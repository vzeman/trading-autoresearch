"""Train and compare online-patched 15-minute transformer trading models.

This is the intraday successor to ``train_online_daily_patch_model.py``.

It builds one shared 15-minute dataset from cached Alpaca minute bars, then
trains multiple transformer-style models under the same causal protocol:

1. Train on the first warmup year.
2. At each later 15-minute decision timestamp, score the current cross-section.
3. Simulate long/flat buy/hold/sell behavior for the next 15-minute interval.
4. Patch-train on the now-realized interval.
5. Repeat chronologically and compare individual models plus ensembles.

The goal is not to bless a live model. It is to create a fair harness where
PatchTST, temporal-fusion, decision/trajectory transformers, Perceiver, JEPA,
and cross-asset attention can compete on the same data.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from prepare import CACHE_DIR
from train_daily_ranker import pick_device


MODEL_NAMES = [
    "patchtst",
    "patchtransformer",
    "itransformer",
    "temporal_fusion",
    "decision_transformer",
    "trajectory_transformer",
    "perceiver",
    "cross_asset",
    "jepa_patch",
]


BASE_FEATURES = [
    "ret_1", "ret_2", "ret_4", "ret_8", "ret_16", "ret_26",
    "vol_8", "vol_26", "volume_z_26",
    "ma8_dist", "ma26_dist", "drawdown_26",
    "hl_range", "body_return",
    "trend_slope_8", "trend_slope_16", "trend_slope_26", "trend_slope_52",
    "trend_consistency_8", "trend_consistency_16", "trend_consistency_26", "trend_consistency_52",
    "trend_accel_8_26", "trend_accel_16_52",
    "pullback_8", "pullback_16", "pullback_26", "pullback_52",
    "breakout_16", "breakout_26", "breakout_52",
    "support_dist_16", "support_dist_26", "support_dist_52",
    "range_compression_8_26", "range_compression_16_52",
    "volume_confirmed_ret_1", "volume_confirmed_ret_4", "volume_confirmed_ret_16",
    "upper_wick_ratio", "lower_wick_ratio", "body_to_range",
    "consecutive_up", "consecutive_down",
    "rsi_14", "rsi_14_signal",
    "macd_line", "macd_signal", "macd_hist",
    "bb_z_20", "bb_width_20", "bb_percent_b_20",
    "stoch_k_14", "stoch_d_14",
    "vwap_dist_26", "vwap_dist_52",
    "atr_14_pct", "atr_26_pct",
    "donchian_pos_20", "donchian_breakout_20", "donchian_breakdown_20",
    "ema_8_21_cross", "ema_21_55_cross",
    "mean_reversion_z_20", "mean_reversion_z_52",
    "algo_momentum_vote", "algo_mean_reversion_vote", "algo_breakout_vote", "algo_trend_quality",
    "spy_ret_1", "spy_ret_4", "spy_ret_16", "spy_ret_26",
    "rel_spy_1", "rel_spy_4", "rel_spy_16", "rel_spy_26",
    "rel_spy_trend_slope_16", "rel_spy_trend_slope_26",
    "rel_spy_trend_accel_8_26", "rel_spy_trend_accel_16_52",
    "mkt_ret_1_mean", "mkt_ret_4_mean", "mkt_ret_16_mean",
    "mkt_pct_positive_16", "mkt_pct_above_ma8", "mkt_dispersion_16",
    "mkt_pct_positive_16_slope", "mkt_pct_above_ma8_slope", "mkt_breadth_accel",
    "rel_mkt_16",
    "tod_sin", "tod_cos",
]


@dataclass(frozen=True)
class Config:
    dataset: str
    output_dir: str
    build_dataset: bool
    start_date: str
    end_date: str
    symbol_limit: int
    universe_mode: str
    universe_rank_cache: str
    min_rows: int
    interval_minutes: int
    seq_len: int
    warmup_days: int
    train_start: str
    train_end: str
    eval_start: str
    eval_end: str
    initial_epochs: int
    patch_epochs: int
    batch_size: int
    hidden_dim: int
    layers: int
    heads: int
    dropout: float
    lr: float
    weight_decay: float
    max_train_samples: int
    max_eval_intervals: int
    min_close: float
    max_abs_return: float
    roundtrip_cost: float
    max_positions: int
    min_pred_profit: float
    max_pred_crash: float
    min_buy_prob: float
    min_pred_score: float
    min_pred_utility: float
    portfolio_exposure: float
    daily_loss_stop: float
    symbol_cooldown_loss: float
    symbol_cooldown_intervals: int
    symbol_daily_cap: int
    spy_momentum_window: int
    spy_momentum_min_return: float
    strategy_momentum_window: int
    strategy_momentum_min_return: float
    filter_falling_stocks: bool
    falling_filter_min_signals: int
    falling_max_ret_4: float
    falling_max_ret_16: float
    falling_max_trend_slope_8: float
    falling_max_trend_slope_16: float
    falling_max_ma8_dist: float
    falling_max_ma26_dist: float
    falling_max_algo_momentum_vote: float
    models: str
    max_ensemble_size: int
    device: str
    seed: int


def cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol}_1m.parquet"


def cached_symbols_alpha(limit: int) -> list[str]:
    symbols = sorted(p.name.removesuffix("_1m.parquet") for p in CACHE_DIR.glob("*_1m.parquet"))
    symbols = [s for s in symbols if not s.startswith("^") and s != "SPY" and cache_path(s).exists()]
    if limit > 0:
        symbols = symbols[:limit]
    return symbols


def yfinance_symbol(symbol: str) -> str:
    return symbol.replace(".", "-")


def recent_dollar_volume_score(symbol: str, lookback_days: int = 60) -> tuple[float, float]:
    try:
        df = pd.read_parquet(cache_path(symbol), columns=["timestamp", "close", "volume"])
    except Exception:
        return 0.0, 0.0
    if df.empty:
        return 0.0, 0.0
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    cutoff = df["timestamp"].max() - pd.Timedelta(days=lookback_days)
    df = df[df["timestamp"] >= cutoff].copy()
    if df.empty:
        return 0.0, 0.0
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    dollar_volume = close * volume
    daily = pd.DataFrame({"timestamp": df["timestamp"], "dollar_volume": dollar_volume})
    daily["date"] = daily["timestamp"].dt.tz_convert("America/New_York").dt.date
    daily_dv = daily.groupby("date")["dollar_volume"].sum()
    return float(daily_dv.median()), float(close.iloc[-1])


def fetch_market_caps(symbols: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    try:
        import yfinance as yf
    except Exception as exc:
        print(f"[15m] yfinance unavailable for valuation ranking: {exc}", flush=True)
        return out
    for symbol in symbols:
        cap = 0.0
        try:
            ticker = yf.Ticker(yfinance_symbol(symbol))
            fast = getattr(ticker, "fast_info", {}) or {}
            if hasattr(fast, "get"):
                cap = float(fast.get("market_cap") or fast.get("marketCap") or 0.0)
            if cap <= 0:
                info = getattr(ticker, "info", {}) or {}
                cap = float(info.get("marketCap") or 0.0)
        except Exception:
            cap = 0.0
        out[symbol] = cap
    return out


def cached_symbols_top_volume_valuation(limit: int, rank_cache: str) -> list[str]:
    cache = Path(rank_cache)
    if cache.exists():
        ranked = pd.read_csv(cache)
        if "symbol" in ranked.columns and len(ranked) >= max(1, limit):
            symbols = ranked["symbol"].astype(str).tolist()
            return symbols[:limit] if limit > 0 else symbols

    symbols = cached_symbols_alpha(0)
    rows = []
    for symbol in symbols:
        median_dollar_volume, last_close = recent_dollar_volume_score(symbol)
        if median_dollar_volume <= 0 or last_close <= 0:
            continue
        rows.append({
            "symbol": symbol,
            "median_dollar_volume_60d": median_dollar_volume,
            "last_close": last_close,
        })
    ranked = pd.DataFrame(rows)
    if ranked.empty:
        return cached_symbols_alpha(limit)
    ranked = ranked.sort_values("median_dollar_volume_60d", ascending=False).reset_index(drop=True)
    candidate_n = min(len(ranked), max(limit * 6 if limit > 0 else 120, 120))
    candidates = ranked.head(candidate_n)["symbol"].astype(str).tolist()
    market_caps = fetch_market_caps(candidates)
    ranked["market_cap"] = ranked["symbol"].map(market_caps).fillna(0.0).astype(float)
    dv_log = np.log1p(ranked["median_dollar_volume_60d"].astype(float))
    cap_log = np.log1p(ranked["market_cap"].astype(float).clip(lower=0.0))
    cap_missing = ranked["market_cap"].astype(float) <= 0
    cap_log = cap_log.where(~cap_missing, dv_log)
    ranked["volume_valuation_score"] = 0.55 * dv_log + 0.45 * cap_log
    ranked = ranked.sort_values("volume_valuation_score", ascending=False).reset_index(drop=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(cache, index=False)
    print(f"[15m] wrote universe ranking path={cache} rows={len(ranked):,}", flush=True)
    if limit > 0:
        return ranked.head(limit)["symbol"].astype(str).tolist()
    return ranked["symbol"].astype(str).tolist()


def cached_symbols(config: Config) -> list[str]:
    if config.universe_mode == "top_volume_valuation":
        symbols = cached_symbols_top_volume_valuation(config.symbol_limit, config.universe_rank_cache)
        print(f"[15m] top volume/valuation universe={symbols}", flush=True)
        return symbols
    if config.universe_mode != "alphabetical":
        raise ValueError(f"unknown universe mode: {config.universe_mode}")
    return cached_symbols_alpha(config.symbol_limit)


def aggregate_15m(symbol: str, interval_minutes: int) -> pd.DataFrame:
    df = pd.read_parquet(cache_path(symbol), columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")
    for col in ("open", "high", "low", "close"):
        df = df[df[col].astype(float) > 0]
    ny = df["timestamp"].dt.tz_convert("America/New_York")
    minute_of_day = ny.dt.hour * 60 + ny.dt.minute
    df = df[(minute_of_day >= 9 * 60 + 30) & (minute_of_day < 16 * 60)].copy()
    if df.empty:
        return df
    bucket = ny.dt.floor(f"{interval_minutes}min")
    df["bucket"] = bucket
    bars = df.groupby("bucket", sort=True).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).reset_index()
    bars = bars.rename(columns={"bucket": "timestamp"})
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    bars["symbol"] = symbol
    return bars


def add_symbol_features(bars: pd.DataFrame, spy: pd.DataFrame) -> pd.DataFrame:
    bars = bars.sort_values("timestamp").reset_index(drop=True)
    close = bars["close"].astype(float)
    volume = bars["volume"].astype(float)
    for n in (1, 2, 4, 8, 16, 26):
        bars[f"ret_{n}"] = np.log(close / close.shift(n))
    for n in (8, 26):
        bars[f"vol_{n}"] = bars["ret_1"].rolling(n, min_periods=max(4, n // 2)).std()
        bars[f"ma{n}_dist"] = close / close.rolling(n, min_periods=max(4, n // 2)).mean() - 1.0
    vol_mean = np.log1p(volume).rolling(26, min_periods=8).mean()
    vol_std = np.log1p(volume).rolling(26, min_periods=8).std()
    bars["volume_z_26"] = (np.log1p(volume) - vol_mean) / vol_std.replace(0.0, np.nan)
    bars["drawdown_26"] = close / close.rolling(26, min_periods=8).max() - 1.0
    bars["hl_range"] = bars["high"].astype(float) / close - bars["low"].astype(float) / close
    bars["body_return"] = bars["close"].astype(float) / bars["open"].astype(float) - 1.0
    log_close = np.log(close)
    for n in (8, 16, 26, 52):
        bars[f"trend_slope_{n}"] = (log_close - log_close.shift(n)) / float(n)
        bars[f"trend_consistency_{n}"] = (bars["ret_1"] > 0).rolling(n, min_periods=max(4, n // 2)).mean()
        roll_high = close.rolling(n, min_periods=max(4, n // 2)).max()
        roll_low = close.rolling(n, min_periods=max(4, n // 2)).min()
        bars[f"pullback_{n}"] = close / roll_high - 1.0
        if n >= 16:
            prev_high = roll_high.shift(1)
            prev_low = roll_low.shift(1)
            bars[f"breakout_{n}"] = close / prev_high - 1.0
            bars[f"support_dist_{n}"] = close / prev_low - 1.0
    bars["trend_accel_8_26"] = bars["trend_slope_8"] - bars["trend_slope_26"]
    bars["trend_accel_16_52"] = bars["trend_slope_16"] - bars["trend_slope_52"]
    candle_range = (bars["high"].astype(float) - bars["low"].astype(float)).replace(0.0, np.nan)
    range_8 = candle_range.rolling(8, min_periods=4).mean()
    range_16 = candle_range.rolling(16, min_periods=8).mean()
    range_26 = candle_range.rolling(26, min_periods=13).mean()
    range_52 = candle_range.rolling(52, min_periods=26).mean()
    bars["range_compression_8_26"] = range_8 / range_26.replace(0.0, np.nan) - 1.0
    bars["range_compression_16_52"] = range_16 / range_52.replace(0.0, np.nan) - 1.0
    bars["volume_confirmed_ret_1"] = bars["ret_1"] * bars["volume_z_26"].clip(-5.0, 5.0)
    bars["volume_confirmed_ret_4"] = bars["ret_4"] * bars["volume_z_26"].clip(-5.0, 5.0)
    bars["volume_confirmed_ret_16"] = bars["ret_16"] * bars["volume_z_26"].clip(-5.0, 5.0)
    open_ = bars["open"].astype(float)
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    body_high = pd.concat([open_, close], axis=1).max(axis=1)
    body_low = pd.concat([open_, close], axis=1).min(axis=1)
    bars["upper_wick_ratio"] = (high - body_high) / candle_range
    bars["lower_wick_ratio"] = (body_low - low) / candle_range
    bars["body_to_range"] = (close - open_).abs() / candle_range
    up = (bars["ret_1"] > 0).astype(int)
    down = (bars["ret_1"] < 0).astype(int)
    up_groups = (up != up.shift()).cumsum()
    down_groups = (down != down.shift()).cumsum()
    bars["consecutive_up"] = up.groupby(up_groups).cumsum().clip(upper=20) / 20.0
    bars["consecutive_down"] = down.groupby(down_groups).cumsum().clip(upper=20) / 20.0

    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = gain / loss.replace(0.0, np.nan)
    bars["rsi_14"] = (100.0 - 100.0 / (1.0 + rs)) / 100.0
    bars["rsi_14_signal"] = (bars["rsi_14"] - 0.5) * 2.0
    ema_8 = close.ewm(span=8, adjust=False, min_periods=8).mean()
    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_21 = close.ewm(span=21, adjust=False, min_periods=21).mean()
    ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    ema_55 = close.ewm(span=55, adjust=False, min_periods=55).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    bars["macd_line"] = macd / close
    bars["macd_signal"] = macd_signal / close
    bars["macd_hist"] = (macd - macd_signal) / close
    ma20 = close.rolling(20, min_periods=10).mean()
    std20 = close.rolling(20, min_periods=10).std()
    upper20 = ma20 + 2.0 * std20
    lower20 = ma20 - 2.0 * std20
    bars["bb_z_20"] = (close - ma20) / std20.replace(0.0, np.nan)
    bars["bb_width_20"] = (upper20 - lower20) / ma20.replace(0.0, np.nan)
    bars["bb_percent_b_20"] = (close - lower20) / (upper20 - lower20).replace(0.0, np.nan)
    low14 = low.rolling(14, min_periods=7).min()
    high14 = high.rolling(14, min_periods=7).max()
    bars["stoch_k_14"] = (close - low14) / (high14 - low14).replace(0.0, np.nan)
    bars["stoch_d_14"] = bars["stoch_k_14"].rolling(3, min_periods=2).mean()
    typical = (high + low + close) / 3.0
    pv = typical * volume
    vwap26 = pv.rolling(26, min_periods=13).sum() / volume.rolling(26, min_periods=13).sum().replace(0.0, np.nan)
    vwap52 = pv.rolling(52, min_periods=26).sum() / volume.rolling(52, min_periods=26).sum().replace(0.0, np.nan)
    bars["vwap_dist_26"] = close / vwap26 - 1.0
    bars["vwap_dist_52"] = close / vwap52 - 1.0
    prev_close = close.shift(1)
    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    bars["atr_14_pct"] = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / close
    bars["atr_26_pct"] = true_range.ewm(alpha=1 / 26, adjust=False, min_periods=26).mean() / close
    don_high20 = high.rolling(20, min_periods=10).max()
    don_low20 = low.rolling(20, min_periods=10).min()
    prev_don_high20 = don_high20.shift(1)
    prev_don_low20 = don_low20.shift(1)
    bars["donchian_pos_20"] = (close - don_low20) / (don_high20 - don_low20).replace(0.0, np.nan)
    bars["donchian_breakout_20"] = close / prev_don_high20 - 1.0
    bars["donchian_breakdown_20"] = close / prev_don_low20 - 1.0
    bars["ema_8_21_cross"] = ema_8 / ema_21 - 1.0
    bars["ema_21_55_cross"] = ema_21 / ema_55 - 1.0
    ma52 = close.rolling(52, min_periods=26).mean()
    std52 = close.rolling(52, min_periods=26).std()
    bars["mean_reversion_z_20"] = bars["bb_z_20"]
    bars["mean_reversion_z_52"] = (close - ma52) / std52.replace(0.0, np.nan)
    bars["algo_momentum_vote"] = (
        (bars["ema_8_21_cross"] > 0).astype(float)
        + (bars["macd_hist"] > 0).astype(float)
        + (bars["rsi_14"] > 0.55).astype(float)
        + (bars["vwap_dist_26"] > 0).astype(float)
    ) / 4.0
    bars["algo_mean_reversion_vote"] = (
        (bars["bb_z_20"] < -1.0).astype(float)
        + (bars["rsi_14"] < 0.35).astype(float)
        + (bars["stoch_k_14"] < 0.20).astype(float)
    ) / 3.0
    bars["algo_breakout_vote"] = (
        (bars["donchian_breakout_20"] > 0).astype(float)
        + (bars["volume_z_26"] > 1.0).astype(float)
        + (bars["range_compression_8_26"] > 0).astype(float)
    ) / 3.0
    bars["algo_trend_quality"] = (
        bars["trend_consistency_26"].fillna(0.5)
        + bars["algo_momentum_vote"].fillna(0.0)
        - bars["algo_mean_reversion_vote"].fillna(0.0)
    ) / 2.0
    bars["future_return"] = close.shift(-1) / close - 1.0
    bars["future_min_return"] = bars["low"].shift(-1) / close - 1.0

    spy_cols = spy[["timestamp", "close"]].rename(columns={"close": "spy_close"})
    bars = bars.merge(spy_cols, on="timestamp", how="inner")
    spy_close = bars["spy_close"].astype(float)
    for n in (1, 4, 16, 26):
        bars[f"spy_ret_{n}"] = np.log(spy_close / spy_close.shift(n))
        bars[f"rel_spy_{n}"] = bars[f"ret_{n}"] - bars[f"spy_ret_{n}"]
    spy_log_close = np.log(spy_close)
    spy_slope_8 = (spy_log_close - spy_log_close.shift(8)) / 8.0
    spy_slope_16 = (spy_log_close - spy_log_close.shift(16)) / 16.0
    spy_slope_26 = (spy_log_close - spy_log_close.shift(26)) / 26.0
    spy_slope_52 = (spy_log_close - spy_log_close.shift(52)) / 52.0
    bars["rel_spy_trend_slope_16"] = bars["trend_slope_16"] - spy_slope_16
    bars["rel_spy_trend_slope_26"] = bars["trend_slope_26"] - spy_slope_26
    bars["rel_spy_trend_accel_8_26"] = bars["trend_accel_8_26"] - (spy_slope_8 - spy_slope_26)
    bars["rel_spy_trend_accel_16_52"] = bars["trend_accel_16_52"] - (spy_slope_16 - spy_slope_52)
    bars["future_spy_return"] = spy_close.shift(-1) / spy_close - 1.0
    bars["future_alpha"] = bars["future_return"] - bars["future_spy_return"]
    ny = bars["timestamp"].dt.tz_convert("America/New_York")
    minutes = ny.dt.hour * 60 + ny.dt.minute - (9 * 60 + 30)
    frac = minutes.astype(float) / float(6.5 * 60)
    bars["tod_sin"] = np.sin(2.0 * np.pi * frac)
    bars["tod_cos"] = np.cos(2.0 * np.pi * frac)
    return bars


def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("timestamp", group_keys=False)
    market = grouped.agg(
        mkt_ret_1_mean=("ret_1", "mean"),
        mkt_ret_4_mean=("ret_4", "mean"),
        mkt_ret_16_mean=("ret_16", "mean"),
        mkt_pct_positive_16=("ret_16", lambda s: float((s > 0).mean())),
        mkt_pct_above_ma8=("ma8_dist", lambda s: float((s > 0).mean())),
        mkt_dispersion_16=("ret_16", "std"),
    ).reset_index()
    out = df.merge(market, on="timestamp", how="left")
    out["rel_mkt_16"] = out["ret_16"] - out["mkt_ret_16_mean"]
    out = out.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    by_time = out[["timestamp", "mkt_pct_positive_16", "mkt_pct_above_ma8"]].drop_duplicates("timestamp").sort_values("timestamp")
    by_time["mkt_pct_positive_16_slope"] = by_time["mkt_pct_positive_16"] - by_time["mkt_pct_positive_16"].shift(16)
    by_time["mkt_pct_above_ma8_slope"] = by_time["mkt_pct_above_ma8"] - by_time["mkt_pct_above_ma8"].shift(16)
    by_time["mkt_breadth_accel"] = (
        by_time["mkt_pct_positive_16"].diff(4) - by_time["mkt_pct_positive_16"].diff(16)
    )
    out = out.merge(
        by_time[["timestamp", "mkt_pct_positive_16_slope", "mkt_pct_above_ma8_slope", "mkt_breadth_accel"]],
        on="timestamp",
        how="left",
    )
    return out


def build_dataset(config: Config) -> pd.DataFrame:
    spy = aggregate_15m("SPY", config.interval_minutes)
    if spy.empty:
        raise RuntimeError("SPY cache is empty; refresh data first")
    symbols = cached_symbols(config)
    frames = []
    for i, symbol in enumerate(symbols, start=1):
        path = cache_path(symbol)
        if not path.exists():
            continue
        try:
            bars = aggregate_15m(symbol, config.interval_minutes)
            if len(bars) < config.min_rows:
                continue
            feat = add_symbol_features(bars, spy)
            if len(feat) >= config.min_rows:
                frames.append(feat)
        except Exception as exc:
            print(f"[15m] skip {symbol}: {exc}", flush=True)
        if i % 50 == 0:
            print(f"[15m] featurized {i}/{len(symbols)} symbols", flush=True)
    if not frames:
        raise RuntimeError("no symbol frames built")
    df = pd.concat(frames, ignore_index=True)
    df = add_market_features(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if config.start_date:
        df = df[df["timestamp"] >= pd.Timestamp(config.start_date, tz="UTC")]
    if config.end_date:
        df = df[df["timestamp"] <= pd.Timestamp(config.end_date, tz="UTC")]
    clean = (
        df["close"].astype(float).ge(config.min_close)
        & df["future_return"].astype(float).between(-config.max_abs_return, config.max_abs_return)
        & df["future_min_return"].astype(float).between(-config.max_abs_return, config.max_abs_return)
        & df["future_spy_return"].astype(float).between(-config.max_abs_return, config.max_abs_return)
    )
    for col in BASE_FEATURES:
        clean &= np.isfinite(df[col].astype(float))
    df = df[clean].copy()
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    symbols = sorted(df["symbol"].unique())
    sym_to_id = {s: i for i, s in enumerate(symbols)}
    df["symbol_id"] = df["symbol"].map(sym_to_id).astype(int)
    df["profit_label"] = (df["future_return"] > config.roundtrip_cost).astype(float)
    df["crash_label"] = (
        (df["future_return"] < -0.006) | (df["future_min_return"] < -0.010)
    ).astype(float)
    df["utility"] = (
        df["future_alpha"].clip(-0.03, 0.03)
        + 0.50 * df["future_return"].clip(-0.03, 0.03)
        - 2.00 * df["future_min_return"].clip(upper=0.0).abs()
        - 0.25 * df["crash_label"]
    )
    ranks = df.groupby("timestamp")["utility"].rank(pct=True)
    df["action_label"] = 1
    df.loc[(ranks >= 0.90) & (df["future_return"] > config.roundtrip_cost), "action_label"] = 2
    df.loc[(df["future_return"] < -config.roundtrip_cost) | (df["crash_label"] > 0), "action_label"] = 0
    out = Path(config.dataset)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"[15m] wrote dataset rows={len(df):,} symbols={len(symbols):,} path={out}", flush=True)
    return df


def load_dataset(config: Config) -> pd.DataFrame:
    path = Path(config.dataset)
    if config.build_dataset or not path.exists():
        return build_dataset(config)
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    print(f"[15m] loaded dataset rows={len(df):,} symbols={df['symbol'].nunique():,}", flush=True)
    return df


def valid_row_indices(df: pd.DataFrame, seq_len: int) -> np.ndarray:
    indices: list[np.ndarray] = []
    for _, group in df.groupby("symbol", sort=False):
        idx = group.index.to_numpy(np.int64)
        if len(idx) > seq_len:
            indices.append(idx[seq_len - 1:-1])
    if not indices:
        return np.array([], dtype=np.int64)
    return np.concatenate(indices)


class SequenceDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        symbol_ids: np.ndarray,
        utility: np.ndarray,
        profit: np.ndarray,
        crash: np.ndarray,
        action: np.ndarray,
        row_indices: np.ndarray,
        seq_len: int,
    ) -> None:
        self.x = x
        self.symbol_ids = symbol_ids
        self.utility = utility
        self.profit = profit
        self.crash = crash
        self.action = action
        self.row_indices = row_indices.astype(np.int64)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.row_indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row = int(self.row_indices[i])
        seq = self.x[row - self.seq_len + 1: row + 1]
        return (
            torch.from_numpy(seq),
            torch.tensor(self.symbol_ids[row], dtype=torch.long),
            torch.tensor(self.utility[row], dtype=torch.float32),
            torch.tensor(self.profit[row], dtype=torch.float32),
            torch.tensor(self.crash[row], dtype=torch.float32),
            torch.tensor(self.action[row], dtype=torch.long),
        )


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1]]


class HeadMixin(nn.Module):
    def make_heads(self, hidden_dim: int) -> None:
        self.utility = nn.Linear(hidden_dim, 1)
        self.profit = nn.Linear(hidden_dim, 1)
        self.crash = nn.Linear(hidden_dim, 1)
        self.action = nn.Linear(hidden_dim, 3)

    def heads(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "utility": self.utility(h).squeeze(-1),
            "profit": self.profit(h).squeeze(-1),
            "crash": self.crash(h).squeeze(-1),
            "action": self.action(h),
        }


class PatchTSTLite(HeadMixin):
    def __init__(self, n_features: int, n_symbols: int, config: Config) -> None:
        super().__init__()
        patch_len = 4
        self.patch_len = patch_len
        self.proj = nn.Linear(n_features * patch_len, config.hidden_dim)
        self.pos = PositionalEncoding(config.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            config.hidden_dim, config.heads, config.hidden_dim * 4,
            config.dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, config.layers)
        self.make_heads(config.hidden_dim)

    def forward(self, x: torch.Tensor, symbol_id: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        b, t, f = x.shape
        usable = (t // self.patch_len) * self.patch_len
        x = x[:, -usable:].reshape(b, usable // self.patch_len, f * self.patch_len)
        h = self.encoder(self.pos(self.proj(x))).mean(dim=1)
        return self.heads(h)


class PatchTransformerLite(HeadMixin):
    def __init__(self, n_features: int, n_symbols: int, config: Config) -> None:
        super().__init__()
        self.patch_len = 8
        self.stride = 4
        self.proj = nn.Linear(n_features * self.patch_len, config.hidden_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        self.symbol = nn.Embedding(n_symbols, config.hidden_dim)
        self.pos = PositionalEncoding(config.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            config.hidden_dim, config.heads, config.hidden_dim * 4,
            config.dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, config.layers)
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.make_heads(config.hidden_dim)

    def forward(self, x: torch.Tensor, symbol_id: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.transpose(2, 3).contiguous().flatten(start_dim=2)
        tokens = self.proj(patches)
        cls = self.cls.expand(x.shape[0], -1, -1)
        if symbol_id is not None:
            cls = cls + self.symbol(symbol_id).unsqueeze(1)
        h = self.encoder(self.pos(torch.cat([cls, tokens], dim=1)))
        return self.heads(self.norm(h[:, 0]))


class ITransformerLite(HeadMixin):
    def __init__(self, n_features: int, n_symbols: int, config: Config) -> None:
        super().__init__()
        self.temporal = nn.Linear(config.seq_len, config.hidden_dim)
        self.feature_embed = nn.Parameter(torch.randn(n_features, config.hidden_dim) * 0.02)
        self.symbol = nn.Embedding(n_symbols, config.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            config.hidden_dim, config.heads, config.hidden_dim * 4,
            config.dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, config.layers)
        self.attn_pool = nn.Linear(config.hidden_dim, 1)
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.make_heads(config.hidden_dim)

    def forward(self, x: torch.Tensor, symbol_id: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        tokens = self.temporal(x.transpose(1, 2)) + self.feature_embed.unsqueeze(0)
        if symbol_id is not None:
            tokens = tokens + self.symbol(symbol_id).unsqueeze(1)
        h = self.encoder(tokens)
        weights = torch.softmax(self.attn_pool(h).squeeze(-1), dim=1).unsqueeze(-1)
        pooled = self.norm((h * weights).sum(dim=1))
        return self.heads(pooled)


class TemporalFusionLite(HeadMixin):
    def __init__(self, n_features: int, n_symbols: int, config: Config) -> None:
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(n_features, n_features), nn.Sigmoid())
        self.proj = nn.Linear(n_features, config.hidden_dim)
        self.symbol = nn.Embedding(n_symbols, config.hidden_dim)
        self.pos = PositionalEncoding(config.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            config.hidden_dim, config.heads, config.hidden_dim * 4,
            config.dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, config.layers)
        self.make_heads(config.hidden_dim)

    def forward(self, x: torch.Tensor, symbol_id: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        gated = x * self.gate(x)
        h = self.proj(gated)
        if symbol_id is not None:
            h = h + self.symbol(symbol_id).unsqueeze(1)
        h = self.encoder(self.pos(h))
        return self.heads(h[:, -1])


class DecisionTransformerLite(HeadMixin):
    def __init__(self, n_features: int, n_symbols: int, config: Config) -> None:
        super().__init__()
        self.state = nn.Linear(n_features, config.hidden_dim)
        self.goal = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        self.pos = PositionalEncoding(config.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            config.hidden_dim, config.heads, config.hidden_dim * 4,
            config.dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, config.layers)
        self.make_heads(config.hidden_dim)

    def forward(self, x: torch.Tensor, symbol_id: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        goal = self.goal.expand(x.shape[0], -1, -1)
        tokens = torch.cat([goal, self.state(x)], dim=1)
        mask = torch.triu(torch.ones(tokens.shape[1], tokens.shape[1], device=tokens.device), diagonal=1).bool()
        h = self.encoder(self.pos(tokens), mask=mask)
        return self.heads(h[:, -1])


class TrajectoryTransformerLite(HeadMixin):
    def __init__(self, n_features: int, n_symbols: int, config: Config) -> None:
        super().__init__()
        self.proj = nn.Linear(n_features, config.hidden_dim)
        self.pos = PositionalEncoding(config.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            config.hidden_dim, config.heads, config.hidden_dim * 4,
            config.dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, config.layers)
        self.make_heads(config.hidden_dim)

    def forward(self, x: torch.Tensor, symbol_id: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        tokens = self.proj(x)
        mask = torch.triu(torch.ones(tokens.shape[1], tokens.shape[1], device=tokens.device), diagonal=1).bool()
        h = self.encoder(self.pos(tokens), mask=mask)
        return self.heads(h[:, -1])


class PerceiverLite(HeadMixin):
    def __init__(self, n_features: int, n_symbols: int, config: Config) -> None:
        super().__init__()
        self.input = nn.Linear(n_features, config.hidden_dim)
        self.latents = nn.Parameter(torch.randn(8, config.hidden_dim) * 0.02)
        self.cross = nn.MultiheadAttention(config.hidden_dim, config.heads, batch_first=True)
        layer = nn.TransformerEncoderLayer(
            config.hidden_dim, config.heads, config.hidden_dim * 4,
            config.dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, config.layers)
        self.make_heads(config.hidden_dim)

    def forward(self, x: torch.Tensor, symbol_id: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        inp = self.input(x)
        lat = self.latents.unsqueeze(0).expand(x.shape[0], -1, -1)
        lat, _ = self.cross(lat, inp, inp, need_weights=False)
        h = self.encoder(lat).mean(dim=1)
        return self.heads(h)


class CrossAssetTransformerLite(HeadMixin):
    def __init__(self, n_features: int, n_symbols: int, config: Config) -> None:
        super().__init__()
        self.proj = nn.Linear(n_features, config.hidden_dim)
        self.symbol = nn.Embedding(n_symbols, config.hidden_dim)
        self.temporal = nn.GRU(config.hidden_dim, config.hidden_dim, batch_first=True)
        self.asset_attn = nn.MultiheadAttention(config.hidden_dim, config.heads, batch_first=True)
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.make_heads(config.hidden_dim)

    def forward(self, x: torch.Tensor, symbol_id: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        h = self.proj(x)
        if symbol_id is not None:
            h = h + self.symbol(symbol_id).unsqueeze(1)
        _, last = self.temporal(h)
        token = last.squeeze(0).unsqueeze(1)
        attended, _ = self.asset_attn(token, token, token, need_weights=False)
        return self.heads(self.norm(attended.squeeze(1)))


class JEPAPatchTransformerLite(HeadMixin):
    def __init__(self, n_features: int, n_symbols: int, config: Config) -> None:
        super().__init__()
        self.proj = nn.Linear(n_features, config.hidden_dim)
        self.pos = PositionalEncoding(config.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            config.hidden_dim, config.heads, config.hidden_dim * 4,
            config.dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, config.layers)
        self.predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.make_heads(config.hidden_dim)

    def forward(self, x: torch.Tensor, symbol_id: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        tokens = self.pos(self.proj(x))
        full = self.encoder(tokens)
        context = full[:, :-1].mean(dim=1)
        target = full[:, -1].detach()
        pred = self.predictor(context)
        out = self.heads(full[:, -1])
        out["aux_loss"] = nn.functional.smooth_l1_loss(pred.contiguous(), target.contiguous())
        return out


def make_model(name: str, n_features: int, n_symbols: int, config: Config) -> nn.Module:
    if name == "patchtst":
        return PatchTSTLite(n_features, n_symbols, config)
    if name == "patchtransformer":
        return PatchTransformerLite(n_features, n_symbols, config)
    if name == "itransformer":
        return ITransformerLite(n_features, n_symbols, config)
    if name == "temporal_fusion":
        return TemporalFusionLite(n_features, n_symbols, config)
    if name == "decision_transformer":
        return DecisionTransformerLite(n_features, n_symbols, config)
    if name == "trajectory_transformer":
        return TrajectoryTransformerLite(n_features, n_symbols, config)
    if name == "perceiver":
        return PerceiverLite(n_features, n_symbols, config)
    if name == "cross_asset":
        return CrossAssetTransformerLite(n_features, n_symbols, config)
    if name == "jepa_patch":
        return JEPAPatchTransformerLite(n_features, n_symbols, config)
    raise ValueError(f"unknown model: {name}")


def make_arrays(df: pd.DataFrame, train_rows: np.ndarray) -> dict:
    x = df[BASE_FEATURES].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
    mean = x[train_rows].mean(axis=0)
    std = np.where(x[train_rows].std(axis=0) < 1e-8, 1.0, x[train_rows].std(axis=0))
    x = np.clip((x - mean) / std, -10.0, 10.0).astype(np.float32)
    return {
        "x": x,
        "symbol_id": df["symbol_id"].astype(int).to_numpy(np.int64),
        "utility": df["utility"].astype(float).to_numpy(np.float32),
        "profit": df["profit_label"].astype(float).to_numpy(np.float32),
        "crash": df["crash_label"].astype(float).to_numpy(np.float32),
        "action": df["action_label"].astype(int).to_numpy(np.int64),
        "mean": mean,
        "std": std,
    }


def model_loss(out: dict[str, torch.Tensor], utility: torch.Tensor, profit: torch.Tensor, crash: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    loss = (
        nn.functional.smooth_l1_loss(out["utility"], utility)
        + 0.25 * nn.functional.binary_cross_entropy_with_logits(out["profit"], profit)
        + 0.75 * nn.functional.binary_cross_entropy_with_logits(out["crash"], crash)
        + 0.50 * nn.functional.cross_entropy(out["action"], action)
    )
    if "aux_loss" in out:
        loss = loss + 0.10 * out["aux_loss"]
    return loss


def train_initial(model: nn.Module, arrays: dict, rows: np.ndarray, config: Config, device: str) -> list[dict]:
    if config.max_train_samples > 0 and len(rows) > config.max_train_samples:
        rng = np.random.default_rng(config.seed)
        rows = np.sort(rng.choice(rows, size=config.max_train_samples, replace=False))
    ds = SequenceDataset(
        arrays["x"], arrays["symbol_id"], arrays["utility"], arrays["profit"],
        arrays["crash"], arrays["action"], rows, config.seq_len,
    )
    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=True, drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    history = []
    for epoch in range(config.initial_epochs):
        model.train()
        losses = []
        for xb, sid, utility, profit, crash, action in loader:
            xb = xb.to(device)
            sid = sid.to(device)
            utility = utility.to(device)
            profit = profit.to(device)
            crash = crash.to(device)
            action = action.to(device)
            opt.zero_grad(set_to_none=True)
            loss = model_loss(model(xb, sid), utility, profit, crash, action)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        row = {"epoch": epoch + 1, "loss": float(np.mean(losses)) if losses else 0.0}
        history.append(row)
        print(f"[15m] epoch {epoch+1}/{config.initial_epochs} loss={row['loss']:.5f}", flush=True)
    return history


def patch_train(model: nn.Module, arrays: dict, rows: np.ndarray, config: Config, device: str) -> float:
    if len(rows) == 0 or config.patch_epochs <= 0:
        return 0.0
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr * 0.5, weight_decay=config.weight_decay)
    losses = []
    ds = SequenceDataset(
        arrays["x"], arrays["symbol_id"], arrays["utility"], arrays["profit"],
        arrays["crash"], arrays["action"], rows, config.seq_len,
    )
    loader = DataLoader(ds, batch_size=min(config.batch_size, max(1, len(ds))), shuffle=True)
    for _ in range(config.patch_epochs):
        for xb, sid, utility, profit, crash, action in loader:
            xb = xb.to(device)
            sid = sid.to(device)
            utility = utility.to(device)
            profit = profit.to(device)
            crash = crash.to(device)
            action = action.to(device)
            opt.zero_grad(set_to_none=True)
            loss = model_loss(model(xb, sid), utility, profit, crash, action)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else 0.0


def score_rows(model: nn.Module, arrays: dict, rows: np.ndarray, config: Config, device: str) -> pd.DataFrame:
    if len(rows) == 0:
        return pd.DataFrame()
    ds = SequenceDataset(
        arrays["x"], arrays["symbol_id"], arrays["utility"], arrays["profit"],
        arrays["crash"], arrays["action"], rows, config.seq_len,
    )
    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=False)
    chunks = []
    model.eval()
    with torch.no_grad():
        offset = 0
        for xb, sid, *_ in loader:
            batch_rows = rows[offset: offset + len(xb)]
            offset += len(xb)
            out = model(xb.to(device), sid.to(device))
            action_prob = torch.softmax(out["action"], dim=-1).detach().cpu().numpy()
            chunks.append(pd.DataFrame({
                "row": batch_rows,
                "pred_utility": out["utility"].detach().cpu().numpy(),
                "pred_profit": torch.sigmoid(out["profit"]).detach().cpu().numpy(),
                "pred_crash": torch.sigmoid(out["crash"]).detach().cpu().numpy(),
                "prob_sell": action_prob[:, 0],
                "prob_hold": action_prob[:, 1],
                "prob_buy": action_prob[:, 2],
            }))
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def max_drawdown(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return 0.0
    peaks = np.maximum.accumulate(arr)
    return float((arr / np.maximum(peaks, 1e-12) - 1.0).min())


class PortfolioState:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.day = None
        self.day_return = 0.0
        self.symbol_blocked_until: dict[str, int] = {}
        self.symbol_day_counts: dict[str, int] = {}
        self.recent_portfolio_returns: list[float] = []
        self.recent_spy_returns: list[float] = []

    def begin_interval(self, ts: pd.Timestamp) -> None:
        current_day = ts.date()
        if current_day != self.day:
            self.day = current_day
            self.day_return = 0.0
            self.symbol_day_counts = {}

    def market_allows_trade(self) -> bool:
        if self.config.daily_loss_stop > -0.99 and self.day_return <= self.config.daily_loss_stop:
            return False
        if self.config.spy_momentum_window > 0:
            if len(self.recent_spy_returns) < self.config.spy_momentum_window:
                return False
            trailing_spy = float(np.prod(1.0 + np.asarray(self.recent_spy_returns[-self.config.spy_momentum_window:])) - 1.0)
            if trailing_spy <= self.config.spy_momentum_min_return:
                return False
        if self.config.strategy_momentum_window > 0:
            if len(self.recent_portfolio_returns) < self.config.strategy_momentum_window:
                return False
            trailing_strategy = float(np.prod(1.0 + np.asarray(self.recent_portfolio_returns[-self.config.strategy_momentum_window:])) - 1.0)
            if trailing_strategy <= self.config.strategy_momentum_min_return:
                return False
        return True

    def filter_symbols(self, candidates: pd.DataFrame, interval_index: int) -> pd.DataFrame:
        if candidates.empty:
            return candidates
        active = candidates.copy()
        if self.config.symbol_cooldown_intervals > 0:
            mask = active["symbol"].astype(str).map(
                lambda s: self.symbol_blocked_until.get(s, -1) <= interval_index
            ).astype(bool)
            active = active.loc[mask].copy()
        if self.config.symbol_daily_cap > 0:
            mask = active["symbol"].astype(str).map(
                lambda s: self.symbol_day_counts.get(s, 0) < self.config.symbol_daily_cap
            ).astype(bool)
            active = active.loc[mask].copy()
        return active

    def record_interval(self, selected: pd.DataFrame, portfolio_ret: float, spy_ret: float, interval_index: int) -> None:
        self.day_return = (1.0 + self.day_return) * (1.0 + portfolio_ret) - 1.0
        self.recent_portfolio_returns.append(float(portfolio_ret))
        self.recent_spy_returns.append(float(spy_ret))
        if selected.empty:
            return
        for _, row in selected.iterrows():
            symbol = str(row["symbol"])
            self.symbol_day_counts[symbol] = self.symbol_day_counts.get(symbol, 0) + 1
            if (
                self.config.symbol_cooldown_intervals > 0
                and float(row["future_return"]) <= self.config.symbol_cooldown_loss
            ):
                self.symbol_blocked_until[symbol] = interval_index + self.config.symbol_cooldown_intervals


def falling_signal_count(rows: pd.DataFrame, config: Config) -> pd.Series:
    signals = []
    thresholds = [
        ("ret_4", config.falling_max_ret_4),
        ("ret_16", config.falling_max_ret_16),
        ("trend_slope_8", config.falling_max_trend_slope_8),
        ("trend_slope_16", config.falling_max_trend_slope_16),
        ("ma8_dist", config.falling_max_ma8_dist),
        ("ma26_dist", config.falling_max_ma26_dist),
        ("algo_momentum_vote", config.falling_max_algo_momentum_vote),
    ]
    for col, threshold in thresholds:
        if col in rows.columns:
            signals.append(rows[col].astype(float).lt(threshold))
    if not signals:
        return pd.Series(0, index=rows.index, dtype=int)
    return pd.concat(signals, axis=1).sum(axis=1).astype(int)


def select_positions(
    scored: pd.DataFrame,
    df: pd.DataFrame,
    config: Config,
    state: PortfolioState | None = None,
    interval_index: int = 0,
) -> pd.DataFrame:
    if scored.empty:
        return scored
    if state is not None and not state.market_allows_trade():
        return scored.iloc[0:0].copy()
    context_cols = [
        "symbol", "future_return", "future_spy_return",
        "ret_4", "ret_16", "trend_slope_8", "trend_slope_16",
        "ma8_dist", "ma26_dist", "algo_momentum_vote",
    ]
    context_cols = [c for c in context_cols if c in df.columns]
    merged = scored.merge(
        df[context_cols].reset_index(names="row"),
        on="row",
        how="left",
    )
    merged["pred_score"] = (
        merged["pred_utility"]
        + 0.05 * merged["pred_profit"]
        + 0.10 * merged["prob_buy"]
        - 0.35 * merged["pred_crash"]
        - 0.05 * merged["prob_sell"]
    )
    merged["falling_signals"] = falling_signal_count(merged, config)
    active = merged[
        (merged["pred_profit"] >= config.min_pred_profit)
        & (merged["pred_crash"] <= config.max_pred_crash)
        & (merged["prob_buy"] >= config.min_buy_prob)
        & (merged["pred_score"] >= config.min_pred_score)
        & (merged["pred_utility"] >= config.min_pred_utility)
    ].copy()
    if config.filter_falling_stocks and not active.empty:
        active = active[active["falling_signals"] < config.falling_filter_min_signals].copy()
    if state is not None:
        active = state.filter_symbols(active, interval_index)
    return active.sort_values("pred_score", ascending=False).head(config.max_positions)


def evaluate_online_model(
    name: str,
    model: nn.Module,
    df: pd.DataFrame,
    arrays: dict,
    eval_rows_by_ts: dict[pd.Timestamp, np.ndarray],
    config: Config,
    device: str,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    equity = 50_000.0
    spy_equity = 50_000.0
    curve_rows = []
    trade_rows = []
    pred_rows = []
    patch_losses = []
    state = PortfolioState(config)
    timestamps = sorted(eval_rows_by_ts)
    if config.max_eval_intervals > 0:
        timestamps = timestamps[: config.max_eval_intervals]
    for i, ts in enumerate(timestamps, start=1):
        rows = eval_rows_by_ts[ts]
        scored = score_rows(model, arrays, rows, config, device)
        scored["timestamp"] = str(ts)
        pred_rows.append(scored)
        state.begin_interval(ts)
        selected = select_positions(scored, df, config, state, i)
        spy_ret = float(df.loc[rows, "future_spy_return"].dropna().iloc[0]) if len(rows) else 0.0
        if selected.empty:
            portfolio_ret = 0.0
            symbols: list[str] = []
        else:
            portfolio_ret = float(config.portfolio_exposure * (selected["future_return"].astype(float).mean() - config.roundtrip_cost))
            symbols = selected["symbol"].astype(str).tolist()
            for _, row in selected.iterrows():
                trade_rows.append({
                    "timestamp": str(ts),
                    "model": name,
                    "symbol": row["symbol"],
                    "pred_score": float(row["pred_score"]),
                    "pred_profit": float(row["pred_profit"]),
                    "pred_crash": float(row["pred_crash"]),
                    "prob_buy": float(row["prob_buy"]),
                    "falling_signals": int(row.get("falling_signals", 0)),
                    "future_return": float(row["future_return"]),
                    "future_spy_return": spy_ret,
                    "future_alpha": float(row["future_return"]) - spy_ret,
                })
        equity *= 1.0 + portfolio_ret
        spy_equity *= 1.0 + spy_ret
        state.record_interval(selected, portfolio_ret, spy_ret, i)
        curve_rows.append({
            "timestamp": str(ts),
            "model": name,
            "equity": equity,
            "spy_equity": spy_equity,
            "portfolio_return": portfolio_ret,
            "spy_return": spy_ret,
            "symbols": ",".join(symbols),
        })
        patch_losses.append(patch_train(model, arrays, rows, config, device))
        if i % 250 == 0:
            print(f"[15m] {name} interval {i}/{len(timestamps)} equity={equity:,.2f}", flush=True)
    curve = pd.DataFrame(curve_rows)
    trades = pd.DataFrame(trade_rows)
    preds = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    trade_returns = trades["future_return"].astype(float).to_numpy() if not trades.empty else np.array([])
    summary = {
        "model": name,
        "decision_intervals": int(len(curve)),
        "active_intervals": int((curve["symbols"].astype(str) != "").sum()) if not curve.empty else 0,
        "trades": int(len(trades)),
        "final_equity": float(equity),
        "spy_final_equity": float(spy_equity),
        "total_return": float(equity / 50_000.0 - 1.0),
        "spy_total_return": float(spy_equity / 50_000.0 - 1.0),
        "active_alpha_return": float(equity / 50_000.0 - spy_equity / 50_000.0),
        "max_drawdown": max_drawdown(curve["equity"].astype(float).tolist()) if not curve.empty else 0.0,
        "trade_profit_rate": float((trade_returns > 0).mean()) if len(trade_returns) else 0.0,
        "mean_patch_loss": float(np.mean(patch_losses)) if patch_losses else 0.0,
    }
    return summary, curve, trades, preds


def evaluate_ensemble(
    names: tuple[str, ...],
    predictions: dict[str, pd.DataFrame],
    df: pd.DataFrame,
    config: Config,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    pred = None
    for name in names:
        cols = ["timestamp", "row", "pred_utility", "pred_profit", "pred_crash", "prob_sell", "prob_hold", "prob_buy"]
        part = predictions[name][cols].copy()
        if pred is None:
            pred = part
        else:
            pred = pred.merge(part, on=["timestamp", "row"], suffixes=("", f"_{name}"))
    if pred is None or pred.empty:
        return {}, pd.DataFrame(), pd.DataFrame()
    base_cols = ["pred_utility", "pred_profit", "pred_crash", "prob_sell", "prob_hold", "prob_buy"]
    for col in base_cols:
        cols = [c for c in pred.columns if c == col or c.startswith(f"{col}_")]
        pred[col] = pred[cols].mean(axis=1)
    equity = 50_000.0
    spy_equity = 50_000.0
    curve_rows = []
    trade_rows = []
    state = PortfolioState(config)
    for i, (ts, group) in enumerate(pred.groupby("timestamp", sort=True), start=1):
        state.begin_interval(pd.Timestamp(ts))
        selected = select_positions(group[["row"] + base_cols].copy(), df, config, state, i)
        rows = group["row"].to_numpy(np.int64)
        spy_ret = float(df.loc[rows, "future_spy_return"].dropna().iloc[0]) if len(rows) else 0.0
        if selected.empty:
            portfolio_ret = 0.0
            symbols: list[str] = []
        else:
            portfolio_ret = float(config.portfolio_exposure * (selected["future_return"].astype(float).mean() - config.roundtrip_cost))
            symbols = selected["symbol"].astype(str).tolist()
            for _, row in selected.iterrows():
                trade_rows.append({
                    "timestamp": ts,
                    "model": "+".join(names),
                    "symbol": row["symbol"],
                    "pred_score": float(row["pred_score"]),
                    "pred_profit": float(row["pred_profit"]),
                    "pred_crash": float(row["pred_crash"]),
                    "prob_buy": float(row["prob_buy"]),
                    "falling_signals": int(row.get("falling_signals", 0)),
                    "future_return": float(row["future_return"]),
                    "future_spy_return": spy_ret,
                    "future_alpha": float(row["future_return"]) - spy_ret,
                })
        equity *= 1.0 + portfolio_ret
        spy_equity *= 1.0 + spy_ret
        state.record_interval(selected, portfolio_ret, spy_ret, i)
        curve_rows.append({
            "timestamp": ts,
            "model": "+".join(names),
            "equity": equity,
            "spy_equity": spy_equity,
            "portfolio_return": portfolio_ret,
            "spy_return": spy_ret,
            "symbols": ",".join(symbols),
        })
    curve = pd.DataFrame(curve_rows)
    trades = pd.DataFrame(trade_rows)
    trade_returns = trades["future_return"].astype(float).to_numpy() if not trades.empty else np.array([])
    summary = {
        "model": "+".join(names),
        "ensemble_size": len(names),
        "decision_intervals": int(len(curve)),
        "active_intervals": int((curve["symbols"].astype(str) != "").sum()) if not curve.empty else 0,
        "trades": int(len(trades)),
        "final_equity": float(equity),
        "spy_final_equity": float(spy_equity),
        "total_return": float(equity / 50_000.0 - 1.0),
        "spy_total_return": float(spy_equity / 50_000.0 - 1.0),
        "active_alpha_return": float(equity / 50_000.0 - spy_equity / 50_000.0),
        "max_drawdown": max_drawdown(curve["equity"].astype(float).tolist()) if not curve.empty else 0.0,
        "trade_profit_rate": float((trade_returns > 0).mean()) if len(trade_returns) else 0.0,
    }
    return summary, curve, trades


def plot_results(curves: pd.DataFrame, output: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if curves.empty:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    n_models = curves["model"].nunique()
    fig_h = max(7.0, min(18.0, 5.5 + 0.16 * n_models))
    fig, ax = plt.subplots(figsize=(14, fig_h))
    for name, group in curves.groupby("model", sort=False):
        group = group.copy()
        group["timestamp"] = pd.to_datetime(group["timestamp"])
        ax.plot(group["timestamp"], group["equity"], linewidth=1.05, alpha=0.85, label=name[:72])
    first = curves.groupby("timestamp", sort=True)["spy_equity"].first().reset_index()
    first["timestamp"] = pd.to_datetime(first["timestamp"])
    ax.plot(first["timestamp"], first["spy_equity"], linestyle="--", color="black", linewidth=1.8, label="SPY")
    ax.set_title(title)
    ax.set_ylabel("Equity ($)")
    ax.grid(alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7, frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0.0, 0.0, 0.74, 1.0))
    fig.savefig(output, dpi=140)
    plt.close(fig)


def plot_leaderboard(leaderboard: pd.DataFrame, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if leaderboard.empty:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    board = leaderboard.copy().sort_values("active_alpha_return", ascending=True)
    labels = board["model"].astype(str)
    y = np.arange(len(board))
    fig_h = max(8.0, min(24.0, 0.28 * len(board) + 2.0))
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.barh(y - 0.18, board["total_return"].astype(float) * 100.0, height=0.34, label="Strategy return")
    ax.barh(y + 0.18, board["active_alpha_return"].astype(float) * 100.0, height=0.34, label="Active alpha vs SPY")
    spy = float(board["spy_total_return"].dropna().iloc[0]) * 100.0 if "spy_total_return" in board else 0.0
    ax.axvline(spy, color="black", linestyle="--", linewidth=1.2, label=f"SPY {spy:.2f}%")
    ax.axvline(0.0, color="gray", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Return (%)")
    ax.set_title("15-minute transformer performance leaderboard: every model and ensemble")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def write_performance_charts(curves: pd.DataFrame, leaderboard: pd.DataFrame, output_dir: Path, docs_dir: Path) -> None:
    individual = curves[curves["model"].isin(MODEL_NAMES)].copy()
    ensembles = curves[~curves["model"].isin(MODEL_NAMES)].copy()
    plot_results(
        curves,
        output_dir / "equity_compare_all_models_and_combinations.png",
        "15-minute transformers: all individual models and ensemble combinations",
    )
    plot_results(
        individual,
        output_dir / "equity_compare_individual_transformers.png",
        "15-minute transformers: individual model performance",
    )
    plot_results(
        ensembles,
        output_dir / "equity_compare_all_combinations.png",
        "15-minute transformers: all ensemble combinations",
    )
    plot_leaderboard(leaderboard, output_dir / "performance_leaderboard_all_models_and_combinations.png")
    plot_results(
        curves,
        docs_dir / "transformer_15m_equity_all_models_and_combinations.png",
        "15-minute transformers: all individual models and ensemble combinations",
    )
    plot_results(
        individual,
        docs_dir / "transformer_15m_equity_individuals.png",
        "15-minute transformers: individual model performance",
    )
    plot_results(
        ensembles,
        docs_dir / "transformer_15m_equity_combinations.png",
        "15-minute transformers: all ensemble combinations",
    )
    plot_leaderboard(leaderboard, docs_dir / "transformer_15m_performance_leaderboard.png")


def run(config: Config) -> dict:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(config.device)
    df = load_dataset(config)
    row_indices = valid_row_indices(df, config.seq_len)
    if len(row_indices) == 0:
        raise RuntimeError("no sequence rows available")
    row_ts = pd.to_datetime(df.loc[row_indices, "timestamp"], utc=True)
    first_ts = df.loc[row_indices, "timestamp"].min()
    if config.eval_start:
        eval_start = pd.Timestamp(config.eval_start, tz="UTC")
        train_end = pd.Timestamp(config.train_end, tz="UTC") if config.train_end else eval_start
    else:
        eval_start = first_ts + pd.Timedelta(days=config.warmup_days)
        train_end = eval_start
    train_start = pd.Timestamp(config.train_start, tz="UTC") if config.train_start else None
    eval_end = pd.Timestamp(config.eval_end, tz="UTC") if config.eval_end else None

    train_mask = row_ts.lt(train_end)
    if train_start is not None:
        train_mask &= row_ts.ge(train_start)
    eval_mask = row_ts.ge(eval_start)
    if eval_end is not None:
        eval_mask &= row_ts.le(eval_end)
    train_rows = row_indices[train_mask.to_numpy()]
    eval_rows = row_indices[eval_mask.to_numpy()]
    if len(train_rows) == 0 or len(eval_rows) == 0:
        raise RuntimeError("not enough warmup/eval rows")
    arrays = make_arrays(df, train_rows)
    eval_rows_by_ts = {
        pd.Timestamp(ts): group.index.to_numpy(np.int64)
        for ts, group in df.loc[eval_rows].groupby("timestamp", sort=True)
    }
    model_names = MODEL_NAMES if config.models == "all" else [m.strip() for m in config.models.split(",") if m.strip()]
    unknown = sorted(set(model_names) - set(MODEL_NAMES))
    if unknown:
        raise ValueError(f"unknown models: {unknown}")
    print(
        f"[15m] device={device} models={model_names} train_rows={len(train_rows):,} "
        f"eval_rows={len(eval_rows):,} eval_intervals={len(eval_rows_by_ts):,} "
        f"train_start={train_start} train_end={train_end} eval_start={eval_start} eval_end={eval_end}",
        flush=True,
    )
    summaries = []
    curves = []
    trades = []
    predictions: dict[str, pd.DataFrame] = {}
    for name in model_names:
        print(f"[15m] training {name}", flush=True)
        model = make_model(name, len(BASE_FEATURES), int(df["symbol_id"].max()) + 1, config).to(device)
        history = train_initial(model, arrays, train_rows, config, device)
        summary, curve, trade_df, pred = evaluate_online_model(name, model, df, arrays, eval_rows_by_ts, config, device)
        summary["initial_train_history"] = history
        summaries.append(summary)
        curves.append(curve)
        trades.append(trade_df)
        predictions[name] = pred
        torch.save(
            {
                "model": name,
                "state_dict": model.state_dict(),
                "feature_cols": BASE_FEATURES,
                "x_mean": arrays["mean"],
                "x_std": arrays["std"],
                "config": asdict(config),
                "summary": summary,
            },
            output_dir / f"{name}.pt",
        )
        print(f"[15m] {name} return={summary['total_return']:.2%} spy={summary['spy_total_return']:.2%}", flush=True)

    ensemble_summaries = []
    max_size = max(1, min(config.max_ensemble_size, len(model_names)))
    for size in range(2, max_size + 1):
        for names in itertools.combinations(model_names, size):
            summary, curve, trade_df = evaluate_ensemble(names, predictions, df, config)
            if summary:
                ensemble_summaries.append(summary)
                curves.append(curve)
                trades.append(trade_df)
    all_curves = pd.concat([c for c in curves if not c.empty], ignore_index=True)
    all_trades = pd.concat([t for t in trades if not t.empty], ignore_index=True) if any(not t.empty for t in trades) else pd.DataFrame()
    result = {
        "config": asdict(config),
        "dataset_rows": int(len(df)),
        "symbols": int(df["symbol"].nunique()),
        "train_rows": int(len(train_rows)),
        "eval_rows": int(len(eval_rows)),
        "train_start": str(train_start) if train_start is not None else "",
        "train_end": str(train_end),
        "eval_start": str(eval_start),
        "eval_end": str(eval_end) if eval_end is not None else "",
        "models": summaries,
        "ensembles": ensemble_summaries,
        "warning": "research_only_15m_online_patched_transformer_harness",
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, default=str))
    all_curves.to_csv(output_dir / "equity_curves.csv", index=False)
    all_trades.to_csv(output_dir / "trades.csv", index=False)
    leaderboard = pd.DataFrame(summaries + ensemble_summaries).sort_values("active_alpha_return", ascending=False)
    leaderboard.to_csv(output_dir / "leaderboard.csv", index=False)
    write_performance_charts(all_curves, leaderboard, output_dir, Path("docs"))
    plot_results(
        all_curves,
        output_dir / "equity_compare.png",
        "15-minute online patched transformer comparison",
    )
    plot_results(
        all_curves,
        Path("docs/transformer_15m_equity_compare.png"),
        "15-minute online patched transformer comparison",
    )
    print(json.dumps(result, indent=2, default=str), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="checkpoints/transformer_15m/shared_15m_dataset.parquet")
    parser.add_argument("--output-dir", default="checkpoints/transformer_15m/exp1_all_models")
    parser.add_argument("--build-dataset", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--start-date", default="2020-11-03")
    parser.add_argument("--end-date", default="2026-05-16")
    parser.add_argument("--symbol-limit", type=int, default=40)
    parser.add_argument("--universe-mode", default="alphabetical", choices=["alphabetical", "top_volume_valuation"])
    parser.add_argument("--universe-rank-cache", default="checkpoints/transformer_15m/top_volume_valuation_universe.csv")
    parser.add_argument("--min-rows", type=int, default=3000)
    parser.add_argument("--interval-minutes", type=int, default=15)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--warmup-days", type=int, default=365)
    parser.add_argument("--train-start", default="")
    parser.add_argument("--train-end", default="")
    parser.add_argument("--eval-start", default="")
    parser.add_argument("--eval-end", default="")
    parser.add_argument("--initial-epochs", type=int, default=8)
    parser.add_argument("--patch-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--max-train-samples", type=int, default=40_000)
    parser.add_argument("--max-eval-intervals", type=int, default=1_500)
    parser.add_argument("--min-close", type=float, default=5.0)
    parser.add_argument("--max-abs-return", type=float, default=0.08)
    parser.add_argument("--roundtrip-cost", type=float, default=0.0008)
    parser.add_argument("--max-positions", type=int, default=3)
    parser.add_argument("--min-pred-profit", type=float, default=0.53)
    parser.add_argument("--max-pred-crash", type=float, default=0.40)
    parser.add_argument("--min-buy-prob", type=float, default=0.40)
    parser.add_argument("--min-pred-score", type=float, default=-1e9)
    parser.add_argument("--min-pred-utility", type=float, default=-1e9)
    parser.add_argument("--portfolio-exposure", type=float, default=1.0)
    parser.add_argument("--daily-loss-stop", type=float, default=-1.0)
    parser.add_argument("--symbol-cooldown-loss", type=float, default=-1.0)
    parser.add_argument("--symbol-cooldown-intervals", type=int, default=0)
    parser.add_argument("--symbol-daily-cap", type=int, default=0)
    parser.add_argument("--spy-momentum-window", type=int, default=0)
    parser.add_argument("--spy-momentum-min-return", type=float, default=0.0)
    parser.add_argument("--strategy-momentum-window", type=int, default=0)
    parser.add_argument("--strategy-momentum-min-return", type=float, default=0.0)
    parser.add_argument("--filter-falling-stocks", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--falling-filter-min-signals", type=int, default=3)
    parser.add_argument("--falling-max-ret-4", type=float, default=0.0)
    parser.add_argument("--falling-max-ret-16", type=float, default=0.0)
    parser.add_argument("--falling-max-trend-slope-8", type=float, default=0.0)
    parser.add_argument("--falling-max-trend-slope-16", type=float, default=0.0)
    parser.add_argument("--falling-max-ma8-dist", type=float, default=0.0)
    parser.add_argument("--falling-max-ma26-dist", type=float, default=0.0)
    parser.add_argument("--falling-max-algo-momentum-vote", type=float, default=0.0)
    parser.add_argument("--models", default="all")
    parser.add_argument("--max-ensemble-size", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    run(Config(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
