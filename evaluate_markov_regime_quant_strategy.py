"""Walk-forward Markov-regime quant strategy inspired by ZVMTeDBmSrI.

The video's useful core is small and testable:

1. Convert price history into bull / sideways / bear regimes.
2. Estimate a transition matrix from past regimes only.
3. Forecast next regime probabilities from the current regime.
4. Trade from the scalar signal P(bull) - P(bear), with size tied to confidence.
5. Use walk-forward evaluation so no future data leaks into the matrix.

This script applies that idea to the cached 15-minute equity feature datasets.
Regime estimation is daily, then the daily signal is used for intraday entries.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


STATE_BEAR = 0
STATE_SIDE = 1
STATE_BULL = 2


@dataclass(frozen=True)
class Config:
    dataset: str
    output_dir: str
    eval_start: str
    eval_end: str
    symbols: str
    top_symbols_limit: int
    universe_rank_cache: str
    regime_window_days: int
    bull_threshold: float
    bear_threshold: float
    min_transition_days: int
    laplace: float
    forecast_horizon_days: int
    min_signal: float
    max_positions: int
    portfolio_exposure: float
    signal_full_exposure: float
    roundtrip_cost: float
    min_close: float
    max_abs_return: float
    require_spy_positive_signal: bool
    require_adaptive_confirmation: bool
    adaptive_min_history_days: int
    filter_falling_stocks: bool
    falling_max_ret_4: float
    falling_max_ret_16: float
    falling_max_ma26_dist: float
    falling_max_trend_slope_16: float
    falling_min_count: int
    trade_cadence: str
    regime_source: str
    transition_lookback_days: int
    transition_halflife_days: float


def max_drawdown(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    return float((values / values.cummax() - 1.0).min())


def sharpe(returns: pd.Series, bars_per_year: int = 252 * 26) -> float:
    r = returns.astype(float)
    std = float(r.std())
    if std < 1e-12:
        return 0.0
    return float(np.sqrt(bars_per_year) * r.mean() / std)


def selected_symbols(config: Config) -> list[str]:
    explicit = [s.strip().upper() for s in config.symbols.split(",") if s.strip()]
    if explicit:
        return explicit
    if config.top_symbols_limit <= 0:
        return []
    path = Path(config.universe_rank_cache)
    if not path.exists():
        return []
    ranked = pd.read_csv(path)
    if "symbol" not in ranked.columns:
        return []
    return ranked["symbol"].astype(str).str.upper().head(config.top_symbols_limit).tolist()


def manual_state(lookback_return: pd.Series, config: Config) -> pd.Series:
    values = lookback_return.astype(float).to_numpy()
    states = np.full(len(values), STATE_SIDE, dtype=np.int8)
    states[values >= config.bull_threshold] = STATE_BULL
    states[values <= config.bear_threshold] = STATE_BEAR
    states[np.isnan(values)] = -1
    return pd.Series(states, index=lookback_return.index)


def adaptive_state_for_symbol(group: pd.DataFrame, config: Config) -> pd.Series:
    """No-lookahead unsupervised confirmation via expanding return quantiles.

    This is deliberately called "adaptive" rather than a full HMM: hmmlearn is
    not installed in this environment. It still captures the transcript's key
    point that bull/bear thresholds should be checked against an unlabeled,
    data-derived regime assignment.
    """
    values = group["lookback_return"].astype(float).to_numpy()
    out = np.full(len(values), -1, dtype=np.int8)
    for i, value in enumerate(values):
        if not np.isfinite(value) or i < config.adaptive_min_history_days:
            continue
        past = values[:i]
        past = past[np.isfinite(past)]
        if len(past) < config.adaptive_min_history_days:
            continue
        q1, q2 = np.quantile(past, [1.0 / 3.0, 2.0 / 3.0])
        if value <= q1:
            out[i] = STATE_BEAR
        elif value >= q2:
            out[i] = STATE_BULL
        else:
            out[i] = STATE_SIDE
    return pd.Series(out, index=group.index)


def add_adaptive_state(daily: pd.DataFrame, config: Config) -> pd.DataFrame:
    daily = daily.copy()
    daily["adaptive_state"] = np.int8(-1)
    for _, group in daily.groupby("symbol", sort=False):
        daily.loc[group.index, "adaptive_state"] = adaptive_state_for_symbol(group, config).astype(np.int8)
    return daily


def transition_matrix(states: np.ndarray, laplace: float, halflife_days: float = 0.0) -> np.ndarray:
    counts = np.full((3, 3), laplace, dtype=np.float64)
    pairs = list(zip(states[:-1], states[1:]))
    if halflife_days and halflife_days > 0.0 and pairs:
        ages = np.arange(len(pairs) - 1, -1, -1, dtype=np.float64)
        weights = np.power(0.5, ages / float(halflife_days))
    else:
        weights = np.ones(len(pairs), dtype=np.float64)
    for (current, nxt), weight in zip(pairs, weights):
        if current >= 0 and nxt >= 0:
            counts[int(current), int(nxt)] += float(weight)
    return counts / counts.sum(axis=1, keepdims=True)


def matrix_power_forecast(matrix: np.ndarray, horizon: int) -> np.ndarray:
    horizon = max(int(horizon), 1)
    if horizon == 1:
        return matrix
    return np.linalg.matrix_power(matrix, horizon)


def daily_from_intraday(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    agg = {"close": ("close", "last"), "volume": ("volume", "sum")}
    if "open" in df.columns:
        agg["open"] = ("open", "first")
    else:
        agg["open"] = ("close", "first")
    if "high" in df.columns:
        agg["high"] = ("high", "max")
    else:
        agg["high"] = ("close", "max")
    if "low" in df.columns:
        agg["low"] = ("low", "min")
    else:
        agg["low"] = ("close", "min")
    daily = (
        df.sort_values(["symbol", "timestamp"])
        .assign(date=lambda x: x["timestamp"].dt.date)
        .groupby(["symbol", "date"], sort=True)
        .agg(**agg)
        .reset_index()
    )
    daily["symbol"] = daily["symbol"].astype(str).str.upper()
    daily["prev_close"] = daily.groupby("symbol")["close"].shift(1)
    daily["gap_return"] = daily["open"].astype(float) / daily["prev_close"].replace(0.0, np.nan).astype(float) - 1.0
    daily["open_to_close_return"] = daily["close"].astype(float) / daily["open"].replace(0.0, np.nan).astype(float) - 1.0
    daily["intraday_range"] = daily["high"].astype(float) / daily["low"].replace(0.0, np.nan).astype(float) - 1.0
    daily["daily_return"] = daily.groupby("symbol")["close"].pct_change()
    daily["lookback_return"] = daily.groupby("symbol")["close"].pct_change(config.regime_window_days)
    daily["manual_state"] = manual_state(daily["lookback_return"], config)
    return add_adaptive_state(daily, config)


def spy_daily_from_intraday(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    if "spy_close" not in df.columns and not (df["symbol"].astype(str).str.upper() == "SPY").any():
        return pd.DataFrame()
    if "spy_close" in df.columns:
        spy_source = df.sort_values("timestamp").assign(
            date=lambda x: x["timestamp"].dt.date,
            symbol="SPY",
            close=lambda x: x["spy_close"].astype(float),
        )
    else:
        spy_source = df[df["symbol"].astype(str).str.upper() == "SPY"].sort_values("timestamp").assign(
            date=lambda x: x["timestamp"].dt.date,
            symbol="SPY",
        )
    spy_agg = {"close": ("close", "last"), "volume": ("volume", "size")}
    if "open" in spy_source.columns:
        spy_agg["open"] = ("open", "first")
    else:
        spy_agg["open"] = ("close", "first")
    if "high" in spy_source.columns:
        spy_agg["high"] = ("high", "max")
    else:
        spy_agg["high"] = ("close", "max")
    if "low" in spy_source.columns:
        spy_agg["low"] = ("low", "min")
    else:
        spy_agg["low"] = ("close", "min")
    spy = spy_source.groupby(["symbol", "date"], sort=True).agg(**spy_agg).reset_index()
    spy["prev_close"] = spy.groupby("symbol")["close"].shift(1)
    spy["gap_return"] = spy["open"].astype(float) / spy["prev_close"].replace(0.0, np.nan).astype(float) - 1.0
    spy["open_to_close_return"] = spy["close"].astype(float) / spy["open"].replace(0.0, np.nan).astype(float) - 1.0
    spy["intraday_range"] = spy["high"].astype(float) / spy["low"].replace(0.0, np.nan).astype(float) - 1.0
    spy["daily_return"] = spy.groupby("symbol")["close"].pct_change()
    spy["lookback_return"] = spy.groupby("symbol")["close"].pct_change(config.regime_window_days)
    spy["manual_state"] = manual_state(spy["lookback_return"], config)
    return add_adaptive_state(spy, config)


def build_daily_signals(daily: pd.DataFrame, config: Config) -> pd.DataFrame:
    rows: list[dict] = []
    for symbol, group in daily.sort_values(["symbol", "date"]).groupby("symbol", sort=False):
        g = group.reset_index(drop=True)
        manual_states = g["manual_state"].to_numpy(np.int8)
        adaptive = g["adaptive_state"].to_numpy(np.int8)
        if config.regime_source == "adaptive":
            states = adaptive.copy()
            states[states < 0] = manual_states[states < 0]
        elif config.regime_source == "fixed":
            states = manual_states
        else:
            raise ValueError(f"unknown regime source {config.regime_source}")
        for i in range(1, len(g)):
            target_date = g.loc[i, "date"]
            current_state = int(states[i - 1])
            current_adaptive = int(adaptive[i - 1])
            if current_state < 0 or i < config.min_transition_days:
                continue
            start = 0 if config.transition_lookback_days <= 0 else max(0, i - config.transition_lookback_days)
            history = states[start:i]
            matrix = transition_matrix(history, config.laplace, config.transition_halflife_days)
            forecast = matrix_power_forecast(matrix, config.forecast_horizon_days)[current_state]
            signal = float(forecast[STATE_BULL] - forecast[STATE_BEAR])
            rows.append(
                {
                    "symbol": symbol,
                    "date": target_date,
                    "daily_return": float(g.loc[i, "daily_return"]) if np.isfinite(g.loc[i, "daily_return"]) else 0.0,
                    "close": float(g.loc[i, "close"]),
                    "current_state": current_state,
                    "manual_state": int(manual_states[i - 1]),
                    "adaptive_state": current_adaptive,
                    "p_bear": float(forecast[STATE_BEAR]),
                    "p_side": float(forecast[STATE_SIDE]),
                    "p_bull": float(forecast[STATE_BULL]),
                    "markov_signal": signal,
                    "state_persistence": float(forecast[current_state]),
                    "adaptive_confirms": bool(
                        current_adaptive == int(manual_states[i - 1])
                        or (int(manual_states[i - 1]) == STATE_BULL and current_adaptive == STATE_SIDE)
                    ),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def falling_count(group: pd.DataFrame, config: Config) -> pd.Series:
    signals = []
    if "ret_4" in group.columns:
        signals.append(group["ret_4"].astype(float).lt(config.falling_max_ret_4))
    if "ret_16" in group.columns:
        signals.append(group["ret_16"].astype(float).lt(config.falling_max_ret_16))
    if "ma26_dist" in group.columns:
        signals.append(group["ma26_dist"].astype(float).lt(config.falling_max_ma26_dist))
    if "trend_slope_16" in group.columns:
        signals.append(group["trend_slope_16"].astype(float).lt(config.falling_max_trend_slope_16))
    if not signals:
        return pd.Series(0, index=group.index, dtype=int)
    return pd.concat(signals, axis=1).sum(axis=1).astype(int)


def normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "close" not in df.columns and "price" in df.columns:
        df["close"] = df["price"]
    if "volume" not in df.columns:
        df["volume"] = 1.0
    if "symbol" not in df.columns:
        raise ValueError("dataset must contain a symbol column")
    if "timestamp" not in df.columns:
        raise ValueError("dataset must contain a timestamp column")
    return df


def read_dataset(config: Config, daily_only: bool = False) -> pd.DataFrame:
    path = Path(config.dataset)
    symbols = selected_symbols(config)
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
        if symbols:
            keep = set(symbols)
            keep.add("SPY")
            files = [p for p in files if p.stem.upper() in keep]
        columns = ["symbol", "timestamp"]
        sample_cols = set(pd.read_parquet(files[0], columns=None).columns) if files else set()
        if "close" in sample_cols:
            columns.append("close")
        elif "price" in sample_cols:
            columns.append("price")
        for col in ["open", "high", "low"]:
            if col in sample_cols:
                columns.append(col)
        if "volume" in sample_cols:
            columns.append("volume")
        frames = []
        for file in files:
            df = pd.read_parquet(file, columns=[c for c in columns if c in sample_cols])
            if "symbol" not in df.columns:
                name = file.stem
                df["symbol"] = name.removesuffix("_1m").removesuffix("_1d").upper()
            frames.append(df)
        if not frames:
            return pd.DataFrame(columns=["symbol", "timestamp", "close", "volume"])
        df = pd.concat(frames, ignore_index=True)
    else:
        if daily_only:
            cols = set(pd.read_parquet(path, columns=None).columns)
            wanted = [
                "symbol", "timestamp", "open", "high", "low", "close", "price", "volume", "spy_close",
                "future_return", "future_spy_return",
            ]
            df = pd.read_parquet(path, columns=[c for c in wanted if c in cols])
        else:
            df = pd.read_parquet(path)
    df = normalize_ohlcv_columns(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["symbol"] = df["symbol"].astype(str).str.upper()
    if symbols:
        allowed = set(symbols)
        if "SPY" in df["symbol"].unique():
            allowed.add("SPY")
        df = df[df["symbol"].isin(allowed)].copy()
    return df


def load_frame(config: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = read_dataset(config, daily_only=False)
    if Path(config.dataset).is_dir():
        raise ValueError("15m cadence requires a feature parquet file, not a per-symbol directory")
    df = df[df["close"].astype(float).ge(config.min_close)].copy()
    df = df[
        df["future_return"].astype(float).between(-config.max_abs_return, config.max_abs_return)
        & df["future_spy_return"].astype(float).between(-config.max_abs_return, config.max_abs_return)
    ].copy()
    daily = daily_from_intraday(df, config)
    signals = build_daily_signals(daily, config)
    spy_signals = build_daily_signals(spy_daily_from_intraday(df, config), config)
    if not spy_signals.empty:
        spy_signals = spy_signals[["date", "markov_signal", "p_bull", "p_bear"]].rename(
            columns={
                "markov_signal": "spy_markov_signal",
                "p_bull": "spy_p_bull",
                "p_bear": "spy_p_bear",
            }
        )
    df["date"] = df["timestamp"].dt.date
    merged = df.merge(signals, on=["symbol", "date"], how="left")
    if not spy_signals.empty:
        merged = merged.merge(spy_signals, on="date", how="left")
    else:
        merged["spy_markov_signal"] = 0.0
    merged["falling_count"] = falling_count(merged, config)
    merged = merged[
        (merged["timestamp"] >= pd.Timestamp(config.eval_start, tz="UTC"))
        & (merged["timestamp"] <= pd.Timestamp(config.eval_end, tz="UTC"))
    ].copy()
    return merged.sort_values(["timestamp", "symbol"]).reset_index(drop=True), daily, signals


def load_daily_eval_frame(config: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = read_dataset(config, daily_only=True)
    df = df[df["close"].astype(float).ge(config.min_close)].copy()
    daily = daily_from_intraday(df, config)
    signals = build_daily_signals(daily, config)
    spy_daily = spy_daily_from_intraday(df, config)
    spy_signals = build_daily_signals(spy_daily, config)
    if not spy_signals.empty:
        spy_signals = spy_signals[["date", "markov_signal", "p_bull", "p_bear", "daily_return"]].rename(
            columns={
                "markov_signal": "spy_markov_signal",
                "p_bull": "spy_p_bull",
                "p_bear": "spy_p_bear",
                "daily_return": "spy_daily_return",
            }
        )
    elif not spy_daily.empty:
        spy_returns = spy_daily[["date", "daily_return"]].rename(columns={"daily_return": "spy_daily_return"})
        spy_signals = spy_returns.assign(spy_markov_signal=0.0, spy_p_bull=0.0, spy_p_bear=0.0)
    else:
        spy_signals = pd.DataFrame(columns=["date", "spy_daily_return", "spy_markov_signal", "spy_p_bull", "spy_p_bear"])
    frame = signals.merge(spy_signals, on="date", how="left")
    frame["timestamp"] = pd.to_datetime(frame["date"].astype(str), utc=True)
    frame = frame[
        (frame["timestamp"] >= pd.Timestamp(config.eval_start, tz="UTC"))
        & (frame["timestamp"] <= pd.Timestamp(config.eval_end, tz="UTC"))
        & frame["daily_return"].astype(float).between(-config.max_abs_return, config.max_abs_return)
        & frame["spy_daily_return"].fillna(0.0).astype(float).between(-config.max_abs_return, config.max_abs_return)
    ].copy()
    return frame.sort_values(["date", "symbol"]).reset_index(drop=True), daily, signals


def strategy_candidates(group: pd.DataFrame, config: Config, variant: str) -> pd.DataFrame:
    candidates = group[group["markov_signal"].astype(float) >= config.min_signal].copy()
    if config.require_spy_positive_signal or "spy_gated" in variant:
        candidates = candidates[candidates["spy_markov_signal"].fillna(0.0).astype(float) > 0.0].copy()
    if config.require_adaptive_confirmation or "confirmed" in variant:
        candidates = candidates[candidates["adaptive_confirms"].fillna(False).astype(bool)].copy()
    if config.filter_falling_stocks:
        candidates = candidates[candidates["falling_count"].astype(int) < config.falling_min_count].copy()
    if candidates.empty:
        return candidates
    score = candidates["markov_signal"].astype(float).copy()
    if "algo_fused" in variant:
        for col, weight in [
            ("rel_spy_16", 0.15),
            ("trend_slope_26", 0.10),
            ("algo_trend_quality", 0.10),
            ("algo_momentum_vote", 0.10),
        ]:
            if col in candidates.columns:
                values = candidates[col].astype(float)
                std = float(values.std())
                z = (values - float(values.mean())) / std if std > 1e-12 else values * 0.0
                score = score + weight * z.clip(-3.0, 3.0)
    candidates["score"] = score
    return candidates.sort_values("score", ascending=False).head(config.max_positions)


def evaluate_variant(df: pd.DataFrame, config: Config, variant: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    equity = 50_000.0
    spy_equity = 50_000.0
    curve_rows: list[dict] = []
    trade_rows: list[dict] = []
    for ts_value, group in df.groupby("timestamp", sort=True):
        ts = pd.Timestamp(ts_value)
        spy_ret = float(group["future_spy_return"].dropna().iloc[0]) if not group.empty else 0.0
        selected = strategy_candidates(group, config, variant)
        symbols: list[str] = []
        if selected.empty:
            portfolio_ret = 0.0
        else:
            signal = selected["markov_signal"].astype(float).clip(lower=0.0)
            raw_weights = signal / max(config.signal_full_exposure, 1e-9)
            raw_weights = raw_weights.clip(upper=1.0)
            if raw_weights.sum() > 0:
                weights = raw_weights / raw_weights.sum() * min(config.portfolio_exposure, float(raw_weights.sum()))
            else:
                weights = pd.Series(1.0 / len(selected), index=selected.index)
            future = selected["future_return"].astype(float)
            portfolio_ret = float((weights * (future - config.roundtrip_cost)).sum())
            symbols = selected["symbol"].astype(str).tolist()
            for idx, row in selected.iterrows():
                trade_rows.append(
                    {
                        "timestamp": str(ts),
                        "variant": variant,
                        "symbol": row["symbol"],
                        "weight": float(weights.loc[idx]),
                        "score": float(row["score"]),
                        "markov_signal": float(row["markov_signal"]),
                        "p_bull": float(row["p_bull"]),
                        "p_bear": float(row["p_bear"]),
                        "current_state": int(row["current_state"]),
                        "adaptive_state": int(row["adaptive_state"]),
                        "spy_markov_signal": float(row.get("spy_markov_signal", 0.0)),
                        "falling_count": int(row["falling_count"]),
                        "future_return": float(row["future_return"]),
                        "future_spy_return": spy_ret,
                    }
                )
        equity *= 1.0 + portfolio_ret
        spy_equity *= 1.0 + spy_ret
        curve_rows.append(
            {
                "timestamp": str(ts),
                "variant": variant,
                "equity": equity,
                "spy_equity": spy_equity,
                "portfolio_return": portfolio_ret,
                "spy_return": spy_ret,
                "symbols": ",".join(symbols),
            }
        )
    curve = pd.DataFrame(curve_rows)
    trades = pd.DataFrame(trade_rows)
    returns = curve["portfolio_return"].astype(float) if not curve.empty else pd.Series(dtype=float)
    spy_returns = curve["spy_return"].astype(float) if not curve.empty else pd.Series(dtype=float)
    summary = {
        "variant": variant,
        "decision_intervals": int(len(curve)),
        "active_intervals": int((curve["symbols"].astype(str) != "").sum()) if not curve.empty else 0,
        "trades": int(len(trades)),
        "final_equity": float(equity),
        "spy_final_equity": float(spy_equity),
        "total_return": float(equity / 50_000.0 - 1.0),
        "spy_total_return": float(spy_equity / 50_000.0 - 1.0),
        "alpha_return": float(equity / 50_000.0 - spy_equity / 50_000.0),
        "max_drawdown": max_drawdown(curve["equity"].astype(float)) if not curve.empty else 0.0,
        "sharpe": sharpe(returns),
        "spy_sharpe": sharpe(spy_returns),
        "trade_profit_rate": float((trades["future_return"].astype(float) > 0.0).mean()) if not trades.empty else 0.0,
        "mean_trade_return": float(trades["future_return"].astype(float).mean()) if not trades.empty else 0.0,
    }
    return summary, curve, trades


def daily_strategy_candidates(group: pd.DataFrame, config: Config, variant: str) -> pd.DataFrame:
    candidates = group[group["markov_signal"].astype(float) >= config.min_signal].copy()
    if config.require_spy_positive_signal or "spy_gated" in variant:
        candidates = candidates[candidates["spy_markov_signal"].fillna(0.0).astype(float) > 0.0].copy()
    if config.require_adaptive_confirmation or "confirmed" in variant:
        candidates = candidates[candidates["adaptive_confirms"].fillna(False).astype(bool)].copy()
    if candidates.empty:
        return candidates
    candidates["score"] = candidates["markov_signal"].astype(float)
    if "algo_fused" in variant:
        candidates["score"] = candidates["score"] + 0.15 * candidates["state_persistence"].astype(float)
    return candidates.sort_values("score", ascending=False).head(config.max_positions)


def evaluate_daily_variant(df: pd.DataFrame, config: Config, variant: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    equity = 50_000.0
    spy_equity = 50_000.0
    curve_rows: list[dict] = []
    trade_rows: list[dict] = []
    for date_value, group in df.groupby("date", sort=True):
        ts = pd.Timestamp(str(date_value), tz="UTC")
        spy_values = group["spy_daily_return"].dropna() if "spy_daily_return" in group.columns else pd.Series(dtype=float)
        spy_ret = float(spy_values.iloc[0]) if not spy_values.empty else 0.0
        selected = daily_strategy_candidates(group, config, variant)
        symbols: list[str] = []
        if selected.empty:
            portfolio_ret = 0.0
        else:
            signal = selected["markov_signal"].astype(float).clip(lower=0.0)
            raw_weights = (signal / max(config.signal_full_exposure, 1e-9)).clip(upper=1.0)
            if raw_weights.sum() > 0:
                weights = raw_weights / raw_weights.sum() * min(config.portfolio_exposure, float(raw_weights.sum()))
            else:
                weights = pd.Series(1.0 / len(selected), index=selected.index)
            daily_ret = selected["daily_return"].astype(float)
            portfolio_ret = float((weights * (daily_ret - config.roundtrip_cost)).sum())
            symbols = selected["symbol"].astype(str).tolist()
            for idx, row in selected.iterrows():
                trade_rows.append(
                    {
                        "timestamp": str(ts),
                        "variant": variant,
                        "symbol": row["symbol"],
                        "weight": float(weights.loc[idx]),
                        "score": float(row["score"]),
                        "markov_signal": float(row["markov_signal"]),
                        "p_bull": float(row["p_bull"]),
                        "p_bear": float(row["p_bear"]),
                        "current_state": int(row["current_state"]),
                        "adaptive_state": int(row["adaptive_state"]),
                        "spy_markov_signal": float(row.get("spy_markov_signal", 0.0)),
                        "daily_return": float(row["daily_return"]),
                        "spy_daily_return": spy_ret,
                    }
                )
        equity *= 1.0 + portfolio_ret
        spy_equity *= 1.0 + spy_ret
        curve_rows.append(
            {
                "timestamp": str(ts),
                "variant": variant,
                "equity": equity,
                "spy_equity": spy_equity,
                "portfolio_return": portfolio_ret,
                "spy_return": spy_ret,
                "symbols": ",".join(symbols),
            }
        )
    curve = pd.DataFrame(curve_rows)
    trades = pd.DataFrame(trade_rows)
    returns = curve["portfolio_return"].astype(float) if not curve.empty else pd.Series(dtype=float)
    spy_returns = curve["spy_return"].astype(float) if not curve.empty else pd.Series(dtype=float)
    summary = {
        "variant": variant,
        "decision_intervals": int(len(curve)),
        "active_intervals": int((curve["symbols"].astype(str) != "").sum()) if not curve.empty else 0,
        "trades": int(len(trades)),
        "final_equity": float(equity),
        "spy_final_equity": float(spy_equity),
        "total_return": float(equity / 50_000.0 - 1.0),
        "spy_total_return": float(spy_equity / 50_000.0 - 1.0),
        "alpha_return": float(equity / 50_000.0 - spy_equity / 50_000.0),
        "max_drawdown": max_drawdown(curve["equity"].astype(float)) if not curve.empty else 0.0,
        "sharpe": sharpe(returns, bars_per_year=252),
        "spy_sharpe": sharpe(spy_returns, bars_per_year=252),
        "trade_profit_rate": float((trades["daily_return"].astype(float) > 0.0).mean()) if not trades.empty else 0.0,
        "mean_trade_return": float(trades["daily_return"].astype(float).mean()) if not trades.empty else 0.0,
    }
    return summary, curve, trades


def plot_results(curves: pd.DataFrame, leaderboard: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    docs = Path("docs")
    docs.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 8))
    for name, group in curves.groupby("variant", sort=False):
        g = group.copy()
        g["timestamp"] = pd.to_datetime(g["timestamp"])
        ax.plot(g["timestamp"], g["equity"], label=name, linewidth=1.25)
    spy = curves.groupby("timestamp", sort=True)["spy_equity"].first().reset_index()
    spy["timestamp"] = pd.to_datetime(spy["timestamp"])
    ax.plot(spy["timestamp"], spy["spy_equity"], "--", color="black", label="SPY", linewidth=1.8)
    ax.set_title("Walk-forward Markov-regime quant strategies vs SPY")
    ax.set_ylabel("Equity ($)")
    ax.grid(alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 0.78, 1.0))
    fig.savefig(output_dir / "markov_regime_equity.png", dpi=140)
    fig.savefig(docs / "markov_regime_equity.png", dpi=140)
    plt.close(fig)

    board = leaderboard.sort_values("alpha_return")
    fig, ax = plt.subplots(figsize=(12, max(4.5, 0.45 * len(board) + 2)))
    y = np.arange(len(board))
    ax.barh(y - 0.18, board["total_return"] * 100, height=0.34, label="Strategy")
    ax.barh(y + 0.18, board["alpha_return"] * 100, height=0.34, label="Alpha vs SPY")
    spy_ret = float(board["spy_total_return"].dropna().iloc[0]) * 100
    ax.axvline(spy_ret, linestyle="--", color="black", label=f"SPY {spy_ret:.2f}%")
    ax.axvline(0.0, color="gray", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(board["variant"])
    ax.set_xlabel("Return (%)")
    ax.set_title("Markov-regime strategy leaderboard")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "markov_regime_leaderboard.png", dpi=140)
    fig.savefig(docs / "markov_regime_leaderboard.png", dpi=140)
    plt.close(fig)


def run(config: Config) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.trade_cadence == "daily":
        df, daily, signals = load_daily_eval_frame(config)
        evaluator = evaluate_daily_variant
    elif config.trade_cadence == "15m":
        df, daily, signals = load_frame(config)
        evaluator = evaluate_variant
    else:
        raise ValueError(f"unknown trade cadence {config.trade_cadence}")
    variants = ["manual", "confirmed", "spy_gated", "confirmed_spy_gated", "algo_fused_spy_gated"]
    summaries = []
    curves = []
    trades = []
    for variant in variants:
        summary, curve, trade_df = evaluator(df, config, variant)
        summaries.append(summary)
        curves.append(curve)
        trades.append(trade_df)
        print(
            f"[markov] {variant} return={summary['total_return']:.2%} "
            f"spy={summary['spy_total_return']:.2%} trades={summary['trades']}",
            flush=True,
        )
    all_curves = pd.concat(curves, ignore_index=True)
    all_trades = pd.concat([t for t in trades if not t.empty], ignore_index=True) if any(not t.empty for t in trades) else pd.DataFrame()
    leaderboard = pd.DataFrame(summaries).sort_values("alpha_return", ascending=False)
    result = {
        "config": asdict(config),
        "rows": int(len(df)),
        "symbols": int(df["symbol"].nunique()),
        "daily_rows": int(len(daily)),
        "signal_rows": int(len(signals)),
        "strategies": summaries,
        "transcript_source": "https://www.youtube.com/watch?v=ZVMTeDBmSrI",
        "transcript_path": "artifacts/transcripts/ZVMTeDBmSrI.txt",
        "warning": "research_only_walk_forward_markov_regime_strategy_not_tradable_advice",
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, default=str))
    leaderboard.to_csv(output_dir / "leaderboard.csv", index=False)
    all_curves.to_csv(output_dir / "equity_curves.csv", index=False)
    all_trades.to_csv(output_dir / "trades.csv", index=False)
    plot_results(all_curves, leaderboard, output_dir)
    print(json.dumps(result, indent=2, default=str), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="checkpoints/transformer_15m/shared_15m_top10_volume_valuation_algo.parquet")
    parser.add_argument("--output-dir", default="checkpoints/transformer_15m/markov_regime_quant_top10_2026")
    parser.add_argument("--eval-start", default="2026-01-01")
    parser.add_argument("--eval-end", default="2026-05-16")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--top-symbols-limit", type=int, default=10)
    parser.add_argument("--universe-rank-cache", default="checkpoints/transformer_15m/top_volume_valuation_universe.csv")
    parser.add_argument("--regime-window-days", type=int, default=20)
    parser.add_argument("--bull-threshold", type=float, default=0.05)
    parser.add_argument("--bear-threshold", type=float, default=-0.05)
    parser.add_argument("--min-transition-days", type=int, default=80)
    parser.add_argument("--laplace", type=float, default=1.0)
    parser.add_argument("--forecast-horizon-days", type=int, default=1)
    parser.add_argument("--min-signal", type=float, default=0.05)
    parser.add_argument("--max-positions", type=int, default=3)
    parser.add_argument("--portfolio-exposure", type=float, default=1.0)
    parser.add_argument("--signal-full-exposure", type=float, default=0.35)
    parser.add_argument("--roundtrip-cost", type=float, default=0.0008)
    parser.add_argument("--min-close", type=float, default=5.0)
    parser.add_argument("--max-abs-return", type=float, default=0.08)
    parser.add_argument("--require-spy-positive-signal", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-adaptive-confirmation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--adaptive-min-history-days", type=int, default=120)
    parser.add_argument("--filter-falling-stocks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--falling-max-ret-4", type=float, default=-0.004)
    parser.add_argument("--falling-max-ret-16", type=float, default=-0.008)
    parser.add_argument("--falling-max-ma26-dist", type=float, default=-0.008)
    parser.add_argument("--falling-max-trend-slope-16", type=float, default=-0.0006)
    parser.add_argument("--falling-min-count", type=int, default=3)
    parser.add_argument("--trade-cadence", choices=["daily", "15m"], default="daily")
    parser.add_argument("--regime-source", choices=["fixed", "adaptive"], default="fixed")
    parser.add_argument("--transition-lookback-days", type=int, default=0)
    parser.add_argument("--transition-halflife-days", type=float, default=0.0)
    run(Config(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
