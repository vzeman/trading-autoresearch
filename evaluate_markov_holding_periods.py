"""Compare adaptive Markov strategies with real multi-day holding periods."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_markov_regime_quant_strategy import (
    Config as MarkovConfig,
    load_daily_eval_frame,
    max_drawdown,
    sharpe,
)


START_EQUITY = 50_000.0


@dataclass(frozen=True)
class HoldingStrategy:
    name: str
    selector: str
    max_hold_days: int
    min_hold_days: int = 1
    exit_signal: float | None = None
    rank_exit_top_n: int = 0
    exit_on_spy_gate: bool = False
    exposure_column: str = ""


def default_markov_config(args: argparse.Namespace) -> MarkovConfig:
    return MarkovConfig(
        dataset=args.dataset,
        output_dir=args.output_dir,
        eval_start=args.eval_start,
        eval_end=args.eval_end,
        symbols=args.symbols,
        top_symbols_limit=args.top_symbols_limit,
        universe_rank_cache=args.universe_rank_cache,
        regime_window_days=args.regime_window_days,
        bull_threshold=args.bull_threshold,
        bear_threshold=args.bear_threshold,
        min_transition_days=args.min_transition_days,
        laplace=args.laplace,
        forecast_horizon_days=args.forecast_horizon_days,
        min_signal=args.min_signal,
        max_positions=args.max_positions,
        portfolio_exposure=args.portfolio_exposure,
        signal_full_exposure=args.signal_full_exposure,
        roundtrip_cost=args.roundtrip_cost,
        min_close=args.min_close,
        max_abs_return=args.max_abs_return,
        require_spy_positive_signal=False,
        require_adaptive_confirmation=False,
        adaptive_min_history_days=args.adaptive_min_history_days,
        filter_falling_stocks=True,
        falling_max_ret_4=-0.004,
        falling_max_ret_16=-0.008,
        falling_max_ma26_dist=-0.008,
        falling_max_trend_slope_16=-0.0006,
        falling_min_count=3,
        trade_cadence="daily",
        regime_source=args.regime_source,
        transition_lookback_days=args.transition_lookback_days,
        transition_halflife_days=args.transition_halflife_days,
    )


def safe_zscore(values: pd.Series) -> pd.Series:
    vals = values.astype(float).replace([np.inf, -np.inf], np.nan)
    std = vals.std(ddof=0)
    if not np.isfinite(std) or std < 1e-12:
        return pd.Series(0.0, index=values.index)
    return (vals - vals.mean()) / std


def add_cross_sectional_scores(frame: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Add leak-resistant alternative model scores from information known before the target day."""
    features = daily.sort_values(["symbol", "date"]).copy()
    if "dollar_volume" not in features.columns:
        features["dollar_volume"] = features["close"].astype(float) * features["volume"].astype(float)
    by_symbol = features.groupby("symbol", sort=False)
    features["ret_3_prev"] = by_symbol["close"].pct_change(3).groupby(features["symbol"]).shift(1)
    features["ret_5_prev"] = by_symbol["close"].pct_change(5).groupby(features["symbol"]).shift(1)
    features["ret_10_prev"] = by_symbol["close"].pct_change(10).groupby(features["symbol"]).shift(1)
    features["ret_20_prev"] = by_symbol["close"].pct_change(20).groupby(features["symbol"]).shift(1)
    features["vol_10_prev"] = by_symbol["daily_return"].rolling(10).std().reset_index(level=0, drop=True).groupby(features["symbol"]).shift(1)
    features["vol_20_prev"] = by_symbol["daily_return"].rolling(20).std().reset_index(level=0, drop=True).groupby(features["symbol"]).shift(1)
    features["avg_vol_5_prev"] = by_symbol["volume"].rolling(5).mean().reset_index(level=0, drop=True).groupby(features["symbol"]).shift(1)
    features["avg_vol_20_prev"] = by_symbol["volume"].rolling(20).mean().reset_index(level=0, drop=True).groupby(features["symbol"]).shift(1)
    features["avg_dollar_vol_5_prev"] = by_symbol["dollar_volume"].rolling(5).mean().reset_index(level=0, drop=True).groupby(features["symbol"]).shift(1)
    features["avg_dollar_vol_20_prev"] = by_symbol["dollar_volume"].rolling(20).mean().reset_index(level=0, drop=True).groupby(features["symbol"]).shift(1)
    features["volume_ratio_prev"] = by_symbol["volume"].shift(1) / features["avg_vol_20_prev"].replace(0.0, np.nan)
    features["volume_ratio_5_20_prev"] = features["avg_vol_5_prev"] / features["avg_vol_20_prev"].replace(0.0, np.nan)
    features["dollar_volume_ratio_5_20_prev"] = features["avg_dollar_vol_5_prev"] / features["avg_dollar_vol_20_prev"].replace(0.0, np.nan)
    features["volume_spike_prev"] = (features["volume_ratio_prev"] - 2.5).clip(lower=0.0)
    features["median_dollar_volume_20_prev"] = (
        by_symbol["dollar_volume"].rolling(20).median().reset_index(level=0, drop=True).groupby(features["symbol"]).shift(1)
    )
    features["roll_max_20_prev"] = by_symbol["close"].rolling(20).max().reset_index(level=0, drop=True).groupby(features["symbol"]).shift(1)
    features["roll_min_20_prev"] = by_symbol["close"].rolling(20).min().reset_index(level=0, drop=True).groupby(features["symbol"]).shift(1)
    features["roll_max_50_prev"] = by_symbol["close"].rolling(50).max().reset_index(level=0, drop=True).groupby(features["symbol"]).shift(1)
    features["close_prev"] = by_symbol["close"].shift(1)
    features["breakout_20_prev"] = features["close_prev"] / features["roll_max_20_prev"].replace(0.0, np.nan) - 1.0
    features["drawdown_20_prev"] = features["close_prev"] / features["roll_max_20_prev"].replace(0.0, np.nan) - 1.0
    features["range_pos_20_prev"] = (
        (features["close_prev"] - features["roll_min_20_prev"])
        / (features["roll_max_20_prev"] - features["roll_min_20_prev"]).replace(0.0, np.nan)
    )
    features["gap_return"] = features.get("gap_return", np.nan)
    features["open_to_close_return"] = features.get("open_to_close_return", np.nan)
    features["gap_return_prev"] = by_symbol["gap_return"].shift(1)
    features["gap_followthrough_prev"] = by_symbol["open_to_close_return"].shift(1)
    features["gap_up_prev"] = features["gap_return_prev"].ge(0.01)
    features["gap_failed_prev"] = features["gap_up_prev"] & features["gap_followthrough_prev"].lt(0.0)
    features["gap_up_count_20_prev"] = (
        features["gap_return"].ge(0.01).astype(float).groupby(features["symbol"]).rolling(20).sum().reset_index(level=0, drop=True).groupby(features["symbol"]).shift(1)
    )
    features["gap_success_count_20_prev"] = (
        (features["gap_return"].ge(0.01) & features["open_to_close_return"].gt(0.0))
        .astype(float)
        .groupby(features["symbol"])
        .rolling(20)
        .sum()
        .reset_index(level=0, drop=True)
        .groupby(features["symbol"])
        .shift(1)
    )
    features["gap_success_rate_20_prev"] = features["gap_success_count_20_prev"] / features["gap_up_count_20_prev"].replace(0.0, np.nan)
    features["gap_over_resistance_20"] = features["open"].astype(float) / features["roll_max_20_prev"].replace(0.0, np.nan) - 1.0
    features["gap_over_resistance_50"] = features["open"].astype(float) / features["roll_max_50_prev"].replace(0.0, np.nan) - 1.0

    spy = features[features["symbol"].astype(str).str.upper() == "SPY"][
        ["date", "ret_5_prev", "ret_10_prev", "ret_20_prev"]
    ].rename(
        columns={
            "ret_5_prev": "spy_ret_5_prev",
            "ret_10_prev": "spy_ret_10_prev",
            "ret_20_prev": "spy_ret_20_prev",
        }
    )
    features = features.merge(spy, on="date", how="left")
    features["rel_ret_5_prev"] = features["ret_5_prev"] - features["spy_ret_5_prev"]
    features["rel_ret_10_prev"] = features["ret_10_prev"] - features["spy_ret_10_prev"]
    features["rel_ret_20_prev"] = features["ret_20_prev"] - features["spy_ret_20_prev"]
    features["price_volume_confirm_prev"] = features["ret_5_prev"] * features["volume_ratio_5_20_prev"]

    score_cols = [
        "ret_3_prev",
        "ret_5_prev",
        "ret_10_prev",
        "ret_20_prev",
        "rel_ret_5_prev",
        "rel_ret_10_prev",
        "rel_ret_20_prev",
        "vol_10_prev",
        "vol_20_prev",
        "volume_ratio_prev",
        "volume_ratio_5_20_prev",
        "dollar_volume_ratio_5_20_prev",
        "volume_spike_prev",
        "price_volume_confirm_prev",
        "median_dollar_volume_20_prev",
        "breakout_20_prev",
        "drawdown_20_prev",
        "range_pos_20_prev",
        "gap_return",
        "gap_return_prev",
        "gap_followthrough_prev",
        "gap_up_count_20_prev",
        "gap_success_rate_20_prev",
        "gap_over_resistance_20",
        "gap_over_resistance_50",
    ]
    for col in score_cols:
        features[f"z_{col}"] = features.groupby("date", sort=False)[col].transform(safe_zscore)

    features["relative_momentum_score"] = (
        0.55 * features["z_rel_ret_5_prev"]
        + 0.35 * features["z_rel_ret_20_prev"]
        + 0.20 * features["z_ret_10_prev"]
        - 0.25 * features["z_vol_10_prev"]
    )
    features["trend_quality_score"] = (
        0.40 * features["z_ret_5_prev"]
        + 0.35 * features["z_ret_20_prev"]
        + 0.25 * features["z_range_pos_20_prev"]
        - 0.35 * features["z_vol_20_prev"]
    )
    features["breakout_quality_score"] = (
        0.65 * features["z_breakout_20_prev"]
        + 0.25 * features["z_volume_ratio_prev"]
        + 0.20 * features["z_rel_ret_5_prev"]
        - 0.30 * features["z_vol_10_prev"]
    )
    features["defensive_trend_score"] = (
        0.45 * features["z_rel_ret_10_prev"]
        + 0.30 * features["z_ret_20_prev"]
        - 0.45 * features["z_vol_20_prev"]
        + 0.20 * features["z_range_pos_20_prev"]
    )
    features["gap_resistance_score"] = (
        0.45 * features["z_gap_return"]
        + 0.45 * features["z_gap_over_resistance_20"]
        + 0.20 * features["z_rel_ret_5_prev"]
        - 0.20 * features["z_vol_10_prev"]
    )
    features["volume_shape_score"] = (
        0.30 * features["z_volume_ratio_5_20_prev"]
        + 0.25 * features["z_dollar_volume_ratio_5_20_prev"]
        + 0.25 * features["z_price_volume_confirm_prev"]
        + 0.15 * features["z_volume_ratio_prev"]
        - 0.20 * features["z_volume_spike_prev"]
    )

    keep = [
        "symbol",
        "date",
        "open",
        "ret_5_prev",
        "ret_20_prev",
        "rel_ret_5_prev",
        "rel_ret_10_prev",
        "rel_ret_20_prev",
        "spy_ret_5_prev",
        "spy_ret_10_prev",
        "spy_ret_20_prev",
        "vol_10_prev",
        "median_dollar_volume_20_prev",
        "volume_ratio_prev",
        "volume_ratio_5_20_prev",
        "dollar_volume_ratio_5_20_prev",
        "volume_spike_prev",
        "price_volume_confirm_prev",
        "gap_return",
        "open_to_close_return",
        "gap_return_prev",
        "gap_followthrough_prev",
        "gap_failed_prev",
        "gap_up_count_20_prev",
        "gap_success_rate_20_prev",
        "gap_over_resistance_20",
        "gap_over_resistance_50",
        "relative_momentum_score",
        "trend_quality_score",
        "breakout_quality_score",
        "defensive_trend_score",
        "gap_resistance_score",
        "volume_shape_score",
    ]
    enriched = frame.merge(features[keep], on=["symbol", "date"], how="left")
    enriched["markov_z_score"] = enriched.groupby("date", sort=False)["markov_signal"].transform(safe_zscore)
    enriched["hybrid_markov_trend_score"] = (
        0.55 * enriched["markov_z_score"].fillna(0.0)
        + 0.30 * enriched["trend_quality_score"].fillna(0.0)
        + 0.15 * enriched["relative_momentum_score"].fillna(0.0)
    )
    enriched["trend_quality_gap_resistance_score"] = (
        0.60 * enriched["trend_quality_score"].fillna(0.0)
        + 0.40 * enriched["gap_resistance_score"].fillna(0.0)
    )
    enriched["trend_quality_avoid_failed_gap_score"] = enriched["trend_quality_score"].fillna(0.0)
    enriched.loc[enriched["gap_failed_prev"].fillna(False).astype(bool), "trend_quality_avoid_failed_gap_score"] -= 0.75
    enriched["trend_quality_volume_shape_score"] = (
        0.72 * enriched["trend_quality_score"].fillna(0.0)
        + 0.28 * enriched["volume_shape_score"].fillna(0.0)
    )
    enriched["trend_quality_avoid_failed_gap_volume_shape_score"] = (
        0.72 * enriched["trend_quality_avoid_failed_gap_score"].fillna(0.0)
        + 0.28 * enriched["volume_shape_score"].fillna(0.0)
    )
    enriched["relative_momentum_volume_shape_score"] = (
        0.75 * enriched["relative_momentum_score"].fillna(0.0)
        + 0.25 * enriched["volume_shape_score"].fillna(0.0)
    )
    spy_ret_5 = enriched["spy_ret_5_prev"].fillna(0.0).astype(float)
    spy_ret_20 = enriched["spy_ret_20_prev"].fillna(0.0).astype(float)
    spy_signal = enriched["spy_markov_signal"].fillna(0.0).astype(float)
    exposure = pd.Series(1.0, index=enriched.index, dtype=float)
    exposure[(spy_ret_5 < 0.0) & (spy_signal <= 0.0)] = 0.65
    exposure[(spy_ret_20 < -0.02) & (spy_signal <= 0.0)] = 0.40
    exposure[(spy_ret_20 < -0.04) & (spy_signal < -0.05)] = 0.20
    enriched["spy_relative_guard_exposure"] = exposure
    return enriched


def score_candidates(group: pd.DataFrame, config: MarkovConfig, selector: str, pool_size: int) -> pd.DataFrame:
    candidates = group[group["symbol"].astype(str).str.upper() != "SPY"].copy()
    score_map = {
        "relative_momentum": "relative_momentum_score",
        "relative_momentum_spy_gated": "relative_momentum_score",
        "relative_momentum_volume_shape": "relative_momentum_volume_shape_score",
        "relative_momentum_volume_shape_spy_gated": "relative_momentum_volume_shape_score",
        "trend_quality": "trend_quality_score",
        "trend_quality_volume_shape": "trend_quality_volume_shape_score",
        "trend_quality_avoid_failed_gap": "trend_quality_avoid_failed_gap_score",
        "trend_quality_avoid_failed_gap_volume_shape": "trend_quality_avoid_failed_gap_volume_shape_score",
        "trend_quality_gap_resistance": "trend_quality_gap_resistance_score",
        "breakout_quality": "breakout_quality_score",
        "defensive_trend": "defensive_trend_score",
        "gap_resistance": "gap_resistance_score",
        "gap_resistance_50": "gap_resistance_score",
        "hybrid_markov_trend": "hybrid_markov_trend_score",
        "hybrid_markov_trend_spy_gated": "hybrid_markov_trend_score",
    }
    score_col = score_map.get(selector)
    if score_col:
        candidates = candidates[candidates[score_col].replace([np.inf, -np.inf], np.nan).notna()].copy()
        candidates = candidates[candidates[score_col].astype(float) > 0.0].copy()
        if selector in {"gap_resistance", "gap_resistance_50", "trend_quality_gap_resistance"}:
            candidates = candidates[candidates["gap_return"].astype(float).ge(0.01)].copy()
            candidates = candidates[candidates["gap_over_resistance_20"].astype(float).ge(0.0)].copy()
        if selector == "gap_resistance_50":
            candidates = candidates[candidates["gap_over_resistance_50"].astype(float).ge(0.0)].copy()
        if selector == "trend_quality_gap_resistance":
            candidates = candidates[candidates["trend_quality_score"].astype(float).gt(0.0)].copy()
        if selector in {"trend_quality_avoid_failed_gap", "trend_quality_avoid_failed_gap_volume_shape"}:
            candidates = candidates[~candidates["gap_failed_prev"].fillna(False).astype(bool)].copy()
        if "hybrid" in selector:
            candidates = candidates[candidates["markov_signal"].astype(float) >= 0.0].copy()
    else:
        candidates = candidates[candidates["markov_signal"].astype(float) >= config.min_signal].copy()
    if "relative_guard" in selector:
        candidates = candidates[candidates["rel_ret_5_prev"].fillna(-1.0).astype(float) > 0.0].copy()
        candidates = candidates[candidates["rel_ret_20_prev"].fillna(-1.0).astype(float) > -0.01].copy()
        candidates = candidates[candidates["ret_5_prev"].fillna(-1.0).astype(float) > -0.015].copy()
        candidates = candidates[candidates["volume_ratio_5_20_prev"].fillna(0.0).astype(float) >= 0.8].copy()
    if "strong_relative_guard" in selector:
        candidates = candidates[candidates["rel_ret_5_prev"].fillna(-1.0).astype(float) >= 0.005].copy()
        candidates = candidates[candidates["rel_ret_20_prev"].fillna(-1.0).astype(float) >= 0.0].copy()
        candidates = candidates[candidates["ret_5_prev"].fillna(-1.0).astype(float) >= 0.0].copy()
        candidates = candidates[candidates["relative_momentum_score"].fillna(-99.0).astype(float) > 0.0].copy()
    if "spy_gated" in selector:
        candidates = candidates[candidates["spy_markov_signal"].fillna(0.0).astype(float) > 0.0].copy()
    if "confirmed" in selector:
        candidates = candidates[candidates["adaptive_confirms"].fillna(False).astype(bool)].copy()
    if candidates.empty:
        candidates["score"] = []
        return candidates
    if score_col:
        score = candidates[score_col].astype(float).copy()
    else:
        score = candidates["markov_signal"].astype(float).copy()
    if "relative_guard" in selector:
        score = (
            score
            + 0.08 * candidates["relative_momentum_score"].fillna(0.0).astype(float)
            + 0.04 * candidates["volume_shape_score"].fillna(0.0).astype(float)
        )
    if "rel_rank" in selector:
        score = (
            score
            + 0.04 * candidates["relative_momentum_score"].fillna(0.0).astype(float)
            + 0.02 * candidates["trend_quality_score"].fillna(0.0).astype(float)
        )
    if "algo_fused" in selector:
        score = score + 0.15 * candidates["state_persistence"].astype(float)
    candidates["score"] = score
    return candidates.sort_values("score", ascending=False).head(pool_size).copy()


def current_selector_score(row: pd.Series | None, selector: str) -> float:
    if row is None:
        return -1.0
    score_map = {
        "relative_momentum": "relative_momentum_score",
        "relative_momentum_spy_gated": "relative_momentum_score",
        "relative_momentum_volume_shape": "relative_momentum_volume_shape_score",
        "relative_momentum_volume_shape_spy_gated": "relative_momentum_volume_shape_score",
        "trend_quality": "trend_quality_score",
        "trend_quality_volume_shape": "trend_quality_volume_shape_score",
        "trend_quality_avoid_failed_gap": "trend_quality_avoid_failed_gap_score",
        "trend_quality_avoid_failed_gap_volume_shape": "trend_quality_avoid_failed_gap_volume_shape_score",
        "trend_quality_gap_resistance": "trend_quality_gap_resistance_score",
        "breakout_quality": "breakout_quality_score",
        "defensive_trend": "defensive_trend_score",
        "gap_resistance": "gap_resistance_score",
        "gap_resistance_50": "gap_resistance_score",
        "hybrid_markov_trend": "hybrid_markov_trend_score",
        "hybrid_markov_trend_spy_gated": "hybrid_markov_trend_score",
    }
    score_col = score_map.get(selector, "markov_signal")
    if score_col not in row or pd.isna(row[score_col]):
        return -1.0
    return float(row[score_col])


def enters_at_open(selector: str) -> bool:
    return "gap_resistance" in selector


def simulate_holding(
    df: pd.DataFrame,
    config: MarkovConfig,
    strategy: HoldingStrategy,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    equity = START_EQUITY
    spy_equity = START_EQUITY
    positions: dict[str, dict] = {}
    prev_weights: dict[str, float] = {}
    curve_rows: list[dict] = []
    trade_rows: list[dict] = []
    closed_holds: list[int] = []
    cost_per_side = config.roundtrip_cost / 2.0

    for date_value, group in df.groupby("date", sort=True):
        ts = pd.Timestamp(str(date_value), tz="UTC")
        group = group.copy()
        by_symbol = group.set_index("symbol", drop=False)
        spy_values = group["spy_daily_return"].dropna() if "spy_daily_return" in group.columns else pd.Series(dtype=float)
        spy_ret = float(spy_values.iloc[0]) if not spy_values.empty else 0.0
        spy_gate_ok = float(group["spy_markov_signal"].fillna(0.0).iloc[0]) > 0.0 if "spy_markov_signal" in group.columns and not group.empty else True

        pool_n = max(config.max_positions, strategy.rank_exit_top_n, 10)
        candidates = score_candidates(group, config, strategy.selector, pool_n)
        candidate_symbols = candidates["symbol"].astype(str).tolist()
        candidate_set = set(candidate_symbols)

        exits: list[str] = []
        for symbol, pos in list(positions.items()):
            row = by_symbol.loc[symbol] if symbol in by_symbol.index else None
            signal = current_selector_score(row, strategy.selector)
            should_exit = False
            if int(pos["age"]) >= strategy.max_hold_days:
                should_exit = True
            if int(pos["age"]) >= strategy.min_hold_days and strategy.exit_signal is not None and signal < strategy.exit_signal:
                should_exit = True
            if int(pos["age"]) >= strategy.min_hold_days and strategy.rank_exit_top_n > 0 and symbol not in set(candidate_symbols[: strategy.rank_exit_top_n]):
                should_exit = True
            if int(pos["age"]) >= strategy.min_hold_days and strategy.exit_on_spy_gate and not spy_gate_ok:
                should_exit = True
            if symbol not in by_symbol.index:
                should_exit = True
            if should_exit:
                exits.append(symbol)

        for symbol in exits:
            closed_holds.append(int(positions[symbol]["age"]))
            del positions[symbol]

        slots = max(config.max_positions - len(positions), 0)
        entries: list[str] = []
        if slots:
            for symbol in candidate_symbols:
                if symbol not in positions and symbol not in entries:
                    entries.append(symbol)
                    if len(entries) >= slots:
                        break
        for symbol in entries:
            row = by_symbol.loc[symbol]
            positions[symbol] = {
                "entry_date": str(ts.date()),
                "age": 0,
                "entry_signal": float(row["markov_signal"]),
                "entry_score": float(row["score"]) if "score" in row else float(row["markov_signal"]),
            }

        symbols = sorted(positions)
        if symbols:
            exposure_scale = 1.0
            if strategy.exposure_column and strategy.exposure_column in group.columns:
                values = group[strategy.exposure_column].dropna().astype(float)
                if not values.empty:
                    exposure_scale = float(np.clip(values.iloc[0], 0.0, 1.0))
            weight = config.portfolio_exposure * exposure_scale / len(symbols)
            weights = {symbol: weight for symbol in symbols}
        else:
            weights = {}

        names = set(prev_weights) | set(weights)
        turnover = float(sum(abs(weights.get(name, 0.0) - prev_weights.get(name, 0.0)) for name in names))
        cost = turnover * cost_per_side
        portfolio_ret = -cost
        entry_set = set(entries)
        for symbol, weight in weights.items():
            if symbol in by_symbol.index:
                row = by_symbol.loc[symbol]
                if symbol in entry_set and enters_at_open(strategy.selector) and "open_to_close_return" in row and pd.notna(row["open_to_close_return"]):
                    realized = float(row["open_to_close_return"])
                else:
                    realized = float(row["daily_return"])
                portfolio_ret += weight * realized

        equity *= 1.0 + portfolio_ret
        spy_equity *= 1.0 + spy_ret

        for symbol in exits:
            trade_rows.append(
                {
                    "timestamp": str(ts),
                    "strategy": strategy.name,
                    "event": "exit",
                    "symbol": symbol,
                    "weight": float(prev_weights.get(symbol, 0.0)),
                    "hold_days": closed_holds[-1] if closed_holds else None,
                    "turnover": turnover,
                }
            )
        for symbol in entries:
            row = by_symbol.loc[symbol]
            trade_rows.append(
                {
                    "timestamp": str(ts),
                    "strategy": strategy.name,
                    "event": "entry",
                    "symbol": symbol,
                    "weight": float(weights.get(symbol, 0.0)),
                    "hold_days": 0,
                    "score": float(row["score"]) if "score" in row else float(row["markov_signal"]),
                    "markov_signal": float(row["markov_signal"]),
                    "gap_return": float(row["gap_return"]) if "gap_return" in row and pd.notna(row["gap_return"]) else None,
                    "gap_over_resistance_20": float(row["gap_over_resistance_20"]) if "gap_over_resistance_20" in row and pd.notna(row["gap_over_resistance_20"]) else None,
                    "gap_over_resistance_50": float(row["gap_over_resistance_50"]) if "gap_over_resistance_50" in row and pd.notna(row["gap_over_resistance_50"]) else None,
                    "volume_shape_score": float(row["volume_shape_score"]) if "volume_shape_score" in row and pd.notna(row["volume_shape_score"]) else None,
                    "volume_ratio_5_20_prev": float(row["volume_ratio_5_20_prev"]) if "volume_ratio_5_20_prev" in row and pd.notna(row["volume_ratio_5_20_prev"]) else None,
                    "dollar_volume_ratio_5_20_prev": float(row["dollar_volume_ratio_5_20_prev"]) if "dollar_volume_ratio_5_20_prev" in row and pd.notna(row["dollar_volume_ratio_5_20_prev"]) else None,
                    "volume_spike_prev": float(row["volume_spike_prev"]) if "volume_spike_prev" in row and pd.notna(row["volume_spike_prev"]) else None,
                    "open_to_close_return": float(row["open_to_close_return"]) if "open_to_close_return" in row and pd.notna(row["open_to_close_return"]) else None,
                    "turnover": turnover,
                }
            )

        curve_rows.append(
            {
                "timestamp": str(ts),
                "strategy": strategy.name,
                "equity": equity,
                "spy_equity": spy_equity,
                "portfolio_return": portfolio_ret,
                "spy_return": spy_ret,
                "turnover": turnover,
                "cost": cost,
                "positions": ",".join(symbols),
                "position_count": len(symbols),
                "entries": ",".join(entries),
                "exits": ",".join(exits),
            }
        )

        for symbol in list(positions):
            positions[symbol]["age"] = int(positions[symbol]["age"]) + 1
        prev_weights = weights

    curve = pd.DataFrame(curve_rows)
    trades = pd.DataFrame(trade_rows)
    returns = curve["portfolio_return"].astype(float) if not curve.empty else pd.Series(dtype=float)
    spy_returns = curve["spy_return"].astype(float) if not curve.empty else pd.Series(dtype=float)
    entries = trades[trades["event"] == "entry"] if not trades.empty else pd.DataFrame()
    exits = trades[trades["event"] == "exit"] if not trades.empty else pd.DataFrame()
    summary = {
        "strategy": strategy.name,
        "selector": strategy.selector,
        "max_hold_days": strategy.max_hold_days,
        "min_hold_days": strategy.min_hold_days,
        "decision_days": int(len(curve)),
        "active_days": int((curve["position_count"].astype(int) > 0).sum()) if not curve.empty else 0,
        "entry_events": int(len(entries)),
        "exit_events": int(len(exits)),
        "total_turnover": float(curve["turnover"].sum()) if not curve.empty else 0.0,
        "avg_daily_turnover": float(curve["turnover"].mean()) if not curve.empty else 0.0,
        "final_equity": float(equity),
        "spy_final_equity": float(spy_equity),
        "total_return": float(equity / START_EQUITY - 1.0),
        "spy_total_return": float(spy_equity / START_EQUITY - 1.0),
        "alpha_return": float(equity / START_EQUITY - spy_equity / START_EQUITY),
        "max_drawdown": max_drawdown(curve["equity"].astype(float)) if not curve.empty else 0.0,
        "sharpe": sharpe(returns, bars_per_year=252),
        "spy_sharpe": sharpe(spy_returns, bars_per_year=252),
        "avg_closed_hold_days": float(np.mean(closed_holds)) if closed_holds else 0.0,
        "open_positions_end": int(len(positions)),
    }
    return summary, curve, trades, pd.DataFrame(closed_holds, columns=["hold_days"])


def monthly_returns(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy, group in curves.groupby("strategy", sort=False):
        g = group.copy()
        g["timestamp"] = pd.to_datetime(g["timestamp"], utc=True)
        g["month"] = g["timestamp"].dt.tz_convert(None).dt.to_period("M").astype(str)
        prev_equity = START_EQUITY
        prev_spy = START_EQUITY
        for month, month_df in g.groupby("month", sort=True):
            end_equity = float(month_df["equity"].iloc[-1])
            end_spy = float(month_df["spy_equity"].iloc[-1])
            rows.append(
                {
                    "strategy": strategy,
                    "month": month,
                    "return": end_equity / prev_equity - 1.0,
                    "spy_return": end_spy / prev_spy - 1.0,
                    "pnl": end_equity - prev_equity,
                    "turnover": float(month_df["turnover"].sum()),
                    "entries": int(month_df["entries"].astype(str).ne("").sum()),
                }
            )
            prev_equity = end_equity
            prev_spy = end_spy
    return pd.DataFrame(rows)


def plot_comparison(curves: pd.DataFrame, leaderboard: pd.DataFrame, monthly: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    docs = Path("docs")
    docs.mkdir(exist_ok=True)
    top = leaderboard.sort_values("alpha_return", ascending=False).head(8)["strategy"].tolist()
    plot_curves = curves[curves["strategy"].isin(top)].copy()
    plot_curves["timestamp"] = pd.to_datetime(plot_curves["timestamp"], utc=True)

    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[2.0, 1.2, 1.4])
    ax_eq = fig.add_subplot(gs[0, :])
    ax_bar = fig.add_subplot(gs[1, 0])
    ax_dd = fig.add_subplot(gs[1, 1])
    ax_month = fig.add_subplot(gs[2, :])

    for strategy, group in plot_curves.groupby("strategy", sort=False):
        ax_eq.plot(group["timestamp"], group["equity"], label=strategy, linewidth=1.6)
    spy = plot_curves.groupby("timestamp", sort=True)["spy_equity"].first().reset_index()
    ax_eq.plot(spy["timestamp"], spy["spy_equity"], "--", color="black", linewidth=2.0, label="SPY")
    ax_eq.set_title("Holding-Period Portfolio Models: Equity Curves")
    ax_eq.set_ylabel("Equity ($)")
    ax_eq.grid(alpha=0.25)
    ax_eq.legend(ncol=2, frameon=False, fontsize=8)

    board = leaderboard.sort_values("total_return", ascending=True)
    y = np.arange(len(board))
    ax_bar.barh(y, board["total_return"] * 100, color="#43aa8b", label="Strategy")
    spy_ret = float(board["spy_total_return"].iloc[0]) * 100
    ax_bar.axvline(spy_ret, color="black", linestyle="--", label=f"SPY {spy_ret:.1f}%")
    ax_bar.axvline(0.0, color="#777", linewidth=0.8)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(board["strategy"], fontsize=8)
    ax_bar.set_xlabel("Return (%)")
    ax_bar.set_title("Total Return")
    ax_bar.grid(axis="x", alpha=0.25)
    ax_bar.legend(frameon=False, fontsize=8)

    ax_dd.barh(y, board["max_drawdown"] * 100, color="#f94144")
    ax_dd.axvline(0.0, color="#777", linewidth=0.8)
    ax_dd.set_yticks(y)
    ax_dd.set_yticklabels([])
    ax_dd.set_xlabel("Max drawdown (%)")
    ax_dd.set_title("Drawdown")
    ax_dd.grid(axis="x", alpha=0.25)

    month_table = monthly[monthly["strategy"].isin(top)].pivot(index="strategy", columns="month", values="return")
    month_table = month_table.reindex(top)
    im = ax_month.imshow(month_table.fillna(0.0).to_numpy() * 100, aspect="auto", cmap="RdYlGn", vmin=-12, vmax=12)
    ax_month.set_yticks(np.arange(len(month_table.index)))
    ax_month.set_yticklabels(month_table.index, fontsize=8)
    ax_month.set_xticks(np.arange(len(month_table.columns)))
    ax_month.set_xticklabels(month_table.columns, rotation=0)
    ax_month.set_title("Monthly Returns (%)")
    for i in range(len(month_table.index)):
        for j in range(len(month_table.columns)):
            val = month_table.iloc[i, j]
            if pd.notna(val):
                ax_month.text(j, i, f"{val*100:+.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax_month, shrink=0.85)

    fig.savefig(output_dir / "markov_holding_period_comparison.png", dpi=160)
    fig.savefig(docs / "markov_holding_period_comparison.png", dpi=160)
    plt.close(fig)


def strategies() -> list[HoldingStrategy]:
    return [
        HoldingStrategy("confirmed_rebalance_daily", "confirmed", max_hold_days=1),
        HoldingStrategy("confirmed_hold_2d", "confirmed", max_hold_days=2),
        HoldingStrategy("confirmed_hold_3d", "confirmed", max_hold_days=3),
        HoldingStrategy("confirmed_hold_5d", "confirmed", max_hold_days=5),
        HoldingStrategy("confirmed_hold_10d", "confirmed", max_hold_days=10),
        HoldingStrategy("confirmed_signal_exit_max10", "confirmed", max_hold_days=10, min_hold_days=2, exit_signal=0.02),
        HoldingStrategy(
            "confirmed_market_exposure_hold_5d",
            "confirmed",
            max_hold_days=5,
            exposure_column="spy_relative_guard_exposure",
        ),
        HoldingStrategy(
            "confirmed_market_exposure_hold_10d",
            "confirmed",
            max_hold_days=10,
            exposure_column="spy_relative_guard_exposure",
        ),
        HoldingStrategy("confirmed_rel_rank_hold_5d", "confirmed_rel_rank", max_hold_days=5),
        HoldingStrategy(
            "confirmed_rel_rank_market_hold_5d",
            "confirmed_rel_rank",
            max_hold_days=5,
            exposure_column="spy_relative_guard_exposure",
        ),
        HoldingStrategy(
            "confirmed_relative_guard_hold_5d",
            "confirmed_relative_guard",
            max_hold_days=5,
            min_hold_days=1,
            exit_signal=0.02,
            exposure_column="spy_relative_guard_exposure",
        ),
        HoldingStrategy(
            "confirmed_relative_guard_exit_max10",
            "confirmed_relative_guard",
            max_hold_days=10,
            min_hold_days=2,
            exit_signal=0.02,
            rank_exit_top_n=10,
            exposure_column="spy_relative_guard_exposure",
        ),
        HoldingStrategy(
            "confirmed_strong_relative_guard_hold_5d",
            "confirmed_strong_relative_guard",
            max_hold_days=5,
            min_hold_days=1,
            exit_signal=0.02,
            exposure_column="spy_relative_guard_exposure",
        ),
        HoldingStrategy("spy_fused_rebalance_daily", "algo_fused_spy_gated", max_hold_days=1),
        HoldingStrategy("spy_fused_hold_3d", "algo_fused_spy_gated", max_hold_days=3),
        HoldingStrategy("spy_fused_hold_5d", "algo_fused_spy_gated", max_hold_days=5),
        HoldingStrategy(
            "spy_fused_rank_exit_max10",
            "algo_fused_spy_gated",
            max_hold_days=10,
            min_hold_days=2,
            exit_signal=0.02,
            rank_exit_top_n=10,
            exit_on_spy_gate=True,
        ),
        HoldingStrategy("relative_momentum_hold_3d", "relative_momentum", max_hold_days=3),
        HoldingStrategy("relative_momentum_hold_5d", "relative_momentum", max_hold_days=5),
        HoldingStrategy("relative_momentum_exit_max10", "relative_momentum_spy_gated", max_hold_days=10, min_hold_days=2, exit_signal=0.0, exit_on_spy_gate=True),
        HoldingStrategy("relative_momentum_volume_shape_exit_max10", "relative_momentum_volume_shape_spy_gated", max_hold_days=10, min_hold_days=2, exit_signal=0.0, exit_on_spy_gate=True),
        HoldingStrategy("trend_quality_hold_3d", "trend_quality", max_hold_days=3),
        HoldingStrategy("trend_quality_hold_5d", "trend_quality", max_hold_days=5),
        HoldingStrategy("trend_quality_volume_shape_hold_3d", "trend_quality_volume_shape", max_hold_days=3),
        HoldingStrategy("trend_quality_avoid_failed_gap_hold_3d", "trend_quality_avoid_failed_gap", max_hold_days=3),
        HoldingStrategy("trend_quality_avoid_failed_gap_volume_shape_hold_3d", "trend_quality_avoid_failed_gap_volume_shape", max_hold_days=3),
        HoldingStrategy("gap_resistance_hold_1d", "gap_resistance", max_hold_days=1),
        HoldingStrategy("gap_resistance_hold_3d", "gap_resistance", max_hold_days=3),
        HoldingStrategy("gap_resistance_50_hold_3d", "gap_resistance_50", max_hold_days=3),
        HoldingStrategy("trend_quality_gap_resistance_hold_3d", "trend_quality_gap_resistance", max_hold_days=3),
        HoldingStrategy("breakout_quality_hold_3d", "breakout_quality", max_hold_days=3),
        HoldingStrategy("defensive_trend_hold_5d", "defensive_trend", max_hold_days=5),
        HoldingStrategy("hybrid_markov_trend_hold_3d", "hybrid_markov_trend", max_hold_days=3),
        HoldingStrategy("hybrid_markov_trend_hold_5d", "hybrid_markov_trend_spy_gated", max_hold_days=5),
        HoldingStrategy("hybrid_markov_trend_exit_max10", "hybrid_markov_trend_spy_gated", max_hold_days=10, min_hold_days=2, exit_signal=0.0, rank_exit_top_n=10, exit_on_spy_gate=True),
    ]


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = default_markov_config(args)
    frame, daily, signals = load_daily_eval_frame(config)
    frame = add_cross_sectional_scores(frame, daily)
    if args.min_median_dollar_volume > 0:
        before = len(frame)
        frame = frame[
            (frame["symbol"].astype(str).str.upper() == "SPY")
            | frame["median_dollar_volume_20_prev"].fillna(0.0).astype(float).ge(args.min_median_dollar_volume)
        ].copy()
        print(
            f"[liquidity] min_median_dollar_volume={args.min_median_dollar_volume:,.0f} "
            f"rows {before}->{len(frame)} symbols={frame['symbol'].nunique()}",
            flush=True,
        )
    summaries = []
    curves = []
    trades = []
    for strategy in strategies():
        summary, curve, trade_df, _ = simulate_holding(frame, config, strategy)
        summaries.append(summary)
        curves.append(curve)
        trades.append(trade_df)
        print(
            f"[holding] {strategy.name} return={summary['total_return']:.2%} "
            f"spy={summary['spy_total_return']:.2%} dd={summary['max_drawdown']:.2%}",
            flush=True,
        )
    all_curves = pd.concat(curves, ignore_index=True)
    all_trades = pd.concat([t for t in trades if not t.empty], ignore_index=True) if any(not t.empty for t in trades) else pd.DataFrame()
    leaderboard = pd.DataFrame(summaries).sort_values("alpha_return", ascending=False)
    monthly = monthly_returns(all_curves)
    result = {
        "config": asdict(config),
        "min_median_dollar_volume": float(args.min_median_dollar_volume),
        "rows": int(len(frame)),
        "symbols": int(frame["symbol"].nunique()),
        "daily_rows": int(len(daily)),
        "signal_rows": int(len(signals)),
        "strategies": summaries,
        "warning": "research_only_holding_period_comparison_not_financial_advice",
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, default=str))
    leaderboard.to_csv(output_dir / "leaderboard.csv", index=False)
    all_curves.to_csv(output_dir / "equity_curves.csv", index=False)
    all_trades.to_csv(output_dir / "trades.csv", index=False)
    monthly.to_csv(output_dir / "monthly_returns.csv", index=False)
    plot_comparison(all_curves, leaderboard, monthly, output_dir)
    print(json.dumps(result, indent=2, default=str), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(Path.home() / ".cache/trading-autoresearch"))
    parser.add_argument("--output-dir", default="checkpoints/transformer_15m/markov_holding_periods_latest")
    parser.add_argument("--eval-start", default="2026-01-01")
    parser.add_argument("--eval-end", default="2026-05-22")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--top-symbols-limit", type=int, default=0)
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
    parser.add_argument("--adaptive-min-history-days", type=int, default=80)
    parser.add_argument("--regime-source", choices=["fixed", "adaptive"], default="adaptive")
    parser.add_argument("--transition-lookback-days", type=int, default=126)
    parser.add_argument("--transition-halflife-days", type=float, default=42.0)
    parser.add_argument("--min-median-dollar-volume", type=float, default=0.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
