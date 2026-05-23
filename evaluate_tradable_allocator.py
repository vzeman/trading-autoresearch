"""Evaluate allocator-scored world-model actions as a sequential tradable policy.

The older planner metrics average independent counterfactual groups. This
script adds a stricter deployment-shaped check:

1. train/use a frozen allocator and world model,
2. choose the trade threshold from calibration data only,
3. on the locked test period, pick at most one action per decision timestamp,
4. hold that position until its realized exit timestamp, otherwise stay cash.

It is still a research simulator, but it catches overlapping-trade optimism and
reports a simple cash-aware equity curve.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from prepare import CACHE_DIR
from train_allocator import apply_allocator
from walk_forward_allocator import choose_threshold, planner_rows, score_dataset, summarize_with_cash
from train_world_model import pick_device


def _time_col(df: pd.DataFrame) -> str:
    return "decision_timestamp" if "decision_timestamp" in df.columns else "timestamp"


def one_candidate_per_timestamp(active: pd.DataFrame) -> pd.DataFrame:
    if active.empty:
        return active.copy()
    time_col = _time_col(active)
    idx = active.reset_index(drop=True).groupby(time_col, sort=False)["pred_score"].idxmax()
    return active.reset_index(drop=True).loc[idx.to_numpy()].sort_values(time_col).reset_index(drop=True)


def entry_candidates(scored: pd.DataFrame) -> pd.DataFrame:
    """Keep actions a cash portfolio can actually initiate."""
    return scored[
        (scored["current_position_frac"].astype(float) == 0.0)
        & (scored["action"].astype(str) == "buy")
        & (scored["target_position_frac"].astype(float) > 0.0)
        & (~scored["symbol"].astype(str).str.startswith("^"))
    ].copy()


def tradability_filters(scored: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    """Apply filters known at decision time before any threshold search."""
    out = scored.copy()
    before = len(out)
    counts: dict[str, int | float] = {"input_rows": int(before)}
    if args.min_price > 0 and "price" in out.columns:
        out = out[out["price"].astype(float) >= float(args.min_price)].copy()
    if args.min_trade_notional > 0 and "trade_notional" in out.columns:
        out = out[out["trade_notional"].astype(float) >= float(args.min_trade_notional)].copy()
    if args.min_state_volume_z_1d > -99 and "state_volume_z_1d" in out.columns:
        out = out[out["state_volume_z_1d"].astype(float) >= float(args.min_state_volume_z_1d)].copy()
    if args.max_state_vol_1d > 0 and "state_vol_1d" in out.columns:
        out = out[out["state_vol_1d"].astype(float) <= float(args.max_state_vol_1d)].copy()
    if args.max_abs_state_ret_1d > 0 and "state_ret_1d" in out.columns:
        out = out[out["state_ret_1d"].astype(float).abs() <= float(args.max_abs_state_ret_1d)].copy()
    counts["output_rows"] = int(len(out))
    counts["dropped_rows"] = int(before - len(out))
    counts["drop_rate"] = float((before - len(out)) / max(before, 1))
    return out, counts


def apply_trade_rule(planner: pd.DataFrame, rule: dict) -> pd.DataFrame:
    out = planner[planner["pred_score"] >= float(rule["score_threshold"])].copy()
    if "max_target_position_frac" in rule:
        out = out[out["target_position_frac"].astype(float) <= float(rule["max_target_position_frac"])].copy()
    if "max_horizon_bars" in rule:
        out = out[out["horizon_bars"].astype(float) <= float(rule["max_horizon_bars"])].copy()
    if "min_pred_profit_label" in rule and "pred_profit_label" in out.columns:
        out = out[out["pred_profit_label"].astype(float) >= float(rule["min_pred_profit_label"])].copy()
    if "min_pred_beat_spy_label" in rule and "pred_beat_spy_label" in out.columns:
        out = out[out["pred_beat_spy_label"].astype(float) >= float(rule["min_pred_beat_spy_label"])].copy()
    if "min_pred_future_alpha_vs_spy" in rule and "pred_future_alpha_vs_spy" in out.columns:
        out = out[out["pred_future_alpha_vs_spy"].astype(float) >= float(rule["min_pred_future_alpha_vs_spy"])].copy()
    if "min_pred_future_min_asset_return" in rule and "pred_future_min_asset_return" in out.columns:
        out = out[out["pred_future_min_asset_return"].astype(float) >= float(rule["min_pred_future_min_asset_return"])].copy()
    if "min_pred_future_asset_max_drawdown" in rule and "pred_future_asset_max_drawdown" in out.columns:
        out = out[out["pred_future_asset_max_drawdown"].astype(float) >= float(rule["min_pred_future_asset_max_drawdown"])].copy()
    if "max_pred_asset_crash_label" in rule and "pred_asset_crash_label" in out.columns:
        out = out[out["pred_asset_crash_label"].astype(float) <= float(rule["max_pred_asset_crash_label"])].copy()
    if "max_pred_severe_adverse_label" in rule and "pred_severe_adverse_label" in out.columns:
        out = out[out["pred_severe_adverse_label"].astype(float) <= float(rule["max_pred_severe_adverse_label"])].copy()
    if "max_pred_score_std" in rule and "pred_score_std" in out.columns:
        out = out[out["pred_score_std"].astype(float) <= float(rule["max_pred_score_std"])].copy()
    return out


def rule_name(rule: dict) -> str:
    parts = [f"q{rule['score_quantile']:.2f}"]
    if "max_target_position_frac" in rule:
        parts.append(f"target<={rule['max_target_position_frac']:.2f}")
    if "max_horizon_bars" in rule:
        parts.append(f"h<={int(rule['max_horizon_bars'])}")
    if "min_pred_profit_label" in rule:
        parts.append(f"p_profit>={rule['min_pred_profit_label']:.2f}")
    if "min_pred_beat_spy_label" in rule:
        parts.append(f"p_beat>={rule['min_pred_beat_spy_label']:.2f}")
    if "min_pred_future_alpha_vs_spy" in rule:
        parts.append(f"pred_alpha>={rule['min_pred_future_alpha_vs_spy']:.4f}")
    if "min_pred_future_min_asset_return" in rule:
        parts.append(f"pred_min_ret>={rule['min_pred_future_min_asset_return']:.4f}")
    if "min_pred_future_asset_max_drawdown" in rule:
        parts.append(f"pred_asset_dd>={rule['min_pred_future_asset_max_drawdown']:.4f}")
    if "max_pred_asset_crash_label" in rule:
        parts.append(f"p_crash<={rule['max_pred_asset_crash_label']:.2f}")
    if "max_pred_severe_adverse_label" in rule:
        parts.append(f"p_severe<={rule['max_pred_severe_adverse_label']:.2f}")
    if "max_pred_score_std" in rule:
        parts.append(f"score_std<={rule['max_pred_score_std']:.4f}")
    return " | ".join(parts)


def candidate_rules(calibration_planner: pd.DataFrame) -> list[dict]:
    score_quantiles = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95)
    max_targets = (0.50, 0.75, 1.00)
    max_horizons = (30, 60, 120)
    min_profit = (None, 0.50, 0.55)
    min_beat = (None, 0.50, 0.55)
    alpha_thresholds: list[float | None] = [None]
    if "pred_future_alpha_vs_spy" in calibration_planner.columns:
        alpha_thresholds += [
            float(calibration_planner["pred_future_alpha_vs_spy"].quantile(q))
            for q in (0.50, 0.60)
        ]
    score_std_thresholds: list[float | None] = [None]
    if "pred_score_std" in calibration_planner.columns:
        score_std_thresholds += [
            float(calibration_planner["pred_score_std"].quantile(q))
            for q in (0.50, 0.75)
        ]
    min_asset_thresholds: list[float | None] = [None]
    if "pred_future_min_asset_return" in calibration_planner.columns:
        min_asset_thresholds += [
            float(calibration_planner["pred_future_min_asset_return"].quantile(q))
            for q in (0.50, 0.60)
        ]
    asset_dd_thresholds: list[float | None] = [None]
    if "pred_future_asset_max_drawdown" in calibration_planner.columns:
        asset_dd_thresholds += [
            float(calibration_planner["pred_future_asset_max_drawdown"].quantile(q))
            for q in (0.50, 0.60)
        ]
    crash_thresholds: list[float | None] = [None]
    if "pred_asset_crash_label" in calibration_planner.columns:
        crash_thresholds += [
            float(calibration_planner["pred_asset_crash_label"].quantile(q))
            for q in (0.40, 0.60)
        ]
    severe_thresholds: list[float | None] = [None]
    if "pred_severe_adverse_label" in calibration_planner.columns:
        severe_thresholds += [
            float(calibration_planner["pred_severe_adverse_label"].quantile(q))
            for q in (0.40, 0.60)
        ]

    base_rules = []
    for q in score_quantiles:
        score_threshold = float(calibration_planner["pred_score"].quantile(q))
        for max_target in max_targets:
            for max_horizon in max_horizons:
                for profit_threshold in min_profit:
                    for beat_threshold in min_beat:
                        for alpha_threshold in alpha_thresholds:
                            for score_std_threshold in score_std_thresholds:
                                rule = {
                                    "name": "",
                                    "score_quantile": float(q),
                                    "score_threshold": score_threshold,
                                    "max_target_position_frac": float(max_target),
                                    "max_horizon_bars": int(max_horizon),
                                }
                                if profit_threshold is not None:
                                    rule["min_pred_profit_label"] = float(profit_threshold)
                                if beat_threshold is not None:
                                    rule["min_pred_beat_spy_label"] = float(beat_threshold)
                                if alpha_threshold is not None:
                                    rule["min_pred_future_alpha_vs_spy"] = float(alpha_threshold)
                                if score_std_threshold is not None:
                                    rule["max_pred_score_std"] = float(score_std_threshold)
                                rule["name"] = rule_name(rule)
                                base_rules.append(rule)

    rules = list(base_rules)
    crash_variants = []
    min_asset_values = [v for v in min_asset_thresholds if v is not None]
    asset_dd_values = [v for v in asset_dd_thresholds if v is not None]
    crash_values = [v for v in crash_thresholds if v is not None]
    severe_values = [v for v in severe_thresholds if v is not None]
    for base in base_rules:
        if min_asset_values or asset_dd_values:
            rule = dict(base)
            if min_asset_values:
                rule["min_pred_future_min_asset_return"] = float(max(min_asset_values))
            if asset_dd_values:
                rule["min_pred_future_asset_max_drawdown"] = float(max(asset_dd_values))
            rule["name"] = rule_name(rule)
            crash_variants.append(rule)
        if crash_values or severe_values:
            rule = dict(base)
            if crash_values:
                rule["max_pred_asset_crash_label"] = float(min(crash_values))
            if severe_values:
                rule["max_pred_severe_adverse_label"] = float(min(severe_values))
            rule["name"] = rule_name(rule)
            crash_variants.append(rule)
        if (min_asset_values or asset_dd_values) and (crash_values or severe_values):
            rule = dict(base)
            if min_asset_values:
                rule["min_pred_future_min_asset_return"] = float(max(min_asset_values))
            if asset_dd_values:
                rule["min_pred_future_asset_max_drawdown"] = float(max(asset_dd_values))
            if crash_values:
                rule["max_pred_asset_crash_label"] = float(min(crash_values))
            if severe_values:
                rule["max_pred_severe_adverse_label"] = float(min(severe_values))
            rule["name"] = rule_name(rule)
            crash_variants.append(rule)
    rules.extend(crash_variants)
    return rules


def score_dataset_ensemble(
    data: Path,
    world_ckpt: dict,
    allocator_ckpts: list[dict],
    device: str,
    batch_size: int,
    limit_rows: int,
    min_horizon_bars: int,
    max_horizon_bars: int,
    seed: int,
) -> pd.DataFrame:
    base = score_dataset(
        data,
        world_ckpt,
        allocator_ckpts[0],
        device,
        batch_size,
        limit_rows,
        min_horizon_bars,
        max_horizon_bars,
        seed,
    )
    score_cols = [base["pred_score"].to_numpy(np.float32)]
    for ckpt in allocator_ckpts[1:]:
        scored = apply_allocator(base.drop(columns=["pred_score"], errors="ignore"), ckpt, device=device, batch_size=batch_size)
        score_cols.append(scored["pred_score"].to_numpy(np.float32))
    if len(score_cols) == 1:
        return base
    scores = np.vstack(score_cols)
    out = base.copy()
    out["pred_score"] = scores.mean(axis=0)
    out["pred_score_std"] = scores.std(axis=0)
    out["pred_score_min"] = scores.min(axis=0)
    out["ensemble_size"] = int(scores.shape[0])
    return out


def _date_range(planner: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    time_col = _time_col(planner)
    start = pd.to_datetime(planner[time_col], utc=True).min()
    exit_col = "exit_timestamp" if "exit_timestamp" in planner.columns else time_col
    end = pd.to_datetime(planner[exit_col], utc=True).max()
    return start, end


def choose_trade_rule(
    calibration_planner: pd.DataFrame,
    min_coverage: float,
    idle_asset: str,
    starting_equity: float,
    max_calibration_drawdown: float,
    min_calibration_trades: int,
    min_calibration_return: float,
    min_calibration_profit_rate: float,
    min_calibration_beat_spy_rate: float,
    validation_planner: pd.DataFrame | None,
    min_validation_trades: int,
    min_validation_return: float,
    min_validation_profit_rate: float,
    min_validation_beat_spy_rate: float,
    extra_roundtrip_bps: float,
    extra_fee_usd: float,
    max_trades_per_symbol: int,
    symbol_cooldown_days: float,
) -> dict:
    total_groups = len(calibration_planner)
    test_start, test_end = _date_range(calibration_planner)
    best: dict | None = None
    evaluated = []
    for rule in candidate_rules(calibration_planner):
        active = apply_trade_rule(calibration_planner, rule)
        coverage = len(active) / max(total_groups, 1)
        if coverage < min_coverage:
            continue
        seq = constrained_sequential_portfolio(
            active,
            starting_equity=starting_equity,
            idle_asset=idle_asset,
            test_start=test_start,
            test_end=test_end,
            include_details=False,
            extra_roundtrip_bps=extra_roundtrip_bps,
            extra_fee_usd=extra_fee_usd,
            max_trades_per_symbol=max_trades_per_symbol,
            symbol_cooldown_days=symbol_cooldown_days,
        )
        if seq["trades"] < min_calibration_trades:
            continue
        if abs(float(seq["max_drawdown"])) > max_calibration_drawdown:
            continue
        if float(seq["total_return"]) < min_calibration_return:
            continue
        if float(seq["profit_rate"]) < min_calibration_profit_rate:
            continue
        if float(seq["beat_spy_rate"]) < min_calibration_beat_spy_rate:
            continue
        validation_seq = None
        validation_coverage = 0.0
        if validation_planner is not None and not validation_planner.empty:
            validation_active = apply_trade_rule(validation_planner, rule)
            validation_coverage = len(validation_active) / max(len(validation_planner), 1)
            validation_start, validation_end = _date_range(validation_planner)
            validation_seq = constrained_sequential_portfolio(
                validation_active,
                starting_equity=starting_equity,
                idle_asset=idle_asset,
                test_start=validation_start,
                test_end=validation_end,
                include_details=False,
                extra_roundtrip_bps=extra_roundtrip_bps,
                extra_fee_usd=extra_fee_usd,
                max_trades_per_symbol=max_trades_per_symbol,
                symbol_cooldown_days=symbol_cooldown_days,
            )
            if validation_seq["trades"] < min_validation_trades:
                continue
            if float(validation_seq["total_return"]) < min_validation_return:
                continue
            if float(validation_seq["profit_rate"]) < min_validation_profit_rate:
                continue
            if float(validation_seq["beat_spy_rate"]) < min_validation_beat_spy_rate:
                continue
            if abs(float(validation_seq["max_drawdown"])) > max_calibration_drawdown:
                continue
        score = (
            4.00 * seq["total_return"]
            - 3.00 * abs(seq["max_drawdown"])
            + 0.08 * seq["profit_rate"]
            + 0.04 * seq["beat_spy_rate"]
            + 0.02 * min(seq["trades"] / 25.0, 1.0)
        )
        if validation_seq is not None:
            score = (
                0.35 * score
                + 0.65 * (
                    4.00 * validation_seq["total_return"]
                    - 3.00 * abs(validation_seq["max_drawdown"])
                    + 0.08 * validation_seq["profit_rate"]
                    + 0.04 * validation_seq["beat_spy_rate"]
                    + 0.02 * min(validation_seq["trades"] / 25.0, 1.0)
                )
            )
        row = {
            "rule": rule,
            "coverage": float(coverage),
            "timestamp_candidates": int(seq["trades"]),
            "objective": float(score),
            "sequential": {
                k: v for k, v in seq.items()
                if k not in ("equity_curve", "trades_detail")
            },
        }
        if validation_seq is not None:
            row["validation_coverage"] = float(validation_coverage)
            row["validation_sequential"] = {
                k: v for k, v in validation_seq.items()
                if k not in ("equity_curve", "trades_detail")
            }
        evaluated.append(row)
        if best is None or row["objective"] > best["objective"]:
            best = row
    if best is None:
        raise RuntimeError("could not select calibrated trade rule")
    evaluated = sorted(evaluated, key=lambda x: x["objective"], reverse=True)
    return {"best": best, "top_candidates": evaluated[:20], "evaluated_rules": int(len(evaluated))}


def _spy_lookup() -> tuple[np.ndarray, np.ndarray]:
    spy = pd.read_parquet(CACHE_DIR / "SPY_1m.parquet", columns=["timestamp", "close"]).sort_values("timestamp")
    ts = pd.to_datetime(spy["timestamp"], utc=True).astype("int64").to_numpy()
    close = spy["close"].astype(float).to_numpy()
    return ts, close


def _price_at(ts_ns: np.ndarray, close: np.ndarray, timestamp: pd.Timestamp) -> float:
    idx = int(np.searchsorted(ts_ns, int(timestamp.value), side="right") - 1)
    idx = max(0, min(idx, len(close) - 1))
    return float(close[idx])


def _asset_return(ts_ns: np.ndarray, close: np.ndarray, start: pd.Timestamp, end: pd.Timestamp) -> float:
    if end <= start:
        return 0.0
    start_px = _price_at(ts_ns, close, start)
    end_px = _price_at(ts_ns, close, end)
    return float(end_px / max(start_px, 1e-12) - 1.0)


def sequential_portfolio(
    candidates: pd.DataFrame,
    starting_equity: float = 50_000.0,
    idle_asset: str = "cash",
    test_start: pd.Timestamp | None = None,
    test_end: pd.Timestamp | None = None,
    include_details: bool = True,
    extra_roundtrip_bps: float = 0.0,
    extra_fee_usd: float = 0.0,
) -> dict:
    time_col = _time_col(candidates)
    if candidates.empty:
        return {
            "starting_equity": starting_equity,
            "final_equity": starting_equity,
            "total_return": 0.0,
            "spy_active_return": 0.0,
            "trades": 0,
            "skipped_overlap": 0,
            "profit_rate": 0.0,
            "beat_spy_rate": 0.0,
            "max_drawdown": 0.0,
            "equity_curve": [],
            "trades_detail": [],
        }

    rows = candidates.copy()
    rows[time_col] = pd.to_datetime(rows[time_col], utc=True)
    if "exit_timestamp" in rows.columns:
        rows["exit_timestamp"] = pd.to_datetime(rows["exit_timestamp"], utc=True)
    else:
        rows["exit_timestamp"] = rows[time_col] + pd.to_timedelta(rows["horizon_bars"].astype(float), unit="m")
    rows = rows.sort_values(time_col).reset_index(drop=True)

    equity = float(starting_equity)
    spy_equity = float(starting_equity)
    ts_ns: np.ndarray | None = None
    spy_close: np.ndarray | None = None
    if idle_asset == "spy":
        ts_ns, spy_close = _spy_lookup()
    portfolio_clock = test_start if test_start is not None else pd.Timestamp(rows[time_col].min())
    next_available = pd.Timestamp.min.tz_localize("UTC")
    curve = []
    details = []
    skipped = 0

    for _, row in rows.iterrows():
        entry_ts = pd.Timestamp(row[time_col])
        if entry_ts < next_available:
            skipped += 1
            continue
        idle_return = 0.0
        if idle_asset == "spy" and ts_ns is not None and spy_close is not None and entry_ts > portfolio_clock:
            idle_return = _asset_return(ts_ns, spy_close, portfolio_clock, entry_ts)
            equity *= 1.0 + idle_return
        exit_ts = pd.Timestamp(row["exit_timestamp"])
        ret = float(row["portfolio_return"])
        target_frac = float(row["target_position_frac"])
        extra_cost_return = target_frac * max(0.0, extra_roundtrip_bps) * 1e-4 + max(0.0, extra_fee_usd) / max(equity, 1e-12)
        ret -= extra_cost_return
        spy_ret = float(row.get("future_spy_return", 0.0))
        before = equity
        equity *= 1.0 + ret
        spy_equity *= 1.0 + spy_ret
        next_available = max(exit_ts, entry_ts)
        portfolio_clock = next_available
        curve.append({"timestamp": str(exit_ts), "equity": equity, "spy_active_equity": spy_equity})
        if include_details:
            details.append({
                "entry_timestamp": str(entry_ts),
                "exit_timestamp": str(exit_ts),
                "symbol": str(row["symbol"]),
                "action": str(row["action"]),
                "horizon_bars": int(row["horizon_bars"]),
                "target_position_frac": float(row["target_position_frac"]),
                "pred_score": float(row["pred_score"]),
                "portfolio_return": ret,
                "extra_cost_return": extra_cost_return,
                "future_spy_return": spy_ret,
                "future_alpha_vs_spy": float(row["future_alpha_vs_spy"]),
                "idle_asset": idle_asset,
                "idle_return_before_entry": idle_return,
                "equity_before": before,
                "equity_after": equity,
            })
        else:
            details.append({
                "portfolio_return": ret,
                "extra_cost_return": extra_cost_return,
                "future_alpha_vs_spy": float(row["future_alpha_vs_spy"]),
            })

    if idle_asset == "spy" and ts_ns is not None and spy_close is not None and test_end is not None and test_end > portfolio_clock:
        tail_return = _asset_return(ts_ns, spy_close, portfolio_clock, test_end)
        equity *= 1.0 + tail_return
        curve.append({"timestamp": str(test_end), "equity": equity, "spy_active_equity": spy_equity})

    eq = np.array([starting_equity] + [float(x["equity"]) for x in curve], dtype=np.float64)
    if len(eq) > 1:
        peaks = np.maximum.accumulate(eq)
        max_dd = float(((eq - peaks) / np.maximum(peaks, 1e-12)).min())
    else:
        max_dd = 0.0
    if details:
        returns = np.array([d["portfolio_return"] for d in details], dtype=np.float64)
        alphas = np.array([d["future_alpha_vs_spy"] for d in details], dtype=np.float64)
        profit_rate = float((returns > 0.0).mean())
        beat_spy_rate = float((alphas > 0.0).mean())
    else:
        profit_rate = 0.0
        beat_spy_rate = 0.0

    return {
        "starting_equity": starting_equity,
        "final_equity": float(equity),
        "total_return": float(equity / starting_equity - 1.0),
        "spy_active_return": float(spy_equity / starting_equity - 1.0),
        "idle_asset": idle_asset,
        "extra_roundtrip_bps": float(extra_roundtrip_bps),
        "extra_fee_usd": float(extra_fee_usd),
        "trades": int(len(details)),
        "skipped_overlap": int(skipped),
        "profit_rate": profit_rate,
        "beat_spy_rate": beat_spy_rate,
        "max_drawdown": max_dd,
        "equity_curve": curve,
        "trades_detail": details if include_details else [],
    }


def constrained_sequential_portfolio(
    active: pd.DataFrame,
    starting_equity: float = 50_000.0,
    idle_asset: str = "cash",
    test_start: pd.Timestamp | None = None,
    test_end: pd.Timestamp | None = None,
    include_details: bool = True,
    extra_roundtrip_bps: float = 0.0,
    extra_fee_usd: float = 0.0,
    max_trades_per_symbol: int = 0,
    symbol_cooldown_days: float = 0.0,
) -> dict:
    """Run the sequential simulator while allowing lower-ranked eligible fallbacks.

    The earlier path picked exactly one highest-scored row per timestamp before
    checking overlap. For concentration controls, that can discard useful
    second-best candidates. This function walks timestamps chronologically and
    picks the highest-scored row that is currently eligible.
    """
    if max_trades_per_symbol <= 0 and symbol_cooldown_days <= 0:
        return sequential_portfolio(
            one_candidate_per_timestamp(active),
            starting_equity=starting_equity,
            idle_asset=idle_asset,
            test_start=test_start,
            test_end=test_end,
            include_details=include_details,
            extra_roundtrip_bps=extra_roundtrip_bps,
            extra_fee_usd=extra_fee_usd,
        )

    time_col = _time_col(active)
    if active.empty:
        return sequential_portfolio(
            active,
            starting_equity=starting_equity,
            idle_asset=idle_asset,
            test_start=test_start,
            test_end=test_end,
            include_details=include_details,
            extra_roundtrip_bps=extra_roundtrip_bps,
            extra_fee_usd=extra_fee_usd,
        )

    rows = active.copy()
    rows[time_col] = pd.to_datetime(rows[time_col], utc=True)
    if "exit_timestamp" in rows.columns:
        rows["exit_timestamp"] = pd.to_datetime(rows["exit_timestamp"], utc=True)
    else:
        rows["exit_timestamp"] = rows[time_col] + pd.to_timedelta(rows["horizon_bars"].astype(float), unit="m")
    rows = rows.sort_values([time_col, "pred_score"], ascending=[True, False]).reset_index(drop=True)

    selected = []
    symbol_counts: dict[str, int] = {}
    symbol_available: dict[str, pd.Timestamp] = {}
    next_available = pd.Timestamp.min.tz_localize("UTC")
    cooldown = pd.Timedelta(days=max(0.0, symbol_cooldown_days))
    skipped_overlap = skipped_concentration = skipped_cooldown = 0

    for entry_ts, group in rows.groupby(time_col, sort=False):
        entry_ts = pd.Timestamp(entry_ts)
        if entry_ts < next_available:
            skipped_overlap += len(group)
            continue
        chosen = None
        for _, row in group.iterrows():
            symbol = str(row["symbol"])
            if max_trades_per_symbol > 0 and symbol_counts.get(symbol, 0) >= max_trades_per_symbol:
                skipped_concentration += 1
                continue
            if symbol_cooldown_days > 0 and entry_ts < symbol_available.get(symbol, pd.Timestamp.min.tz_localize("UTC")):
                skipped_cooldown += 1
                continue
            chosen = row
            break
        if chosen is None:
            continue
        selected.append(chosen)
        symbol = str(chosen["symbol"])
        symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
        exit_ts = pd.Timestamp(chosen["exit_timestamp"])
        next_available = max(exit_ts, entry_ts)
        symbol_available[symbol] = next_available + cooldown

    selected_df = pd.DataFrame(selected)
    seq = sequential_portfolio(
        selected_df,
        starting_equity=starting_equity,
        idle_asset=idle_asset,
        test_start=test_start,
        test_end=test_end,
        include_details=include_details,
        extra_roundtrip_bps=extra_roundtrip_bps,
        extra_fee_usd=extra_fee_usd,
    )
    seq["skipped_overlap_candidates"] = int(skipped_overlap)
    seq["skipped_concentration_candidates"] = int(skipped_concentration)
    seq["skipped_cooldown_candidates"] = int(skipped_cooldown)
    seq["max_trades_per_symbol"] = int(max_trades_per_symbol)
    seq["symbol_cooldown_days"] = float(symbol_cooldown_days)
    seq["symbol_trade_counts"] = {k: int(v) for k, v in sorted(symbol_counts.items(), key=lambda kv: (-kv[1], kv[0]))}
    return seq


def split_planner_by_time(planner: pd.DataFrame, validation_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    frac = max(0.0, min(float(validation_fraction), 0.8))
    if frac <= 0.0 or planner.empty:
        return planner, None
    time_col = _time_col(planner)
    times = pd.to_datetime(planner[time_col], utc=True)
    cutoff = times.quantile(1.0 - frac)
    search = planner.loc[times <= cutoff].reset_index(drop=True)
    validation = planner.loc[times > cutoff].reset_index(drop=True)
    if search.empty or validation.empty:
        return planner, None
    return search, validation


def run(args: argparse.Namespace) -> dict:
    device = pick_device(args.device)
    world_ckpt = torch.load(args.world_checkpoint, map_location="cpu", weights_only=False)
    allocator_paths = [args.allocator_checkpoint] + list(args.ensemble_allocator_checkpoints)
    allocator_ckpts = [torch.load(path, map_location="cpu", weights_only=False) for path in allocator_paths]

    calibration_scored = score_dataset_ensemble(
        Path(args.calibration_data),
        world_ckpt,
        allocator_ckpts,
        device,
        args.batch_size,
        args.limit_rows,
        args.min_horizon_bars,
        args.max_horizon_bars,
        args.seed,
    )
    if args.entry_only:
        calibration_scored = entry_candidates(calibration_scored)
    calibration_scored, calibration_filter_summary = tradability_filters(calibration_scored, args)
    calibration_planner = planner_rows(calibration_scored)
    rule_search_planner, rule_validation_planner = split_planner_by_time(
        calibration_planner,
        args.rule_validation_fraction,
    )
    threshold_choice = choose_threshold(calibration_planner, args.min_coverage, args.objective_mode)
    selected = threshold_choice["best"]
    if args.fixed_score_quantile >= 0:
        selected = min(
            threshold_choice["candidates"],
            key=lambda row: abs(float(row["quantile"]) - float(args.fixed_score_quantile)),
        )
    fixed_rule = {
        "name": f"fixed_threshold_q{selected['quantile']:.2f}",
        "score_quantile": float(selected["quantile"]),
        "score_threshold": float(selected["threshold"]),
    }
    if args.fixed_max_target_position_frac > 0:
        fixed_rule["max_target_position_frac"] = float(args.fixed_max_target_position_frac)
    if args.fixed_max_horizon_bars > 0:
        fixed_rule["max_horizon_bars"] = int(args.fixed_max_horizon_bars)
    fixed_rule["name"] = rule_name(fixed_rule)
    calibrated_rule = None
    selected_rule = fixed_rule
    if args.rule_mode == "calibrated":
        try:
            calibrated_rule = choose_trade_rule(
                rule_search_planner,
                min_coverage=args.min_coverage,
                idle_asset=args.idle_asset,
                starting_equity=args.starting_equity,
                max_calibration_drawdown=args.max_calibration_drawdown,
                min_calibration_trades=args.min_calibration_trades,
                min_calibration_return=args.min_calibration_return,
                min_calibration_profit_rate=args.min_calibration_profit_rate,
                min_calibration_beat_spy_rate=args.min_calibration_beat_spy_rate,
                validation_planner=rule_validation_planner,
                min_validation_trades=args.min_validation_trades,
                min_validation_return=args.min_validation_return,
                min_validation_profit_rate=args.min_validation_profit_rate,
                min_validation_beat_spy_rate=args.min_validation_beat_spy_rate,
                extra_roundtrip_bps=args.extra_roundtrip_bps,
                extra_fee_usd=args.extra_fee_usd,
                max_trades_per_symbol=args.max_trades_per_symbol,
                symbol_cooldown_days=args.symbol_cooldown_days,
            )
            selected_rule = calibrated_rule["best"]["rule"]
        except RuntimeError as exc:
            calibrated_rule = {"error": str(exc), "best": None, "top_candidates": [], "evaluated_rules": 0}
            selected_rule = {
                "name": "no_trade_no_calibrated_rule",
                "score_quantile": 1.0,
                "score_threshold": float("inf"),
            }

    test_scored = score_dataset_ensemble(
        Path(args.test_data),
        world_ckpt,
        allocator_ckpts,
        device,
        args.batch_size,
        args.limit_rows,
        args.min_horizon_bars,
        args.max_horizon_bars,
        args.seed,
    )
    if args.entry_only:
        test_scored = entry_candidates(test_scored)
    test_scored, test_filter_summary = tradability_filters(test_scored, args)
    test_planner = planner_rows(test_scored)
    time_col = _time_col(test_planner)
    test_start = pd.to_datetime(test_planner[time_col], utc=True).min()
    exit_col = "exit_timestamp" if "exit_timestamp" in test_planner.columns else time_col
    test_end = pd.to_datetime(test_planner[exit_col], utc=True).max()
    active_groups = apply_trade_rule(test_planner, selected_rule)
    timestamp_candidates = one_candidate_per_timestamp(active_groups)
    sequential = constrained_sequential_portfolio(
        active_groups,
        args.starting_equity,
        idle_asset=args.idle_asset,
        test_start=test_start,
        test_end=test_end,
        extra_roundtrip_bps=args.extra_roundtrip_bps,
        extra_fee_usd=args.extra_fee_usd,
        max_trades_per_symbol=args.max_trades_per_symbol,
        symbol_cooldown_days=args.symbol_cooldown_days,
    )

    payload = {
        "world_checkpoint": args.world_checkpoint,
        "allocator_checkpoint": args.allocator_checkpoint,
        "ensemble_allocator_checkpoints": list(args.ensemble_allocator_checkpoints),
        "calibration_data": args.calibration_data,
        "test_data": args.test_data,
        "device": device,
        "objective_mode": args.objective_mode,
        "entry_only": bool(args.entry_only),
        "idle_asset": args.idle_asset,
        "rule_mode": args.rule_mode,
        "extra_roundtrip_bps": float(args.extra_roundtrip_bps),
        "extra_fee_usd": float(args.extra_fee_usd),
        "max_trades_per_symbol": int(args.max_trades_per_symbol),
        "symbol_cooldown_days": float(args.symbol_cooldown_days),
        "tradability_filters": {
            "min_price": float(args.min_price),
            "min_trade_notional": float(args.min_trade_notional),
            "min_state_volume_z_1d": float(args.min_state_volume_z_1d),
            "max_state_vol_1d": float(args.max_state_vol_1d),
            "max_abs_state_ret_1d": float(args.max_abs_state_ret_1d),
        },
        "calibrated_rule_constraints": {
            "rule_validation_fraction": float(args.rule_validation_fraction),
            "min_calibration_trades": int(args.min_calibration_trades),
            "min_calibration_return": float(args.min_calibration_return),
            "min_calibration_profit_rate": float(args.min_calibration_profit_rate),
            "min_calibration_beat_spy_rate": float(args.min_calibration_beat_spy_rate),
            "min_validation_trades": int(args.min_validation_trades),
            "min_validation_return": float(args.min_validation_return),
            "min_validation_profit_rate": float(args.min_validation_profit_rate),
            "min_validation_beat_spy_rate": float(args.min_validation_beat_spy_rate),
            "max_calibration_drawdown": float(args.max_calibration_drawdown),
        },
        "calibration_filter_summary": calibration_filter_summary,
        "test_filter_summary": test_filter_summary,
        "selected_threshold": selected,
        "selected_trade_rule": selected_rule,
        "calibrated_rule_search": calibrated_rule,
        "calibration_groups": int(len(calibration_planner)),
        "rule_search_groups": int(len(rule_search_planner)),
        "rule_validation_groups": int(len(rule_validation_planner)) if rule_validation_planner is not None else 0,
        "test_groups": int(len(test_planner)),
        "active_group_summary": summarize_with_cash(
            "locked_test_active_groups",
            active_groups,
            len(test_planner),
            selected_rule["score_threshold"],
            selected_rule["score_quantile"],
        ),
        "timestamp_candidates": int(len(timestamp_candidates)),
        "sequential_portfolio": sequential,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps(payload, indent=2, default=str), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-data", required=True)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--world-checkpoint", required=True)
    parser.add_argument("--allocator-checkpoint", required=True)
    parser.add_argument("--ensemble-allocator-checkpoints", nargs="*", default=[], help="optional extra allocator checkpoints; scores are averaged and calibrated rules may gate disagreement")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--limit-rows", type=int, default=0)
    parser.add_argument("--min-horizon-bars", type=int, default=0)
    parser.add_argument("--max-horizon-bars", type=int, default=120)
    parser.add_argument("--min-coverage", type=float, default=0.05)
    parser.add_argument("--objective-mode", choices=["cash_return", "active_return", "hybrid"], default="hybrid")
    parser.add_argument("--rule-mode", choices=["fixed_threshold", "calibrated"], default="fixed_threshold")
    parser.add_argument("--fixed-score-quantile", type=float, default=-1.0, help="force fixed-threshold mode to use the nearest calibration quantile candidate")
    parser.add_argument("--fixed-max-target-position-frac", type=float, default=0.0, help="optional max target allocation for fixed-threshold rules")
    parser.add_argument("--fixed-max-horizon-bars", type=int, default=0, help="optional max horizon for fixed-threshold rules")
    parser.add_argument("--max-calibration-drawdown", type=float, default=0.18)
    parser.add_argument("--rule-validation-fraction", type=float, default=0.0, help="reserve the latest calibration slice for rule validation")
    parser.add_argument("--min-calibration-trades", type=int, default=10)
    parser.add_argument("--min-calibration-return", type=float, default=0.0)
    parser.add_argument("--min-calibration-profit-rate", type=float, default=0.0)
    parser.add_argument("--min-calibration-beat-spy-rate", type=float, default=0.0)
    parser.add_argument("--min-validation-trades", type=int, default=0)
    parser.add_argument("--min-validation-return", type=float, default=0.0)
    parser.add_argument("--min-validation-profit-rate", type=float, default=0.0)
    parser.add_argument("--min-validation-beat-spy-rate", type=float, default=0.0)
    parser.add_argument("--extra-roundtrip-bps", type=float, default=0.0, help="extra per-active-trade cost stress in bps of target exposure")
    parser.add_argument("--extra-fee-usd", type=float, default=0.0, help="extra flat cost stress per active trade")
    parser.add_argument("--max-trades-per-symbol", type=int, default=0, help="optional cap on selected active trades per symbol")
    parser.add_argument("--symbol-cooldown-days", type=float, default=0.0, help="optional cooldown before reusing the same symbol")
    parser.add_argument("--min-price", type=float, default=0.0, help="observable liquidity guard: minimum entry price")
    parser.add_argument("--min-trade-notional", type=float, default=0.0, help="observable liquidity guard: minimum simulated trade notional")
    parser.add_argument("--min-state-volume-z-1d", type=float, default=-99.0, help="observable liquidity guard using one-day volume z-score")
    parser.add_argument("--max-state-vol-1d", type=float, default=0.0, help="observable risk guard using one-day state volatility")
    parser.add_argument("--max-abs-state-ret-1d", type=float, default=0.0, help="observable event guard using absolute one-day state return")
    parser.add_argument("--entry-only", action=argparse.BooleanOptionalAction, default=True, help="only allow buy entries from cash")
    parser.add_argument("--idle-asset", choices=["cash", "spy"], default="cash", help="asset held between model-selected trades")
    parser.add_argument("--starting-equity", type=float, default=50_000.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
