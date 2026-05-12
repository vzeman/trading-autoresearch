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
    ].copy()


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
        spy_ret = float(row.get("future_spy_return", 0.0))
        before = equity
        equity *= 1.0 + ret
        spy_equity *= 1.0 + spy_ret
        next_available = max(exit_ts, entry_ts)
        portfolio_clock = next_available
        curve.append({"timestamp": str(exit_ts), "equity": equity, "spy_active_equity": spy_equity})
        details.append({
            "entry_timestamp": str(entry_ts),
            "exit_timestamp": str(exit_ts),
            "symbol": str(row["symbol"]),
            "action": str(row["action"]),
            "horizon_bars": int(row["horizon_bars"]),
            "target_position_frac": float(row["target_position_frac"]),
            "pred_score": float(row["pred_score"]),
            "portfolio_return": ret,
            "future_spy_return": spy_ret,
            "future_alpha_vs_spy": float(row["future_alpha_vs_spy"]),
            "idle_asset": idle_asset,
            "idle_return_before_entry": idle_return,
            "equity_before": before,
            "equity_after": equity,
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
        "trades": int(len(details)),
        "skipped_overlap": int(skipped),
        "profit_rate": profit_rate,
        "beat_spy_rate": beat_spy_rate,
        "max_drawdown": max_dd,
        "equity_curve": curve,
        "trades_detail": details,
    }


def run(args: argparse.Namespace) -> dict:
    device = pick_device(args.device)
    world_ckpt = torch.load(args.world_checkpoint, map_location="cpu", weights_only=False)
    allocator_ckpt = torch.load(args.allocator_checkpoint, map_location="cpu", weights_only=False)

    calibration_scored = score_dataset(
        Path(args.calibration_data),
        world_ckpt,
        allocator_ckpt,
        device,
        args.batch_size,
        args.limit_rows,
        args.min_horizon_bars,
        args.max_horizon_bars,
        args.seed,
    )
    if args.entry_only:
        calibration_scored = entry_candidates(calibration_scored)
    calibration_planner = planner_rows(calibration_scored)
    threshold_choice = choose_threshold(calibration_planner, args.min_coverage, args.objective_mode)
    selected = threshold_choice["best"]

    test_scored = score_dataset(
        Path(args.test_data),
        world_ckpt,
        allocator_ckpt,
        device,
        args.batch_size,
        args.limit_rows,
        args.min_horizon_bars,
        args.max_horizon_bars,
        args.seed,
    )
    if args.entry_only:
        test_scored = entry_candidates(test_scored)
    test_planner = planner_rows(test_scored)
    time_col = _time_col(test_planner)
    test_start = pd.to_datetime(test_planner[time_col], utc=True).min()
    exit_col = "exit_timestamp" if "exit_timestamp" in test_planner.columns else time_col
    test_end = pd.to_datetime(test_planner[exit_col], utc=True).max()
    active_groups = test_planner[test_planner["pred_score"] >= selected["threshold"]].copy()
    timestamp_candidates = one_candidate_per_timestamp(active_groups)
    sequential = sequential_portfolio(
        timestamp_candidates,
        args.starting_equity,
        idle_asset=args.idle_asset,
        test_start=test_start,
        test_end=test_end,
    )

    payload = {
        "world_checkpoint": args.world_checkpoint,
        "allocator_checkpoint": args.allocator_checkpoint,
        "calibration_data": args.calibration_data,
        "test_data": args.test_data,
        "device": device,
        "objective_mode": args.objective_mode,
        "entry_only": bool(args.entry_only),
        "idle_asset": args.idle_asset,
        "selected_threshold": selected,
        "calibration_groups": int(len(calibration_planner)),
        "test_groups": int(len(test_planner)),
        "active_group_summary": summarize_with_cash(
            "locked_test_active_groups",
            active_groups,
            len(test_planner),
            selected["threshold"],
            selected["quantile"],
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
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--limit-rows", type=int, default=0)
    parser.add_argument("--min-horizon-bars", type=int, default=0)
    parser.add_argument("--max-horizon-bars", type=int, default=120)
    parser.add_argument("--min-coverage", type=float, default=0.05)
    parser.add_argument("--objective-mode", choices=["cash_return", "active_return", "hybrid"], default="hybrid")
    parser.add_argument("--entry-only", action=argparse.BooleanOptionalAction, default=True, help="only allow buy entries from cash")
    parser.add_argument("--idle-asset", choices=["cash", "spy"], default="cash", help="asset held between model-selected trades")
    parser.add_argument("--starting-equity", type=float, default=50_000.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
