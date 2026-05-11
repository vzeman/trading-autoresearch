"""Walk-forward threshold evaluation for allocator-scored world-model trades.

This script freezes a trained world model and allocator, scores candidate
actions, chooses one action per timestamp/horizon, and then chooses the
cash/trade threshold from past data only before applying it to the next window.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluate_world_model import load_frame, predict, select_by_idx, summarize_selection
from train_allocator import apply_allocator, filter_frame
from train_world_model import pick_device


QUANTILES = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95)


def score_dataset(
    data: Path,
    world_ckpt: dict,
    allocator_ckpt: dict,
    device: str,
    batch_size: int,
    limit_rows: int,
    min_horizon_bars: int,
    max_horizon_bars: int,
    seed: int,
) -> pd.DataFrame:
    df = filter_frame(load_frame(data, limit_rows, seed=seed), min_horizon_bars, max_horizon_bars)
    scored = predict(df, world_ckpt, device=device, batch_size=batch_size)
    return apply_allocator(scored, allocator_ckpt, device=device, batch_size=batch_size)


def planner_rows(scored: pd.DataFrame) -> pd.DataFrame:
    time_col = "decision_timestamp" if "decision_timestamp" in scored.columns else "timestamp"
    group_cols = [time_col, "horizon_bars"]
    candidates = scored.reset_index(drop=True)
    idx = candidates.groupby(group_cols, sort=False)["pred_score"].idxmax()
    return select_by_idx(candidates, idx)


def summarize_with_cash(name: str, active: pd.DataFrame, total_groups: int, threshold: float, quantile: float) -> dict:
    summary = summarize_selection(name, active)
    summary["threshold"] = float(threshold)
    summary["quantile"] = float(quantile)
    summary["active_groups"] = int(len(active))
    summary["cash_groups"] = int(max(total_groups - len(active), 0))
    summary["coverage"] = float(len(active) / max(total_groups, 1))
    summary["portfolio_mean_return_with_cash"] = float(active["portfolio_return"].sum() / max(total_groups, 1))
    summary["portfolio_mean_pnl_with_cash"] = float(active["portfolio_pnl"].sum() / max(total_groups, 1))
    summary["beat_spy_rate_with_cash"] = float(active["beat_spy_label"].sum() / max(total_groups, 1))
    return summary


def objective(summary: dict, min_coverage: float, mode: str) -> float:
    if summary["coverage"] < min_coverage:
        return -1e9
    if mode == "cash_return":
        return (
            summary["portfolio_mean_return_with_cash"]
            + 0.25 * summary["mean_future_alpha_vs_spy"] * summary["coverage"]
            - 0.10 * abs(summary["mean_max_drawdown"]) * summary["coverage"]
        )
    if mode == "active_return":
        return (
            summary["mean_portfolio_return"]
            + 0.25 * summary["mean_future_alpha_vs_spy"]
            + 0.02 * summary["beat_spy_rate"]
            - 0.10 * abs(summary["mean_max_drawdown"])
        )
    if mode == "hybrid":
        return (
            0.50 * summary["portfolio_mean_return_with_cash"]
            + 0.50 * summary["mean_portfolio_return"] * min(summary["coverage"], 0.25)
            + 0.25 * summary["mean_future_alpha_vs_spy"] * summary["coverage"]
            - 0.10 * abs(summary["mean_max_drawdown"]) * summary["coverage"]
        )
    raise ValueError(f"unknown objective mode: {mode}")


def choose_threshold(calibration: pd.DataFrame, min_coverage: float, objective_mode: str) -> dict:
    total_groups = len(calibration)
    best: dict | None = None
    all_results = []
    for q in QUANTILES:
        threshold = float(calibration["pred_score"].quantile(q))
        active = calibration[calibration["pred_score"] >= threshold].copy()
        summary = summarize_with_cash(f"calibration_q{q:.2f}", active, total_groups, threshold, q)
        summary["objective"] = float(objective(summary, min_coverage, objective_mode))
        all_results.append(summary)
        if best is None or summary["objective"] > best["objective"]:
            best = summary
    if best is None:
        raise RuntimeError("could not choose threshold")
    return {"best": best, "candidates": all_results}


def split_by_time(planner: pd.DataFrame, folds: int) -> list[pd.DataFrame]:
    time_col = "decision_timestamp" if "decision_timestamp" in planner.columns else "timestamp"
    times = pd.Series(pd.to_datetime(planner[time_col], utc=True).sort_values().unique())
    chunks = np.array_split(times.to_numpy(), folds)
    out = []
    ts = pd.to_datetime(planner[time_col], utc=True)
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        mask = ts.isin(pd.to_datetime(chunk, utc=True))
        fold = planner[mask.to_numpy()].copy().sort_values([time_col, "horizon_bars"]).reset_index(drop=True)
        if len(fold):
            out.append(fold)
    return out


def evaluate_fixed_calibration(calibration: pd.DataFrame, test: pd.DataFrame, min_coverage: float, objective_mode: str) -> dict:
    total_groups = len(test)
    rows = []
    best = choose_threshold(calibration, min_coverage, objective_mode)["best"]
    for q in QUANTILES:
        threshold = float(calibration["pred_score"].quantile(q))
        active = test[test["pred_score"] >= threshold].copy()
        rows.append(summarize_with_cash(f"fixed_calibration_q{q:.2f}", active, total_groups, threshold, q))
    active_best = test[test["pred_score"] >= best["threshold"]].copy()
    applied_best = summarize_with_cash("fixed_calibration_selected", active_best, total_groups, best["threshold"], best["quantile"])
    return {
        "selected_from_calibration": best,
        "applied_selected": applied_best,
        "all_quantiles": rows,
    }


def run_walk_forward(calibration: pd.DataFrame, test: pd.DataFrame, folds: int, expanding: bool, min_coverage: float, objective_mode: str) -> dict:
    fold_frames = split_by_time(test, folds)
    past = calibration.copy()
    fold_results = []
    active_frames = []
    for i, fold in enumerate(fold_frames, start=1):
        choice = choose_threshold(past, min_coverage, objective_mode)
        selected = choice["best"]
        active = fold[fold["pred_score"] >= selected["threshold"]].copy()
        applied = summarize_with_cash(
            f"walk_forward_fold_{i}",
            active,
            len(fold),
            selected["threshold"],
            selected["quantile"],
        )
        fold_results.append({
            "fold": i,
            "groups": int(len(fold)),
            "selected_quantile": selected["quantile"],
            "selected_threshold": selected["threshold"],
            "calibration_objective": selected["objective"],
            "applied": applied,
        })
        active_frames.append(active)
        if expanding:
            past = pd.concat([past, fold], ignore_index=True)
        else:
            past = fold

    active_all = pd.concat(active_frames, ignore_index=True) if active_frames else test.iloc[:0].copy()
    aggregate = summarize_with_cash("walk_forward_aggregate", active_all, len(test), float("nan"), float("nan"))
    return {
        "folds": fold_results,
        "aggregate": aggregate,
    }


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
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-coverage", type=float, default=0.05)
    parser.add_argument("--objective-mode", choices=["cash_return", "active_return", "hybrid"], default="cash_return")
    parser.add_argument("--rolling", action="store_true", help="use only the previous fold after the initial calibration")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

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
    calibration_planner = planner_rows(calibration_scored)
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
    test_planner = planner_rows(test_scored)

    fixed = evaluate_fixed_calibration(calibration_planner, test_planner, args.min_coverage, args.objective_mode)
    walk = run_walk_forward(
        calibration_planner,
        test_planner,
        args.folds,
        expanding=not args.rolling,
        min_coverage=args.min_coverage,
        objective_mode=args.objective_mode,
    )
    payload = {
        "world_checkpoint": args.world_checkpoint,
        "allocator_checkpoint": args.allocator_checkpoint,
        "calibration_data": args.calibration_data,
        "test_data": args.test_data,
        "device": device,
        "min_horizon_bars": int(args.min_horizon_bars),
        "max_horizon_bars": int(args.max_horizon_bars),
        "calibration_groups": int(len(calibration_planner)),
        "test_groups": int(len(test_planner)),
        "folds": int(args.folds),
        "expanding": not args.rolling,
        "min_coverage": float(args.min_coverage),
        "objective_mode": args.objective_mode,
        "fixed_calibration": fixed,
        "walk_forward": walk,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps(payload, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
