"""Tune a lightweight second-stage planner score for a trained world model.

The world model predicts candidate outcomes. This script tunes a linear
allocator score on a validation slice, then applies the fixed score to an
untouched test dataset. It is deliberately simple: no external ML libraries,
just deterministic random search over score weights.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluate_world_model import evaluate_planner, load_frame, predict
from train_world_model import pick_device


SCORE_COLUMNS = [
    "pred_portfolio_return",
    "pred_future_alpha_vs_spy",
    "pred_beat_spy_label",
    "pred_profit_label",
    "pred_rank_top_quartile",
    "pred_max_drawdown",
    "pred_path_vol",
    "target_position_frac",
    "trade_notional",
    "horizon_bars",
]


def filter_frame(df: pd.DataFrame, min_horizon: int, max_horizon: int) -> pd.DataFrame:
    if min_horizon > 0:
        df = df[df["horizon_bars"] >= min_horizon]
    if max_horizon > 0:
        df = df[df["horizon_bars"] <= max_horizon]
    if df.empty:
        raise RuntimeError("no rows left after horizon filtering")
    return df.reset_index(drop=True)


def validation_slice(df: pd.DataFrame, score_all: bool) -> pd.DataFrame:
    if score_all:
        return df.reset_index(drop=True)
    time_col = "decision_timestamp" if "decision_timestamp" in df.columns else "timestamp"
    ts = pd.to_datetime(df[time_col], utc=True)
    return df[ts > ts.quantile(0.80)].reset_index(drop=True)


def set_score(df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    score = np.zeros(len(out), dtype=np.float64)
    for col, weight in weights.items():
        if col == "bias":
            score += weight
            continue
        if col not in out.columns:
            continue
        values = out[col].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float64)
        if col == "horizon_bars":
            values = np.log1p(values)
        if col == "trade_notional":
            values = values / 50_000.0
        score += weight * values
    out["pred_score"] = score
    return out


def objective(result: dict, threshold_key: str) -> float:
    if threshold_key == "planner":
        row = result["planner"]
        return row["mean_portfolio_return"] + 0.25 * row["mean_future_alpha_vs_spy"] - 0.10 * abs(row["mean_max_drawdown"])
    for row in result["threshold_planner"]:
        if row["name"] == threshold_key:
            return (
                row["portfolio_mean_return_with_cash"]
                + 0.25 * row["mean_future_alpha_vs_spy"] * row["coverage"]
                - 0.10 * abs(row["mean_max_drawdown"]) * row["coverage"]
            )
    raise KeyError(threshold_key)


def random_weights(rng: np.random.Generator) -> dict[str, float]:
    return {
        "pred_portfolio_return": float(rng.uniform(0.0, 4.0)),
        "pred_future_alpha_vs_spy": float(rng.uniform(0.0, 2.0)),
        "pred_beat_spy_label": float(rng.uniform(0.0, 1.2)),
        "pred_profit_label": float(rng.uniform(0.0, 0.8)),
        "pred_rank_top_quartile": float(rng.uniform(0.0, 1.2)),
        "pred_max_drawdown": float(rng.uniform(0.0, 1.5)),
        "pred_path_vol": float(rng.uniform(-2.0, 0.0)),
        "target_position_frac": float(rng.uniform(-0.4, 0.4)),
        "trade_notional": float(rng.uniform(-0.3, 0.1)),
        "horizon_bars": float(rng.uniform(-0.3, 0.3)),
    }


def tune(scored: pd.DataFrame, trials: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    candidates: list[dict[str, float]] = [
        {
            "pred_portfolio_return": 1.0,
            "pred_future_alpha_vs_spy": 0.20,
            "pred_beat_spy_label": 0.50,
            "pred_profit_label": 0.25,
            "pred_rank_top_quartile": 0.50,
            "pred_max_drawdown": 0.50,
            "pred_path_vol": -0.10,
        }
    ]
    candidates.extend(random_weights(rng) for _ in range(max(0, trials)))
    threshold_keys = ["planner"] + [f"planner_q{q:.2f}" for q in (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95)]
    best = {"score": -1e9, "weights": {}, "threshold_key": "", "result": {}}
    for weights in candidates:
        result = evaluate_planner(set_score(scored, weights))
        for key in threshold_keys:
            score = objective(result, key)
            if score > best["score"]:
                best = {
                    "score": float(score),
                    "weights": weights,
                    "threshold_key": key,
                    "result": result,
                }
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--limit-rows", type=int, default=0)
    parser.add_argument("--min-horizon-bars", type=int, default=0)
    parser.add_argument("--max-horizon-bars", type=int, default=0)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = pick_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    train_df = filter_frame(load_frame(Path(args.train_data), args.limit_rows, seed=args.seed), args.min_horizon_bars, args.max_horizon_bars)
    train_df = validation_slice(train_df, score_all=False)
    train_scored = predict(train_df, ckpt, device=device, batch_size=args.batch_size)
    best = tune(train_scored, trials=args.trials, seed=args.seed)

    test_df = filter_frame(load_frame(Path(args.test_data), args.limit_rows, seed=args.seed), args.min_horizon_bars, args.max_horizon_bars)
    test_scored = predict(test_df, ckpt, device=device, batch_size=args.batch_size)
    test_result = evaluate_planner(set_score(test_scored, best["weights"]))

    payload = {
        "checkpoint": args.checkpoint,
        "train_rows_scored": int(len(train_scored)),
        "test_rows_scored": int(len(test_scored)),
        "min_horizon_bars": args.min_horizon_bars,
        "max_horizon_bars": args.max_horizon_bars,
        "trials": args.trials,
        "device": device,
        "best_validation": best,
        "test_result": test_result,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps(payload, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
