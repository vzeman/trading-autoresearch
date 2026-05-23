"""Evaluate the action-conditioned world model as a planner.

This script scores candidate action rows from the counterfactual dataset, picks
the best action per decision time and horizon, and compares realized outcomes
against simple baselines and the oracle best candidate.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_world_model import PortfolioWorldModel, TARGET_REGRESSION, TARGET_CLASSIFICATION, TARGET_CLIPS, action_keys, pick_device


def load_frame(path: Path, limit_rows: int = 0, seed: int = 0) -> pd.DataFrame:
    def sort_cols(frame: pd.DataFrame) -> list[str]:
        cols = ["decision_timestamp" if "decision_timestamp" in frame.columns else "timestamp"]
        if "symbol" in frame.columns:
            cols.append("symbol")
        return cols

    if path.is_dir():
        frames = []
        shards = sorted(path.glob("*.parquet"))
        per_shard = max(1, int(math.ceil(limit_rows / len(shards)))) if limit_rows > 0 and shards else 0
        for shard_idx, shard in enumerate(shards):
            df = pd.read_parquet(shard)
            if limit_rows > 0:
                n = min(per_shard, len(df))
                if n < len(df):
                    df = df.sample(n=n, random_state=seed + shard_idx).sort_values(sort_cols(df))
            frames.append(df)
        if not frames:
            raise RuntimeError(f"no parquet shards found under {path}")
        out = pd.concat(frames, ignore_index=True)
        if limit_rows > 0 and len(out) > limit_rows:
            out = out.sample(n=limit_rows, random_state=seed).sort_values(sort_cols(out))
        return out.reset_index(drop=True)
    df = pd.read_parquet(path)
    return df.head(limit_rows) if limit_rows > 0 else df


def prepare_inputs(df: pd.DataFrame, ckpt: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_cols = list(ckpt["feature_cols"])
    x = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
    if "horizon_bars" in feature_cols:
        x[:, feature_cols.index("horizon_bars")] = np.log1p(x[:, feature_cols.index("horizon_bars")])
    if "trade_notional" in feature_cols:
        x[:, feature_cols.index("trade_notional")] /= 50_000.0
    if "price" in feature_cols:
        x[:, feature_cols.index("price")] = np.log1p(np.maximum(x[:, feature_cols.index("price")], 0.0))
    x = (x - ckpt["x_mean"]) / ckpt["x_std"]
    x = np.clip(x, -20.0, 20.0).astype(np.float32)

    sym_to_id = {s: i for i, s in enumerate(ckpt["symbols"])}
    action_to_id = {a: i for i, a in enumerate(ckpt["actions"])}
    symbol_id = df["symbol"].astype(str).map(sym_to_id).fillna(0).to_numpy(np.int64)
    raw_action = df["action"].astype(str)
    keyed_action = action_keys(df)
    action_source = keyed_action if any("|" in str(a) for a in ckpt["actions"]) else raw_action
    action_id = action_source.map(action_to_id).fillna(0).to_numpy(np.int64)
    return x, symbol_id, action_id


def predict(df: pd.DataFrame, ckpt: dict, device: str, batch_size: int) -> pd.DataFrame:
    x, symbol_id, action_id = prepare_inputs(df, ckpt)
    target_regression = list(ckpt.get("target_regression", TARGET_REGRESSION))
    target_classification = list(ckpt.get("target_classification", TARGET_CLASSIFICATION))
    model = PortfolioWorldModel(
        n_features=len(ckpt["feature_cols"]),
        n_symbols=len(ckpt["symbols"]),
        n_actions=len(ckpt["actions"]),
        hidden_dim=ckpt["config"]["hidden_dim"],
        n_layers=ckpt["config"]["n_layers"],
        dropout=0.0,
        n_reg_targets=len(target_regression),
        n_cls_targets=len(target_classification),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    preds_reg = []
    preds_cls = []
    y_mean = torch.as_tensor(ckpt["y_mean"], device=device)
    y_std = torch.as_tensor(ckpt["y_std"], device=device)
    with torch.no_grad():
        for i in range(0, len(df), batch_size):
            xb = torch.from_numpy(x[i : i + batch_size]).to(device)
            sid = torch.from_numpy(symbol_id[i : i + batch_size].copy()).to(device)
            aid = torch.from_numpy(action_id[i : i + batch_size].copy()).to(device)
            reg, cls, rank = model(xb, sid, aid)
            reg_raw = reg * y_std + y_mean
            preds_reg.append(reg_raw.detach().cpu().numpy())
            preds_cls.append(torch.sigmoid(cls).detach().cpu().numpy())
            if i == 0:
                preds_rank = []
            preds_rank.append(torch.sigmoid(rank).detach().cpu().numpy())
    reg_arr = np.concatenate(preds_reg, axis=0)
    cls_arr = np.concatenate(preds_cls, axis=0)
    rank_arr = np.concatenate(preds_rank, axis=0)
    out = df.copy()
    for j, name in enumerate(target_regression):
        out[f"pred_{name}"] = reg_arr[:, j]
        if name in TARGET_CLIPS:
            lo, hi = TARGET_CLIPS[name]
            out[f"pred_{name}"] = out[f"pred_{name}"].clip(lo, hi)
    for j, name in enumerate(target_classification):
        out[f"pred_{name}"] = cls_arr[:, j]
    out["pred_rank_top_quartile"] = rank_arr
    for name in TARGET_REGRESSION:
        col = f"pred_{name}"
        if col not in out.columns:
            out[col] = 0.0
    for name in TARGET_CLASSIFICATION:
        col = f"pred_{name}"
        if col not in out.columns:
            out[col] = 0.0
    out["pred_score"] = (
        out["pred_portfolio_return"]
        + 0.20 * out["pred_future_alpha_vs_spy"]
        + 0.50 * out["pred_beat_spy_label"]
        + 0.25 * out["pred_profit_label"]
        + 0.50 * out["pred_rank_top_quartile"]
        + 0.50 * out["pred_max_drawdown"]
        - 0.10 * out["pred_path_vol"]
        - 0.30 * out["pred_asset_crash_label"]
        - 0.50 * out["pred_severe_adverse_label"]
        + 0.20 * out["pred_future_min_asset_return"]
        + 0.20 * out["pred_future_asset_max_drawdown"]
    )
    return out


def summarize_selection(name: str, selected: pd.DataFrame) -> dict:
    if len(selected) == 0:
        return {
            "name": name,
            "groups": 0,
            "mean_portfolio_return": 0.0,
            "median_portfolio_return": 0.0,
            "mean_pnl": 0.0,
            "mean_max_drawdown": 0.0,
            "profit_rate": 0.0,
            "beat_spy_rate": 0.0,
            "mean_future_alpha_vs_spy": 0.0,
            "action_mix": {},
            "horizon_mix": {},
        }
    return {
        "name": name,
        "groups": int(len(selected)),
        "mean_portfolio_return": float(selected["portfolio_return"].mean()),
        "median_portfolio_return": float(selected["portfolio_return"].median()),
        "mean_pnl": float(selected["portfolio_pnl"].mean()),
        "mean_max_drawdown": float(selected["max_drawdown"].mean()),
        "profit_rate": float(selected["profit_label"].mean()),
        "beat_spy_rate": float(selected["beat_spy_label"].mean()),
        "mean_future_alpha_vs_spy": float(selected["future_alpha_vs_spy"].mean()),
        "action_mix": selected["action"].value_counts().to_dict(),
        "horizon_mix": selected["horizon_bars"].value_counts().to_dict(),
    }


def select_by_idx(df: pd.DataFrame, idx: pd.Series) -> pd.DataFrame:
    return df.loc[idx.to_numpy()].reset_index(drop=True)


def evaluate_planner(scored: pd.DataFrame) -> dict:
    time_col = "decision_timestamp" if "decision_timestamp" in scored.columns else "timestamp"
    group_cols = [time_col, "horizon_bars"]
    candidates = scored.reset_index(drop=True)
    by_group = candidates.groupby(group_cols, sort=False)

    planner = select_by_idx(candidates, by_group["pred_score"].idxmax())
    oracle = select_by_idx(candidates, by_group["portfolio_return"].idxmax())
    random_pick = candidates.groupby(group_cols, sort=False).sample(n=1, random_state=0).reset_index(drop=True)

    buy_rows = candidates[candidates["action"] == "buy"]
    if len(buy_rows):
        buy_best = select_by_idx(buy_rows, buy_rows.groupby(group_cols, sort=False)["pred_score"].idxmax())
    else:
        buy_best = planner.iloc[:0].copy()

    hold_cash = candidates[
        (candidates["action"] == "hold")
        & (candidates["current_position_frac"] == 0.0)
        & (candidates["target_position_frac"] == 0.0)
    ]
    if len(hold_cash):
        hold_cash = select_by_idx(hold_cash, hold_cash.groupby(group_cols, sort=False)["pred_score"].idxmax())
    else:
        hold_cash = planner.iloc[:0].copy()

    threshold_results = []
    for q in (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95):
        threshold = float(planner["pred_score"].quantile(q))
        active = planner[planner["pred_score"] >= threshold].copy()
        inactive_groups = len(planner) - len(active)
        summary = summarize_selection(f"planner_q{q:.2f}", active)
        total_groups = len(planner)
        summary["threshold"] = threshold
        summary["active_groups"] = int(len(active))
        summary["cash_groups"] = int(inactive_groups)
        summary["coverage"] = float(len(active) / max(total_groups, 1))
        summary["portfolio_mean_return_with_cash"] = float(active["portfolio_return"].sum() / max(total_groups, 1))
        summary["portfolio_mean_pnl_with_cash"] = float(active["portfolio_pnl"].sum() / max(total_groups, 1))
        summary["beat_spy_rate_with_cash"] = float(active["beat_spy_label"].sum() / max(total_groups, 1))
        threshold_results.append(summary)

    best_threshold = max(
        threshold_results,
        key=lambda r: (r["portfolio_mean_return_with_cash"], r["beat_spy_rate_with_cash"]),
    ) if threshold_results else {}

    return {
        "planner": summarize_selection("planner", planner),
        "threshold_planner": threshold_results,
        "best_threshold_planner": best_threshold,
        "buy_only_planner": summarize_selection("buy_only_planner", buy_best) if len(buy_best) else {},
        "hold_cash": summarize_selection("hold_cash", hold_cash) if len(hold_cash) else {},
        "random": summarize_selection("random", random_pick),
        "oracle": summarize_selection("oracle", oracle),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/world_model/top100_train_counterfactual")
    parser.add_argument("--checkpoint", default="checkpoints/world_model/world_model_v1.pt")
    parser.add_argument("--output", default="checkpoints/world_model/world_model_v1_eval.json")
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--limit-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--score-all", action="store_true", help="score every loaded row instead of taking the final validation fraction")
    parser.add_argument("--min-horizon-bars", type=int, default=0)
    parser.add_argument("--max-horizon-bars", type=int, default=0)
    parser.add_argument("--device", default=os.environ.get("WORLD_MODEL_DEVICE", "auto"))
    args = parser.parse_args()

    device = pick_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    df = load_frame(Path(args.data), args.limit_rows, seed=args.seed)
    if args.min_horizon_bars > 0:
        df = df[df["horizon_bars"] >= args.min_horizon_bars].reset_index(drop=True)
    if args.max_horizon_bars > 0:
        df = df[df["horizon_bars"] <= args.max_horizon_bars].reset_index(drop=True)
    if df.empty:
        raise RuntimeError("no rows left after horizon filtering")
    time_col = "decision_timestamp" if "decision_timestamp" in df.columns else "timestamp"
    if args.score_all:
        val_df = df.reset_index(drop=True)
    else:
        timestamps = pd.to_datetime(df[time_col], utc=True)
        cutoff = timestamps.quantile(0.80)
        val_df = df[timestamps > cutoff].reset_index(drop=True)
    scored = predict(val_df, ckpt, device=device, batch_size=args.batch_size)
    result = evaluate_planner(scored)
    result["rows_scored"] = int(len(scored))
    result["groups"] = int(scored.groupby([time_col, "horizon_bars"]).ngroups)
    result["time_col"] = time_col
    result["min_horizon_bars"] = int(args.min_horizon_bars)
    result["max_horizon_bars"] = int(args.max_horizon_bars)
    result["device"] = device
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
