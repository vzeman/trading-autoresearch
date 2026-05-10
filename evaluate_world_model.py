"""Evaluate the action-conditioned world model as a planner.

This script scores candidate action rows from the counterfactual dataset, picks
the best action per (timestamp, horizon), and compares realized outcomes against
simple baselines and the oracle best candidate.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_world_model import PortfolioWorldModel, TARGET_REGRESSION, TARGET_CLASSIFICATION, pick_device


def load_frame(path: Path, limit_rows: int = 0) -> pd.DataFrame:
    if path.is_dir():
        frames = []
        remaining = limit_rows
        for shard in sorted(path.glob("*.parquet")):
            df = pd.read_parquet(shard)
            if limit_rows > 0:
                if remaining <= 0:
                    break
                df = df.head(remaining)
                remaining -= len(df)
            frames.append(df)
        if not frames:
            raise RuntimeError(f"no parquet shards found under {path}")
        return pd.concat(frames, ignore_index=True)
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
    action_id = df["action"].astype(str).map(action_to_id).fillna(0).to_numpy(np.int64)
    return x, symbol_id, action_id


def predict(df: pd.DataFrame, ckpt: dict, device: str, batch_size: int) -> pd.DataFrame:
    x, symbol_id, action_id = prepare_inputs(df, ckpt)
    model = PortfolioWorldModel(
        n_features=len(ckpt["feature_cols"]),
        n_symbols=len(ckpt["symbols"]),
        n_actions=len(ckpt["actions"]),
        hidden_dim=ckpt["config"]["hidden_dim"],
        n_layers=ckpt["config"]["n_layers"],
        dropout=0.0,
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
            reg, cls = model(xb, sid, aid)
            reg_raw = reg * y_std + y_mean
            preds_reg.append(reg_raw.detach().cpu().numpy())
            preds_cls.append(torch.sigmoid(cls).detach().cpu().numpy())
    reg_arr = np.concatenate(preds_reg, axis=0)
    cls_arr = np.concatenate(preds_cls, axis=0)
    out = df.copy()
    for j, name in enumerate(TARGET_REGRESSION):
        out[f"pred_{name}"] = reg_arr[:, j]
    for j, name in enumerate(TARGET_CLASSIFICATION):
        out[f"pred_{name}"] = cls_arr[:, j]
    out["pred_score"] = (
        out["pred_portfolio_return"]
        + 0.20 * out["pred_future_alpha_vs_spy"]
        + 0.50 * out["pred_beat_spy_label"]
        + 0.25 * out["pred_profit_label"]
        + 0.50 * out["pred_max_drawdown"]
        - 0.10 * out["pred_path_vol"]
    )
    return out


def summarize_selection(name: str, selected: pd.DataFrame) -> dict:
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
    group_cols = ["timestamp", "horizon_bars"]
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

    return {
        "planner": summarize_selection("planner", planner),
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
    parser.add_argument("--device", default=os.environ.get("WORLD_MODEL_DEVICE", "auto"))
    args = parser.parse_args()

    device = pick_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    df = load_frame(Path(args.data), args.limit_rows)
    timestamps = pd.to_datetime(df["timestamp"], utc=True)
    cutoff = timestamps.quantile(0.80)
    val_df = df[timestamps > cutoff].reset_index(drop=True)
    scored = predict(val_df, ckpt, device=device, batch_size=args.batch_size)
    result = evaluate_planner(scored)
    result["rows_scored"] = int(len(scored))
    result["groups"] = int(scored.groupby(["timestamp", "horizon_bars"]).ngroups)
    result["device"] = device
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
