"""Generate live-style recommendations from a trained listwise daily ranker.

This is a research signal only. The current listwise model is a 2026 specialist
candidate, not a validated all-regime allocator.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from recommend_daily_ranker_consensus import build_live_features, load_dataset_features
from train_daily_listwise_ranker import ListwiseRanker
from train_daily_ranker import FEATURE_COLS, pick_device


def load_checkpoint(path: str, device: str) -> tuple[ListwiseRanker, dict, dict, dict, list[str]]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    feature_cols = ckpt.get("feature_cols", FEATURE_COLS)
    model = ListwiseRanker(
        len(feature_cols),
        int(cfg.get("hidden_dim", 160)),
        float(cfg.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["arrays"], ckpt["rule"], cfg, feature_cols


def score_day(day: pd.DataFrame, checkpoint: str, device: str, batch_size: int) -> pd.DataFrame:
    model, arrays, rule, cfg, feature_cols = load_checkpoint(checkpoint, device)
    x = day[feature_cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
    x = np.clip((x - arrays["x_mean"]) / arrays["x_std"], -10.0, 10.0).astype(np.float32)
    preds = []
    xt = torch.from_numpy(x)
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = xt[start:start + batch_size].to(device)
            score, profit, crash = model(xb)
            preds.append(torch.stack([score, torch.sigmoid(profit), torch.sigmoid(crash)], dim=1).cpu().numpy())
    arr = np.concatenate(preds, axis=0) if preds else np.zeros((0, 3), dtype=np.float32)
    out = day.copy().reset_index(drop=True)
    out["pred_utility"] = arr[:, 0]
    out["pred_profit"] = arr[:, 1]
    out["pred_crash"] = arr[:, 2]
    out["pred_top"] = out["pred_utility"].rank(pct=True).astype(float)
    out["pred_score"] = (
        out["pred_utility"]
        + float(cfg.get("score_profit_weight", 0.04)) * out["pred_profit"]
        + float(cfg.get("score_top_weight", 0.08)) * out["pred_top"]
        - float(cfg.get("score_crash_weight", 0.10)) * out["pred_crash"]
    )
    return out


def apply_rule(scored: pd.DataFrame, rule: dict) -> pd.DataFrame:
    if rule.get("no_trade"):
        return scored.iloc[:0].copy()
    active = scored[
        (scored["pred_score"] >= rule.get("score_threshold", float("inf")))
        & (scored["pred_profit"] >= rule.get("min_profit", 1.0))
        & (scored["pred_crash"] <= rule.get("max_crash", 0.0))
    ].copy()
    filters = {
        "min_spy_ret_20d": ("spy_ret_20d", ">="),
        "min_ret_20d": ("ret_20d", ">="),
        "min_rel_spy_20d": ("rel_spy_20d", ">="),
        "min_drawdown_60d": ("drawdown_60d", ">="),
        "max_vol_20d_rank": ("xsec_vol_20d_rank", "<="),
        "min_mkt_pct_positive_20d": ("mkt_pct_positive_20d", ">="),
        "min_mkt_pct_above_ma20": ("mkt_pct_above_ma20", ">="),
        "min_mkt_ret_20d_mean": ("mkt_ret_20d_mean", ">="),
        "max_mkt_ret_20d_dispersion": ("mkt_ret_20d_dispersion", "<="),
    }
    for rule_key, (col, op) in filters.items():
        val = rule.get(rule_key)
        if val is None or col not in active.columns:
            continue
        active = active[active[col] >= val] if op == ">=" else active[active[col] <= val]
    return active.sort_values("pred_score", ascending=False)


def rule_step_counts(scored: pd.DataFrame, rule: dict) -> dict:
    if rule.get("no_trade"):
        return {"no_trade_rule": int(len(scored)), "final": 0}
    active = scored.copy()
    counts = {"start": int(len(active))}
    steps = [
        ("score_threshold", lambda df: df["pred_score"] >= rule.get("score_threshold", float("inf"))),
        ("min_profit", lambda df: df["pred_profit"] >= rule.get("min_profit", 1.0)),
        ("max_crash", lambda df: df["pred_crash"] <= rule.get("max_crash", 0.0)),
    ]
    filters = {
        "min_spy_ret_20d": ("spy_ret_20d", ">="),
        "min_ret_20d": ("ret_20d", ">="),
        "min_rel_spy_20d": ("rel_spy_20d", ">="),
        "min_drawdown_60d": ("drawdown_60d", ">="),
        "max_vol_20d_rank": ("xsec_vol_20d_rank", "<="),
        "min_mkt_pct_positive_20d": ("mkt_pct_positive_20d", ">="),
        "min_mkt_pct_above_ma20": ("mkt_pct_above_ma20", ">="),
        "min_mkt_ret_20d_mean": ("mkt_ret_20d_mean", ">="),
        "max_mkt_ret_20d_dispersion": ("mkt_ret_20d_dispersion", "<="),
    }
    for name, predicate in steps:
        active = active[predicate(active)].copy()
        counts[name] = int(len(active))
    for rule_key, (col, op) in filters.items():
        val = rule.get(rule_key)
        if val is None or col not in active.columns:
            continue
        active = active[active[col] >= val] if op == ">=" else active[active[col] <= val]
        counts[rule_key] = int(len(active))
    counts["final"] = int(len(active))
    return counts


def run(args: argparse.Namespace) -> dict:
    device = pick_device(args.device)
    day, latest_date = load_dataset_features(args) if args.use_dataset else build_live_features(args)
    model, arrays, rule, checkpoint_config, feature_cols = load_checkpoint(args.checkpoint, device)
    del model, arrays
    scored = score_day(day, args.checkpoint, device, args.batch_size)
    active = apply_rule(scored, rule)
    top_all = scored.sort_values("pred_score", ascending=False).head(args.top_k)
    recs = active.head(args.top_k)[
        [
            "date", "symbol", "pred_score", "pred_utility", "pred_profit", "pred_crash",
            "pred_top", "ret_20d", "spy_ret_20d", "rel_spy_20d", "drawdown_60d",
            "mkt_pct_positive_20d", "mkt_pct_above_ma20",
        ]
    ].to_dict(orient="records") if not active.empty else []
    payload = {
        "decision_date": str(pd.Timestamp(latest_date).date()),
        "checkpoint": args.checkpoint,
        "checkpoint_config": checkpoint_config,
        "feature_cols": feature_cols,
        "feature_source": "dataset" if args.use_dataset else "live_cache",
        "device": device,
        "rule": rule,
        "diagnostics": {
            "feature_rows": int(len(day)),
            "rule_pass_rows": int(len(active)),
            "rule_step_counts": rule_step_counts(scored, rule),
            "top_unfiltered": top_all[["symbol", "pred_score", "pred_profit", "pred_crash", "ret_20d", "rel_spy_20d"]].to_dict(orient="records"),
            "market": {
                key: float(day[key].dropna().iloc[0]) if key in day.columns and not day[key].dropna().empty else None
                for key in (
                    "spy_ret_20d",
                    "mkt_pct_positive_20d",
                    "mkt_pct_above_ma20",
                    "mkt_ret_20d_mean",
                    "mkt_ret_20d_dispersion",
                )
            },
        },
        "recommendations": recs,
        "decision": "buy_candidates" if recs else "no_trade",
        "warning": "research_only_listwise_specialist_not_validated_across_folds",
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps(payload, indent=2, default=str), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/daily_listwise_ranker/exp5_riskadj_train2025_2026_strictcal/daily_listwise_ranker.pt")
    parser.add_argument("--dataset", default="checkpoints/daily_ranker/exp11_latest_dataset_h5_2026/daily_ranker_dataset.parquet")
    parser.add_argument("--output", default="checkpoints/daily_listwise_ranker/latest_listwise_recommendation_live.json")
    parser.add_argument("--date", default="")
    parser.add_argument("--use-dataset", action="store_true")
    parser.add_argument("--symbol-limit", type=int, default=503)
    parser.add_argument("--live-feature-cache", default="checkpoints/daily_ranker/latest_live_features.parquet")
    parser.add_argument("--refresh-live-features", action="store_true")
    parser.add_argument("--horizon-days", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="auto")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
