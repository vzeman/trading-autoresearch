"""Evaluate a consensus of trained daily ranker checkpoints.

Each checkpoint scores the target period with its own normalization and
calibrated rule. A trade is allowed only when enough independent checkpoints
select the same symbol on the same decision date.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_daily_ranker import DailyRanker, FEATURE_COLS, simulate


@dataclass(frozen=True)
class Config:
    dataset: str
    checkpoints: list[str]
    output: str
    test_start: str
    test_end: str
    horizon_days: int
    min_votes: int
    min_pred_profit: float
    max_pred_crash: float
    min_pred_top: float
    min_raw_score_quantile: float
    min_spy_ret_20d: float | None
    min_rel_spy_20d: float | None
    min_ret_20d: float | None
    min_drawdown_60d: float | None
    min_mkt_pct_positive_20d: float | None
    min_mkt_pct_above_ma20: float | None
    min_mkt_ret_20d_mean: float | None
    max_mkt_ret_20d_dispersion: float | None
    top_k: int
    max_positions: int
    batch_size: int
    device: str


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_checkpoint(path: str, device: str) -> tuple[DailyRanker, dict, dict, list[str]]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    feature_cols = ckpt.get("feature_cols", FEATURE_COLS)
    cfg = ckpt["config"]
    hidden_dim = int(cfg.get("hidden_dim", 128))
    dropout = float(cfg.get("dropout", 0.0))
    model = DailyRanker(len(feature_cols), hidden_dim, dropout).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["arrays"], ckpt["rule"], feature_cols


def score_checkpoint(
    df: pd.DataFrame,
    checkpoint_path: str,
    device: str,
    batch_size: int,
) -> pd.DataFrame:
    model, arrays, rule, feature_cols = load_checkpoint(checkpoint_path, device)
    x = df[feature_cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
    mean = arrays["x_mean"]
    std = arrays["x_std"]
    x = np.clip((x - mean) / std, -10.0, 10.0).astype(np.float32)
    preds = []
    with torch.no_grad():
        xt = torch.from_numpy(x)
        for start in range(0, len(x), batch_size):
            xb = xt[start:start + batch_size].to(device)
            util, profit, crash, top = model(xb)
            preds.append(torch.stack([util, torch.sigmoid(profit), torch.sigmoid(crash), torch.sigmoid(top)], dim=1).cpu().numpy())
    arr = np.concatenate(preds, axis=0) if preds else np.zeros((0, 4), dtype=np.float32)
    out = df.copy().reset_index(drop=True)
    out["pred_utility"] = arr[:, 0] * arrays["utility_std"] + arrays["utility_mean"]
    out["pred_profit"] = arr[:, 1]
    out["pred_crash"] = arr[:, 2]
    out["pred_top"] = arr[:, 3]
    out["pred_score"] = out["pred_utility"] + 0.04 * out["pred_profit"] + 0.08 * out["pred_top"] - 0.10 * out["pred_crash"]
    if rule.get("no_trade"):
        return out.iloc[:0].copy()
    selected = simulate_active_rows(out, rule)
    selected["checkpoint"] = checkpoint_path
    return selected


def simulate_active_rows(scored: pd.DataFrame, rule: dict) -> pd.DataFrame:
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
        if op == ">=":
            active = active[active[col] >= val]
        else:
            active = active[active[col] <= val]
    return active


def consensus_frame(scored_parts: list[pd.DataFrame], config: Config) -> pd.DataFrame:
    if not scored_parts:
        return pd.DataFrame()
    scored = pd.concat(scored_parts, ignore_index=True)
    if scored.empty:
        return scored
    grouped = scored.groupby(["date", "symbol"], as_index=False).agg(
        votes=("checkpoint", "nunique"),
        pred_score=("pred_score", "mean"),
        pred_profit=("pred_profit", "mean"),
        pred_crash=("pred_crash", "mean"),
        pred_top=("pred_top", "mean"),
        future_return=("future_return", "first"),
        future_spy_return=("future_spy_return", "first"),
        future_alpha=("future_alpha", "first"),
        ret_20d=("ret_20d", "first"),
        spy_ret_20d=("spy_ret_20d", "first"),
        rel_spy_20d=("rel_spy_20d", "first"),
        drawdown_60d=("drawdown_60d", "first"),
        xsec_vol_20d_rank=("xsec_vol_20d_rank", "first"),
        mkt_pct_positive_20d=("mkt_pct_positive_20d", "first"),
        mkt_pct_above_ma20=("mkt_pct_above_ma20", "first"),
        mkt_ret_20d_mean=("mkt_ret_20d_mean", "first"),
        mkt_ret_20d_dispersion=("mkt_ret_20d_dispersion", "first"),
    )
    out = grouped[grouped["votes"] >= config.min_votes].copy()
    if out.empty:
        return out
    raw_score = out["pred_score"].astype(float)
    out["raw_consensus_score"] = raw_score
    if config.min_raw_score_quantile > 0:
        threshold = float(raw_score.quantile(config.min_raw_score_quantile))
        out = out[out["raw_consensus_score"] >= threshold].copy()
    if config.min_pred_profit > 0:
        out = out[out["pred_profit"] >= config.min_pred_profit].copy()
    if config.max_pred_crash < 1:
        out = out[out["pred_crash"] <= config.max_pred_crash].copy()
    if config.min_pred_top > 0:
        out = out[out["pred_top"] >= config.min_pred_top].copy()
    if config.min_spy_ret_20d is not None:
        out = out[out["spy_ret_20d"] >= config.min_spy_ret_20d].copy()
    if config.min_rel_spy_20d is not None:
        out = out[out["rel_spy_20d"] >= config.min_rel_spy_20d].copy()
    if config.min_ret_20d is not None:
        out = out[out["ret_20d"] >= config.min_ret_20d].copy()
    if config.min_drawdown_60d is not None:
        out = out[out["drawdown_60d"] >= config.min_drawdown_60d].copy()
    if config.min_mkt_pct_positive_20d is not None:
        out = out[out["mkt_pct_positive_20d"] >= config.min_mkt_pct_positive_20d].copy()
    if config.min_mkt_pct_above_ma20 is not None:
        out = out[out["mkt_pct_above_ma20"] >= config.min_mkt_pct_above_ma20].copy()
    if config.min_mkt_ret_20d_mean is not None:
        out = out[out["mkt_ret_20d_mean"] >= config.min_mkt_ret_20d_mean].copy()
    if config.max_mkt_ret_20d_dispersion is not None:
        out = out[out["mkt_ret_20d_dispersion"] <= config.max_mkt_ret_20d_dispersion].copy()
    if out.empty:
        return out
    out["pred_score"] = out["votes"].astype(float) + 0.10 * out["pred_score"].astype(float)
    return out


def run(config: Config) -> dict:
    device = pick_device(config.device)
    df = pd.read_parquet(config.dataset)
    dates = pd.to_datetime(df["date"], utc=True)
    test = df[(dates >= pd.Timestamp(config.test_start, tz="UTC")) & (dates < pd.Timestamp(config.test_end, tz="UTC"))].copy()
    scored_parts = [score_checkpoint(test, path, device, config.batch_size) for path in config.checkpoints]
    consensus = consensus_frame(scored_parts, config)
    if consensus.empty:
        result = {
            "trades": 0,
            "periods": 0,
            "total_return": 0.0,
            "spy_active_return": 0.0,
            "active_alpha_return": 0.0,
            "profit_rate": 0.0,
            "mean_return": 0.0,
            "max_drawdown": 0.0,
            "beat_spy_rate": 0.0,
        }
    else:
        result = simulate(
            consensus,
            config.top_k,
            config.max_positions,
            config.horizon_days,
            score_threshold=-float("inf"),
            min_profit=0.0,
            max_crash=1.0,
        )
    payload = {
        "config": asdict(config),
        "rows": int(len(df)),
        "test_rows": int(len(test)),
        "consensus_rows": int(len(consensus)),
        "result": result,
    }
    Path(config.output).parent.mkdir(parents=True, exist_ok=True)
    Path(config.output).write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps(payload, indent=2, default=str), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", dest="checkpoints", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--test-start", required=True)
    parser.add_argument("--test-end", required=True)
    parser.add_argument("--horizon-days", type=int, default=5)
    parser.add_argument("--min-votes", type=int, default=2)
    parser.add_argument("--min-pred-profit", type=float, default=0.0)
    parser.add_argument("--max-pred-crash", type=float, default=1.0)
    parser.add_argument("--min-pred-top", type=float, default=0.0)
    parser.add_argument("--min-raw-score-quantile", type=float, default=0.0)
    parser.add_argument("--min-spy-ret-20d", type=float, default=None)
    parser.add_argument("--min-rel-spy-20d", type=float, default=None)
    parser.add_argument("--min-ret-20d", type=float, default=None)
    parser.add_argument("--min-drawdown-60d", type=float, default=None)
    parser.add_argument("--min-mkt-pct-positive-20d", type=float, default=None)
    parser.add_argument("--min-mkt-pct-above-ma20", type=float, default=None)
    parser.add_argument("--min-mkt-ret-20d-mean", type=float, default=None)
    parser.add_argument("--max-mkt-ret-20d-dispersion", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-positions", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="auto")
    run(Config(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
