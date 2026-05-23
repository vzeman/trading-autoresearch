"""Evaluate non-ML daily ranking rules on the daily ranker dataset.

The goal is to keep the neural model honest. If a simple observed-feature rule
cannot survive validation and the locked year, a larger model is likely fitting
noise rather than learning a tradable edge.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from train_daily_ranker import simulate


@dataclass(frozen=True)
class Config:
    dataset: str
    output: str
    train_end: str
    test_start: str
    test_end: str
    validation_fraction: float
    horizon_days: int
    top_k: int
    max_positions: int
    min_validation_trades: int
    min_validation_return: float
    min_validation_profit_rate: float
    min_validation_beat_spy_rate: float
    max_validation_drawdown: float


def split_masks(df: pd.DataFrame, config: Config) -> tuple[np.ndarray, np.ndarray]:
    dates = pd.to_datetime(df["date"], utc=True)
    train_end = pd.Timestamp(config.train_end, tz="UTC")
    test_start = pd.Timestamp(config.test_start, tz="UTC")
    test_end = pd.Timestamp(config.test_end, tz="UTC")
    train_all = dates < train_end
    val_cutoff = dates[train_all].quantile(1.0 - max(0.0, min(config.validation_fraction, 0.8)))
    val_mask = ((dates >= val_cutoff) & (dates < train_end)).to_numpy()
    test_mask = ((dates >= test_start) & (dates < test_end)).to_numpy()
    return val_mask, test_mask


def zscore(series: pd.Series) -> pd.Series:
    std = series.std()
    if not np.isfinite(std) or std < 1e-8:
        return series * 0.0
    return (series - series.mean()) / std


def add_rule_score(df: pd.DataFrame, formula: str) -> pd.DataFrame:
    out = df.copy()
    grouped = out.groupby("date", group_keys=False)
    for col in ("ret_5d", "ret_20d", "ret_60d", "rel_spy_20d", "rel_spy_60d", "ma20_dist", "ma60_dist"):
        out[f"z_{col}"] = grouped[col].transform(zscore).clip(-5.0, 5.0)
    out["xsec_ret_60d_rank"] = grouped["ret_60d"].rank(pct=True)

    formulas = {
        "trend_low_vol": (
            0.45 * out["xsec_ret_20d_rank"]
            + 0.25 * out["xsec_ret_60d_rank"]
            + 0.20 * out["xsec_drawdown_60d_rank"]
            - 0.20 * out["xsec_vol_20d_rank"]
        ),
        "relative_trend": (
            0.35 * out["z_rel_spy_20d"]
            + 0.25 * out["z_rel_spy_60d"]
            + 0.25 * out["xsec_ret_20d_rank"]
            - 0.15 * out["xsec_vol_20d_rank"]
        ),
        "quality_momentum": (
            0.30 * out["z_ret_20d"]
            + 0.25 * out["z_ret_60d"]
            + 0.20 * out["z_ma20_dist"]
            + 0.15 * out["xsec_drawdown_60d_rank"]
            - 0.20 * out["xsec_vol_20d_rank"]
        ),
        "short_reversal_in_uptrend": (
            -0.35 * out["z_ret_5d"]
            + 0.35 * out["z_ret_60d"]
            + 0.20 * out["xsec_drawdown_60d_rank"]
            - 0.10 * out["xsec_vol_20d_rank"]
        ),
        "defensive_relative": (
            0.35 * out["z_rel_spy_20d"]
            + 0.30 * out["xsec_drawdown_60d_rank"]
            - 0.30 * out["xsec_vol_20d_rank"]
            + 0.15 * out["xsec_volume_z_20d_rank"]
        ),
    }
    out["pred_score"] = formulas[formula].replace([np.inf, -np.inf], np.nan).fillna(-999.0)
    out["pred_profit"] = 1.0
    out["pred_crash"] = 0.0
    return out


def choose_rule(val: pd.DataFrame, config: Config) -> dict:
    formulas = ("trend_low_vol", "relative_trend", "quality_momentum", "defensive_relative")
    min_spy_filters = (0.0,)
    min_ret_20_filters = (None, 0.03)
    min_ret_60_filters = (None, 0.05)
    min_rel_filters = (None, 0.0)
    drawdown_filters = (-0.15, -0.05)
    vol_filters = (0.80, 0.60)
    best = None
    candidates = []
    for formula in formulas:
        scored = add_rule_score(val, formula)
        thresholds = [float(scored["pred_score"].quantile(q)) for q in (0.80, 0.90, 0.95)]
        for threshold in thresholds:
            for min_spy in min_spy_filters:
                for min_ret_20 in min_ret_20_filters:
                    for min_ret_60 in min_ret_60_filters:
                        for min_rel in min_rel_filters:
                            for min_dd in drawdown_filters:
                                for max_vol in vol_filters:
                                    active = scored
                                    if min_ret_60 is not None:
                                        active = active[active["ret_60d"] >= min_ret_60]
                                    result = simulate(
                                        active,
                                        config.top_k,
                                        config.max_positions,
                                        config.horizon_days,
                                        threshold,
                                        min_profit=0.0,
                                        max_crash=1.0,
                                        min_spy_ret_20d=min_spy,
                                        min_ret_20d=min_ret_20,
                                        min_rel_spy_20d=min_rel,
                                        min_drawdown_60d=min_dd,
                                        max_vol_20d_rank=max_vol,
                                    )
                                    if result["trades"] < config.min_validation_trades:
                                        continue
                                    if result["total_return"] < config.min_validation_return:
                                        continue
                                    if result["profit_rate"] < config.min_validation_profit_rate:
                                        continue
                                    if result["beat_spy_rate"] < config.min_validation_beat_spy_rate:
                                        continue
                                    if abs(result["max_drawdown"]) > config.max_validation_drawdown:
                                        continue
                                    score = (
                                        result["active_alpha_return"]
                                        + 0.50 * result["total_return"]
                                        - 3.0 * abs(result["max_drawdown"])
                                        + 0.05 * result["profit_rate"]
                                    )
                                    row = {
                                        "formula": formula,
                                        "score_threshold": threshold,
                                        "min_spy_ret_20d": min_spy,
                                        "min_ret_20d": min_ret_20,
                                        "min_ret_60d": min_ret_60,
                                        "min_rel_spy_20d": min_rel,
                                        "min_drawdown_60d": min_dd,
                                        "max_vol_20d_rank": max_vol,
                                        "objective": score,
                                        "validation": result,
                                    }
                                    candidates.append(row)
                                    if best is None or score > best["objective"]:
                                        best = row
    if best is None:
        return {"rule": {"no_trade": True}, "candidates": []}
    rule_keys = (
        "formula", "score_threshold", "min_spy_ret_20d", "min_ret_20d", "min_ret_60d",
        "min_rel_spy_20d", "min_drawdown_60d", "max_vol_20d_rank",
    )
    return {"rule": {k: best[k] for k in rule_keys}, "best": best, "candidates": sorted(candidates, key=lambda x: x["objective"], reverse=True)[:20]}


def apply_rule(df: pd.DataFrame, rule: dict, config: Config) -> dict:
    if rule.get("no_trade"):
        return {"trades": 0, "periods": 0, "total_return": 0.0, "spy_active_return": 0.0, "active_alpha_return": 0.0}
    scored = add_rule_score(df, rule["formula"])
    if rule.get("min_ret_60d") is not None:
        scored = scored[scored["ret_60d"] >= rule["min_ret_60d"]]
    return simulate(
        scored,
        config.top_k,
        config.max_positions,
        config.horizon_days,
        rule["score_threshold"],
        min_profit=0.0,
        max_crash=1.0,
        min_spy_ret_20d=rule.get("min_spy_ret_20d"),
        min_ret_20d=rule.get("min_ret_20d"),
        min_rel_spy_20d=rule.get("min_rel_spy_20d"),
        min_drawdown_60d=rule.get("min_drawdown_60d"),
        max_vol_20d_rank=rule.get("max_vol_20d_rank"),
    )


def run(config: Config) -> dict:
    df = pd.read_parquet(config.dataset)
    val_mask, test_mask = split_masks(df, config)
    val = df[val_mask].copy()
    test = df[test_mask].copy()
    choice = choose_rule(val, config)
    test_result = apply_rule(test, choice["rule"], config)
    payload = {
        "config": asdict(config),
        "rows": int(len(df)),
        "validation_rows": int(val_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "rule_selection": choice,
        "test_result": test_result,
    }
    Path(config.output).parent.mkdir(parents=True, exist_ok=True)
    Path(config.output).write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps(payload, indent=2, default=str), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="checkpoints/daily_ranker/exp2_2025_regime_h5/daily_ranker_dataset.parquet")
    parser.add_argument("--output", default="checkpoints/daily_ranker/rule_baseline_h5_2025.json")
    parser.add_argument("--train-end", default="2024-01-01")
    parser.add_argument("--test-start", default="2025-01-01")
    parser.add_argument("--test-end", default="2026-01-01")
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--horizon-days", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-positions", type=int, default=3)
    parser.add_argument("--min-validation-trades", type=int, default=30)
    parser.add_argument("--min-validation-return", type=float, default=0.0)
    parser.add_argument("--min-validation-profit-rate", type=float, default=0.52)
    parser.add_argument("--min-validation-beat-spy-rate", type=float, default=0.50)
    parser.add_argument("--max-validation-drawdown", type=float, default=0.08)
    run(Config(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
