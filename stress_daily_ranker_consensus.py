"""Stress-test the fixed regime-gated daily-ranker consensus protocol.

This script intentionally does not reselect rules. It loads the best passing
rule from a sweep, applies that same protocol to the locked folds, exports the
actual simulated trades, and summarizes monthly/quarterly concentration.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_daily_ranker_consensus import Config as ConsensusConfig
from evaluate_daily_ranker_consensus import consensus_frame, pick_device, score_checkpoint
from recommend_daily_ranker_consensus import load_best_rule
from sweep_daily_ranker_consensus import DATASET, default_folds


def select_trades(scored: pd.DataFrame, top_k: int, max_positions: int, horizon_days: int) -> pd.DataFrame:
    selected = []
    next_date = pd.Timestamp.min.tz_localize("UTC")
    ordered = scored.sort_values(["date", "pred_score"], ascending=[True, False])
    for date, group in ordered.groupby("date", sort=True):
        date = pd.Timestamp(date)
        if date < next_date:
            continue
        selected.append(group.head(top_k))
        next_date = date + pd.Timedelta(days=max(1, horizon_days))
    if not selected:
        return scored.iloc[:0].copy()
    trades = pd.concat(selected, ignore_index=True)
    if max_positions > 0 and not trades.empty:
        trades = trades.groupby("date", group_keys=False).head(max_positions).reset_index(drop=True)
    return trades


def summarize_returns(trades: pd.DataFrame, roundtrip_cost: float) -> dict:
    if trades.empty:
        return {
            "trades": 0,
            "periods": 0,
            "total_return": 0.0,
            "spy_active_return": 0.0,
            "active_alpha_return": 0.0,
            "profit_rate": 0.0,
            "beat_spy_rate": 0.0,
            "mean_return": 0.0,
            "max_drawdown": 0.0,
        }
    out = trades.copy()
    out["_net_return"] = out["future_return"].astype(float) - roundtrip_cost
    out["_spy_return"] = out["future_spy_return"].astype(float)
    period = out.groupby("date")["_net_return"].mean().sort_index()
    spy_period = out.groupby("date")["_spy_return"].mean().reindex(period.index).fillna(0.0)
    eq = (1.0 + period.to_numpy()).cumprod()
    spy_eq = (1.0 + spy_period.to_numpy()).cumprod()
    curve = np.r_[1.0, eq]
    peaks = np.maximum.accumulate(curve)
    dd = (curve - peaks) / np.maximum(peaks, 1e-12)
    return {
        "trades": int(len(out)),
        "periods": int(len(period)),
        "total_return": float(eq[-1] - 1.0),
        "spy_active_return": float(spy_eq[-1] - 1.0),
        "active_alpha_return": float((eq[-1] - 1.0) - (spy_eq[-1] - 1.0)),
        "profit_rate": float((out["_net_return"] > 0).mean()),
        "beat_spy_rate": float((out["future_alpha"].astype(float) > 0).mean()),
        "mean_return": float(out["_net_return"].mean()),
        "max_drawdown": float(dd.min()),
    }


def period_table(trades: pd.DataFrame, freq: str, roundtrip_cost: float) -> list[dict]:
    if trades.empty:
        return []
    rows = []
    work = trades.copy()
    work["period"] = pd.to_datetime(work["date"], utc=True).dt.to_period(freq).astype(str)
    for period, group in work.groupby("period", sort=True):
        row = summarize_returns(group, roundtrip_cost)
        row["period"] = period
        rows.append(row)
    return rows


def consensus_config(args: argparse.Namespace, fold, rule: dict, device: str) -> ConsensusConfig:
    checkpoints = list(fold.checkpoints)
    min_votes = max(2, len(checkpoints) - int(rule["min_vote_gap"]))
    return ConsensusConfig(
        dataset=args.dataset,
        checkpoints=checkpoints,
        output="",
        test_start=fold.test_start,
        test_end=fold.test_end,
        horizon_days=args.horizon_days,
        min_votes=min_votes,
        min_pred_profit=rule["min_pred_profit"],
        max_pred_crash=rule["max_pred_crash"],
        min_pred_top=rule["min_pred_top"],
        min_raw_score_quantile=rule["min_raw_score_quantile"],
        min_spy_ret_20d=rule["min_spy_ret_20d"],
        min_rel_spy_20d=rule["min_rel_spy_20d"],
        min_ret_20d=rule["min_ret_20d"],
        min_drawdown_60d=rule["min_drawdown_60d"],
        min_mkt_pct_positive_20d=rule["min_mkt_pct_positive_20d"],
        min_mkt_pct_above_ma20=rule["min_mkt_pct_above_ma20"],
        min_mkt_ret_20d_mean=rule["min_mkt_ret_20d_mean"],
        max_mkt_ret_20d_dispersion=rule["max_mkt_ret_20d_dispersion"],
        top_k=args.top_k,
        max_positions=args.max_positions,
        batch_size=args.batch_size,
        device=device,
    )


def run(args: argparse.Namespace) -> dict:
    device = pick_device(args.device)
    rule = load_best_rule(args.protocol_sweep, args.allow_unpassed_rule)
    df = pd.read_parquet(args.dataset)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    trades_by_fold = []
    fold_summaries = []

    for fold in default_folds():
        dates = pd.to_datetime(df["date"], utc=True)
        test = df[(dates >= pd.Timestamp(fold.test_start, tz="UTC")) & (dates < pd.Timestamp(fold.test_end, tz="UTC"))].copy()
        scored_parts = [
            score_checkpoint(test, checkpoint, device, args.batch_size)
            for checkpoint in fold.checkpoints
        ]
        config = consensus_config(args, fold, rule, device)
        consensus = consensus_frame(scored_parts, config)
        trades = select_trades(consensus, args.top_k, args.max_positions, args.horizon_days)
        if not trades.empty:
            trades = trades.copy()
            trades["fold"] = fold.name
            trades["_net_return"] = trades["future_return"].astype(float) - args.roundtrip_cost
            trades_by_fold.append(trades)
        summary = summarize_returns(trades, args.roundtrip_cost)
        summary.update({
            "fold": fold.name,
            "consensus_rows": int(len(consensus)),
            "test_start": fold.test_start,
            "test_end": fold.test_end,
            "min_votes": config.min_votes,
        })
        fold_summaries.append(summary)

    all_trades = pd.concat(trades_by_fold, ignore_index=True) if trades_by_fold else pd.DataFrame()
    aggregate = summarize_returns(all_trades, args.roundtrip_cost)
    months = period_table(all_trades, "M", args.roundtrip_cost)
    quarters = period_table(all_trades, "Q", args.roundtrip_cost)
    negative_months = [row for row in months if row["active_alpha_return"] < 0]
    payload = {
        "config": {
            "dataset": args.dataset,
            "protocol_sweep": args.protocol_sweep,
            "horizon_days": args.horizon_days,
            "top_k": args.top_k,
            "max_positions": args.max_positions,
            "roundtrip_cost": args.roundtrip_cost,
            "device": device,
        },
        "rule": rule,
        "folds": fold_summaries,
        "aggregate": aggregate,
        "monthly_periods": months,
        "quarterly_periods": quarters,
        "stress": {
            "positive_alpha_months": sum(1 for row in months if row["active_alpha_return"] > 0),
            "negative_alpha_months": len(negative_months),
            "worst_month": min(months, key=lambda row: row["active_alpha_return"]) if months else None,
            "best_month": max(months, key=lambda row: row["active_alpha_return"]) if months else None,
        },
        "trade_csv": args.trade_csv,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2, default=str))
    if args.trade_csv and not all_trades.empty:
        keep = [
            "fold", "date", "symbol", "votes", "pred_score", "pred_profit", "pred_crash",
            "pred_top", "future_return", "future_spy_return", "future_alpha", "_net_return",
            "ret_20d", "spy_ret_20d", "rel_spy_20d", "mkt_pct_positive_20d",
            "mkt_pct_above_ma20",
        ]
        Path(args.trade_csv).parent.mkdir(parents=True, exist_ok=True)
        all_trades[[c for c in keep if c in all_trades.columns]].to_csv(args.trade_csv, index=False)
    print(json.dumps(payload, indent=2, default=str), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--protocol-sweep", default="checkpoints/daily_ranker/consensus_protocol_sweep_regime_min3_with2025.json")
    parser.add_argument("--output", default="checkpoints/daily_ranker/consensus_protocol_stress.json")
    parser.add_argument("--trade-csv", default="checkpoints/daily_ranker/consensus_protocol_trades.csv")
    parser.add_argument("--horizon-days", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-positions", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--roundtrip-cost", type=float, default=0.0015)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-unpassed-rule", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
