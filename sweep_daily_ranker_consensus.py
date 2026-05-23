"""Sweep daily-ranker consensus protocol settings across locked folds.

This script is deliberately conservative: no-trade is allowed for a fold, but
negative active alpha is not. A protocol is only interesting if it trades in at
least a configurable number of folds and has positive aggregate active alpha.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import pandas as pd

from evaluate_daily_ranker_consensus import Config as ConsensusConfig
from evaluate_daily_ranker_consensus import consensus_frame, pick_device, score_checkpoint
from train_daily_ranker import simulate


DATASET = "checkpoints/daily_ranker/exp11_latest_dataset_h5_2026/daily_ranker_dataset.parquet"
CKPT_2021 = "checkpoints/daily_ranker/exp10_long_history_alpha_h5_train2021_test2023/daily_ranker.pt"
CKPT_2022 = "checkpoints/daily_ranker/exp8_long_history_alpha_h5_test2023/daily_ranker.pt"
CKPT_2023 = "checkpoints/daily_ranker/exp8_long_history_alpha_h5_test2024/daily_ranker.pt"
CKPT_2024 = "checkpoints/daily_ranker/exp8_long_history_alpha_h5_2025/daily_ranker.pt"
CKPT_2025 = "checkpoints/daily_ranker/exp12_train2025_alpha_h5_2026/daily_ranker_recalibrated_loose.pt"


@dataclass(frozen=True)
class FoldSpec:
    name: str
    test_start: str
    test_end: str
    checkpoints: tuple[str, ...]


@dataclass(frozen=True)
class SweepConfig:
    dataset: str
    output: str
    horizon_days: int
    top_k: int
    max_positions: int
    batch_size: int
    device: str
    min_traded_folds: int
    min_total_trades: int
    min_total_active_alpha: float


def default_folds() -> list[FoldSpec]:
    return [
        FoldSpec("2023", "2023-01-01", "2024-01-01", (CKPT_2021, CKPT_2022)),
        FoldSpec("2024", "2024-01-01", "2025-01-01", (CKPT_2021, CKPT_2022, CKPT_2023)),
        FoldSpec("2025", "2025-01-01", "2026-01-01", (CKPT_2021, CKPT_2022, CKPT_2023, CKPT_2024)),
        FoldSpec("2026_ytd", "2026-01-01", "2026-05-10", (CKPT_2021, CKPT_2022, CKPT_2023, CKPT_2024, CKPT_2025)),
    ]


def checkpoint_scores(df: pd.DataFrame, fold: FoldSpec, device: str, batch_size: int) -> dict[str, pd.DataFrame]:
    dates = pd.to_datetime(df["date"], utc=True)
    test = df[(dates >= pd.Timestamp(fold.test_start, tz="UTC")) & (dates < pd.Timestamp(fold.test_end, tz="UTC"))].copy()
    return {
        checkpoint: score_checkpoint(test, checkpoint, device, batch_size)
        for checkpoint in fold.checkpoints
    }


def evaluate_fold(
    scored_by_checkpoint: dict[str, pd.DataFrame],
    fold: FoldSpec,
    config: SweepConfig,
    rule: dict,
) -> dict:
    consensus_config = ConsensusConfig(
        dataset=config.dataset,
        checkpoints=list(fold.checkpoints),
        output="",
        test_start=fold.test_start,
        test_end=fold.test_end,
        horizon_days=config.horizon_days,
        min_votes=rule["min_votes"],
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
        top_k=config.top_k,
        max_positions=config.max_positions,
        batch_size=config.batch_size,
        device=config.device,
    )
    consensus = consensus_frame(list(scored_by_checkpoint.values()), consensus_config)
    if consensus.empty:
        return {
            "fold": fold.name,
            "consensus_rows": 0,
            "trades": 0,
            "periods": 0,
            "total_return": 0.0,
            "spy_active_return": 0.0,
            "active_alpha_return": 0.0,
            "profit_rate": 0.0,
            "beat_spy_rate": 0.0,
            "max_drawdown": 0.0,
        }
    result = simulate(
        consensus,
        config.top_k,
        config.max_positions,
        config.horizon_days,
        score_threshold=-float("inf"),
        min_profit=0.0,
        max_crash=1.0,
    )
    return {"fold": fold.name, "consensus_rows": int(len(consensus)), **result}


def candidate_rules(folds: list[FoldSpec]) -> list[dict]:
    max_votes = max(len(f.checkpoints) for f in folds)
    rules = []
    for (
        min_vote_gap,
        min_pred_profit,
        max_pred_crash,
        min_pred_top,
        min_raw_score_quantile,
        min_spy_ret_20d,
        min_rel_spy_20d,
        min_ret_20d,
        min_drawdown_60d,
        min_mkt_pct_positive_20d,
        min_mkt_pct_above_ma20,
        min_mkt_ret_20d_mean,
        max_mkt_ret_20d_dispersion,
    ) in product(
        (1,),
        (0.0,),
        (1.0,),
        (0.0,),
        (0.0,),
        (None, 0.0, 0.03),
        (None, 0.0),
        (None, 0.03),
        (None,),
        (None, 0.55, 0.60),
        (None, 0.55, 0.60),
        (None, 0.0),
        (None, 0.08),
    ):
        rules.append(
            {
                "min_vote_gap": min_vote_gap,
                "max_votes": max_votes,
                "min_pred_profit": min_pred_profit,
                "max_pred_crash": max_pred_crash,
                "min_pred_top": min_pred_top,
                "min_raw_score_quantile": min_raw_score_quantile,
                "min_spy_ret_20d": min_spy_ret_20d,
                "min_rel_spy_20d": min_rel_spy_20d,
                "min_ret_20d": min_ret_20d,
                "min_drawdown_60d": min_drawdown_60d,
                "min_mkt_pct_positive_20d": min_mkt_pct_positive_20d,
                "min_mkt_pct_above_ma20": min_mkt_pct_above_ma20,
                "min_mkt_ret_20d_mean": min_mkt_ret_20d_mean,
                "max_mkt_ret_20d_dispersion": max_mkt_ret_20d_dispersion,
            }
        )
    return rules


def rule_for_fold(base_rule: dict, fold: FoldSpec) -> dict:
    out = dict(base_rule)
    out["min_votes"] = max(2, len(fold.checkpoints) - base_rule["min_vote_gap"])
    return out


def summarize(rule: dict, fold_results: list[dict], config: SweepConfig) -> dict:
    traded = [r for r in fold_results if r["trades"] > 0]
    negative = [r for r in fold_results if r["active_alpha_return"] < -1e-12]
    total_trades = sum(r["trades"] for r in fold_results)
    total_alpha = sum(r["active_alpha_return"] for r in fold_results)
    total_return = sum(r["total_return"] for r in fold_results)
    max_drawdown = min((r["max_drawdown"] for r in fold_results), default=0.0)
    objective = total_alpha + 0.25 * total_return - 0.50 * abs(max_drawdown) + 0.01 * total_trades
    passed = (
        not negative
        and len(traded) >= config.min_traded_folds
        and total_trades >= config.min_total_trades
        and total_alpha >= config.min_total_active_alpha
    )
    return {
        "rule": rule,
        "passed": passed,
        "objective": objective,
        "traded_folds": len(traded),
        "total_trades": total_trades,
        "total_return_sum": total_return,
        "total_active_alpha_sum": total_alpha,
        "worst_fold_drawdown": max_drawdown,
        "folds": fold_results,
    }


def run(config: SweepConfig) -> dict:
    device = pick_device(config.device)
    folds = default_folds()
    df = pd.read_parquet(config.dataset)
    scored = {
        fold.name: checkpoint_scores(df, fold, device, config.batch_size)
        for fold in folds
    }
    summaries = []
    for base_rule in candidate_rules(folds):
        fold_results = [
            evaluate_fold(scored[fold.name], fold, config, rule_for_fold(base_rule, fold))
            for fold in folds
        ]
        summaries.append(summarize(base_rule, fold_results, config))
    summaries.sort(key=lambda row: (row["passed"], row["objective"]), reverse=True)
    payload = {
        "config": asdict(config),
        "folds": [asdict(fold) for fold in folds],
        "passed_count": sum(1 for row in summaries if row["passed"]),
        "top": summaries[:20],
    }
    Path(config.output).parent.mkdir(parents=True, exist_ok=True)
    Path(config.output).write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps(payload, indent=2, default=str), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--output", default="checkpoints/daily_ranker/consensus_protocol_sweep.json")
    parser.add_argument("--horizon-days", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-positions", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--min-traded-folds", type=int, default=3)
    parser.add_argument("--min-total-trades", type=int, default=40)
    parser.add_argument("--min-total-active-alpha", type=float, default=0.05)
    run(SweepConfig(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
