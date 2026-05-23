"""Recalibrate the saved daily-ranker trade rule without retraining weights."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch

from train_daily_ranker import (
    Config,
    DailyRanker,
    FEATURE_COLS,
    choose_rule,
    make_arrays,
    pick_device,
    score_frame,
    simulate,
    split_masks,
)


def run(args: argparse.Namespace) -> dict:
    device = pick_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = Config(**ckpt["config"])
    config = replace(
        config,
        min_validation_trades=args.min_validation_trades,
        min_validation_return=args.min_validation_return,
        min_validation_active_alpha=args.min_validation_active_alpha,
        min_validation_profit_rate=args.min_validation_profit_rate,
        min_validation_beat_spy_rate=args.min_validation_beat_spy_rate,
        max_validation_drawdown=args.max_validation_drawdown,
        rule_validation_fraction=args.rule_validation_fraction,
        min_rule_validation_trades=args.min_rule_validation_trades,
        observed_score_weight=args.observed_score_weight,
        device=device,
    )
    df = pd.read_parquet(args.dataset)
    train_mask, val_mask, test_mask = split_masks(df, config)
    arrays = make_arrays(df, train_mask)
    hidden_dim = int(ckpt["config"].get("hidden_dim", 160))
    dropout = float(ckpt["config"].get("dropout", 0.0))
    model = DailyRanker(len(FEATURE_COLS), hidden_dim, dropout).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    val_scored = score_frame(model, df, arrays, val_mask, device, config.batch_size, config.observed_score_weight)
    test_scored = score_frame(model, df, arrays, test_mask, device, config.batch_size, config.observed_score_weight)
    choice = choose_rule(val_scored, config)
    rule = choice["rule"]
    test_result = simulate(
        test_scored,
        config.top_k,
        config.max_positions,
        config.horizon_days,
        rule.get("score_threshold", float("inf")),
        rule.get("min_profit", 1.0),
        rule.get("max_crash", 0.0),
        min_spy_ret_20d=rule.get("min_spy_ret_20d"),
        min_ret_20d=rule.get("min_ret_20d"),
        min_rel_spy_20d=rule.get("min_rel_spy_20d"),
        min_drawdown_60d=rule.get("min_drawdown_60d"),
        max_vol_20d_rank=rule.get("max_vol_20d_rank"),
        min_mkt_pct_positive_20d=rule.get("min_mkt_pct_positive_20d"),
        min_mkt_pct_above_ma20=rule.get("min_mkt_pct_above_ma20"),
        min_mkt_ret_20d_mean=rule.get("min_mkt_ret_20d_mean"),
        max_mkt_ret_20d_dispersion=rule.get("max_mkt_ret_20d_dispersion"),
    )

    out_ckpt = dict(ckpt)
    out_ckpt["config"] = {**ckpt["config"], **config.__dict__}
    out_ckpt["rule"] = rule
    out_ckpt["recalibration"] = {
        "source_checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "rule_selection": choice,
        "test_result": test_result,
    }
    Path(args.output_checkpoint).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_ckpt, args.output_checkpoint)
    payload = {
        "source_checkpoint": args.checkpoint,
        "output_checkpoint": args.output_checkpoint,
        "dataset": args.dataset,
        "rule_selection": choice,
        "test_result": test_result,
    }
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps(payload, indent=2, default=str), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--min-validation-trades", type=int, default=10)
    parser.add_argument("--min-validation-return", type=float, default=-1.0)
    parser.add_argument("--min-validation-active-alpha", type=float, default=-1.0)
    parser.add_argument("--min-validation-profit-rate", type=float, default=0.0)
    parser.add_argument("--min-validation-beat-spy-rate", type=float, default=0.0)
    parser.add_argument("--max-validation-drawdown", type=float, default=1.0)
    parser.add_argument("--rule-validation-fraction", type=float, default=0.0)
    parser.add_argument("--min-rule-validation-trades", type=int, default=1)
    parser.add_argument("--observed-score-weight", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
