"""Strict walk-forward tradability test for allocator-scored world-model trades.

Unlike the planner-level walk-forward check, this script uses the sequential
portfolio simulator from evaluate_tradable_allocator.py. Each test fold selects
its threshold/rule from past data only, applies it to the next unseen window,
then all selected rows are stitched into one continuous portfolio path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from evaluate_tradable_allocator import (
    _date_range,
    apply_trade_rule,
    choose_trade_rule,
    constrained_sequential_portfolio,
    entry_candidates,
    rule_name,
    score_dataset_ensemble,
    tradability_filters,
)
from train_world_model import pick_device
from walk_forward_allocator import choose_threshold, planner_rows, split_by_time, summarize_with_cash


def _score_planner(
    data: str,
    world_ckpt: dict,
    allocator_ckpts: list[dict],
    args: argparse.Namespace,
    device: str,
) -> tuple[pd.DataFrame, dict]:
    scored = score_dataset_ensemble(
        Path(data),
        world_ckpt,
        allocator_ckpts,
        device,
        args.batch_size,
        args.limit_rows,
        args.min_horizon_bars,
        args.max_horizon_bars,
        args.seed,
    )
    if args.entry_only:
        scored = entry_candidates(scored)
    scored, filter_summary = tradability_filters(scored, args)
    return planner_rows(scored), filter_summary


def _fixed_rule_from_choice(past: pd.DataFrame, args: argparse.Namespace) -> tuple[dict, dict]:
    threshold_choice = choose_threshold(past, args.min_coverage, args.objective_mode)
    selected = threshold_choice["best"]
    if args.fixed_score_quantile >= 0:
        selected = min(
            threshold_choice["candidates"],
            key=lambda row: abs(float(row["quantile"]) - float(args.fixed_score_quantile)),
        )
    rule = {
        "name": f"fixed_threshold_q{selected['quantile']:.2f}",
        "score_quantile": float(selected["quantile"]),
        "score_threshold": float(selected["threshold"]),
    }
    if args.fixed_max_target_position_frac > 0:
        rule["max_target_position_frac"] = float(args.fixed_max_target_position_frac)
    if args.fixed_max_horizon_bars > 0:
        rule["max_horizon_bars"] = int(args.fixed_max_horizon_bars)
    rule["name"] = rule_name(rule)
    return rule, {"selected_threshold": selected, "threshold_candidates": threshold_choice["candidates"]}


def _select_rule(past: pd.DataFrame, args: argparse.Namespace) -> tuple[dict, dict]:
    if args.rule_mode == "fixed_threshold":
        return _fixed_rule_from_choice(past, args)
    calibrated = choose_trade_rule(
        past,
        min_coverage=args.min_coverage,
        idle_asset=args.idle_asset,
        starting_equity=args.starting_equity,
        max_calibration_drawdown=args.max_calibration_drawdown,
        extra_roundtrip_bps=args.extra_roundtrip_bps,
        extra_fee_usd=args.extra_fee_usd,
        max_trades_per_symbol=args.max_trades_per_symbol,
        symbol_cooldown_days=args.symbol_cooldown_days,
    )
    return calibrated["best"]["rule"], {"calibrated_rule_search": calibrated}


def _fold_bounds(fold: pd.DataFrame) -> dict:
    start, end = _date_range(fold)
    return {"start": str(start), "end": str(end)}


def run(args: argparse.Namespace) -> dict:
    device = pick_device(args.device)
    world_ckpt = torch.load(args.world_checkpoint, map_location="cpu", weights_only=False)
    allocator_paths = [args.allocator_checkpoint] + list(args.ensemble_allocator_checkpoints)
    allocator_ckpts = [torch.load(path, map_location="cpu", weights_only=False) for path in allocator_paths]

    calibration_planner, calibration_filter_summary = _score_planner(
        args.calibration_data,
        world_ckpt,
        allocator_ckpts,
        args,
        device,
    )
    test_planner, test_filter_summary = _score_planner(
        args.test_data,
        world_ckpt,
        allocator_ckpts,
        args,
        device,
    )

    past = calibration_planner.copy()
    fold_results = []
    active_frames = []
    for fold_idx, fold in enumerate(split_by_time(test_planner, args.folds), start=1):
        rule, selection = _select_rule(past, args)
        active = apply_trade_rule(fold, rule)
        fold_start, fold_end = _date_range(fold)
        fold_seq = constrained_sequential_portfolio(
            active,
            starting_equity=args.starting_equity,
            idle_asset=args.idle_asset,
            test_start=fold_start,
            test_end=fold_end,
            include_details=False,
            extra_roundtrip_bps=args.extra_roundtrip_bps,
            extra_fee_usd=args.extra_fee_usd,
            max_trades_per_symbol=args.max_trades_per_symbol,
            symbol_cooldown_days=args.symbol_cooldown_days,
        )
        fold_results.append({
            "fold": int(fold_idx),
            "bounds": _fold_bounds(fold),
            "past_groups": int(len(past)),
            "test_groups": int(len(fold)),
            "active_groups": int(len(active)),
            "coverage": float(len(active) / max(len(fold), 1)),
            "selected_trade_rule": rule,
            "selection": selection,
            "active_group_summary": summarize_with_cash(
                f"walk_forward_fold_{fold_idx}_active_groups",
                active,
                len(fold),
                rule["score_threshold"],
                rule["score_quantile"],
            ),
            "sequential_portfolio": fold_seq,
        })
        active_frames.append(active)
        if args.rolling:
            past = fold.copy()
        else:
            past = pd.concat([past, fold], ignore_index=True)

    active_all = pd.concat(active_frames, ignore_index=True) if active_frames else test_planner.iloc[:0].copy()
    test_start, test_end = _date_range(test_planner)
    sequential = constrained_sequential_portfolio(
        active_all,
        starting_equity=args.starting_equity,
        idle_asset=args.idle_asset,
        test_start=test_start,
        test_end=test_end,
        include_details=True,
        extra_roundtrip_bps=args.extra_roundtrip_bps,
        extra_fee_usd=args.extra_fee_usd,
        max_trades_per_symbol=args.max_trades_per_symbol,
        symbol_cooldown_days=args.symbol_cooldown_days,
    )

    payload = {
        "world_checkpoint": args.world_checkpoint,
        "allocator_checkpoint": args.allocator_checkpoint,
        "ensemble_allocator_checkpoints": list(args.ensemble_allocator_checkpoints),
        "calibration_data": args.calibration_data,
        "test_data": args.test_data,
        "device": device,
        "objective_mode": args.objective_mode,
        "entry_only": bool(args.entry_only),
        "idle_asset": args.idle_asset,
        "rule_mode": args.rule_mode,
        "fixed_score_quantile": float(args.fixed_score_quantile),
        "extra_roundtrip_bps": float(args.extra_roundtrip_bps),
        "extra_fee_usd": float(args.extra_fee_usd),
        "max_trades_per_symbol": int(args.max_trades_per_symbol),
        "symbol_cooldown_days": float(args.symbol_cooldown_days),
        "tradability_filters": {
            "min_price": float(args.min_price),
            "min_trade_notional": float(args.min_trade_notional),
            "min_state_volume_z_1d": float(args.min_state_volume_z_1d),
            "max_state_vol_1d": float(args.max_state_vol_1d),
            "max_abs_state_ret_1d": float(args.max_abs_state_ret_1d),
        },
        "calibration_filter_summary": calibration_filter_summary,
        "test_filter_summary": test_filter_summary,
        "calibration_groups": int(len(calibration_planner)),
        "test_groups": int(len(test_planner)),
        "folds": int(args.folds),
        "expanding": not args.rolling,
        "fold_results": fold_results,
        "active_group_summary": summarize_with_cash(
            "walk_forward_active_groups",
            active_all,
            len(test_planner),
            float("nan"),
            float("nan"),
        ),
        "sequential_portfolio": sequential,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps(payload, indent=2, default=str), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-data", required=True)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--world-checkpoint", required=True)
    parser.add_argument("--allocator-checkpoint", required=True)
    parser.add_argument("--ensemble-allocator-checkpoints", nargs="*", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--limit-rows", type=int, default=0)
    parser.add_argument("--min-horizon-bars", type=int, default=0)
    parser.add_argument("--max-horizon-bars", type=int, default=120)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-coverage", type=float, default=0.05)
    parser.add_argument("--objective-mode", choices=["cash_return", "active_return", "hybrid"], default="hybrid")
    parser.add_argument("--rule-mode", choices=["fixed_threshold", "calibrated"], default="fixed_threshold")
    parser.add_argument("--fixed-score-quantile", type=float, default=-1.0)
    parser.add_argument("--fixed-max-target-position-frac", type=float, default=0.0)
    parser.add_argument("--fixed-max-horizon-bars", type=int, default=0)
    parser.add_argument("--max-calibration-drawdown", type=float, default=0.18)
    parser.add_argument("--extra-roundtrip-bps", type=float, default=0.0)
    parser.add_argument("--extra-fee-usd", type=float, default=0.0)
    parser.add_argument("--max-trades-per-symbol", type=int, default=0)
    parser.add_argument("--symbol-cooldown-days", type=float, default=0.0)
    parser.add_argument("--min-price", type=float, default=0.0)
    parser.add_argument("--min-trade-notional", type=float, default=0.0)
    parser.add_argument("--min-state-volume-z-1d", type=float, default=-99.0)
    parser.add_argument("--max-state-vol-1d", type=float, default=0.0)
    parser.add_argument("--max-abs-state-ret-1d", type=float, default=0.0)
    parser.add_argument("--entry-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--idle-asset", choices=["cash", "spy"], default="cash")
    parser.add_argument("--starting-equity", type=float, default=50_000.0)
    parser.add_argument("--rolling", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
