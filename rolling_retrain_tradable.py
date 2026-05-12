"""Run full yearly rolling retrains for the tradable world-model policy.

Unlike rolling_yearly_tradable_eval.py, this script retrains the world model
and allocator inside each yearly fold before evaluating the next calendar year.
It is the main robustness path before paper trading.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import pandas as pd

from rolling_yearly_tradable_eval import _date_range, _year_dataset_args


def _run(cmd: list[str], dry_run: bool) -> dict:
    started = time.time()
    if dry_run:
        return {"cmd": cmd, "returncode": 0, "seconds": 0.0, "dry_run": True}
    proc = subprocess.run(cmd, check=False)
    return {
        "cmd": cmd,
        "returncode": int(proc.returncode),
        "seconds": float(time.time() - started),
        "dry_run": False,
    }


def _json_metric(path: Path, key: str, default: float = 0.0) -> float:
    if not path.exists():
        return default
    payload = json.loads(path.read_text())
    cur = payload
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    try:
        return float(cur)
    except (TypeError, ValueError):
        return default


def _fold_years(args: argparse.Namespace, source_dirs: list[Path]) -> range:
    first_start = min(_date_range(path)[0] for path in source_dirs)
    last_end = max(_date_range(path)[1] for path in source_dirs)
    start_year = args.start_year or int(first_start.year) + args.train_years
    end_year = args.end_year or int(last_end.year)
    return range(start_year, end_year + 1)


def _train_world_cmd(args: argparse.Namespace, train_dir: Path, world_ckpt: Path) -> list[str]:
    return [
        args.python,
        "train_world_model.py",
        "--data",
        str(train_dir),
        "--output",
        str(world_ckpt),
        "--epochs",
        str(args.world_epochs),
        "--batch-size",
        str(args.world_batch_size),
        "--hidden-dim",
        str(args.world_hidden_dim),
        "--n-layers",
        str(args.world_layers),
        "--dropout",
        str(args.world_dropout),
        "--lr",
        str(args.world_lr),
        "--weight-decay",
        str(args.world_weight_decay),
        "--val-gap-days",
        str(args.val_gap_days),
        "--max-horizon-bars",
        str(args.max_horizon_bars),
        "--rank-loss-coef",
        str(args.rank_loss_coef),
        "--device",
        args.device,
    ] + (["--limit-rows", str(args.limit_rows)] if args.limit_rows > 0 else [])


def _train_allocator_cmd(args: argparse.Namespace, train_dir: Path, test_dir: Path, world_ckpt: Path, allocator_ckpt: Path) -> list[str]:
    return [
        args.python,
        "train_allocator.py",
        "--train-data",
        str(train_dir),
        "--test-data",
        str(test_dir),
        "--world-checkpoint",
        str(world_ckpt),
        "--output",
        str(allocator_ckpt),
        "--epochs",
        str(args.allocator_epochs),
        "--batch-size",
        str(args.allocator_batch_size),
        "--hidden-dim",
        str(args.allocator_hidden_dim),
        "--n-layers",
        str(args.allocator_layers),
        "--dropout",
        str(args.allocator_dropout),
        "--lr",
        str(args.allocator_lr),
        "--weight-decay",
        str(args.allocator_weight_decay),
        "--val-gap-days",
        str(args.val_gap_days),
        "--max-horizon-bars",
        str(args.max_horizon_bars),
        "--top-quantile",
        str(args.top_quantile),
        "--feature-mode",
        args.feature_mode,
        "--utility-mode",
        args.utility_mode,
        "--extra-roundtrip-bps",
        str(args.extra_roundtrip_bps),
        "--drawdown-penalty",
        str(args.drawdown_penalty),
        "--volatility-penalty",
        str(args.volatility_penalty),
        "--device",
        args.device,
    ] + (["--limit-rows", str(args.limit_rows)] if args.limit_rows > 0 else [])


def _eval_cmd(args: argparse.Namespace, train_dir: Path, test_dir: Path, world_ckpt: Path, allocator_ckpt: Path, output: Path) -> list[str]:
    return [
        args.python,
        "evaluate_tradable_allocator.py",
        "--calibration-data",
        str(train_dir),
        "--test-data",
        str(test_dir),
        "--world-checkpoint",
        str(world_ckpt),
        "--allocator-checkpoint",
        str(allocator_ckpt),
        "--output",
        str(output),
        "--batch-size",
        str(args.eval_batch_size),
        "--max-horizon-bars",
        str(args.max_horizon_bars),
        "--min-coverage",
        str(args.min_coverage),
        "--objective-mode",
        args.objective_mode,
        "--rule-mode",
        args.rule_mode,
        "--max-calibration-drawdown",
        str(args.max_calibration_drawdown),
        "--extra-roundtrip-bps",
        str(args.extra_roundtrip_bps),
        "--max-trades-per-symbol",
        str(args.max_trades_per_symbol),
        "--symbol-cooldown-days",
        str(args.symbol_cooldown_days),
        "--min-price",
        str(args.min_price),
        "--min-trade-notional",
        str(args.min_trade_notional),
        "--min-state-volume-z-1d",
        str(args.min_state_volume_z_1d),
        "--max-state-vol-1d",
        str(args.max_state_vol_1d),
        "--max-abs-state-ret-1d",
        str(args.max_abs_state_ret_1d),
        "--entry-only",
        "--idle-asset",
        args.idle_asset,
        "--device",
        args.device,
    ] + (["--limit-rows", str(args.limit_rows)] if args.limit_rows > 0 else [])


def run(args: argparse.Namespace) -> dict:
    source_dirs = [Path(p) for p in args.source_data]
    work_dir = Path(args.work_dir)
    output_dir = Path(args.output_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    folds = []

    for year in _fold_years(args, source_dirs):
        fold_dir = work_dir / f"fold_{year}"
        train_dir = fold_dir / "train_data"
        test_dir = fold_dir / "test_data"
        world_ckpt = output_dir / f"world_model_fold_{year}.pt"
        allocator_ckpt = output_dir / f"allocator_fold_{year}.pt"
        eval_output = output_dir / f"tradable_fold_{year}.json"
        train_start = f"{year - args.train_years}-01-01"
        train_end = f"{year}-01-01"
        test_start = f"{year}-01-01"
        test_end = f"{year + 1}-01-01"
        fold_dir.mkdir(parents=True, exist_ok=True)
        if args.rebuild_slices or not train_dir.exists():
            _year_dataset_args(source_dirs, train_dir, train_start, train_end)
        if args.rebuild_slices or not test_dir.exists():
            _year_dataset_args(source_dirs, test_dir, test_start, test_end)

        commands = []
        if args.force or not world_ckpt.exists():
            commands.append(_run(_train_world_cmd(args, train_dir, world_ckpt), args.dry_run))
        if args.force or not allocator_ckpt.exists():
            commands.append(_run(_train_allocator_cmd(args, train_dir, test_dir, world_ckpt, allocator_ckpt), args.dry_run))
        if args.force or not eval_output.exists():
            commands.append(_run(_eval_cmd(args, train_dir, test_dir, world_ckpt, allocator_ckpt, eval_output), args.dry_run))
        failed = [cmd for cmd in commands if cmd["returncode"] != 0]
        fold = {
            "year": int(year),
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "train_data": str(train_dir),
            "test_data": str(test_dir),
            "world_checkpoint": str(world_ckpt),
            "allocator_checkpoint": str(allocator_ckpt),
            "evaluation": str(eval_output),
            "commands": commands,
            "ok": not failed,
        }
        if eval_output.exists():
            fold.update({
                "total_return": _json_metric(eval_output, "sequential_portfolio.total_return"),
                "max_drawdown": _json_metric(eval_output, "sequential_portfolio.max_drawdown"),
                "trades": int(_json_metric(eval_output, "sequential_portfolio.trades")),
                "profit_rate": _json_metric(eval_output, "sequential_portfolio.profit_rate"),
                "beat_spy_rate": _json_metric(eval_output, "sequential_portfolio.beat_spy_rate"),
            })
        folds.append(fold)
        if failed:
            break

    valid = [fold for fold in folds if fold.get("ok") and "total_return" in fold]
    returns = pd.Series([fold["total_return"] for fold in valid], dtype=float)
    summary = {
        "folds": folds,
        "valid_years": int(len(valid)),
        "mean_return": float(returns.mean()) if len(returns) else 0.0,
        "median_return": float(returns.median()) if len(returns) else 0.0,
        "positive_year_rate": float((returns > 0).mean()) if len(returns) else 0.0,
        "config": vars(args),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", nargs="+", default=["data/world_model/locked5y_train_intraday120", "data/world_model/locked1y_test_intraday120"])
    parser.add_argument("--output", default="checkpoints/world_model/rolling_retrain_tradable_summary.json")
    parser.add_argument("--output-dir", default="checkpoints/world_model/rolling_retrain")
    parser.add_argument("--work-dir", default="data/world_model/rolling_retrain")
    parser.add_argument("--start-year", type=int, default=0)
    parser.add_argument("--end-year", type=int, default=0)
    parser.add_argument("--train-years", type=int, default=3)
    parser.add_argument("--python", default=".venv/bin/python")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--rebuild-slices", action="store_true")
    parser.add_argument("--limit-rows", type=int, default=0)
    parser.add_argument("--max-horizon-bars", type=int, default=120)
    parser.add_argument("--val-gap-days", type=float, default=14.0)
    parser.add_argument("--world-epochs", type=int, default=4)
    parser.add_argument("--world-batch-size", type=int, default=32768)
    parser.add_argument("--world-hidden-dim", type=int, default=256)
    parser.add_argument("--world-layers", type=int, default=4)
    parser.add_argument("--world-dropout", type=float, default=0.10)
    parser.add_argument("--world-lr", type=float, default=3e-4)
    parser.add_argument("--world-weight-decay", type=float, default=1e-4)
    parser.add_argument("--rank-loss-coef", type=float, default=0.50)
    parser.add_argument("--allocator-epochs", type=int, default=4)
    parser.add_argument("--allocator-batch-size", type=int, default=32768)
    parser.add_argument("--allocator-hidden-dim", type=int, default=192)
    parser.add_argument("--allocator-layers", type=int, default=3)
    parser.add_argument("--allocator-dropout", type=float, default=0.25)
    parser.add_argument("--allocator-lr", type=float, default=1e-4)
    parser.add_argument("--allocator-weight-decay", type=float, default=1e-3)
    parser.add_argument("--top-quantile", type=float, default=0.80)
    parser.add_argument("--feature-mode", choices=["compact", "market"], default="compact")
    parser.add_argument("--utility-mode", choices=["default", "stress_adjusted", "stress_convex", "tradable_stress"], default="stress_adjusted")
    parser.add_argument("--drawdown-penalty", type=float, default=0.50)
    parser.add_argument("--volatility-penalty", type=float, default=0.25)
    parser.add_argument("--eval-batch-size", type=int, default=32768)
    parser.add_argument("--min-coverage", type=float, default=0.05)
    parser.add_argument("--objective-mode", choices=["cash_return", "active_return", "hybrid"], default="hybrid")
    parser.add_argument("--rule-mode", choices=["fixed_threshold", "calibrated"], default="fixed_threshold")
    parser.add_argument("--max-calibration-drawdown", type=float, default=0.18)
    parser.add_argument("--extra-roundtrip-bps", type=float, default=10.0)
    parser.add_argument("--max-trades-per-symbol", type=int, default=3)
    parser.add_argument("--symbol-cooldown-days", type=float, default=10.0)
    parser.add_argument("--min-price", type=float, default=5.0)
    parser.add_argument("--min-trade-notional", type=float, default=1000.0)
    parser.add_argument("--min-state-volume-z-1d", type=float, default=-3.0)
    parser.add_argument("--max-state-vol-1d", type=float, default=0.08)
    parser.add_argument("--max-abs-state-ret-1d", type=float, default=0.20)
    parser.add_argument("--idle-asset", choices=["cash", "spy"], default="spy")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
