"""Run yearly sequential tradable evaluations from existing world-model datasets.

This is a fast robustness harness around evaluate_tradable_allocator.py. It
does not retrain the world model per year; use it to find calendar-year
fragility before spending GPU time on full rolling retrains.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pyarrow.parquet as pq

from evaluate_tradable_allocator import run as run_tradable_eval


def _date_range(data_dir: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []
    for path in data_dir.glob("*.parquet"):
        columns = set(pq.read_schema(path).names)
        wanted = ["decision_timestamp"]
        if "exit_timestamp" in columns:
            wanted.append("exit_timestamp")
        elif "horizon_bars" in columns:
            wanted.append("horizon_bars")
        frame = pd.read_parquet(path, columns=wanted)
        if frame.empty:
            continue
        decision_ts = pd.to_datetime(frame["decision_timestamp"], utc=True)
        starts.append(decision_ts.min())
        if "exit_timestamp" in frame.columns:
            ends.append(pd.to_datetime(frame["exit_timestamp"], utc=True).max())
        else:
            ends.append((decision_ts + pd.to_timedelta(frame["horizon_bars"].astype(float), unit="m")).max())
    if not starts or not ends:
        raise RuntimeError(f"no timestamped rows found in {data_dir}")
    return min(starts), max(ends)


def _year_dataset_args(source_dirs: list[Path], output_dir: Path, start: str, end: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    written = 0
    rows = 0
    by_symbol: dict[str, list[pd.DataFrame]] = {}
    for source_dir in source_dirs:
        for path in source_dir.glob("*.parquet"):
            frame = pd.read_parquet(path)
            if frame.empty:
                continue
            decision_ts = pd.to_datetime(frame["decision_timestamp"], utc=True)
            if "exit_timestamp" in frame.columns:
                exit_ts = pd.to_datetime(frame["exit_timestamp"], utc=True)
            else:
                exit_ts = decision_ts + pd.to_timedelta(frame["horizon_bars"].astype(float), unit="m")
            sliced = frame[(decision_ts >= start_ts) & (exit_ts < end_ts)].copy()
            if sliced.empty:
                continue
            sliced["exit_timestamp"] = exit_ts.loc[sliced.index].to_numpy()
            by_symbol.setdefault(path.name, []).append(sliced)
    for name, frames in by_symbol.items():
        combined = pd.concat(frames, ignore_index=True).drop_duplicates()
        combined.to_parquet(output_dir / name, index=False)
        written += 1
        rows += len(combined)
    if written == 0:
        raise RuntimeError(f"no rows for {start}..{end} in {source_dirs}")
    metadata = {"source": [str(p) for p in source_dirs], "start": start, "end": end, "symbols": written, "rows": rows}
    (output_dir / "_slice.json").write_text(json.dumps(metadata, indent=2, default=str))


def _make_args(base: argparse.Namespace, calibration_data: Path, test_data: Path, output: Path) -> SimpleNamespace:
    return SimpleNamespace(
        calibration_data=str(calibration_data),
        test_data=str(test_data),
        world_checkpoint=base.world_checkpoint,
        allocator_checkpoint=base.allocator_checkpoint,
        ensemble_allocator_checkpoints=base.ensemble_allocator_checkpoints,
        output=str(output),
        batch_size=base.batch_size,
        limit_rows=base.limit_rows,
        min_horizon_bars=base.min_horizon_bars,
        max_horizon_bars=base.max_horizon_bars,
        min_coverage=base.min_coverage,
        objective_mode=base.objective_mode,
        rule_mode=base.rule_mode,
        max_calibration_drawdown=base.max_calibration_drawdown,
        extra_roundtrip_bps=base.extra_roundtrip_bps,
        extra_fee_usd=base.extra_fee_usd,
        max_trades_per_symbol=base.max_trades_per_symbol,
        symbol_cooldown_days=base.symbol_cooldown_days,
        min_price=base.min_price,
        min_trade_notional=base.min_trade_notional,
        min_state_volume_z_1d=base.min_state_volume_z_1d,
        max_state_vol_1d=base.max_state_vol_1d,
        max_abs_state_ret_1d=base.max_abs_state_ret_1d,
        entry_only=base.entry_only,
        idle_asset=base.idle_asset,
        starting_equity=base.starting_equity,
        seed=base.seed,
        device=base.device,
    )


def run(args: argparse.Namespace) -> dict:
    source_dirs = [Path(p) for p in args.source_data]
    first_start = min(_date_range(path)[0] for path in source_dirs)
    last_end = max(_date_range(path)[1] for path in source_dirs)
    start_year = args.start_year or int(first_start.year) + 1
    end_year = args.end_year or int(last_end.year)
    work_dir = Path(args.work_dir)
    output_dir = Path(args.output_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_results = []
    for year in range(start_year, end_year + 1):
        test_start = f"{year}-01-01"
        test_end = f"{year + 1}-01-01"
        calibration_start = f"{year - args.train_years}-01-01"
        calibration_end = test_start
        calibration_dir = work_dir / f"calibration_{calibration_start}_{calibration_end}"
        test_dir = work_dir / f"test_{test_start}_{test_end}"
        if not calibration_dir.exists():
            _year_dataset_args(source_dirs, calibration_dir, calibration_start, calibration_end)
        if not test_dir.exists():
            _year_dataset_args(source_dirs, test_dir, test_start, test_end)
        fold_output = output_dir / f"tradable_year_{year}.json"
        try:
            payload = run_tradable_eval(_make_args(args, calibration_dir, test_dir, fold_output))
            seq = payload["sequential_portfolio"]
            fold_results.append({
                "year": year,
                "output": str(fold_output),
                "total_return": float(seq["total_return"]),
                "max_drawdown": float(seq["max_drawdown"]),
                "trades": int(seq["trades"]),
                "profit_rate": float(seq["profit_rate"]),
                "beat_spy_rate": float(seq["beat_spy_rate"]),
            })
        except Exception as exc:
            fold_results.append({"year": year, "error": str(exc)})

    valid = [row for row in fold_results if "error" not in row]
    summary = {
        "folds": fold_results,
        "valid_years": int(len(valid)),
        "mean_return": float(pd.Series([r["total_return"] for r in valid]).mean()) if valid else 0.0,
        "median_return": float(pd.Series([r["total_return"] for r in valid]).median()) if valid else 0.0,
        "positive_year_rate": float(pd.Series([r["total_return"] > 0 for r in valid]).mean()) if valid else 0.0,
        "mean_max_drawdown": float(pd.Series([r["max_drawdown"] for r in valid]).mean()) if valid else 0.0,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", nargs="+", default=["data/world_model/locked5y_train_intraday120", "data/world_model/locked1y_test_intraday120"])
    parser.add_argument("--world-checkpoint", required=True)
    parser.add_argument("--allocator-checkpoint", required=True)
    parser.add_argument("--ensemble-allocator-checkpoints", nargs="*", default=[])
    parser.add_argument("--output", default="checkpoints/world_model/rolling_yearly_tradable_summary.json")
    parser.add_argument("--output-dir", default="checkpoints/world_model/rolling_yearly")
    parser.add_argument("--work-dir", default="data/world_model/yearly_slices")
    parser.add_argument("--start-year", type=int, default=0)
    parser.add_argument("--end-year", type=int, default=0)
    parser.add_argument("--train-years", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--limit-rows", type=int, default=0)
    parser.add_argument("--min-horizon-bars", type=int, default=0)
    parser.add_argument("--max-horizon-bars", type=int, default=120)
    parser.add_argument("--min-coverage", type=float, default=0.05)
    parser.add_argument("--objective-mode", choices=["cash_return", "active_return", "hybrid"], default="hybrid")
    parser.add_argument("--rule-mode", choices=["fixed_threshold", "calibrated"], default="calibrated")
    parser.add_argument("--max-calibration-drawdown", type=float, default=0.18)
    parser.add_argument("--extra-roundtrip-bps", type=float, default=10.0)
    parser.add_argument("--extra-fee-usd", type=float, default=0.0)
    parser.add_argument("--max-trades-per-symbol", type=int, default=3)
    parser.add_argument("--symbol-cooldown-days", type=float, default=10.0)
    parser.add_argument("--min-price", type=float, default=5.0)
    parser.add_argument("--min-trade-notional", type=float, default=1000.0)
    parser.add_argument("--min-state-volume-z-1d", type=float, default=-3.0)
    parser.add_argument("--max-state-vol-1d", type=float, default=0.08)
    parser.add_argument("--max-abs-state-ret-1d", type=float, default=0.20)
    parser.add_argument("--entry-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--idle-asset", choices=["cash", "spy"], default="spy")
    parser.add_argument("--starting-equity", type=float, default=50_000.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
