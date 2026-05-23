"""Walk-forward SPY/cash regime overlay for the daily trading system.

The stock selector can correctly choose no-trade, but a deployable portfolio
still needs a market-exposure decision. This script tests simple causal
date-level rules that either hold SPY for the next horizon or hold cash.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


DATASET = "checkpoints/daily_ranker/exp11_latest_dataset_h5_2026/daily_ranker_dataset.parquet"


@dataclass(frozen=True)
class FoldSpec:
    name: str
    train_start: str
    test_start: str
    test_end: str


@dataclass(frozen=True)
class Config:
    dataset: str
    output: str
    horizon_days: int
    roundtrip_cost: float
    min_train_periods: int
    min_train_spy_exposure: float
    min_test_spy_exposure: float
    min_fold_active_alpha: float


def default_folds() -> list[FoldSpec]:
    return [
        FoldSpec("2023", "2016-01-01", "2023-01-01", "2024-01-01"),
        FoldSpec("2024", "2016-01-01", "2024-01-01", "2025-01-01"),
        FoldSpec("2025", "2016-01-01", "2025-01-01", "2026-01-01"),
        FoldSpec("2026_ytd", "2016-01-01", "2026-01-01", "2026-05-10"),
    ]


def load_daily_market(dataset: str) -> pd.DataFrame:
    cols = [
        "date",
        "symbol",
        "future_spy_return",
        "spy_ret_5d",
        "spy_ret_20d",
        "spy_ret_60d",
        "mkt_ret_5d_mean",
        "mkt_ret_20d_mean",
        "mkt_ret_60d_mean",
        "mkt_ret_20d_dispersion",
        "mkt_pct_positive_20d",
        "mkt_pct_above_ma20",
        "mkt_pct_drawdown_gt_10",
        "mkt_pct_low_vol",
    ]
    df = pd.read_parquet(dataset, columns=cols)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    rows = []
    for date, group in df.groupby("date", sort=True):
        spy = group[group["symbol"] == "SPY"]
        source = spy.iloc[0] if not spy.empty else group.iloc[0]
        row = {col: source[col] for col in cols if col not in {"symbol"}}
        row["universe_count"] = int(group["symbol"].nunique())
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    numeric = [c for c in out.columns if c != "date"]
    out[numeric] = out[numeric].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    return out


def candidate_rules() -> list[dict]:
    rules = []
    for (
        min_spy_ret_20d,
        min_spy_ret_60d,
        min_mkt_pct_positive_20d,
        min_mkt_pct_above_ma20,
        min_mkt_ret_20d_mean,
        max_mkt_ret_20d_dispersion,
        min_mkt_pct_drawdown_gt_10,
    ) in product(
        (None, -0.05, 0.0, 0.03),
        (None, -0.05, 0.0, 0.05),
        (None, 0.45, 0.50, 0.55, 0.60),
        (None, 0.45, 0.50, 0.55, 0.60),
        (None, -0.02, 0.0, 0.02),
        (None, 0.08, 0.12, 0.16),
        (None, 0.50, 0.60, 0.70),
    ):
        rules.append({
            "min_spy_ret_20d": min_spy_ret_20d,
            "min_spy_ret_60d": min_spy_ret_60d,
            "min_mkt_pct_positive_20d": min_mkt_pct_positive_20d,
            "min_mkt_pct_above_ma20": min_mkt_pct_above_ma20,
            "min_mkt_ret_20d_mean": min_mkt_ret_20d_mean,
            "max_mkt_ret_20d_dispersion": max_mkt_ret_20d_dispersion,
            "min_mkt_pct_drawdown_gt_10": min_mkt_pct_drawdown_gt_10,
        })
    rules.append({
        "min_spy_ret_20d": None,
        "min_spy_ret_60d": None,
        "min_mkt_pct_positive_20d": None,
        "min_mkt_pct_above_ma20": None,
        "min_mkt_ret_20d_mean": None,
        "max_mkt_ret_20d_dispersion": None,
        "min_mkt_pct_drawdown_gt_10": None,
    })
    return rules


def apply_rule(df: pd.DataFrame, rule: dict) -> pd.Series:
    active = pd.Series(True, index=df.index)
    checks = {
        "min_spy_ret_20d": ("spy_ret_20d", ">="),
        "min_spy_ret_60d": ("spy_ret_60d", ">="),
        "min_mkt_pct_positive_20d": ("mkt_pct_positive_20d", ">="),
        "min_mkt_pct_above_ma20": ("mkt_pct_above_ma20", ">="),
        "min_mkt_ret_20d_mean": ("mkt_ret_20d_mean", ">="),
        "max_mkt_ret_20d_dispersion": ("mkt_ret_20d_dispersion", "<="),
        "min_mkt_pct_drawdown_gt_10": ("mkt_pct_drawdown_gt_10", ">="),
    }
    for key, (col, op) in checks.items():
        val = rule.get(key)
        if val is None:
            continue
        if op == ">=":
            active &= df[col].astype(float) >= float(val)
        else:
            active &= df[col].astype(float) <= float(val)
    return active


def non_overlapping(df: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    selected = []
    next_date = pd.Timestamp.min.tz_localize("UTC")
    for _, row in df.sort_values("date").iterrows():
        date = pd.Timestamp(row["date"])
        if date < next_date:
            continue
        selected.append(row)
        next_date = date + pd.Timedelta(days=max(1, horizon_days))
    return pd.DataFrame(selected).reset_index(drop=True) if selected else df.iloc[:0].copy()


def simulate_periods(periods: pd.DataFrame, rule: dict, config: Config) -> dict:
    if periods.empty:
        return empty_result()
    in_spy = apply_rule(periods, rule).astype(float).to_numpy()
    spy = periods["future_spy_return"].astype(float).to_numpy()
    strategy = np.where(in_spy > 0, spy - config.roundtrip_cost, 0.0)
    eq = (1.0 + strategy).cumprod()
    spy_eq = (1.0 + spy).cumprod()
    curve = np.r_[1.0, eq]
    peaks = np.maximum.accumulate(curve)
    dd = (curve - peaks) / np.maximum(peaks, 1e-12)
    return {
        "periods": int(len(periods)),
        "spy_periods": int(in_spy.sum()),
        "spy_exposure": float(in_spy.mean()),
        "total_return": float(eq[-1] - 1.0),
        "buy_hold_spy_return": float(spy_eq[-1] - 1.0),
        "active_alpha_return": float((eq[-1] - 1.0) - (spy_eq[-1] - 1.0)),
        "win_rate": float((strategy > 0).mean()),
        "avoid_loss_rate": float(((in_spy == 0) & (spy < 0)).mean()),
        "miss_gain_rate": float(((in_spy == 0) & (spy > 0)).mean()),
        "mean_return": float(strategy.mean()),
        "max_drawdown": float(dd.min()),
    }


def simulate(df: pd.DataFrame, rule: dict, config: Config) -> dict:
    return simulate_periods(non_overlapping(df, config.horizon_days), rule, config)


def empty_result() -> dict:
    return {
        "periods": 0,
        "spy_periods": 0,
        "spy_exposure": 0.0,
        "total_return": 0.0,
        "buy_hold_spy_return": 0.0,
        "active_alpha_return": 0.0,
        "win_rate": 0.0,
        "avoid_loss_rate": 0.0,
        "miss_gain_rate": 0.0,
        "mean_return": 0.0,
        "max_drawdown": 0.0,
    }


def objective(result: dict) -> float:
    return (
        1.50 * result["active_alpha_return"]
        + 0.50 * result["total_return"]
        - 1.50 * abs(result["max_drawdown"])
        - 0.10 * max(0.0, 0.20 - result["spy_exposure"])
    )


def choose_rule(train: pd.DataFrame, config: Config) -> dict:
    best = None
    train_periods = non_overlapping(train, config.horizon_days)
    for rule in candidate_rules():
        result = simulate_periods(train_periods, rule, config)
        if result["periods"] < config.min_train_periods:
            continue
        if result["spy_exposure"] < config.min_train_spy_exposure:
            continue
        row = {"rule": rule, "train_result": result, "objective": objective(result)}
        if best is None or row["objective"] > best["objective"]:
            best = row
    if best is None:
        return {"rule": candidate_rules()[-1], "train_result": empty_result(), "objective": -float("inf")}
    return best


def run(config: Config) -> dict:
    daily = load_daily_market(config.dataset)
    fold_rows = []
    for fold in default_folds():
        dates = pd.to_datetime(daily["date"], utc=True)
        train = daily[(dates >= pd.Timestamp(fold.train_start, tz="UTC")) & (dates < pd.Timestamp(fold.test_start, tz="UTC"))].copy()
        test = daily[(dates >= pd.Timestamp(fold.test_start, tz="UTC")) & (dates < pd.Timestamp(fold.test_end, tz="UTC"))].copy()
        choice = choose_rule(train, config)
        test_result = simulate_periods(non_overlapping(test, config.horizon_days), choice["rule"], config)
        passed = (
            test_result["spy_exposure"] >= config.min_test_spy_exposure
            and test_result["active_alpha_return"] >= config.min_fold_active_alpha
        )
        fold_rows.append({
            "fold": asdict(fold),
            "rule": choice["rule"],
            "train_result": choice["train_result"],
            "train_objective": choice["objective"],
            "test_result": test_result,
            "passed": passed,
        })
    aggregate = {
        "passed_folds": sum(1 for row in fold_rows if row["passed"]),
        "total_active_alpha_sum": float(sum(row["test_result"]["active_alpha_return"] for row in fold_rows)),
        "total_return_sum": float(sum(row["test_result"]["total_return"] for row in fold_rows)),
        "buy_hold_spy_sum": float(sum(row["test_result"]["buy_hold_spy_return"] for row in fold_rows)),
        "worst_fold_alpha": float(min(row["test_result"]["active_alpha_return"] for row in fold_rows)),
        "worst_fold_drawdown": float(min(row["test_result"]["max_drawdown"] for row in fold_rows)),
    }
    payload = {
        "config": asdict(config),
        "rows": int(len(daily)),
        "folds": fold_rows,
        "aggregate": aggregate,
    }
    Path(config.output).parent.mkdir(parents=True, exist_ok=True)
    Path(config.output).write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps(payload, indent=2, default=str), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--output", default="checkpoints/daily_ranker/market_regime_overlay.json")
    parser.add_argument("--horizon-days", type=int, default=5)
    parser.add_argument("--roundtrip-cost", type=float, default=0.0002)
    parser.add_argument("--min-train-periods", type=int, default=120)
    parser.add_argument("--min-train-spy-exposure", type=float, default=0.15)
    parser.add_argument("--min-test-spy-exposure", type=float, default=0.05)
    parser.add_argument("--min-fold-active-alpha", type=float, default=0.0)
    run(Config(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
