"""Calibrate a 15-minute transformer ensemble before testing 2026.

This script focuses on the first promising 15-minute result:
``temporal_fusion + perceiver``.

Protocol:

1. Train both models on pre-calibration history.
2. Walk through the calibration period, predicting first and patch-training
   after each realized 15-minute interval.
3. Sweep ensemble trading gates only on calibration predictions.
4. Continue the same patched models into the test period.
5. Evaluate the frozen calibrated gate on the untouched test period.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_15m_transformer_ensemble import (
    Config as BaseConfig,
    evaluate_online_model,
    load_dataset,
    make_arrays,
    make_model,
    max_drawdown,
    pick_device,
    plot_results,
    train_initial,
    valid_row_indices,
)


@dataclass(frozen=True)
class Config:
    dataset: str
    output_dir: str
    train_start: str
    train_end: str
    calibration_start: str
    calibration_end: str
    test_start: str
    test_end: str
    initial_epochs: int
    patch_epochs: int
    max_train_samples: int
    batch_size: int
    hidden_dim: int
    layers: int
    heads: int
    dropout: float
    lr: float
    weight_decay: float
    seq_len: int
    max_positions: int
    roundtrip_cost: float
    min_calibration_trades: int
    min_calibration_return: float
    min_calibration_alpha: float
    max_calibration_drawdown: float
    device: str
    seed: int


def base_config(config: Config, min_profit: float = 0.0, max_crash: float = 1.0, min_buy: float = 0.0) -> BaseConfig:
    return BaseConfig(
        dataset=config.dataset,
        output_dir=config.output_dir,
        build_dataset=False,
        start_date="2020-11-03",
        end_date="2026-05-16",
        symbol_limit=40,
        universe_mode="alphabetical",
        universe_rank_cache="checkpoints/transformer_15m/top_volume_valuation_universe.csv",
        min_rows=3000,
        interval_minutes=15,
        seq_len=config.seq_len,
        warmup_days=365,
        train_start=config.train_start,
        train_end=config.train_end,
        eval_start=config.test_start,
        eval_end=config.test_end,
        initial_epochs=config.initial_epochs,
        patch_epochs=config.patch_epochs,
        batch_size=config.batch_size,
        hidden_dim=config.hidden_dim,
        layers=config.layers,
        heads=config.heads,
        dropout=config.dropout,
        lr=config.lr,
        weight_decay=config.weight_decay,
        max_train_samples=config.max_train_samples,
        max_eval_intervals=0,
        min_close=5.0,
        max_abs_return=0.08,
        roundtrip_cost=config.roundtrip_cost,
        max_positions=config.max_positions,
        min_pred_profit=min_profit,
        max_pred_crash=max_crash,
        min_buy_prob=min_buy,
        min_pred_score=-1e9,
        min_pred_utility=-1e9,
        portfolio_exposure=1.0,
        daily_loss_stop=-1.0,
        symbol_cooldown_loss=-1.0,
        symbol_cooldown_intervals=0,
        symbol_daily_cap=0,
        spy_momentum_window=0,
        spy_momentum_min_return=0.0,
        strategy_momentum_window=0,
        strategy_momentum_min_return=0.0,
        filter_falling_stocks=False,
        falling_filter_min_signals=3,
        falling_max_ret_4=0.0,
        falling_max_ret_16=0.0,
        falling_max_trend_slope_8=0.0,
        falling_max_trend_slope_16=0.0,
        falling_max_ma8_dist=0.0,
        falling_max_ma26_dist=0.0,
        falling_max_algo_momentum_vote=0.0,
        models="temporal_fusion,perceiver",
        max_ensemble_size=2,
        device=config.device,
        seed=config.seed,
    )


def rows_by_timestamp(df: pd.DataFrame, row_indices: np.ndarray, start: str, end: str) -> dict[pd.Timestamp, np.ndarray]:
    ts = pd.to_datetime(df.loc[row_indices, "timestamp"], utc=True)
    mask = ts.ge(pd.Timestamp(start, tz="UTC")) & ts.lt(pd.Timestamp(end, tz="UTC"))
    rows = row_indices[mask.to_numpy()]
    return {
        pd.Timestamp(t): group.index.to_numpy(np.int64)
        for t, group in df.loc[rows].groupby("timestamp", sort=True)
    }


def evaluate_with_gate(
    merged: pd.DataFrame,
    config: Config,
    min_profit: float,
    max_crash: float,
    min_buy: float,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    active = merged[
        (merged["pred_profit"] >= min_profit)
        & (merged["pred_crash"] <= max_crash)
        & (merged["prob_buy"] >= min_buy)
    ].copy()
    if not active.empty:
        active["pred_score"] = (
            active["pred_utility"]
            + 0.05 * active["pred_profit"]
            + 0.10 * active["prob_buy"]
            - 0.35 * active["pred_crash"]
            - 0.05 * active["prob_sell"]
        )
    equity = 50_000.0
    spy_equity = 50_000.0
    curve_rows = []
    trade_rows = []
    active_by_ts = {ts: group for ts, group in active.groupby("timestamp", sort=False)} if not active.empty else {}
    for ts, group in merged.groupby("timestamp", sort=True):
        spy_ret = float(group["future_spy_return"].dropna().iloc[0]) if not group.empty else 0.0
        candidates = active_by_ts.get(ts)
        if candidates is None or candidates.empty:
            selected = candidates
            portfolio_ret = 0.0
            symbols: list[str] = []
        else:
            selected = candidates.sort_values("pred_score", ascending=False).head(config.max_positions)
            portfolio_ret = float(selected["future_return"].astype(float).mean() - config.roundtrip_cost)
            symbols = selected["symbol"].astype(str).tolist()
            for _, row in selected.iterrows():
                trade_rows.append({
                    "timestamp": ts,
                    "model": "temporal_fusion+perceiver",
                    "symbol": row["symbol"],
                    "pred_score": float(row["pred_score"]),
                    "future_return": float(row["future_return"]),
                    "future_spy_return": spy_ret,
                    "future_alpha": float(row["future_return"]) - spy_ret,
                })
        equity *= 1.0 + portfolio_ret
        spy_equity *= 1.0 + spy_ret
        curve_rows.append({
            "timestamp": ts,
            "model": "temporal_fusion+perceiver",
            "equity": equity,
            "spy_equity": spy_equity,
            "portfolio_return": portfolio_ret,
            "spy_return": spy_ret,
            "symbols": ",".join(symbols),
        })
    curve = pd.DataFrame(curve_rows)
    trades = pd.DataFrame(trade_rows)
    trade_returns = trades["future_return"].astype(float).to_numpy() if not trades.empty else np.array([])
    summary = {
        "model": "temporal_fusion+perceiver",
        "decision_intervals": int(len(curve)),
        "active_intervals": int((curve["symbols"].astype(str) != "").sum()) if not curve.empty else 0,
        "trades": int(len(trades)),
        "final_equity": float(equity),
        "spy_final_equity": float(spy_equity),
        "total_return": float(equity / 50_000.0 - 1.0),
        "spy_total_return": float(spy_equity / 50_000.0 - 1.0),
        "active_alpha_return": float(equity / 50_000.0 - spy_equity / 50_000.0),
        "max_drawdown": max_drawdown(curve["equity"].astype(float).tolist()) if not curve.empty else 0.0,
        "trade_profit_rate": float((trade_returns > 0).mean()) if len(trade_returns) else 0.0,
    }
    return summary, curve, trades


def merge_predictions(predictions: dict[str, pd.DataFrame], df: pd.DataFrame) -> pd.DataFrame:
    cols = ["timestamp", "row", "pred_utility", "pred_profit", "pred_crash", "prob_sell", "prob_hold", "prob_buy"]
    merged = predictions["temporal_fusion"][cols].merge(
        predictions["perceiver"][cols],
        on=["timestamp", "row"],
        suffixes=("_tf", "_pv"),
    )
    for col in ("pred_utility", "pred_profit", "pred_crash", "prob_sell", "prob_hold", "prob_buy"):
        merged[col] = 0.5 * (merged[f"{col}_tf"] + merged[f"{col}_pv"])
    meta = df[["symbol", "future_return", "future_spy_return"]].reset_index(names="row")
    return merged[["timestamp", "row", "pred_utility", "pred_profit", "pred_crash", "prob_sell", "prob_hold", "prob_buy"]].merge(
        meta, on="row", how="left"
    )


def calibrate(merged: pd.DataFrame, config: Config) -> dict:
    rows = []
    for min_profit in (0.48, 0.50, 0.52, 0.54, 0.56, 0.58):
        for max_crash in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
            for min_buy in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45):
                summary, _, _ = evaluate_with_gate(merged, config, min_profit, max_crash, min_buy)
                if not summary:
                    continue
                row = {
                    "min_pred_profit": min_profit,
                    "max_pred_crash": max_crash,
                    "min_buy_prob": min_buy,
                    **summary,
                }
                row["passes"] = (
                    row["trades"] >= config.min_calibration_trades
                    and row["total_return"] >= config.min_calibration_return
                    and row["active_alpha_return"] >= config.min_calibration_alpha
                    and row["max_drawdown"] >= -abs(config.max_calibration_drawdown)
                )
                rows.append(row)
    board = pd.DataFrame(rows)
    if board.empty:
        raise RuntimeError("calibration produced no rules")
    passed = board[board["passes"]].copy()
    candidates = passed if not passed.empty else board.copy()
    candidates["objective"] = (
        candidates["active_alpha_return"]
        + 0.35 * candidates["total_return"]
        + 0.15 * candidates["trade_profit_rate"]
        + 0.10 * candidates["max_drawdown"]
    )
    best = candidates.sort_values(["objective", "active_alpha_return", "total_return"], ascending=False).iloc[0].to_dict()
    best["passed_hard_filters"] = bool(best.get("passes", False))
    return {"best": best, "board": board}


def plot_two_curves(cal_curve: pd.DataFrame, test_curve: pd.DataFrame, output: Path) -> None:
    curves = pd.concat([
        cal_curve.assign(model="calibration_2025"),
        test_curve.assign(model="test_2026"),
    ], ignore_index=True)
    plot_results(curves, output, "Calibrated temporal_fusion+perceiver: 2025 calibration and 2026 test")


def run(config: Config) -> dict:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(config.device)
    bcfg = base_config(config)
    df = load_dataset(bcfg)
    row_indices = valid_row_indices(df, config.seq_len)
    ts = pd.to_datetime(df.loc[row_indices, "timestamp"], utc=True)
    train_mask = ts.ge(pd.Timestamp(config.train_start, tz="UTC")) & ts.lt(pd.Timestamp(config.train_end, tz="UTC"))
    train_rows = row_indices[train_mask.to_numpy()]
    if len(train_rows) == 0:
        raise RuntimeError("empty train rows")
    arrays = make_arrays(df, train_rows)
    cal_groups = rows_by_timestamp(df, row_indices, config.calibration_start, config.calibration_end)
    test_groups = rows_by_timestamp(df, row_indices, config.test_start, config.test_end)
    print(
        f"[cal15m] train_rows={len(train_rows):,} calibration_intervals={len(cal_groups):,} "
        f"test_intervals={len(test_groups):,} device={device}",
        flush=True,
    )

    predictions_cal: dict[str, pd.DataFrame] = {}
    predictions_test: dict[str, pd.DataFrame] = {}
    model_summaries = {}
    for name in ("temporal_fusion", "perceiver"):
        print(f"[cal15m] training {name}", flush=True)
        model = make_model(name, len(arrays["mean"]), int(df["symbol_id"].max()) + 1, bcfg).to(device)
        history = train_initial(model, arrays, train_rows, bcfg, device)
        loose = replace(bcfg, min_pred_profit=0.0, max_pred_crash=1.0, min_buy_prob=0.0)
        cal_summary, cal_curve, cal_trades, cal_pred = evaluate_online_model(
            name, model, df, arrays, cal_groups, loose, device
        )
        test_summary, test_curve, test_trades, test_pred = evaluate_online_model(
            name, model, df, arrays, test_groups, loose, device
        )
        predictions_cal[name] = cal_pred
        predictions_test[name] = test_pred
        model_summaries[name] = {
            "history": history,
            "loose_calibration_summary": cal_summary,
            "loose_test_summary": test_summary,
        }
        torch.save(
            {
                "model": name,
                "state_dict": model.state_dict(),
                "config": asdict(config),
                "train_history": history,
            },
            output_dir / f"{name}_patched_through_2026.pt",
        )

    merged_cal = merge_predictions(predictions_cal, df)
    merged_test = merge_predictions(predictions_test, df)
    merged_cal.to_parquet(output_dir / "merged_calibration_predictions.parquet", index=False)
    merged_test.to_parquet(output_dir / "merged_test_predictions.parquet", index=False)
    calibration = calibrate(merged_cal, config)
    best = calibration["best"]
    print(f"[cal15m] selected rule={best}", flush=True)
    calibration["board"].sort_values("objective" if "objective" in calibration["board"].columns else "active_alpha_return", ascending=False).to_csv(
        output_dir / "calibration_grid.csv", index=False
    )
    cal_summary, cal_curve, cal_trades = evaluate_with_gate(
        merged_cal, config,
        float(best["min_pred_profit"]),
        float(best["max_pred_crash"]),
        float(best["min_buy_prob"]),
    )
    test_summary, test_curve, test_trades = evaluate_with_gate(
        merged_test, config,
        float(best["min_pred_profit"]),
        float(best["max_pred_crash"]),
        float(best["min_buy_prob"]),
    )
    cal_curve.to_csv(output_dir / "calibrated_2025_curve.csv", index=False)
    test_curve.to_csv(output_dir / "calibrated_2026_curve.csv", index=False)
    cal_trades.to_csv(output_dir / "calibrated_2025_trades.csv", index=False)
    test_trades.to_csv(output_dir / "calibrated_2026_trades.csv", index=False)
    plot_two_curves(cal_curve, test_curve, output_dir / "calibrated_2025_2026_equity.png")
    plot_two_curves(cal_curve, test_curve, Path("docs/transformer_15m_calibrated_2025_2026_equity.png"))
    result = {
        "config": asdict(config),
        "selected_rule": {
            "min_pred_profit": float(best["min_pred_profit"]),
            "max_pred_crash": float(best["max_pred_crash"]),
            "min_buy_prob": float(best["min_buy_prob"]),
            "passed_hard_filters": bool(best["passed_hard_filters"]),
        },
        "model_summaries": model_summaries,
        "calibration_summary": cal_summary,
        "test_summary": test_summary,
        "warning": "research_only_calibrated_15m_transformer_ensemble",
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="checkpoints/transformer_15m/shared_15m_40sym.parquet")
    parser.add_argument("--output-dir", default="checkpoints/transformer_15m/exp3_calibrated_tf_perceiver_2025_2026")
    parser.add_argument("--train-start", default="2021-01-01")
    parser.add_argument("--train-end", default="2025-01-01")
    parser.add_argument("--calibration-start", default="2025-01-01")
    parser.add_argument("--calibration-end", default="2026-01-01")
    parser.add_argument("--test-start", default="2026-01-01")
    parser.add_argument("--test-end", default="2026-05-16")
    parser.add_argument("--initial-epochs", type=int, default=16)
    parser.add_argument("--patch-epochs", type=int, default=1)
    parser.add_argument("--max-train-samples", type=int, default=250_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--max-positions", type=int, default=3)
    parser.add_argument("--roundtrip-cost", type=float, default=0.0008)
    parser.add_argument("--min-calibration-trades", type=int, default=80)
    parser.add_argument("--min-calibration-return", type=float, default=0.0)
    parser.add_argument("--min-calibration-alpha", type=float, default=0.0)
    parser.add_argument("--max-calibration-drawdown", type=float, default=0.20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    run(Config(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
