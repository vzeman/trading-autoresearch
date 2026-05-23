"""Validate daily listwise stop-loss probes against cached 1-minute bars."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_listwise_drawdown_stop import default_folds, score_period, selected_trades, simulate_overlay, Overlay
from prepare import CACHE_DIR
from train_daily_ranker import pick_device


def add_future_dates(df: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], utc=True)
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
    out["future_date"] = out.groupby("symbol")["date"].shift(-horizon_days)
    return out


def _minute_bars(symbol: str) -> pd.DataFrame:
    path = CACHE_DIR / f"{symbol}_1m.parquet"
    if not path.exists():
        return pd.DataFrame()
    bars = pd.read_parquet(path, columns=["timestamp", "open", "high", "low", "close"])
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    bars["local_date"] = bars["timestamp"].dt.tz_convert("America/New_York").dt.date
    return bars.sort_values("timestamp").reset_index(drop=True)


def intraday_trade_return(row: pd.Series, stop_loss: float, slippage_bps: float) -> tuple[float, bool, str]:
    symbol = str(row["symbol"])
    bars = _minute_bars(symbol)
    if bars.empty or pd.isna(row.get("future_date")):
        return float(row["future_return"]), False, "missing_bars"
    entry_date = pd.Timestamp(row["date"]).tz_convert("America/New_York").date()
    exit_date = pd.Timestamp(row["future_date"]).tz_convert("America/New_York").date()
    window = bars[(bars["local_date"] > entry_date) & (bars["local_date"] <= exit_date)].copy()
    if window.empty:
        return float(row["future_return"]), False, "empty_window"
    entry = float(row["close"])
    if not np.isfinite(entry) or entry <= 0:
        return float(row["future_return"]), False, "bad_entry"
    stop_price = entry * (1.0 - stop_loss)
    slip = max(0.0, slippage_bps) * 1e-4
    for bar in window.itertuples(index=False):
        open_px = float(bar.open)
        low_px = float(bar.low)
        if open_px <= stop_price:
            return open_px / entry - 1.0 - slip, True, "gap_stop"
        if low_px <= stop_price:
            return -stop_loss - slip, True, "stop"
    exit_px = float(window.iloc[-1]["close"])
    return exit_px / entry - 1.0, False, "target_close"


def intraday_periods(trades: pd.DataFrame, stop_loss: float, roundtrip_cost: float, slippage_bps: float) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["date", "return_", "spy_return", "stopped_positions", "positions", "gap_stops"])
    work = trades.copy()
    returns = []
    stopped = []
    gap_stops = []
    reasons = []
    for _, row in work.iterrows():
        ret, did_stop, reason = intraday_trade_return(row, stop_loss, slippage_bps)
        returns.append(ret - roundtrip_cost)
        stopped.append(int(did_stop))
        gap_stops.append(int(reason == "gap_stop"))
        reasons.append(reason)
    work["_net_return"] = returns
    work["_stopped"] = stopped
    work["_gap_stop"] = gap_stops
    work["_reason"] = reasons
    spy = work.get("future_spy_return", work["future_return"] - work["future_alpha"]).astype(float)
    work["_spy_return"] = spy.to_numpy()
    return work.groupby("date", as_index=False).agg(
        return_=("_net_return", "mean"),
        spy_return=("_spy_return", "mean"),
        stopped_positions=("_stopped", "sum"),
        gap_stops=("_gap_stop", "sum"),
        positions=("symbol", "count"),
    )


def run(args: argparse.Namespace) -> dict:
    device = pick_device(args.device)
    df = add_future_dates(pd.read_parquet(args.dataset), args.horizon_days)
    payload = {"config": vars(args), "folds": [], "warning": "research_only_intraday_stop_validation"}
    for fold in default_folds():
        scored, rule = score_period(df, fold, device, args.batch_size)
        trades = selected_trades(scored, rule, args.top_k, args.max_positions, args.horizon_days)
        fold_rows = []
        for stop_loss in args.stop_loss:
            periods = intraday_periods(trades, stop_loss, args.roundtrip_cost, args.slippage_bps)
            result = simulate_overlay(periods, Overlay(stop_loss=stop_loss, drawdown_stop=None, cooldown_days=0))
            fold_rows.append({"stop_loss": float(stop_loss), "result": result})
        payload["folds"].append({
            "fold": fold.name,
            "checkpoint": fold.checkpoint,
            "candidate_trades": int(len(trades)),
            "results": fold_rows,
        })
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps(payload, indent=2, default=str), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="checkpoints/daily_ranker/exp11_latest_dataset_h5_2026/daily_ranker_dataset.parquet")
    parser.add_argument("--output", default="checkpoints/daily_listwise_ranker/riskadj_intraday_stop_eval.json")
    parser.add_argument("--stop-loss", type=float, nargs="+", default=[0.03, 0.05, 0.08])
    parser.add_argument("--roundtrip-cost", type=float, default=0.0015)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-positions", type=int, default=3)
    parser.add_argument("--horizon-days", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="auto")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
