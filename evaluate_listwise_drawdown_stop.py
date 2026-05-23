"""Evaluate stop-loss and drawdown-cooldown overlays for listwise checkpoints.

This script does not retrain. It reloads saved listwise ranker checkpoints,
scores their locked test windows, applies the checkpoint's saved rule, then
tests simple portfolio-level risk overlays on the selected trades.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from recommend_daily_listwise_ranker import apply_rule
from train_daily_listwise_ranker import ListwiseRanker
from train_daily_ranker import FEATURE_COLS, pick_device


@dataclass(frozen=True)
class Fold:
    name: str
    checkpoint: str
    test_start: str
    test_end: str


@dataclass(frozen=True)
class Overlay:
    stop_loss: float | None
    drawdown_stop: float | None
    cooldown_days: int


def default_folds() -> list[Fold]:
    root = "checkpoints/daily_listwise_ranker"
    return [
        Fold("2023", f"{root}/exp5_riskadj_train2022_test2023_strictcal/daily_listwise_ranker.pt", "2023-01-01", "2024-01-01"),
        Fold("2024", f"{root}/exp5_riskadj_train2023_test2024_strictcal/daily_listwise_ranker.pt", "2024-01-01", "2025-01-01"),
        Fold("2025", f"{root}/exp5_riskadj_train2024_test2025_strictcal/daily_listwise_ranker.pt", "2025-01-01", "2026-01-01"),
        Fold("2026_ytd", f"{root}/exp5_riskadj_train2025_2026_strictcal/daily_listwise_ranker.pt", "2026-01-01", "2026-05-10"),
    ]


def load_checkpoint(path: str, device: str) -> tuple[ListwiseRanker, dict, dict, dict, list[str]]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    feature_cols = ckpt.get("feature_cols", FEATURE_COLS)
    model = ListwiseRanker(len(feature_cols), int(cfg.get("hidden_dim", 160)), float(cfg.get("dropout", 0.0))).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["arrays"], ckpt["rule"], cfg, feature_cols


def score_period(df: pd.DataFrame, fold: Fold, device: str, batch_size: int) -> tuple[pd.DataFrame, dict]:
    model, arrays, rule, cfg, feature_cols = load_checkpoint(fold.checkpoint, device)
    dates = pd.to_datetime(df["date"], utc=True)
    mask = (dates >= pd.Timestamp(fold.test_start, tz="UTC")) & (dates < pd.Timestamp(fold.test_end, tz="UTC"))
    work = df.loc[mask].copy().reset_index(drop=True)
    if work.empty:
        return work, rule
    x = work[feature_cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
    x = np.clip((x - arrays["x_mean"]) / arrays["x_std"], -10.0, 10.0).astype(np.float32)
    preds = []
    xt = torch.from_numpy(x)
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = xt[start:start + batch_size].to(device)
            score, profit, crash = model(xb)
            preds.append(torch.stack([score, torch.sigmoid(profit), torch.sigmoid(crash)], dim=1).cpu().numpy())
    arr = np.concatenate(preds, axis=0) if preds else np.zeros((0, 3), dtype=np.float32)
    work["pred_utility"] = arr[:, 0]
    work["pred_profit"] = arr[:, 1]
    work["pred_crash"] = arr[:, 2]
    work["pred_top"] = work.groupby("date")["pred_utility"].rank(pct=True).astype(float)
    work["pred_score"] = (
        work["pred_utility"]
        + float(cfg.get("score_profit_weight", 0.04)) * work["pred_profit"]
        + float(cfg.get("score_top_weight", 0.08)) * work["pred_top"]
        - float(cfg.get("score_crash_weight", 0.10)) * work["pred_crash"]
    )
    return work, rule


def selected_trades(scored: pd.DataFrame, rule: dict, top_k: int, max_positions: int, horizon_days: int) -> pd.DataFrame:
    active = apply_rule(scored, rule)
    selected = []
    next_date = pd.Timestamp.min.tz_localize("UTC")
    for date, group in active.sort_values(["date", "pred_score"], ascending=[True, False]).groupby("date", sort=True):
        ts = pd.Timestamp(date)
        if ts < next_date:
            continue
        selected.append(group.head(top_k))
        next_date = ts + pd.Timedelta(days=max(1, horizon_days))
    out = pd.concat(selected, ignore_index=True) if selected else active.iloc[:0].copy()
    if max_positions > 0 and not out.empty:
        out = out.groupby("date", group_keys=False).head(max_positions).reset_index(drop=True)
    return out


def period_returns(trades: pd.DataFrame, stop_loss: float | None, roundtrip_cost: float) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["date", "return", "spy_return", "stopped_positions", "positions"])
    work = trades.copy()
    raw = work["future_return"].astype(float).to_numpy()
    mins = work.get("future_min_return", pd.Series(0.0, index=work.index)).astype(float).to_numpy()
    stopped = np.zeros(len(work), dtype=bool)
    if stop_loss is not None and stop_loss > 0:
        stopped = mins <= -float(stop_loss)
        raw = np.where(stopped, -float(stop_loss), raw)
    work["_net_return"] = raw - roundtrip_cost
    work["_stopped"] = stopped.astype(int)
    spy = work.get("future_spy_return", work["future_return"] - work["future_alpha"]).astype(float)
    work["_spy_return"] = spy.to_numpy()
    return work.groupby("date", as_index=False).agg(
        return_=("_net_return", "mean"),
        spy_return=("_spy_return", "mean"),
        stopped_positions=("_stopped", "sum"),
        positions=("symbol", "count"),
    )


def simulate_overlay(periods: pd.DataFrame, overlay: Overlay) -> dict:
    if periods.empty:
        return {
            "periods": 0,
            "trades": 0,
            "total_return": 0.0,
            "spy_active_return": 0.0,
            "active_alpha_return": 0.0,
            "max_drawdown": 0.0,
            "stopped_positions": 0,
            "skipped_periods": 0,
        }
    equity = 1.0
    spy_equity = 1.0
    peak = 1.0
    max_dd = 0.0
    cooldown_until = pd.Timestamp.min.tz_localize("UTC")
    used = 0
    trades = 0
    skipped = 0
    stopped_positions = 0
    for _, row in periods.sort_values("date").iterrows():
        ts = pd.Timestamp(row["date"])
        if overlay.drawdown_stop is not None and ts < cooldown_until:
            skipped += 1
            continue
        equity *= 1.0 + float(row["return_"])
        spy_equity *= 1.0 + float(row["spy_return"])
        peak = max(peak, equity)
        dd = equity / max(peak, 1e-12) - 1.0
        max_dd = min(max_dd, dd)
        used += 1
        trades += int(row["positions"])
        stopped_positions += int(row["stopped_positions"])
        if overlay.drawdown_stop is not None and dd <= -float(overlay.drawdown_stop):
            cooldown_until = ts + pd.Timedelta(days=max(1, overlay.cooldown_days))
    total = equity - 1.0
    spy_total = spy_equity - 1.0
    return {
        "periods": int(used),
        "trades": int(trades),
        "total_return": float(total),
        "spy_active_return": float(spy_total),
        "active_alpha_return": float(total - spy_total),
        "max_drawdown": float(max_dd),
        "stopped_positions": int(stopped_positions),
        "skipped_periods": int(skipped),
    }


def overlays() -> list[Overlay]:
    out = [Overlay(None, None, 0)]
    for stop_loss in (0.03, 0.05, 0.08):
        out.append(Overlay(stop_loss, None, 0))
    for stop_loss in (0.03, 0.05, 0.08):
        for dd_stop in (0.10, 0.15, 0.20):
            out.append(Overlay(stop_loss, dd_stop, 30))
    return out


def run(args: argparse.Namespace) -> dict:
    device = pick_device(args.device)
    df = pd.read_parquet(args.dataset)
    fold_payloads = []
    scored_periods: dict[str, pd.DataFrame] = {}
    for fold in default_folds():
        scored, rule = score_period(df, fold, device, args.batch_size)
        trades = selected_trades(scored, rule, args.top_k, args.max_positions, args.horizon_days)
        scored_periods[fold.name] = trades
        fold_payloads.append({"fold": asdict(fold), "candidate_trades": int(len(trades)), "rule": rule})

    candidates = []
    for overlay in overlays():
        fold_results = []
        for fold in default_folds():
            periods = period_returns(scored_periods[fold.name], overlay.stop_loss, args.roundtrip_cost)
            result = simulate_overlay(periods, overlay)
            fold_results.append({"fold": fold.name, "result": result})
        traded = [x["result"] for x in fold_results if x["result"]["periods"] > 0]
        candidates.append({
            "overlay": asdict(overlay),
            "fold_results": fold_results,
            "traded_folds": int(len(traded)),
            "sum_return": float(sum(x["total_return"] for x in traded)),
            "sum_active_alpha": float(sum(x["active_alpha_return"] for x in traded)),
            "worst_drawdown": float(min((x["max_drawdown"] for x in traded), default=0.0)),
        })
    candidates.sort(
        key=lambda x: (
            x["traded_folds"],
            x["sum_active_alpha"] + 0.25 * x["sum_return"] - 0.75 * abs(x["worst_drawdown"]),
        ),
        reverse=True,
    )
    payload = {
        "config": vars(args),
        "folds": fold_payloads,
        "best": candidates[0] if candidates else None,
        "candidates": candidates,
        "warning": "stop_loss_uses_future_min_return_backtest_assumption_research_only",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps(payload, indent=2, default=str), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="checkpoints/daily_ranker/exp11_latest_dataset_h5_2026/daily_ranker_dataset.parquet")
    parser.add_argument("--output", default="checkpoints/daily_listwise_ranker/riskadj_drawdown_stop_eval.json")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-positions", type=int, default=3)
    parser.add_argument("--horizon-days", type=int, default=5)
    parser.add_argument("--roundtrip-cost", type=float, default=0.0015)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="auto")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
