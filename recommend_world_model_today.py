"""Generate latest buy candidates from the action-conditioned world model.

This is a research signal, not an order router. It builds live-style candidate
rows from the latest cached bars without realized future outcomes, scores them
with a trained world model and allocator, then applies the current q80/75
tradability rule.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluate_world_model import predict
from prepare import CACHE_DIR, STARTING_CASH_USD, fetch_bars
from train_allocator import apply_allocator
from train_world_model import pick_device
from top500_universe import load_top500_symbols
from world_model_dataset import (
    BuildConfig,
    CONTEXT_BARS,
    DEFAULT_ACTIONS_FULL,
    USE_FEATURES,
    _cache_path,
    _rolling_state_features,
    _source_bars_for_split,
    _safe_log_return,
    _safe_volume_z,
    _safe_window_vol,
    _trade_to_target,
)
from experiment import fetch_context, featurize


def _latest_regular_timestamp(symbols: list[str], min_fresh_symbols: int) -> pd.Timestamp:
    latest: list[pd.Timestamp] = []
    for sym in symbols:
        path = _cache_path(sym)
        if not path.exists():
            continue
        try:
            frame = pd.read_parquet(path, columns=["timestamp"])
        except Exception:
            continue
        if frame.empty:
            continue
        ts = pd.to_datetime(frame["timestamp"], utc=True).max()
        if 13 <= ts.hour <= 21:
            latest.append(ts)
    if not latest:
        raise RuntimeError("no cached timestamps found")
    counts = pd.Series(latest).dt.floor("min").value_counts().sort_index()
    eligible = counts[counts >= min_fresh_symbols]
    if not eligible.empty:
        return pd.Timestamp(eligible.index.max())
    return pd.Timestamp(max(latest))


def _context_config(symbols: list[str], action_mode: str) -> BuildConfig:
    return BuildConfig(
        symbols=symbols,
        samples_per_symbol=1,
        seed=0,
        horizons=[15, 30, 60, 120],
        actions_per_timestamp=len(DEFAULT_ACTIONS_FULL),
        use_top500=False,
        symbol_limit=0,
        cached_only=True,
        split_name="all",
        context_bars=CONTEXT_BARS,
        output="",
        action_mode=action_mode,
        cross_sectional=True,
        shared_timestamps=True,
        shard_by_symbol=False,
    )


def _parse_utc_timestamp(text: str) -> pd.Timestamp:
    ts = pd.Timestamp(text)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _causal_tail(source: pd.DataFrame, decision_ts: pd.Timestamp, bars: int) -> pd.DataFrame:
    ts = pd.to_datetime(source["timestamp"], utc=True)
    out = source.loc[ts <= decision_ts].sort_values("timestamp")
    if bars > 0 and len(out) > bars:
        out = out.tail(bars)
    return out.reset_index(drop=True)


def _build_live_cross_sectional_features(
    symbols: list[str],
    decision_ts: pd.Timestamp,
    context: dict[str, pd.DataFrame],
    min_bars: int,
    feature_tail_bars: int,
) -> dict[str, dict[int, dict[str, float]]]:
    """Compute cross-sectional features for a single live timestamp."""
    records: list[dict[str, float | str | int]] = []
    windows = ((30, "30m"), (120, "2h"), (390, "1d"))
    ts_ns = int(pd.Timestamp(decision_ts).value)

    for sym in symbols:
        try:
            bars = fetch_bars(sym, force=False)
        except Exception:
            continue
        source = _causal_tail(_source_bars_for_split(bars, "all"), decision_ts, feature_tail_bars)
        if len(source) < max(min_bars, max(w for w, _ in windows) + 2):
            continue
        feat = featurize(source, context=context).dropna().reset_index(drop=True)
        if len(feat) < max(w for w, _ in windows) + 2:
            continue
        ts_arr = pd.to_datetime(feat["timestamp"], utc=True).astype("int64").to_numpy()
        i = int(np.searchsorted(ts_arr, ts_ns, side="right") - 1)
        if i < max(w for w, _ in windows):
            continue
        close = feat["close"].to_numpy(np.float32)
        volume = source.sort_values("timestamp")["volume"].to_numpy(np.float32)[-len(feat):]
        row: dict[str, float | str | int] = {
            "symbol": sym,
            "decision_ns": ts_ns,
            "xsec_price": float(close[i]),
        }
        for bars_n, label in windows:
            row[f"xsec_ret_{label}"] = _safe_log_return(close, i - bars_n, i)
        row["xsec_vol_1d"] = _safe_window_vol(close, i, 390)
        row["xsec_volume_z_1d"] = _safe_volume_z(volume, i, 390)
        records.append(row)

    if not records:
        return {}
    df = pd.DataFrame(records)
    out = df[["symbol", "decision_ns"]].copy()
    grouped = df.groupby("decision_ns", sort=False)
    out["xsec_universe_count"] = grouped["symbol"].transform("count").astype(float)
    metric_cols = ["xsec_ret_30m", "xsec_ret_2h", "xsec_ret_1d", "xsec_vol_1d", "xsec_volume_z_1d"]
    for col in metric_cols:
        g = grouped[col]
        out[f"{col}_mean"] = g.transform("mean")
        out[f"{col}_median"] = g.transform("median")
        out[f"{col}_std"] = g.transform("std").fillna(0.0)
        out[f"{col}_p10"] = g.transform(lambda s: s.quantile(0.10))
        out[f"{col}_p90"] = g.transform(lambda s: s.quantile(0.90))
        out[f"{col}_dispersion"] = out[f"{col}_p90"] - out[f"{col}_p10"]
        out[f"{col}_rank_pct"] = grouped[col].rank(pct=True, method="average")
        out[f"{col}_minus_median"] = df[col] - out[f"{col}_median"]
        if col.startswith("xsec_ret_"):
            out[f"{col}_up_frac"] = grouped[col].transform(lambda s: float((s > 0).mean()))

    feature_cols = [c for c in out.columns if c not in ("symbol", "decision_ns")]
    out[feature_cols] = out[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    feature_map: dict[str, dict[int, dict[str, float]]] = {}
    for rec in out.to_dict(orient="records"):
        sym = str(rec.pop("symbol"))
        rec_ts_ns = int(rec.pop("decision_ns"))
        feature_map.setdefault(sym, {})[rec_ts_ns] = {k: float(v) for k, v in rec.items()}
    return feature_map


def _candidate_rows_for_symbol(
    sym: str,
    decision_ts: pd.Timestamp,
    context: dict[str, pd.DataFrame],
    cross_sectional: dict[str, dict[int, dict[str, float]]],
    min_bars: int,
    feature_tail_bars: int,
) -> list[dict]:
    try:
        bars = fetch_bars(sym, force=False)
    except Exception:
        return []
    source = _causal_tail(_source_bars_for_split(bars, "all"), decision_ts, feature_tail_bars)
    if len(source) < min_bars:
        return []
    feat = featurize(source, context=context).dropna().reset_index(drop=True)
    if len(feat) < CONTEXT_BARS + 2:
        return []
    ts_arr = pd.to_datetime(feat["timestamp"], utc=True).astype("int64").to_numpy()
    i = int(np.searchsorted(ts_arr, pd.Timestamp(decision_ts).value, side="right") - 1)
    if i < CONTEXT_BARS:
        return []
    close = feat["close"].to_numpy(np.float32)
    volume = source.sort_values("timestamp")["volume"].to_numpy(np.float32)[-len(feat):]
    price = float(close[i])
    base = {
        "symbol": sym,
        "timestamp": feat["timestamp"].iloc[i],
        "decision_timestamp": pd.Timestamp(decision_ts),
        "split": "live",
        "price": price,
    }
    for name in USE_FEATURES:
        base[f"feat_{name}"] = float(feat[name].iloc[i])
    base.update(_rolling_state_features(close, volume, i))
    base.update(cross_sectional.get(sym, {}).get(int(pd.Timestamp(decision_ts).value), {}))

    rows: list[dict] = []
    for action, current_frac, target_frac in DEFAULT_ACTIONS_FULL:
        if action != "buy" or current_frac != 0.0 or target_frac > 0.75:
            continue
        cash, qty, fee, slippage, notional = _trade_to_target(price, current_frac, target_frac, STARTING_CASH_USD)
        del cash, qty
        for horizon in (15, 30, 60, 120):
            row = dict(base)
            row.update({
                "action": action,
                "horizon_bars": int(horizon),
                "current_position_frac": float(current_frac),
                "target_position_frac": float(target_frac),
                "trade_notional": float(notional),
                "fees": float(fee),
                "slippage": float(slippage),
            })
            rows.append(row)
    return rows


def build_candidates(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    symbols = load_top500_symbols() if args.top500 else sorted(p.name.removesuffix("_1m.parquet") for p in CACHE_DIR.glob("*_1m.parquet"))
    if args.limit > 0:
        symbols = symbols[: args.limit]
    if args.symbols:
        wanted = {s.strip().upper() for part in args.symbols for s in part.split(",") if s.strip()}
        symbols = [s for s in symbols if s in wanted]
    decision_ts = _parse_utc_timestamp(args.decision_timestamp) if args.decision_timestamp else _latest_regular_timestamp(symbols, args.min_fresh_symbols)
    context = fetch_context(force=False)
    cross = _build_live_cross_sectional_features(
        symbols=symbols,
        decision_ts=decision_ts,
        context=context,
        min_bars=args.min_bars,
        feature_tail_bars=args.feature_tail_bars,
    )

    rows: list[dict] = []
    for sym in symbols:
        rows.extend(_candidate_rows_for_symbol(
            sym=sym,
            decision_ts=decision_ts,
            context=context,
            cross_sectional=cross,
            min_bars=args.min_bars,
            feature_tail_bars=args.feature_tail_bars,
        ))
    frame = pd.DataFrame(rows)
    meta = {
        "decision_timestamp": str(decision_ts),
        "symbols_requested": len(symbols),
        "symbols_with_candidates": int(frame["symbol"].nunique()) if not frame.empty else 0,
        "candidate_rows": int(len(frame)),
    }
    return frame, meta


def run(args: argparse.Namespace) -> dict:
    device = pick_device(args.device)
    world_ckpt = torch.load(args.world_checkpoint, map_location="cpu", weights_only=False)
    allocator_ckpt = torch.load(args.allocator_checkpoint, map_location="cpu", weights_only=False)
    candidates, meta = build_candidates(args)
    if candidates.empty:
        raise RuntimeError("no live candidates built")
    scored = predict(candidates, world_ckpt, device=device, batch_size=args.batch_size)
    scored = apply_allocator(scored, allocator_ckpt, device=device, batch_size=args.batch_size)
    gate_counts = {"scored_rows": int(len(scored))}
    eligible = scored[scored["target_position_frac"].astype(float) <= args.max_target_position_frac].copy()
    gate_counts["after_max_target"] = int(len(eligible))
    eligible = eligible[eligible["pred_score"].astype(float) >= args.score_threshold].copy()
    gate_counts["after_score_threshold"] = int(len(eligible))
    if not args.no_strict_risk_gates:
        if "pred_profit_label" in eligible.columns:
            eligible = eligible[eligible["pred_profit_label"].astype(float) >= args.min_pred_profit_label].copy()
        gate_counts["after_min_pred_profit"] = int(len(eligible))
        if "pred_beat_spy_label" in eligible.columns:
            eligible = eligible[eligible["pred_beat_spy_label"].astype(float) >= args.min_pred_beat_spy_label].copy()
        gate_counts["after_min_pred_beat_spy"] = int(len(eligible))
        if "pred_future_alpha_vs_spy" in eligible.columns:
            eligible = eligible[eligible["pred_future_alpha_vs_spy"].astype(float) >= args.min_pred_future_alpha_vs_spy].copy()
        gate_counts["after_min_pred_alpha"] = int(len(eligible))
        if "pred_max_drawdown" in eligible.columns:
            eligible = eligible[eligible["pred_max_drawdown"].astype(float) >= args.min_pred_max_drawdown].copy()
        gate_counts["after_min_pred_drawdown"] = int(len(eligible))
        if "pred_future_min_asset_return" in eligible.columns:
            eligible = eligible[eligible["pred_future_min_asset_return"].astype(float) >= args.min_pred_future_min_asset_return].copy()
        gate_counts["after_min_pred_future_min_asset_return"] = int(len(eligible))
        if "pred_future_asset_max_drawdown" in eligible.columns:
            eligible = eligible[eligible["pred_future_asset_max_drawdown"].astype(float) >= args.min_pred_future_asset_max_drawdown].copy()
        gate_counts["after_min_pred_future_asset_max_drawdown"] = int(len(eligible))
        if "pred_asset_crash_label" in eligible.columns:
            eligible = eligible[eligible["pred_asset_crash_label"].astype(float) <= args.max_pred_asset_crash_label].copy()
        gate_counts["after_max_pred_asset_crash"] = int(len(eligible))
        if "pred_severe_adverse_label" in eligible.columns:
            eligible = eligible[eligible["pred_severe_adverse_label"].astype(float) <= args.max_pred_severe_adverse_label].copy()
        gate_counts["after_max_pred_severe_adverse"] = int(len(eligible))
        observable_filters = [
            ("state_ret_30m", ">=", args.min_state_ret_30m),
            ("state_ret_2h", ">=", args.min_state_ret_2h),
            ("state_ret_1d", ">=", args.min_state_ret_1d),
            ("state_ret_5d", ">=", args.min_state_ret_5d),
            ("state_drawdown_5d", ">=", args.min_state_drawdown_5d),
            ("state_vol_1d", "<=", args.max_state_vol_1d),
        ]
        for col, op, threshold in observable_filters:
            if col not in eligible.columns:
                continue
            if op == ">=":
                eligible = eligible[eligible[col].astype(float) >= float(threshold)].copy()
            else:
                eligible = eligible[eligible[col].astype(float) <= float(threshold)].copy()
            gate_counts[f"after_{col}"] = int(len(eligible))
    sort_cols = ["pred_score", "pred_portfolio_return", "pred_beat_spy_label", "pred_profit_label"]
    best = eligible.sort_values(sort_cols, ascending=False).groupby("symbol", as_index=False).head(1)
    best = best.sort_values(sort_cols, ascending=False).head(args.top)
    fields = [
        "symbol", "decision_timestamp", "timestamp", "price", "horizon_bars",
        "target_position_frac", "pred_score", "pred_portfolio_return",
        "pred_future_alpha_vs_spy", "pred_max_drawdown", "pred_path_vol",
        "pred_future_min_asset_return", "pred_future_asset_max_drawdown",
        "pred_profit_label", "pred_beat_spy_label", "pred_asset_crash_label",
        "pred_severe_adverse_label", "state_ret_30m",
        "state_ret_2h", "state_ret_1d", "state_ret_5d",
        "state_drawdown_5d", "state_vol_1d",
    ]
    recommendations = best[[c for c in fields if c in best.columns]].copy()
    payload = {
        "meta": meta,
        "device": device,
        "world_checkpoint": args.world_checkpoint,
        "allocator_checkpoint": args.allocator_checkpoint,
        "score_threshold": float(args.score_threshold),
        "max_target_position_frac": float(args.max_target_position_frac),
        "strict_risk_gates": not bool(args.no_strict_risk_gates),
        "risk_gate_settings": {
            "min_pred_profit_label": float(args.min_pred_profit_label),
            "min_pred_beat_spy_label": float(args.min_pred_beat_spy_label),
            "min_pred_future_alpha_vs_spy": float(args.min_pred_future_alpha_vs_spy),
            "min_pred_max_drawdown": float(args.min_pred_max_drawdown),
            "min_pred_future_min_asset_return": float(args.min_pred_future_min_asset_return),
            "min_pred_future_asset_max_drawdown": float(args.min_pred_future_asset_max_drawdown),
            "max_pred_asset_crash_label": float(args.max_pred_asset_crash_label),
            "max_pred_severe_adverse_label": float(args.max_pred_severe_adverse_label),
            "min_state_ret_30m": float(args.min_state_ret_30m),
            "min_state_ret_2h": float(args.min_state_ret_2h),
            "min_state_ret_1d": float(args.min_state_ret_1d),
            "min_state_ret_5d": float(args.min_state_ret_5d),
            "min_state_drawdown_5d": float(args.min_state_drawdown_5d),
            "max_state_vol_1d": float(args.max_state_vol_1d),
        },
        "gate_counts": gate_counts,
        "eligible_rows": int(len(eligible)),
        "eligible_symbols": int(eligible["symbol"].nunique()) if not eligible.empty else 0,
        "recommendations": json.loads(recommendations.to_json(orient="records", date_format="iso")),
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-checkpoint", default="checkpoints/world_model/rolling_retrain_top500_adjusted_liquid_xsec_q80_cap75/world_model_fold_2026.pt")
    parser.add_argument("--allocator-checkpoint", default="checkpoints/world_model/rolling_retrain_top500_adjusted_liquid_xsec_q80_cap75/allocator_fold_2026.pt")
    parser.add_argument("--score-threshold", type=float, default=0.3246140003204346)
    parser.add_argument("--max-target-position-frac", type=float, default=0.75)
    parser.add_argument("--no-strict-risk-gates", action="store_true", help="disable crash-avoidance gates; for diagnostics only")
    parser.add_argument("--min-pred-profit-label", type=float, default=0.60)
    parser.add_argument("--min-pred-beat-spy-label", type=float, default=0.55)
    parser.add_argument("--min-pred-future-alpha-vs-spy", type=float, default=0.0)
    parser.add_argument("--min-pred-max-drawdown", type=float, default=-0.006)
    parser.add_argument("--min-pred-future-min-asset-return", type=float, default=-0.015)
    parser.add_argument("--min-pred-future-asset-max-drawdown", type=float, default=-0.015)
    parser.add_argument("--max-pred-asset-crash-label", type=float, default=0.35)
    parser.add_argument("--max-pred-severe-adverse-label", type=float, default=0.20)
    parser.add_argument("--min-state-ret-30m", type=float, default=-0.004)
    parser.add_argument("--min-state-ret-2h", type=float, default=-0.010)
    parser.add_argument("--min-state-ret-1d", type=float, default=-0.025)
    parser.add_argument("--min-state-ret-5d", type=float, default=-0.060)
    parser.add_argument("--min-state-drawdown-5d", type=float, default=-0.080)
    parser.add_argument("--max-state-vol-1d", type=float, default=0.025)
    parser.add_argument("--output", default="checkpoints/world_model/latest_world_model_recommendations.json")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--top500", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--symbols", nargs="*", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-bars", type=int, default=CONTEXT_BARS + 120 + 2)
    parser.add_argument("--feature-tail-bars", type=int, default=25000)
    parser.add_argument("--min-fresh-symbols", type=int, default=50)
    parser.add_argument("--decision-timestamp", default="")
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--device", default="auto")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
