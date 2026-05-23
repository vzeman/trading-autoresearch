"""Recommend candidates from the best daily-ranker consensus protocol.

This is a research helper, not an order router. It loads the best passing rule
from a protocol sweep, scores the latest available dataset date with the prior
checkpoints, and prints the consensus symbols that pass the regime gate.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_daily_ranker_consensus import Config as ConsensusConfig
from evaluate_daily_ranker_consensus import consensus_frame, pick_device, score_checkpoint
from sweep_daily_ranker_consensus import CKPT_2021, CKPT_2022, CKPT_2023, CKPT_2024, CKPT_2025, DATASET
from train_daily_ranker import FEATURE_COLS, add_features, add_market_context, daily_bars
from top500_universe import load_top500_symbols


def load_best_rule(path: str, allow_unpassed: bool = False) -> dict:
    payload = json.loads(Path(path).read_text())
    for row in payload.get("top", []):
        if row.get("passed"):
            return row["rule"]
    if allow_unpassed and payload.get("top"):
        return payload["top"][0]["rule"]
    raise RuntimeError(f"no passing rule found in {path}")


def _cached_symbols(limit: int) -> list[str]:
    symbols = ["SPY"] + [sym for sym in load_top500_symbols() if sym != "SPY"]
    out = []
    seen = set()
    for sym in symbols:
        if sym in seen:
            continue
        seen.add(sym)
        try:
            daily_bars(sym)
        except Exception:
            continue
        out.append(sym)
        if limit > 0 and len(out) >= limit:
            break
    return out


def build_live_features(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.Timestamp]:
    if args.live_feature_cache and Path(args.live_feature_cache).exists() and not args.refresh_live_features:
        day = pd.read_parquet(args.live_feature_cache)
        missing = [col for col in FEATURE_COLS if col not in day.columns]
        if not missing:
            day["date"] = pd.to_datetime(day["date"], utc=True)
            latest_date = pd.to_datetime(day["date"], utc=True).max()
            return day, latest_date
        print(f"[daily-consensus-live] cached live features missing {len(missing)} columns; rebuilding", flush=True)

    symbols = _cached_symbols(args.symbol_limit)
    if "SPY" not in symbols:
        raise RuntimeError("SPY cache is required for daily-ranker live features")
    spy_daily = daily_bars("SPY")
    frames = []
    for idx, sym in enumerate(symbols, start=1):
        try:
            feat = add_features(daily_bars(sym), spy_daily, args.horizon_days)
            frames.append(feat)
        except Exception as exc:
            print(f"[daily-consensus-live] skip {sym}: {exc}", flush=True)
        if idx % 100 == 0:
            print(f"[daily-consensus-live] featurized {idx}/{len(symbols)}", flush=True)
    if not frames:
        raise RuntimeError("no live feature rows could be built")
    df = pd.concat(frames, ignore_index=True)
    for col in ("ret_5d", "ret_20d", "vol_20d", "drawdown_60d", "volume_z_20d"):
        rank_col = {
            "ret_5d": "xsec_ret_5d_rank",
            "ret_20d": "xsec_ret_20d_rank",
            "vol_20d": "xsec_vol_20d_rank",
            "drawdown_60d": "xsec_drawdown_60d_rank",
            "volume_z_20d": "xsec_volume_z_20d_rank",
        }[col]
        df[rank_col] = df.groupby("date")[col].rank(pct=True)
    df = add_market_context(df)
    feature_ready = df[["date", "symbol", *FEATURE_COLS]].replace([np.inf, -np.inf], np.nan).dropna()
    if args.date:
        latest_date = pd.Timestamp(args.date, tz="UTC")
    else:
        latest_date = pd.to_datetime(feature_ready["date"], utc=True).max()
    day = df[pd.to_datetime(df["date"], utc=True) == latest_date].copy()
    day = day.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS).reset_index(drop=True)
    if day.empty:
        raise RuntimeError(f"no feature-complete rows for decision date {latest_date.date()}")
    day["future_return"] = 0.0
    day["future_spy_return"] = 0.0
    day["future_alpha"] = 0.0
    if args.live_feature_cache:
        Path(args.live_feature_cache).parent.mkdir(parents=True, exist_ok=True)
        day.to_parquet(args.live_feature_cache, index=False)
    return day, latest_date


def load_dataset_features(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.Timestamp]:
    df = pd.read_parquet(args.dataset)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    latest_date = pd.Timestamp(args.date, tz="UTC") if args.date else df["date"].max()
    day = df[df["date"] == latest_date].copy()
    if day.empty:
        raise RuntimeError(f"no rows for decision date {latest_date.date()}")
    return day, latest_date


def run(args: argparse.Namespace) -> dict:
    device = pick_device(args.device)
    day, latest_date = load_dataset_features(args) if args.use_dataset else build_live_features(args)

    try:
        rule = load_best_rule(args.protocol_sweep, args.allow_unpassed_rule)
    except RuntimeError as exc:
        payload = {
            "decision_date": str(latest_date.date()),
            "dataset": args.dataset,
            "feature_source": "dataset" if args.use_dataset else "live_cache",
            "protocol_sweep": args.protocol_sweep,
            "device": device,
            "rule": None,
            "min_votes": None,
            "diagnostics": {
                "feature_rows": int(len(day)),
                "reason": str(exc),
            },
            "recommendations": [],
            "decision": "no_trade",
        }
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(payload, indent=2, default=str))
        print(json.dumps(payload, indent=2, default=str), flush=True)
        return payload

    checkpoints = [CKPT_2021, CKPT_2022, CKPT_2023, CKPT_2024, CKPT_2025]
    min_votes = max(2, len(checkpoints) - int(rule["min_vote_gap"]))
    config = ConsensusConfig(
        dataset=args.dataset,
        checkpoints=checkpoints,
        output="",
        test_start=str(latest_date.date()),
        test_end=str((latest_date + pd.Timedelta(days=1)).date()),
        horizon_days=args.horizon_days,
        min_votes=min_votes,
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
        top_k=args.top_k,
        max_positions=args.max_positions,
        batch_size=args.batch_size,
        device=device,
    )
    scored_parts = [score_checkpoint(day, path, device, args.batch_size) for path in checkpoints]
    checkpoint_counts = {
        path: int(len(frame))
        for path, frame in zip(checkpoints, scored_parts, strict=True)
    }
    prefilter_config = ConsensusConfig(
        **{
            **asdict(config),
            "min_spy_ret_20d": None,
            "min_rel_spy_20d": None,
            "min_ret_20d": None,
            "min_drawdown_60d": None,
            "min_mkt_pct_positive_20d": None,
            "min_mkt_pct_above_ma20": None,
            "min_mkt_ret_20d_mean": None,
            "max_mkt_ret_20d_dispersion": None,
        }
    )
    prefilter_consensus = consensus_frame(scored_parts, prefilter_config)
    consensus = consensus_frame(scored_parts, config)
    if consensus.empty:
        recommendations = []
    else:
        recommendations = (
            consensus.sort_values(["pred_score", "votes"], ascending=[False, False])
            .head(args.top_k)[
                [
                    "date",
                    "symbol",
                    "votes",
                    "pred_score",
                    "pred_profit",
                    "pred_crash",
                    "pred_top",
                    "ret_20d",
                    "spy_ret_20d",
                    "rel_spy_20d",
                    "mkt_pct_positive_20d",
                    "mkt_pct_above_ma20",
                ]
            ]
            .to_dict(orient="records")
        )
    payload = {
        "decision_date": str(latest_date.date()),
        "dataset": args.dataset,
        "feature_source": "dataset" if args.use_dataset else "live_cache",
        "protocol_sweep": args.protocol_sweep,
        "device": device,
        "rule": rule,
        "min_votes": min_votes,
        "diagnostics": {
            "feature_rows": int(len(day)),
            "checkpoint_selected_rows": checkpoint_counts,
            "consensus_rows_before_regime_filters": int(len(prefilter_consensus)),
            "consensus_rows_after_regime_filters": int(len(consensus)),
            "market": {
                key: float(day[key].dropna().iloc[0]) if key in day.columns and not day[key].dropna().empty else None
                for key in (
                    "spy_ret_20d",
                    "mkt_pct_positive_20d",
                    "mkt_pct_above_ma20",
                    "mkt_ret_20d_mean",
                    "mkt_ret_20d_dispersion",
                )
            },
        },
        "recommendations": recommendations,
        "decision": "buy_candidates" if recommendations else "no_trade",
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps(payload, indent=2, default=str), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--protocol-sweep", default="checkpoints/daily_ranker/consensus_protocol_sweep_regime_min3_with2025.json")
    parser.add_argument("--output", default="checkpoints/daily_ranker/latest_consensus_recommendation.json")
    parser.add_argument("--date", default="")
    parser.add_argument("--use-dataset", action="store_true")
    parser.add_argument("--symbol-limit", type=int, default=503)
    parser.add_argument("--live-feature-cache", default="checkpoints/daily_ranker/latest_live_features.parquet")
    parser.add_argument("--refresh-live-features", action="store_true")
    parser.add_argument("--allow-unpassed-rule", action="store_true")
    parser.add_argument("--horizon-days", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-positions", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="auto")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
