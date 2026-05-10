"""Build an action-conditioned portfolio world-model dataset.

This is intentionally separate from experiment.py/evaluator.py. It reuses the
cached bars and causal feature extraction, then generates counterfactual
portfolio-action outcomes:

    state(t, symbol, portfolio) + action + horizon -> realized portfolio value

The output is a parquet table suitable for training a world model that predicts
what happens if we buy, sell, or hold a symbol over intraday/daily/weekly
horizons under realistic fees and slippage.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from prepare import (
    CACHE_DIR,
    EVAL_DAYS,
    FEE_PER_TRADE_USD,
    MIN_TRADE_NOTIONAL_USD,
    SLIPPAGE_BPS,
    STARTING_CASH_USD,
    UNIVERSE,
    fetch_bars,
    split,
)
from experiment import USE_FEATURES, featurize, fetch_context


OUTPUT_DIR = Path("data/world_model")
CONTEXT_BARS = 390 * 20
DEFAULT_HORIZONS = (30, 120, 390, 1170, 1950)
DEFAULT_ACTIONS = (
    ("hold", 0.00, 0.00),
    ("hold", 0.05, 0.05),
    ("hold", 0.10, 0.10),
    ("hold", 0.20, 0.20),
    ("buy", 0.00, 0.05),
    ("buy", 0.00, 0.10),
    ("buy", 0.00, 0.20),
    ("buy", 0.05, 0.10),
    ("buy", 0.10, 0.20),
    ("sell", 0.05, 0.00),
    ("sell", 0.10, 0.00),
    ("sell", 0.20, 0.00),
)


@dataclass(frozen=True)
class BuildConfig:
    symbols: list[str]
    samples_per_symbol: int
    seed: int
    horizons: list[int]
    actions_per_timestamp: int
    use_top500: bool
    symbol_limit: int
    cached_only: bool
    split_name: str
    context_bars: int
    output: str
    shard_by_symbol: bool = False


def _cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol}_1m.parquet"


def _load_symbols(use_top500: bool, limit: int, cached_only: bool) -> list[str]:
    if use_top500:
        from top500_universe import load_top500_symbols

        symbols = load_top500_symbols()
    else:
        symbols = list(UNIVERSE)

    out: list[str] = []
    seen: set[str] = set()
    for sym in symbols:
        if sym in seen:
            continue
        seen.add(sym)
        if cached_only and not _cache_path(sym).exists():
            continue
        out.append(sym)
        if limit > 0 and len(out) >= limit:
            break
    return out


def _safe_log_return(close: np.ndarray, start: int, end: int) -> float:
    if start < 0 or end < 0 or start >= len(close) or end >= len(close):
        return 0.0
    return float(math.log(float(close[end]) / max(float(close[start]), 1e-12)))


def _rolling_state_features(close: np.ndarray, volume: np.ndarray, i: int) -> dict[str, float]:
    row: dict[str, float] = {}
    for bars, label in ((30, "30m"), (120, "2h"), (390, "1d"), (1950, "5d"), (7800, "20d")):
        if i - bars < 0:
            row[f"state_ret_{label}"] = 0.0
            row[f"state_vol_{label}"] = 0.0
            row[f"state_volume_z_{label}"] = 0.0
            continue
        window = close[i - bars : i + 1].astype(np.float64)
        rets = np.diff(np.log(np.maximum(window, 1e-12)))
        row[f"state_ret_{label}"] = _safe_log_return(close, i - bars, i)
        row[f"state_vol_{label}"] = float(np.std(rets, ddof=1)) if len(rets) > 2 else 0.0
        vol_window = volume[i - bars : i + 1].astype(np.float64)
        vol_mean = float(np.mean(vol_window))
        vol_std = float(np.std(vol_window, ddof=1)) if len(vol_window) > 2 else 0.0
        row[f"state_volume_z_{label}"] = float((volume[i] - vol_mean) / max(vol_std, 1e-9))
    recent_peak = float(np.max(close[max(0, i - 1950) : i + 1]))
    row["state_drawdown_5d"] = float(close[i] / max(recent_peak, 1e-12) - 1.0)
    return row


def _trade_to_target(
    current_price: float,
    current_frac: float,
    target_frac: float,
    starting_equity: float,
) -> tuple[float, float, float, float, float]:
    """Return cash, qty, fee, slippage, traded_notional after a target action."""
    current_price = max(float(current_price), 1e-9)
    qty_before = (starting_equity * current_frac) / current_price
    cash_before = starting_equity - qty_before * current_price
    qty_after = (starting_equity * target_frac) / current_price
    delta = qty_after - qty_before
    notional = abs(delta) * current_price
    if notional < MIN_TRADE_NOTIONAL_USD:
        return cash_before, qty_before, 0.0, 0.0, 0.0

    side = 1.0 if delta > 0 else -1.0
    fill_price = current_price * (1.0 + side * SLIPPAGE_BPS * 1e-4)
    slippage = abs(delta) * abs(fill_price - current_price)
    cash_after = cash_before - delta * fill_price - FEE_PER_TRADE_USD
    return cash_after, qty_after, FEE_PER_TRADE_USD, slippage, notional


def _outcome_for_action(
    close: np.ndarray,
    spy_close: np.ndarray | None,
    i: int,
    horizon: int,
    action: str,
    current_frac: float,
    target_frac: float,
) -> dict[str, float | int | str]:
    current_price = float(close[i])
    cash, qty, fee, slippage, notional = _trade_to_target(
        current_price=current_price,
        current_frac=current_frac,
        target_frac=target_frac,
        starting_equity=STARTING_CASH_USD,
    )
    path_prices = close[i : i + horizon + 1].astype(np.float64)
    equity = cash + qty * path_prices
    final_equity = float(equity[-1])
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.maximum(peak, 1e-12)
    rets = np.diff(equity) / np.maximum(equity[:-1], 1e-12)
    portfolio_return = final_equity / STARTING_CASH_USD - 1.0
    asset_return = float(path_prices[-1] / max(path_prices[0], 1e-12) - 1.0)
    spy_return = 0.0
    if spy_close is not None and i + horizon < len(spy_close):
        spy_return = float(spy_close[i + horizon] / max(spy_close[i], 1e-12) - 1.0)
    return {
        "action": action,
        "horizon_bars": int(horizon),
        "current_position_frac": float(current_frac),
        "target_position_frac": float(target_frac),
        "trade_notional": float(notional),
        "fees": float(fee),
        "slippage": float(slippage),
        "future_asset_return": asset_return,
        "future_spy_return": spy_return,
        "future_alpha_vs_spy": asset_return - spy_return,
        "final_equity": final_equity,
        "portfolio_return": float(portfolio_return),
        "portfolio_pnl": float(final_equity - STARTING_CASH_USD),
        "max_drawdown": float(np.min(dd)),
        "path_vol": float(np.std(rets, ddof=1)) if len(rets) > 2 else 0.0,
        "min_equity": float(np.min(equity)),
        "max_equity": float(np.max(equity)),
        "profit_label": int(portfolio_return > 0),
        "beat_spy_label": int(portfolio_return > spy_return),
    }


def _choose_indices(
    n: int,
    samples: int,
    min_i: int,
    max_horizon: int,
    rng: np.random.Generator,
) -> np.ndarray:
    hi = n - max_horizon - 1
    if hi <= min_i:
        return np.empty(0, dtype=np.int64)
    available = np.arange(min_i, hi, dtype=np.int64)
    if samples <= 0 or samples >= len(available):
        return available
    return np.sort(rng.choice(available, size=samples, replace=False))


def _align_spy_to_symbol(feat: pd.DataFrame, spy_bars: pd.DataFrame | None) -> np.ndarray | None:
    if spy_bars is None or spy_bars.empty:
        return None
    spy = spy_bars[["timestamp", "close"]].sort_values("timestamp")
    aligned = pd.merge_asof(
        feat[["timestamp"]].sort_values("timestamp"),
        spy,
        on="timestamp",
        direction="backward",
    )
    return aligned["close"].ffill().bfill().to_numpy(np.float32)


def _build_symbol_rows(
    sym: str,
    config: BuildConfig,
    context: dict[str, pd.DataFrame],
    spy_bars: pd.DataFrame | None,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    max_horizon = max(config.horizons)
    action_specs = list(DEFAULT_ACTIONS)

    try:
        bars = fetch_bars(sym, force=not config.cached_only)
    except Exception as exc:
        return pd.DataFrame(), {"status": "failed", "error": str(exc)}
    train_bars, eval_bars = split(bars)
    source_bars = train_bars if config.split_name == "train" else eval_bars
    if len(source_bars) < config.context_bars + max_horizon + 2:
        return pd.DataFrame(), {"status": "skipped", "bars": int(len(source_bars))}

    feat = featurize(source_bars, context=context)
    feat = feat.dropna().reset_index(drop=True)
    close = feat["close"].to_numpy(np.float32)
    volume = source_bars.sort_values("timestamp")["volume"].to_numpy(np.float32)[-len(feat):]
    spy_close = _align_spy_to_symbol(feat, spy_bars)
    valid_indices = _choose_indices(
        n=len(feat),
        samples=config.samples_per_symbol,
        min_i=config.context_bars,
        max_horizon=max_horizon,
        rng=rng,
    )
    if len(valid_indices) == 0:
        return pd.DataFrame(), {"status": "skipped", "bars": int(len(feat))}

    for i in valid_indices:
        base = {
            "symbol": sym,
            "timestamp": feat["timestamp"].iloc[int(i)],
            "split": config.split_name,
            "price": float(close[i]),
        }
        for name in USE_FEATURES:
            base[f"feat_{name}"] = float(feat[name].iloc[int(i)])
        base.update(_rolling_state_features(close, volume, int(i)))

        sampled_actions = rng.choice(
            len(action_specs),
            size=min(config.actions_per_timestamp, len(action_specs)),
            replace=False,
        )
        for action_idx in sampled_actions:
            action, current_frac, target_frac = action_specs[int(action_idx)]
            for horizon in config.horizons:
                row = dict(base)
                row.update(
                    _outcome_for_action(
                        close=close,
                        spy_close=spy_close,
                        i=int(i),
                        horizon=int(horizon),
                        action=action,
                        current_frac=float(current_frac),
                        target_frac=float(target_frac),
                    )
                )
                rows.append(row)
    stats = {
        "status": "ok",
        "bars": int(len(feat)),
        "timestamps_sampled": int(len(valid_indices)),
        "rows": int(len(rows)),
    }
    return pd.DataFrame(rows), stats


def _metadata(config: BuildConfig, row_count: int, symbol_stats: dict[str, dict], df: pd.DataFrame | None = None) -> dict:
    if df is not None and not df.empty:
        feature_columns = [c for c in df.columns if c.startswith("feat_") or c.startswith("state_")]
    else:
        feature_columns = [f"feat_{name}" for name in USE_FEATURES] + [
            f"state_{kind}_{label}"
            for label in ("30m", "2h", "1d", "5d", "20d")
            for kind in ("ret", "vol", "volume_z")
        ] + ["state_drawdown_5d"]
    return {
        "config": asdict(config),
        "rows": int(row_count),
        "symbols_ok": int(sum(1 for s in symbol_stats.values() if s.get("status") == "ok")),
        "symbol_stats": symbol_stats,
        "feature_columns": feature_columns,
        "target_columns": [
            "final_equity",
            "portfolio_return",
            "portfolio_pnl",
            "max_drawdown",
            "path_vol",
            "future_asset_return",
            "future_spy_return",
            "future_alpha_vs_spy",
            "profit_label",
            "beat_spy_label",
        ],
    }


def build_dataset(config: BuildConfig) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(config.seed)
    context = fetch_context(force=False)
    spy_bars = fetch_bars("SPY", force=False) if _cache_path("SPY").exists() else None
    frames: list[pd.DataFrame] = []
    symbol_stats: dict[str, dict] = {}

    for sym in config.symbols:
        df_sym, stats = _build_symbol_rows(sym, config, context, spy_bars, rng)
        symbol_stats[sym] = stats
        if not df_sym.empty:
            frames.append(df_sym)
        print(
            f"[world-dataset] {sym}: status={stats.get('status')} "
            f"bars={stats.get('bars', 0):,} sampled_ts={stats.get('timestamps_sampled', 0):,} "
            f"rows={stats.get('rows', 0):,}",
            flush=True,
        )

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return df, _metadata(config, len(df), symbol_stats, df)


def build_dataset_sharded(config: BuildConfig, output_dir: Path) -> dict:
    rng = np.random.default_rng(config.seed)
    context = fetch_context(force=False)
    spy_bars = fetch_bars("SPY", force=False) if _cache_path("SPY").exists() else None
    output_dir.mkdir(parents=True, exist_ok=True)
    symbol_stats: dict[str, dict] = {}
    rows_total = 0
    first_df: pd.DataFrame | None = None

    for sym in config.symbols:
        df_sym, stats = _build_symbol_rows(sym, config, context, spy_bars, rng)
        if not df_sym.empty:
            shard = output_dir / f"{sym}.parquet"
            df_sym.to_parquet(shard, index=False)
            stats = dict(stats)
            stats["shard"] = str(shard)
            rows_total += len(df_sym)
            if first_df is None:
                first_df = df_sym.head(1)
        symbol_stats[sym] = stats
        print(
            f"[world-dataset] {sym}: status={stats.get('status')} "
            f"bars={stats.get('bars', 0):,} sampled_ts={stats.get('timestamps_sampled', 0):,} "
            f"rows={stats.get('rows', 0):,}",
            flush=True,
        )
    return _metadata(config, rows_total, symbol_stats, first_df)


def _parse_horizons(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(OUTPUT_DIR / "train_counterfactual.parquet"))
    parser.add_argument("--metadata-output", default="")
    parser.add_argument("--top500", action="store_true", help="use current S&P 500 symbols")
    parser.add_argument("--symbol-limit", type=int, default=25)
    parser.add_argument("--samples-per-symbol", type=int, default=500)
    parser.add_argument("--actions-per-timestamp", type=int, default=6)
    parser.add_argument("--horizons", default=",".join(str(x) for x in DEFAULT_HORIZONS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=["train", "eval"], default="train")
    parser.add_argument("--context-bars", type=int, default=CONTEXT_BARS)
    parser.add_argument("--allow-download", action="store_true", help="fetch missing symbol bars instead of using cache only")
    parser.add_argument("--shard-by-symbol", action="store_true", help="write one parquet shard per symbol; recommended for large builds")
    args = parser.parse_args(list(argv) if argv is not None else None)

    symbols = _load_symbols(
        use_top500=args.top500,
        limit=args.symbol_limit,
        cached_only=not args.allow_download,
    )
    if not symbols:
        raise SystemExit("no symbols available; refresh cache or disable --top500")
    output = Path(args.output)
    metadata_output = Path(args.metadata_output) if args.metadata_output else output.with_suffix(".metadata.json")
    config = BuildConfig(
        symbols=symbols,
        samples_per_symbol=args.samples_per_symbol,
        seed=args.seed,
        horizons=_parse_horizons(args.horizons),
        actions_per_timestamp=args.actions_per_timestamp,
        use_top500=args.top500,
        symbol_limit=args.symbol_limit,
        cached_only=not args.allow_download,
        split_name=args.split,
        context_bars=args.context_bars,
        output=str(output),
        shard_by_symbol=args.shard_by_symbol,
    )
    if args.shard_by_symbol:
        meta = build_dataset_sharded(config, output)
    else:
        df, meta = build_dataset(config)
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output, index=False)
    metadata_output.write_text(json.dumps(meta, indent=2, default=str))
    print(f"[world-dataset] wrote {meta['rows']:,} rows -> {output}", flush=True)
    print(f"[world-dataset] wrote metadata -> {metadata_output}", flush=True)


if __name__ == "__main__":
    main()
