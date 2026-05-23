"""Sweep regime segmentation windows for Markov/trend portfolio strategies.

This compares how the bull/sideways/bear segmentation lookback changes the
portfolio result. It reuses the holding-period simulator and varies
`regime_window_days`, optionally across fixed and adaptive regime sources.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from evaluate_markov_holding_periods import (
    add_cross_sectional_scores,
    default_markov_config,
    simulate_holding,
    strategies,
)
from evaluate_markov_regime_quant_strategy import (
    STATE_BEAR,
    STATE_BULL,
    STATE_SIDE,
    load_daily_eval_frame,
)


DEFAULT_STRATEGIES = [
    "confirmed_rebalance_daily",
    "confirmed_signal_exit_max10",
    "spy_fused_hold_5d",
    "relative_momentum_exit_max10",
    "trend_quality_hold_3d",
    "hybrid_markov_trend_hold_3d",
    "hybrid_markov_trend_exit_max10",
]


def parse_csv_ints(text: str) -> list[int]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return values


def parse_csv_strings(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def make_args(base: argparse.Namespace, window: int, regime_source: str, output_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        dataset=base.dataset,
        output_dir=str(output_dir),
        eval_start=base.eval_start,
        eval_end=base.eval_end,
        symbols=base.symbols,
        top_symbols_limit=base.top_symbols_limit,
        universe_rank_cache=base.universe_rank_cache,
        regime_window_days=int(window),
        bull_threshold=base.bull_threshold,
        bear_threshold=base.bear_threshold,
        min_transition_days=base.min_transition_days,
        laplace=base.laplace,
        forecast_horizon_days=base.forecast_horizon_days,
        min_signal=base.min_signal,
        max_positions=base.max_positions,
        portfolio_exposure=base.portfolio_exposure,
        signal_full_exposure=base.signal_full_exposure,
        roundtrip_cost=base.roundtrip_cost,
        min_close=base.min_close,
        max_abs_return=base.max_abs_return,
        adaptive_min_history_days=base.adaptive_min_history_days,
        regime_source=regime_source,
        transition_lookback_days=base.transition_lookback_days,
        transition_halflife_days=base.transition_halflife_days,
    )


def regime_distribution(daily: pd.DataFrame, signals: pd.DataFrame, source: str) -> dict[str, float]:
    """Return the active state distribution actually used by generated signals."""
    if signals.empty or "current_state" not in signals.columns:
        states = pd.Series(dtype=int)
    else:
        states = signals["current_state"].astype(int)
    valid = states[states >= 0]
    total = max(int(len(valid)), 1)
    raw_column = "adaptive_state" if source == "adaptive" else "manual_state"
    raw_states = daily[raw_column].astype(int)
    raw_valid = raw_states[raw_states >= 0]
    return {
        "bear_pct": float((valid == STATE_BEAR).sum() / total),
        "sideways_pct": float((valid == STATE_SIDE).sum() / total),
        "bull_pct": float((valid == STATE_BULL).sum() / total),
        "valid_regime_rows": int(len(valid)),
        "raw_valid_regime_rows": int(len(raw_valid)),
    }


def plot_sweep(results: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    docs = Path("docs")
    docs.mkdir(exist_ok=True)
    top_by_best_alpha = (
        results.groupby("strategy")["alpha_return"].max().sort_values(ascending=False).head(8).index.tolist()
    )
    plot_df = results[results["strategy"].isin(top_by_best_alpha)].copy()

    fig = plt.figure(figsize=(16, 11), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1.5, 1.1])
    ax_return = fig.add_subplot(gs[0, 0])
    ax_alpha = fig.add_subplot(gs[0, 1])
    ax_dd = fig.add_subplot(gs[1, 0])
    ax_heat = fig.add_subplot(gs[1, 1])
    ax_best = fig.add_subplot(gs[2, :])

    for (regime_source, strategy), group in plot_df.groupby(["regime_source", "strategy"], sort=False):
        label = f"{strategy} ({regime_source})"
        group = group.sort_values("regime_window_days")
        ax_return.plot(group["regime_window_days"], group["total_return"] * 100, marker="o", linewidth=1.5, label=label)
        ax_alpha.plot(group["regime_window_days"], group["alpha_return"] * 100, marker="o", linewidth=1.5, label=label)
        ax_dd.plot(group["regime_window_days"], group["max_drawdown"] * 100, marker="o", linewidth=1.5, label=label)

    for ax, title, ylabel in [
        (ax_return, "Total Return By Regime Window", "Return (%)"),
        (ax_alpha, "Alpha Versus SPY", "Alpha (%)"),
        (ax_dd, "Max Drawdown", "Drawdown (%)"),
    ]:
        ax.axhline(0.0, color="#777", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("Regime lookback window (trading days)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)

    ax_return.legend(frameon=False, fontsize=7, ncol=2)

    best = (
        results.sort_values("alpha_return", ascending=False)
        .groupby(["regime_source", "regime_window_days"], sort=False)
        .head(1)
        .copy()
    )
    pivot = best.pivot(index="regime_source", columns="regime_window_days", values="alpha_return").sort_index()
    im = ax_heat.imshow(pivot.fillna(0.0).to_numpy() * 100, aspect="auto", cmap="RdYlGn", vmin=-15, vmax=15)
    ax_heat.set_title("Best Strategy Alpha Per Window (%)")
    ax_heat.set_yticks(np.arange(len(pivot.index)))
    ax_heat.set_yticklabels(pivot.index)
    ax_heat.set_xticks(np.arange(len(pivot.columns)))
    ax_heat.set_xticklabels(pivot.columns)
    ax_heat.set_xlabel("Regime lookback window")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                ax_heat.text(j, i, f"{val*100:+.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax_heat, shrink=0.85)

    best_all = results.sort_values("alpha_return", ascending=True).tail(12)
    labels = [
        f"{r.strategy}\n{r.regime_source}, {int(r.regime_window_days)}d"
        for r in best_all.itertuples(index=False)
    ]
    y = np.arange(len(best_all))
    ax_best.barh(y, best_all["alpha_return"] * 100, color="#43aa8b")
    ax_best.axvline(0.0, color="#777", linewidth=0.8)
    ax_best.set_yticks(y)
    ax_best.set_yticklabels(labels, fontsize=8)
    ax_best.set_xlabel("Alpha vs SPY (%)")
    ax_best.set_title("Top Window/Strategy Combinations")
    ax_best.grid(axis="x", alpha=0.25)

    for path in [
        output_dir / "markov_regime_window_sweep.png",
        docs / "markov_regime_window_sweep.png",
    ]:
        fig.savefig(path, dpi=160)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    requested_strategies = set(parse_csv_strings(args.strategies))
    selected_strategies = [s for s in strategies() if s.name in requested_strategies]
    if not selected_strategies:
        raise ValueError("none of the requested strategies matched evaluate_markov_holding_periods.strategies()")

    rows: list[dict] = []
    config_rows: list[dict] = []
    windows = parse_csv_ints(args.regime_windows)
    regime_sources = parse_csv_strings(args.regime_sources)
    for regime_source in regime_sources:
        for window in windows:
            run_dir = output_dir / f"{regime_source}_window_{window:03d}"
            config = default_markov_config(make_args(args, window, regime_source, run_dir))
            frame, daily, signals = load_daily_eval_frame(config)
            frame = add_cross_sectional_scores(frame, daily)
            dist = regime_distribution(daily, signals, regime_source)
            spy_return = np.nan
            print(
                f"[sweep] source={regime_source} window={window} "
                f"rows={len(frame)} signals={len(signals)} regimes={dist}",
                flush=True,
            )
            for strategy in selected_strategies:
                summary, _, _, _ = simulate_holding(frame, config, strategy)
                row = {
                    **summary,
                    "regime_source": regime_source,
                    "regime_window_days": int(window),
                    "frame_rows": int(len(frame)),
                    "daily_rows": int(len(daily)),
                    "signal_rows": int(len(signals)),
                    **dist,
                }
                rows.append(row)
                spy_return = summary["spy_total_return"]
                print(
                    f"  {strategy.name}: return={summary['total_return']:.2%} "
                    f"alpha={summary['alpha_return']:.2%} dd={summary['max_drawdown']:.2%}",
                    flush=True,
                )
            config_rows.append(
                {
                    "regime_source": regime_source,
                    "regime_window_days": int(window),
                    "spy_total_return": float(spy_return) if np.isfinite(spy_return) else None,
                    "config": asdict(config),
                    **dist,
                }
            )

    results = pd.DataFrame(rows).sort_values(["alpha_return"], ascending=False)
    configs = pd.DataFrame(config_rows)
    results.to_csv(output_dir / "window_sweep_results.csv", index=False)
    configs.to_json(output_dir / "window_sweep_configs.json", orient="records", indent=2)
    summary = {
        "dataset": args.dataset,
        "eval_start": args.eval_start,
        "eval_end": args.eval_end,
        "regime_windows": windows,
        "regime_sources": regime_sources,
        "strategies": [s.name for s in selected_strategies],
        "top_results": results.head(20).to_dict(orient="records"),
        "warning": "research_only_regime_window_sweep_not_financial_advice",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    plot_sweep(results, output_dir)
    print(json.dumps(summary, indent=2, default=str), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(Path.home() / ".cache/trading-autoresearch"))
    parser.add_argument("--output-dir", default="checkpoints/transformer_15m/markov_regime_window_sweep")
    parser.add_argument("--eval-start", default="2026-01-01")
    parser.add_argument("--eval-end", default="2026-05-22")
    parser.add_argument("--regime-windows", default="5,10,20,40,60,90,120")
    parser.add_argument("--regime-sources", default="adaptive,fixed")
    parser.add_argument("--strategies", default=",".join(DEFAULT_STRATEGIES))
    parser.add_argument("--symbols", default="")
    parser.add_argument("--top-symbols-limit", type=int, default=0)
    parser.add_argument("--universe-rank-cache", default="checkpoints/transformer_15m/top_volume_valuation_universe.csv")
    parser.add_argument("--bull-threshold", type=float, default=0.05)
    parser.add_argument("--bear-threshold", type=float, default=-0.05)
    parser.add_argument("--min-transition-days", type=int, default=80)
    parser.add_argument("--laplace", type=float, default=1.0)
    parser.add_argument("--forecast-horizon-days", type=int, default=1)
    parser.add_argument("--min-signal", type=float, default=0.05)
    parser.add_argument("--max-positions", type=int, default=3)
    parser.add_argument("--portfolio-exposure", type=float, default=1.0)
    parser.add_argument("--signal-full-exposure", type=float, default=0.35)
    parser.add_argument("--roundtrip-cost", type=float, default=0.0008)
    parser.add_argument("--min-close", type=float, default=5.0)
    parser.add_argument("--max-abs-return", type=float, default=0.08)
    parser.add_argument("--adaptive-min-history-days", type=int, default=80)
    parser.add_argument("--transition-lookback-days", type=int, default=126)
    parser.add_argument("--transition-halflife-days", type=float, default=42.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
