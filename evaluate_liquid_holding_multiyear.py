"""Evaluate holding-period strategies on high-volume stocks across year folds."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from evaluate_markov_holding_periods import (
    add_cross_sectional_scores,
    default_markov_config,
    simulate_holding,
    strategies,
)
from evaluate_markov_regime_quant_strategy import load_daily_eval_frame


DEFAULT_STRATEGIES = [
    "trend_quality_hold_3d",
    "trend_quality_volume_shape_hold_3d",
    "trend_quality_avoid_failed_gap_hold_3d",
    "trend_quality_avoid_failed_gap_volume_shape_hold_3d",
    "hybrid_markov_trend_hold_3d",
    "relative_momentum_exit_max10",
    "relative_momentum_volume_shape_exit_max10",
]


def csv_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def csv_strings(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def config_args(base: argparse.Namespace, year: int, output_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        dataset=base.dataset,
        output_dir=str(output_dir / f"fold_{year}"),
        eval_start=f"{year}-01-01",
        eval_end=f"{year}-12-31" if year < base.final_year else base.final_eval_end,
        symbols=base.symbols,
        top_symbols_limit=base.top_symbols_limit,
        universe_rank_cache=base.universe_rank_cache,
        regime_window_days=base.regime_window_days,
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
        regime_source=base.regime_source,
        transition_lookback_days=base.transition_lookback_days,
        transition_halflife_days=base.transition_halflife_days,
    )


def apply_liquidity_filter(frame: pd.DataFrame, min_median_dollar_volume: float) -> pd.DataFrame:
    if min_median_dollar_volume <= 0:
        return frame
    return frame[
        frame["median_dollar_volume_20_prev"].fillna(0.0).astype(float).ge(float(min_median_dollar_volume))
    ].copy()


def monthly_returns_by_fold(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (year, strategy), group in curves.groupby(["year", "strategy"], sort=False):
        g = group.copy()
        g["timestamp"] = pd.to_datetime(g["timestamp"], utc=True)
        g["month"] = g["timestamp"].dt.tz_convert(None).dt.to_period("M").astype(str)
        prev_equity = float(g["equity"].iloc[0])
        prev_spy = float(g["spy_equity"].iloc[0])
        for month, month_df in g.groupby("month", sort=True):
            end_equity = float(month_df["equity"].iloc[-1])
            end_spy = float(month_df["spy_equity"].iloc[-1])
            rows.append(
                {
                    "year": int(year),
                    "strategy": strategy,
                    "month": month,
                    "return": end_equity / prev_equity - 1.0,
                    "spy_return": end_spy / prev_spy - 1.0,
                    "pnl": end_equity - prev_equity,
                    "turnover": float(month_df["turnover"].sum()),
                    "entries": int(month_df["entries"].astype(str).ne("").sum()),
                }
            )
            prev_equity = end_equity
            prev_spy = end_spy
    return pd.DataFrame(rows)


def plot_multiyear(results: pd.DataFrame, curves: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    docs = Path("docs")
    docs.mkdir(exist_ok=True)
    results = results.copy()
    curves = curves.copy()
    curves["timestamp"] = pd.to_datetime(curves["timestamp"], utc=True)
    curves["fold_label"] = curves["year"].astype(str) + " " + curves["strategy"]

    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.6, 1.3, 1.4])
    ax_equity = fig.add_subplot(gs[0, :])
    ax_fold = fig.add_subplot(gs[1, 0])
    ax_dd = fig.add_subplot(gs[1, 1])
    ax_alpha = fig.add_subplot(gs[2, :])

    top_strats = results.groupby("strategy")["alpha_return"].mean().sort_values(ascending=False).head(5).index.tolist()
    for (year, strategy), group in curves[curves["strategy"].isin(top_strats)].groupby(["year", "strategy"], sort=True):
        label = f"{year} {strategy}"
        ax_equity.plot(group["timestamp"], group["equity"], linewidth=1.3, label=label)
    spy = curves.groupby("timestamp", sort=True)["spy_equity"].first().reset_index()
    ax_equity.plot(spy["timestamp"], spy["spy_equity"], "--", color="black", linewidth=1.8, label="SPY")
    ax_equity.set_title("High-Volume Strategy Equity By Fold")
    ax_equity.set_ylabel("Equity ($)")
    ax_equity.grid(alpha=0.25)
    ax_equity.legend(frameon=False, fontsize=7, ncol=3)

    fold = results.pivot(index="year", columns="strategy", values="total_return")
    fold = fold[[c for c in top_strats if c in fold.columns]]
    fold.plot(kind="bar", ax=ax_fold)
    ax_fold.set_title("Fold Returns")
    ax_fold.set_ylabel("Return")
    ax_fold.axhline(0.0, color="#777", linewidth=0.8)
    ax_fold.grid(axis="y", alpha=0.25)
    ax_fold.legend(frameon=False, fontsize=7)

    dd = results.pivot(index="year", columns="strategy", values="max_drawdown")
    dd = dd[[c for c in top_strats if c in dd.columns]]
    dd.plot(kind="bar", ax=ax_dd, color=plt.cm.Reds_r([0.25, 0.35, 0.45, 0.55, 0.65][: len(dd.columns)]))
    ax_dd.set_title("Fold Max Drawdown")
    ax_dd.set_ylabel("Drawdown")
    ax_dd.axhline(0.0, color="#777", linewidth=0.8)
    ax_dd.grid(axis="y", alpha=0.25)
    ax_dd.legend(frameon=False, fontsize=7)

    board = results.sort_values("alpha_return", ascending=True)
    labels = [f"{int(r.year)} {r.strategy}" for r in board.itertuples(index=False)]
    ax_alpha.barh(range(len(board)), board["alpha_return"] * 100, color="#43aa8b")
    ax_alpha.set_yticks(range(len(board)))
    ax_alpha.set_yticklabels(labels, fontsize=7)
    ax_alpha.axvline(0.0, color="#777", linewidth=0.8)
    ax_alpha.set_xlabel("Alpha vs SPY (%)")
    ax_alpha.set_title("Every Fold/Strategy Alpha")
    ax_alpha.grid(axis="x", alpha=0.25)

    docs_name = (
        "liquid_multiyear_top10_strategy_comparison.png"
        if "top10" in output_dir.name.lower()
        else "liquid_multiyear_strategy_comparison.png"
    )
    for path in [output_dir / "liquid_multiyear_strategy_comparison.png", docs / docs_name]:
        fig.savefig(path, dpi=160)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = {s.name: s for s in strategies() if s.name in set(csv_strings(args.strategies))}
    if not selected:
        raise ValueError("No matching strategies selected")

    result_rows = []
    curve_frames = []
    trade_frames = []
    for year in csv_ints(args.years):
        cargs = config_args(args, year, output_dir)
        config = default_markov_config(cargs)
        frame, daily, signals = load_daily_eval_frame(config)
        frame = add_cross_sectional_scores(frame, daily)
        before_rows = len(frame)
        before_symbols = int(frame["symbol"].nunique()) if not frame.empty else 0
        frame = apply_liquidity_filter(frame, args.min_median_dollar_volume)
        after_symbols = int(frame["symbol"].nunique()) if not frame.empty else 0
        print(
            f"[fold {year}] rows {before_rows}->{len(frame)} symbols {before_symbols}->{after_symbols} "
            f"min_dv={args.min_median_dollar_volume:,.0f}",
            flush=True,
        )
        for strategy in selected.values():
            summary, curve, trades, _ = simulate_holding(frame, config, strategy)
            summary.update(
                {
                    "year": int(year),
                    "eval_start": cargs.eval_start,
                    "eval_end": cargs.eval_end,
                    "liquid_rows": int(len(frame)),
                    "liquid_symbols": after_symbols,
                    "pre_filter_rows": before_rows,
                    "pre_filter_symbols": before_symbols,
                    "min_median_dollar_volume": float(args.min_median_dollar_volume),
                }
            )
            result_rows.append(summary)
            curve = curve.copy()
            curve["year"] = int(year)
            curve_frames.append(curve)
            if not trades.empty:
                trades = trades.copy()
                trades["year"] = int(year)
                trade_frames.append(trades)
            print(
                f"  {strategy.name}: return={summary['total_return']:.2%} "
                f"spy={summary['spy_total_return']:.2%} alpha={summary['alpha_return']:.2%} "
                f"dd={summary['max_drawdown']:.2%}",
                flush=True,
            )

    results = pd.DataFrame(result_rows).sort_values(["year", "alpha_return"], ascending=[True, False])
    curves = pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame()
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    monthly = monthly_returns_by_fold(curves) if not curves.empty else pd.DataFrame()
    results.to_csv(output_dir / "liquid_multiyear_results.csv", index=False)
    curves.to_csv(output_dir / "liquid_multiyear_equity_curves.csv", index=False)
    trades.to_csv(output_dir / "liquid_multiyear_trades.csv", index=False)
    monthly.to_csv(output_dir / "liquid_multiyear_monthly_returns.csv", index=False)
    summary = {
        "config": vars(args),
        "markov_config_template": asdict(default_markov_config(config_args(args, csv_ints(args.years)[0], output_dir))),
        "top_results": results.sort_values("alpha_return", ascending=False).head(20).to_dict(orient="records"),
        "warning": "research_only_liquid_multiyear_simulation_not_financial_advice",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    plot_multiyear(results, curves, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="checkpoints/transformer_15m/shared_15m_40sym_algo.parquet")
    parser.add_argument("--output-dir", default="checkpoints/transformer_15m/liquid_holding_multiyear")
    parser.add_argument("--years", default="2023,2024,2025,2026")
    parser.add_argument("--final-year", type=int, default=2026)
    parser.add_argument("--final-eval-end", default="2026-05-14")
    parser.add_argument("--strategies", default=",".join(DEFAULT_STRATEGIES))
    parser.add_argument("--min-median-dollar-volume", type=float, default=20_000_000.0)
    parser.add_argument("--symbols", default="")
    parser.add_argument("--top-symbols-limit", type=int, default=0)
    parser.add_argument("--universe-rank-cache", default="checkpoints/transformer_15m/top_volume_valuation_universe.csv")
    parser.add_argument("--regime-window-days", type=int, default=20)
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
    parser.add_argument("--regime-source", choices=["fixed", "adaptive"], default="adaptive")
    parser.add_argument("--transition-lookback-days", type=int, default=126)
    parser.add_argument("--transition-halflife-days", type=float, default=42.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
