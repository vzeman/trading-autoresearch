"""Quarterly walk-forward portfolio evaluation on the full top-100 cache.

The experiment uses the 100 SPY holding-weight constituents already downloaded
from Alpaca plus SPY as the benchmark/context symbol. Signals are computed in a
walk-forward way:

- Markov transition matrices for a target day use only earlier daily states.
- Cross-sectional strategy features are shifted to information known before the
  target day, except gap-at-open strategies which use the target day's opening
  gap and enter at the open.
- Each quarterly fold is evaluated out-of-sample after at least one full
  calendar year of prior data.

This is research output, not financial advice.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_markov_holding_periods import (
    START_EQUITY,
    add_cross_sectional_scores,
    default_markov_config,
    simulate_holding,
    strategies as holding_strategies,
)
from evaluate_markov_regime_quant_strategy import (
    build_daily_signals,
    daily_strategy_candidates,
    max_drawdown,
    manual_state,
    sharpe,
)


def load_top100_symbols(path: Path) -> list[str]:
    with path.open() as f:
        return [row["symbol"].strip().upper() for row in csv.DictReader(f) if row.get("symbol")]


def cache_path(cache_dir: Path, symbol: str) -> Path:
    return cache_dir / f"{symbol}_1m.parquet"


def build_daily_panel(symbols: list[str], cache_dir: Path, output: Path, refresh: bool) -> pd.DataFrame:
    if output.exists() and not refresh:
        daily = pd.read_parquet(output)
        daily["date"] = pd.to_datetime(daily["date"]).dt.date
        return daily

    frames: list[pd.DataFrame] = []
    for idx, symbol in enumerate(symbols, start=1):
        path = cache_path(cache_dir, symbol)
        if not path.exists():
            raise FileNotFoundError(f"missing cache for {symbol}: {path}")
        df = pd.read_parquet(path, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["symbol"] = symbol
        df["date"] = df["timestamp"].dt.date
        daily = (
            df.sort_values("timestamp")
            .groupby(["symbol", "date"], sort=True)
            .agg(
                open=("open", "first"),
                high=("high", "max"),
                low=("low", "min"),
                close=("close", "last"),
                volume=("volume", "sum"),
                first_ts=("timestamp", "first"),
                last_ts=("timestamp", "last"),
                minute_bars=("timestamp", "size"),
            )
            .reset_index()
        )
        frames.append(daily)
        print(f"[daily-panel] {idx:03d}/{len(symbols)} {symbol} days={len(daily):,}", flush=True)

    panel = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    panel["timestamp"] = pd.to_datetime(panel["date"].astype(str), utc=True)
    by_symbol = panel.groupby("symbol", sort=False)
    panel["prev_close"] = by_symbol["close"].shift(1)
    panel["gap_return"] = panel["open"].astype(float) / panel["prev_close"].replace(0.0, np.nan).astype(float) - 1.0
    panel["open_to_close_return"] = panel["close"].astype(float) / panel["open"].replace(0.0, np.nan).astype(float) - 1.0
    panel["intraday_range"] = panel["high"].astype(float) / panel["low"].replace(0.0, np.nan).astype(float) - 1.0
    panel["dollar_volume"] = panel["close"].astype(float) * panel["volume"].astype(float)
    panel["daily_return"] = by_symbol["close"].pct_change()
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output, index=False)
    return panel


def prepare_frame(daily: pd.DataFrame, config: object, eval_start: str, eval_end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = daily.copy()
    daily["symbol"] = daily["symbol"].astype(str).str.upper()
    daily["date"] = pd.to_datetime(daily["date"]).dt.date
    daily["timestamp"] = pd.to_datetime(daily["date"].astype(str), utc=True)
    daily = daily[daily["close"].astype(float).ge(config.min_close)].copy()
    daily["lookback_return"] = daily.groupby("symbol")["close"].pct_change(config.regime_window_days)
    daily["manual_state"] = manual_state(daily["lookback_return"], config)

    from evaluate_markov_regime_quant_strategy import add_adaptive_state

    daily = add_adaptive_state(daily, config)
    signals = build_daily_signals(daily, config)
    spy_signals = build_daily_signals(daily[daily["symbol"] == "SPY"].copy(), config)
    if not spy_signals.empty:
        spy_signals = spy_signals[["date", "markov_signal", "p_bull", "p_bear", "daily_return"]].rename(
            columns={
                "markov_signal": "spy_markov_signal",
                "p_bull": "spy_p_bull",
                "p_bear": "spy_p_bear",
                "daily_return": "spy_daily_return",
            }
        )
    else:
        spy_returns = daily[daily["symbol"] == "SPY"][["date", "daily_return"]].rename(
            columns={"daily_return": "spy_daily_return"}
        )
        spy_signals = spy_returns.assign(spy_markov_signal=0.0, spy_p_bull=0.0, spy_p_bear=0.0)

    frame = signals.merge(spy_signals, on="date", how="left")
    frame["timestamp"] = pd.to_datetime(frame["date"].astype(str), utc=True)
    start = pd.Timestamp(eval_start, tz="UTC")
    end = pd.Timestamp(eval_end, tz="UTC")
    frame = frame[
        frame["timestamp"].ge(start)
        & frame["timestamp"].le(end)
        & frame["daily_return"].astype(float).between(-config.max_abs_return, config.max_abs_return)
        & frame["spy_daily_return"].fillna(0.0).astype(float).between(-config.max_abs_return, config.max_abs_return)
    ].copy()
    frame = add_cross_sectional_scores(frame.sort_values(["date", "symbol"]).reset_index(drop=True), daily)
    return frame, signals


def quarter_folds(start: str, end: str) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    quarter_starts = pd.date_range(start_ts, end_ts, freq="QS", tz="UTC")
    folds = []
    for q_start in quarter_starts:
        q_end = min(q_start + pd.DateOffset(months=3) - pd.Timedelta(days=1), end_ts)
        if q_start <= end_ts:
            folds.append((f"{q_start.year}Q{q_start.quarter}", q_start, q_end))
    return folds


def summarize_curve(strategy: str, family: str, curve: pd.DataFrame, trades: pd.DataFrame | None = None) -> dict:
    if curve.empty:
        return {
            "family": family,
            "strategy": strategy,
            "decision_days": 0,
            "active_days": 0,
            "trades": 0,
            "total_return": 0.0,
            "spy_total_return": 0.0,
            "alpha_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "spy_sharpe": 0.0,
        }
    returns = curve["portfolio_return"].astype(float)
    spy_returns = curve["spy_return"].astype(float)
    equity = float(curve["equity"].iloc[-1])
    spy_equity = float(curve["spy_equity"].iloc[-1])
    active_col = "position_count" if "position_count" in curve.columns else "symbols"
    active_days = int((curve[active_col].astype(str) != "").sum()) if active_col == "symbols" else int((curve[active_col].astype(int) > 0).sum())
    return {
        "family": family,
        "strategy": strategy,
        "decision_days": int(len(curve)),
        "active_days": active_days,
        "trades": int(len(trades)) if trades is not None else 0,
        "total_return": float(equity / START_EQUITY - 1.0),
        "spy_total_return": float(spy_equity / START_EQUITY - 1.0),
        "alpha_return": float(equity / START_EQUITY - spy_equity / START_EQUITY),
        "max_drawdown": max_drawdown(curve["equity"].astype(float)),
        "sharpe": sharpe(returns, bars_per_year=252),
        "spy_sharpe": sharpe(spy_returns, bars_per_year=252),
    }


def simulate_markov_daily(frame: pd.DataFrame, config: object, variant: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    equity = START_EQUITY
    spy_equity = START_EQUITY
    curve_rows: list[dict] = []
    trade_rows: list[dict] = []
    for date_value, group in frame.groupby("date", sort=True):
        ts = pd.Timestamp(str(date_value), tz="UTC")
        spy_values = group["spy_daily_return"].dropna()
        spy_ret = float(spy_values.iloc[0]) if not spy_values.empty else 0.0
        selected = daily_strategy_candidates(group, config, variant)
        if selected.empty:
            portfolio_ret = 0.0
            symbols: list[str] = []
        else:
            signal = selected["markov_signal"].astype(float).clip(lower=0.0)
            raw_weights = (signal / max(config.signal_full_exposure, 1e-9)).clip(upper=1.0)
            if raw_weights.sum() > 0:
                weights = raw_weights / raw_weights.sum() * min(config.portfolio_exposure, float(raw_weights.sum()))
            else:
                weights = pd.Series(1.0 / len(selected), index=selected.index)
            daily_ret = selected["daily_return"].astype(float)
            portfolio_ret = float((weights * (daily_ret - config.roundtrip_cost)).sum())
            symbols = selected["symbol"].astype(str).tolist()
            for idx, row in selected.iterrows():
                trade_rows.append(
                    {
                        "timestamp": str(ts),
                        "strategy": f"markov_daily_{variant}",
                        "symbol": row["symbol"],
                        "weight": float(weights.loc[idx]),
                        "score": float(row["score"]),
                        "markov_signal": float(row["markov_signal"]),
                        "daily_return": float(row["daily_return"]),
                        "spy_daily_return": spy_ret,
                    }
                )
        equity *= 1.0 + portfolio_ret
        spy_equity *= 1.0 + spy_ret
        curve_rows.append(
            {
                "timestamp": str(ts),
                "strategy": f"markov_daily_{variant}",
                "equity": equity,
                "spy_equity": spy_equity,
                "portfolio_return": portfolio_ret,
                "spy_return": spy_ret,
                "positions": ",".join(symbols),
                "position_count": len(symbols),
                "entries": ",".join(symbols),
                "exits": "",
            }
        )
    curve = pd.DataFrame(curve_rows)
    trades = pd.DataFrame(trade_rows)
    return summarize_curve(f"markov_daily_{variant}", "markov_daily", curve, trades), curve, trades


def quarter_return_rows(continuous: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy, group in continuous.groupby("strategy", sort=False):
        g = group.copy()
        g["timestamp"] = pd.to_datetime(g["timestamp"], utc=True)
        g["quarter"] = g["timestamp"].dt.to_period("Q").astype(str)
        prev_equity = START_EQUITY
        prev_spy = START_EQUITY
        for quarter, qdf in g.groupby("quarter", sort=True):
            end_equity = float(qdf["equity"].iloc[-1])
            end_spy = float(qdf["spy_equity"].iloc[-1])
            rows.append(
                {
                    "strategy": strategy,
                    "quarter": quarter,
                    "return": end_equity / prev_equity - 1.0,
                    "spy_return": end_spy / prev_spy - 1.0,
                    "alpha_return": end_equity / prev_equity - end_spy / prev_spy,
                    "end_equity": end_equity,
                    "spy_end_equity": end_spy,
                }
            )
            prev_equity = end_equity
            prev_spy = end_spy
    return pd.DataFrame(rows)


def plot_outputs(continuous: pd.DataFrame, fold_results: pd.DataFrame, quarter_returns: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    docs = Path("docs")
    docs.mkdir(exist_ok=True)

    leaderboard = (
        fold_results.groupby(["family", "strategy"], as_index=False)
        .agg(
            avg_alpha=("alpha_return", "mean"),
            avg_return=("total_return", "mean"),
            worst_alpha=("alpha_return", "min"),
            beat_spy=("alpha_return", lambda s: int((s > 0).sum())),
            folds=("alpha_return", "size"),
        )
        .sort_values(["avg_alpha", "worst_alpha"], ascending=False)
    )
    top = leaderboard.head(10)["strategy"].tolist()

    curves = continuous[continuous["strategy"].isin(top)].copy()
    curves["timestamp"] = pd.to_datetime(curves["timestamp"], utc=True)
    fig = plt.figure(figsize=(17, 13), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[2.0, 1.2, 1.8])
    ax_eq = fig.add_subplot(gs[0, :])
    ax_alpha = fig.add_subplot(gs[1, 0])
    ax_worst = fig.add_subplot(gs[1, 1])
    ax_heat = fig.add_subplot(gs[2, :])

    for strategy, group in curves.groupby("strategy", sort=False):
        ax_eq.plot(group["timestamp"], group["equity"], label=strategy, linewidth=1.35)
    spy = curves.groupby("timestamp", sort=True)["spy_equity"].first().reset_index()
    ax_eq.plot(spy["timestamp"], spy["spy_equity"], "--", color="black", label="SPY", linewidth=2.0)
    ax_eq.set_title("Top-100 Walk-Forward Continuous Portfolio Equity")
    ax_eq.set_ylabel("Equity ($)")
    ax_eq.grid(alpha=0.25)
    ax_eq.legend(ncol=2, frameon=False, fontsize=8)

    board = leaderboard.head(15).sort_values("avg_alpha")
    y = np.arange(len(board))
    ax_alpha.barh(y, board["avg_alpha"] * 100, color="#2a9d8f")
    ax_alpha.axvline(0, color="#777", linewidth=0.8)
    ax_alpha.set_yticks(y)
    ax_alpha.set_yticklabels(board["strategy"], fontsize=8)
    ax_alpha.set_xlabel("Avg quarterly alpha (%)")
    ax_alpha.set_title("Average Fold Alpha")
    ax_alpha.grid(axis="x", alpha=0.25)

    ax_worst.barh(y, board["worst_alpha"] * 100, color="#e76f51")
    ax_worst.axvline(0, color="#777", linewidth=0.8)
    ax_worst.set_yticks(y)
    ax_worst.set_yticklabels([])
    ax_worst.set_xlabel("Worst quarterly alpha (%)")
    ax_worst.set_title("Worst Fold Alpha")
    ax_worst.grid(axis="x", alpha=0.25)

    heat = quarter_returns[quarter_returns["strategy"].isin(top)].pivot(index="strategy", columns="quarter", values="alpha_return")
    heat = heat.reindex(top)
    im = ax_heat.imshow(heat.fillna(0.0).to_numpy() * 100, aspect="auto", cmap="RdYlGn", vmin=-15, vmax=15)
    ax_heat.set_yticks(np.arange(len(heat.index)))
    ax_heat.set_yticklabels(heat.index, fontsize=8)
    ax_heat.set_xticks(np.arange(len(heat.columns)))
    ax_heat.set_xticklabels(heat.columns, rotation=45, ha="right", fontsize=8)
    ax_heat.set_title("Continuous Quarterly Alpha Vs SPY (%)")
    for i in range(len(heat.index)):
        for j in range(len(heat.columns)):
            val = heat.iloc[i, j]
            if pd.notna(val):
                ax_heat.text(j, i, f"{val*100:+.1f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax_heat, shrink=0.8)

    chart = output_dir / "top100_quarterly_walk_forward.png"
    fig.savefig(chart, dpi=160)
    fig.savefig(docs / "top100_quarterly_walk_forward.png", dpi=160)
    plt.close(fig)

    leaderboard.to_csv(output_dir / "strategy_quarterly_leaderboard.csv", index=False)


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    top100 = load_top100_symbols(Path(args.symbols_csv))
    symbols = top100 + (["SPY"] if "SPY" not in top100 else [])
    daily = build_daily_panel(symbols, Path(args.cache_dir), Path(args.daily_panel), args.refresh_daily)

    config_args = argparse.Namespace(
        dataset=args.daily_panel,
        output_dir=args.output_dir,
        eval_start=args.eval_start,
        eval_end=args.eval_end,
        symbols=",".join(top100),
        top_symbols_limit=0,
        universe_rank_cache=args.symbols_csv,
        regime_window_days=args.regime_window_days,
        bull_threshold=args.bull_threshold,
        bear_threshold=args.bear_threshold,
        min_transition_days=args.min_transition_days,
        laplace=args.laplace,
        forecast_horizon_days=args.forecast_horizon_days,
        min_signal=args.min_signal,
        max_positions=args.max_positions,
        portfolio_exposure=args.portfolio_exposure,
        signal_full_exposure=args.signal_full_exposure,
        roundtrip_cost=args.roundtrip_cost,
        min_close=args.min_close,
        max_abs_return=args.max_abs_return,
        adaptive_min_history_days=args.adaptive_min_history_days,
        regime_source=args.regime_source,
        transition_lookback_days=args.transition_lookback_days,
        transition_halflife_days=args.transition_halflife_days,
    )
    config = default_markov_config(config_args)
    frame, signals = prepare_frame(daily, config, args.eval_start, args.eval_end)
    print(
        f"[quarterly] frame rows={len(frame):,} symbols={frame['symbol'].nunique()} "
        f"dates={frame['date'].nunique()} strategies={len(holding_strategies()) + 5}",
        flush=True,
    )

    fold_summaries: list[dict] = []
    fold_curves: list[pd.DataFrame] = []
    fold_trades: list[pd.DataFrame] = []
    folds = quarter_folds(args.eval_start, args.eval_end)
    hold_strategies = holding_strategies()
    markov_variants = ["manual", "confirmed", "spy_gated", "confirmed_spy_gated", "algo_fused_spy_gated"]

    for fold_label, q_start, q_end in folds:
        q_frame = frame[frame["timestamp"].between(q_start, q_end)].copy()
        if q_frame.empty:
            continue
        train_end = q_start - pd.Timedelta(days=1)
        print(f"[quarterly] {fold_label} train<= {train_end.date()} eval={q_start.date()}->{q_end.date()} rows={len(q_frame):,}", flush=True)
        for strategy in hold_strategies:
            summary, curve, trades, _ = simulate_holding(q_frame, config, strategy)
            summary.update(
                {
                    "quarter": fold_label,
                    "train_end": str(train_end.date()),
                    "eval_start": str(q_start.date()),
                    "eval_end": str(q_end.date()),
                    "family": "holding",
                }
            )
            fold_summaries.append(summary)
            curve["quarter"] = fold_label
            curve["family"] = "holding"
            fold_curves.append(curve)
            if not trades.empty:
                trades["quarter"] = fold_label
                trades["family"] = "holding"
                fold_trades.append(trades)
        for variant in markov_variants:
            summary, curve, trades = simulate_markov_daily(q_frame, config, variant)
            summary.update(
                {
                    "quarter": fold_label,
                    "train_end": str(train_end.date()),
                    "eval_start": str(q_start.date()),
                    "eval_end": str(q_end.date()),
                }
            )
            fold_summaries.append(summary)
            curve["quarter"] = fold_label
            curve["family"] = "markov_daily"
            fold_curves.append(curve)
            if not trades.empty:
                trades["quarter"] = fold_label
                trades["family"] = "markov_daily"
                fold_trades.append(trades)

    continuous_curves: list[pd.DataFrame] = []
    continuous_trades: list[pd.DataFrame] = []
    continuous_summaries: list[dict] = []
    for strategy in hold_strategies:
        summary, curve, trades, _ = simulate_holding(frame, config, strategy)
        summary.update({"family": "holding"})
        continuous_summaries.append(summary)
        curve["family"] = "holding"
        continuous_curves.append(curve)
        if not trades.empty:
            trades["family"] = "holding"
            continuous_trades.append(trades)
    for variant in markov_variants:
        summary, curve, trades = simulate_markov_daily(frame, config, variant)
        continuous_summaries.append(summary)
        curve["family"] = "markov_daily"
        continuous_curves.append(curve)
        if not trades.empty:
            trades["family"] = "markov_daily"
            continuous_trades.append(trades)

    fold_results = pd.DataFrame(fold_summaries)
    all_fold_curves = pd.concat(fold_curves, ignore_index=True)
    all_fold_trades = pd.concat(fold_trades, ignore_index=True) if fold_trades else pd.DataFrame()
    continuous = pd.concat(continuous_curves, ignore_index=True)
    continuous_trade_df = pd.concat(continuous_trades, ignore_index=True) if continuous_trades else pd.DataFrame()
    continuous_results = pd.DataFrame(continuous_summaries).sort_values("alpha_return", ascending=False)
    quarter_returns = quarter_return_rows(continuous)

    fold_results.to_csv(output_dir / "quarterly_fold_results.csv", index=False)
    all_fold_curves.to_csv(output_dir / "quarterly_fold_equity_curves.csv", index=False)
    all_fold_trades.to_csv(output_dir / "quarterly_fold_trades.csv", index=False)
    continuous.to_csv(output_dir / "continuous_equity_curves.csv", index=False)
    continuous_trade_df.to_csv(output_dir / "continuous_trades.csv", index=False)
    continuous_results.to_csv(output_dir / "continuous_leaderboard.csv", index=False)
    quarter_returns.to_csv(output_dir / "continuous_quarter_returns.csv", index=False)
    plot_outputs(continuous, fold_results, quarter_returns, output_dir)

    leaderboard = pd.read_csv(output_dir / "strategy_quarterly_leaderboard.csv")
    summary = {
        "warning": "research_only_not_financial_advice",
        "symbols_traded": len(top100),
        "context_symbols": ["SPY"],
        "daily_panel": str(args.daily_panel),
        "eval_start": args.eval_start,
        "eval_end": args.eval_end,
        "quarter_folds": len(folds),
        "strategies": len(hold_strategies) + len(markov_variants),
        "frame_rows": int(len(frame)),
        "signal_rows": int(len(signals)),
        "config": asdict(config),
        "top_quarterly_strategies": leaderboard.head(10).to_dict(orient="records"),
        "top_continuous_strategies": continuous_results.head(10).to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols-csv", default="checkpoints/data_download/sp500_top100_spy_holdings_2026-05-21.csv")
    parser.add_argument("--cache-dir", default=str(Path.home() / ".cache/trading-autoresearch"))
    parser.add_argument("--daily-panel", default="checkpoints/data_download/sp500_top100_plus_spy_daily_ohlcv.parquet")
    parser.add_argument("--output-dir", default="checkpoints/transformer_15m/top100_quarterly_walk_forward")
    parser.add_argument("--eval-start", default="2018-01-01")
    parser.add_argument("--eval-end", default="2026-05-22")
    parser.add_argument("--refresh-daily", action="store_true")
    parser.add_argument("--regime-window-days", type=int, default=60)
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
