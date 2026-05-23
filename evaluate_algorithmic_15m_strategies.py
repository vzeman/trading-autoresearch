"""Evaluate pure algorithmic 15-minute trading rules without neural models."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


STRATEGIES = [
    "momentum_breakout",
    "trend_pullback",
    "spy_relative_strength",
    "mean_reversion",
    "algo_vote",
    "consensus",
    "vwap_trend_reclaim",
    "opening_range_breakout",
    "volatility_breakout",
    "liquidity_momentum",
    "pullback_continuation",
    "adaptive_consensus",
]


@dataclass(frozen=True)
class Config:
    dataset: str
    output_dir: str
    symbols: str
    universe_rank_cache: str
    top_symbols_limit: int
    eval_start: str
    eval_end: str
    roundtrip_cost: float
    max_positions: int
    min_score_quantile: float
    min_score: float
    max_abs_return: float
    min_close: float
    portfolio_exposure: float
    symbol_cooldown_loss: float
    symbol_cooldown_intervals: int
    symbol_daily_cap: int
    daily_loss_stop: float
    spy_momentum_window: int
    spy_momentum_min_return: float
    strategy_momentum_window: int
    strategy_momentum_min_return: float
    filter_falling_stocks: bool
    falling_filter_min_signals: int
    falling_max_ret_4: float
    falling_max_ret_16: float
    falling_max_trend_slope_8: float
    falling_max_trend_slope_16: float
    falling_max_ma8_dist: float
    falling_max_ma26_dist: float
    falling_max_algo_momentum_vote: float


class PortfolioState:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.day = None
        self.day_return = 0.0
        self.symbol_blocked_until: dict[str, int] = {}
        self.symbol_day_counts: dict[str, int] = {}
        self.recent_portfolio_returns: list[float] = []
        self.recent_spy_returns: list[float] = []

    def begin_interval(self, ts: pd.Timestamp) -> None:
        day = ts.date()
        if day != self.day:
            self.day = day
            self.day_return = 0.0
            self.symbol_day_counts = {}

    def can_trade(self) -> bool:
        if self.config.daily_loss_stop > -0.99 and self.day_return <= self.config.daily_loss_stop:
            return False
        if self.config.spy_momentum_window > 0:
            if len(self.recent_spy_returns) < self.config.spy_momentum_window:
                return False
            ret = float(np.prod(1.0 + np.asarray(self.recent_spy_returns[-self.config.spy_momentum_window:])) - 1.0)
            if ret <= self.config.spy_momentum_min_return:
                return False
        if self.config.strategy_momentum_window > 0:
            if len(self.recent_portfolio_returns) < self.config.strategy_momentum_window:
                return False
            ret = float(np.prod(1.0 + np.asarray(self.recent_portfolio_returns[-self.config.strategy_momentum_window:])) - 1.0)
            if ret <= self.config.strategy_momentum_min_return:
                return False
        return True

    def filter_candidates(self, candidates: pd.DataFrame, interval_index: int) -> pd.DataFrame:
        active = candidates.copy()
        if self.config.symbol_cooldown_intervals > 0 and not active.empty:
            mask = active["symbol"].astype(str).map(
                lambda s: self.symbol_blocked_until.get(s, -1) <= interval_index
            ).astype(bool)
            active = active.loc[mask].copy()
        if self.config.symbol_daily_cap > 0 and not active.empty:
            mask = active["symbol"].astype(str).map(
                lambda s: self.symbol_day_counts.get(s, 0) < self.config.symbol_daily_cap
            ).astype(bool)
            active = active.loc[mask].copy()
        return active

    def record(self, selected: pd.DataFrame, portfolio_return: float, spy_return: float, interval_index: int) -> None:
        self.day_return = (1.0 + self.day_return) * (1.0 + portfolio_return) - 1.0
        self.recent_portfolio_returns.append(float(portfolio_return))
        self.recent_spy_returns.append(float(spy_return))
        for _, row in selected.iterrows():
            symbol = str(row["symbol"])
            self.symbol_day_counts[symbol] = self.symbol_day_counts.get(symbol, 0) + 1
            if (
                self.config.symbol_cooldown_intervals > 0
                and float(row["future_return"]) <= self.config.symbol_cooldown_loss
            ):
                self.symbol_blocked_until[symbol] = interval_index + self.config.symbol_cooldown_intervals


def max_drawdown(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    return float((values / values.cummax() - 1.0).min())


def zscore(series: pd.Series) -> pd.Series:
    s = series.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    std = float(s.std())
    if std < 1e-12:
        return pd.Series(0.0, index=series.index)
    return ((s - float(s.mean())) / std).clip(-5.0, 5.0)


def rank01(series: pd.Series) -> pd.Series:
    return series.astype(float).rank(pct=True).fillna(0.5)


def falling_signal_count(df: pd.DataFrame, config: Config) -> pd.Series:
    thresholds = [
        ("ret_4", config.falling_max_ret_4),
        ("ret_16", config.falling_max_ret_16),
        ("trend_slope_8", config.falling_max_trend_slope_8),
        ("trend_slope_16", config.falling_max_trend_slope_16),
        ("ma8_dist", config.falling_max_ma8_dist),
        ("ma26_dist", config.falling_max_ma26_dist),
        ("algo_momentum_vote", config.falling_max_algo_momentum_vote),
    ]
    signals = [df[col].astype(float).lt(threshold) for col, threshold in thresholds if col in df.columns]
    if not signals:
        return pd.Series(0, index=df.index, dtype=int)
    return pd.concat(signals, axis=1).sum(axis=1).astype(int)


def add_rule_scores(group: pd.DataFrame) -> pd.DataFrame:
    g = group.copy()
    momentum = (
        0.30 * zscore(g["ret_16"])
        + 0.20 * zscore(g["ret_4"])
        + 0.20 * zscore(g["trend_slope_16"])
        + 0.15 * zscore(g["macd_hist"])
        + 0.15 * zscore(g["volume_confirmed_ret_16"])
    )
    breakout = (
        0.30 * zscore(g["breakout_26"])
        + 0.25 * zscore(g["donchian_pos_20"])
        + 0.20 * zscore(g["donchian_breakout_20"])
        + 0.15 * zscore(g["range_compression_16_52"] * g["ret_4"])
        + 0.10 * zscore(g["volume_z_26"])
    )
    trend_pullback = (
        0.35 * zscore(g["algo_trend_quality"])
        + 0.25 * zscore(g["trend_slope_26"])
        + 0.20 * zscore(g["pullback_16"])
        - 0.10 * zscore(g["bb_z_20"].clip(lower=0.0))
        + 0.10 * zscore(g["ema_8_21_cross"] + g["ema_21_55_cross"])
    )
    spy_relative = (
        0.35 * zscore(g["rel_spy_16"])
        + 0.20 * zscore(g["rel_spy_trend_slope_16"])
        + 0.20 * zscore(g["rel_mkt_16"])
        + 0.15 * zscore(g["rel_spy_trend_accel_8_26"])
        + 0.10 * zscore(g["mkt_pct_positive_16_slope"])
    )
    mean_reversion = (
        -0.30 * zscore(g["mean_reversion_z_20"])
        -0.20 * zscore(g["bb_z_20"])
        + 0.20 * zscore(0.50 - g["stoch_k_14"])
        + 0.15 * zscore(g["lower_wick_ratio"])
        + 0.15 * zscore(g["trend_slope_26"])
    )
    algo_vote = (
        0.35 * zscore(g["algo_momentum_vote"])
        + 0.25 * zscore(g["algo_breakout_vote"])
        + 0.20 * zscore(g["algo_trend_quality"])
        - 0.10 * zscore(g["algo_mean_reversion_vote"].abs())
        + 0.10 * zscore(g["rel_spy_16"])
    )

    g["score_momentum_breakout"] = 0.65 * momentum + 0.35 * breakout
    g["score_trend_pullback"] = trend_pullback
    g["score_spy_relative_strength"] = spy_relative
    g["score_mean_reversion"] = mean_reversion
    g["score_algo_vote"] = algo_vote
    g["score_consensus"] = (
        rank01(g["score_momentum_breakout"])
        + rank01(g["score_trend_pullback"])
        + rank01(g["score_spy_relative_strength"])
        + rank01(g["score_algo_vote"])
        + 0.50 * rank01(g["score_mean_reversion"])
    )
    vwap_trend = (
        0.30 * zscore(-g["vwap_dist_26"].abs())
        + 0.25 * zscore(g["trend_slope_26"])
        + 0.20 * zscore(g["rel_spy_16"])
        + 0.15 * zscore(g["ema_8_21_cross"] + g["ema_21_55_cross"])
        + 0.10 * zscore(g["volume_z_26"])
    )
    opening_range = (
        0.35 * zscore(g["orb_breakout"])
        + 0.20 * zscore(g["from_day_open"])
        + 0.20 * zscore(g["volume_confirmed_ret_4"])
        + 0.15 * zscore(g["rel_spy_4"])
        - 0.10 * zscore(g["atr_14_pct"])
    )
    volatility_breakout = (
        0.30 * zscore(g["donchian_breakout_20"])
        + 0.25 * zscore(g["breakout_52"])
        + 0.20 * zscore(g["range_compression_16_52"])
        + 0.15 * zscore(g["atr_14_pct"])
        + 0.10 * zscore(g["volume_z_26"])
    )
    liquidity_momentum = (
        0.35 * zscore(g["volume_confirmed_ret_16"])
        + 0.25 * zscore(g["rel_spy_16"])
        + 0.20 * zscore(g["mkt_pct_positive_16"])
        + 0.10 * zscore(g["mkt_breadth_accel"])
        + 0.10 * zscore(g["algo_momentum_vote"])
    )
    pullback_continuation = (
        0.30 * zscore(g["trend_slope_52"])
        + 0.20 * zscore(g["pullback_26"])
        + 0.20 * zscore(g["lower_wick_ratio"])
        - 0.15 * zscore(g["bb_z_20"].clip(lower=0.0))
        + 0.15 * zscore(g["rel_spy_16"])
    )

    g["score_vwap_trend_reclaim"] = vwap_trend
    g["score_opening_range_breakout"] = opening_range
    g["score_volatility_breakout"] = volatility_breakout
    g["score_liquidity_momentum"] = liquidity_momentum
    g["score_pullback_continuation"] = pullback_continuation
    g["score_adaptive_consensus"] = (
        rank01(g["score_vwap_trend_reclaim"])
        + rank01(g["score_opening_range_breakout"])
        + rank01(g["score_volatility_breakout"])
        + rank01(g["score_liquidity_momentum"])
        + rank01(g["score_pullback_continuation"])
        + 0.50 * rank01(g["score_spy_relative_strength"])
    )
    return g


def base_strategy_mask(df: pd.DataFrame, strategy: str) -> pd.Series:
    if strategy == "momentum_breakout":
        return (
            (df["ret_4"] > 0.0)
            & (df["ret_16"] > 0.0)
            & (df["macd_hist"] > -0.002)
            & (df["algo_momentum_vote"] > -0.25)
        )
    if strategy == "trend_pullback":
        return (
            (df["trend_slope_26"] > 0.0)
            & (df["ma26_dist"] > -0.02)
            & (df["ret_4"] > -0.012)
            & (df["rsi_14"].between(0.35, 0.75))
        )
    if strategy == "spy_relative_strength":
        return (
            (df["rel_spy_16"] > 0.0)
            & (df["rel_mkt_16"] > -0.005)
            & (df["trend_slope_16"] > -0.001)
        )
    if strategy == "mean_reversion":
        return (
            (df["bb_z_20"] < -0.5)
            & (df["stoch_k_14"] < 0.45)
            & (df["trend_slope_52"] > -0.0015)
            & (df["ret_16"] > -0.04)
        )
    if strategy == "algo_vote":
        return (
            (df["algo_momentum_vote"] > 0.0)
            & (df["algo_breakout_vote"] > -0.25)
            & (df["algo_trend_quality"] > -0.25)
        )
    if strategy == "consensus":
        return (
            (df["ret_16"] > -0.025)
            & (df["ma26_dist"] > -0.035)
            & (df["algo_trend_quality"] > -0.50)
        )
    if strategy == "vwap_trend_reclaim":
        return (
            (df["trend_slope_26"] > 0.0)
            & (df["vwap_dist_26"] > -0.006)
            & (df["ma26_dist"] > -0.012)
            & (df["ret_4"] > -0.006)
            & (df["rsi_14"].between(0.42, 0.72))
        )
    if strategy == "opening_range_breakout":
        return (
            (df["orb_breakout"] > 0.0)
            & (df["from_day_open"] > 0.002)
            & (df["volume_z_26"] > -0.25)
            & (df["rel_spy_4"] > -0.004)
        )
    if strategy == "volatility_breakout":
        return (
            (df["donchian_breakout_20"] > 0.0)
            & (df["range_compression_16_52"] > -0.25)
            & (df["atr_14_pct"].between(0.001, 0.025))
            & (df["volume_z_26"] > 0.0)
        )
    if strategy == "liquidity_momentum":
        return (
            (df["volume_confirmed_ret_16"] > 0.0)
            & (df["rel_spy_16"] > -0.002)
            & (df["mkt_pct_positive_16"] > 0.45)
            & (df["algo_momentum_vote"] > -0.25)
        )
    if strategy == "pullback_continuation":
        return (
            (df["trend_slope_52"] > 0.0)
            & (df["ret_16"].between(-0.025, 0.010))
            & (df["ma26_dist"] > -0.02)
            & (df["rsi_14"].between(0.35, 0.62))
            & (df["lower_wick_ratio"] > 0.15)
        )
    if strategy == "adaptive_consensus":
        return (
            (df["ret_26"] > -0.035)
            & (df["ma26_dist"] > -0.035)
            & (df["rel_spy_16"] > -0.015)
            & (df["atr_14_pct"] < 0.035)
            & (df["mkt_pct_positive_16"] > 0.35)
        )
    raise ValueError(f"unknown strategy {strategy}")


def selected_symbols(config: Config) -> list[str]:
    explicit = [s.strip().upper() for s in config.symbols.split(",") if s.strip()]
    if explicit:
        return explicit
    if config.top_symbols_limit <= 0:
        return []
    rank_path = Path(config.universe_rank_cache)
    if rank_path.exists():
        ranked = pd.read_csv(rank_path)
        if "symbol" in ranked.columns:
            return ranked["symbol"].astype(str).str.upper().head(config.top_symbols_limit).tolist()
    return []


def add_intraday_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["symbol", "timestamp"]).copy()
    df["session"] = df["timestamp"].dt.date.astype(str)
    grouped = df.groupby(["symbol", "session"], sort=False)
    df["bar_in_day"] = grouped.cumcount()
    day_open = grouped["open"].transform("first").replace(0.0, np.nan)
    df["from_day_open"] = df["close"] / day_open - 1.0
    first_two_high = df[df["bar_in_day"] < 2].groupby(["symbol", "session"])["high"].transform("max")
    first_two_low = df[df["bar_in_day"] < 2].groupby(["symbol", "session"])["low"].transform("min")
    df["orb_high"] = first_two_high.reindex(df.index).groupby([df["symbol"], df["session"]]).transform("first")
    df["orb_low"] = first_two_low.reindex(df.index).groupby([df["symbol"], df["session"]]).transform("first")
    df["orb_breakout"] = np.where(
        (df["bar_in_day"] >= 2) & df["orb_high"].notna(),
        df["close"] / df["orb_high"].replace(0.0, np.nan) - 1.0,
        0.0,
    )
    df["orb_breakdown"] = np.where(
        (df["bar_in_day"] >= 2) & df["orb_low"].notna(),
        df["close"] / df["orb_low"].replace(0.0, np.nan) - 1.0,
        0.0,
    )
    return df.fillna(0.0)


def load_eval_frame(config: Config) -> pd.DataFrame:
    df = pd.read_parquet(config.dataset)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    symbols = selected_symbols(config)
    if symbols:
        df = df[df["symbol"].astype(str).str.upper().isin(symbols)].copy()
    df = df[
        (df["timestamp"] >= pd.Timestamp(config.eval_start, tz="UTC"))
        & (df["timestamp"] <= pd.Timestamp(config.eval_end, tz="UTC"))
        & df["close"].astype(float).ge(config.min_close)
        & df["future_return"].astype(float).between(-config.max_abs_return, config.max_abs_return)
        & df["future_spy_return"].astype(float).between(-config.max_abs_return, config.max_abs_return)
    ].copy()
    df = add_intraday_features(df)
    df["falling_signals"] = falling_signal_count(df, config)
    scored = [add_rule_scores(group) for _, group in df.groupby("timestamp", sort=True)]
    return pd.concat(scored, ignore_index=True).sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def evaluate_strategy(df: pd.DataFrame, strategy: str, config: Config) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    score_col = f"score_{strategy}"
    equity = 50_000.0
    spy_equity = 50_000.0
    state = PortfolioState(config)
    curve_rows = []
    trade_rows = []
    timestamps = sorted(df["timestamp"].unique())
    for i, ts_value in enumerate(timestamps, start=1):
        ts = pd.Timestamp(ts_value)
        group = df[df["timestamp"] == ts_value].copy()
        state.begin_interval(ts)
        spy_ret = float(group["future_spy_return"].dropna().iloc[0]) if not group.empty else 0.0
        selected = pd.DataFrame()
        if state.can_trade() and not group.empty:
            candidates = group[base_strategy_mask(group, strategy)].copy()
            if config.filter_falling_stocks and not candidates.empty:
                candidates = candidates[candidates["falling_signals"] < config.falling_filter_min_signals].copy()
            if not candidates.empty:
                threshold = candidates[score_col].quantile(config.min_score_quantile)
                threshold = max(float(threshold), config.min_score)
                candidates = candidates[candidates[score_col] >= threshold].copy()
            candidates = state.filter_candidates(candidates, i)
            selected = candidates.sort_values(score_col, ascending=False).head(config.max_positions)

        if selected.empty:
            portfolio_ret = 0.0
            symbols = []
        else:
            portfolio_ret = float(config.portfolio_exposure * (selected["future_return"].astype(float).mean() - config.roundtrip_cost))
            symbols = selected["symbol"].astype(str).tolist()
            for _, row in selected.iterrows():
                trade_rows.append(
                    {
                        "timestamp": str(ts),
                        "model": strategy,
                        "symbol": row["symbol"],
                        "score": float(row[score_col]),
                        "falling_signals": int(row["falling_signals"]),
                        "future_return": float(row["future_return"]),
                        "future_spy_return": spy_ret,
                        "future_alpha": float(row["future_return"]) - spy_ret,
                    }
                )

        equity *= 1.0 + portfolio_ret
        spy_equity *= 1.0 + spy_ret
        state.record(selected, portfolio_ret, spy_ret, i)
        curve_rows.append(
            {
                "timestamp": str(ts),
                "model": strategy,
                "equity": equity,
                "spy_equity": spy_equity,
                "portfolio_return": portfolio_ret,
                "spy_return": spy_ret,
                "symbols": ",".join(symbols),
            }
        )

    curve = pd.DataFrame(curve_rows)
    trades = pd.DataFrame(trade_rows)
    trade_returns = trades["future_return"].astype(float).to_numpy() if not trades.empty else np.array([])
    summary = {
        "model": strategy,
        "decision_intervals": int(len(curve)),
        "active_intervals": int((curve["symbols"].astype(str) != "").sum()) if not curve.empty else 0,
        "trades": int(len(trades)),
        "final_equity": float(equity),
        "spy_final_equity": float(spy_equity),
        "total_return": float(equity / 50_000.0 - 1.0),
        "spy_total_return": float(spy_equity / 50_000.0 - 1.0),
        "active_alpha_return": float(equity / 50_000.0 - spy_equity / 50_000.0),
        "max_drawdown": max_drawdown(curve["equity"].astype(float)) if not curve.empty else 0.0,
        "trade_profit_rate": float((trade_returns > 0.0).mean()) if len(trade_returns) else 0.0,
        "mean_trade_return": float(trade_returns.mean()) if len(trade_returns) else 0.0,
    }
    return summary, curve, trades


def plot_results(curves: pd.DataFrame, leaderboard: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 8))
    for name, group in curves.groupby("model", sort=False):
        group = group.copy()
        group["timestamp"] = pd.to_datetime(group["timestamp"])
        ax.plot(group["timestamp"], group["equity"], label=name, linewidth=1.3)
    first = curves.groupby("timestamp", sort=True)["spy_equity"].first().reset_index()
    first["timestamp"] = pd.to_datetime(first["timestamp"])
    ax.plot(first["timestamp"], first["spy_equity"], "--", color="black", label="SPY", linewidth=1.8)
    ax.set_title("Pure algorithmic 15-minute strategies vs SPY")
    ax.set_ylabel("Equity ($)")
    ax.grid(alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 0.78, 1.0))
    fig.savefig(output_dir / "algo_15m_equity.png", dpi=140)
    Path("docs").mkdir(exist_ok=True)
    fig.savefig("docs/algo_15m_equity.png", dpi=140)
    plt.close(fig)

    board = leaderboard.sort_values("active_alpha_return")
    fig, ax = plt.subplots(figsize=(12, max(5, 0.45 * len(board) + 2)))
    y = np.arange(len(board))
    ax.barh(y - 0.18, board["total_return"] * 100, height=0.34, label="Strategy")
    ax.barh(y + 0.18, board["active_alpha_return"] * 100, height=0.34, label="Alpha vs SPY")
    spy = float(board["spy_total_return"].dropna().iloc[0]) * 100
    ax.axvline(spy, linestyle="--", color="black", label=f"SPY {spy:.2f}%")
    ax.axvline(0.0, color="gray", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(board["model"])
    ax.set_xlabel("Return (%)")
    ax.set_title("Pure algorithmic strategy leaderboard")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "algo_15m_leaderboard.png", dpi=140)
    fig.savefig("docs/algo_15m_leaderboard.png", dpi=140)
    plt.close(fig)


def run(config: Config) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_eval_frame(config)
    summaries = []
    curves = []
    trades = []
    for strategy in STRATEGIES:
        summary, curve, trade_df = evaluate_strategy(df, strategy, config)
        summaries.append(summary)
        curves.append(curve)
        trades.append(trade_df)
        print(f"[algo15m] {strategy} return={summary['total_return']:.2%} spy={summary['spy_total_return']:.2%}", flush=True)

    all_curves = pd.concat(curves, ignore_index=True)
    all_trades = pd.concat([t for t in trades if not t.empty], ignore_index=True) if any(not t.empty for t in trades) else pd.DataFrame()
    leaderboard = pd.DataFrame(summaries).sort_values("active_alpha_return", ascending=False)
    result = {
        "config": asdict(config),
        "rows": int(len(df)),
        "symbols": int(df["symbol"].nunique()),
        "strategies": summaries,
        "warning": "research_only_pure_algorithmic_15m_baseline",
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, default=str))
    leaderboard.to_csv(output_dir / "leaderboard.csv", index=False)
    all_curves.to_csv(output_dir / "equity_curves.csv", index=False)
    all_trades.to_csv(output_dir / "trades.csv", index=False)
    plot_results(all_curves, leaderboard, output_dir)
    print(json.dumps(result, indent=2, default=str), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="checkpoints/transformer_15m/shared_15m_40sym_algo.parquet")
    parser.add_argument("--output-dir", default="checkpoints/transformer_15m/algo_15m_baseline_2026")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--universe-rank-cache", default="checkpoints/transformer_15m/top_volume_valuation_universe.csv")
    parser.add_argument("--top-symbols-limit", type=int, default=0)
    parser.add_argument("--eval-start", default="2026-01-01")
    parser.add_argument("--eval-end", default="2026-05-16")
    parser.add_argument("--roundtrip-cost", type=float, default=0.0008)
    parser.add_argument("--max-positions", type=int, default=3)
    parser.add_argument("--min-score-quantile", type=float, default=0.85)
    parser.add_argument("--min-score", type=float, default=-1e9)
    parser.add_argument("--max-abs-return", type=float, default=0.08)
    parser.add_argument("--min-close", type=float, default=5.0)
    parser.add_argument("--portfolio-exposure", type=float, default=1.0)
    parser.add_argument("--symbol-cooldown-loss", type=float, default=-0.02)
    parser.add_argument("--symbol-cooldown-intervals", type=int, default=78)
    parser.add_argument("--symbol-daily-cap", type=int, default=0)
    parser.add_argument("--daily-loss-stop", type=float, default=-1.0)
    parser.add_argument("--spy-momentum-window", type=int, default=0)
    parser.add_argument("--spy-momentum-min-return", type=float, default=0.0)
    parser.add_argument("--strategy-momentum-window", type=int, default=0)
    parser.add_argument("--strategy-momentum-min-return", type=float, default=0.0)
    parser.add_argument("--filter-falling-stocks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--falling-filter-min-signals", type=int, default=5)
    parser.add_argument("--falling-max-ret-4", type=float, default=-0.004)
    parser.add_argument("--falling-max-ret-16", type=float, default=-0.008)
    parser.add_argument("--falling-max-trend-slope-8", type=float, default=-0.001)
    parser.add_argument("--falling-max-trend-slope-16", type=float, default=-0.0006)
    parser.add_argument("--falling-max-ma8-dist", type=float, default=-0.004)
    parser.add_argument("--falling-max-ma26-dist", type=float, default=-0.008)
    parser.add_argument("--falling-max-algo-momentum-vote", type=float, default=-0.5)
    run(Config(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
