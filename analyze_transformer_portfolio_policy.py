"""Analyze portfolio-management overlays for 15-minute transformer experiments.

This script does not retrain models. It replays an existing experiment's
``trades.csv`` and ``equity_curves.csv`` to diagnose whether losses come from
model selection, transaction costs, concentration, or weak portfolio rules.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Policy:
    name: str
    max_pos: int = 3
    exposure: float = 1.0
    cost: float = 0.0008
    daily_stop: float | None = None
    cooldown_loss: float | None = None
    cooldown_intervals: int = 0
    spy_gate: int = 0
    strategy_gate: int = 0
    symbol_daily_cap: int | None = None


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    return float((equity / equity.cummax() - 1.0).min())


def load_model_name(output_dir: Path, requested: str) -> str:
    if requested:
        return requested
    leaderboard = pd.read_csv(output_dir / "leaderboard.csv")
    return str(leaderboard.sort_values("active_alpha_return", ascending=False).iloc[0]["model"])


def replay_policy(
    policy: Policy,
    trades: pd.DataFrame,
    curves: pd.DataFrame,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    all_ts = list(curves["timestamp"])
    ts_index = {ts: i for i, ts in enumerate(all_ts)}
    spy = dict(zip(curves["timestamp"], curves["spy_return"]))
    by_ts = {
        ts: group.sort_values("pred_score", ascending=False).copy()
        for ts, group in trades.groupby("timestamp", sort=True)
    }

    equity = 50_000.0
    spy_equity = 50_000.0
    day = None
    day_ret = 0.0
    blocked_until: dict[str, int] = {}
    symbol_day_counts: dict[str, int] = {}
    recent_returns: list[float] = []
    curve_rows = []
    trade_rows = []

    for ts in all_ts:
        current_day = ts.date()
        if current_day != day:
            day = current_day
            day_ret = 0.0
            symbol_day_counts = {}

        allow_trade = True
        if policy.daily_stop is not None and day_ret <= policy.daily_stop:
            allow_trade = False
        if policy.spy_gate > 0:
            i = ts_index[ts]
            start = max(0, i - policy.spy_gate)
            trailing_spy = np.prod([1.0 + spy[all_ts[j]] for j in range(start, i)]) - 1.0
            if i < policy.spy_gate or trailing_spy <= 0.0:
                allow_trade = False
        if policy.strategy_gate > 0 and len(recent_returns) >= policy.strategy_gate:
            if float(np.mean(recent_returns[-policy.strategy_gate:])) <= 0.0:
                allow_trade = False

        selected = pd.DataFrame()
        if allow_trade and ts in by_ts:
            candidates = by_ts[ts].copy()
            if policy.cooldown_loss is not None:
                candidates = candidates[
                    candidates["symbol"].map(lambda s: blocked_until.get(s, -1) <= ts_index[ts])
                ]
            if policy.symbol_daily_cap is not None:
                candidates = candidates[
                    candidates["symbol"].map(lambda s: symbol_day_counts.get(s, 0) < policy.symbol_daily_cap)
                ]
            selected = candidates.head(policy.max_pos)

        if selected.empty:
            portfolio_return = 0.0
            symbols = ""
        else:
            gross = float(selected["future_return"].astype(float).mean())
            portfolio_return = policy.exposure * (gross - policy.cost)
            symbols = ",".join(selected["symbol"].astype(str))
            for _, row in selected.iterrows():
                symbol = str(row["symbol"])
                symbol_day_counts[symbol] = symbol_day_counts.get(symbol, 0) + 1
                trade_rows.append(
                    {
                        "timestamp": ts,
                        "symbol": symbol,
                        "future_return": float(row["future_return"]),
                        "future_alpha": float(row["future_alpha"]),
                        "pred_score": float(row["pred_score"]),
                    }
                )
                if policy.cooldown_loss is not None and float(row["future_return"]) <= policy.cooldown_loss:
                    blocked_until[symbol] = ts_index[ts] + policy.cooldown_intervals

        equity *= 1.0 + portfolio_return
        spy_equity *= 1.0 + spy[ts]
        day_ret = (1.0 + day_ret) * (1.0 + portfolio_return) - 1.0
        recent_returns.append(portfolio_return)
        curve_rows.append(
            {
                "timestamp": ts,
                "equity": equity,
                "spy_equity": spy_equity,
                "portfolio_return": portfolio_return,
                "spy_return": spy[ts],
                "symbols": symbols,
            }
        )

    curve = pd.DataFrame(curve_rows)
    replayed_trades = pd.DataFrame(trade_rows)
    trade_returns = replayed_trades["future_return"].astype(float) if not replayed_trades.empty else pd.Series(dtype=float)
    active = curve["symbols"].astype(str) != ""
    summary = {
        "policy": policy.name,
        "total_return": float(equity / 50_000.0 - 1.0),
        "spy_total_return": float(spy_equity / 50_000.0 - 1.0),
        "active_alpha_return": float(equity / 50_000.0 - spy_equity / 50_000.0),
        "max_drawdown": max_drawdown(curve["equity"]),
        "active_intervals": int(active.sum()),
        "trades": int(len(replayed_trades)),
        "trade_profit_rate": float((trade_returns > 0.0).mean()) if len(trade_returns) else 0.0,
        "mean_trade_gross_return": float(trade_returns.mean()) if len(trade_returns) else 0.0,
        "mean_active_interval_net_return": float(curve.loc[active, "portfolio_return"].mean()) if active.any() else 0.0,
    }
    return summary, curve, replayed_trades


def build_policies(base_cost: float) -> list[Policy]:
    policies: list[Policy] = [Policy("baseline", cost=base_cost)]
    policies += [Policy(f"top{n}", max_pos=n, cost=base_cost) for n in [1, 2, 3]]
    policies += [Policy(f"exposure_{x}", exposure=x, cost=base_cost) for x in [0.25, 0.5, 0.75, 1.0]]
    policies += [Policy(f"daily_stop_{s}", daily_stop=s, cost=base_cost) for s in [-0.005, -0.01, -0.015, -0.02, -0.03]]
    for loss in [-0.01, -0.015, -0.02]:
        for cooldown in [26, 78]:
            policies.append(
                Policy(
                    f"symbol_cool_{loss}_{cooldown}",
                    cooldown_loss=loss,
                    cooldown_intervals=cooldown,
                    cost=base_cost,
                )
            )
    policies += [Policy(f"spy_gate_{n}", spy_gate=n, cost=base_cost) for n in [8, 16, 26, 52, 104]]
    policies += [Policy(f"strategy_gate_{n}", strategy_gate=n, cost=base_cost) for n in [10, 25, 50, 100]]
    policies += [Policy(f"symbol_daily_cap_{n}", symbol_daily_cap=n, cost=base_cost) for n in [1, 2, 3, 5, 10]]
    for max_pos in [1, 2, 3]:
        for stop in [None, -0.01, -0.02]:
            for cooldown in [None, -0.015]:
                for spy_gate in [0, 26, 52]:
                    name = f"combo_top{max_pos}_stop{stop}_cool{cooldown}_spy{spy_gate}"
                    policies.append(
                        Policy(
                            name,
                            max_pos=max_pos,
                            daily_stop=stop,
                            cooldown_loss=cooldown,
                            cooldown_intervals=26,
                            spy_gate=spy_gate,
                            cost=base_cost,
                        )
                    )
    return policies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--cost", type=float, default=0.0008)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    model = load_model_name(output_dir, args.model)
    trades = pd.read_csv(output_dir / "trades.csv", parse_dates=["timestamp"])
    curves = pd.read_csv(output_dir / "equity_curves.csv", parse_dates=["timestamp"])
    trades = trades[trades["model"] == model].copy()
    curves = curves[curves["model"] == model].copy().sort_values("timestamp")
    if trades.empty or curves.empty:
        raise RuntimeError(f"no trades/curves found for model {model!r}")

    summaries = []
    curves_by_policy = {}
    trades_by_policy = {}
    for policy in build_policies(args.cost):
        summary, curve, policy_trades = replay_policy(policy, trades, curves)
        summaries.append(summary)
        curves_by_policy[policy.name] = curve
        trades_by_policy[policy.name] = policy_trades

    summary_df = pd.DataFrame(summaries).sort_values(["active_alpha_return", "total_return"], ascending=False)
    summary_df.to_csv(output_dir / "portfolio_policy_analysis.csv", index=False)

    best_policy = str(summary_df.iloc[0]["policy"])
    curves_by_policy[best_policy].to_csv(output_dir / "portfolio_policy_analysis_best_curve.csv", index=False)
    trades_by_policy[best_policy].to_csv(output_dir / "portfolio_policy_analysis_best_trades.csv", index=False)

    cost_rows = []
    for bps in [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15]:
        cost = bps / 10_000.0
        baseline, _, _ = replay_policy(Policy("baseline", cost=cost), trades, curves)
        cooldown, _, _ = replay_policy(
            Policy("symbol_cool_-0.02_78", cost=cost, cooldown_loss=-0.02, cooldown_intervals=78),
            trades,
            curves,
        )
        cost_rows.append(
            {
                "cost_bps": bps,
                "baseline_return": baseline["total_return"],
                "cooldown_return": cooldown["total_return"],
                "spy_return": baseline["spy_total_return"],
            }
        )
    pd.DataFrame(cost_rows).to_csv(output_dir / "portfolio_policy_cost_sensitivity.csv", index=False)

    print(f"model={model}")
    print(summary_df.head(20).to_string(index=False))
    print(f"best_policy={best_policy}")


if __name__ == "__main__":
    main()
