"""Plot world-model portfolio diagnostics from local result JSON files."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


RESULTS = [
    ("q80 hybrid retrain", "retrained_allocator_intraday120_q80_hybrid_3ep.json"),
    ("q80 learned gate", "retrained_allocator_intraday120_q80_hybrid_gate_3ep.json"),
    ("q90 hybrid retrain", "retrained_allocator_intraday120_q90_hybrid_3ep.json"),
    ("q80 cash objective", "retrained_allocator_intraday120_q80_cash_3ep.json"),
    ("q80 score sizing", "retrained_allocator_intraday120_q80_hybrid_sized_3ep.json"),
    ("q80 regime gate", "retrained_allocator_intraday120_q80_hybrid_regime_3ep.json"),
    ("q80 market context", "retrained_allocator_intraday120_q80_hybrid_marketctx_3ep.json"),
]


def load_result(results_dir: Path, filename: str) -> dict:
    path = results_dir / filename
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def pct(v: float) -> float:
    return 100.0 * float(v)


def save_summary(rows: list[dict], output: Path) -> None:
    fields = [
        "name",
        "active_groups",
        "coverage",
        "mean_active_return",
        "return_with_cash",
        "active_profit_rate",
        "active_beat_spy_rate",
        "mean_alpha_vs_spy",
    ]
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fields})


def plot_equity_curve(result: dict, output: Path) -> None:
    folds = result["folds"]
    strategy = [1.0]
    selected_spy = [1.0]
    active = [1.0]
    labels = ["start"]
    for fold in folds:
        applied = fold["applied"]
        cash_return = float(applied["portfolio_mean_return_with_cash"])
        active_return = float(applied["mean_portfolio_return"])
        coverage = float(applied["coverage"])
        active_alpha = float(applied["mean_future_alpha_vs_spy"])
        implied_spy_cash = cash_return - active_alpha * coverage
        strategy.append(strategy[-1] * (1.0 + cash_return))
        selected_spy.append(selected_spy[-1] * (1.0 + implied_spy_cash))
        active.append(active[-1] * (1.0 + active_return))
        labels.append(f"fold {fold['fold']}")

    plt.figure(figsize=(12, 7))
    x = range(len(strategy))
    plt.plot(x, strategy, marker="o", linewidth=2.8, label="winning strategy, cash when no trade")
    plt.plot(x, selected_spy, marker="o", linewidth=2.0, linestyle="--", label="implied SPY on selected groups, cash otherwise")
    plt.plot(x, active, marker="o", linewidth=2.0, linestyle=":", label="active selected trades only")
    plt.axhline(1.0, color="#555555", linewidth=1.0)
    plt.xticks(list(x), labels)
    plt.ylabel("Growth of $1.00")
    plt.title("World-model winning policy: walk-forward simulated equity")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def plot_fold_returns(result: dict, output: Path) -> None:
    folds = result["folds"]
    names = [f"fold {f['fold']}" for f in folds]
    cash_returns = [pct(f["applied"]["portfolio_mean_return_with_cash"]) for f in folds]
    active_returns = [pct(f["applied"]["mean_portfolio_return"]) for f in folds]
    coverage = [pct(f["applied"]["coverage"]) for f in folds]

    fig, ax1 = plt.subplots(figsize=(12, 7))
    x = range(len(folds))
    width = 0.36
    ax1.bar([i - width / 2 for i in x], cash_returns, width=width, label="return with cash")
    ax1.bar([i + width / 2 for i in x], active_returns, width=width, label="active trade return")
    ax1.axhline(0, color="#333333", linewidth=1.0)
    ax1.set_ylabel("Return per fold (%)")
    ax1.set_xticks(list(x), names)
    ax1.grid(True, axis="y", alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(list(x), coverage, color="#111111", marker="D", linewidth=2.2, label="coverage")
    ax2.set_ylabel("Coverage (%)")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")
    plt.title("Winning policy fold behavior")
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def plot_experiment_bars(rows: list[dict], output: Path) -> None:
    rows = sorted(rows, key=lambda r: r["return_with_cash"], reverse=True)
    names = [r["name"] for r in rows]
    cash = [pct(r["return_with_cash"]) for r in rows]
    active = [pct(r["mean_active_return"]) for r in rows]
    x = range(len(rows))

    plt.figure(figsize=(13, 7))
    width = 0.38
    plt.bar([i - width / 2 for i in x], cash, width=width, label="return with cash")
    plt.bar([i + width / 2 for i in x], active, width=width, label="active return")
    plt.axhline(0, color="#333333", linewidth=1.0)
    plt.xticks(list(x), names, rotation=24, ha="right")
    plt.ylabel("Mean return (%)")
    plt.title("World-model allocator experiment comparison")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def plot_coverage_scatter(rows: list[dict], output: Path) -> None:
    plt.figure(figsize=(11, 7))
    for row in rows:
        x = pct(row["coverage"])
        y = pct(row["mean_active_return"])
        size = 110 + 900 * max(row["return_with_cash"], 0.0)
        plt.scatter(x, y, s=size, alpha=0.85)
        plt.annotate(row["name"], (x, y), textcoords="offset points", xytext=(7, 5), fontsize=9)
    plt.axhline(0, color="#333333", linewidth=1.0)
    plt.xlabel("Coverage: groups traded (%)")
    plt.ylabel("Mean active return (%)")
    plt.title("Coverage versus selected-trade quality")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def plot_trade_quality(rows: list[dict], output: Path) -> None:
    rows = sorted(rows, key=lambda r: r["active_beat_spy_rate"], reverse=True)
    names = [r["name"] for r in rows]
    profit = [pct(r["active_profit_rate"]) for r in rows]
    beat = [pct(r["active_beat_spy_rate"]) for r in rows]
    x = range(len(rows))
    width = 0.38
    plt.figure(figsize=(13, 7))
    plt.bar([i - width / 2 for i in x], profit, width=width, label="profit rate")
    plt.bar([i + width / 2 for i in x], beat, width=width, label="beat-SPY rate")
    plt.axhline(50, color="#333333", linewidth=1.0, linestyle="--")
    plt.xticks(list(x), names, rotation=24, ha="right")
    plt.ylabel("Active groups (%)")
    plt.title("Trade quality by allocator variant")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="checkpoints/world_model")
    parser.add_argument("--output-dir", default="docs/world_model_charts")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    loaded = {}
    for name, filename in RESULTS:
        data = load_result(results_dir, filename)
        agg = data["aggregate"]
        loaded[name] = data
        rows.append(
            {
                "name": name,
                "active_groups": int(agg["active_groups"]),
                "coverage": float(agg["coverage"]),
                "mean_active_return": float(agg["mean_portfolio_return"]),
                "return_with_cash": float(agg["portfolio_mean_return_with_cash"]),
                "active_profit_rate": float(agg["profit_rate"]),
                "active_beat_spy_rate": float(agg["beat_spy_rate"]),
                "mean_alpha_vs_spy": float(agg["mean_future_alpha_vs_spy"]),
            }
        )

    save_summary(rows, output_dir / "world_model_experiment_summary.csv")
    plot_equity_curve(loaded["q80 hybrid retrain"], output_dir / "winning_strategy_equity.png")
    plot_fold_returns(loaded["q80 hybrid retrain"], output_dir / "winning_strategy_fold_returns.png")
    plot_experiment_bars(rows, output_dir / "experiment_comparison.png")
    plot_coverage_scatter(rows, output_dir / "coverage_vs_return.png")
    plot_trade_quality(rows, output_dir / "trade_quality.png")

    print(f"Wrote {len(list(output_dir.glob('*')))} files to {output_dir}")


if __name__ == "__main__":
    main()
