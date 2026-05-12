"""Plot locked one-year sequential tradable world-model evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prepare import CACHE_DIR


def load_curve(path: Path) -> tuple[pd.DataFrame, dict]:
    payload = json.loads(path.read_text())
    curve = pd.DataFrame(payload["sequential_portfolio"]["equity_curve"])
    curve["timestamp"] = pd.to_datetime(curve["timestamp"], utc=True)
    curve = curve.sort_values("timestamp")
    return curve, payload


def spy_curve(start: pd.Timestamp, end: pd.Timestamp, starting_equity: float) -> pd.DataFrame:
    spy = pd.read_parquet(CACHE_DIR / "SPY_1m.parquet", columns=["timestamp", "close"]).sort_values("timestamp")
    spy["timestamp"] = pd.to_datetime(spy["timestamp"], utc=True)
    window = spy[(spy["timestamp"] >= start) & (spy["timestamp"] <= end)].copy()
    if window.empty:
        raise RuntimeError("no SPY rows for requested chart window")
    window["equity"] = starting_equity * window["close"].astype(float) / float(window["close"].iloc[0])
    return window[["timestamp", "equity"]]


def drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


def load_optional_curve(path: Path | None) -> tuple[pd.DataFrame, dict] | None:
    if path is None or not path.exists():
        return None
    return load_curve(path)


def plot_equity(
    spy_idle_json: Path,
    cash_idle_json: Path,
    risk_json: Path | None,
    stress_json: Path | None,
    stress_trained_json: Path | None,
    output: Path,
) -> None:
    spy_idle, payload_spy = load_curve(spy_idle_json)
    cash_idle, _ = load_curve(cash_idle_json)
    risk = load_optional_curve(risk_json)
    stress = load_optional_curve(stress_json)
    stress_trained = load_optional_curve(stress_trained_json)
    curves = [spy_idle, cash_idle]
    if risk is not None:
        curves.append(risk[0])
    if stress is not None:
        curves.append(stress[0])
    if stress_trained is not None:
        curves.append(stress_trained[0])
    start = min(curve["timestamp"].min() for curve in curves)
    end = max(curve["timestamp"].max() for curve in curves)
    starting = float(payload_spy["sequential_portfolio"]["starting_equity"])
    spy = spy_curve(start, end, starting)

    plt.figure(figsize=(13, 7))
    plt.plot(spy["timestamp"], spy["equity"], linewidth=2.2, linestyle="--", label="SPY buy-and-hold")
    plt.plot(cash_idle["timestamp"], cash_idle["equity"], linewidth=1.9, alpha=0.75, label="fixed q70 model, cash idle")
    plt.plot(spy_idle["timestamp"], spy_idle["equity"], linewidth=2.7, label="fixed q70 model, SPY idle")
    if risk is not None:
        risk_curve, _ = risk
        plt.plot(risk_curve["timestamp"], risk_curve["equity"], linewidth=2.7, label="risk-controlled model, SPY idle")
    if stress is not None:
        stress_curve, _ = stress
        plt.plot(stress_curve["timestamp"], stress_curve["equity"], linewidth=2.3, linestyle=":", label="fixed q70, cost/cap stress")
    if stress_trained is not None:
        stress_trained_curve, _ = stress_trained
        plt.plot(stress_trained_curve["timestamp"], stress_trained_curve["equity"], linewidth=2.2, linestyle="-.", label="stress-trained allocator, cost/cap")
    plt.axhline(starting, color="#555555", linewidth=1.0)
    plt.title("Locked one-year sequential tradable simulation")
    plt.ylabel("Portfolio equity ($)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def plot_drawdown(
    spy_idle_json: Path,
    cash_idle_json: Path,
    risk_json: Path | None,
    stress_json: Path | None,
    stress_trained_json: Path | None,
    output: Path,
) -> None:
    spy_idle, payload_spy = load_curve(spy_idle_json)
    cash_idle, _ = load_curve(cash_idle_json)
    risk = load_optional_curve(risk_json)
    stress = load_optional_curve(stress_json)
    stress_trained = load_optional_curve(stress_trained_json)
    curves = [spy_idle, cash_idle]
    if risk is not None:
        curves.append(risk[0])
    if stress is not None:
        curves.append(stress[0])
    if stress_trained is not None:
        curves.append(stress_trained[0])
    start = min(curve["timestamp"].min() for curve in curves)
    end = max(curve["timestamp"].max() for curve in curves)
    starting = float(payload_spy["sequential_portfolio"]["starting_equity"])
    spy = spy_curve(start, end, starting)

    plt.figure(figsize=(13, 7))
    plt.plot(spy["timestamp"], 100 * drawdown(spy["equity"]), linewidth=2.0, linestyle="--", label="SPY buy-and-hold")
    plt.plot(cash_idle["timestamp"], 100 * drawdown(cash_idle["equity"]), linewidth=1.8, alpha=0.75, label="fixed q70 model, cash idle")
    plt.plot(spy_idle["timestamp"], 100 * drawdown(spy_idle["equity"]), linewidth=2.5, label="fixed q70 model, SPY idle")
    if risk is not None:
        risk_curve, _ = risk
        plt.plot(risk_curve["timestamp"], 100 * drawdown(risk_curve["equity"]), linewidth=2.5, label="risk-controlled model, SPY idle")
    if stress is not None:
        stress_curve, _ = stress
        plt.plot(stress_curve["timestamp"], 100 * drawdown(stress_curve["equity"]), linewidth=2.2, linestyle=":", label="fixed q70, cost/cap stress")
    if stress_trained is not None:
        stress_trained_curve, _ = stress_trained
        plt.plot(stress_trained_curve["timestamp"], 100 * drawdown(stress_trained_curve["equity"]), linewidth=2.0, linestyle="-.", label="stress-trained allocator, cost/cap")
    plt.axhline(0, color="#555555", linewidth=1.0)
    plt.title("Locked one-year drawdown")
    plt.ylabel("Drawdown (%)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def plot_trade_returns(spy_idle_json: Path, output: Path) -> None:
    payload = json.loads(spy_idle_json.read_text())
    trades = payload["sequential_portfolio"]["trades_detail"]
    returns = np.array([t["portfolio_return"] for t in trades], dtype=float) * 100
    spy_returns = np.array([t["future_spy_return"] for t in trades], dtype=float) * 100

    plt.figure(figsize=(12, 7))
    bins = np.linspace(min(returns.min(), spy_returns.min()), max(returns.max(), spy_returns.max()), 28)
    plt.hist(spy_returns, bins=bins, alpha=0.55, label="SPY over active windows")
    plt.hist(returns, bins=bins, alpha=0.70, label="selected stocks")
    plt.axvline(0, color="#333333", linewidth=1.0)
    plt.title("Risk-controlled trade return distribution on locked year")
    plt.xlabel("Return per active window (%)")
    plt.ylabel("Trades")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spy-idle-json", default="checkpoints/world_model/tradable_locked1y_intraday120_q80_hybrid_entryonly_spyidle.json")
    parser.add_argument("--cash-idle-json", default="checkpoints/world_model/tradable_locked1y_intraday120_q80_hybrid_entryonly.json")
    parser.add_argument("--risk-json", default="checkpoints/world_model/tradable_locked1y_intraday120_q80_calibrated_dd18_spyidle.json")
    parser.add_argument("--stress-json", default="checkpoints/world_model/tradable_locked1y_intraday120_q80_fixed_cost10_cap3_cd10_spyidle.json")
    parser.add_argument("--stress-trained-json", default="checkpoints/world_model/tradable_locked1y_intraday120_q80_stress10_fixed_cost10_cap3_cd10_spyidle.json")
    parser.add_argument("--output-dir", default="docs/world_model_charts")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    spy_idle_json = Path(args.spy_idle_json)
    cash_idle_json = Path(args.cash_idle_json)
    risk_json = Path(args.risk_json) if args.risk_json else None
    stress_json = Path(args.stress_json) if args.stress_json else None
    stress_trained_json = Path(args.stress_trained_json) if args.stress_trained_json else None
    plot_equity(spy_idle_json, cash_idle_json, risk_json, stress_json, stress_trained_json, output / "locked1y_tradable_equity.png")
    plot_drawdown(spy_idle_json, cash_idle_json, risk_json, stress_json, stress_trained_json, output / "locked1y_tradable_drawdown.png")
    plot_trade_returns(risk_json if risk_json and risk_json.exists() else spy_idle_json, output / "locked1y_trade_returns.png")
    print(f"Wrote locked tradable charts to {output}")


if __name__ == "__main__":
    main()
