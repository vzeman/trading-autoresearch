"""Baseline strategies — no model training, just naive policies.

Run once to populate docs/baselines.png so we can compare any trained
result against:
  1. Untrained model (random weights, HOLD-biased action head → mostly idle)
  2. Buy-and-hold SPY (passive market benchmark)
  3. Buy-and-hold equal-weight 5-symbol portfolio

This proves that any future Sharpe gain over these baselines is
attributable to model training, not just to the data window.
"""
from __future__ import annotations
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from prepare import (
    UNIVERSE, STARTING_CASH_USD, NOTIONAL_PER_SYMBOL_USD,
    FEE_PER_TRADE_USD, SLIPPAGE_BPS,
    PaperBroker, prepare_all, split, sharpe_ratio, max_drawdown_pct,
)
from experiment import (
    PatchTransformer, USE_FEATURES, featurize, fetch_context,
    PATCH_LEN, CONTEXT_PATCHES, D_MODEL, N_HEADS, N_LAYERS, D_FF,
    DROPOUT, PRED_HORIZON, simulate,
)

REPO = Path(__file__).resolve().parent
DOCS = REPO / "docs"
DOCS.mkdir(exist_ok=True)


def load_eval_features() -> dict[str, pd.DataFrame]:
    bars = prepare_all()
    ctx = fetch_context()
    eval_feat = {}
    for sym in UNIVERSE:
        _tr, ev = split(bars[sym])
        if len(ev) > 50:
            eval_feat[sym] = featurize(ev, context=ctx)
    return eval_feat


def buy_and_hold_single(eval_feat: dict[str, pd.DataFrame], symbol: str) -> tuple[list, dict]:
    """Buy NOTIONAL_PER_SYMBOL_USD of `symbol` at first eval bar, hold to end."""
    if symbol not in eval_feat:
        return [], {"sharpe": 0, "pnl": 0, "dd": 0, "trades": 0}
    df = eval_feat[symbol]
    closes = df["close"].to_numpy(np.float64)
    times = df["timestamp"].tolist()
    qty = NOTIONAL_PER_SYMBOL_USD / max(closes[0], 1e-9)
    cost_at_entry = qty * closes[0] + FEE_PER_TRADE_USD
    cash_remaining = STARTING_CASH_USD - cost_at_entry
    eq_curve = [(times[i], cash_remaining + qty * closes[i]) for i in range(len(closes))]
    summary = {
        "sharpe": sharpe_ratio(eq_curve),
        "pnl": eq_curve[-1][1] - STARTING_CASH_USD,
        "dd": max_drawdown_pct(eq_curve),
        "trades": 1,
    }
    return eq_curve, summary


def buy_and_hold_equalweight(eval_feat: dict[str, pd.DataFrame]) -> tuple[list, dict]:
    """Buy NOTIONAL_PER_SYMBOL_USD of EACH symbol at start; hold."""
    syms = list(eval_feat.keys())
    if not syms:
        return [], {"sharpe": 0, "pnl": 0, "dd": 0, "trades": 0}
    # Use the timestamps from the FIRST symbol as the master timeline
    master_times = eval_feat[syms[0]]["timestamp"].tolist()
    qty_per_sym = {}
    cash = STARTING_CASH_USD
    for sym in syms:
        c0 = float(eval_feat[sym]["close"].iloc[0])
        qty_per_sym[sym] = NOTIONAL_PER_SYMBOL_USD / max(c0, 1e-9)
        cash -= NOTIONAL_PER_SYMBOL_USD + FEE_PER_TRADE_USD
    # equity at each timestamp = cash + sum(qty * latest_close_per_sym at that ts)
    last_close = {sym: float(eval_feat[sym]["close"].iloc[0]) for sym in syms}
    closes_by_sym = {sym: eval_feat[sym]["close"].to_numpy(np.float64) for sym in syms}
    eq_curve = []
    for i, t in enumerate(master_times):
        for sym in syms:
            if i < len(closes_by_sym[sym]):
                last_close[sym] = float(closes_by_sym[sym][i])
        eq = cash + sum(qty_per_sym[s] * last_close[s] for s in syms)
        eq_curve.append((t, eq))
    summary = {
        "sharpe": sharpe_ratio(eq_curve),
        "pnl": eq_curve[-1][1] - STARTING_CASH_USD,
        "dd": max_drawdown_pct(eq_curve),
        "trades": len(syms),
    }
    return eq_curve, summary


def untrained_model(eval_feat: dict[str, pd.DataFrame], seed: int = 0) -> tuple[list, dict]:
    """Random-init transformer (no pretrain, no RL) run through the standard
    simulate() pipeline. Shows what the model does BEFORE any training.
    Expected: HOLD bias dominates → very few trades → flat equity.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cpu"
    model = PatchTransformer(
        n_features=len(USE_FEATURES), patch_len=PATCH_LEN, context_patches=CONTEXT_PATCHES,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
        dropout=DROPOUT, pred_horizon=PRED_HORIZON,
    ).to(device)
    broker = simulate(model, eval_feat, device, learn=False)
    summary = {
        "sharpe": sharpe_ratio(broker.equity_curve) if broker.equity_curve else 0,
        "pnl": (broker.equity_curve[-1][1] - STARTING_CASH_USD) if broker.equity_curve else 0,
        "dd": max_drawdown_pct(broker.equity_curve) if broker.equity_curve else 0,
        "trades": broker.n_trades,
    }
    return broker.equity_curve, summary


def render_chart(curves: dict[str, list], summaries: dict[str, dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, eq in curves.items():
        if not eq:
            continue
        ts = [e[0] for e in eq]
        vals = [e[1] for e in eq]
        s = summaries[name]
        ax.plot(ts, vals, label=f"{name}  (sharpe {s['sharpe']:+.2f}, pnl ${s['pnl']:+,.0f})", alpha=0.85)
    ax.axhline(y=STARTING_CASH_USD, linestyle="--", color="gray", alpha=0.5, label="start")
    ax.set_title("Naive baselines on the held-out eval slice (no training)", fontsize=11)
    ax.set_xlabel("time (UTC)")
    ax.set_ylabel("portfolio equity ($)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main() -> None:
    print("[baselines] loading eval features …", flush=True)
    eval_feat = load_eval_features()
    syms = list(eval_feat.keys())
    print(f"[baselines] eval slice: {len(syms)} symbols, {len(eval_feat[syms[0]]):,} bars each")
    if not syms:
        print("[baselines] no eval data — bail")
        return

    curves: dict[str, list] = {}
    summaries: dict[str, dict] = {}

    # Buy-and-hold each individual symbol
    for sym in syms:
        c, s = buy_and_hold_single(eval_feat, sym)
        curves[f"B&H {sym}"] = c
        summaries[f"B&H {sym}"] = s
        print(f"[baselines] B&H {sym}: sharpe {s['sharpe']:+.2f}  pnl ${s['pnl']:+,.0f}  dd {s['dd']:+.1f}%")

    # Buy-and-hold equal-weight portfolio
    c, s = buy_and_hold_equalweight(eval_feat)
    curves["B&H equal-weight"] = c
    summaries["B&H equal-weight"] = s
    print(f"[baselines] B&H equal-weight: sharpe {s['sharpe']:+.2f}  pnl ${s['pnl']:+,.0f}  dd {s['dd']:+.1f}%")

    # Untrained model
    print("[baselines] running untrained model …")
    c, s = untrained_model(eval_feat, seed=0)
    curves["Untrained model (seed 0)"] = c
    summaries["Untrained model (seed 0)"] = s
    print(f"[baselines] untrained: sharpe {s['sharpe']:+.2f}  pnl ${s['pnl']:+,.0f}  dd {s['dd']:+.1f}%  trades {s['trades']}")

    out = DOCS / "baselines.png"
    render_chart(curves, summaries, out)
    print(f"[baselines] chart → {out}")


if __name__ == "__main__":
    main()
