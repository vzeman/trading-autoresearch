"""Frozen evaluator. AGENT MUST NOT MODIFY THIS FILE.

Runs experiment.train_and_eval() with N_SEEDS, computes portfolio metrics on
each, and prints a single canonical summary block the agent can grep.

Output format (ALWAYS this exact shape so the agent can parse it):

    ---
    sharpe:           1.234
    sharpe_ci_low:    0.456
    sharpe_ci_high:   1.987
    max_dd_pct:       -2.3
    pnl_usd:          234.56
    pnl_pct:          0.469
    trades:           78
    fees_usd:         78.0
    slippage_usd:     12.4
    elapsed_seconds:  295.4
    seeds_completed:  3
    ---

Decision rule the agent should follow:
  - Higher `sharpe_ci_low` is better (statistically robust).
  - HARD CONSTRAINT: `max_dd_pct` must be > MAX_DD_FLOOR (default −10%).
    Anything that violates is auto-discarded regardless of Sharpe.
  - Tiebreaker: simpler diff wins.
"""
from __future__ import annotations
import os
import sys
import time
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from prepare import (
    N_SEEDS, STARTING_CASH_USD,
    sharpe_ratio, max_drawdown_pct, bootstrap_sharpe_ci,
)

REPO = Path(__file__).resolve().parent
DOCS = REPO / "docs"
DOCS.mkdir(exist_ok=True)
RESULTS_TSV = REPO / "results.tsv"
README = REPO / "README.md"


def _spy_benchmark_curve() -> tuple[list, float]:
    """Buy-and-hold full SP500 (SPY): invest STARTING_CASH at start of eval slice.
    Returns (equity_curve, final_pnl). Cached prices from prepare.fetch_bars."""
    try:
        from prepare import fetch_bars, split
        bars = fetch_bars("SPY")
        _, ev = split(bars)
        if len(ev) < 2:
            return [], 0.0
        closes = ev["close"].to_numpy()
        times = ev["timestamp"].tolist()
        qty = STARTING_CASH_USD / max(closes[0], 1e-9)
        curve = [(times[i], qty * closes[i]) for i in range(len(closes))]
        pnl = curve[-1][1] - STARTING_CASH_USD
        return curve, pnl
    except Exception:
        return [], 0.0

# Constraint: an experiment that draws down more than this is auto-rejected
MAX_DD_FLOOR_PCT = -10.0

# Per-seed safety ceiling (NOT a tight budget — only stops genuinely runaway runs).
# Each seed runs to completion regardless of wall time; we only abort if a single
# seed exceeds this. Default: 1 hour. The agent should generally pick configs
# that finish faster, but slow hardware should not cause partial results.
MAX_SECONDS_PER_SEED = 3600


def run() -> dict:
    from experiment import train_and_eval   # imported here so import errors → discard

    t0 = time.time()
    sharpes: list[float] = []
    sharpes_ci_low: list[float] = []
    sharpes_ci_high: list[float] = []
    dds: list[float] = []
    pnls: list[float] = []
    trades_list: list[int] = []
    fees_list: list[float] = []
    slip_list: list[float] = []
    equity_curves: list[list[tuple]] = []
    trades_per_seed: list[list[tuple]] = []
    picker_curves: list[list[tuple]] = []
    picker_trades_per_seed: list[list[tuple]] = []
    picker_pnls: list[float] = []
    picker_sharpes: list[float] = []
    picker_dds: list[float] = []
    picker_trades_n: list[int] = []
    picker_fees_l: list[float] = []
    seeds_done = 0

    for seed in range(N_SEEDS):
        print(f"\n[evaluator] === seed {seed+1}/{N_SEEDS} ===", flush=True)
        seed_t0 = time.time()
        result = train_and_eval(seed=seed)
        # Backward-compat: 4 / 5 / 9 tuple lengths supported.
        # 9-tuple = primary + picker secondary strategy.
        picker_eq, picker_nt, picker_fees, picker_trades = [], 0, 0.0, []
        if len(result) == 9:
            eq, n_trades, fees, slip, trades, picker_eq, picker_nt, picker_fees, picker_trades = result
        elif len(result) == 5:
            eq, n_trades, fees, slip, trades = result
        else:
            eq, n_trades, fees, slip = result
            trades = []
        seed_elapsed = time.time() - seed_t0
        if seed_elapsed > MAX_SECONDS_PER_SEED:
            print(f"[evaluator] WARNING seed {seed} took {seed_elapsed:.0f}s "
                  f"(safety ceiling {MAX_SECONDS_PER_SEED}s) — consider a smaller config",
                  flush=True)
        if not eq or len(eq) < 5:
            print(f"[evaluator] seed {seed}: empty equity curve, skipping", flush=True)
            continue
        s = sharpe_ratio(eq)
        ci_lo, ci_hi = bootstrap_sharpe_ci(eq)
        dd = max_drawdown_pct(eq)
        pnl = eq[-1][1] - eq[0][1]
        sharpes.append(s); sharpes_ci_low.append(ci_lo); sharpes_ci_high.append(ci_hi)
        dds.append(dd); pnls.append(pnl); trades_list.append(n_trades)
        fees_list.append(fees); slip_list.append(slip)
        equity_curves.append(eq)
        trades_per_seed.append(trades)
        picker_curves.append(picker_eq)
        picker_trades_per_seed.append(picker_trades)
        if picker_eq and len(picker_eq) > 5:
            picker_pnls.append(picker_eq[-1][1] - picker_eq[0][1])
            picker_sharpes.append(sharpe_ratio(picker_eq))
            picker_dds.append(max_drawdown_pct(picker_eq))
        else:
            picker_pnls.append(0.0); picker_sharpes.append(0.0); picker_dds.append(0.0)
        picker_trades_n.append(picker_nt)
        picker_fees_l.append(picker_fees)
        seeds_done += 1
        print(f"[evaluator] seed {seed}: sharpe={s:+.3f}  dd={dd:+.2f}%  pnl=${pnl:+,.2f}  trades={n_trades}", flush=True)

    elapsed = time.time() - t0
    if seeds_done == 0:
        print("\n---\nsharpe:           0.000\nsharpe_ci_low:    -999.000\nsharpe_ci_high:   0.000\nmax_dd_pct:       -100.0\npnl_usd:          0.00\npnl_pct:          0.000\ntrades:           0\nfees_usd:         0.00\nslippage_usd:     0.00\nelapsed_seconds:  {:.1f}\nseeds_completed:  0\n---".format(elapsed))
        return {}

    sharpe_med = float(np.median(sharpes))
    ci_low_med = float(np.median(sharpes_ci_low))
    ci_high_med = float(np.median(sharpes_ci_high))
    dd_worst = float(min(dds))           # most negative
    pnl_med = float(np.median(pnls))
    pnl_pct = pnl_med / STARTING_CASH_USD * 100
    trades_med = int(np.median(trades_list))
    fees_med = float(np.median(fees_list))
    slip_med = float(np.median(slip_list))

    summary = dict(
        sharpe=sharpe_med, sharpe_ci_low=ci_low_med, sharpe_ci_high=ci_high_med,
        max_dd_pct=dd_worst, pnl_usd=pnl_med, pnl_pct=pnl_pct,
        trades=trades_med, fees_usd=fees_med, slippage_usd=slip_med,
        elapsed_seconds=elapsed, seeds_completed=seeds_done,
    )
    # Picker (secondary strategy) summary
    summary["picker_sharpe"] = float(np.median(picker_sharpes)) if picker_sharpes else 0.0
    summary["picker_pnl_usd"] = float(np.median(picker_pnls)) if picker_pnls else 0.0
    summary["picker_pnl_pct"] = summary["picker_pnl_usd"] / STARTING_CASH_USD * 100
    summary["picker_max_dd_pct"] = float(min(picker_dds)) if picker_dds else 0.0
    summary["picker_trades"] = int(np.median(picker_trades_n)) if picker_trades_n else 0
    summary["picker_fees_usd"] = float(np.median(picker_fees_l)) if picker_fees_l else 0.0

    print("\n---")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k:<17} {v:.3f}".rstrip("0").rstrip(".") if abs(v) > 0 else f"{k:<17} 0.000")
        else:
            print(f"{k:<17} {v}")
    # Re-print in canonical exact format that the agent should grep:
    print("\n--- canonical ---")
    print(f"sharpe:           {sharpe_med:+.3f}")
    print(f"sharpe_ci_low:    {ci_low_med:+.3f}")
    print(f"sharpe_ci_high:   {ci_high_med:+.3f}")
    print(f"max_dd_pct:       {dd_worst:+.2f}")
    print(f"pnl_usd:          {pnl_med:+.2f}")
    print(f"pnl_pct:          {pnl_pct:+.3f}")
    print(f"trades:           {trades_med}")
    print(f"fees_usd:         {fees_med:.2f}")
    print(f"slippage_usd:     {slip_med:.2f}")
    print(f"elapsed_seconds:  {elapsed:.1f}")
    print(f"seeds_completed:  {seeds_done}")
    print(f"---")

    if dd_worst < MAX_DD_FLOOR_PCT:
        print(f"\n[evaluator] CONSTRAINT VIOLATED: max_dd_pct {dd_worst:.2f}% < floor {MAX_DD_FLOOR_PCT}%")
        print(f"[evaluator] Recommend: DISCARD this experiment")

    # ----- Render charts + update results.tsv + update README -----
    commit = _git_short_hash()
    desc = _last_commit_subject()
    status = "discard" if dd_worst < MAX_DD_FLOOR_PCT else "auto"  # agent overwrites this
    _render_equity_chart(equity_curves, commit, summary, trades_per_seed=trades_per_seed)

    # Picker (secondary strategy) — separate chart
    if any(picker_curves):
        picker_summary = {
            "pnl_med": float(np.median(picker_pnls)) if picker_pnls else 0.0,
            "trades_med": int(np.median(picker_trades_n)) if picker_trades_n else 0,
        }
        _render_picker_chart(picker_curves, commit, picker_summary,
                             picker_trades_per_seed=picker_trades_per_seed)
    _append_results_row(commit, summary, status, desc)
    _render_progress_chart()
    _update_readme(summary, commit)

    return summary


# ----------------------------------------------------------------------
# Chart + README helpers
# ----------------------------------------------------------------------

def _git_short_hash() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short=7", "HEAD"], cwd=REPO).decode().strip()
        return out or "untracked"
    except Exception:
        return "untracked"


def _last_commit_subject() -> str:
    try:
        out = subprocess.check_output(["git", "log", "-1", "--pretty=%s"], cwd=REPO).decode().strip()
        return out[:120].replace("\t", " ")
    except Exception:
        return "(no commit)"


def _render_equity_chart(curves: list[list[tuple]], commit: str, summary: dict,
                         trades_per_seed: list[list[tuple]] | None = None) -> None:
    """Per-experiment equity curve PNG → docs/equity_latest.png + docs/equity_<commit>.png.

    Overlays:
      - Each per-seed equity line (thin, colored)
      - SP500 (SPY) buy-and-hold reference line (thick black dashed)
      - Start-cash horizontal line
      - Trade markers (vertical lines, seed 0): green=BUY, red=SELL
    """
    if not curves:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, eq in enumerate(curves):
        if not eq:
            continue
        ts = [e[0] for e in eq]
        vals = [e[1] for e in eq]
        ax.plot(ts, vals, alpha=0.7, label=f"seed {i}", linewidth=1.0)
    # SP500 buy-and-hold benchmark — thick dashed black
    spy_curve, spy_pnl = _spy_benchmark_curve()
    if spy_curve:
        spy_ts = [e[0] for e in spy_curve]
        spy_vals = [e[1] for e in spy_curve]
        ax.plot(spy_ts, spy_vals, color="black", linestyle="--", linewidth=2.0,
                alpha=0.8, label=f"SP500 B&H (${spy_pnl:+,.0f})", zorder=5)
    ax.axhline(y=STARTING_CASH_USD, linestyle=":", color="gray", alpha=0.5, label="start")

    # Trade markers — show seed 0's trades only (chart would be too busy with all 10).
    n_trades_drawn = 0
    if trades_per_seed and trades_per_seed[0]:
        seed0_trades = trades_per_seed[0]
        for ts, sym, side in seed0_trades:
            color = "#22c55e" if side == "BUY" else "#ef4444"
            ax.axvline(ts, color=color, alpha=0.35, linewidth=0.7, linestyle=":")
        n_trades_drawn = len(seed0_trades)
        # Add legend handles for the markers
        from matplotlib.lines import Line2D
        marker_handles = [
            Line2D([0], [0], color="#22c55e", linestyle=":", label=f"BUY (seed 0)"),
            Line2D([0], [0], color="#ef4444", linestyle=":", label=f"SELL (seed 0)"),
        ]
        existing = ax.get_legend_handles_labels()
        ax.legend(handles=existing[0] + marker_handles, loc="best", fontsize=7, ncol=2)
    else:
        ax.legend(loc="best", fontsize=8)

    title_extra = f" · {n_trades_drawn} trade markers (seed 0)" if n_trades_drawn else ""
    ax.set_title(
        f"Equity — commit {commit}  ·  "
        f"sharpe {summary['sharpe']:+.2f} (CI low {summary['sharpe_ci_low']:+.2f})  ·  "
        f"DD {summary['max_dd_pct']:+.1f}%  ·  {summary['trades']} trades{title_extra}",
        fontsize=10,
    )
    ax.set_xlabel("time (UTC)")
    ax.set_ylabel("portfolio equity ($)")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(DOCS / "equity_latest.png", dpi=110)
    fig.savefig(DOCS / f"equity_{commit}.png", dpi=110)
    plt.close(fig)


def _render_picker_chart(curves: list[list[tuple]], commit: str, summary: dict,
                         picker_trades_per_seed: list[list[tuple]] | None = None) -> None:
    """Best-stock picker equity curve PNG → docs/picker_latest.png + docs/picker_<commit>.png."""
    if not any(curves):
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, eq in enumerate(curves):
        if not eq:
            continue
        ts = [e[0] for e in eq]
        vals = [e[1] for e in eq]
        ax.plot(ts, vals, alpha=0.7, label=f"seed {i}", linewidth=1.0)
    # SP500 benchmark for picker chart too
    spy_curve, spy_pnl = _spy_benchmark_curve()
    if spy_curve:
        spy_ts = [e[0] for e in spy_curve]
        spy_vals = [e[1] for e in spy_curve]
        ax.plot(spy_ts, spy_vals, color="black", linestyle="--", linewidth=2.0,
                alpha=0.8, label=f"SP500 B&H (${spy_pnl:+,.0f})", zorder=5)
    ax.axhline(y=STARTING_CASH_USD, linestyle=":", color="gray", alpha=0.5, label="start")

    n_picker_trades = 0
    if picker_trades_per_seed and picker_trades_per_seed[0]:
        for ts, sym, side in picker_trades_per_seed[0]:
            color = "#22c55e" if side == "BUY" else "#ef4444"
            ax.axvline(ts, color=color, alpha=0.4, linewidth=0.7, linestyle=":")
        n_picker_trades = len(picker_trades_per_seed[0])
        from matplotlib.lines import Line2D
        marker_handles = [
            Line2D([0], [0], color="#22c55e", linestyle=":", label="BUY (seed 0)"),
            Line2D([0], [0], color="#ef4444", linestyle=":", label="SELL (seed 0)"),
        ]
        existing = ax.get_legend_handles_labels()
        ax.legend(handles=existing[0] + marker_handles, loc="best", fontsize=7, ncol=2)
    else:
        ax.legend(loc="best", fontsize=8)

    ax.set_title(
        f"Best-Stock Picker — commit {commit}  ·  "
        f"median PnL ${summary.get('pnl_med', 0):+,.2f}  ·  "
        f"{summary.get('trades_med', 0)} trades  ·  {n_picker_trades} markers (seed 0)",
        fontsize=10,
    )
    ax.set_xlabel("time (UTC)")
    ax.set_ylabel("portfolio equity ($)")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(DOCS / "picker_latest.png", dpi=110)
    fig.savefig(DOCS / f"picker_{commit}.png", dpi=110)
    plt.close(fig)


def _render_progress_chart() -> None:
    """Sharpe over experiments PNG → docs/progress.png."""
    if not RESULTS_TSV.exists():
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows: list[dict] = []
    with open(RESULTS_TSV) as f:
        header = f.readline().strip().split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != len(header):
                continue
            rows.append(dict(zip(header, parts)))
    if not rows:
        return
    xs = list(range(1, len(rows) + 1))
    sharpe = [float(r.get("sharpe", 0)) for r in rows]
    ci_low = [float(r.get("sharpe_ci_low", 0)) for r in rows]
    status = [r.get("status", "auto") for r in rows]
    colors = {"keep": "#22c55e", "discard": "#ef4444", "crash": "#9ca3af", "auto": "#3b82f6"}
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xs, sharpe, "-", color="#3b82f6", label="sharpe (median of seeds)", alpha=0.8)
    ax.plot(xs, ci_low, "--", color="#94a3b8", label="sharpe CI low (5%)", alpha=0.8)
    for x, s, st in zip(xs, sharpe, status):
        ax.scatter([x], [s], c=colors.get(st, "#3b82f6"), s=30, edgecolors="white", linewidths=0.5)
    # running best on CI-low
    best = float("-inf")
    best_xs, best_ys = [], []
    for x, c, st in zip(xs, ci_low, status):
        if st == "keep" and c > best:
            best = c
        best_xs.append(x); best_ys.append(best if best > -1e9 else 0.0)
    ax.plot(best_xs, best_ys, color="#16a34a", linewidth=2, label="running best CI low (kept)")
    ax.axhline(0, color="black", alpha=0.4, linewidth=0.5)
    ax.set_xlabel("experiment #")
    ax.set_ylabel("annualized Sharpe")
    ax.set_title(f"Autoresearch progress — {len(rows)} experiments")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(DOCS / "progress.png", dpi=110)
    plt.close(fig)


RESULTS_HEADER = "commit\tsharpe\tsharpe_ci_low\tmax_dd_pct\tpnl_usd\ttrades\tstatus\tdescription"


def _append_results_row(commit: str, summary: dict, status: str, description: str) -> None:
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text(RESULTS_HEADER + "\n")
    with open(RESULTS_TSV, "a") as f:
        f.write("\t".join([
            commit,
            f"{summary['sharpe']:.4f}",
            f"{summary['sharpe_ci_low']:.4f}",
            f"{summary['max_dd_pct']:.2f}",
            f"{summary['pnl_usd']:.2f}",
            str(int(summary['trades'])),
            status,
            description.replace("\t", " ")[:120],
        ]) + "\n")


README_START = "<!-- RESULTS_START -->"
README_END = "<!-- RESULTS_END -->"


def _strategy_comparison_md(summary: dict) -> str:
    """Side-by-side comparison table — highlight the winner per metric.
    Includes SP500 (SPY) buy-and-hold as a passive baseline."""
    primary = {
        "name": "Primary (full portfolio every-bar)",
        "sharpe": summary.get("sharpe", 0.0),
        "pnl": summary.get("pnl_usd", 0.0),
        "pnl_pct": summary.get("pnl_pct", 0.0),
        "dd": summary.get("max_dd_pct", 0.0),
        "trades": int(summary.get("trades", 0)),
        "fees": summary.get("fees_usd", 0.0),
    }
    picker = {
        "name": "Picker (best-stock, $1k cooldown 5min)",
        "sharpe": summary.get("picker_sharpe", 0.0),
        "pnl": summary.get("picker_pnl_usd", 0.0),
        "pnl_pct": summary.get("picker_pnl_pct", 0.0),
        "dd": summary.get("picker_max_dd_pct", 0.0),
        "trades": int(summary.get("picker_trades", 0)),
        "fees": summary.get("picker_fees_usd", 0.0),
    }
    # SP500 buy-and-hold benchmark — naive comparison
    spy_curve, spy_pnl = _spy_benchmark_curve()
    spy_pnl_pct = spy_pnl / STARTING_CASH_USD * 100 if STARTING_CASH_USD > 0 else 0
    spy_sharpe = sharpe_ratio(spy_curve) if spy_curve else 0.0
    spy_dd = max_drawdown_pct(spy_curve) if spy_curve else 0.0
    spy = {
        "name": "**SP500 (SPY) buy-and-hold** — passive benchmark",
        "sharpe": spy_sharpe,
        "pnl": spy_pnl,
        "pnl_pct": spy_pnl_pct,
        "dd": spy_dd,
        "trades": 1,
        "fees": 1.0,   # one $1 fee at entry
    }
    strategies = [primary, picker, spy]

    def winner(key: str, higher_better: bool) -> int:
        vals = [s[key] for s in strategies]
        return vals.index(max(vals) if higher_better else min(vals))

    win = {
        "sharpe": winner("sharpe", True),
        "pnl": winner("pnl", True),
        "dd": winner("dd", True),     # higher (less negative) is better
        "fees": winner("fees", False),
    }

    def cell(s_idx: int, key: str, fmt: str) -> str:
        v = strategies[s_idx][key]
        text = fmt.format(v)
        return f"**{text}** 🏆" if win.get(key) == s_idx else text

    lines = [
        "| Strategy | Sharpe | Net PnL | PnL % | Max DD % | Trades | Fees |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for i, s in enumerate(strategies):
        lines.append(
            f"| {s['name']} | "
            f"{cell(i, 'sharpe', '{:+.3f}')} | "
            f"{cell(i, 'pnl', '${:+,.2f}')} | "
            f"{s['pnl_pct']:+.3f}% | "
            f"{cell(i, 'dd', '{:+.2f}%')} | "
            f"{s['trades']} | "
            f"{cell(i, 'fees', '${:.2f}')} |"
        )
    overall_winner = strategies[win["sharpe"]]["name"]
    lines.append("")
    lines.append(f"**Best by Sharpe:** {overall_winner}")
    return "\n".join(lines)


def _update_readme(summary: dict, commit: str) -> None:
    if not README.exists():
        return
    rows: list[dict] = []
    if RESULTS_TSV.exists():
        with open(RESULTS_TSV) as f:
            header = f.readline().strip().split("\t")
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == len(header):
                    rows.append(dict(zip(header, parts)))

    # Top-5 by sharpe_ci_low (only kept rows)
    kept = [r for r in rows if r.get("status") == "keep"]
    kept.sort(key=lambda r: float(r.get("sharpe_ci_low", "0")), reverse=True)
    top = kept[:5]

    lines = [README_START, "",
             f"_Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_  ",
             f"_Total experiments: **{len(rows)}**  ·  kept: **{len(kept)}**  ·  latest commit: `{commit}`_",
             "",
             "### Latest experiment — primary strategy (full portfolio)",
             "",
             f"![equity curve](docs/equity_latest.png)",
             "",
             "### Latest experiment — best-stock picker (secondary strategy)",
             "",
             f"![picker equity](docs/picker_latest.png)",
             "",
             "### Strategy comparison @ this checkpoint",
             "",
             _strategy_comparison_md(summary),
             "",
             "### Detailed metrics — primary strategy",
             "",
             f"| metric | value |",
             f"|---|---|",
             f"| Sharpe (median over seeds) | **{summary['sharpe']:+.3f}** |",
             f"| Sharpe — bootstrap CI low (5%) | **{summary['sharpe_ci_low']:+.3f}** |",
             f"| Sharpe — bootstrap CI high (95%) | {summary['sharpe_ci_high']:+.3f} |",
             f"| Max drawdown | {summary['max_dd_pct']:+.2f}% |",
             f"| Net PnL | ${summary['pnl_usd']:+,.2f} ({summary['pnl_pct']:+.3f}%) |",
             f"| Trades | {int(summary['trades'])} |",
             f"| Fees / slippage | ${summary['fees_usd']:.2f} / ${summary['slippage_usd']:.2f} |",
             f"| Wall time | {summary['elapsed_seconds']:.1f}s |",
             f"| Seeds completed | {summary['seeds_completed']} |",
             "",
             "### Progress over all experiments",
             "",
             f"![progress](docs/progress.png)",
             "",
             "### Leaderboard (top 5 kept by Sharpe CI-low)",
             "",
             f"| # | commit | Sharpe | CI-low | DD% | PnL | Trades | Description |",
             f"|---|---|---:|---:|---:|---:|---:|---|"]
    for i, r in enumerate(top):
        lines.append(
            f"| {i+1} | `{r['commit']}` | "
            f"{float(r['sharpe']):+.2f} | {float(r['sharpe_ci_low']):+.2f} | "
            f"{float(r['max_dd_pct']):+.2f} | ${float(r['pnl_usd']):+,.2f} | "
            f"{r['trades']} | {r['description']} |"
        )
    lines.extend(["", README_END])
    block = "\n".join(lines)

    txt = README.read_text()
    if README_START in txt and README_END in txt:
        before = txt.split(README_START)[0]
        after = txt.split(README_END)[-1]
        new = before + block + after
    else:
        new = txt.rstrip() + "\n\n" + block + "\n"
    README.write_text(new)


if __name__ == "__main__":
    run()
