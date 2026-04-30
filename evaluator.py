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
    N_SEEDS, TIME_BUDGET_SECONDS, STARTING_CASH_USD,
    sharpe_ratio, max_drawdown_pct, bootstrap_sharpe_ci,
)

REPO = Path(__file__).resolve().parent
DOCS = REPO / "docs"
DOCS.mkdir(exist_ok=True)
RESULTS_TSV = REPO / "results.tsv"
README = REPO / "README.md"

# Constraint: an experiment that draws down more than this is auto-rejected
MAX_DD_FLOOR_PCT = -10.0


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
    seeds_done = 0

    for seed in range(N_SEEDS):
        if time.time() - t0 > TIME_BUDGET_SECONDS:
            print(f"[evaluator] time budget hit after {seeds_done} seeds", flush=True)
            break
        print(f"\n[evaluator] === seed {seed+1}/{N_SEEDS} ===", flush=True)
        eq, n_trades, fees, slip = train_and_eval(seed=seed)
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
    _render_equity_chart(equity_curves, commit, summary)
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


def _render_equity_chart(curves: list[list[tuple]], commit: str, summary: dict) -> None:
    """Per-experiment equity curve PNG → docs/equity_latest.png + docs/equity_<commit>.png."""
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
        ax.plot(ts, vals, alpha=0.7, label=f"seed {i}")
    ax.axhline(y=STARTING_CASH_USD, linestyle="--", color="gray", alpha=0.5, label="start")
    ax.set_title(
        f"Equity curve — commit {commit}  ·  "
        f"sharpe {summary['sharpe']:+.2f} (CI low {summary['sharpe_ci_low']:+.2f})  ·  "
        f"DD {summary['max_dd_pct']:+.1f}%  ·  {summary['trades']} trades"
    )
    ax.set_xlabel("time (UTC)")
    ax.set_ylabel("portfolio equity ($)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(DOCS / "equity_latest.png", dpi=110)
    fig.savefig(DOCS / f"equity_{commit}.png", dpi=110)
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
             "### Latest experiment",
             "",
             f"![equity curve](docs/equity_latest.png)",
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
