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


def _spy_aligned(strategy_curve: list, spy_curve: list) -> np.ndarray:
    """Interpolate SPY equity at each strategy timestamp. Returns array same len as strategy_curve."""
    if not strategy_curve or not spy_curve:
        return np.array([])
    s_t = np.array([t.value for t, _ in strategy_curve], dtype=np.int64)
    spy_t = np.array([t.value for t, _ in spy_curve], dtype=np.int64)
    spy_v = np.array([v for _, v in spy_curve], dtype=np.float64)
    return np.interp(s_t, spy_t, spy_v)


def _pct_time_over_spy(strategy_curve: list, spy_curve: list) -> float:
    """Fraction of bars where strategy equity > SPY benchmark. Returns 0..100."""
    if not strategy_curve or not spy_curve:
        return 0.0
    s_v = np.array([v for _, v in strategy_curve], dtype=np.float64)
    spy_at = _spy_aligned(strategy_curve, spy_curve)
    if s_v.size == 0 or spy_at.size != s_v.size:
        return 0.0
    return float((s_v > spy_at).mean() * 100.0)


def _allocation_curves(equity_curves: list[list], cash_curves: list[list]) -> list[list]:
    """Per-seed allocation% = 100 * (equity - cash) / equity. Returns list of [(ts, pct)]."""
    out = []
    for eq, ca in zip(equity_curves, cash_curves):
        if not eq or not ca or len(eq) != len(ca):
            out.append([])
            continue
        alloc = []
        for (t1, e), (t2, c) in zip(eq, ca):
            denom = e if e > 1e-9 else 1e-9
            pct = max(0.0, min(100.0, 100.0 * (e - c) / denom))
            alloc.append((t1, pct))
        out.append(alloc)
    return out


def _draw_allocation_axis(ax, equity_curves, cash_curves) -> float:
    """Draw per-seed allocation% lines on a subplot. Returns avg allocation% across seeds."""
    alloc_curves = _allocation_curves(equity_curves, cash_curves)
    avg_alloc = 0.0
    n_seen = 0
    for i, ac in enumerate(alloc_curves):
        if not ac:
            continue
        ts = [t for t, _ in ac]
        vs = [v for _, v in ac]
        ax.plot(ts, vs, alpha=0.6, linewidth=0.8, label=f"seed {i}")
        avg_alloc += float(np.mean(vs))
        n_seen += 1
    if n_seen > 0:
        avg_alloc /= n_seen
    # median curve overlay
    med = _median_curve(alloc_curves)
    if med:
        med_t = [t for t, _ in med]
        med_v = [v for _, v in med]
        ax.plot(med_t, med_v, color="#1f2937", linewidth=1.6, label="median", zorder=4)
    ax.set_ylim(0, 100)
    ax.set_ylabel("alloc %")
    ax.grid(True, alpha=0.3)
    ax.axhline(50, color="gray", linestyle=":", alpha=0.4)
    return avg_alloc


def _median_curve(curves: list[list]) -> list:
    """Build a 'median' equity curve across seeds, aligned to seed-0 timestamps."""
    valid = [c for c in curves if c and len(c) > 1]
    if not valid:
        return []
    base_t = [t for t, _ in valid[0]]
    base_int = np.array([t.value for t in base_t], dtype=np.int64)
    stack = []
    for c in valid:
        ct = np.array([t.value for t, _ in c], dtype=np.int64)
        cv = np.array([v for _, v in c], dtype=np.float64)
        stack.append(np.interp(base_int, ct, cv))
    med = np.median(np.stack(stack, axis=0), axis=0)
    return list(zip(base_t, med.tolist()))

# Constraint: an experiment that draws down more than this is auto-rejected
MAX_DD_FLOOR_PCT = -10.0

# Per-seed safety ceiling (NOT a tight budget — only stops genuinely runaway runs).
# Each seed runs to completion regardless of wall time; we only abort if a single
# seed exceeds this. Default: 1 hour. The agent should generally pick configs
# that finish faster, but slow hardware should not cause partial results.
MAX_SECONDS_PER_SEED = 3600


def _seed_worker(seed: int):
    """Pickle-safe worker: imports + runs one seed in a subprocess."""
    import os
    # Pin per-worker torch threads to avoid oversubscription across multiple workers.
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    import torch
    torch.set_num_threads(2)
    from experiment import train_and_eval
    return seed, train_and_eval(seed=seed)


def _maybe_parallel_seeds(n_workers: int):
    """Return list of (seed, result_tuple). If n_workers>1, run via multiprocessing."""
    seeds = list(range(N_SEEDS))
    if n_workers <= 1 or N_SEEDS <= 1:
        return [_seed_worker(s) for s in seeds]
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    print(f"[evaluator] running {N_SEEDS} seeds × {n_workers} parallel workers", flush=True)
    with ctx.Pool(min(n_workers, N_SEEDS)) as pool:
        # imap_unordered streams results as they complete — print as we go
        out = []
        for seed_idx, result in pool.imap_unordered(_seed_worker, seeds):
            out.append((seed_idx, result))
            print(f"[evaluator] seed {seed_idx} returned ({len(out)}/{N_SEEDS} done)", flush=True)
    out.sort(key=lambda x: x[0])
    return out


def run(n_workers: int = 3) -> dict:
    """Run evaluator. n_workers=1 is sequential; >1 uses multiprocessing.Pool."""
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
    weighted_curves: list[list[tuple]] = []
    weighted_trades_per_seed: list[list[tuple]] = []
    weighted_pnls: list[float] = []
    weighted_sharpes: list[float] = []
    weighted_dds: list[float] = []
    weighted_trades_n: list[int] = []
    weighted_fees_l: list[float] = []
    seeds_done = 0

    primary_cash_curves: list[list[tuple]] = []
    picker_cash_curves: list[list[tuple]] = []
    weighted_cash_curves: list[list[tuple]] = []

    seed_results = _maybe_parallel_seeds(n_workers)
    for seed, result in seed_results:
        # Tuple shapes supported (newest first):
        #   6 = WEIGHTED-ONLY (current): (eq, ntrades, fees, slip, trades, cash_curve)
        #   16 = legacy with primary+picker+weighted+3 cash curves
        #   13 = legacy primary+picker+weighted (no cash curves)
        #   9, 5, 4 = older still
        picker_eq, picker_nt, picker_fees, picker_trades = [], 0, 0.0, []
        weighted_eq, weighted_nt, weighted_fees, weighted_trades = [], 0, 0.0, []
        primary_cash, picker_cash, weighted_cash = [], [], []
        if len(result) == 6:
            # WEIGHTED-ONLY: pipe weighted into both "primary" eq slot and weighted slot
            # so existing summary fields keep working until the deeper cleanup lands.
            eq, n_trades, fees, slip, trades, weighted_cash = result
            weighted_eq, weighted_nt, weighted_fees, weighted_trades = eq, n_trades, fees, trades
            primary_cash = weighted_cash
        elif len(result) == 16:
            (eq, n_trades, fees, slip, trades,
             picker_eq, picker_nt, picker_fees, picker_trades,
             weighted_eq, weighted_nt, weighted_fees, weighted_trades,
             primary_cash, picker_cash, weighted_cash) = result
        elif len(result) == 13:
            (eq, n_trades, fees, slip, trades,
             picker_eq, picker_nt, picker_fees, picker_trades,
             weighted_eq, weighted_nt, weighted_fees, weighted_trades) = result
        elif len(result) == 9:
            eq, n_trades, fees, slip, trades, picker_eq, picker_nt, picker_fees, picker_trades = result
        elif len(result) == 5:
            eq, n_trades, fees, slip, trades = result
        else:
            eq, n_trades, fees, slip = result
            trades = []
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
        # Weighted strategy stats
        weighted_curves.append(weighted_eq)
        weighted_trades_per_seed.append(weighted_trades)
        if weighted_eq and len(weighted_eq) > 5:
            weighted_pnls.append(weighted_eq[-1][1] - weighted_eq[0][1])
            weighted_sharpes.append(sharpe_ratio(weighted_eq))
            weighted_dds.append(max_drawdown_pct(weighted_eq))
        else:
            weighted_pnls.append(0.0); weighted_sharpes.append(0.0); weighted_dds.append(0.0)
        weighted_trades_n.append(weighted_nt)
        weighted_fees_l.append(weighted_fees)
        primary_cash_curves.append(primary_cash)
        picker_cash_curves.append(picker_cash)
        weighted_cash_curves.append(weighted_cash)
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
    # Weighted (Strategy 3) summary
    summary["weighted_sharpe"] = float(np.median(weighted_sharpes)) if weighted_sharpes else 0.0
    summary["weighted_pnl_usd"] = float(np.median(weighted_pnls)) if weighted_pnls else 0.0
    summary["weighted_pnl_pct"] = summary["weighted_pnl_usd"] / STARTING_CASH_USD * 100
    summary["weighted_max_dd_pct"] = float(min(weighted_dds)) if weighted_dds else 0.0
    summary["weighted_trades"] = int(np.median(weighted_trades_n)) if weighted_trades_n else 0
    summary["weighted_fees_usd"] = float(np.median(weighted_fees_l)) if weighted_fees_l else 0.0

    # % of time each strategy's median equity is above SPY buy-and-hold benchmark
    spy_curve_for_pct, _ = _spy_benchmark_curve()
    summary["primary_pct_over_spy"] = _pct_time_over_spy(_median_curve(equity_curves), spy_curve_for_pct)
    summary["picker_pct_over_spy"] = _pct_time_over_spy(_median_curve(picker_curves), spy_curve_for_pct)
    summary["weighted_pct_over_spy"] = _pct_time_over_spy(_median_curve(weighted_curves), spy_curve_for_pct)

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
    # Weighted is the only live strategy. Render full eval + 1-month sub-window.
    if any(weighted_curves):
        w_summary = {
            "pnl_med": float(np.median(weighted_pnls)) if weighted_pnls else 0.0,
            "sharpe_med": float(np.median(weighted_sharpes)) if weighted_sharpes else 0.0,
            "trades_med": int(np.median(weighted_trades_n)) if weighted_trades_n else 0,
        }
        _render_weighted_chart(weighted_curves, commit, w_summary,
                               trades_per_seed=weighted_trades_per_seed,
                               cash_curves=weighted_cash_curves)
        _render_weighted_window_chart(weighted_curves, commit, w_summary,
                                      trades_per_seed=weighted_trades_per_seed,
                                      cash_curves=weighted_cash_curves,
                                      window_days=30, suffix="1m", label="1 month")
    # exp80: multi-profile equity comparison chart (overlays all simulator strategies)
    try:
        _render_profile_compare_chart(commit)
    except Exception as e:
        print(f"[profile-compare-chart] failed: {e}", flush=True)
    _append_results_row(commit, summary, status, desc)
    _render_progress_chart()
    _update_readme(summary, commit)

    return summary


# ----------------------------------------------------------------------
# Chart + README helpers
# ----------------------------------------------------------------------

def _git_short_hash() -> str:
    """Return the experiment commit SHA. The driver may set EXPERIMENT_COMMIT if
    it added throwaway commits (e.g. LIVE-block) on top — without this override
    the chart filenames would key off the wrong (driver) commit."""
    override = os.environ.get("EXPERIMENT_COMMIT")
    if override:
        return override.strip()
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
                         trades_per_seed: list[list[tuple]] | None = None,
                         cash_curves: list[list[tuple]] | None = None) -> None:
    """Per-experiment equity curve PNG → docs/equity_latest.png + docs/equity_<commit>.png.

    Top subplot: equity vs SPY benchmark, shading where strategy beats SPY, trade markers.
    Bottom subplot: per-seed portfolio allocation% over time.
    """
    if not curves:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if cash_curves and any(cash_curves):
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 6.5),
                                       gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax2 = None
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

    # Shade where median strategy curve > SPY benchmark
    pct_over = 0.0
    med_curve = _median_curve(curves)
    if spy_curve and med_curve:
        med_t = [t for t, _ in med_curve]
        med_v = np.array([v for _, v in med_curve], dtype=np.float64)
        spy_at_med = _spy_aligned(med_curve, spy_curve)
        ax.fill_between(med_t, med_v, spy_at_med, where=(med_v > spy_at_med),
                        interpolate=True, color="#22c55e", alpha=0.18, zorder=1, label="median > SPY")
        ax.fill_between(med_t, med_v, spy_at_med, where=(med_v < spy_at_med),
                        interpolate=True, color="#ef4444", alpha=0.10, zorder=1)
        pct_over = _pct_time_over_spy(med_curve, spy_curve)

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

    avg_alloc = 0.0
    if ax2 is not None:
        avg_alloc = _draw_allocation_axis(ax2, curves, cash_curves)
        ax2.set_xlabel("time (UTC)")

    title_extra = f" · {n_trades_drawn} trade markers (seed 0)" if n_trades_drawn else ""
    alloc_extra = f"  ·  avg alloc {avg_alloc:.0f}%" if ax2 is not None else ""
    ax.set_title(
        f"Equity — commit {commit}  ·  "
        f"sharpe {summary['sharpe']:+.2f} (CI low {summary['sharpe_ci_low']:+.2f})  ·  "
        f"DD {summary['max_dd_pct']:+.1f}%  ·  {summary['trades']} trades  ·  "
        f"over SPY {pct_over:.0f}% of time{alloc_extra}{title_extra}",
        fontsize=10,
    )
    if ax2 is None:
        ax.set_xlabel("time (UTC)")
    ax.set_ylabel("portfolio equity ($)")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(DOCS / "equity_latest.png", dpi=110)
    fig.savefig(DOCS / f"equity_{commit}.png", dpi=110)
    plt.close(fig)


def _render_picker_chart(curves: list[list[tuple]], commit: str, summary: dict,
                         picker_trades_per_seed: list[list[tuple]] | None = None,
                         cash_curves: list[list[tuple]] | None = None) -> None:
    """Best-stock picker equity curve PNG → docs/picker_latest.png + docs/picker_<commit>.png."""
    if not any(curves):
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if cash_curves and any(cash_curves):
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 6.5),
                                       gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax2 = None
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

    pct_over_p = 0.0
    med_p = _median_curve(curves)
    if spy_curve and med_p:
        med_t = [t for t, _ in med_p]
        med_v = np.array([v for _, v in med_p], dtype=np.float64)
        spy_at = _spy_aligned(med_p, spy_curve)
        ax.fill_between(med_t, med_v, spy_at, where=(med_v > spy_at),
                        interpolate=True, color="#22c55e", alpha=0.18, zorder=1, label="median > SPY")
        ax.fill_between(med_t, med_v, spy_at, where=(med_v < spy_at),
                        interpolate=True, color="#ef4444", alpha=0.10, zorder=1)
        pct_over_p = _pct_time_over_spy(med_p, spy_curve)

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

    avg_alloc_p = 0.0
    if ax2 is not None:
        avg_alloc_p = _draw_allocation_axis(ax2, curves, cash_curves)
        ax2.set_xlabel("time (UTC)")

    alloc_extra = f"  ·  avg alloc {avg_alloc_p:.0f}%" if ax2 is not None else ""
    ax.set_title(
        f"Best-Stock Picker — commit {commit}  ·  "
        f"median PnL ${summary.get('pnl_med', 0):+,.2f}  ·  "
        f"{summary.get('trades_med', 0)} trades  ·  "
        f"over SPY {pct_over_p:.0f}%{alloc_extra}  ·  {n_picker_trades} markers (seed 0)",
        fontsize=10,
    )
    if ax2 is None:
        ax.set_xlabel("time (UTC)")
    ax.set_ylabel("portfolio equity ($)")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(DOCS / "picker_latest.png", dpi=110)
    fig.savefig(DOCS / f"picker_{commit}.png", dpi=110)
    plt.close(fig)


def _render_weighted_chart(curves: list[list[tuple]], commit: str, summary: dict,
                           trades_per_seed: list[list[tuple]] | None = None,
                           cash_curves: list[list[tuple]] | None = None) -> None:
    """Confidence-weighted strategy equity curve → docs/weighted_latest.png + commit copy."""
    if not any(curves):
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if cash_curves and any(cash_curves):
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 6.5),
                                       gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax2 = None
    for i, eq in enumerate(curves):
        if not eq:
            continue
        ts = [e[0] for e in eq]
        vals = [e[1] for e in eq]
        ax.plot(ts, vals, alpha=0.7, label=f"seed {i}", linewidth=1.0)
    spy_curve, spy_pnl = _spy_benchmark_curve()
    if spy_curve:
        spy_ts = [e[0] for e in spy_curve]
        spy_vals = [e[1] for e in spy_curve]
        ax.plot(spy_ts, spy_vals, color="black", linestyle="--", linewidth=2.0,
                alpha=0.8, label=f"SP500 B&H (${spy_pnl:+,.0f})", zorder=5)
    ax.axhline(y=STARTING_CASH_USD, linestyle=":", color="gray", alpha=0.5, label="start")

    pct_over_w = 0.0
    med_w = _median_curve(curves)
    if spy_curve and med_w:
        med_t = [t for t, _ in med_w]
        med_v = np.array([v for _, v in med_w], dtype=np.float64)
        spy_at = _spy_aligned(med_w, spy_curve)
        ax.fill_between(med_t, med_v, spy_at, where=(med_v > spy_at),
                        interpolate=True, color="#22c55e", alpha=0.18, zorder=1, label="median > SPY")
        ax.fill_between(med_t, med_v, spy_at, where=(med_v < spy_at),
                        interpolate=True, color="#ef4444", alpha=0.10, zorder=1)
        pct_over_w = _pct_time_over_spy(med_w, spy_curve)

    n_w_trades = 0
    if trades_per_seed and trades_per_seed[0]:
        for ts, sym, side in trades_per_seed[0]:
            color = "#22c55e" if side == "BUY" else "#ef4444"
            ax.axvline(ts, color=color, alpha=0.4, linewidth=0.7, linestyle=":")
        n_w_trades = len(trades_per_seed[0])
        from matplotlib.lines import Line2D
        marker_handles = [
            Line2D([0], [0], color="#22c55e", linestyle=":", label="BUY (seed 0)"),
            Line2D([0], [0], color="#ef4444", linestyle=":", label="SELL (seed 0)"),
        ]
        existing = ax.get_legend_handles_labels()
        ax.legend(handles=existing[0] + marker_handles, loc="best", fontsize=7, ncol=2)
    else:
        ax.legend(loc="best", fontsize=8)

    avg_alloc_w = 0.0
    if ax2 is not None:
        avg_alloc_w = _draw_allocation_axis(ax2, curves, cash_curves)
        ax2.set_xlabel("time (UTC)")

    alloc_extra = f"  ·  avg alloc {avg_alloc_w:.0f}%" if ax2 is not None else ""
    ax.set_title(
        f"Weighted Dynamic Sizing — commit {commit}  ·  "
        f"sharpe {summary.get('sharpe_med', 0):+.2f}  ·  "
        f"median PnL ${summary.get('pnl_med', 0):+,.2f}  ·  "
        f"{summary.get('trades_med', 0)} trades  ·  "
        f"over SPY {pct_over_w:.0f}%{alloc_extra}  ·  {n_w_trades} markers (seed 0)",
        fontsize=10,
    )
    if ax2 is None:
        ax.set_xlabel("time (UTC)")
    ax.set_ylabel("portfolio equity ($)")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(DOCS / "weighted_latest.png", dpi=110)
    fig.savefig(DOCS / f"weighted_{commit}.png", dpi=110)
    plt.close(fig)


def _slice_window(curves: list[list[tuple]], days: int) -> list[list[tuple]]:
    """Truncate each per-seed curve to the first `days` of trading from its start."""
    import pandas as pd
    out = []
    for c in curves:
        if not c:
            out.append([])
            continue
        t0 = c[0][0]
        cutoff = t0 + pd.Timedelta(days=days)
        out.append([(t, v) for (t, v) in c if t <= cutoff])
    return out


def _render_weighted_window_chart(curves, commit, summary, *, trades_per_seed=None,
                                  cash_curves=None, window_days=30, suffix="1m",
                                  label="1 month") -> None:
    """Render the weighted strategy over a smaller time window (e.g. first 30 days)."""
    if not any(curves):
        return
    sliced = _slice_window(curves, window_days)
    sliced_cash = _slice_window(cash_curves or [], window_days) if cash_curves else None
    sliced_trades = None
    if trades_per_seed:
        import pandas as pd
        sliced_trades = []
        for tlist, c in zip(trades_per_seed, curves):
            if not c or not tlist:
                sliced_trades.append([])
                continue
            cutoff = c[0][0] + pd.Timedelta(days=window_days)
            sliced_trades.append([t for t in tlist if t[0] <= cutoff])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if sliced_cash and any(sliced_cash):
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 6.5),
                                       gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax2 = None

    for i, eq in enumerate(sliced):
        if not eq:
            continue
        ax.plot([t for t, _ in eq], [v for _, v in eq], alpha=0.7, linewidth=1.0, label=f"seed {i}")

    spy_curve, _ = _spy_benchmark_curve()
    spy_window = []
    if spy_curve and sliced and sliced[0]:
        import pandas as pd
        cutoff = sliced[0][0][0] + pd.Timedelta(days=window_days)
        start = sliced[0][0][0]
        spy_window = [(t, v) for (t, v) in spy_curve if start <= t <= cutoff]

    if spy_window:
        ax.plot([t for t, _ in spy_window], [v for _, v in spy_window],
                color="black", linestyle="--", linewidth=2.0, alpha=0.8,
                label="SP500 B&H", zorder=5)

    ax.axhline(y=STARTING_CASH_USD, linestyle=":", color="gray", alpha=0.5, label="start")

    pct_over = 0.0
    med = _median_curve(sliced)
    if spy_window and med:
        med_t = [t for t, _ in med]
        med_v = np.array([v for _, v in med], dtype=np.float64)
        spy_at = _spy_aligned(med, spy_window)
        ax.fill_between(med_t, med_v, spy_at, where=(med_v > spy_at),
                        interpolate=True, color="#22c55e", alpha=0.18, zorder=1)
        ax.fill_between(med_t, med_v, spy_at, where=(med_v < spy_at),
                        interpolate=True, color="#ef4444", alpha=0.10, zorder=1)
        pct_over = _pct_time_over_spy(med, spy_window)

    n_trades_drawn = 0
    if sliced_trades and sliced_trades[0]:
        for ts, sym, side in sliced_trades[0]:
            color = "#22c55e" if side == "BUY" else "#ef4444"
            ax.axvline(ts, color=color, alpha=0.4, linewidth=0.7, linestyle=":")
        n_trades_drawn = len(sliced_trades[0])

    avg_alloc = 0.0
    if ax2 is not None:
        avg_alloc = _draw_allocation_axis(ax2, sliced, sliced_cash)
        ax2.set_xlabel("time (UTC)")

    # final PnL inside the window for the median seed
    pnl_window = 0.0
    if med:
        pnl_window = med[-1][1] - STARTING_CASH_USD

    alloc_extra = f"  ·  avg alloc {avg_alloc:.0f}%" if ax2 is not None else ""
    ax.set_title(
        f"Weighted ({label}) — commit {commit}  ·  "
        f"PnL ${pnl_window:+,.2f}  ·  over SPY {pct_over:.0f}%{alloc_extra}  ·  "
        f"{n_trades_drawn} markers (seed 0)",
        fontsize=10,
    )
    if ax2 is None:
        ax.set_xlabel("time (UTC)")
    ax.set_ylabel("portfolio equity ($)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=7, ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(DOCS / f"weighted_{suffix}_latest.png", dpi=110)
    fig.savefig(DOCS / f"weighted_{suffix}_{commit}.png", dpi=110)
    plt.close(fig)


def _render_profile_compare_chart(commit: str) -> None:
    """exp80: overlay equity curves of every profile (intraday/intraweek/intramonth/
    longterm/daily_capped/weekly_capped/monthly_capped/topN_pickers/spy_buyhold)
    so the user can visually compare trade-frequency variants on one chart.
    Uses median-across-seeds curve for each profile.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as _np
    import pandas as _pd

    curves_by_profile: dict = {}
    for f in sorted((REPO / "checkpoints").glob("last_seed*_profile_curves.json")):
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        for pname, pts in data.items():
            try:
                # convert string ts → datetime
                tsv = [(_pd.to_datetime(t), float(v)) for (t, v) in pts]
            except Exception:
                continue
            curves_by_profile.setdefault(pname, []).append(tsv)
    if not curves_by_profile:
        return
    # For each profile, take median final equity across seeds (sort by it for legend order)
    final_eq = {}
    for pname, seed_curves in curves_by_profile.items():
        ends = [c[-1][1] for c in seed_curves if c]
        final_eq[pname] = _np.median(ends) if ends else 0.0
    sorted_profiles = sorted(curves_by_profile.keys(), key=lambda p: -final_eq[p])

    fig, ax = plt.subplots(figsize=(11, 6.2))
    cmap = plt.get_cmap("tab20")
    for i, pname in enumerate(sorted_profiles):
        seed_curves = curves_by_profile[pname]
        if not seed_curves:
            continue
        # Pick the seed whose final equity is closest to the median (representative)
        ends = [c[-1][1] for c in seed_curves if c]
        if not ends:
            continue
        med = _np.median(ends)
        rep_idx = int(_np.argmin([abs(e - med) for e in ends]))
        rep = seed_curves[rep_idx]
        if not rep:
            continue
        ts = [t for (t, _v) in rep]
        ev = [v for (_t, v) in rep]
        is_spy = pname == "spy_buyhold"
        ax.plot(ts, ev, label=f"{pname}  (${final_eq[pname]:,.0f})",
                color=cmap(i % 20), linewidth=2.0 if is_spy else 1.3,
                linestyle="--" if is_spy else "-",
                alpha=0.95 if is_spy else 0.80)
    ax.axhline(50000, color="grey", linewidth=0.7, alpha=0.5, linestyle=":")
    ax.set_title(
        f"Strategy comparison — equity curves (median seed) — commit {commit}",
        fontsize=10,
    )
    ax.set_xlabel("time (UTC)")
    ax.set_ylabel("portfolio equity ($)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=7, ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(DOCS / "profile_compare_latest.png", dpi=110)
    fig.savefig(DOCS / f"profile_compare_{commit}.png", dpi=110)
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
    weighted = {
        "name": "Weighted (Kelly-sized, max 20% free cash, ≤5/step)",
        "sharpe": summary.get("weighted_sharpe", 0.0),
        "pnl": summary.get("weighted_pnl_usd", 0.0),
        "pnl_pct": summary.get("weighted_pnl_pct", 0.0),
        "dd": summary.get("weighted_max_dd_pct", 0.0),
        "trades": int(summary.get("weighted_trades", 0)),
        "fees": summary.get("weighted_fees_usd", 0.0),
        "over_spy": summary.get("weighted_pct_over_spy", 0.0),
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
        "over_spy": 0.0,  # SPY is the benchmark; trivially 0% strictly above itself
    }
    strategies = [weighted, spy]

    def winner(key: str, higher_better: bool) -> int:
        vals = [s[key] for s in strategies]
        return vals.index(max(vals) if higher_better else min(vals))

    win = {
        "sharpe": winner("sharpe", True),
        "pnl": winner("pnl", True),
        "dd": winner("dd", True),     # higher (less negative) is better
        "fees": winner("fees", False),
        "over_spy": winner("over_spy", True),
    }

    def cell(s_idx: int, key: str, fmt: str) -> str:
        v = strategies[s_idx][key]
        text = fmt.format(v)
        return f"**{text}** 🏆" if win.get(key) == s_idx else text

    lines = [
        "| Strategy | Sharpe | Net PnL | PnL % | Max DD % | Trades | Fees | % time > SPY |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, s in enumerate(strategies):
        lines.append(
            f"| {s['name']} | "
            f"{cell(i, 'sharpe', '{:+.3f}')} | "
            f"{cell(i, 'pnl', '${:+,.2f}')} | "
            f"{s['pnl_pct']:+.3f}% | "
            f"{cell(i, 'dd', '{:+.2f}%')} | "
            f"{s['trades']} | "
            f"{cell(i, 'fees', '${:.2f}')} | "
            f"{cell(i, 'over_spy', '{:.0f}%')} |"
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
             "### Weighted strategy — full eval window (~73 days)",
             "",
             f"![weighted equity](docs/weighted_latest.png)",
             "",
             "### Weighted strategy — first month of eval",
             "",
             f"![weighted 1m](docs/weighted_1m_latest.png)",
             "",
             "### Strategy vs SPY benchmark",
             "",
             _strategy_comparison_md(summary),
             "",
             "### Detailed metrics — weighted strategy",
             "",
             f"| metric | value |",
             f"|---|---|",
             f"| Sharpe (median over seeds) | **{summary.get('weighted_sharpe', 0):+.3f}** |",
             f"| Net PnL | ${summary.get('weighted_pnl_usd', 0):+,.2f} ({summary.get('weighted_pnl_pct', 0):+.3f}%) |",
             f"| Max drawdown | {summary.get('weighted_max_dd_pct', 0):+.2f}% |",
             f"| Trades | {int(summary.get('weighted_trades', 0))} |",
             f"| % time above SPY | {summary.get('weighted_pct_over_spy', 0):.0f}% |",
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
