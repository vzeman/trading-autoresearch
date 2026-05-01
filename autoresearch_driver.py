"""One iteration of the autoresearch loop.

Usage: python autoresearch_driver.py "description of change"

Assumes the agent has ALREADY edited experiment.py and committed. This script:
  1. Runs evaluator.py, captures stdout to run.log
  2. Parses the canonical metrics block
  3. Decides keep/discard per program.md rules:
       - max_dd_pct < -10 → discard (auto)
       - sharpe_ci_low > prior best (across status=keep rows) → keep
       - otherwise → discard
  4. Updates the last row of results.tsv with the decided status
  5. discard → git reset --hard HEAD~1
  6. keep → promote checkpoints/last_seed*.pt → checkpoints/best/<commit>_ci<ci>_<seed>.pt
                and update checkpoints/best.json

Exit code 0 always (so the outer loop continues even after crashes).
Prints a STATUS=... line for the agent to grep.
"""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
RESULTS_TSV = REPO / "results.tsv"
RUN_LOG = REPO / "run.log"
CHECKPOINTS = REPO / "checkpoints"
BEST_DIR = CHECKPOINTS / "best"
BEST_JSON = CHECKPOINTS / "best.json"
ITERATIONS_DIR = REPO / "iterations"
LATEST_LINK = ITERATIONS_DIR / "latest.md"
DOCS = REPO / "docs"
DD_FLOOR = -10.0


def git(args: list[str]) -> str:
    return subprocess.check_output(["git"] + args, cwd=REPO).decode().strip()


def parse_canonical(text: str) -> dict[str, str]:
    out, in_block = {}, False
    for line in text.splitlines():
        if line.strip() == "--- canonical ---":
            in_block = True
            continue
        if not in_block:
            continue
        if line.strip() == "---":
            break
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def best_kept_ci_low() -> float:
    """Highest sharpe_ci_low across status=keep rows with REAL trades.

    Skips rows with trades==0 (their ci_low=0.0 is artificial — bootstrap on
    an empty equity curve returns 0, which would set an unbeatable bar).
    """
    if not RESULTS_TSV.exists():
        return float("-inf")
    best = float("-inf")
    with open(RESULTS_TSV) as f:
        f.readline()  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8 or parts[6] != "keep":
                continue
            try:
                trades = int(parts[5])
                if trades <= 0:
                    continue
                v = float(parts[2])
                best = max(best, v)
            except ValueError:
                continue
    return best


def update_last_row_status(status: str, description: str) -> None:
    if not RESULTS_TSV.exists():
        return
    lines = RESULTS_TSV.read_text().splitlines()
    if len(lines) < 2:
        return
    parts = lines[-1].split("\t")
    if len(parts) < 7:
        return
    parts[6] = status
    if len(parts) >= 8 and description:
        parts[7] = description.replace("\t", " ")[:120]
    lines[-1] = "\t".join(parts)
    RESULTS_TSV.write_text("\n".join(lines) + "\n")


def append_crash_row(commit: str, description: str) -> None:
    header = "commit\tsharpe\tsharpe_ci_low\tmax_dd_pct\tpnl_usd\ttrades\tstatus\tdescription"
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text(header + "\n")
    with open(RESULTS_TSV, "a") as f:
        desc = description.replace("\t", " ")[:120]
        f.write(f"{commit}\t0.0000\t0.0000\t0.00\t0.00\t0\tcrash\t{desc}\n")


def promote_checkpoints(commit: str, ci_low: float) -> list[str]:
    BEST_DIR.mkdir(parents=True, exist_ok=True)
    promoted = []
    for src in sorted(CHECKPOINTS.glob("last_seed*.pt")):
        dst = BEST_DIR / f"{commit}_ci{ci_low:+.4f}_{src.name}"
        shutil.copy2(src, dst)
        promoted.append(dst.name)
    return promoted


def write_best_json(commit: str, metrics: dict[str, str]) -> None:
    BEST_JSON.write_text(json.dumps({
        "commit": commit,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
    }, indent=2))


def safe_reset_head_minus_1() -> None:
    """git reset --hard HEAD~1 BUT preserve results.tsv.

    results.tsv is tracked, so a reset would wipe the row evaluator just
    appended (and that we just status-stamped). Capture it, reset, restore.
    """
    backup = RESULTS_TSV.read_text() if RESULTS_TSV.exists() else None
    git(["reset", "--hard", "HEAD~1"])
    if backup is not None:
        RESULTS_TSV.write_text(backup)


def push_to_origin() -> None:
    """Push current state + auto-generated docs to origin/main. Idempotent.

    Stages docs/*.png + README.md + results.tsv + iterations/ (if changed),
    commits, pushes. Failures are non-fatal.
    """
    try:
        subprocess.run(
            ["git", "add", "README.md", "docs", "results.tsv", "iterations"],
            cwd=REPO, check=False,
        )
        diff = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO).returncode
        if diff != 0:
            subprocess.run(
                ["git", "commit", "-m", "docs: refresh iteration page + charts + results"],
                cwd=REPO, check=False, capture_output=True,
            )
        subprocess.run(["git", "push", "origin", "main"], cwd=REPO, check=False, capture_output=True)
    except Exception as e:
        print(f"  push_to_origin failed: {e}", flush=True)


README_START = "<!-- LATEST_ITER_START -->"
README_END = "<!-- LATEST_ITER_END -->"


def update_readme_for_iteration(
    iter_num: int, commit: str, description: str, status: str, metrics: dict[str, str],
) -> None:
    """Replace the LATEST_ITER block in README with the latest iteration summary.

    The block embeds the latest equity chart (so it shows on the GitHub homepage)
    and links to the full iter_*.md detail page.
    """
    readme = REPO / "README.md"
    if not readme.exists():
        return

    badge = {"keep": "🟢 KEEP — new best", "discard": "🔴 DISCARD",
             "crash": "💥 CRASH"}.get(status, status)

    # Best record (from best.json if it exists)
    best_block = ""
    if BEST_JSON.exists():
        try:
            best = json.loads(BEST_JSON.read_text())
            bm = best.get("metrics", {})
            best_block = (
                f"### Current best (`{best.get('commit','?')}`)\n\n"
                f"| metric | value |\n|---|---|\n"
                f"| Sharpe (median) | **{bm.get('sharpe','—')}** |\n"
                f"| Sharpe CI low (5%) | {bm.get('sharpe_ci_low','—')} |\n"
                f"| Net PnL | **${bm.get('pnl_usd','0')}** ({bm.get('pnl_pct','0')}%) |\n"
                f"| Max drawdown | {bm.get('max_dd_pct','0')}% |\n"
                f"| Trades | {bm.get('trades','0')} |\n"
                f"| Saved at | {best.get('timestamp','—')} |\n\n"
                f"![weighted equity, current best](docs/weighted_latest.png)\n\n"
                f"![first month](docs/weighted_1m_latest.png)\n\n"
            )
        except Exception:
            pass

    iter_filename = f"iter_{iter_num:03d}_{commit}.md"
    block = (
        f"{README_START}\n\n"
        f"_Last iteration: **{time.strftime('%Y-%m-%d %H:%M UTC')}** · `{commit}` · {badge}_  \n"
        f"📄 **[Full iteration report → iterations/{iter_filename}](iterations/{iter_filename})** · "
        f"📁 [all iterations](iterations/)\n\n"
        f"### Latest iteration: iter {iter_num:03d} — {commit}\n\n"
        f"{badge} · {description}\n\n"
        f"| metric | value |\n|---|---|\n"
        f"| Sharpe (median) | **{metrics.get('sharpe','—')}** |\n"
        f"| Sharpe CI low (5%) | {metrics.get('sharpe_ci_low','—')} |\n"
        f"| Net PnL | **${metrics.get('pnl_usd','0')}** ({metrics.get('pnl_pct','0')}%) |\n"
        f"| Max drawdown | {metrics.get('max_dd_pct','0')}% |\n"
        f"| Trades | {metrics.get('trades','0')} |\n"
        f"| Wall time | {metrics.get('elapsed_seconds','—')}s |\n\n"
        f"![iteration equity](docs/weighted_{commit}.png)\n\n"
        f"{best_block}"
        f"### Progress over all experiments\n\n"
        f"![progress](docs/progress.png)\n\n"
        f"{README_END}\n"
    )

    txt = readme.read_text()
    if README_START in txt and README_END in txt:
        before = txt.split(README_START, 1)[0].rstrip() + "\n\n"
        after = "\n" + txt.split(README_END, 1)[1].lstrip()
        readme.write_text(before + block + after)
    else:
        # First time: append at end
        readme.write_text(txt.rstrip() + "\n\n" + block)


def next_iter_number() -> int:
    """Iteration number = number of rows in results.tsv (excluding header).

    Increments monotonically across runs. Used purely for filenames.
    """
    if not RESULTS_TSV.exists():
        return 1
    with open(RESULTS_TSV) as f:
        return max(1, sum(1 for _ in f))  # header counts as 1; first row → 1, etc.


def write_iteration_md(
    iter_num: int, commit: str, description: str, status: str, reason: str,
    metrics: dict[str, str], elapsed: float, run_log_excerpt: str,
) -> Path:
    """Write iterations/iter_<NNN>_<commit>.md with full result + chart links.

    Updates iterations/latest.md as a copy so README can link to a stable URL.
    """
    ITERATIONS_DIR.mkdir(exist_ok=True)
    fname = f"iter_{iter_num:03d}_{commit}.md"
    p = ITERATIONS_DIR / fname

    # Status badge
    badge = {"keep": "🟢 KEEP", "discard": "🔴 DISCARD", "crash": "💥 CRASH"}.get(status, status)

    # Try to find the seed-by-seed lines from run.log
    seed_lines = []
    for ln in run_log_excerpt.splitlines():
        if "[evaluator] seed" in ln and "sharpe=" in ln:
            seed_lines.append(ln.strip())

    # Chart paths (relative from iterations/ → ../docs/)
    weighted_full = f"../docs/weighted_{commit}.png"
    weighted_1m = f"../docs/weighted_1m_{commit}.png"

    md = []
    md.append(f"# iter {iter_num:03d} — {commit}")
    md.append("")
    md.append(f"**{badge}** · {description}")
    md.append("")
    md.append(f"_{time.strftime('%Y-%m-%d %H:%M UTC')} · {elapsed:.0f}s wall_")
    md.append("")
    md.append("## Result")
    md.append("")
    md.append(f"| metric | value |")
    md.append(f"|---|---|")
    md.append(f"| Sharpe (median) | **{metrics.get('sharpe', '—')}** |")
    md.append(f"| Sharpe CI low (5%) | {metrics.get('sharpe_ci_low', '—')} |")
    md.append(f"| Sharpe CI high (95%) | {metrics.get('sharpe_ci_high', '—')} |")
    md.append(f"| Net PnL | **${metrics.get('pnl_usd', '0')}** ({metrics.get('pnl_pct', '0')}%) |")
    md.append(f"| Max drawdown | {metrics.get('max_dd_pct', '0')}% |")
    md.append(f"| Trades | {metrics.get('trades', '0')} |")
    md.append(f"| Fees | ${metrics.get('fees_usd', '0')} |")
    md.append(f"| Seeds completed | {metrics.get('seeds_completed', '?')} |")
    md.append("")
    md.append(f"**Decision reason:** {reason}")
    md.append("")
    if seed_lines:
        md.append("## Per-seed details")
        md.append("")
        md.append("```")
        md.extend(seed_lines)
        md.append("```")
        md.append("")
    md.append("## Equity curve (full eval window, ~73 days)")
    md.append("")
    md.append(f"![weighted equity]({weighted_full})")
    md.append("")
    md.append("## Equity curve (first month)")
    md.append("")
    md.append(f"![weighted 1m]({weighted_1m})")
    md.append("")
    md.append("## Diff vs previous experiment")
    md.append("")
    md.append("```diff")
    try:
        diff = subprocess.check_output(
            ["git", "show", "--no-color", "--stat", "--format=%h %s%n%n%b", commit],
            cwd=REPO, text=True,
        )
        md.append(diff.strip())
    except Exception:
        md.append("(commit info unavailable)")
    md.append("```")
    md.append("")
    md.append(f"---")
    md.append(f"")
    md.append(f"[← all iterations](.) · [back to README](../README.md)")
    p.write_text("\n".join(md) + "\n")

    # Update latest.md so README can link to a stable filename
    LATEST_LINK.write_text(p.read_text())

    return p


def main() -> None:
    description = sys.argv[1] if len(sys.argv) > 1 else "(no description)"
    commit = git(["rev-parse", "--short=7", "HEAD"])
    print(f"=== iter @ {commit}  {description}", flush=True)

    t0 = time.time()
    proc = subprocess.run(
        [str(REPO / ".venv" / "bin" / "python"), str(REPO / "evaluator.py")],
        cwd=REPO, capture_output=True, text=True, timeout=12 * 3600,
    )
    elapsed = time.time() - t0
    RUN_LOG.write_text(proc.stdout + "\n--STDERR--\n" + proc.stderr)

    if proc.returncode != 0:
        print(f"STATUS=crash exit={proc.returncode} elapsed={elapsed:.0f}s", flush=True)
        append_crash_row(commit, description)
        safe_reset_head_minus_1()
        return

    metrics = parse_canonical(proc.stdout)
    try:
        ci_low = float(metrics["sharpe_ci_low"])
        dd = float(metrics["max_dd_pct"])
        sharpe = float(metrics.get("sharpe", "0"))
        trades = int(metrics.get("trades", "0"))
    except (KeyError, ValueError) as e:
        print(f"STATUS=crash reason=parse_failed: {e}", flush=True)
        append_crash_row(commit, description)
        safe_reset_head_minus_1()
        return

    iter_num = next_iter_number()
    prior_best = best_kept_ci_low()
    if dd < DD_FLOOR:
        status, reason = "discard", f"dd={dd:+.2f} < {DD_FLOOR}"
        update_last_row_status(status, description)
        write_iteration_md(iter_num, commit, description, status, reason, metrics, elapsed, proc.stdout)
        safe_reset_head_minus_1()
    elif trades <= 0 or abs(sharpe) < 1e-9:
        # 0-trade results are uninformative; never KEEP them even if ci_low=0
        # technically "improves" over -2.5. Discard so the experiment doesn't
        # become a poison-pill baseline.
        status, reason = "discard", f"trades={trades} sharpe={sharpe:+.3f} — strategy didn't trade"
        update_last_row_status(status, description)
        write_iteration_md(iter_num, commit, description, status, reason, metrics, elapsed, proc.stdout)
        safe_reset_head_minus_1()
    elif ci_low > prior_best:
        status, reason = "keep", f"ci_low={ci_low:+.4f} > prior best {prior_best:+.4f}"
        update_last_row_status(status, description)
        promoted = promote_checkpoints(commit, ci_low)
        write_best_json(commit, metrics)
        write_iteration_md(iter_num, commit, description, status, reason, metrics, elapsed, proc.stdout)
        update_readme_for_iteration(iter_num, commit, description, status, metrics)
        print(f"  promoted {len(promoted)} checkpoint(s) → {BEST_DIR}", flush=True)
    else:
        status, reason = "discard", f"ci_low={ci_low:+.4f} ≤ prior best {prior_best:+.4f}"
        update_last_row_status(status, description)
        write_iteration_md(iter_num, commit, description, status, reason, metrics, elapsed, proc.stdout)
        safe_reset_head_minus_1()

    # For non-keep iterations, still update README to show the latest iteration page link
    if status != "keep":
        update_readme_for_iteration(iter_num, commit, description, status, metrics)

    # Auto-push docs + README + iterations + results.tsv so GitHub stays in sync.
    push_to_origin()

    print(
        f"STATUS={status}  COMMIT={commit}  SHARPE={sharpe:+.3f}  "
        f"CI_LOW={ci_low:+.4f}  DD={dd:+.2f}  ELAPSED={elapsed:.0f}s  REASON={reason}",
        flush=True,
    )


if __name__ == "__main__":
    main()
