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

    Stages docs/*.png + README.md + results.tsv (if changed), commits with
    a generic message if there's something to commit, then pushes.
    Failures are non-fatal — printed and swallowed.
    """
    try:
        subprocess.run(["git", "add", "README.md", "docs", "results.tsv"], cwd=REPO, check=False)
        # Commit only if there are staged changes
        diff = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO).returncode
        if diff != 0:
            subprocess.run(
                ["git", "commit", "-m", "docs: refresh charts + results after iter"],
                cwd=REPO, check=False, capture_output=True,
            )
        subprocess.run(["git", "push", "origin", "main"], cwd=REPO, check=False, capture_output=True)
    except Exception as e:
        print(f"  push_to_origin failed: {e}", flush=True)


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

    prior_best = best_kept_ci_low()
    if dd < DD_FLOOR:
        status, reason = "discard", f"dd={dd:+.2f} < {DD_FLOOR}"
        update_last_row_status(status, description)
        safe_reset_head_minus_1()
    elif trades <= 0 or abs(sharpe) < 1e-9:
        # 0-trade results are uninformative; never KEEP them even if ci_low=0
        # technically "improves" over -2.5. Discard so the experiment doesn't
        # become a poison-pill baseline.
        status, reason = "discard", f"trades={trades} sharpe={sharpe:+.3f} — strategy didn't trade"
        update_last_row_status(status, description)
        safe_reset_head_minus_1()
    elif ci_low > prior_best:
        status, reason = "keep", f"ci_low={ci_low:+.4f} > prior best {prior_best:+.4f}"
        update_last_row_status(status, description)
        promoted = promote_checkpoints(commit, ci_low)
        write_best_json(commit, metrics)
        print(f"  promoted {len(promoted)} checkpoint(s) → {BEST_DIR}", flush=True)
    else:
        status, reason = "discard", f"ci_low={ci_low:+.4f} ≤ prior best {prior_best:+.4f}"
        update_last_row_status(status, description)
        safe_reset_head_minus_1()

    # Auto-push docs + README + results.tsv so GitHub stays in sync.
    push_to_origin()

    print(
        f"STATUS={status}  COMMIT={commit}  SHARPE={sharpe:+.3f}  "
        f"CI_LOW={ci_low:+.4f}  DD={dd:+.2f}  ELAPSED={elapsed:.0f}s  REASON={reason}",
        flush=True,
    )


if __name__ == "__main__":
    main()
