"""Render a chart of training loss falling, from checkpoints/last_train_loss.jsonl.

Usage:
    .venv/bin/python scripts/chart_training_loss.py [output_png]

Default output: docs/training_loss_latest.png

Read the live JSONL written by supervised_pretrain (one row per print interval)
and chart NLL + multi-horizon NLL + ranking loss over training steps.
Works even while pretrain is still running — re-run the script to refresh.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
LOG_PATH = REPO / "checkpoints" / "last_train_loss.jsonl"
OUT_DEFAULT = REPO / "docs" / "training_loss_latest.png"


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def render(rows: list[dict], out_path: Path) -> None:
    if not rows:
        print(f"no rows in {LOG_PATH}", file=sys.stderr)
        return
    x = list(range(1, len(rows) + 1))
    nll = [r.get("nll_running_mean", r.get("nll", 0.0)) for r in rows]
    has_mh = any("mh_nll" in r for r in rows)
    has_rank = any("rank" in r for r in rows)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(x, nll, color="#1f77b4", linewidth=1.6, label="NLL (Gaussian)")
    if has_mh:
        mh = [r.get("mh_nll", float("nan")) for r in rows]
        ax.plot(x, mh, color="#ff7f0e", linewidth=1.4, label="multi-horizon NLL", alpha=0.85)
    if has_rank:
        rk = [r.get("rank", float("nan")) for r in rows]
        ax2 = ax.twinx()
        ax2.plot(x, rk, color="#2ca02c", linewidth=1.2, linestyle="--",
                 label="ranking loss (right axis)", alpha=0.85)
        ax2.set_ylabel("ranking loss", color="#2ca02c")
        ax2.tick_params(axis="y", labelcolor="#2ca02c")

    last_row = rows[-1]
    epoch = last_row.get("epoch", 1)
    frac = last_row.get("frac_through", 1.0)
    pct = frac * 100
    last_nll = last_row.get("nll_running_mean", last_row.get("nll", 0.0))
    title = (
        f"Pretrain training loss — {len(rows)} samples (every batch print) · "
        f"epoch {epoch} · {pct:.0f}% through epoch · current NLL {last_nll:+.4f}"
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("training-progress sample (≈50/epoch)")
    ax.set_ylabel("NLL loss")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    if has_rank:
        ax2.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"wrote {out_path} ({len(rows)} samples)")


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT_DEFAULT
    rows = load_rows(LOG_PATH)
    if not rows:
        print(f"No log data at {LOG_PATH} — pretrain has not written any samples yet.")
        print("(Older runs / cached pretrain do not produce live loss data.)")
        sys.exit(0)
    render(rows, out)


if __name__ == "__main__":
    main()
