# trading-autoresearch

Karpathy-style [autoresearch](https://github.com/karpathy/autoresearch) harness, adapted for **portfolio management research**: an LLM agent autonomously iterates on a small intraday transformer + RL policy overnight, keeping changes that robustly improve risk-adjusted returns.

## Latest results

<!-- RESULTS_START -->

_Last updated: 2026-04-30 07:49 UTC_  
_Total experiments: **15**  ·  kept: **6**  ·  latest commit: `3c5a1c7`_

### Latest experiment

![equity curve](docs/equity_latest.png)

| metric | value |
|---|---|
| Sharpe (median over seeds) | **+2.348** |
| Sharpe — bootstrap CI low (5%) | **-6.483** |
| Sharpe — bootstrap CI high (95%) | +10.170 |
| Max drawdown | -0.30% |
| Net PnL | $+75.39 (+0.151%) |
| Trades | 13 |
| Fees / slippage | $13.00 / $2.61 |
| Wall time | 305.1s |
| Seeds completed | 10 |

### Progress over all experiments

![progress](docs/progress.png)

### Leaderboard (top 5 kept by Sharpe CI-low)

| # | commit | Sharpe | CI-low | DD% | PnL | Trades | Description |
|---|---|---:|---:|---:|---:|---:|---|
| 1 | `aeff147` | -0.39 | -4.63 | -0.31 | $-13.06 | 7 | exp1: HOLD bias 3.0→1.0 — ci_low improved -5.61→-4.63, DD -9.18→-0.31% |
| 2 | `4a6dea7` | -0.32 | -5.61 | -9.18 | $-10.56 | 5 | baseline (v2 features, HOLD bias 3.0) |
| 3 | `dff38d6` | +2.31 | -6.49 | -0.30 | $+74.27 | 14 | exp10 KEEP 🎯 HOLD bias 1.0→1.5 — sharpe +2.06→+2.31, per-seed range collapsed to [+2.28,+2.35], all DDs -0.30%, trades 13-15. Two discrete equilibria. New best. |
| 4 | `8616861` | +2.06 | -6.66 | -0.32 | $+66.02 | 21 | exp7 KEEP 🚀 RL_LR 3e-5→2e-5 — ALL 10 SEEDS POSITIVE. Median sharpe +2.06, all DD ≤-0.32%, pnl +$56-$73, trades 15-27. First profitable AND stable config. |
| 5 | `ed1886c` | -0.65 | -9.70 | -0.31 | $-12.98 | 8 | exp6 KEEP: RL_LR 1e-5→3e-5 — sharpe -1.08→-0.65 (+0.43), per-seed range tightened to [-0.78,-0.43], trades 5-11. Best so far. |

<!-- RESULTS_END -->

```
You wake up. The agent ran 87 experiments while you slept.
The leaderboard is sorted by Sharpe lower-CI. The top three are reproducible.
```

## What's different from Karpathy's original

| Original `autoresearch` | This repo |
|---|---|
| LLM training (`train.py` → val_bpb) | Trading model + RL policy (`experiment.py` → portfolio Sharpe) |
| Single deterministic metric | Multi-seed median Sharpe + bootstrap CI lower bound |
| One file, one metric | One file, **one primary metric + one hard constraint** (max DD ≥ −10%) |
| H100 GPU expected | MPS / CUDA / CPU all fine; ~5 min per experiment on M-series MacBook |
| Data baked into prepare.py (FineWeb) | Free yfinance 1-min bars for 5 liquid US names, cached locally |

The **risk** of running an LLM agent against a backtest is overfitting to a single eval window. We push back with three guardrails:

1. **Bootstrap CI on Sharpe** — improvements have to be statistically real.
2. **Multi-seed runs** — RL is stochastic; median of 3 seeds, not best of 3.
3. **Hard drawdown constraint** — Sharpe-only optimization can hide tail risk.

## File layout

```
prepare.py     # frozen — data download, broker, metrics, train/eval split
experiment.py  # the file the agent edits — model + RL policy + train loop
evaluator.py   # frozen — runs experiment with N seeds, prints canonical metrics
program.md     # the agent's instructions (the "skill")
results.tsv    # append-only public log of every experiment (committed)
docs/          # auto-generated equity + progress charts (committed)
pyproject.toml # uv / pip dependencies
```

## Quick start

```bash
# 1. Install (Python 3.10+; uv recommended)
pip install -e .

# 2. Cache the data (one-time, ~30s for 5 symbols × 28d × 1m bars)
python prepare.py

# 3. Single experiment with the baseline experiment.py (~3-5 min on MPS)
python evaluator.py
```

Expected output ends with:

```
--- canonical ---
sharpe:           +0.123
sharpe_ci_low:    -0.456
sharpe_ci_high:   +0.789
max_dd_pct:       -1.23
pnl_usd:          +12.34
pnl_pct:          +0.025
trades:           42
fees_usd:         42.00
slippage_usd:     5.61
elapsed_seconds:  287.4
seeds_completed:  3
---
```

## Running the agent loop

Open this repo in [Claude Code](https://claude.com/claude-code) (or another agent harness with shell + edit access), then:

```
Hi, please read program.md and let's start a new autoresearch run.
```

The agent will:
- Create branch `autoresearch/<tag>`
- Initialize `results.tsv` with the header
- Run the baseline once
- Loop forever: hypothesize → edit `experiment.py` → run evaluator → keep/discard → log

It runs until you interrupt it (Ctrl-C). On wake-up, sort `results.tsv` by `sharpe_ci_low` to see what worked.

## What the agent CAN and CANNOT change

See `program.md` — short version: the agent owns `experiment.py` (features + model + policy + training); it must NEVER touch `prepare.py` (the simulator) or `evaluator.py` (the contract). Adding new pip dependencies is forbidden. There's no hard time budget per experiment, but the agent should aim for ~3 min per seed (~10 min total) so the loop iterates quickly.

## The contract

`experiment.py` MUST export:

```python
def train_and_eval(seed: int) -> tuple[
    list[tuple[pd.Timestamp, float]],   # equity curve from broker.equity_curve on EVAL slice
    int,                                  # n_trades
    float,                                # total_fees
    float,                                # total_slippage
]: ...
```

If this signature breaks, the evaluator will crash and the experiment will be auto-discarded.

## Honest limitations

1. **yfinance free tier exposes ~30d of 1-min bars** — the eval window is ~9 trading days. Statistical significance is genuinely limited; treat any single overnight result as exploratory.
2. **No live trading** — this is a research harness; the broker is a simulator. Wiring to a real broker is left to the reader.
3. **Per-tick HFT it isn't** — 1-min bars are coarse. The architecture would extend down to seconds with a paid data feed.
4. **The agent is biased toward what the LLM has seen in pretraining**; it'll suggest standard moves first (LR sweeps, deeper nets, dropout). Genuinely novel architectures are rare.

## Inspiration

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — the original. Read his README + `program.md` first.
- [vzeman/trading](https://github.com/vzeman/trading) — sister repo with the IBKR portfolio-management skills (different focus, manual workflows).

## License

MIT — copy, fork, modify, anything.
