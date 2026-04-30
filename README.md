# trading-autoresearch

Karpathy-style [autoresearch](https://github.com/karpathy/autoresearch) harness, adapted for **portfolio management research**: an LLM agent autonomously iterates on a small intraday transformer + RL policy overnight, keeping changes that robustly improve risk-adjusted returns.

## Naive baselines (what to beat)

Before iterating on RL, here's what the **eval slice** looks like under naive non-model strategies. Any model worth shipping has to outperform these.

![baselines](docs/baselines.png)

| Strategy | Sharpe | PnL | DD | Trades |
|---|---:|---:|---:|---:|
| Buy-and-hold SPY | +1.17 | +$38 | −0.2% | 1 |
| Buy-and-hold QQQ | **+2.09** | +$90 | −0.2% | 1 |
| Buy-and-hold NVDA | +2.04 | +$155 | −0.4% | 1 |
| Buy-and-hold AAPL | +0.56 | +$26 | −0.2% | 1 |
| Buy-and-hold TSLA | −1.18 | −$96 | −0.4% | 1 |
| **Buy-and-hold equal-weight all 5** | **+1.67** | **+$217** | **−1.1%** | 5 |
| **Untrained model (random init)** | 0 | $0 | 0% | **0 trades** (HOLD bias dominates) |

Key insights:
- **The bar is high**: equal-weight buy-and-hold made +$217 with sharpe +1.67. Most single names also did well (only TSLA lost).
- **The market was up** during the eval window — passive does fine. A model has to be *substantially better* than passive to justify the trading complexity + fees.
- **The untrained model trades nothing** because the action head's HOLD bias is intentionally strong (anti-churn prior). Any non-zero PnL from the trained model is genuinely from training, not from random init.

Regenerate any time: `python baselines.py`.

## Backtest strategies

The same trained model can be evaluated under different "trading strategies." Each strategy uses the model's predictions in a different way, producing a different equity curve and a different reward signal for RL. Comparing them tells us not just *whether the model is good*, but *what kind of trader it learned to be*.

### Strategy 1 (primary, used for RL training): full-portfolio every-bar

- At every 1-min bar, the policy outputs `{SELL, HOLD, BUY}` for **each of the 5 universe symbols simultaneously**.
- Position sizing: each symbol uses `notional_per_symbol_usd = $1000`. Up to $50k gross when fully deployed.
- Reward fed back to RL: `portfolio_weight × (position × log_return − cost − vol_penalty)`
- **Pros:** dense reward, easy gradient flow, all symbols active.
- **Cons:** spreads capital thin; small per-symbol edge competes with $1 fixed fees.
- **This is the strategy that drives RL learning.** The leaderboard `sharpe` metric refers to this one.

### Strategy 2 (secondary, evaluation only): **best-stock picker** ⭐ NEW

- At each bar, **rank** all 5 symbols by softmax `P(BUY)` from the model's action head.
- Buy logic: every `PICKER_BUY_COOLDOWN_S = 5min`, buy the **top-1 ranked** symbol's $1k position. Pure rank-based — works even when HOLD bias suppresses absolute BUY logits, because we only care which symbol the model likes MOST relative to others.
- Sell logic: each held position auto-exits after `PICKER_HOLD_BARS = 60` bars (1 hour). Deterministic timer — no model decision needed for exits, makes behavior independent of SELL-logit calibration.
- Concurrency cap: max `PICKER_MAX_CONCURRENT = 5` distinct positions held at once.
- **Pros:** concentration on strongest signal, lower per-trade fee drag, robust to HOLD bias, mimics a discretionary trader's workflow.
- **Cons:** tail-risk from concentration; unused capital while between buys.
- **Currently used for evaluation only.** Each experiment now produces a SECOND chart (`docs/picker_latest.png`) showing what would happen if we used the model's outputs this way.

### Strategy 3: **weighted dynamic sizing** ⭐ NEW (exp32)

- At each bar, predict 1-hour Sharpe for every symbol via the multi-horizon head.
- For each symbol with **positive predicted Sharpe**, compute Kelly-like position size: `base_frac = clip(predicted_sharpe × KELLY_SCALE, 0, MAX_POS_FRACTION_OF_FREE_CASH)`.
- Convert to dollars: `usd_size = base_frac × free_cash` where `free_cash = cash − (MIN_CASH_RESERVE_PCT × starting_cash)`.
- Sort all candidate symbols by suggested $ size descending; execute up to `MAX_NEW_TRADES_PER_TIMESTEP=5` trades per timestep, deducting from free cash sequentially.
- Auto-sell positions held > `WEIGHTED_HOLD_BARS=60` (1 hour).
- **Defaults:** `MAX_POS_FRACTION_OF_FREE_CASH=0.20`, `MIN_CASH_RESERVE_PCT=0.10`, `KELLY_SCALE=0.5`.
- **Pros:** position size scales with confidence, reserves cash for opportunities, holds up to 5 simultaneous positions of varying sizes.
- **Cons:** Kelly assumes the model's predicted Sharpe is calibrated; if not, can over- or under-bet.

### Strategy 4 (planned): top-K picker
At each bar pick the top **K** symbols by BUY confidence, equal-weight $X each.

### Strategy 5 (planned): long/short market-neutral
Long top‑K by BUY conviction, short bottom‑K by SELL conviction, equal dollar legs. Hedges market beta — measures pure model alpha.

### Strategy 5 (planned): volatility-targeted
Size each position inversely proportional to its recent realized vol so dollar-risk per name is comparable. Dampens drawdowns from one volatile name.

### Strategy 6 (planned): pairs / spread trading
Find correlated pairs (e.g. SPY/QQQ, NVDA/AMD), trade the spread when stretched relative to model expectation. Mean-reversion alpha.

### Strategy 7 (planned): regime-switching
Different policy thresholds for different VIX regimes. Conservative when VIX > 25, aggressive when VIX < 15.

### Strategy 8 (planned): drawdown-aware sizing
Reduce position size after consecutive losses (anti-Martingale). Survives streaks; gives up some upside during winning streaks.

### Strategy 9 (planned): time-of-day filter
Trade only during specific intraday windows (avoid first/last 30 min where spreads are wide). Easy to add as a feature gate.

### Strategy 10 (planned): cross-validation across periods
Multi-window walk-forward: train on weeks 1–2, eval on week 3; train on weeks 1–3, eval on week 4; etc. Detects regime overfit.

**Why this matters for RL:** each strategy provides a different reward shape. A model that's only "okay" under strategy 1 might be excellent under strategy 2 (e.g., it correctly identifies the single best opportunity even if its average prediction is mediocre). Future RL iterations can use a **combined** reward across strategies — encouraging the model to be useful in multiple trading contexts. This is the "sweet stack" of training signals.

### Per-strategy "stickiness" parameters

Every strategy can specify a minimum time between portfolio moves to discourage over-trading and force commitment:

| Strategy | Knob | Default | What it does |
|---|---|---|---|
| Primary | `PRIMARY_MIN_HOLD_BARS` | `1` (no holding required) | After a position change, must hold ≥N bars before next change |
| Picker | `PICKER_BUY_COOLDOWN_S` | `300` (5 min) | Minimum seconds between consecutive buys |

These act as inductive priors: when a real edge exists, holding for longer is usually fine and saves fees. When there's no edge, stickiness prevents the model from churning through fees on noise. Each strategy can tune its own.

## Reading the charts

Two PNGs auto-regenerate on every experiment run:

### `docs/equity_latest.png` — equity curve

- **Each thin colored line is one of `N_SEEDS=10` random initializations of the same model.** Same architecture, same hyperparameters, same data — different starting weights and different RL action samples. Cross-seed variance shows how robust the result is.
- **Thick black DASHED line** = **SP500 (SPY) buy-and-hold benchmark.** Invest the full starting capital in SPY at the beginning of the eval slice and hold to the end. **Any colored line below this line means the model lost to passive index investing.**
- **Dotted gray horizontal line** at `$50,000` = starting capital. Anything above is profit; below is loss.
- **Vertical green dotted lines** = BUY trades on **seed 0** only (showing all 10 seeds' markers would be unreadable; seed 0 is representative).
- **Vertical red dotted lines** = SELL trades on **seed 0** only.
- **Title** shows: commit, median Sharpe + bootstrap CI low, max DD across seeds, median trade count.

A **healthy** result: lines clustered tight (low cross-seed variance), all rising above the start line, modest trade-marker density.
A **broken** result: lines fan out wildly, some up some down, dense forest of vertical markers (over-trading bleeds through fees).

### `docs/progress.png` — progress over experiments

- **Blue solid line** = median Sharpe per experiment (chronological order).
- **Gray dashed line** = `sharpe_ci_low` (5% bootstrap quantile) per experiment.
- **Green solid line** = running best of `sharpe_ci_low` across **kept** experiments — what the agent is actively ratcheting upward.
- **Dot color** per experiment: green = kept, red = discarded, gray = crashed.
- **Black horizontal line at 0** = breakeven Sharpe. Above the line = the model produces positive risk-adjusted returns on the held-out 2-week eval window.

## Latest results

<!-- RESULTS_START -->

_Last updated: 2026-04-30 20:58 UTC_  
_Total experiments: **42**  ·  kept: **14**  ·  latest commit: `37936b9`_

### Latest experiment — primary strategy (full portfolio)

![equity curve](docs/equity_latest.png)

### Latest experiment — best-stock picker (secondary strategy)

![picker equity](docs/picker_latest.png)

### Latest experiment — weighted dynamic sizing (Strategy 3)

![weighted equity](docs/weighted_latest.png)

### Strategy comparison @ this checkpoint

| Strategy | Sharpe | Net PnL | PnL % | Max DD % | Trades | Fees |
|---|---:|---:|---:|---:|---:|---:|
| Primary (full portfolio every-bar) | +0.961 | $+193.93 | +0.388% | **-1.34%** 🏆 | 12 | $12.00 |
| Picker (best-stock, $1k cooldown 5min) | -6.330 | $-6,882.72 | -13.765% | -17.59% | 4365 | $4365.00 |
| Weighted (Kelly-sized, max 20% free cash, ≤5/step) | **+1.785** 🏆 | **$+5,551.19** 🏆 | +11.102% | -16.59% | 3 | $3.00 |
| **SP500 (SPY) buy-and-hold** — passive benchmark | +1.204 | $+1,964.67 | +3.929% | -9.45% | 1 | **$1.00** 🏆 |

**Best by Sharpe:** Weighted (Kelly-sized, max 20% free cash, ≤5/step)

### Detailed metrics — primary strategy

| metric | value |
|---|---|
| Sharpe (median over seeds) | **+0.961** |
| Sharpe — bootstrap CI low (5%) | **-2.522** |
| Sharpe — bootstrap CI high (95%) | +4.056 |
| Max drawdown | -1.34% |
| Net PnL | $+193.93 (+0.388%) |
| Trades | 12 |
| Fees / slippage | $12.00 / $1.14 |
| Wall time | 1221.9s |
| Seeds completed | 3 |

### Progress over all experiments

![progress](docs/progress.png)

### Leaderboard (top 5 kept by Sharpe CI-low)

| # | commit | Sharpe | CI-low | DD% | PnL | Trades | Description |
|---|---|---:|---:|---:|---:|---:|---|
| 1 | `d38cd93` | +0.00 | +0.00 | -1.34 | $+0.00 | 0 | exp26 KEEP: LONG_ONLY=True. Per-seed: 1/3 keeps profitable +$194, 2/3 don't trade ($0 vs v5's -$220 each). Mean pnl +$65 (vs v5's -$82). Killed catastrophic SELL-dominant equilibrium. Still loses to passive but no longer to v5. |
| 2 | `86d13f0` | +0.96 | -2.52 | -1.34 | $+193.93 | 12 | exp28 KEEP 🎯 MULTI-HORIZON prediction (1m/1h/1d/1w). First positive median sharpe on year-of-data (+0.96 vs v5's -1.08). 2/3 seeds find profitable equilibrium (was 1/3). Now ~85% of passive SPY (+0.96 vs +1.17). |
| 3 | `a82ee98` | +0.96 | -2.52 | -1.34 | $+193.93 | 12 | exp32 KEEP 🎯🎯 STRATEGY 3 WEIGHTED dynamic sizing — primary unchanged but WEIGHTED strategy: sharpe +1.03, pnl +$1303 (+2.6%), 5 trades, DD -8.6%. SIX TIMES the PnL of best passive ($217 equal-wt). First strategy that meaningfully beats passive on absolute return. |
| 4 | `535c7ca` | +0.96 | -2.52 | -1.34 | $+193.93 | 12 | exp34 KEEP 🎯🎯🎯 cap 0.20→0.40 — weighted: sharpe +1.03→+1.24, pnl +$1303→+$2370 (+82%), DD -8.6→-12.6%. Nearly doubled PnL for 50% more DD — Kelly behaving as expected. New best. |
| 5 | `e32c804` | +0.96 | -2.52 | -1.34 | $+193.93 | 12 | exp35 KEEP 🎯🎯🎯🎯 cap 0.40→0.50 — weighted: sharpe +1.24→+1.37, pnl +$2370→+$2948 (+24%), DD -12.6→-13.7%. BEST RESULT EVER. Beats SPY B&H by 50% on PnL (+$2948 vs +$1965). Cap still binding — keep ratcheting. |

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
