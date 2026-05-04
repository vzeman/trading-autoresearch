# iter 154 — 6ccdd60

**🔴 DISCARD** · exp154: top2 with 81.25pct reserve

_2026-05-05 01:30 UTC · 370s wall_

## Result

| metric | value |
|---|---|
| Sharpe (median) | **+3.213** |
| Sharpe CI low (5%) | +0.900 |
| Sharpe CI high (95%) | +5.683 |
| % time above SPY | 59.366% |
| Net PnL | **$+4277.70** (+8.555%) |
| Max drawdown | -2.05% |
| Trades | 1 |
| Fees | $1.00 |
| Seeds completed | 3 |

**Decision reason:** objective=+1.1636 ≤ prior best +1.1643 (ci_low=+0.9000, over_spy=59.4%, pnl=+8.55%)

## Winning strategy

Canonical strategy for this iteration: **top4 cross-sectional picker** — rank symbols by the transformer's 4h + 1d forecast Sharpe, buy the top four once enough symbols are ready, hold through the eval window, and keep 1 median trades after costs.

A **seed** is one independent training/evaluation run with a different random initialization and sampling path. The gate uses median/worst-tail statistics across seeds so one lucky seed cannot define the best checkpoint.

Positive seed transaction tables are shown later in this report; losing or flat seed transaction tables are omitted to keep reports focused on actionable winners.

## Per-seed details

```
[evaluator] seed 0: sharpe=+3.213  dd=-2.05%  pnl=$+4,277.70  trades=1
[evaluator] seed 1: sharpe=+3.213  dd=-2.05%  pnl=$+4,277.70  trades=1
[evaluator] seed 2: sharpe=+0.596  dd=-1.40%  pnl=$+268.71  trades=1
```

## Equity curve (full eval window, ~73 days)

![weighted equity](../docs/weighted_6ccdd60.png)

## Equity curve (first month)

![weighted 1m](../docs/weighted_1m_6ccdd60.png)

## Strategy comparison (equity curves)

Overlays every profile (intraday/intraweek/intramonth/longterm + 
daily-capped/weekly-capped/monthly-capped trade-frequency variants 
+ topN pickers + SPY benchmark) on one chart, using the median-seed run.

![strategy comparison](../docs/profile_compare_6ccdd60.png)

## Recent live-style simulations vs SP500

Each chart rebases the winning strategy and SP500 to $50,000 at the start of the trailing window, ending at the latest available bar.

### Trailing 1 day

![winning strategy trailing 1 day](../docs/winning_1d_6ccdd60.png)

### Trailing 1 week

![winning strategy trailing 1 week](../docs/winning_1w_6ccdd60.png)

### Trailing 1 month

![winning strategy trailing 1 month](../docs/winning_1mo_6ccdd60.png)

### Trailing 3 months

![winning strategy trailing 3 months](../docs/winning_3mo_6ccdd60.png)

### Trailing 6 months

![winning strategy trailing 6 months](../docs/winning_6mo_6ccdd60.png)

## Trader profile comparison

Same trained model, different time-horizon strategies + SPY benchmark + passive top-N pickers.

| profile | sharpe | PnL ($) | PnL % | trades | DD % | horizon |
|---|---:|---:|---:|---:|---:|---:|
| **daily_capped** | -2.008 | $-8.79 | -0.02% | 2 | -0.02% | 1d |
| **intraday** | -12.965 | $-6,935.75 | -13.87% | 4993 | -13.87% | 2h |
| **intramonth** | +0.000 | $+0.00 | +0.00% | 2 | -0.04% | 30d |
| **intraweek** | -4.735 | $-2,650.27 | -5.30% | 981 | -5.70% | 5d |
| **longterm** | +0.000 | $+0.00 | +0.00% | 2 | -0.04% | 30d |
| **monthly_capped** | +0.000 | $+0.00 | +0.00% | 0 | +0.00% | 30d |
| **spy_buyhold** | +0.980 | $+378.15 | +0.76% | 1 | -1.84% | - |
| **top10_picker** | +1.286 | $+1,409.36 | +2.82% | 9 | -2.85% | - |
| **top1_picker** | +0.000 | $+0.00 | +0.00% | 1 | -1.72% | - |
| **top20_picker** | +0.969 | $+722.50 | +1.45% | 19 | -2.71% | - |
| **top3_picker** | +2.288 | $+4,133.60 | +8.27% | 2 | -2.80% | - |
| **top4_picker** | +0.480 | $+267.90 | +0.54% | 3 | -2.54% | - |
| **top5_picker** | +1.519 | $+2,916.79 | +5.83% | 4 | -2.76% | - |
| **weekly_capped** | -0.536 | $-246.41 | -0.49% | 68 | -2.39% | 5d |

**Best active strategy: `top3_picker` (sharpe +2.288) — BEATS SPY ✓**

## Out-of-symbol holdout eval

Tested on **JPM, WMT, V, DIS, JNJ** — large-caps the model NEVER saw during training.

| seed | sharpe | PnL | trades | DD% |
|---:|---:|---:|---:|---:|
| 0 | +0.451 | $+163.39 | 5 | -1.79% |
| 1 | +0.337 | $+122.94 | 11 | -1.80% |
| 2 | +0.451 | $+163.39 | 5 | -1.79% |
| 3 | +0.327 | $+504.54 | 5 | -9.19% |
| 4 | +0.000 | $+0.00 | 0 | +0.00% |

**Median holdout sharpe: +0.337** (vs in-symbol +3.213)

## Transactions

_(no profitable per-seed transaction table; losing/flat seeds omitted)_

## Diff vs previous experiment

```diff
6ccdd60 exp154: top2 with 81.25pct reserve



 experiment.py | 4 ++--
 1 file changed, 2 insertions(+), 2 deletions(-)
```

---

[← all iterations](.) · [back to README](../README.md)
