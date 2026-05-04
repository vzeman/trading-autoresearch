# iter 153 — d5b40ef

**🟢 KEEP** · exp153: SPY-alpha objective with top2 82.5pct reserve

_2026-05-05 01:24 UTC · 372s wall_

## Result

| metric | value |
|---|---|
| Sharpe (median) | **+3.214** |
| Sharpe CI low (5%) | +0.904 |
| Sharpe CI high (95%) | +5.684 |
| % time above SPY | 60.114% |
| Net PnL | **$+3994.86** (+7.990%) |
| Max drawdown | -1.91% |
| Trades | 1 |
| Fees | $1.00 |
| Seeds completed | 3 |

**Decision reason:** objective=+1.1642 > prior best +1.1617 (ci_low=+0.9040, over_spy=60.1%, pnl=+7.99%)

## Winning strategy

Canonical strategy for this iteration: **top4 cross-sectional picker** — rank symbols by the transformer's 4h + 1d forecast Sharpe, buy the top four once enough symbols are ready, hold through the eval window, and keep 1 median trades after costs.

A **seed** is one independent training/evaluation run with a different random initialization and sampling path. The gate uses median/worst-tail statistics across seeds so one lucky seed cannot define the best checkpoint.

Positive seed transaction tables are shown later in this report; losing or flat seed transaction tables are omitted to keep reports focused on actionable winners.

## Per-seed details

```
[evaluator] seed 0: sharpe=+3.214  dd=-1.91%  pnl=$+3,994.86  trades=1
[evaluator] seed 1: sharpe=+3.214  dd=-1.91%  pnl=$+3,994.86  trades=1
[evaluator] seed 2: sharpe=+0.596  dd=-1.31%  pnl=$+250.73  trades=1
```

## Equity curve (full eval window, ~73 days)

![weighted equity](../docs/weighted_d5b40ef.png)

## Equity curve (first month)

![weighted 1m](../docs/weighted_1m_d5b40ef.png)

## Strategy comparison (equity curves)

Overlays every profile (intraday/intraweek/intramonth/longterm + 
daily-capped/weekly-capped/monthly-capped trade-frequency variants 
+ topN pickers + SPY benchmark) on one chart, using the median-seed run.

![strategy comparison](../docs/profile_compare_d5b40ef.png)

## Recent live-style simulations vs SP500

Each chart rebases the winning strategy and SP500 to $50,000 at the start of the trailing window, ending at the latest available bar.

### Trailing 1 day

![winning strategy trailing 1 day](../docs/winning_1d_d5b40ef.png)

### Trailing 1 week

![winning strategy trailing 1 week](../docs/winning_1w_d5b40ef.png)

### Trailing 1 month

![winning strategy trailing 1 month](../docs/winning_1mo_d5b40ef.png)

### Trailing 3 months

![winning strategy trailing 3 months](../docs/winning_3mo_d5b40ef.png)

### Trailing 6 months

![winning strategy trailing 6 months](../docs/winning_6mo_d5b40ef.png)

## Trader profile comparison

Same trained model, different time-horizon strategies + SPY benchmark + passive top-N pickers.

| profile | sharpe | PnL ($) | PnL % | trades | DD % | horizon |
|---|---:|---:|---:|---:|---:|---:|
| **daily_capped** | -2.102 | $-8.25 | -0.02% | 2 | -0.02% | 1d |
| **intraday** | -12.965 | $-6,374.62 | -12.75% | 4649 | -12.75% | 2h |
| **intramonth** | +0.000 | $+0.00 | +0.00% | 2 | -0.04% | 30d |
| **intraweek** | -5.269 | $-2,622.26 | -5.24% | 981 | -5.35% | 5d |
| **longterm** | +0.000 | $+0.00 | +0.00% | 2 | -0.04% | 30d |
| **monthly_capped** | +0.000 | $+0.00 | +0.00% | 0 | +0.00% | 30d |
| **spy_buyhold** | +0.980 | $+352.93 | +0.71% | 1 | -1.72% | - |
| **top10_picker** | +1.287 | $+1,316.69 | +2.63% | 9 | -2.66% | - |
| **top1_picker** | +0.000 | $+0.00 | +0.00% | 1 | -1.61% | - |
| **top20_picker** | +0.968 | $+673.60 | +1.35% | 19 | -2.53% | - |
| **top3_picker** | +2.288 | $+3,861.12 | +7.72% | 2 | -2.61% | - |
| **top4_picker** | +0.485 | $+252.89 | +0.51% | 3 | -2.37% | - |
| **top5_picker** | +1.523 | $+2,724.39 | +5.45% | 4 | -2.57% | - |
| **weekly_capped** | -0.803 | $-296.14 | -0.59% | 67 | -2.21% | 5d |

**Best active strategy: `top3_picker` (sharpe +2.288) — BEATS SPY ✓**

## Out-of-symbol holdout eval

Tested on **JPM, WMT, V, DIS, JNJ** — large-caps the model NEVER saw during training.

| seed | sharpe | PnL | trades | DD% |
|---:|---:|---:|---:|---:|
| 0 | +0.460 | $+156.19 | 5 | -1.68% |
| 1 | +0.356 | $+121.79 | 11 | -1.68% |
| 2 | +0.460 | $+156.19 | 5 | -1.68% |
| 3 | +0.327 | $+504.54 | 5 | -9.19% |
| 4 | +0.000 | $+0.00 | 0 | +0.00% |

**Median holdout sharpe: +0.356** (vs in-symbol +3.214)

## Transactions

_(no profitable per-seed transaction table; losing/flat seeds omitted)_

## Diff vs previous experiment

```diff
d5b40ef exp153: SPY-alpha objective with top2 82.5pct reserve



 autoresearch_driver.py | 29 +++++++++++++++++++----------
 experiment.py          |  4 ++--
 2 files changed, 21 insertions(+), 12 deletions(-)
```

---

[← all iterations](.) · [back to README](../README.md)
