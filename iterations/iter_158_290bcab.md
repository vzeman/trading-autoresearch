# iter 158 — 290bcab

**🟢 KEEP** · exp158: top2 with 82.34375pct reserve

_2026-05-05 01:58 UTC · 372s wall_

## Result

| metric | value |
|---|---|
| Sharpe (median) | **+3.214** |
| Sharpe CI low (5%) | +0.904 |
| Sharpe CI high (95%) | +5.684 |
| % time above SPY | 60.058% |
| Net PnL | **$+4030.24** (+8.060%) |
| Max drawdown | -1.93% |
| Trades | 1 |
| Fees | $1.00 |
| Seeds completed | 3 |

**Decision reason:** objective=+1.1648 > prior best +1.1645 (ci_low=+0.9040, over_spy=60.1%, pnl=+8.06%)

## Winning strategy

Canonical strategy for this iteration: **top4 cross-sectional picker** — rank symbols by the transformer's 4h + 1d forecast Sharpe, buy the top four once enough symbols are ready, hold through the eval window, and keep 1 median trades after costs.

A **seed** is one independent training/evaluation run with a different random initialization and sampling path. The gate uses median/worst-tail statistics across seeds so one lucky seed cannot define the best checkpoint.

Positive seed transaction tables are shown later in this report; losing or flat seed transaction tables are omitted to keep reports focused on actionable winners.

## Per-seed details

```
[evaluator] seed 0: sharpe=+3.214  dd=-1.93%  pnl=$+4,030.24  trades=1
[evaluator] seed 1: sharpe=+3.214  dd=-1.93%  pnl=$+4,030.24  trades=1
[evaluator] seed 2: sharpe=+0.596  dd=-1.32%  pnl=$+252.98  trades=1
```

## Equity curve (full eval window, ~73 days)

![weighted equity](../docs/weighted_290bcab.png)

## Equity curve (first month)

![weighted 1m](../docs/weighted_1m_290bcab.png)

## Strategy comparison (equity curves)

Overlays every profile (intraday/intraweek/intramonth/longterm + 
daily-capped/weekly-capped/monthly-capped trade-frequency variants 
+ topN pickers + SPY benchmark) on one chart, using the median-seed run.

![strategy comparison](../docs/profile_compare_290bcab.png)

## Recent live-style simulations vs SP500

Each chart rebases the winning strategy and SP500 to $50,000 at the start of the trailing window, ending at the latest available bar.

### Trailing 1 day

![winning strategy trailing 1 day](../docs/winning_1d_290bcab.png)

### Trailing 1 week

![winning strategy trailing 1 week](../docs/winning_1w_290bcab.png)

### Trailing 1 month

![winning strategy trailing 1 month](../docs/winning_1mo_290bcab.png)

### Trailing 3 months

![winning strategy trailing 3 months](../docs/winning_3mo_290bcab.png)

### Trailing 6 months

![winning strategy trailing 6 months](../docs/winning_6mo_290bcab.png)

## Trader profile comparison

Same trained model, different time-horizon strategies + SPY benchmark + passive top-N pickers.

| profile | sharpe | PnL ($) | PnL % | trades | DD % | horizon |
|---|---:|---:|---:|---:|---:|---:|
| **daily_capped** | -2.102 | $-8.31 | -0.02% | 2 | -0.02% | 1d |
| **intraday** | -12.965 | $-6,451.50 | -12.90% | 4701 | -12.90% | 2h |
| **intramonth** | +0.000 | $+0.00 | +0.00% | 2 | -0.04% | 30d |
| **intraweek** | -5.111 | $-2,580.19 | -5.16% | 981 | -5.31% | 5d |
| **longterm** | +0.000 | $+0.00 | +0.00% | 2 | -0.04% | 30d |
| **monthly_capped** | +0.000 | $+0.00 | +0.00% | 0 | +0.00% | 30d |
| **spy_buyhold** | +0.980 | $+356.08 | +0.71% | 1 | -1.73% | - |
| **top10_picker** | +1.287 | $+1,328.29 | +2.66% | 9 | -2.68% | - |
| **top1_picker** | +0.000 | $+0.00 | +0.00% | 1 | -1.62% | - |
| **top20_picker** | +0.968 | $+679.72 | +1.36% | 19 | -2.55% | - |
| **top3_picker** | +2.288 | $+3,895.21 | +7.79% | 2 | -2.64% | - |
| **top4_picker** | +0.484 | $+254.79 | +0.51% | 3 | -2.39% | - |
| **top5_picker** | +1.523 | $+2,748.46 | +5.50% | 4 | -2.60% | - |
| **weekly_capped** | -0.491 | $-224.28 | -0.45% | 67 | -2.23% | 5d |

**Best active strategy: `top3_picker` (sharpe +2.288) — BEATS SPY ✓**

## Out-of-symbol holdout eval

Tested on **JPM, WMT, V, DIS, JNJ** — large-caps the model NEVER saw during training.

| seed | sharpe | PnL | trades | DD% |
|---:|---:|---:|---:|---:|
| 0 | +0.459 | $+157.12 | 5 | -1.69% |
| 1 | +0.354 | $+121.99 | 11 | -1.69% |
| 2 | +0.459 | $+157.12 | 5 | -1.69% |
| 3 | +0.327 | $+504.54 | 5 | -9.19% |
| 4 | +0.000 | $+0.00 | 0 | +0.00% |

**Median holdout sharpe: +0.354** (vs in-symbol +3.214)

## Transactions

_(no profitable per-seed transaction table; losing/flat seeds omitted)_

## Diff vs previous experiment

```diff
290bcab exp158: top2 with 82.34375pct reserve



 experiment.py | 4 ++--
 1 file changed, 2 insertions(+), 2 deletions(-)
```

---

[← all iterations](.) · [back to README](../README.md)
