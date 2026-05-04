# iter 112 — d5b2c32

**🟢 KEEP** · exp112: reward time above SPY benchmark

_2026-05-04 04:59 UTC · 737s wall_

## Result

| metric | value |
|---|---|
| Sharpe (median) | **+1.550** |
| Sharpe CI low (5%) | -0.398 |
| Sharpe CI high (95%) | +4.333 |
| % time above SPY | 28.039% |
| Net PnL | **$+4750.35** (+9.501%) |
| Max drawdown | -8.77% |
| Trades | 2 |
| Fees | $2.00 |
| Seeds completed | 3 |

**Decision reason:** objective=-0.3700 > prior best -0.3976 (ci_low=-0.3980, over_spy=28.0%)

## Per-seed details

```
[evaluator] seed 0: sharpe=+1.550  dd=-8.77%  pnl=$+4,750.35  trades=3
[evaluator] seed 1: sharpe=+2.113  dd=-7.77%  pnl=$+6,098.08  trades=2
[evaluator] seed 2: sharpe=+0.000  dd=+0.00%  pnl=$+0.00  trades=0
```

## Equity curve (full eval window, ~73 days)

![weighted equity](../docs/weighted_d5b2c32.png)

## Equity curve (first month)

![weighted 1m](../docs/weighted_1m_d5b2c32.png)

## Strategy comparison (equity curves)

Overlays every profile (intraday/intraweek/intramonth/longterm + 
daily-capped/weekly-capped/monthly-capped trade-frequency variants 
+ topN pickers + SPY benchmark) on one chart, using the median-seed run.

![strategy comparison](../docs/profile_compare_d5b2c32.png)

## Trader profile comparison

Same trained model, different time-horizon strategies + SPY benchmark + passive top-N pickers.

| profile | sharpe | PnL ($) | PnL % | trades | DD % | horizon |
|---|---:|---:|---:|---:|---:|---:|
| **daily_capped** | -2.005 | $-58.28 | -0.12% | 2 | -0.12% | 1d |
| **intraday** | -12.965 | $-24,204.37 | -48.41% | 5210 | -48.41% | 2h |
| **intramonth** | -0.920 | $-99.24 | -0.20% | 2 | -0.23% | 30d |
| **intraweek** | -4.723 | $-8,460.17 | -16.92% | 981 | -17.64% | 5d |
| **longterm** | +0.000 | $+0.00 | +0.00% | 2 | -0.23% | 30d |
| **monthly_capped** | +0.000 | $+0.00 | +0.00% | 0 | +0.00% | 30d |
| **spy_buyhold** | +0.999 | $+1,437.18 | +2.87% | 1 | -6.95% | - |
| **top10_picker** | +1.244 | $+3,337.96 | +6.68% | 9 | -8.27% | - |
| **top1_picker** | +0.000 | $+0.00 | +0.00% | 0 | +0.00% | - |
| **top20_picker** | +1.061 | $+1,840.82 | +3.68% | 18 | -8.81% | - |
| **top3_picker** | +2.088 | $+8,072.73 | +16.15% | 2 | -10.42% | - |
| **top4_picker** | +1.008 | $+2,038.88 | +4.08% | 3 | -6.66% | - |
| **top5_picker** | +1.455 | $+3,845.88 | +7.69% | 4 | -6.97% | - |
| **weekly_capped** | -1.634 | $-2,486.75 | -4.97% | 88 | -6.16% | 5d |

**Best active strategy: `top3_picker` (sharpe +2.088) — BEATS SPY ✓**

## Out-of-symbol holdout eval

Tested on **JPM, WMT, V, DIS, JNJ** — large-caps the model NEVER saw during training.

| seed | sharpe | PnL | trades | DD% |
|---:|---:|---:|---:|---:|
| 0 | +0.212 | $+223.49 | 5 | -6.67% |
| 1 | -0.143 | $-279.59 | 11 | -6.18% |
| 2 | +0.212 | $+223.49 | 5 | -6.67% |
| 3 | +0.327 | $+504.54 | 5 | -9.19% |
| 4 | +0.000 | $+0.00 | 0 | +0.00% |

**Median holdout sharpe: +0.212** (vs in-symbol +1.550)

## Transactions

_(no profitable per-seed transaction table; losing/flat seeds omitted)_

## Diff vs previous experiment

```diff
d5b2c32 exp112: reward time above SPY benchmark



 autoresearch_driver.py | 48 +++++++++++++++++++++++++++++++++++++-----------
 evaluator.py           |  5 ++++-
 experiment.py          |  4 ++--
 results.tsv            |  2 +-
 4 files changed, 44 insertions(+), 15 deletions(-)
```

---

[← all iterations](.) · [back to README](../README.md)
