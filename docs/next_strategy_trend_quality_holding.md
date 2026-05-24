# Next Possible Strategy: Trend-Quality Holding Portfolio

Status: promising research candidate, not live-trading approved.

The latest holding-period experiment suggests the next practical branch should
be a simple cross-sectional trend-quality portfolio rather than another pure
Markov/Jepa forecasting run. On the recent 2026 slice, the strongest result was
`trend_quality_hold_3d`: +17.83% portfolio return versus +9.65% for SPY, with
-7.36% max drawdown.

This strategy is intentionally simple:

1. Build daily features from the latest intraday cache.
2. Rank stocks cross-sectionally using only information available before the
   target day.
3. Buy the top three candidates.
4. Hold each entry for three trading days.
5. Re-rank and rotate after the holding period.

## Current Result

Test window: effective active period from March 2026 through 2026-05-22. The
script starts at 2026-01-01, but the adaptive Markov family requires enough
history before it emits comparable signals.

| rank | model / strategy | return | alpha vs SPY | max DD | Sharpe | avg hold |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `trend_quality_hold_3d` | +17.83% | +8.18% | -7.36% | 2.63 | 2.80d |
| 2 | `hybrid_markov_trend_hold_3d` | +14.76% | +5.11% | -9.58% | 2.00 | 2.80d |
| 3 | `relative_momentum_exit_max10` | +11.49% | +1.85% | -6.15% | 2.35 | 7.00d |
| benchmark | SPY buy-and-hold | +9.65% | baseline | | 2.83 | |

Chart:

![holding-period comparison](markov_holding_period_comparison.png)

Artifacts:

- `checkpoints/transformer_15m/markov_holding_periods_latest/summary.json`
- `checkpoints/transformer_15m/markov_holding_periods_latest/leaderboard.csv`
- `checkpoints/transformer_15m/markov_holding_periods_latest/equity_curves.csv`
- `checkpoints/transformer_15m/markov_holding_periods_latest/trades.csv`
- `checkpoints/transformer_15m/markov_holding_periods_latest/monthly_returns.csv`

## Feature Definition

All features are shifted so the score for day `t` uses only data known before
day `t`.

`trend_quality_score` combines:

- recent 5-day return,
- recent 20-day return,
- price position inside the recent 20-day range,
- a penalty for 20-day realized volatility.

The current implementation uses cross-sectional z-scores per date:

```text
trend_quality_score =
  0.40 * z(ret_5_prev)
+ 0.35 * z(ret_20_prev)
+ 0.25 * z(range_pos_20_prev)
- 0.35 * z(vol_20_prev)
```

The related variants are:

- `relative_momentum`: stock return minus SPY return, penalized for volatility.
- `breakout_quality`: proximity to a 20-day high plus volume confirmation.
- `defensive_trend`: relative strength with a stronger volatility penalty.
- `hybrid_markov_trend`: adaptive Markov signal blended with trend and relative
  momentum.

## Trading Rule Candidate

Candidate rule for the next branch:

1. Universe: current liquid cached S&P 500 symbols, excluding SPY as an entry.
2. Signal: `trend_quality_score > 0`.
3. Portfolio: top three ranked symbols.
4. Holding period: three trading days.
5. Weighting: equal weight, 100% gross exposure while active.
6. Cost model: 8 bps roundtrip cost in the current research script.
7. Benchmark: SPY buy-and-hold over the same active dates.

The lower-drawdown alternative to keep on the shortlist is
`relative_momentum_exit_max10`. It returned less than `trend_quality_hold_3d`
but had better max drawdown in the recent test.

## Reproduce

```bash
.venv/bin/python evaluate_markov_holding_periods.py \
  --dataset /Users/viktorzeman/.cache/trading-autoresearch \
  --output-dir checkpoints/transformer_15m/markov_holding_periods_latest \
  --eval-start 2026-01-01 \
  --eval-end 2026-05-22
```

The implementation lives in:

- `evaluate_markov_holding_periods.py`
- `evaluate_markov_regime_quant_strategy.py`

## Why This Is Next

The experiment changed the conclusion from "train longer" to "improve the
portfolio rule and feature family":

- Pure Markov rebalance was negative on the recent slice.
- Markov with SPY gating improved drawdown but still trailed SPY.
- Trend-quality and Markov+trend hybrids beat SPY on the recent slice.
- Holding period mattered: three days worked better than daily churn or longer
  five-day holds for the best trend-quality score.

This points to a practical next experiment: validate whether short holding
periods plus trend-quality ranking survive walk-forward testing.

## Required Validation Before Paper Trading

Do not promote this to a tradable model until it passes these gates:

1. Walk-forward folds across multiple years, not just 2026 YTD.
2. Frozen parameter selection: choose the scoring formula and holding period on
   past folds only, then test on later unseen folds.
3. Transaction-cost stress: 8 bps, 15 bps, and 25 bps roundtrip.
4. Concentration stress: max trades per symbol, symbol cooldown, and sector
   concentration limits.
5. Down-market stress: verify behavior when SPY is falling and when volatility
   is high.
6. Live paper trading: compare realized fills and slippage against the backtest.

## Next Implementation Steps

1. Add walk-forward evaluation for `trend_quality_hold_3d`,
   `hybrid_markov_trend_hold_3d`, and `relative_momentum_exit_max10`.
2. Add SPY/cash as an explicit idle allocation instead of treating no-position
   days as flat cash only.
3. Add volatility targeting so the strategy can reduce exposure when recent
   portfolio volatility rises.
4. Add a drawdown cooldown: reduce or stop exposure after portfolio drawdown
   breaches a calibrated threshold.
5. Compare equal-weight top three against volatility-weighted top three.

## Gap-Up Follow-Up

Added a gap-up-over-resistance diagnostic in
`docs/gap_up_resistance_strategy.md`.

The first pass found that directly buying gap-ups over 20-day or 50-day
resistance was poor on the recent 2026 slice. The better use of gap data was as
a risk filter for the original strategy: penalize stocks whose previous gap-up
failed intraday.

Latest comparison:

| strategy | return | SPY | alpha | max DD |
|---|---:|---:|---:|---:|
| `trend_quality_avoid_failed_gap_hold_3d` | +29.90% | +9.65% | +20.25% | -7.34% |
| `trend_quality_hold_3d` | +17.83% | +9.65% | +8.18% | -7.36% |

This keeps the original strategy shape, but adds gap-failure awareness. It
needs walk-forward validation before it can replace the original candidate.

## High-Volume Multi-Year Follow-Up

Added a liquidity-filtered multi-year diagnostic in
`docs/liquid_multiyear_volume_strategy.md`.

The test re-ran the holding strategy family across 2021 through 2026 YTD and
restricted entries to higher-volume stocks. The strict top-10
volume-and-valuation universe was much stronger than the broader `$20M` daily
dollar-volume threshold.

Best strict-liquid candidate:

| strategy | avg return | avg alpha vs SPY | SPY-beating folds | worst alpha | worst DD |
|---|---:|---:|---:|---:|---:|
| `trend_quality_hold_3d` | +25.65% | +14.61% | 5 / 6 | -17.14% | -41.50% |

Conclusion: high-volume filtering helps, but the system is still not
tradable-ready. The next fix should be an adaptive market-risk gate so the
portfolio reduces exposure or switches strategy during SPY bear regimes like
2022.

## Volume-Shape Signal Follow-Up

Added volume-shape features to the holding strategy family. The new signal
rewards constructive volume build-up and penalizes one-day blowoff spikes:

```text
volume_shape_score =
  0.30 * z(avg_volume_5_prev / avg_volume_20_prev)
+ 0.25 * z(avg_dollar_volume_5_prev / avg_dollar_volume_20_prev)
+ 0.25 * z(ret_5_prev * avg_volume_5_prev / avg_volume_20_prev)
+ 0.15 * z(previous_day_volume / avg_volume_20_prev)
- 0.20 * z(max(previous_day_volume / avg_volume_20_prev - 2.5, 0))
```

New tested variants:

- `trend_quality_volume_shape_hold_3d`
- `trend_quality_avoid_failed_gap_volume_shape_hold_3d`
- `relative_momentum_volume_shape_exit_max10`

Result: volume shape is useful but conditional. In the strict top-10 universe,
`trend_quality_volume_shape_hold_3d` averaged +10.30% alpha versus +14.61% for
the original `trend_quality_hold_3d`. It did help specific folds, such as 2023
top-10 and the broader `$20M` universe in 2024-2026 for the failed-gap variant.
Use it as an ensemble signal, not as a full replacement for trend quality.

## Volume-State Transformer Gate

Added `train_volume_state_transformer.py` and documented the experiment in
`docs/volume_state_transformer_gate.md`.

This trains a separate iTransformer-style market-state model from 40-day
market-wide volume-shape sequences. Its output is shifted forward one trading
session and used as a risk gate for `trend_quality_hold_3d`.

Final top-10 walk-forward result with threshold `0.80`:

| strategy | avg return | avg alpha vs SPY | SPY-beating folds | worst alpha | avg DD | worst DD |
|---|---:|---:|---:|---:|---:|---:|
| `trend_quality_hold_3d` | +20.37% | +11.51% | 4 / 5 | -17.14% | -26.24% | -41.50% |
| `trend_quality_hold_3d_volume_state_gate` | +17.44% | +8.58% | 4 / 5 | -13.18% | -23.76% | -39.14% |

Follow-up: added a leader-aware gate after analyzing why 2025 failed. The
failure was not that risk was fake. The risky days had a much higher actual
risk-label rate, but the top liquid leaders still outperformed SPY. The new
rule blocks only when the transformer sees risk and the top trend-quality
leaders do not have enough relative strength.

Updated result:

| strategy | avg return | avg alpha vs SPY | worst alpha | avg DD | worst DD |
|---|---:|---:|---:|---:|---:|
| `trend_quality_hold_3d` | +20.37% | +11.51% | -17.14% | -26.24% | -41.50% |
| `trend_quality_hold_3d_volume_state_leader_gate` | +20.35% | +11.49% | -13.18% | -23.64% | -39.14% |

Conclusion: leader-aware volume-state gating is not a return booster yet, but
it preserves return while improving worst-case behavior. Next version should
learn the leader-strength/action threshold walk-forward rather than fixing it
at `0.24`.
