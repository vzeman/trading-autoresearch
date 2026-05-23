# Markov Regime Quant Strategy Experiment

Date: 2026-05-23

## Winning Strategy In This Iteration

There is no deployable winning strategy yet.

The best single 2026 result came from the daily Markov-regime strategy:

- universe: cached 40-symbol 15-minute dataset, no rank filter
- cadence: daily, because the source method forecasts next-day regimes
- regime labels: 20-trading-day return
  - bull: `>= +5%`
  - bear: `<= -5%`
  - sideways: between those thresholds
- walk-forward training: every target day uses only prior daily regime
  transitions for that symbol
- signal: `P(next_state=bull) - P(next_state=bear)`
- position sizing: confidence-weighted long allocation, max 3 positions
- cost assumption: 8 bps round trip

Locked 2026 result, 40-symbol universe:

- best variant: `manual`
- total return: `+22.41%`
- SPY return: `+9.78%`
- max drawdown: `-10.35%`
- Sharpe: `2.17`
- trades: `276`

This is interesting, but not tradable by itself because the same fixed method
failed prior years.

## Transcript-Derived Method

Source video:

`https://www.youtube.com/watch?v=ZVMTeDBmSrI`

Transcript downloaded to:

`artifacts/transcripts/ZVMTeDBmSrI.txt`

The useful quant framework extracted from the transcript:

- Do not trade from chart feelings. Convert market behavior into numerical
  regimes.
- Label each asset into bull, sideways, or bear states.
- Estimate a 3x3 transition matrix from historical state transitions.
- Forecast next-day state probabilities from the current state.
- Use state persistence/stickiness as the core edge hypothesis.
- Generate a scalar trade signal as `P(bull) - P(bear)`.
- Size exposure by signal strength.
- Use walk-forward backtesting so the matrix never learns from future data.
- Check subjective state labels against an unlabeled/adaptive regime detector.

The script implements the HMM idea conservatively as an adaptive, no-lookahead
regime confirmation using expanding quantiles. A full HMM library is not
available in this local environment, so this is not represented as a true
Gaussian HMM.

## Implementation

New script:

`evaluate_markov_regime_quant_strategy.py`

Primary daily command:

```bash
.venv/bin/python evaluate_markov_regime_quant_strategy.py \
  --dataset checkpoints/transformer_15m/shared_15m_40sym_algo.parquet \
  --output-dir checkpoints/transformer_15m/markov_regime_quant_40sym_daily_2026 \
  --trade-cadence daily \
  --top-symbols-limit 0
```

Important implementation note: the first test incorrectly applied a next-day
regime forecast every 15 minutes. That version was crushed by overtrading and
transaction costs. The daily cadence is the correct interpretation of the
method described in the transcript.

## Results

Daily 2026, top-10 universe:

- best variant: `confirmed_spy_gated`
- total return: `+22.51%`
- SPY return: `+9.78%`
- max drawdown: `-12.50%`
- Sharpe: `2.06`
- trades: `211`

Daily 2026, full cached 40-symbol universe:

- best variant: `manual`
- total return: `+22.41%`
- SPY return: `+9.78%`
- max drawdown: `-10.35%`
- Sharpe: `2.17`
- trades: `276`

Daily 2025, same 40-symbol method:

- best variant: `spy_gated`
- total return: `+1.56%`
- SPY return: `+6.41%`
- max drawdown: `-20.79%`
- Sharpe: `0.19`
- trades: `669`

Daily 2024, same 40-symbol method:

- best variant: `manual`
- total return: `-13.27%`
- SPY return: `+23.29%`
- max drawdown: `-33.20%`
- Sharpe: `-0.47`
- trades: `756`

Robustness sweep:

`checkpoints/transformer_15m/markov_regime_quant_sweep_2024_2026.csv`

The sweep tried stricter minimum signals, 1-3 positions, and lower exposure
across 2024, 2025, and 2026. No fixed configuration beat SPY across all three
folds. Lower exposure reduced losses but also reduced 2026 upside.

## Conclusion

The Markov-regime method is a useful feature and strategy component, not a
standalone trading system. It found a strong 2026 pocket, but failed the
cross-year robustness test. The right next step is to use Markov regime features
inside the broader allocator as:

- market/state context,
- a no-trade gate during weak SPY regimes,
- a confidence feature for position sizing,
- a candidate generator that still requires cross-sectional alpha confirmation.

Do not use this strategy live until it passes multi-year walk-forward tests with
positive active alpha, controlled drawdown, and stable behavior after costs.

## Adaptive Markov Follow-Up

Implemented after testing the fixed transition matrix:

- `evaluate_markov_regime_quant_strategy.py` can now read both:
  - single feature parquet files,
  - per-symbol parquet directories such as
    `data/world_model/cached193_shared500_full_xsec_intraday120`.
- `--regime-source adaptive` uses expanding, no-lookahead return quantiles as
  the active bull/sideways/bear regime source.
- `--transition-lookback-days` limits the transition matrix to recent history.
- `--transition-halflife-days` exponentially downweights older transitions.

The tested adaptive setting was:

```bash
--regime-source adaptive \
--transition-lookback-days 252 \
--transition-halflife-days 63
```

This makes the method adapt day by day:

1. At each date, only prior data is used.
2. The regime label can drift with the asset's own expanding distribution.
3. The transition matrix uses approximately the last trading year.
4. Recent transitions carry more weight than old transitions.

### Adaptive Results On Current Local Data

Compact 40-symbol transformer cache, 2026:

| method | best variant | return | SPY | alpha | max DD | trades |
|---|---:|---:|---:|---:|---:|---:|
| fixed | manual | +22.41% | +9.78% | +12.62% | -10.35% | 276 |
| adaptive | confirmed | +22.65% | +9.78% | +12.87% | -10.31% | 276 |

Broad 193-symbol current cache, 2025 YTD through 2025-11-05:

| method | best variant | return | SPY | alpha | max DD | trades |
|---|---:|---:|---:|---:|---:|---:|
| fixed | confirmed | -15.03% | +26.45% | -41.47% | -20.93% | 262 |
| adaptive | algo_fused_spy_gated | +12.69% | +26.45% | -13.75% | -7.12% | 162 |

Broad 193-symbol current cache, 2024:

| method | best variant | return | SPY | alpha | max DD | trades |
|---|---:|---:|---:|---:|---:|---:|
| fixed | algo_fused_spy_gated | +17.85% | +24.63% | -6.78% | -16.49% | 231 |
| adaptive | confirmed_spy_gated | -3.97% | +24.63% | -28.60% | -11.18% | 228 |

Interpretation:

- Adaptation clearly helped the broad 2025 cache, changing the strategy from
  strongly negative to positive and reducing drawdown.
- Adaptation slightly improved the compact 2026 result.
- Adaptation damaged 2024 performance, so the adaptive parameters cannot be
  chosen once and trusted blindly.

The next serious version should calibrate the adaptive parameters in a nested
walk-forward loop. Candidate parameters:

- regime source: fixed vs adaptive,
- transition lookback: 63, 126, 252, 504 days,
- transition half-life: 21, 63, 126 days,
- SPY gate on/off,
- exposure cap and max position count.

The rule must be selected on past folds only, then evaluated on a later locked
fold. Without this nested selection, the adaptive method can still overfit.

## Holding-Period Alternative Models

Added `evaluate_markov_holding_periods.py` to compare portfolio behavior when
signals are held for multiple days instead of rebalanced every day. The script
uses the latest local Alpaca cache and writes:

- `checkpoints/transformer_15m/markov_holding_periods_latest/summary.json`
- `checkpoints/transformer_15m/markov_holding_periods_latest/leaderboard.csv`
- `checkpoints/transformer_15m/markov_holding_periods_latest/equity_curves.csv`
- `checkpoints/transformer_15m/markov_holding_periods_latest/monthly_returns.csv`
- `docs/markov_holding_period_comparison.png`

Besides the existing Markov variants, this run added several leak-resistant
cross-sectional selectors. Every feature is shifted so the score for a target
day is based on information available before that day:

- `relative_momentum`: stock return minus SPY return over recent windows,
  penalized for realized volatility.
- `trend_quality`: recent absolute trend, range position, and volatility.
- `breakout_quality`: proximity to a 20-day breakout, volume confirmation, and
  volatility penalty.
- `defensive_trend`: relative strength with stronger volatility penalty.
- `hybrid_markov_trend`: adaptive Markov signal blended with trend quality and
  relative momentum.

Run command:

```bash
.venv/bin/python evaluate_markov_holding_periods.py \
  --dataset /Users/viktorzeman/.cache/trading-autoresearch \
  --output-dir checkpoints/transformer_15m/markov_holding_periods_latest \
  --eval-start 2026-01-01 \
  --eval-end 2026-05-22
```

### Latest 2026 Slice

The effective test starts in March because the adaptive Markov model requires
history before emitting signals. SPY returned +9.65% over the comparable active
period.

| rank | model / strategy | return | alpha vs SPY | max DD | Sharpe | avg hold |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `trend_quality_hold_3d` | +17.83% | +8.18% | -7.36% | 2.63 | 2.80d |
| 2 | `hybrid_markov_trend_hold_3d` | +14.76% | +5.11% | -9.58% | 2.00 | 2.80d |
| 3 | `relative_momentum_exit_max10` | +11.49% | +1.85% | -6.15% | 2.35 | 7.00d |
| 4 | `hybrid_markov_trend_exit_max10` | +6.99% | -2.65% | -13.45% | 1.19 | 4.11d |
| 5 | `spy_fused_hold_5d` | +6.39% | -3.26% | -6.92% | 1.70 | 4.59d |

Interpretation:

- Holding periods matter. The daily Markov rebalance was negative, while the
  best 3-day trend-quality portfolio beat SPY on this recent locked slice.
- Pure Markov is still not good enough by itself. The stronger results came
  from cross-sectional trend and relative-strength features.
- `relative_momentum_exit_max10` is interesting because it beat SPY with the
  lowest drawdown among the winners, but it traded less often and was active
  only when the SPY gate was positive.
- This is not yet a tradable final model. It is a promising candidate family
  that needs walk-forward validation across older folds and live paper trading
  before any real capital is used.
