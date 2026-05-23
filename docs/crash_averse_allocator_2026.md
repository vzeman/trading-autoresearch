# Crash-Averse Allocator 2026 Iteration

Date: 2026-05-15

## Why this iteration exists

The previous live watchlist ranked stocks that were actively selling off. The
main failure was that small predicted upside could outrank weak beat-SPY
confidence and crash risk. Those rows were not tradable-quality buy signals.

## Code changes

- Added `crash_averse` allocator utility in `train_allocator.py`.
- Added positive-class weighting for allocator top-label training. Without this,
  a top-10% classifier could report roughly 90% accuracy by predicting every row
  as not-top.
- Added `--train-entry-only` to train allocators only on cash-to-buy rows, which
  matches the deployment question: "from cash, should we buy this symbol?"
- Added strict live gates in `recommend_world_model_today.py`:
  - minimum predicted profit probability,
  - minimum predicted beat-SPY probability,
  - positive predicted alpha,
  - predicted drawdown cap,
  - observable momentum and volatility guards.

## Trained checkpoints

World model reused:

`checkpoints/world_model/rolling_retrain_top500_adjusted_liquid_xsec_q80_cap75/world_model_fold_2026.pt`

Experimental allocators:

`checkpoints/world_model/crash_averse_allocator_2026/allocator_fold_2026.pt`

`checkpoints/world_model/crash_averse_allocator_2026_weighted/allocator_fold_2026.pt`

`checkpoints/world_model/crash_averse_entry_allocator_2026/allocator_fold_2026.pt`

## Locked 2026 results

Best entry-only planner slice, not sequential:

- q90 active groups: 19
- mean portfolio return: +0.0037%
- beat-SPY rate: 57.9%
- mean alpha vs SPY: +0.1575%
- mean max drawdown: -0.0185%

Strict sequential simulation with q90 and idle capital in SPY:

- total return: +4.39%
- trades: 10
- active trade profit rate: 20%
- active beat-SPY rate: 50%
- max drawdown: -6.19%

This was not acceptable because the positive total return mostly came from idle
SPY exposure, not from good active trades.

Calibrated rule search with idle capital in cash:

- selected rule: q95, target <= 50%, horizon <= 30 bars, positive predicted alpha
- trades: 3
- active trade profit rate: 0%
- active beat-SPY rate: 66.7%
- total return: -0.024%
- max drawdown: -0.024%

This is safer, but still not profitable. It is not deployable.

## Live recommendation check

Using the crash-averse entry allocator on the latest refreshed Alpaca snapshot:

- decision timestamp: 2026-05-14 19:59:00 UTC
- scored rows: 10,060
- rows after calibrated score threshold: 0
- final recommendations: 0

## Conclusion

The current world model and allocator family should remain quarantined. The
right next step is not threshold tuning. The base world model must be retrained
to predict adverse excursion and crash outcomes directly, and validation must
require active-trade profitability across walk-forward folds before any paper
trading use.

## Crash-Aware Base World Model Follow-Up

Implemented on 2026-05-15 after the conclusion above:

- `world_model_dataset.py` now writes explicit adverse/crash targets for new
  datasets:
  - `future_min_asset_return`
  - `future_asset_max_drawdown`
  - `asset_crash_label`
  - `severe_adverse_label`
- `train_world_model.py` backfills those targets for older parquet datasets, so
  existing folds can train crash-aware heads without rebuilding the full cache.
- `evaluate_world_model.py` now loads checkpoint-specific target heads and
  includes predicted crash/adverse risk in `pred_score`.
- `train_allocator.py` now exposes the predicted crash/adverse outputs as
  allocator features.
- `recommend_world_model_today.py` now has live gates for predicted asset crash
  probability, severe adverse probability, minimum predicted asset return, and
  predicted asset drawdown.

New base world checkpoint:

`checkpoints/world_model/crash_aware_world_2026/world_model_fold_2026.pt`

Training details:

- rows: 3,889,116
- features: 103
- symbols: 386
- actions: 49
- regression heads: 6
- classification heads: 4
- early stopped after 4 epochs, best epoch 2

New entry-only crash-averse allocator:

`checkpoints/world_model/crash_aware_world_2026/allocator_entry_crash_averse_fold_2026.pt`

Strict locked-2026 calibrated cash evaluation:

- selected calibration rule: q50, target <= 50%, horizon <= 60 bars,
  predicted profit >= 0.55, predicted alpha >= 0.000238
- calibration sequential result: +0.22%, 11 trades, 63.6% profitable,
  63.6% beat-SPY, -0.12% max drawdown
- locked 2026 test result: -0.047%, 2 trades, 0% profitable,
  50.0% beat-SPY, -0.047% max drawdown

Latest live Alpaca check using this experimental model:

- decision timestamp: 2026-05-14 19:59:00 UTC
- scored rows: 10,060
- rows after calibrated score threshold: 0
- final recommendations: 0

Interpretation: adding crash/adverse heads improved the model shape and live
quarantine behavior, but the locked test did not prove active trade edge. This
checkpoint is useful research infrastructure, not a tradable model.

## Nested Calibration Gate

Added after the crash-aware locked test still failed:

- `evaluate_tradable_allocator.py` now supports nested rule validation via
  `--rule-validation-fraction`.
- Rule search can require:
  - minimum calibration trades,
  - minimum calibration return,
  - minimum calibration profit rate,
  - minimum calibration beat-SPY rate,
  - minimum validation trades,
  - minimum validation return,
  - minimum validation profit rate,
  - minimum validation beat-SPY rate.
- Rule search also has focused crash-head variants instead of a full Cartesian
  explosion across crash thresholds.

Strict nested run attempted:

```bash
.venv/bin/python evaluate_tradable_allocator.py \
  --calibration-data data/world_model/rolling_retrain_top500_adjusted_liquid_xsec_q80_cap75/fold_2026/train_data \
  --test-data data/world_model/rolling_retrain_top500_adjusted_liquid_xsec_q80_cap75/fold_2026/test_data \
  --world-checkpoint checkpoints/world_model/crash_aware_world_2026/world_model_fold_2026.pt \
  --allocator-checkpoint checkpoints/world_model/crash_aware_world_2026/allocator_entry_crash_averse_fold_2026.pt \
  --rule-mode calibrated \
  --rule-validation-fraction 0.25 \
  --min-calibration-trades 12 \
  --min-calibration-return 0.0005 \
  --min-calibration-profit-rate 0.55 \
  --min-calibration-beat-spy-rate 0.55 \
  --min-validation-trades 3 \
  --min-validation-return 0.0 \
  --min-validation-profit-rate 0.50 \
  --min-validation-beat-spy-rate 0.50 \
  --idle-asset cash
```

Result: no calibrated trade rule survived. The run intentionally failed with:

`RuntimeError: could not select calibrated trade rule`

Interpretation: this is a stronger quarantine signal. The model can sometimes
find calibration-only pockets, but those pockets do not survive a later
calibration holdout. Until a candidate passes nested calibration and then the
locked test with positive active-trade PnL, the system should output no buys.

## Cross-Year 2025 Replication Attempt

Trained the same crash-aware world model and entry-only crash-averse allocator
on the existing 2025 rolling fold:

World checkpoint:

`checkpoints/world_model/crash_aware_world_2025/world_model_fold_2025.pt`

Allocator checkpoint:

`checkpoints/world_model/crash_aware_world_2025/allocator_entry_crash_averse_fold_2025.pt`

World training:

- rows: 3,029,376
- train rows: 2,404,272
- validation rows: 588,048
- best epoch: 2
- best validation loss: 1.5520

Allocator training:

- entry-only rows: 371,505
- best epoch: 8
- validation planner was superficially positive:
  - all planner groups: +0.0185% mean return, 63.3% profitable
  - q90: +0.0352% mean return, 69.2% profitable, 76.9% beat-SPY

Locked 2025 planner test was weak/negative:

- all planner groups: -0.00085% mean return, 52.0% profitable
- q95: -0.00442% mean return, 70.0% profitable, but still negative after
  costs/selection

Nested calibration:

`checkpoints/world_model/crash_aware_world_2025/tradable_entry_crash_averse_2025_nested_no_trade.json`

Result:

- calibration groups: 656
- rule-search groups: 492
- rule-validation groups: 164
- selected rule: `no_trade_no_calibrated_rule`
- locked test trades: 0

## Daily Ranker / Hybrid Baseline Iteration

Added on 2026-05-15 after the action-conditioned world model still failed
locked tests.

New files/checkpoints:

- `train_daily_ranker.py`
- `evaluate_daily_rule_baseline.py`
- `checkpoints/daily_ranker/exp2_2025_regime_h5/daily_ranker.pt`
- `checkpoints/daily_ranker/exp2_2025_regime_h20/daily_ranker.pt`
- `checkpoints/daily_ranker/exp3_hybrid_h5/daily_ranker.pt`
- `checkpoints/daily_ranker/exp3_hybrid_h20/daily_ranker.pt`

Dataset/features:

- Daily bars are derived from cached Alpaca minute bars.
- Universe: cached top-500 symbols plus SPY, limited to 503 symbols in these
  runs.
- Features:
  - stock returns over 1, 5, 20, and 60 trading days,
  - 20/60 day realized volatility,
  - 20 day volume z-score,
  - distance from 20/60 day moving averages,
  - 60 day drawdown,
  - SPY returns over 1, 5, 20, and 60 trading days,
  - relative stock-vs-SPY returns over 5, 20, and 60 trading days,
  - daily cross-sectional ranks for return, volatility, drawdown, and volume.
- Labels:
  - future return over the selected horizon,
  - future minimum return during the horizon,
  - future alpha versus SPY,
  - profit label,
  - crash/adverse label,
  - risk-adjusted utility that rewards return/alpha and penalizes adverse
    excursion/crash.

Model:

- Two-layer MLP with LayerNorm/GELU/dropout.
- Heads:
  - utility regression,
  - profit classification,
  - crash classification.
- Rule search validates the trained score with observable gates:
  - predicted score quantile,
  - predicted profit minimum,
  - predicted crash maximum,
  - SPY 20 day regime filter,
  - stock 20 day trend filter,
  - stock-vs-SPY relative strength filter,
  - 60 day drawdown filter,
  - 20 day volatility-rank cap.

Important simulator correction:

- The original daily-ranker simulation compounded overlapping horizon labels
  every day, which was too optimistic.
- It now only opens a new selection after the horizon has elapsed and subtracts
  roundtrip trading cost.
- It also reports SPY return over the exact same active periods, so raw return
  cannot be confused with active edge.

2025 locked-year results:

- `exp2_2025_regime_h5`: -14.59% total return, -23.49% max drawdown,
  36.1% beat-SPY trade rate.
- `exp2_2025_regime_h20`: +0.76% total return, -17.40% max drawdown,
  35.1% beat-SPY trade rate.
- Rule-only baseline with strict validation gates selected no-trade for both
  5-day and 20-day horizons.

Hybrid observed-score iteration:

- Added `--observed-score-weight` to blend a small observed trend/risk score
  into the neural score.
- Observed score favors 20 day cross-sectional momentum, shallow drawdown,
  low volatility rank, and positive stock-vs-SPY 20 day relative strength.

Hybrid locked-year results:

- `exp3_hybrid_h5`, train before 2024, locked 2025:
  - strategy: +13.01%
  - SPY over active periods: +22.07%
  - active alpha: -9.07%
  - max drawdown: -11.05%
  - verdict: better raw return, still not active edge.
- `exp3_hybrid_h20`, train before 2024, locked 2025:
  - strategy: +10.07%
  - SPY over active periods: +36.73%
  - active alpha: -26.66%
  - max drawdown: -12.08%
  - verdict: not tradable.

Cross-year h5 hybrid check:

- Train before 2022, locked 2023:
  - strategy: -12.04%
  - SPY over active periods: +6.43%
  - active alpha: -18.47%
  - max drawdown: -19.09%
  - verdict: failed.
- Train before 2023, locked 2024:
  - strategy: +22.07%
  - SPY over active periods: +31.13%
  - active alpha: -9.06%
  - max drawdown: -6.93%
  - verdict: profitable but still inferior to SPY.

Conclusion:

The daily ranker/hybrid family is not tradable. It can find validation pockets
and sometimes produce positive raw return, but it has not beaten SPY across
unseen yearly folds. The current project gate should remain:

1. No live buys unless a model beats SPY active-period return across multiple
   locked yearly folds.
2. Require positive active alpha, not just positive raw return.
3. Keep no-trade as the default deployment behavior when no calibrated rule
   survives validation.
4. Next model work should focus on stronger walk-forward validation, better
   market-regime conditioning, and portfolio-level objectives rather than simply
   making the neural network larger.

## Nested Daily Ranker Follow-Up

Added after the hybrid ranker produced positive raw returns but negative active
alpha versus SPY.

Code changes:

- `train_daily_ranker.py` now supports nested rule validation inside the
  validation window:
  - rule-search slice chooses candidate thresholds,
  - later rule-holdout slice must independently pass,
  - locked test is only evaluated after a rule survives both slices.
- Rule validation now requires:
  - positive total return,
  - positive active alpha versus SPY over the same holding windows,
  - minimum profit rate,
  - minimum beat-SPY trade rate,
  - maximum drawdown cap.
- The simulator now reports `spy_active_return` and `active_alpha_return`.

Nested results on locked 2025:

- `exp4_nested_h5_2025`: no calibrated rule survived nested validation.
- `exp4_nested_h20_2025`: no calibrated rule survived nested validation.

Interpretation: the previous hybrid returns were validation-fragile. With
nested validation, the safe decision is no-trade.

## Market Context Features

Added market-wide context so each stock row can see how the broader universe is
moving, not just SPY:

- mean market returns over 1, 5, 20, and 60 days,
- median 20 day market return,
- mean stock-vs-SPY 20 day relative return,
- mean 20 day volatility,
- 20 day return dispersion,
- percentage of stocks positive over 20 days,
- percentage above 20 day moving average,
- percentage not in deeper than 10% 60 day drawdown,
- percentage in the lower half of volatility rank,
- stock return relative to market mean over 20 and 60 days.

Nested 2025 results:

- `exp5_market_context_h5_2025`: no calibrated rule survived nested validation.
- `exp5_market_context_h20_2025`: no calibrated rule survived nested validation.

Interpretation: market breadth features are useful context, but in this model
they overfit quickly and did not create stable active alpha.

## Top-Utility Ranking Head

Added a cross-sectional top-decile target:

- `top_utility_label` marks symbols in the top 10% of same-day future utility.
- The model now has four heads:
  - risk-adjusted utility regression,
  - profit probability,
  - crash probability,
  - top-decile utility probability.
- The score blends predicted utility, profit, top-decile probability, and crash
  risk.

Nested 2025 results:

- `exp6_tophead_h5_2025`: no calibrated rule survived nested validation.
- `exp6_tophead_h20_2025`: no calibrated rule survived nested validation.

Interpretation: directly training a cross-sectional top selector did not fix the
instability.

## Alpha-Only Target

Added `--utility-mode alpha` to train the utility head and top-decile label on
excess return versus SPY, rather than raw return.

Alpha utility:

- rewards future alpha versus SPY,
- penalizes adverse excursion,
- penalizes crash labels.

Nested 2025 results:

- `exp7_alpha_target_h5_2025`: no calibrated rule survived nested validation.
- `exp7_alpha_target_h20_2025`: no calibrated rule survived nested validation.

Interpretation: when the model is asked to prove actual alpha instead of raw
return, the current daily feature/model family still cannot find a tradable
rule. This is the right failure mode: no-trade is safer than forcing weak buys.

Current hard conclusion:

The latest daily ranker family is not tradable. The project now has stronger
validation machinery, richer market-context features, and a better alpha target,
but no candidate has beaten SPY across unseen year folds. The next serious
research pivot should be one of:

1. A true walk-forward ensemble that trains on multiple historical windows and
   only trades when independent folds agree.
2. More data depth: older Alpaca history or another source, including delisted
   names if possible to reduce survivorship bias.
3. A portfolio policy model trained on sequential portfolio state with explicit
   cash/SPY/stock allocation and turnover constraints, not independent row
   ranking.
4. Intraday-specific models evaluated on intraday execution assumptions rather
   than daily close-to-close labels.

## Long-History Alpha Target

Added after the post-2020 daily ranker repeatedly no-traded under nested
validation.

Data coverage check:

- local cache contains 513 symbols,
- about 190 symbols have useful coverage back to early 2016,
- nearly 500 symbols have useful coverage from mid-2020 onward,
- latest bars are mostly current through 2026-05-14.

Experiment:

- `exp8_long_history_alpha_h5_2025`
- cached-all universe, variable history back to 2016,
- alpha utility target,
- nested validation,
- 5-day horizon.

Locked-year results:

| Fold | Strategy | SPY active | Active alpha | Max DD | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| 2023 | -23.35% | +11.92% | -35.27% | -31.37% | failed |
| 2024 | +34.41% | +9.00% | +25.41% | -5.86% | strong |
| 2025 | +7.33% | +10.75% | -3.41% | -7.62% | failed alpha |

Interpretation:

Longer history helped enough to produce one genuinely strong locked year
(2024), but it did not generalize. The 2023 failure is severe and the 2025 fold
still lags SPY on active periods. This candidate remains research-only.

## Breadth-Gated Rule Search

Added market breadth rule gates:

- minimum percentage of stocks positive over 20 days,
- minimum percentage of stocks above 20 day moving average,
- minimum mean market 20 day return,
- maximum cross-sectional 20 day return dispersion.

Experiment:

- `exp9_breadth_gated_h5_*`
- same long-history alpha dataset,
- compact breadth-gated grid.

Locked-year results:

| Fold | Strategy | SPY active | Active alpha | Max DD | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| 2023 | -5.59% | +1.88% | -7.48% | -8.01% | improved but failed |
| 2024 | 0.00% | 0.00% | 0.00% | 0.00% | no-trade, missed winner |
| 2025 | +7.33% | +10.75% | -3.41% | -7.62% | failed alpha |

Interpretation:

Breadth gates reduced the 2023 damage but also killed the 2024 winner and did
not fix 2025. The code hooks are useful, but this gate configuration is not an
accepted model.

Updated conclusion:

The best current signal is conditional and regime-specific, not yet tradable.
The next improvement should not be another single-rule search. It should require
agreement across independently trained folds before allowing a live trade, or it
should pivot to a sequential portfolio policy that can explicitly choose SPY or
cash when stock-picking alpha is weak.

## Fold-Consensus Ranker Gate

Added `evaluate_daily_ranker_consensus.py`.

Purpose:

- Load multiple independently trained daily ranker checkpoints.
- Score the same target period with each checkpoint's own normalization and
  calibrated rule.
- Allow a trade only when enough checkpoints independently select the same
  symbol on the same date.

Consensus tests:

| Target | Checkpoints | Vote rule | Strategy | SPY active | Active alpha | Max DD | Verdict |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 2023 | train<2021 + train<2022 | 2 of 2 | 0.00% | 0.00% | 0.00% | 0.00% | no-trade, avoided bad 2023 |
| 2024 | train<2022 + train<2023 | 2 of 2 | +27.12% | +19.46% | +7.65% | -4.09% | positive alpha |
| 2025 | train<2022 + train<2023 + train<2024 | 2 of 3 | -4.40% | +13.74% | -18.14% | -10.62% | failed |
| 2025 | train<2022 + train<2023 + train<2024 | 3 of 3 | +5.24% | +4.97% | +0.27% | -3.81% | tiny positive alpha |

Interpretation:

Consensus is the most promising safety layer so far:

- it blocks the severe 2023 failure when older folds disagree,
- it preserves positive active alpha in 2024,
- strict 3-of-3 consensus barely clears active SPY in 2025.

However, the 2025 alpha margin is only +0.27 percentage points with a sub-50%
profit/beat-SPY trade rate. This is not tradable proof. The next required gate
is stronger: consensus must beat SPY across multiple locked folds with a minimum
active-alpha margin and a minimum trade count, or it should output no-trade.

## Historical Consensus Protocol

Tested a causal-ish protocol: each target year uses only checkpoints trained
with earlier cutoffs.

Protocol results:

| Target | Prior checkpoints | Vote rule | Strategy | SPY active | Active alpha | Max DD |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 2023 | train<2021, train<2022 | 2 of 2 | 0.00% | 0.00% | 0.00% | 0.00% |
| 2024 | train<2021, train<2022, train<2023 | 2 of 3 | +27.12% | +19.46% | +7.65% | -4.09% |
| 2025 | train<2021, train<2022, train<2023, train<2024 | 3 of 4 | +5.24% | +4.97% | +0.27% | -3.81% |

This is the best safety shape so far:

- bad 2023 is blocked,
- 2024 remains positive and beats active SPY,
- 2025 barely beats active SPY but by too small a margin.

Current deployment decision: still not tradable. A production gate should require
meaningful active-alpha margin, e.g. at least several percentage points across
multiple locked folds, not +0.27% in one partial-success fold.

## Partial 2026 Protocol Check

Built a latest-through-2026 dataset:

- `checkpoints/daily_ranker/exp11_latest_dataset_h5_2026/daily_ranker_dataset.parquet`
- start date: 2016-01-01
- end date: 2026-05-10
- rows: 924,894
- 2026 test rows: 44,286

Applied the prior-checkpoint consensus protocol to 2026 YTD:

| Target | Prior checkpoints | Vote rule | Strategy | SPY active | Active alpha | Max DD |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 2026 YTD | train<2021, train<2022, train<2023, train<2024 | 3 of 4 | +0.86% | +4.46% | -3.60% | -3.88% |
| 2026 YTD | train<2021, train<2022, train<2023, train<2024 | 4 of 4 | 0.00% | 0.00% | 0.00% | 0.00% |

Interpretation:

The 3-vote protocol is still not reliable in the current 2026 regime. The
4-of-4 requirement no-trades, which is safer. Current live stance remains:
no-trade unless a consensus protocol passes multi-year locked folds with a
meaningful alpha margin.

Interpretation: the 2025 replication confirms the 2026 result. The crash-aware
setup can find attractive-looking slices inside training/validation, but those
slices do not survive nested calibration. This architecture is still research
only and should remain in no-trade mode.

## Regime-Gated Consensus Sweep

Added consensus-level market-regime filters to `evaluate_daily_ranker_consensus.py`
and a focused protocol sweep in `sweep_daily_ranker_consensus.py`.

The important change is that the ensemble no longer needs to decide only from
single-stock model scores. A trade can now require the whole market to be in a
constructive breadth regime before any stock selection is accepted.

Best protocol found:

- prior-checkpoint consensus, with one model allowed to disagree,
- minimum market percentage positive over 20 days: `0.55`,
- minimum market percentage above 20 day moving average: `0.55`,
- no extra predicted-profit or predicted-crash threshold at the consensus layer,
- 5 trading day holding horizon,
- up to 3 selected symbols per rebalance,
- round-trip cost retained at `0.15%`.

Sweep output:

- `checkpoints/daily_ranker/consensus_protocol_sweep_regime.json`
- candidate protocols: `432`
- passing protocols: `16`

Locked-fold result for the best passing protocol:

| Fold | Trades | Strategy | SPY active | Active alpha | Max DD | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2023 | 12 | +7.24% | +5.47% | +1.77% | -4.16% | breadth gate avoided the previous 2023 crash |
| 2024 | 15 | +4.60% | +1.92% | +2.68% | -2.46% | lower return than earlier 2024 winner, but still positive alpha |
| 2025 | 10 | +1.49% | -0.06% | +1.55% | -4.42% | small but positive alpha |
| 2026 YTD | 5 | +3.90% | +1.18% | +2.72% | -0.98% | very small sample |

Aggregate:

- total trades: `42`,
- summed strategy return across folds: `+17.22%`,
- summed active alpha across folds: `+8.72%`,
- worst fold drawdown: `-4.42%`.

Interpretation:

This is the first protocol that passed the locked-fold screen including 2026
YTD. The useful lesson is that stock selection must be conditional on market
breadth. The model was weak when forced to trade in bad breadth regimes; the
breadth gate turned several bad periods into no-trade periods.

Deployment decision:

Still not production tradable. The sample is only 42 trades across four folds,
and 2026 YTD has only 5 trades. Treat this as the current best research
candidate, not as a live allocator. The next validation must add more historical
years and require this same protocol shape to survive without tuning the gate on
the test folds.

Added `recommend_daily_ranker_consensus.py` as a practical research helper. It
loads the best passing regime-gated protocol and applies it to the latest
available dataset date.

Latest run:

- command: `.venv/bin/python recommend_daily_ranker_consensus.py --device mps`
- latest labeled dataset date: `2026-05-07`
- decision: `no_trade`
- output: `checkpoints/daily_ranker/latest_consensus_recommendation.json`

Important limitation: the current dataset builder drops rows without future
labels, so the helper uses the latest labeled row, not a truly live unlabeled
market row. A live version needs a feature-only inference dataset path.

### Live-Feature Consensus Recommendation

Updated `recommend_daily_ranker_consensus.py` so the default path builds
feature-only daily rows directly from cached bars. This removes the future-label
dependency for live inference. Dataset mode is still available with
`--use-dataset`.

The helper now writes and reuses a feature cache:

- feature cache: `checkpoints/daily_ranker/latest_live_features.parquet`
- live recommendation: `checkpoints/daily_ranker/latest_consensus_recommendation_live.json`
- cached rerun time: about `1s` after the feature cache exists

Latest live-cache run:

- command: `.venv/bin/python recommend_daily_ranker_consensus.py --device mps --refresh-live-features --output checkpoints/daily_ranker/latest_consensus_recommendation_live.json`
- decision date: `2026-05-14`
- feature rows: `503`
- decision: `no_trade`

Diagnostics:

| Check | Value |
| --- | ---: |
| train<2021 selected rows | 0 |
| train<2022 selected rows | 0 |
| train<2023 selected rows | 0 |
| train<2024 selected rows | 0 |
| consensus rows before regime filters | 0 |
| consensus rows after regime filters | 0 |
| SPY 20 day return | +6.39% |
| market pct positive 20 day | 43.54% |
| market pct above MA20 | 41.75% |
| market 20 day mean return | -0.14% |
| market 20 day dispersion | 15.03% |

Interpretation:

The current no-trade is not just the breadth gate. All four prior checkpoints
selected zero candidates on the latest live-cache date. The breadth regime also
fails the winning protocol's 55%/55% requirements. This is a correct defensive
decision under the current protocol.

### Strict Consensus Correction

Stress testing exposed a protocol bug in the first regime-gated sweep: for the
2023 fold there were only two prior checkpoints, and `min_vote_gap=1` reduced
the vote requirement to `1 of 2`. That is not real consensus.

Fix:

- `sweep_daily_ranker_consensus.py` now floors every fold at `min_votes >= 2`.
- `recommend_daily_ranker_consensus.py` and `stress_daily_ranker_consensus.py`
  use the same vote floor.
- the strict sweep output is
  `checkpoints/daily_ranker/consensus_protocol_sweep_regime_min2.json`.

Strict sweep result:

- passing protocols: `0`
- best unpassed protocol total trades: `76`
- best unpassed summed return: `+33.22%`
- best unpassed summed active alpha: `+4.32%`
- worst fold drawdown: `-4.09%`

Best unpassed fold details:

| Fold | Trades | Strategy | SPY active | Active alpha | Max DD | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2023 | 0 | 0.00% | 0.00% | 0.00% | 0.00% | no-trade |
| 2024 | 39 | +27.12% | +19.46% | +7.65% | -4.09% | good |
| 2025 | 29 | +5.24% | +4.97% | +0.27% | -3.81% | too thin |
| 2026 YTD | 8 | +0.86% | +4.46% | -3.60% | -3.88% | failed |

Updated deployment decision:

The earlier regime-gated passing result is invalidated. With real minimum
2-vote consensus, no protocol passes the locked-fold requirement. The strict
live recommender therefore defaults to no-trade when the strict sweep has no
passing protocol.

Latest strict live run:

- command: `.venv/bin/python recommend_daily_ranker_consensus.py --device mps --output checkpoints/daily_ranker/latest_consensus_recommendation_live_strict.json`
- decision date: `2026-05-14`
- decision: `no_trade`
- reason: no passing rule found in the strict min-2 consensus sweep.

### Fresh Train-Through-2025 Checkpoint

Trained a proper causal 2026 checkpoint instead of relying on the earlier
1-epoch placeholder:

- output: `checkpoints/daily_ranker/exp12_train2025_alpha_h5_2026/daily_ranker.pt`
- train end: `2025-01-01`
- test window: `2026-01-01` to `2026-05-10`
- epochs: `12`
- hidden dim: `160`
- target: alpha utility
- best validation loss: `0.7155` at epoch 3

Result:

- calibrated rule: no-trade
- 2026 YTD test result: no trades

Then recalibrated the same weights with a looser validation gate:

- script: `recalibrate_daily_ranker_rule.py`
- output checkpoint:
  `checkpoints/daily_ranker/exp12_train2025_alpha_h5_2026/daily_ranker_recalibrated_loose.pt`
- validation rule found a mild positive slice:
  - validation trades: `40`
  - validation return: `+2.92%`
  - validation active alpha: `+1.44%`
  - validation max drawdown: `-6.05%`
- 2026 YTD test result: still no trades

Interpretation:

The freshest causal checkpoint does not find actionable 2026 trades under its
own rule, even when recalibrated loosely. This supports the current defensive
no-trade stance.

### Strict Min-3 Fold Requirement

The previous min-2 sweep could pass by trading only 2024 and 2025 while
no-trading 2023 and 2026. That is too weak for a model intended to beat SPY
reliably.

Updated sweep default:

- minimum traded folds: `3`
- 2026 fold includes the fresh train-through-2025 recalibrated checkpoint
- output:
  `checkpoints/daily_ranker/consensus_protocol_sweep_regime_min3_with2025.json`

Strict min-3 result:

- passing protocols: `0`
- best unpassed protocol trades only `2` folds
- best unpassed total trades: `68`
- best unpassed summed active alpha: `+7.92%`
- 2026 YTD: no-trade

Latest strict min-3 live run:

- output:
  `checkpoints/daily_ranker/latest_consensus_recommendation_live_strict_min3.json`
- decision date: `2026-05-14`
- decision: `no_trade`
- reason: no passing rule found in the strict min-3 sweep.

Updated conclusion:

Still not tradable. The system has become safer and more honest, but the
research edge is concentrated in 2024/2025 and does not yet produce a robust
multi-fold protocol that can trade through 2026.

## Market Regime Overlay Attempt

Added `evaluate_market_regime_overlay.py` to test a SPY/cash fallback. The goal
was to decide whether no-stock-trade periods should still hold SPY or cash.

Walk-forward result:

| Fold | Strategy | Buy-hold SPY | Active alpha | SPY exposure | Max DD |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2023 | +13.56% | +25.42% | -11.85% | 46.15% | -3.31% |
| 2024 | -1.96% | +25.10% | -27.06% | 56.60% | -10.81% |
| 2025 | +7.17% | +23.87% | -16.70% | 24.53% | -1.30% |
| 2026 YTD | +8.54% | +8.95% | -0.41% | 100.00% | -8.70% |

Interpretation:

The SPY/cash overlay reduced exposure and drawdown in some folds, but it failed
the real objective: it underperformed buy-and-hold SPY in every locked fold.
Do not use this overlay as a live fallback.

## Listwise Cross-Sectional Ranker

Added `train_daily_listwise_ranker.py`.

Difference from the row-wise daily ranker:

- trains one decision date at a time,
- uses a listwise softmax objective across the stock cross-section,
- learns which symbols should rank highest on a given date, instead of treating
  every symbol-date row independently.

First loose 2026 run was promising but had an unsafe calibration issue: the
selected rule had a strong holdout validation slice but an ugly search slice.
Tightened calibration to require the search slice not be catastrophic:

- min search return: `-5%`
- min search active alpha: `-5%`
- min profit rate: `45%`
- min beat-SPY rate: `45%`
- max validation drawdown: `20%`

Locked-fold results for the tighter listwise model:

| Fold | Train end | Trades | Strategy | SPY active | Active alpha | Max DD | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2023 | 2022-01-01 | 60 | -8.58% | +1.11% | -9.69% | -22.10% | failed |
| 2024 | 2023-01-01 | 0 | 0.00% | 0.00% | 0.00% | 0.00% | no-trade |
| 2025 | 2024-01-01 | 0 | 0.00% | 0.00% | 0.00% | 0.00% | no-trade |
| 2026 YTD | 2025-01-01 | 7 | +14.68% | +4.71% | +9.97% | -1.74% | promising but tiny |

Interpretation:

This is the first model family that found positive 2026 stock-picking alpha
after stricter checks. However, it failed badly in 2023 and no-traded 2024/2025,
so it is not a deployable all-year allocator. The useful next direction is to
treat listwise ranking as a specialist candidate generator and require a
separate regime/safety gate before using it live.

### Live Listwise Specialist Check

Added `recommend_daily_listwise_ranker.py`.

Purpose:

- load the best 2026 listwise specialist checkpoint,
- build or reuse live feature rows,
- apply the checkpoint's saved rule,
- print filter-by-filter diagnostics.

Latest run:

- command: `.venv/bin/python recommend_daily_listwise_ranker.py --device mps`
- decision date: `2026-05-14`
- feature rows: `503`
- decision: `no_trade`
- output: `checkpoints/daily_listwise_ranker/latest_listwise_recommendation_live.json`

Rule diagnostics:

| Step | Remaining rows |
| --- | ---: |
| start | 503 |
| score threshold | 67 |
| min profit | 17 |
| max crash | 17 |
| SPY 20 day regime | 17 |
| drawdown safety gate | 0 |
| max volatility rank | 0 |
| final | 0 |

Top unfiltered names were `DDOG`, `MU`, and `AMD`, but they did not pass the
saved rule. The final blocker was the drawdown safety gate, which requires
stocks to be within 5% of their 60 day high.

Interpretation:

Even the 2026-specialist listwise model currently says no-trade. This is a
healthy refusal: the raw scorer sees momentum names, but the rule gate blocks
them because they are too extended/drawn down relative to the calibrated safety
condition. This remains research-only because the listwise model failed 2023.

## Strict Calibration And Risk-Adjusted Listwise Iteration

The first listwise model still looked too easy to overfit, so the next
iteration made calibration stricter and then changed the training target.

Strict calibration settings:

- min validation return: `0%`
- min validation active alpha: `0%`
- min profit rate: `52%`
- min beat-SPY rate: `50%`
- max validation drawdown: `10%`
- rule holdout fraction: `25%`

Strict calibration results:

| Model | Fold | Trades | Strategy | SPY active | Active alpha | Max DD | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| utility target | 2023 | 51 | -19.13% | -0.39% | -18.74% | -30.13% | failed |
| utility target | 2026 YTD | 0 | 0.00% | 0.00% | 0.00% | 0.00% | no-trade |
| alpha target | 2023 | 63 | -18.62% | +1.90% | -20.52% | -42.73% | failed |
| alpha target | 2026 YTD | 0 | 0.00% | 0.00% | 0.00% | 0.00% | no-trade |

Interpretation:

Simply tightening the rule or ranking future alpha directly was not enough.
The validation slices could still look excellent while the locked 2023 test
collapsed. The failure mode was high-beta/high-crash selection.

Added a risk-adjusted listwise mode to `train_daily_listwise_ranker.py`:

- new target: `future_alpha - downside_penalty * max(-future_min_return, 0)`
- default downside penalty used here: `2.0`
- increased crash auxiliary loss from `0.30` to `0.80`
- increased scoring crash penalty from `0.10` to `0.35`
- stored all scoring weights in the checkpoint config
- updated `recommend_daily_listwise_ranker.py` to read those weights from the
  checkpoint for live-style recommendations

Locked-fold risk-adjusted results:

| Fold | Train end | Trades | Strategy | SPY active | Active alpha | Max DD | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2023 | 2022-01-01 | 65 | +39.02% | +8.89% | +30.13% | -34.89% | return passed, DD failed |
| 2024 | 2023-01-01 | 0 | 0.00% | 0.00% | 0.00% | 0.00% | no-trade |
| 2025 | 2024-01-01 | 0 | 0.00% | 0.00% | 0.00% | 0.00% | no-trade |
| 2026 YTD | 2025-01-01 | 17 | +2.94% | +1.51% | +1.43% | -4.99% | small positive |

This is a meaningful improvement over the previous listwise variants because
it no longer loses money in the 2023/2026 checks. However, it is still not
tradable:

- 2023 drawdown is too large for a live allocator.
- 2024 and 2025 no-trade under strict calibration, so the strategy is not a
  steady SPY-beater.
- 2026 YTD alpha is positive but small and based on only `17` trades.

Latest live-style risk-adjusted check:

- command:
  `.venv/bin/python recommend_daily_listwise_ranker.py --device mps --output checkpoints/daily_listwise_ranker/latest_riskadj_listwise_recommendation_live.json`
- checkpoint:
  `checkpoints/daily_listwise_ranker/exp5_riskadj_train2025_2026_strictcal/daily_listwise_ranker.pt`
- decision date: `2026-05-14`
- decision: `no_trade`

Live diagnostics:

| Step | Remaining rows |
| --- | ---: |
| start | 503 |
| score threshold | 43 |
| min profit | 5 |
| max crash | 5 |
| SPY 20 day regime | 5 |
| drawdown safety gate | 3 |
| max volatility rank | 0 |
| market breadth gate | 0 |
| final | 0 |

Top unfiltered names were `SNDK`, `INTC`, and `MU`, but predicted crash was too
high and market breadth was weak (`43.54%` of symbols positive over 20 days,
`41.75%` above MA20). The model correctly refused to recommend buys in this
state.

Current best interpretation:

The risk-adjusted listwise model is the best research branch so far, but it is
not the final trading model. The next useful work should focus on controlling
realized portfolio drawdown, not just selecting higher expected-return symbols.
Candidate next steps:

- add a walk-forward portfolio-level drawdown stop and cooldown, calibrated only
  on training/validation windows;
- add a second model that predicts whether the next week is safe for concentrated
  single-stock exposure;
- require a multi-checkpoint ensemble agreement for the risk-adjusted model,
  similar to the earlier daily-ranker consensus protocol;
- evaluate a fallback allocation benchmark explicitly: stock specialist,
  SPY, or cash, with the decision trained only on past folds.

## Stop-Loss And Drawdown Overlay Probe

Added `evaluate_listwise_drawdown_stop.py`.

Purpose:

- reload the saved risk-adjusted listwise checkpoints,
- score each locked fold,
- apply each checkpoint's saved rule,
- test simple stop-loss and drawdown-cooldown overlays without retraining.

Important caveat:

The stop-loss simulation uses `future_min_return` as a proxy for whether a stop
would have been hit during the holding window. This is useful for research, but
it is optimistic unless validated on real intraday bars with realistic fills,
spreads, gaps, and slippage. Treat it as a lead, not as tradable evidence.

Best overlay in the probe:

- stop loss: `3%`
- portfolio drawdown cooldown: none
- output:
  `checkpoints/daily_listwise_ranker/riskadj_drawdown_stop_eval.json`

Results:

| Fold | Periods | Trades | Strategy | SPY active | Active alpha | Max DD | Stopped positions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2023 | 25 | 65 | +98.06% | +8.89% | +89.18% | -11.08% | 26 |
| 2024 | 0 | 0 | 0.00% | 0.00% | 0.00% | 0.00% | 0 |
| 2025 | 0 | 0 | 0.00% | 0.00% | 0.00% | 0.00% | 0 |
| 2026 YTD | 7 | 17 | +2.14% | +1.51% | +0.64% | -3.15% | 5 |

Baseline without stop loss:

| Fold | Periods | Trades | Strategy | SPY active | Active alpha | Max DD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2023 | 25 | 65 | +39.02% | +8.89% | +30.13% | -34.89% |
| 2026 YTD | 7 | 17 | +2.94% | +1.51% | +1.43% | -4.99% |

Interpretation:

The ranking signal may be finding real upside in 2023/2026, but it requires a
loss-control mechanism. The stop-loss probe is encouraging because it shows the
large 2023 drawdown is concentrated in positions that had meaningful adverse
excursions. It is not yet sufficient for live trading because the stop behavior
must be validated using intraday bars and realistic stop execution.

Next best step:

Download/use intraday bars for the selected historical trades and rerun the
same stop-loss logic with realistic stop fills. If the 3% stop still improves
2023 without destroying 2026, then promote stop-loss behavior into the official
walk-forward protocol. If it fails with intraday realism, the model remains
research-only.

## Intraday Stop-Loss Validation

Found local 1-minute Alpaca cache with `513` symbols in:

`/Users/viktorzeman/.cache/trading-autoresearch`

Added `evaluate_listwise_intraday_stops.py`.

This validator is stricter than the daily stop proxy:

- selected trades come from the locked daily listwise checkpoints;
- entry price is the daily signal close;
- stop is checked on cached 1-minute bars after the signal date and through the
  target exit date;
- if the next available minute opens below the stop, it exits at that worse open
  price;
- otherwise, stop exits at the stop level with `5 bps` slippage;
- if no stop is hit, exit uses the minute close at the end of the target window;
- the same `15 bps` roundtrip cost is applied.

Output:

`checkpoints/daily_listwise_ranker/riskadj_intraday_stop_eval.json`

Intraday stop results:

| Stop | Fold | Periods | Trades | Strategy | SPY active | Active alpha | Max DD | Stopped positions |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3% | 2023 | 25 | 65 | +16.43% | +8.89% | +7.54% | -13.71% | 36 |
| 3% | 2026 YTD | 7 | 17 | +1.26% | +1.51% | -0.24% | -3.43% | 6 |
| 5% | 2023 | 25 | 65 | +43.72% | +8.89% | +34.83% | -21.46% | 28 |
| 5% | 2026 YTD | 7 | 17 | +2.05% | +1.51% | +0.55% | -4.66% | 3 |
| 8% | 2023 | 25 | 65 | +59.00% | +8.89% | +50.12% | -24.99% | 9 |
| 8% | 2026 YTD | 7 | 17 | +2.25% | +1.51% | +0.75% | -5.63% | 1 |

2024 and 2025 still no-traded because their locked checkpoints selected no
candidates.

Interpretation:

The intraday validation removes the unrealistic `+98%` daily-proxy result, but
the signal does not disappear. A `5%` or `8%` stop keeps positive alpha in both
traded folds. The tradeoff is still not live-ready:

- `3%` controls drawdown better but underperforms active SPY in 2026 YTD.
- `5%` is the best balance so far, but 2023 max drawdown is still `-21.46%`.
- `8%` gives more return but too much drawdown.
- Only two folds trade; 2024 and 2025 still no-trade.

Updated conclusion:

The best current branch is risk-adjusted listwise ranking plus realistic
intraday stop validation. It has a research edge, but it is not yet a
production allocator. To become tradable, it needs either lower drawdown at the
same return level or a separate regime model that decides when this specialist
is allowed to deploy capital.

## SPY-Relative Speed And Direction Features

Added a causal SPY-relative feature family to `train_daily_ranker.py`.

Motivation:

Raw relative return is useful, but it does not fully tell the model whether a
stock is:

- rising with SPY,
- rising faster than SPY,
- lagging SPY while moving in the same direction,
- diverging from SPY,
- moving against SPY.

New feature groups:

- `rel_spy_1d`, plus existing `rel_spy_5d`, `rel_spy_20d`, `rel_spy_60d`
- `rel_spy_speed_{1,5,20,60}d`
  - normalized speed difference:
    `(stock_return - spy_return) / (abs(spy_return) + 0.01)`, clipped to
    `[-5, 5]`
- `spy_same_dir_{1,5,20,60}d`
- `spy_opposite_dir_{1,5,20,60}d`
- `spy_lagging_same_dir_{5,20,60}d`
- `spy_leading_same_dir_{5,20,60}d`

Compatibility change:

Updated `recommend_daily_listwise_ranker.py` so live scoring uses the exact
`feature_cols` saved in each checkpoint. This prevents older checkpoints from
breaking when `FEATURE_COLS` grows.

Updated `recommend_daily_ranker_consensus.py` so stale live feature caches are
rebuilt when newly required columns are missing.

Fresh dataset:

`checkpoints/daily_ranker/exp13_relspy_speed_features_h5/daily_ranker_dataset.parquet`

Dataset build command:

```bash
.venv/bin/python train_daily_ranker.py \
  --output-dir checkpoints/daily_ranker/exp13_relspy_speed_features_h5 \
  --start-date 2016-01-01 \
  --end-date 2026-05-10 \
  --train-end 2025-01-01 \
  --test-start 2026-01-01 \
  --test-end 2026-05-10 \
  --horizon-days 5 \
  --cached-all \
  --symbol-limit 0 \
  --epochs 1 \
  --device mps \
  --utility-mode alpha \
  --min-validation-trades 20 \
  --rule-validation-fraction 0.25 \
  --min-rule-validation-trades 5
```

The one-epoch row-wise run was only a cheap dataset materialization sanity
check, but it was directionally interesting:

| Fold | Trades | Strategy | SPY active | Active alpha | Max DD |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2026 YTD | 33 | +20.23% | +7.46% | +12.78% | -0.67% |

Do not treat this as validated yet because it is a one-epoch row-wise run and
needs the same locked fold protocol as the rest of the research.

Risk-adjusted listwise rerun on the new feature dataset:

| Fold | Train end | Trades | Strategy | SPY active | Active alpha | Max DD | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2023 | 2022-01-01 | 53 | +29.02% | +11.68% | +17.34% | -16.10% | improved drawdown |
| 2024 | 2023-01-01 | 0 | 0.00% | 0.00% | 0.00% | 0.00% | no-trade |
| 2025 | 2024-01-01 | 24 | +6.97% | +1.79% | +5.18% | -2.77% | good small fold |
| 2026 YTD | 2025-01-01 | 0 | 0.00% | 0.00% | 0.00% | 0.00% | no-trade |

Comparison to previous risk-adjusted listwise branch:

- 2023 became less explosive but much safer:
  - old: `+39.02%`, max DD `-34.89%`
  - new: `+29.02%`, max DD `-16.10%`
- 2025 improved from no-trade to a clean positive fold:
  - new: `+6.97%`, active alpha `+5.18%`, max DD `-2.77%`
- 2026 became more conservative under strict listwise calibration:
  - old: `+2.94%`, active alpha `+1.43%`
  - new: no-trade

Interpretation:

The user's SPY-relative speed/direction idea helped. It appears to reduce
dangerous 2023 behavior and produces a cleaner 2025 fold. It did not solve the
main deployment problem: the system still does not trade all folds, and 2026
strict listwise calibration refuses to deploy capital.

Next best branch:

Combine the strengths:

- use SPY-relative speed/direction features,
- keep risk-adjusted listwise training,
- add an ensemble/regime gate that can choose between:
  - old risk-adjusted specialist,
  - new SPY-relative specialist,
  - row-wise SPY-relative ranker,
  - cash/no-trade.

The row-wise SPY-relative result is promising enough to deserve a proper
walk-forward validation, but it is not yet evidence of a tradable model.

## 32-Epoch SPY-Relative Risk-Adjusted Listwise Test

Question:

Would simply training the SPY-relative risk-adjusted listwise model longer help?

Experiment:

- same dataset:
  `checkpoints/daily_ranker/exp13_relspy_speed_features_h5/daily_ranker_dataset.parquet`
- same strict calibration settings as the 8-epoch run
- same MPS device
- epochs increased from `8` to `32`
- best validation-loss checkpoint still selected before rule calibration

32-epoch locked-fold results:

| Fold | Train end | Trades | Strategy | SPY active | Active alpha | Max DD | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2023 | 2022-01-01 | 44 | +10.56% | +7.82% | +2.74% | -17.12% | worse than 8 epochs |
| 2024 | 2023-01-01 | 0 | 0.00% | 0.00% | 0.00% | 0.00% | no-trade |
| 2025 | 2024-01-01 | 25 | -14.62% | +1.79% | -16.41% | -16.43% | failed |
| 2026 YTD | 2025-01-01 | 0 | 0.00% | 0.00% | 0.00% | 0.00% | no-trade |

Comparison to 8 epochs:

| Fold | 8-epoch active alpha | 32-epoch active alpha | Change |
| --- | ---: | ---: | --- |
| 2023 | +17.34% | +2.74% | worse |
| 2024 | 0.00% | 0.00% | unchanged no-trade |
| 2025 | +5.18% | -16.41% | much worse |
| 2026 YTD | 0.00% | 0.00% | unchanged no-trade |

Training behavior:

Training loss kept falling, but validation loss generally stopped improving
early and became noisy. For example:

- 2023 best validation loss occurred around epoch `10`, but locked test
  performance was worse than the 8-epoch model.
- 2025 best validation loss occurred around epoch `12`, but locked test
  performance flipped negative.
- 2026 best validation loss matched the earlier epoch `6` checkpoint and still
  produced no valid strict rule.

Conclusion:

More epochs did not solve the model quality problem. This looks like
overfitting/regime instability, not undertraining.

Do not continue by simply increasing epochs. The next useful branches are:

- checkpoint/epoch ensembling instead of selecting one late checkpoint;
- walk-forward validation of the row-wise SPY-relative ranker, because its quick
  one-epoch sanity result was much stronger than the listwise 32-epoch result;
- regime gating between old risk-adjusted specialist, new SPY-relative
  specialist, row-wise SPY-relative ranker, and no-trade/cash.

## Online Daily Patch Model

Added `train_online_daily_patch_model.py` as a separate causal continual
training experiment.

Protocol:

1. Build or load a one-day-horizon daily dataset.
2. Train the model on the first year of usable rows.
3. For each later trading day:
   - score the current cross-section before seeing the next-day result,
   - buy up to three selected symbols for the next close-to-close interval or
     hold cash,
   - record portfolio return and SPY benchmark return,
   - patch-train the same model on that day's now-realized label,
   - continue day by day until the latest cached date.

Model:

- two-layer LayerNorm/GELU/dropout MLP,
- listwise cross-sectional utility head,
- profit-probability auxiliary head,
- crash-probability auxiliary head,
- score blend rewards utility/profit/top-rank and penalizes predicted crash.

Important correction:

The first online runs exposed a benchmark-data bug. Local SPY minute data starts
on `2020-07-27`, while many stocks have local history back to 2016. The old
feature builder backfilled missing SPY prices into earlier years, making
pre-2020 SPY returns look like zero and distorting alpha labels. The online
trainer now requires real benchmark history before using a row:

- `--require-benchmark-history`,
- `--benchmark-symbol SPY`,
- `--benchmark-warmup-days 70`.

Invalidated artifact runs:

| Run | Filter | Result | Status |
| --- | --- | ---: | --- |
| `exp1_h1_relspy` | no split/bad-tick filter | infinite/absurd equity | invalid |
| `exp2_h1_relspy_clean` | +/-35% daily cap | absurd compounding | invalid |
| `exp3_h1_relspy_strict_clean` | +/-12% cap, price >= $5 | absurd compounding | invalid because SPY history was fake before 2020 |

First credible online run:

`checkpoints/online_daily_patch/exp7_h1_real_spy_strict_8ep_fixed_benchmark`

Settings:

- usable dates: `2020-11-03` to `2026-05-13`,
- first-year warmup: `2020-11-03` to `2021-11-03`,
- online decision period: `2021-11-04` to `2026-05-13`,
- rows after benchmark/tradability filters: `695,046`,
- initial epochs: `8`,
- patch epochs per day: `1`,
- max positions: `3`,
- minimum price: `$5`,
- daily move cap: `+/-12%`,
- SPY daily move cap: `+/-8%`,
- roundtrip cost: `0.30%`.

Result from `$50,000` starting equity:

| Metric | Online patched model | SPY buy-and-hold |
| --- | ---: | ---: |
| Final equity | `$21,146` | `$72,801` |
| Total return | `-57.71%` | `+45.60%` |
| Max drawdown | `-61.42%` | `-25.22%` |

Additional diagnostics:

- decision days: `1,132`,
- active buy days: `230`,
- selected trades: `402`,
- trade profit rate: `49.0%`.

Chart:

`docs/online_daily_patch_equity.png`

Benchmark accounting fix:

The first credible run still had a reporting weakness: the SPY benchmark return
for a day was read from the currently top-scored row. Different models could
therefore report slightly different SPY curves if the top row came from a
symbol with missing dates. `train_online_daily_patch_model.py` now builds the
benchmark curve directly from `daily_bars("SPY")` by decision date, and trade
diagnostics use the same date-level SPY return.

Longer-epoch test:

`checkpoints/online_daily_patch/exp6_h1_real_spy_strict_32ep_fixed_benchmark`

Same protocol and filters, but initial warmup training increased from `8` to
`32` epochs.

| Initial epochs | Active days | Trades | Final equity | Total return | Max DD | Trade profit rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 230 | 402 | `$21,146` | `-57.71%` | `-61.42%` | `49.0%` |
| 32 | 462 | 1,000 | `$18,121` | `-63.76%` | `-80.59%` | `51.7%` |

Comparison chart:

`docs/online_daily_patch_epoch_compare.png`

Interpretation:

The online patching framework now works and is causal, but the first credible
model is decisively not tradable. More epochs lowered training loss but made
the portfolio worse by increasing trade count, drawdown, and realized losses.
The artifact runs were useful because they found data-quality and
benchmark-alignment failures. After those failures were removed, the model
underperformed SPY badly. Future work should keep this online protocol as a
validation harness, but the model itself needs a better objective and stricter
risk controls before any live recommendation can be trusted.

Next protocol decision:

Do not spend more time increasing daily warmup epochs. Use `8` initial epochs
as the default for online patching unless a validation run proves otherwise.
The better next branch is the same causal patch-and-retrain protocol on
15-minute bars:

1. Aggregate cached Alpaca minute bars into 15-minute candles.
2. Train on the first year of benchmark-aligned 15-minute intervals.
3. At each interval, predict only the next 15-minute move, not the whole next
   day.
4. Simulate buy/sell/hold for the next interval with explicit turnover costs.
5. Patch-train on the realized interval once it becomes known.
6. Continue interval by interval, matching how real trading would adapt as new
   market data arrives.

Why this is preferred:

- the model no longer has to predict overnight and full-day noise from one
  daily close,
- stop/risk control can react inside the trading day,
- patching becomes closer to the real deployment loop,
- the next label is much nearer in time, which should reduce regime drift.

The 15-minute model should still be treated as research-only until it beats SPY
or a cash/SPY fallback across unseen walk-forward periods after realistic
transaction costs.

## 15-Minute Transformer Family Harness

Added `train_15m_transformer_ensemble.py`.

Purpose:

- build one shared 15-minute dataset from cached Alpaca 1-minute bars,
- train several transformer-style model families under the same protocol,
- run the causal online loop:
  - train on the first year,
  - predict the next 15-minute interval,
  - simulate buy/hold/sell,
  - patch-train on the realized interval,
  - repeat chronologically,
- compare individual models and pair/triple ensembles.

Implemented model families:

- `patchtst`: PatchTST-style temporal patch transformer,
- `temporal_fusion`: temporal-fusion-style gated transformer,
- `decision_transformer`: causal goal-token decision transformer,
- `trajectory_transformer`: causal state trajectory transformer,
- `perceiver`: Perceiver-style latent bottleneck transformer,
- `cross_asset`: lightweight symbol-aware/cross-asset attention variant,
- `jepa_patch`: JEPA-style latent patch predictor plus trading heads.

Dataset/features:

- 15-minute candles aggregated from cached 1-minute OHLCV bars,
- regular market hours only,
- sequence length: `32` intervals,
- features include:
  - short intraday returns over 1, 2, 4, 8, 16, and 26 intervals,
  - realized volatility,
  - volume z-score,
  - moving-average distance,
  - drawdown,
  - candle range/body,
  - SPY returns and stock-vs-SPY relative returns,
  - market breadth/dispersion context,
  - time-of-day sine/cosine.

Smoke test:

- dataset: `checkpoints/transformer_15m/smoke_15m_dataset.parquet`,
- symbols: `12`,
- rows: `427,301`,
- result: all seven architectures trained, patched online, saved checkpoints,
  and generated pairwise ensemble outputs. This validated the harness.

First 8-epoch pilot:

- command output:
  `checkpoints/transformer_15m/exp1_all_models_8ep_40sym`,
- dataset:
  `checkpoints/transformer_15m/shared_15m_40sym.parquet`,
- symbols: `40`,
- rows: `1,402,850`,
- train rows in first warmup year: `253,122`,
- online evaluation rows: `1,148,448`,
- evaluated intervals: `1,500`,
- initial epochs: `8`,
- patch epochs per interval: `1`,
- max train samples per model: `40,000`,
- max positions: `3`,
- roundtrip cost: `0.08%`,
- loose gates:
  - predicted profit >= `0.48`,
  - predicted crash <= `0.60`,
  - buy probability >= `0.25`.

Best results by active alpha, including ensembles:

| Rank | Model | Trades | Strategy | SPY | Active alpha | Max DD |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `patchtst+perceiver` | 286 | `-22.93%` | `-6.25%` | `-16.68%` | `-29.47%` |
| 2 | `perceiver` | 256 | `-24.45%` | `-6.25%` | `-18.20%` | `-26.81%` |
| 3 | `trajectory_transformer+cross_asset+jepa_patch` | 448 | `-26.00%` | `-6.25%` | `-19.75%` | `-31.88%` |
| 4 | `trajectory_transformer` | 434 | `-30.22%` | `-6.25%` | `-23.97%` | `-34.50%` |
| 5 | `decision_transformer+trajectory_transformer+jepa_patch` | 474 | `-32.49%` | `-6.25%` | `-26.24%` | `-38.67%` |

Individual model results:

| Model | Trades | Strategy | SPY | Active alpha | Max DD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `perceiver` | 256 | `-24.45%` | `-6.25%` | `-18.20%` | `-26.81%` |
| `trajectory_transformer` | 434 | `-30.22%` | `-6.25%` | `-23.97%` | `-34.50%` |
| `patchtst` | 407 | `-32.73%` | `-6.25%` | `-26.48%` | `-35.15%` |
| `jepa_patch` | 557 | `-41.70%` | `-6.25%` | `-35.45%` | `-44.21%` |
| `cross_asset` | 666 | `-43.15%` | `-6.25%` | `-36.90%` | `-46.89%` |
| `decision_transformer` | 480 | `-49.33%` | `-6.25%` | `-43.08%` | `-51.13%` |
| `temporal_fusion` | 859 | `-53.11%` | `-6.25%` | `-46.86%` | `-56.13%` |

Artifacts:

- leaderboard:
  `checkpoints/transformer_15m/exp1_all_models_8ep_40sym/leaderboard.csv`,
- summary:
  `checkpoints/transformer_15m/exp1_all_models_8ep_40sym/summary.json`,
- curves:
  `checkpoints/transformer_15m/exp1_all_models_8ep_40sym/equity_curves.csv`,
- trades:
  `checkpoints/transformer_15m/exp1_all_models_8ep_40sym/trades.csv`,
- chart:
  `docs/transformer_15m_equity_compare.png`.

Charting update:

`train_15m_transformer_ensemble.py` now always writes readable performance
charts for every run:

- all individual transformers and all ensemble combinations:
  `docs/transformer_15m_equity_all_models_and_combinations.png`,
- individual transformers only:
  `docs/transformer_15m_equity_individuals.png`,
- all tested combinations only:
  `docs/transformer_15m_equity_combinations.png`,
- leaderboard bar chart for every model and combination:
  `docs/transformer_15m_performance_leaderboard.png`.

Interpretation:

The transformer harness is now working, but the first all-model 15-minute
pilot is not tradable. Every model and every tested pair/triple ensemble
underperformed SPY. The loose gates produced too many false positives, and
transaction costs plus noisy 15-minute labels overwhelmed the learned signal.

Useful lesson:

Architecture variety alone is not enough. The next branch should keep this
harness but add calibration before live-style selection:

1. Train all models as above.
2. Use a validation slice after warmup to calibrate per-model thresholds.
3. Require a model or ensemble to beat SPY on validation before it is allowed
   to trade in the online test slice.
4. Add a no-trade default when no calibrated transformer passes.
5. Consider holding SPY or cash as an explicit action instead of forcing stock
   picks from weak intraday signals.

## 2026 Holdout 15-Minute Transformer Test

Question:

Would more training and a clean 2026 holdout improve the transformer family?

Change:

`train_15m_transformer_ensemble.py` now supports explicit chronological split
controls:

- `--train-start`,
- `--train-end`,
- `--eval-start`,
- `--eval-end`.

This lets us train on 2021-2025 and evaluate only on 2026. The 2026 rows are
not used in initial training. During the 2026 online walk, the model still
patch-trains after each interval is predicted and realized, which matches the
intended real-time deployment loop.

Experiment:

- output:
  `checkpoints/transformer_15m/exp2_train2021_2025_eval2026_16ep_40sym`,
- dataset:
  `checkpoints/transformer_15m/shared_15m_40sym.parquet`,
- symbols: `40`,
- dataset rows: `1,402,850`,
- initial training window: `2021-01-01` to `2026-01-01`,
- 2026 evaluation window: `2026-01-01` to `2026-05-16`,
- eligible training rows: `1,268,941`,
- sampled training rows per model: `250,000`,
- 2026 evaluation rows: `94,248`,
- 2026 decision intervals: `2,390`,
- initial epochs: `16`,
- patch epochs per interval: `1`,
- max positions: `3`,
- roundtrip cost: `0.08%`,
- same loose gates as the first 15-minute pilot.

Best results by active alpha:

| Rank | Model | Trades | Strategy | SPY | Active alpha | Max DD | Profit rate |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `temporal_fusion+perceiver` | 449 | `+15.75%` | `+9.58%` | `+6.16%` | `-14.98%` | `54.57%` |
| 2 | `trajectory_transformer+perceiver+jepa_patch` | 438 | `+1.97%` | `+9.58%` | `-7.61%` | `-18.52%` | `53.88%` |
| 3 | `temporal_fusion+perceiver+jepa_patch` | 443 | `+1.01%` | `+9.58%` | `-8.58%` | `-18.82%` | `53.95%` |
| 4 | `trajectory_transformer+cross_asset+jepa_patch` | 512 | `+0.87%` | `+9.58%` | `-8.71%` | `-19.15%` | `52.73%` |
| 5 | `temporal_fusion+decision_transformer+perceiver` | 466 | `-0.42%` | `+9.58%` | `-10.01%` | `-19.62%` | `53.22%` |

Individual model results:

| Model | Trades | Strategy | SPY | Active alpha | Max DD | Profit rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `perceiver` | 468 | `-9.88%` | `+9.58%` | `-19.47%` | `-18.28%` | `53.21%` |
| `trajectory_transformer` | 597 | `-24.89%` | `+9.58%` | `-34.48%` | `-26.10%` | `51.59%` |
| `temporal_fusion` | 807 | `-30.79%` | `+9.58%` | `-40.38%` | `-35.74%` | `50.93%` |
| `cross_asset` | 669 | `-32.46%` | `+9.58%` | `-42.04%` | `-41.85%` | `50.67%` |
| `jepa_patch` | 667 | `-33.73%` | `+9.58%` | `-43.31%` | `-43.66%` | `49.33%` |
| `decision_transformer` | 822 | `-36.72%` | `+9.58%` | `-46.31%` | `-36.99%` | `51.95%` |
| `patchtst` | 736 | `-40.25%` | `+9.58%` | `-49.83%` | `-42.26%` | `50.82%` |

Artifacts:

- leaderboard:
  `checkpoints/transformer_15m/exp2_train2021_2025_eval2026_16ep_40sym/leaderboard.csv`,
- summary:
  `checkpoints/transformer_15m/exp2_train2021_2025_eval2026_16ep_40sym/summary.json`,
- all curves:
  `checkpoints/transformer_15m/exp2_train2021_2025_eval2026_16ep_40sym/equity_curves.csv`,
- trades:
  `checkpoints/transformer_15m/exp2_train2021_2025_eval2026_16ep_40sym/trades.csv`,
- charts:
  - `docs/transformer_15m_equity_all_models_and_combinations.png`,
  - `docs/transformer_15m_equity_individuals.png`,
  - `docs/transformer_15m_equity_combinations.png`,
  - `docs/transformer_15m_performance_leaderboard.png`.

Interpretation:

This is the best 15-minute transformer result so far. Training on the longer
2021-2025 window and holding out 2026 produced one ensemble that beat SPY:
`temporal_fusion+perceiver`.

However, only `1` of `63` tested individual/ensemble variants had positive
active alpha. This is promising but not robust enough for live trading. The
next step should not be simply more epochs. It should validate whether
`temporal_fusion+perceiver` survives:

1. stricter threshold calibration on a pre-2026 validation slice,
2. other held-out periods, not only 2026 YTD,
3. more symbols,
4. realistic spread/slippage stress,
5. a no-trade or SPY/cash fallback when the ensemble confidence is weak.

## Pre-2026 Calibration Test For Temporal Fusion + Perceiver

Question:

Was the 2026 `temporal_fusion+perceiver` win reusable, or did it depend on
choosing thresholds after seeing 2026?

Added `calibrate_15m_transformer_ensemble.py`.

Protocol:

1. Train `temporal_fusion` and `perceiver` on data before 2025.
2. Walk through 2025 causally:
   - predict each 15-minute interval,
   - then patch-train after the interval is realized.
3. Sweep ensemble thresholds on 2025 only.
4. Continue the same patched models into 2026.
5. Test the frozen 2025-selected rule on 2026.

Experiment:

- output:
  `checkpoints/transformer_15m/exp3_calibrated_tf_perceiver_2025_2026`,
- training window: `2021-01-01` to `2025-01-01`,
- calibration window: `2025-01-01` to `2026-01-01`,
- test window: `2026-01-01` to `2026-05-16`,
- eligible training rows: `1,018,440`,
- sampled training rows per model: `250,000`,
- calibration intervals: `6,440`,
- test intervals: `2,390`,
- initial epochs: `16`,
- patch epochs per interval: `1`.

Selected 2025 rule:

- `min_pred_profit`: `0.56`,
- `max_pred_crash`: `0.60`,
- `min_buy_prob`: `0.20`,
- passed the calibration hard filters.

Calibration result on 2025:

| Metric | Ensemble | SPY |
| --- | ---: | ---: |
| Trades | `402` | |
| Return | `+51.53%` | `+14.48%` |
| Active alpha | `+37.05%` | |
| Max drawdown | `-10.86%` | |
| Trade profit rate | `53.48%` | |

Untouched 2026 test result with the frozen 2025 rule:

| Metric | Ensemble | SPY |
| --- | ---: | ---: |
| Trades | `148` | |
| Return | `-16.44%` | `+9.58%` |
| Active alpha | `-26.03%` | |
| Max drawdown | `-18.37%` | |
| Trade profit rate | `41.89%` | |

Chart:

`docs/transformer_15m_calibrated_2025_2026_equity.png`

Interpretation:

This invalidates the loose 2026 win as deployment evidence. The ensemble can
find a very strong 2025 threshold slice, but that rule did not generalize to
2026. The model family is learning some patterns, but the thresholded trade
policy is regime-fragile.

Updated conclusion:

Do not deploy `temporal_fusion+perceiver` yet. The next work should focus on
robustness, not more epochs:

1. rolling yearly calibration and test folds,
2. a regime classifier that can choose no-trade/cash/SPY,
3. stricter calibration that requires the rule to pass multiple sub-slices
   inside 2025,
4. cost/slippage stress,
5. more symbols only after the calibration protocol survives.

## Planned 15-Minute Pattern And Trend Features

The current 15-minute transformer dataset already includes short returns,
volatility, drawdown, volume z-score, SPY-relative returns, breadth, and
time-of-day features. The next dataset version should add explicit pattern and
trend descriptors so the transformers do not need to rediscover every local
shape from raw returns.

Candidate causal features:

- trend slope over 8, 16, 26, and 52 intervals,
- trend consistency: percentage of positive candles over each window,
- acceleration: short-window slope minus long-window slope,
- pullback depth from recent 8/16/26 interval highs,
- breakout distance above recent 16/26/52 interval highs,
- support/resistance distance to rolling lows/highs,
- compression/expansion: current range versus rolling average range,
- volume-confirmed move: return multiplied by volume z-score,
- wick/body ratios for rejection and continuation candles,
- consecutive up/down candle counts,
- SPY-relative trend slope and acceleration,
- market-breadth trend slope and breadth acceleration.

Training goal:

Use these features as additional causal inputs, not as labels. Labels should
remain next-interval return/alpha/crash/action. This keeps the model honest:
patterns are only allowed to summarize the past and present, never the future.

Validation requirement:

Pattern features should only be accepted if they improve a pre-2026 calibration
protocol and then improve the untouched 2026 test. If they only improve the
calibration year, treat them as overfit.

## Train-All-Available Pre-2026 Transformer Test

Question:

Would using every available 15-minute training row through the end of 2025 fix
the 2026 performance?

Experiment:

- output:
  `checkpoints/transformer_15m/exp4_train_allrows_to2025_eval2026_tf_perceiver`,
- dataset:
  `checkpoints/transformer_15m/shared_15m_40sym.parquet`,
- models:
  - `temporal_fusion`,
  - `perceiver`,
  - `temporal_fusion+perceiver`,
- train window: `2020-11-03` to `2026-01-01`,
- test window: `2026-01-01` to `2026-05-16`,
- training rows: `1,307,322`,
- training sample cap: none,
- 2026 evaluation rows: `94,248`,
- 2026 decision intervals: `2,390`,
- initial epochs: `16`,
- patch epochs per interval: `1`,
- batch size: `1024`,
- same loose gates as the prior 2026 holdout.

Results:

| Model | Trades | Strategy | SPY | Active alpha | Max DD | Profit rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `perceiver` | 482 | `-12.54%` | `+9.58%` | `-22.12%` | `-22.46%` | `53.73%` |
| `temporal_fusion` | 903 | `-23.52%` | `+9.58%` | `-33.10%` | `-34.09%` | `50.50%` |
| `temporal_fusion+perceiver` | 431 | `-31.46%` | `+9.58%` | `-41.04%` | `-33.66%` | `49.88%` |

Interpretation:

Training on all available pre-2026 rows did not improve the strategy. It made
the Perceiver less bad than many other variants, but still far behind SPY. The
ensemble that previously won in the sampled 2021-2025 run did not survive when
trained on all rows through 2025.

Conclusion:

The failure is not simply lack of training data or too few epochs. The next
model iteration should add better causal pattern/trend features and a stronger
regime/no-trade policy, then validate with the same 2026 holdout.

## Pattern/Trend Feature Perceiver Test

Implemented the planned causal pattern/trend feature family in
`train_15m_transformer_ensemble.py`.

New feature groups:

- trend slopes over 8, 16, 26, and 52 intervals,
- trend consistency over 8, 16, 26, and 52 intervals,
- short-vs-long trend acceleration,
- pullback from rolling highs,
- breakout distance above prior rolling highs,
- support distance above prior rolling lows,
- range compression/expansion,
- volume-confirmed returns,
- upper/lower wick ratios,
- body-to-range ratio,
- consecutive up/down candle counts,
- SPY-relative trend slope and acceleration,
- market breadth slope and breadth acceleration.

Experiment:

- output:
  `checkpoints/transformer_15m/exp5_pattern_features_perceiver_2026`,
- dataset:
  `checkpoints/transformer_15m/shared_15m_40sym_pattern.parquet`,
- rows: `1,353,250`,
- symbols: `40`,
- model: `perceiver`,
- train window: `2020-11-03` to `2026-01-01`,
- test window: `2026-01-01` to `2026-05-16`,
- training rows: `1,260,154`,
- 2026 evaluation rows: `91,816`,
- initial epochs: `16`,
- patch epochs per interval: `1`,
- same loose gates as the previous Perceiver run.

Result:

| Model | Trades | Strategy | SPY | Active alpha | Max DD | Profit rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| old Perceiver features | 482 | `-12.54%` | `+9.58%` | `-22.12%` | `-22.46%` | `53.73%` |
| pattern Perceiver features | 576 | `-35.10%` | `+9.52%` | `-44.63%` | `-37.38%` | `50.52%` |

Interpretation:

The first pattern-feature run made the Perceiver worse. The model traded more,
profit rate fell, drawdown deepened, and active alpha deteriorated. This does
not prove every pattern feature is useless, but it does show that adding many
hand-engineered pattern columns without feature selection or stricter
calibration can increase overfitting/noise.

Next implication:

Do not keep blindly adding features. The next pattern-feature attempt should
use either:

- feature ablation/selection on a pre-2026 validation protocol, or
- a regime/no-trade head that uses these features only to decide whether
  trading is allowed, not to force more stock picks.

## Top-20 Volume/Valuation Universe Restriction

Added a top volume/valuation universe mode to
`train_15m_transformer_ensemble.py`:

- `--universe-mode top_volume_valuation`,
- `--universe-rank-cache`.

Ranking method:

- compute recent 60-day median daily dollar volume from local cached 1-minute
  bars,
- fetch market cap from yfinance when available,
- score symbols with a blend of log dollar volume and log market cap,
- fall back to liquidity when market cap is unavailable.

Top 20 selected in this run:

`NVDA`, `GOOGL`, `AAPL`, `AMZN`, `MSFT`, `TSLA`, `GOOG`, `AVGO`, `META`, `MU`,
`AMD`, `INTC`, `LLY`, `NFLX`, `XOM`, `ORCL`, `V`, `WMT`, `SNDK`, `JPM`.

Ranking cache:

`checkpoints/transformer_15m/top_volume_valuation_universe.csv`

Experiment:

- output:
  `checkpoints/transformer_15m/exp6_top20_volume_valuation_perceiver_2026`,
- dataset:
  `checkpoints/transformer_15m/shared_15m_top20_volume_valuation_pattern.parquet`,
- rows: `685,221`,
- symbols: `20`,
- model: `perceiver`,
- train window: `2020-11-03` to `2026-01-01`,
- test window: `2026-01-01` to `2026-05-16`,
- training rows: `636,812`,
- 2026 evaluation rows: `47,769`,
- initial epochs: `16`,
- patch epochs per interval: `1`,
- same loose gates as the prior Perceiver run.

Result:

| Model | Trades | Strategy | SPY | Active alpha | Max DD | Profit rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `top20_volume_valuation_perceiver` | 2,415 | `-47.39%` | `+9.29%` | `-56.68%` | `-58.72%` | `51.14%` |

Interpretation:

Restricting to the top 20 liquid/high-valuation names did not help by itself.
It made the model trade far too often: `1,304` active intervals out of `2,390`.
This confirms the key issue is not just universe quality. The strategy needs a
strong no-trade/regime gate and much stricter calibration before the stock
selector is allowed to deploy capital.

## Algorithmic Trading Feature Transformer Test

New experiment:

Use the stronger 15-minute transformer harness with only `temporal_fusion` and
`perceiver`, but expand the input data with classic algorithmic-trading
features and strategy votes.

Feature additions in `train_15m_transformer_ensemble.py`:

- RSI 14 and centered RSI signal,
- MACD line/signal/histogram,
- Bollinger z-score, width, and percent-b,
- stochastic %K/%D,
- rolling VWAP distance over 26 and 52 bars,
- ATR 14/26 as percent of close,
- Donchian 20-bar position and breakout/breakdown flags,
- EMA 8/21 and 21/55 crossover state,
- mean-reversion z-scores over 20 and 52 bars,
- aggregated momentum, mean-reversion, breakout, and trend-quality votes.

Experiment:

- output:
  `checkpoints/transformer_15m/exp7_algo_features_tf_perceiver_2026`,
- dataset:
  `checkpoints/transformer_15m/shared_15m_40sym_algo.parquet`,
- rows: `1,353,246`,
- symbols: `40`,
- training rows: `1,222,712`,
- 2026 evaluation rows: `91,816`,
- train window: `2021-01-01` to `2026-01-01`,
- test window: `2026-01-01` to `2026-05-16`,
- models: `temporal_fusion`, `perceiver`, and their two-model ensemble,
- initial epochs: `16`,
- patch epochs per interval: `1`,
- device: `mps`,
- max sampled training rows: `250,000`,
- gates: `min_pred_profit=0.48`, `max_pred_crash=0.60`,
  `min_buy_prob=0.25`.

Result:

| Model | Active intervals | Trades | Strategy | SPY | Active alpha | Max DD | Profit rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `perceiver` | 338 | 530 | `-18.69%` | `+9.52%` | `-28.22%` | `-26.11%` | `50.57%` |
| `temporal_fusion+perceiver` | 311 | 455 | `-27.98%` | `+9.52%` | `-37.51%` | `-27.54%` | `50.55%` |
| `temporal_fusion` | 620 | 844 | `-59.74%` | `+9.52%` | `-69.27%` | `-59.82%` | `49.05%` |

Interpretation:

The algorithmic trading feature expansion did not improve the 2026 holdout.
Perceiver handled the wider feature set much better than Temporal Fusion, but
it still lost money and underperformed SPY badly. The ensemble also lost,
because the weak Temporal Fusion signals contaminated the combined ranking.

This result reinforces the same lesson as the pattern-feature and top-20
experiments: more columns are not enough. The next useful improvement should
focus on feature selection, a frozen validation-calibrated no-trade/regime
gate, and explicit risk-adjusted objectives instead of simply giving the
selector more technical indicators.

## iTransformer and PatchTransformer Algo-Feature Ensemble Test

New model variants added to `train_15m_transformer_ensemble.py`:

- `patchtransformer`: overlapping 8-bar temporal patches with stride 4, a CLS
  token, symbol embedding, transformer encoder, and the shared utility/profit/
  crash/action heads.
- `itransformer`: iTransformer-style inverted tokens where each feature becomes
  a token and its 32-bar history is projected into the hidden dimension before
  feature-token attention.

Both variants use the same expanded algorithmic trading input schema as the
prior algo-feature test. No separate feature set was created; all models read
from `BASE_FEATURES`, which now includes the RSI/MACD/Bollinger/stochastic/
VWAP/ATR/Donchian/EMA/mean-reversion/strategy-vote features.

Experiment:

- output:
  `checkpoints/transformer_15m/exp8_algo_all_transformers_2026`,
- dataset:
  `checkpoints/transformer_15m/shared_15m_40sym_algo.parquet`,
- rows: `1,353,246`,
- symbols: `40`,
- training rows: `1,222,712`,
- 2026 evaluation rows: `91,816`,
- train window: `2021-01-01` to `2026-01-01`,
- test window: `2026-01-01` to `2026-05-16`,
- models:
  `patchtst`, `patchtransformer`, `itransformer`, `temporal_fusion`,
  `decision_transformer`, `trajectory_transformer`, `perceiver`,
  `cross_asset`, `jepa_patch`,
- ensembles: every 2-model and 3-model combination,
- total leaderboard rows: `129`,
- initial epochs: `16`,
- patch epochs per interval: `1`,
- device: `mps`,
- max sampled training rows: `250,000`,
- gates: `min_pred_profit=0.48`, `max_pred_crash=0.60`,
  `min_buy_prob=0.25`.

Best overall results:

| Rank | Model | Trades | Strategy | SPY | Active alpha | Max DD | Profit rate |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `patchtransformer+trajectory_transformer+cross_asset` | 331 | `-4.20%` | `+9.52%` | `-13.73%` | `-14.53%` | `52.57%` |
| 2 | `patchtst+patchtransformer+cross_asset` | 409 | `-7.44%` | `+9.52%` | `-16.97%` | `-22.83%` | `52.32%` |
| 3 | `patchtst+itransformer+cross_asset` | 335 | `-8.50%` | `+9.52%` | `-18.02%` | `-20.88%` | `54.03%` |
| 4 | `patchtst+cross_asset` | 438 | `-10.61%` | `+9.52%` | `-20.13%` | `-17.89%` | `54.11%` |
| 5 | `patchtransformer+temporal_fusion+cross_asset` | 430 | `-11.18%` | `+9.52%` | `-20.70%` | `-15.83%` | `55.35%` |

Individual model ranking:

| Model | Trades | Strategy | SPY | Active alpha | Max DD | Profit rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `trajectory_transformer` | 558 | `-19.04%` | `+9.52%` | `-28.57%` | `-24.59%` | `50.72%` |
| `itransformer` | 371 | `-23.23%` | `+9.52%` | `-32.75%` | `-26.86%` | `52.02%` |
| `patchtransformer` | 501 | `-26.70%` | `+9.52%` | `-36.22%` | `-33.83%` | `51.70%` |
| `jepa_patch` | 508 | `-32.65%` | `+9.52%` | `-42.17%` | `-34.31%` | `50.39%` |
| `decision_transformer` | 529 | `-35.87%` | `+9.52%` | `-45.39%` | `-38.08%` | `49.15%` |
| `perceiver` | 631 | `-36.02%` | `+9.52%` | `-45.54%` | `-43.49%` | `48.65%` |
| `cross_asset` | 423 | `-37.91%` | `+9.52%` | `-47.43%` | `-38.25%` | `49.17%` |
| `patchtst` | 677 | `-40.14%` | `+9.52%` | `-49.66%` | `-41.31%` | `50.66%` |
| `temporal_fusion` | 894 | `-41.71%` | `+9.52%` | `-51.23%` | `-42.21%` | `53.91%` |

Interpretation:

Adding `itransformer` and `patchtransformer` helped the architecture search but
did not create a tradable model. The best ensemble reduced the damage from
large double-digit losses to `-4.20%`, but it still failed the actual goal:
beat SPY on unseen 2026 data.

Useful signal from this run:

- ensembles are materially safer than the worst individual models,
- `cross_asset` is weak alone but appears useful as a diversifying ensemble
  component,
- `trajectory_transformer`, `itransformer`, and `patchtransformer` are the best
  individual architecture candidates from this group,
- `temporal_fusion` and `perceiver` are not reliable in this algo-feature setup,
  despite being plausible model families.

Next implication:

Do not continue by only adding more transformer variants. The best model is
still selecting too many losing intervals. The next experiment should make the
primary problem a calibrated abstention/risk task:

- train a separate no-trade/regime gate on pre-2026 validation folds,
- require the gate to predict positive expected portfolio return after costs,
- optimize drawdown/Sortino/alpha, not only per-row utility labels,
- only allow the selector/ensemble to trade when the gate approves the market
  regime.

## Longer Training Check on Best Algo-Feature Ensemble

Question:

Would more initial epochs help the best algo-feature transformer ensemble?

Experiment:

- output:
  `checkpoints/transformer_15m/exp9_algo_best_combo_32ep_2026`,
- dataset:
  `checkpoints/transformer_15m/shared_15m_40sym_algo.parquet`,
- train window: `2021-01-01` to `2026-01-01`,
- test window: `2026-01-01` to `2026-05-16`,
- models:
  `patchtransformer`, `trajectory_transformer`, `cross_asset`,
- initial epochs: `32`,
- patch epochs per interval: `1`,
- device: `mps`,
- same 40-symbol universe, feature schema, costs, and gates as the 16-epoch
  algo-feature ensemble test.

Result:

| Model | Active intervals | Trades | Strategy | SPY | Active alpha | Max DD | Profit rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `patchtransformer+trajectory_transformer+cross_asset` | 367 | 447 | `-3.17%` | `+9.52%` | `-12.69%` | `-22.64%` | `48.77%` |
| `trajectory_transformer` | 631 | 880 | `-11.32%` | `+9.52%` | `-20.84%` | `-16.82%` | `52.50%` |
| `trajectory_transformer+cross_asset` | 411 | 501 | `-12.28%` | `+9.52%` | `-21.80%` | `-17.83%` | `47.90%` |
| `patchtransformer+cross_asset` | 342 | 413 | `-14.32%` | `+9.52%` | `-23.85%` | `-21.36%` | `51.09%` |
| `patchtransformer` | 567 | 699 | `-31.91%` | `+9.52%` | `-41.43%` | `-34.47%` | `50.64%` |
| `cross_asset` | 342 | 415 | `-35.25%` | `+9.52%` | `-44.77%` | `-35.29%` | `46.27%` |

Comparison to the prior 16-epoch best ensemble:

| Setup | Strategy | SPY | Active alpha | Max DD |
| --- | ---: | ---: | ---: | ---: |
| 16 epochs, best triple | `-4.20%` | `+9.52%` | `-13.73%` | `-14.53%` |
| 32 epochs, same triple | `-3.17%` | `+9.52%` | `-12.69%` | `-22.64%` |

Interpretation:

More epochs helped the best triple slightly on total return and helped
`trajectory_transformer` materially as an individual model. But the same longer
training worsened `patchtransformer`, worsened drawdown for the triple, and
still failed to beat SPY.

Conclusion:

Training longer is not the main missing ingredient. It can improve selected
architectures, especially `trajectory_transformer`, but without a validation
selected checkpoint and a regime/no-trade gate, longer training risks fitting
historical quirks while still deploying capital during bad intervals.

## Portfolio Failure Analysis

Added reusable analysis script:

`analyze_transformer_portfolio_policy.py`

Purpose:

Replay an existing 15-minute transformer experiment without retraining and test
portfolio-management overlays such as:

- top-1/top-2/top-3 position caps,
- exposure scaling,
- daily loss stops,
- symbol cooldowns after sharp losses,
- SPY momentum gates,
- strategy momentum gates,
- per-symbol daily trade caps,
- transaction-cost sensitivity.

Analyzed run:

- experiment:
  `checkpoints/transformer_15m/exp9_algo_best_combo_32ep_2026`,
- model:
  `patchtransformer+trajectory_transformer+cross_asset`,
- baseline:
  `-3.17%` strategy return vs `+9.52%` SPY,
- baseline active intervals: `367`,
- baseline trades: `447`,
- baseline max drawdown: `-22.64%`.

Key diagnosis:

1. The strategy has a tiny gross edge, but transaction cost consumes it.

   The best 32-epoch triple had average active-interval gross return of about
   `+0.0764%`, while the assumed roundtrip cost is `0.0800%`. That turns a
   small gross edge into a slightly negative expected active interval.

   Cost sensitivity:

   | Roundtrip cost | Baseline strategy | Symbol-cooldown strategy | SPY |
   | ---: | ---: | ---: | ---: |
   | `0 bps` | `+29.87%` | `+38.54%` | `+9.52%` |
   | `4 bps` | `+12.14%` | `+22.25%` | `+9.52%` |
   | `6 bps` | `+4.21%` | `+14.83%` | `+9.52%` |
   | `8 bps` | `-3.17%` | `+7.87%` | `+9.52%` |
   | `10 bps` | `-10.02%` | `+1.32%` | `+9.52%` |

   Implication: the model cannot be deployed on names/venues where real
   roundtrip spread plus slippage is near `8 bps` or worse. It needs lower-cost
   liquid names, fewer trades, or a much stronger expected-return threshold.

2. Losses are clustered in time, especially April and May 2026.

   Baseline monthly path for the best 32-epoch triple:

   | Month | Strategy | SPY | Active intervals |
   | --- | ---: | ---: | ---: |
   | `2026-01` | `-2.2%` | `+1.4%` | 56 |
   | `2026-02` | `+6.6%` | `-1.4%` | 27 |
   | `2026-03` | `+9.0%` | `-4.3%` | 39 |
   | `2026-04` | `-7.8%` | `+10.4%` | 148 |
   | `2026-05` | `-7.6%` | `+3.6%` | 97 |

   The model did well when SPY was weak in February/March, but it failed badly
   when SPY rallied in April/May. This suggests the issue is not only market
   crash avoidance. It is also stock selection and sector/symbol rotation
   during risk-on regimes.

3. A few repeated symbols drive much of the portfolio risk.

   Worst intervals repeatedly included `ALB`, `APP`, `AKAM`, and `AMD`.
   A simple causal symbol cooldown after a `-2%` trade loss, blocking that
   symbol for `78` decision intervals, improved the 8 bps result from `-3.17%`
   to `+7.87%` and reduced max drawdown from `-22.64%` to `-8.44%`.

   This is not a final tradable result because it was selected after inspecting
   the 2026 holdout. But it proves the portfolio layer is a major failure point.

4. The selector can trade even when the combined score is negative.

   The current selection logic gates on predicted profit/crash/buy probability,
   then ranks by:

   `utility + 0.05 * profit + 0.10 * buy_prob - 0.35 * crash - 0.05 * sell_prob`

   It does not require the final `pred_score` to be positive or to exceed
   expected cost. In the best 32-epoch triple, average selected `pred_score` was
   still negative. This means the system can deploy capital into the least bad
   candidates at a timestamp instead of abstaining.

5. Position sizing is too naive.

   The current simulation equal-weights up to three selected names and subtracts
   one interval cost. It does not:

   - size by predicted edge after cost,
   - scale by realized volatility,
   - cap repeated single-symbol exposure over several days,
   - reduce risk after recent model losses,
   - enforce a portfolio expected-return hurdle.

   Also, if real trading cost is charged per position rather than once per
   active interval, the current backtest is optimistic.

Best portfolio overlay found in this diagnostic sweep:

| Policy | Strategy | SPY | Active alpha | Max DD | Active intervals | Trades |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | `-3.17%` | `+9.52%` | `-12.69%` | `-22.64%` | 367 | 447 |
| symbol cooldown after `-2%`, pause 78 intervals | `+7.87%` | `+9.52%` | `-1.66%` | `-8.44%` | 313 | 359 |

Interpretation:

The model is closer than the raw negative return suggests, but it is not
portfolio-ready. The weak point is not only prediction. The portfolio manager
needs to reject low-edge trades, account for costs explicitly, cool down
symbols after sharp losses, and reduce/stop exposure after the strategy enters
a bad local regime.

Next portfolio-management experiment:

- add a required minimum predicted score/edge after costs,
- add causal symbol cooldown after large realized 15-minute losses,
- add per-symbol daily trade caps,
- run the cooldown/score thresholds on pre-2026 validation only,
- freeze the chosen portfolio rules,
- test once on 2026.

Until that validation-frozen portfolio layer beats SPY, the model remains
research-only.

## Integrated Portfolio Manager in Transformer Harness

Implemented stateful portfolio-management controls directly in
`train_15m_transformer_ensemble.py`.

New controls:

- `--min-pred-score`,
- `--min-pred-utility`,
- `--portfolio-exposure`,
- `--daily-loss-stop`,
- `--symbol-cooldown-loss`,
- `--symbol-cooldown-intervals`,
- `--symbol-daily-cap`,
- `--spy-momentum-window`,
- `--spy-momentum-min-return`,
- `--strategy-momentum-window`,
- `--strategy-momentum-min-return`,
- `--filter-falling-stocks`,
- `--falling-filter-min-signals`,
- severe-falling thresholds for recent returns, trend slopes, MA distance, and
  algo momentum vote.

The evaluator now keeps causal portfolio state:

- current trading day and daily return,
- symbol cooldowns after large realized losses,
- per-symbol daily trade counts,
- recent portfolio returns,
- recent SPY returns.

This means the live-style replay can now decide not to trade even if the model
emits a candidate.

Smoke test:

- output: `checkpoints/transformer_15m/smoke_managed_portfolio`,
- result: passed after fixing an empty-mask edge case in symbol filtering.

### Broad Falling-Stock Filter Run

Experiment:

- output:
  `checkpoints/transformer_15m/exp10_managed_best_combo_32ep_2026`,
- models:
  `patchtransformer`, `trajectory_transformer`, `cross_asset`,
- initial epochs: `32`,
- manager:
  broad falling-stock filter, symbol cooldown after `-2%` loss for `78`
  intervals, per-symbol daily cap `3`, `min_pred_score=-0.10`.

Result:

| Model | Active intervals | Trades | Strategy | SPY | Active alpha | Max DD | Profit rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `patchtransformer` | 167 | 192 | `-2.96%` | `+9.52%` | `-12.49%` | `-14.88%` | `48.44%` |
| `patchtransformer+trajectory_transformer` | 159 | 172 | `-4.11%` | `+9.52%` | `-13.64%` | `-9.68%` | `51.16%` |
| `patchtransformer+cross_asset` | 109 | 114 | `-5.73%` | `+9.52%` | `-15.25%` | `-7.72%` | `47.37%` |
| `patchtransformer+trajectory_transformer+cross_asset` | 117 | 119 | `-11.44%` | `+9.52%` | `-20.96%` | `-13.99%` | `47.90%` |

Interpretation:

The broad falling-stock filter greatly improved `patchtransformer` by avoiding
many obvious losers, but it over-filtered the ensemble and removed too many
useful trades.

### Strict Severe-Falling Filter Run

Experiment:

- output:
  `checkpoints/transformer_15m/exp11_managed_severe_falling_32ep_2026`,
- models:
  `patchtransformer`, `trajectory_transformer`, `cross_asset`,
- initial epochs: `32`,
- manager:
  symbol cooldown after `-2%` loss for `78` intervals,
  severe-falling filter only when at least `5` bearish signals are present,
  using thresholds:
  `ret_4 < -0.004`, `ret_16 < -0.008`,
  `trend_slope_8 < -0.001`, `trend_slope_16 < -0.0006`,
  `ma8_dist < -0.004`, `ma26_dist < -0.008`,
  `algo_momentum_vote < -0.5`.

Result:

| Model | Active intervals | Trades | Strategy | SPY | Active alpha | Max DD | Profit rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `patchtransformer+trajectory_transformer+cross_asset` | 254 | 289 | `+5.28%` | `+9.52%` | `-4.25%` | `-7.87%` | `50.52%` |
| `trajectory_transformer` | 440 | 579 | `-9.33%` | `+9.52%` | `-18.86%` | `-19.42%` | `52.33%` |
| `patchtransformer+trajectory_transformer` | 306 | 365 | `-14.21%` | `+9.52%` | `-23.74%` | `-16.99%` | `51.51%` |
| `patchtransformer` | 376 | 436 | `-22.35%` | `+9.52%` | `-31.87%` | `-27.13%` | `50.69%` |
| `cross_asset` | 217 | 249 | `-31.77%` | `+9.52%` | `-41.29%` | `-31.77%` | `45.38%` |

Interpretation:

The integrated portfolio manager materially improved the best ensemble:

- raw 32-epoch triple: `-3.17%`, max DD `-22.64%`,
- strict managed triple: `+5.28%`, max DD `-7.87%`.

This is the best integrated 2026 result so far, but it still does not beat SPY
(`+9.52%`). It remains research-only.

Important lesson:

Avoiding falling stocks helps only when the definition is strict. A broad
filter blocks normal pullbacks that can recover. A severe-falling filter plus
symbol cooldown is much better, but the thresholds were informed by 2026
analysis, so they must be validated/frozen on pre-2026 data before being used
as evidence of tradability.

## Pure Algorithmic 15-Minute Baseline

Added baseline script:

`evaluate_algorithmic_15m_strategies.py`

Purpose:

Evaluate classic rule-based trading strategies without any transformer or
learned model. The script uses the same 15-minute feature dataset and the same
kind of portfolio controls:

- severe falling-stock filter,
- symbol cooldown after large losses,
- optional SPY momentum gate,
- optional daily loss stop,
- optional per-symbol daily cap,
- transaction costs,
- max position cap.

Strategies tested:

- `momentum_breakout`,
- `trend_pullback`,
- `spy_relative_strength`,
- `mean_reversion`,
- `algo_vote`,
- `consensus`.

### Default Pure Algo Run

Experiment:

- output:
  `checkpoints/transformer_15m/algo_15m_baseline_2026`,
- dataset:
  `checkpoints/transformer_15m/shared_15m_40sym_algo.parquet`,
- test window:
  `2026-01-01` to `2026-05-16`,
- roundtrip cost: `8 bps`,
- max positions: `3`,
- severe falling-stock filter enabled,
- symbol cooldown after `-2%` loss for `78` intervals.

Result:

| Strategy | Active intervals | Trades | Strategy | SPY | Active alpha | Max DD | Profit rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `momentum_breakout` | 2,359 | 4,913 | `-81.52%` | `+9.46%` | `-90.98%` | `-81.64%` | `49.73%` |
| `trend_pullback` | 2,385 | 6,109 | `-83.03%` | `+9.46%` | `-92.49%` | `-83.04%` | `49.94%` |
| `spy_relative_strength` | 2,382 | 6,221 | `-83.26%` | `+9.46%` | `-92.72%` | `-83.78%` | `50.06%` |
| `mean_reversion` | 2,371 | 4,083 | `-84.77%` | `+9.46%` | `-94.23%` | `-84.88%` | `48.67%` |
| `algo_vote` | 2,389 | 7,045 | `-85.95%` | `+9.46%` | `-95.41%` | `-86.04%` | `49.06%` |
| `consensus` | 2,391 | 7,146 | `-81.55%` | `+9.46%` | `-91.01%` | `-81.58%` | `50.46%` |

Diagnosis:

The default pure algorithmic rules overtrade. Even when the hit rate is near
50%, the average trade return is far below the `8 bps` roundtrip cost, so the
portfolio bleeds almost every active interval.

### Strict Top-1 Algorithmic Run

Experiment:

- output:
  `checkpoints/transformer_15m/algo_15m_strict_q985_top1_spy26_2026`,
- max positions: `1`,
- score quantile: `0.985`,
- SPY momentum gate: `26` intervals,
- same severe falling filter and cooldown.

Best result:

| Strategy | Active intervals | Trades | Strategy | SPY | Active alpha | Max DD | Profit rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `consensus` | 1,211 | 1,211 | `-54.53%` | `+9.46%` | `-63.99%` | `-56.78%` | `50.70%` |
| `trend_pullback` | 1,235 | 1,235 | `-54.79%` | `+9.46%` | `-64.26%` | `-54.82%` | `49.72%` |

Still far too much trading and no durable edge.

### Risk-Clamped Pure Algo Run

Experiment:

- output:
  `checkpoints/transformer_15m/algo_15m_risk_clamped_no_strategy_gate_2026`,
- max positions: `1`,
- score quantile: `0.985`,
- SPY momentum gate: `26` intervals,
- daily loss stop: `-0.5%`,
- per-symbol daily cap: `1`,
- portfolio exposure: `0.5`.

Best result:

| Strategy | Active intervals | Trades | Strategy | SPY | Active alpha | Max DD | Profit rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `trend_pullback` | 278 | 278 | `-7.79%` | `+9.46%` | `-17.25%` | `-9.25%` | `52.16%` |
| `consensus` | 257 | 257 | `-8.33%` | `+9.46%` | `-17.79%` | `-9.38%` | `51.36%` |

Interpretation:

Risk clamps prevent catastrophic overtrading, but pure algorithmic rules still
do not beat cash or SPY. Their average trade return is still below transaction
cost after realistic filtering.

Comparison to transformer ensemble:

| Approach | Best 2026 return | SPY | Max DD |
| --- | ---: | ---: | ---: |
| pure algorithmic, default | `-81.52%` | `+9.46%` | `-81.64%` |
| pure algorithmic, strict top-1 | `-54.53%` | `+9.46%` | `-56.78%` |
| pure algorithmic, risk-clamped | `-7.79%` | `+9.46%` | `-9.25%` |
| managed transformer triple | `+5.28%` | `+9.52%` | `-7.87%` |

Conclusion:

The transformer ensemble is adding real signal compared with hand-coded
technical rules. The portfolio manager is also essential. Pure algorithmic
strategies alone are not enough on this 15-minute dataset.

## Advanced Pure Algorithmic Top-10 Test

The first top-10 attempt using
`checkpoints/transformer_15m/shared_15m_40sym_algo.parquet` was misleading
because that dataset was built from an alphabetic 40-symbol universe. Only two
of the volume/valuation top-10 names were present.

Built a true top-10 volume/valuation dataset:

`checkpoints/transformer_15m/shared_15m_top10_volume_valuation_algo.parquet`

Universe:

`AAPL`, `AMZN`, `AVGO`, `GOOG`, `GOOGL`, `META`, `MSFT`, `MU`, `NVDA`, `TSLA`

Rows: `356,053`

Added more advanced rule families to `evaluate_algorithmic_15m_strategies.py`:

- `vwap_trend_reclaim`,
- `opening_range_breakout`,
- `volatility_breakout`,
- `liquidity_momentum`,
- `pullback_continuation`,
- `adaptive_consensus`.

These include intraday features such as day-open return and opening-range
breakout state, plus VWAP/volume/volatility/regime filters.

### Advanced Top-10 Run

Experiment:

- output:
  `checkpoints/transformer_15m/algo_15m_advanced_true_top10_2026`,
- dataset:
  `checkpoints/transformer_15m/shared_15m_top10_volume_valuation_algo.parquet`,
- test window:
  `2026-01-01` to `2026-05-16`,
- max positions: `2`,
- score quantile: `0.90`,
- SPY momentum gate: `16`,
- severe falling-stock filter enabled,
- symbol cooldown after `-2%` loss for `78` intervals,
- roundtrip cost: `8 bps`.

Best result:

| Strategy | Active intervals | Trades | Strategy | SPY | Active alpha | Max DD | Profit rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `volatility_breakout` | 425 | 425 | `-14.78%` | `+9.22%` | `-24.01%` | `-25.99%` | `50.59%` |
| `liquidity_momentum` | 937 | 937 | `-23.33%` | `+9.22%` | `-32.56%` | `-34.40%` | `51.01%` |
| `mean_reversion` | 832 | 832 | `-42.77%` | `+9.22%` | `-51.99%` | `-44.54%` | `52.16%` |

### Risk-Clamped Advanced Top-10 Run

Experiment:

- output:
  `checkpoints/transformer_15m/algo_15m_advanced_true_top10_risk_clamped_2026`,
- max positions: `1`,
- score quantile: `0.985`,
- SPY momentum gate: `26`,
- daily loss stop: `-0.5%`,
- per-symbol daily cap: `1`,
- portfolio exposure: `0.5`,
- same severe falling filter and cooldown.

Best result:

| Strategy | Active intervals | Trades | Strategy | SPY | Active alpha | Max DD | Profit rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `volatility_breakout` | 187 | 187 | `-1.33%` | `+9.22%` | `-10.55%` | `-4.76%` | `50.80%` |
| `liquidity_momentum` | 236 | 236 | `-3.88%` | `+9.22%` | `-13.11%` | `-8.95%` | `51.69%` |
| `pullback_continuation` | 283 | 283 | `-4.07%` | `+9.22%` | `-13.30%` | `-5.55%` | `54.42%` |
| `spy_relative_strength` | 188 | 188 | `-4.31%` | `+9.22%` | `-13.53%` | `-7.16%` | `49.47%` |

Interpretation:

Advanced rules plus top-10 filtering are much safer than naive pure
algorithmic trading, but they still do not beat cash or SPY. The best rule,
`volatility_breakout`, gets close to flat after risk clamps (`-1.33%`) and has
low drawdown, but it still cannot generate enough net edge after costs.

Comparison:

| Approach | Best 2026 return | SPY | Max DD |
| --- | ---: | ---: | ---: |
| pure algorithmic, 40-symbol risk-clamped | `-7.79%` | `+9.46%` | `-9.25%` |
| advanced pure algorithmic, true top-10 | `-14.78%` | `+9.22%` | `-25.99%` |
| advanced pure algorithmic, true top-10 risk-clamped | `-1.33%` | `+9.22%` | `-4.76%` |
| managed transformer triple | `+5.28%` | `+9.52%` | `-7.87%` |

Conclusion:

The advanced top-10 pure algorithmic approach improves safety but still lacks
return. It is useful as a portfolio/risk overlay candidate, especially
`volatility_breakout`, but not as a standalone trading model.
