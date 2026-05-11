# Action-Conditioned Portfolio World-Model Dataset

This dataset is the first step away from the old forecaster architecture.
Instead of training only:

```text
past market window -> future return
```

it generates counterfactual rows for:

```text
market state + portfolio state + action + horizon -> future portfolio outcome
```

The builder lives in `world_model_dataset.py` and writes parquet files under
`data/world_model/`, which is gitignored.

## Current Read-First Status

As of 2026-05-11, the best local direction is the plain intraday-120
action-conditioned world model plus a second-stage allocator/ranker. The
cross-sectional and narrower horizon specialists are not champions.

Current base world-model checkpoint:

```text
checkpoints/world_model/world_model_full500_intraday120_8m_actionkey.pt
```

Best allocator checkpoints:

```text
checkpoints/world_model/allocator_intraday120_q80.pt
checkpoints/world_model/allocator_intraday120_q90.pt
```

The allocator is trained by `train_allocator.py`. It scores candidate rows with
the frozen world model, then learns a compact planner score from predicted
outcomes plus action/portfolio metadata. It does not train on the untouched eval
split.

Untouched base world-model eval result on
`cached193_eval_shared250_full_counterfactual` with
`--max-horizon-bars 120`:

| selector | groups | mean return | profit rate | beat-SPY rate | mean alpha vs SPY |
|---|---:|---:|---:|---:|---:|
| forced planner | 1000 | +0.001974 | 51.7% | 51.3% | +0.001630 |
| fixed-score q80 | 200 | +0.005555 | 60.5% | 58.5% | +0.004511 |
| fixed-score q85 | 150 | +0.006954 | 62.7% | 61.3% | +0.006069 |
| buy-only | 1000 | +0.001977 | n/a | 52.2% | n/a |
| random | 1000 | +0.000021 | n/a | 45.4% | n/a |

Allocator untouched eval results:

| allocator | selector | groups | mean return | profit rate | beat-SPY rate | mean alpha vs SPY |
|---|---|---:|---:|---:|---:|---:|
| q80-label allocator | forced planner | 1000 | +0.001457 | 51.0% | 51.3% | +0.001440 |
| q80-label allocator | q80 | 200 | +0.005947 | 63.0% | 59.5% | +0.006031 |
| q80-label allocator | q90 | 100 | +0.009776 | 69.0% | 66.0% | +0.010586 |
| q80-label allocator | q95 | 50 | +0.013363 | 76.0% | 72.0% | +0.013620 |
| q90-label allocator | forced planner | 1000 | +0.000450 | 52.7% | 52.3% | +0.000506 |
| q90-label allocator | q80 | 200 | +0.006515 | 63.0% | 61.5% | +0.006214 |
| q90-label allocator | q90 | 100 | +0.009726 | 73.0% | 69.0% | +0.009925 |
| q90-label allocator | q95 | 50 | +0.010786 | 72.0% | 68.0% | +0.010702 |

Interpretation: this is the strongest result so far. The q80-label allocator
has the best ultra-selective q95 return/beat-SPY slice; the q90-label allocator
is stronger around q80/q90 and has better broad beat-SPY rates. This is still
not deployment-ready. Do not claim the model can reliably beat SPY in live
trading until it survives walk-forward evaluation, transaction-cost stress,
liquidity filters, max-position/cash behavior, and a locked final test period.

Recent follow-up iterations:

| iteration | checkpoint/script | untouched eval readout | decision |
|---|---|---|---|
| Cross-sectional features | `world_model_full500_xsec_intraday120_actionkey.pt` | forced planner -0.000386, beat-SPY 49.7%; q80 +0.001003, beat-SPY 56.5% | not champion |
| Regularized xsec | `world_model_full500_xsec_intraday120_regularized.pt` | forced planner +0.000392, beat-SPY 50.0%; q90 +0.002170, beat-SPY 60.0% | not champion |
| Score tuner | `tune_planner_score.py` on the intraday-120 model | fixed tuned planner +0.001190, beat-SPY 51.7%; worse than the original fixed planner and q80/q85 | keep as diagnostic, not champion |
| Horizon specialist 30-60 | `world_model_full500_h30_60_8m_actionkey.pt` | forced planner +0.001527, beat-SPY 49.0%; q60 +0.003254, beat-SPY 49.5% | not champion |
| Horizon specialist 15-30 | `world_model_full500_h15_30_8m_actionkey.pt` | forced planner +0.000047, beat-SPY 47.8%; q95 +0.004248, beat-SPY 68.0% over only 25 groups | not champion, maybe useful as a selective sub-signal |
| Second-stage allocator | `allocator_intraday120_q80.pt`, `allocator_intraday120_q90.pt` | q80/q90/q95 threshold slices improved over the fixed planner | current best direction |

Recommended next experiments:

- Use walk-forward model selection before touching the final eval split again.
- Extend `train_allocator.py` with walk-forward folds and a locked threshold
  chosen on validation only.
- Add peer/market context more carefully: prune xsec features, add sector/peer
  ranks, and test them behind walk-forward validation.
- Stress every candidate with fees, slippage, liquidity, max-position, and
  cash-when-no-trade behavior.

## Row Semantics

Each row asks one causal question:

```text
At timestamp t, for symbol S, if the portfolio currently has position p and
we change it to target position q, what happens over horizon H?
```

The initial action space was intentionally simple:

- `hold`: keep 0%, 5%, 10%, or 20% exposure.
- `buy`: move from 0% or small exposure to 5%, 10%, or 20%.
- `sell`: move from 5%, 10%, or 20% exposure to cash.

Later iterations add two larger grids:

- `rich`: 0%, 2%, 5%, 10%, 15%, 20%, and 30% target exposures.
- `full`: 0%, 5%, 10%, 25%, 50%, 75%, and 100% target exposures.

Use `--action-mode full` when the goal is to decide whether to deploy capital
into one symbol and compare against SPY. The capped grids can find alpha, but
they are structurally disadvantaged against a full-SPY benchmark.

Fees and slippage use the same constants as the frozen simulator.

## State Features

The dataset reuses the existing causal `experiment.featurize()` output:

- symbol returns, ranges, volatility, EMA distances, volume features
- SPY/TLT/UUP context features
- SPY multi-horizon regime features

It also adds compact state summaries:

- returns over 30m, 2h, 1d, 5d, 20d
- realized volatility over the same windows
- volume z-scores over the same windows
- 5d drawdown from recent high

These are tabular state features for the first model. A later LeWorld-style
model can extend the builder to emit sequence windows keyed by the same row ids.

## Targets

For each action and horizon, the builder computes realized outcomes:

- `final_equity`
- `portfolio_return`
- `portfolio_pnl`
- `max_drawdown`
- `path_vol`
- `min_equity`
- `max_equity`
- `future_asset_return`
- `future_spy_return`
- `future_alpha_vs_spy`
- `profit_label`
- `beat_spy_label`

These targets let us train value, risk, and ranking heads. The eventual world
model should predict a distribution of outcomes, not just one expected return.

Training clips regression targets to robust ranges before normalization:

- `portfolio_return`: -100% to +300%
- `max_drawdown`: -100% to 0%
- `path_vol`: 0% to 20%
- `future_alpha_vs_spy`: -100% to +300%

The same clips are applied to predictions during planner evaluation. This keeps
bad adjusted-price artifacts from dominating the score.

## Example

Small smoke build:

```bash
.venv/bin/python world_model_dataset.py \
  --top500 \
  --symbol-limit 5 \
  --samples-per-symbol 25 \
  --actions-per-timestamp 4 \
  --horizons 30,390,1950 \
  --output data/world_model/smoke.parquet
```

Larger exploratory build:

```bash
.venv/bin/python world_model_dataset.py \
  --top500 \
  --symbol-limit 100 \
  --samples-per-symbol 1000 \
  --actions-per-timestamp 8 \
  --horizons 30,120,390,1170,1950 \
  --shard-by-symbol \
  --output data/world_model/top100_train_counterfactual
```

Sharded mode is recommended for larger builds because it writes one parquet file
per symbol instead of keeping millions of rows in memory.

## First Built Dataset

The first local dataset was built with:

```bash
.venv/bin/python world_model_dataset.py \
  --top500 \
  --symbol-limit 100 \
  --samples-per-symbol 200 \
  --actions-per-timestamp 8 \
  --horizons 30,120,390,1170,1950 \
  --shard-by-symbol \
  --output data/world_model/top100_train_counterfactual
```

Local result:

- 100 parquet shards
- 800,000 counterfactual rows
- 75 columns per row
- 52 state feature columns
- 10 target columns
- about 47 MB on disk

The generated parquet files are under `data/world_model/`, which is ignored by
git. Rebuild them locally when needed.

## Next Model

The next model should be action-conditioned:

```text
z_t = encoder(market_state, portfolio_state)
z_{t+h} = predictor(z_t, action, horizon)
outcome = heads(z_{t+h})
```

Training losses:

- latent prediction loss
- SIGReg/LeJEPA anti-collapse regularizer
- regression loss for portfolio return and drawdown
- quantile loss for downside/upside outcomes
- ranking loss across candidate actions from the same timestamp

The first benchmark should be simple: each week, score candidate buy-and-hold
actions for the next week and compare the selected basket against SPY and the
old top20 picker.

## Current World-Model Iteration

The current pipeline is a separate action-conditioned world model, not the old
forecaster. Important changes:

- Sharded counterfactual datasets can use every cached symbol with `--cached-all`.
- Shared decision timestamps are enabled with `--shared-timestamps`, so rank
  labels and planner choices compare symbols/actions at the same market moment.
- Rich horizons are enabled with `--horizon-mode rich`:
  15, 30, 60, 120, 240, 390, 780, 1170, 1950, 3900, and 7800 bars.
- The trainer uses symbol dropout, a rank top-quartile head, a validation gap,
  early stopping, balanced shard sampling, and robust target clipping.
- Action embeddings use full transition keys such as `buy|0.00->1.00`, not only
  the coarse `buy`, `sell`, and `hold` labels.
- The evaluator supports balanced sampled evaluation and cash thresholding.

Datasets built locally:

| dataset | rows | symbols | action mode | timestamps | size |
|---|---:|---:|---|---:|---:|
| `cached193_shared_rich_counterfactual` | 6,077,544 | 193 | `rich` | 250 | 338 MB |
| `cached193_shared1000_rich_counterfactual` | 16,263,368 | 193 | `rich` | 1,000 | 979 MB |
| `cached193_shared500_full_counterfactual` | 12,203,532 | 193 | `full` | 500 | 681 MB |

Best current checkpoint:

```text
checkpoints/world_model/world_model_full500_8m_actionkey.pt
```

It was trained on a balanced 8M-row sample from the full-action dataset:

```bash
.venv/bin/python train_world_model.py \
  --data data/world_model/cached193_shared500_full_counterfactual \
  --limit-rows 8000000 \
  --epochs 12 \
  --batch-size 32768 \
  --hidden-dim 384 \
  --n-layers 4 \
  --dropout 0.30 \
  --lr 1e-4 \
  --weight-decay 1e-3 \
  --symbol-dropout 0.20 \
  --rank-loss-coef 0.75 \
  --patience 4 \
  --val-gap-days 14 \
  --output checkpoints/world_model/world_model_full500_8m_actionkey.pt
```

Best validation epoch:

- epoch: 4
- train rows: 6,306,663
- validation rows: 1,584,536
- validation groups: 1,177
- action categories: 49 transition keys
- profit-label accuracy: 58.3%
- beat-SPY-label accuracy: 57.7%
- device: Apple MPS

Planner result:

| selector | groups | mean return | mean PnL | profit rate | beat-SPY rate | mean alpha vs SPY |
|---|---:|---:|---:|---:|---:|---:|
| world-model planner | 1,177 | +0.018575 | $+928.77 | 56.0% | 53.4% | +0.017090 |
| q50 threshold planner | 589 | +0.035727 | $+1,786.36 | 56.5% | 54.3% | +0.035373 |
| q70 threshold planner | 353 | +0.052078 | $+2,603.92 | 56.9% | 55.5% | +0.056499 |
| q95 threshold planner | 59 | +0.149285 | $+7,464.26 | 61.0% | 66.1% | +0.183409 |
| buy-only planner | 1,177 | +0.019118 | $+955.89 | 55.0% | 52.9% | +0.016535 |
| random candidate | 1,177 | +0.001563 | $+78.17 | 44.8% | 41.5% | -0.002412 |
| oracle candidate | 1,177 | +0.155908 | $+7,795.42 | 100.0% | 99.7% | +0.149831 |

Interpretation: this is the first iteration that beats random and beats SPY on
the validation candidate groups. The q50/q70 threshold planners are the most
interesting operating points so far. It is promising, but not yet
"tradable-ready": the validation sample is still in-sample to model selection,
the strategy needs walk-forward evaluation on untouched dates, and
portfolio-level sizing across multiple simultaneous symbols is not implemented
yet.

## Untouched Eval Split Check

After selecting the best validation checkpoint, an untouched eval dataset was
built from `prepare.py`'s chronological eval split, which is the last 180
calendar days of cached bars:

```bash
.venv/bin/python world_model_dataset.py \
  --cached-all \
  --symbol-limit 0 \
  --samples-per-symbol 250 \
  --actions-per-timestamp 12 \
  --action-mode full \
  --horizon-mode rich \
  --shared-timestamps \
  --shard-by-symbol \
  --split eval \
  --output data/world_model/cached193_eval_shared250_full_counterfactual
```

Local result:

- rows: 4,351,908
- scored groups: 2,750
- rows scored: 4,351,908

Evaluation command:

```bash
.venv/bin/python evaluate_world_model.py \
  --data data/world_model/cached193_eval_shared250_full_counterfactual \
  --checkpoint checkpoints/world_model/world_model_full500_8m_actionkey.pt \
  --batch-size 32768 \
  --score-all \
  --output checkpoints/world_model/world_model_full500_8m_actionkey_eval_split_all.json
```

Untouched eval result:

| selector | groups | mean return | mean PnL | profit rate | beat-SPY rate | mean alpha vs SPY |
|---|---:|---:|---:|---:|---:|---:|
| world-model planner | 2,750 | -0.032021 | $-1,601.04 | 47.7% | 47.0% | -0.038085 |
| q50 threshold planner | 1,375 | -0.056262 | $-2,813.08 | 41.3% | 40.9% | -0.065537 |
| buy-only planner | 2,750 | -0.029169 | $-1,458.46 | 48.0% | 47.3% | -0.035552 |
| hold cash | 2,739 | 0.000000 | $0.00 | 0.0% | 49.3% | +0.001788 |
| random candidate | 2,750 | +0.001161 | $+58.07 | 42.5% | 46.3% | +0.001054 |
| oracle candidate | 2,750 | +0.302798 | $+15,139.90 | 99.7% | 99.7% | +0.302914 |

Interpretation: the model is not tradable-ready. It overfits the training/held
validation regime and becomes overconfident on long-horizon buys in the
untouched eval regime. The oracle result shows the candidate set still contains
large opportunity, but the current model does not rank it robustly. Next work
should focus on walk-forward training/evaluation, regime-balanced sampling,
horizon-specific planners, and an abstention/calibration objective that is
validated only on untouched periods.

## Short-Horizon Follow-Up

The first fix was to train a dedicated short-horizon model using only horizons
up to one trading day (`horizon_bars <= 390`). This directly targets the failure
mode from the full-horizon model, which was overconfident on long-horizon buys.

Training command:

```bash
.venv/bin/python train_world_model.py \
  --data data/world_model/cached193_shared500_full_counterfactual \
  --limit-rows 8000000 \
  --max-horizon-bars 390 \
  --epochs 12 \
  --batch-size 32768 \
  --hidden-dim 384 \
  --n-layers 4 \
  --dropout 0.30 \
  --lr 1e-4 \
  --weight-decay 1e-3 \
  --symbol-dropout 0.20 \
  --rank-loss-coef 0.75 \
  --patience 4 \
  --val-gap-days 14 \
  --output checkpoints/world_model/world_model_full500_short_8m_actionkey.pt
```

Training result:

- rows after horizon filtering: 4,358,466
- train rows: 3,440,256
- validation rows: 864,278
- best epoch: 8
- validation profit accuracy: 59.5%
- validation beat-SPY accuracy: 55.8%

Untouched eval command:

```bash
.venv/bin/python evaluate_world_model.py \
  --data data/world_model/cached193_eval_shared250_full_counterfactual \
  --checkpoint checkpoints/world_model/world_model_full500_short_8m_actionkey.pt \
  --batch-size 32768 \
  --score-all \
  --max-horizon-bars 390 \
  --output checkpoints/world_model/world_model_full500_short_8m_actionkey_eval_split_all.json
```

Untouched eval result:

| selector | groups | mean return | mean PnL | profit rate | beat-SPY rate | mean alpha vs SPY |
|---|---:|---:|---:|---:|---:|---:|
| short-horizon planner | 1,500 | +0.000859 | $+42.96 | 49.5% | 51.0% | +0.000924 |
| q80 threshold planner | 300 | +0.003518 | $+175.91 | 56.3% | 55.0% | +0.003536 |
| q95 threshold planner | 75 | +0.008193 | $+409.66 | 65.3% | 61.3% | +0.007588 |
| buy-only planner | 1,500 | +0.001023 | $+51.13 | 51.1% | 51.3% | +0.001134 |
| random candidate | 1,500 | -0.000030 | $-1.49 | 40.1% | 46.1% | -0.000400 |
| oracle candidate | 1,500 | +0.067838 | $+3,391.89 | 99.6% | 99.6% | +0.067580 |

Interpretation: the short-horizon model does generalize better than the
full-horizon model, but the edge is still small. It is a useful direction, not a
finished trading model. The next step should split intraday, one-day, and
multi-day horizons into separate planners and train the abstention threshold on
walk-forward validation only.

## Short-Horizon Fine-Tune

The short-horizon checkpoint was fine-tuned for 5 more low-learning-rate epochs:

```bash
.venv/bin/python train_world_model.py \
  --data data/world_model/cached193_shared500_full_counterfactual \
  --limit-rows 8000000 \
  --max-horizon-bars 390 \
  --epochs 5 \
  --batch-size 32768 \
  --hidden-dim 384 \
  --n-layers 4 \
  --dropout 0.30 \
  --lr 3e-5 \
  --weight-decay 1e-3 \
  --symbol-dropout 0.20 \
  --rank-loss-coef 0.75 \
  --patience 3 \
  --val-gap-days 14 \
  --init-checkpoint checkpoints/world_model/world_model_full500_short_8m_actionkey.pt \
  --output checkpoints/world_model/world_model_full500_short_8m_actionkey_ft5.pt
```

Fine-tune result:

- best extra epoch: 2
- best validation loss: 1.3701
- validation profit accuracy: 59.6%
- validation beat-SPY accuracy: 55.9%

Untouched eval result:

| selector | groups | mean return | mean PnL | profit rate | beat-SPY rate | mean alpha vs SPY |
|---|---:|---:|---:|---:|---:|---:|
| fine-tuned planner | 1,500 | +0.001050 | $+52.50 | 49.7% | 51.2% | +0.000905 |
| q80 threshold planner | 300 | +0.003406 | $+170.28 | 53.7% | 52.0% | +0.003152 |
| q95 threshold planner | 75 | +0.007071 | $+353.54 | 61.3% | 57.3% | +0.005581 |
| buy-only planner | 1,500 | +0.001342 | $+67.10 | 51.3% | 51.3% | +0.001343 |
| random candidate | 1,500 | -0.000030 | $-1.49 | 40.1% | 46.1% | -0.000400 |

Interpretation: the fine-tune slightly improved the forced planner mean return
but weakened the thresholded high-confidence slices compared with the previous
short-horizon checkpoint. Keep `world_model_full500_short_8m_actionkey.pt` as
the safer baseline for now.

## Intraday 120-Bar Variant

The next horizon split trained only the 15, 30, 60, and 120 bar horizons:

```bash
.venv/bin/python train_world_model.py \
  --data data/world_model/cached193_shared500_full_counterfactual \
  --limit-rows 8000000 \
  --max-horizon-bars 120 \
  --epochs 12 \
  --batch-size 32768 \
  --hidden-dim 384 \
  --n-layers 4 \
  --dropout 0.30 \
  --lr 1e-4 \
  --weight-decay 1e-3 \
  --symbol-dropout 0.20 \
  --rank-loss-coef 0.75 \
  --patience 4 \
  --val-gap-days 14 \
  --output checkpoints/world_model/world_model_full500_intraday120_8m_actionkey.pt
```

Untouched eval result:

| selector | groups | mean return | mean PnL | profit rate | beat-SPY rate | mean alpha vs SPY |
|---|---:|---:|---:|---:|---:|---:|
| intraday-120 planner | 1,000 | +0.001974 | $+98.70 | 51.7% | 51.3% | +0.001630 |
| q50 threshold planner | 500 | +0.003245 | $+162.25 | 53.0% | 51.2% | +0.002514 |
| q80 threshold planner | 200 | +0.005555 | $+277.74 | 60.5% | 58.5% | +0.004511 |
| q85 threshold planner | 150 | +0.006954 | $+347.69 | 62.7% | 61.3% | +0.006069 |
| buy-only planner | 1,000 | +0.001977 | $+98.83 | 52.1% | 52.2% | +0.001711 |
| random candidate | 1,000 | +0.000021 | $+1.06 | 39.7% | 45.4% | +0.000091 |

Interpretation: this is the best plain model family so far. It is still not a
large enough edge for deployment, but the high-confidence slices are materially
better than random on the untouched eval split.

## Cross-Sectional Feature Attempt

The dataset builder now supports `--cross-sectional`. It adds universe movement
features at each shared decision timestamp:

- breadth/count features
- 30m, 2h, and 1d universe return mean/median/std/p10/p90/dispersion
- per-symbol cross-sectional rank percentile
- symbol return minus universe median
- up-fraction features
- 1d volatility and volume-z cross-sectional summaries/ranks

Train/eval datasets:

```bash
.venv/bin/python world_model_dataset.py \
  --cached-all \
  --symbol-limit 0 \
  --samples-per-symbol 500 \
  --actions-per-timestamp 12 \
  --action-mode full \
  --horizons 15,30,60,120 \
  --shared-timestamps \
  --cross-sectional \
  --shard-by-symbol \
  --output data/world_model/cached193_shared500_full_xsec_intraday120

.venv/bin/python world_model_dataset.py \
  --cached-all \
  --symbol-limit 0 \
  --samples-per-symbol 250 \
  --actions-per-timestamp 12 \
  --action-mode full \
  --horizons 15,30,60,120 \
  --shared-timestamps \
  --cross-sectional \
  --shard-by-symbol \
  --split eval \
  --output data/world_model/cached193_eval_shared250_full_xsec_intraday120
```

Local dataset results:

- train rows: 4,543,488
- eval rows: 1,937,808
- model features: 103 total, including 44 `xsec_` features

Two xsec models were trained:

| model | planner return | planner beat-SPY | best threshold | threshold return | threshold beat-SPY |
|---|---:|---:|---|---:|---:|
| plain intraday-120 baseline | +0.001974 | 51.3% | q50 | +0.003245 | 51.2% |
| xsec MLP | -0.000386 | 49.7% | q80 | +0.001003 | 56.5% |
| xsec regularized MLP | +0.000392 | 50.0% | q90 | +0.002170 | 60.0% |

Interpretation: adding cross-sectional features naively did not improve the
main planner. It helped some narrow thresholded beat-SPY slices, but returns
were weaker than the plain intraday-120 model. Keep the feature code, but do not
use the current xsec checkpoints as champions. Better next experiments:
feature pruning, sector/peer grouping, a separate cross-sectional ranker, and
calibrating the threshold on walk-forward validation rather than final eval.

## First Trained Model

The first baseline trainer is `train_world_model.py`. It trains a compact
tabular action-conditioned model:

- symbol embedding
- action embedding
- normalized market/portfolio state features
- MLP latent trunk
- regression heads for return, drawdown, volatility, and alpha
- classification heads for profit and beat-SPY labels

Training command:

```bash
.venv/bin/python train_world_model.py \
  --data data/world_model/top100_train_counterfactual \
  --epochs 12 \
  --batch-size 8192 \
  --hidden-dim 256 \
  --n-layers 4 \
  --dropout 0.10 \
  --output checkpoints/world_model/world_model_v1.pt
```

Local training result:

- rows: 800,000
- train rows: 640,000
- validation rows: 160,000
- device: Apple MPS
- elapsed: 64.7 seconds
- best epoch: 2
- best validation loss: 1.0979
- validation MAE portfolio return: 0.002119
- validation MAE max drawdown: 0.001421
- validation MAE future alpha vs SPY: 0.023300
- profit-label accuracy: 67.1%
- beat-SPY-label accuracy: 59.0%

The trainer saves the best validation epoch, not the final epoch, because this
first model overfits after epoch 2.

## First Planner Evaluation

The planner evaluator is `evaluate_world_model.py`. It loads the trained
checkpoint, scores validation-set candidate actions, and chooses the best action
per `(timestamp, horizon)` by a simple risk-adjusted score:

```text
pred_return
+ 0.20 * pred_alpha_vs_spy
+ 0.50 * pred_beat_spy_probability
+ 0.25 * pred_profit_probability
+ 0.50 * pred_max_drawdown
- 0.10 * pred_path_vol
```

Evaluation command:

```bash
.venv/bin/python evaluate_world_model.py \
  --data data/world_model/top100_train_counterfactual \
  --checkpoint checkpoints/world_model/world_model_v1.pt \
  --output checkpoints/world_model/world_model_v1_eval.json
```

Validation-set planner result:

| selector | groups | mean return | mean PnL | profit rate | beat-SPY rate | mean alpha vs SPY |
|---|---:|---:|---:|---:|---:|---:|
| world-model planner | 19,565 | +0.000217 | $+10.87 | 51.3% | 41.3% | -0.001735 |
| buy-only planner | 19,565 | +0.000167 | $+8.35 | 50.8% | 40.9% | -0.001773 |
| random candidate | 19,565 | +0.000053 | $+2.64 | 34.5% | 40.6% | -0.001701 |
| oracle candidate | 19,565 | +0.002609 | $+130.46 | 52.0% | 48.4% | -0.001384 |

Interpretation: the first world model learns enough to beat random candidate
selection on mean realized return, but it is not a tradable edge yet. It still
loses to SPY most of the time, and even the oracle over this candidate set has
only a 48.4% beat-SPY rate. The next dataset/model iteration should improve the
candidate set and train directly for action ranking within each timestamp.
