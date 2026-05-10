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
