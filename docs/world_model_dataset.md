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

The action space in v1 is intentionally simple:

- `hold`: keep 0%, 5%, 10%, or 20% exposure.
- `buy`: move from 0% or small exposure to 5%, 10%, or 20%.
- `sell`: move from 5%, 10%, or 20% exposure to cash.

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
