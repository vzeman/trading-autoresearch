# Autoresearch Policy

This repo optimizes a portfolio-management model under a continuous research loop.
Every iteration must make an explicit decision about whether to train weights, change
portfolio strategy, or both.

## Long-Term Objective

Maximize robust out-of-sample portfolio management quality, not just training loss:

- Beat SPY/SP500 for longer fractions of the evaluation window.
- Improve Sharpe lower confidence bound.
- Preserve low drawdown and survivability.
- Keep realistic trading costs, including fixed fees, slippage, and liquidity impact.
- Adapt to newly received Alpaca market data without leaking future information.

## Iteration Decision Checklist

Before every experiment, decide and document the primary lever:

1. **Fresh model training**
   - Use when the data cache changed materially, features changed, model architecture changed, or cached weights may be stale.
   - Choose `PRETRAIN_EPOCHS` deliberately.
   - Default is `1` because prior tests showed lower loss can overfit and hurt trading.
   - Try `2+` epochs only as an explicit experiment, then judge by out-of-sample portfolio metrics.

2. **Portfolio-management optimization**
   - Use cached weights when only strategy/risk allocation changes.
   - Test top-N selection, reserve/cash policy, rebalance cadence, position caps, turnover limits, market-regime filters, and SPY-relative objectives.
   - Reward strategies that stay above SPY/SP500 longer, while also requiring positive risk-adjusted return.

3. **Feature/model research**
   - Add only causal features.
   - Prefer market-context features that improve regime awareness: SPY trends, broad ETF context, volatility, dispersion, drawdown, breadth, and relative strength.
   - Validate every feature through held-out trading performance, not feature intuition.

4. **Data adaptation**
   - Refresh missing/latest Alpaca data regularly.
   - Prefer async refresh during long training so training does not block on data download.
   - Log data freshness in the iteration report.

## Epoch Policy

Training loss is diagnostic, not the target.

- If `nll_running_mean` drops but Sharpe CI-low, drawdown, or SPY-outperformance worsens, treat it as overfit.
- If a fresh train improves evaluation, keep the checkpoint and consider a local epoch sweep around it.
- If fresh training repeatedly fails but cached strategy sweeps improve, prioritize portfolio-management research.
- Always keep the best checkpoint by the objective gate; never overwrite it with an unproven run.

## Reporting Requirements

Each iteration report should make clear:

- Whether it trained weights or reused cached weights.
- Epoch counts and loss trend when training was performed.
- Winning strategy definition and what the seed means.
- SPY/SP500 comparison, including time above SPY.
- 1d, 1w, 1mo, 3mo, and 6mo live-style simulations when available.
- Positive-seed transaction table only; omit transaction tables for negative seeds.

## Continuous Loop Rule

After an iteration completes:

1. Push code/report/results to GitHub.
2. Decide the next highest-value lever: more training, strategy optimization, feature/model change, or data refresh.
3. Start the next iteration immediately unless manually interrupted.

