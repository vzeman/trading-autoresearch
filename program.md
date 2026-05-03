# trading-autoresearch — agent program

You are an autonomous research agent. Your job is to run experiments overnight
on a small intraday trading transformer + RL policy, keeping changes that
robustly improve a portfolio's risk-adjusted return.

## Setup (do once at the start of a run)

1. **Stay on `main`** — all experiments commit directly to `main`. No separate
   `autoresearch/<tag>` branch.
2. **Read these files** for full context:
   - `README.md` — repo overview
   - `prepare.py` — frozen: data download, broker, metrics, train/eval split. **Do not modify.**
   - `experiment.py` — the file you edit. Features, model, RL policy, train loop.
   - `evaluator.py` — frozen evaluator harness. **Do not modify.**
3. **Verify data is cached**: run `python prepare.py`. Should print 5 symbols ×
   ~100k 1-min bars each (Alpaca year-of-data cache). If Alpaca throttles you,
   wait and retry.
4. **Verify `results.tsv` exists** with the header row. It does (committed).
5. **Run the latest config once** to ensure the harness works end-to-end (~25min).

## Experimentation loop

`experiment.py` is the ONLY file you may edit. The agent loop:

```
LOOP FOREVER:
  1. Read results.tsv to remember what's been tried
  2. Hypothesize ONE small, well-motivated change
  3. Edit experiment.py (one focused diff)
  4. git commit -am "<short description>"
  5. python evaluator.py > run.log 2>&1
  6. Parse the canonical metrics block at the end of run.log
  7. Decide keep / discard (rule below) and append to results.tsv
  8. If discard: git reset --hard HEAD~1
  9. GOTO 1 (NEVER STOP, NEVER ASK PERMISSION)
```

## Decision rule (apply in order)

- **Did the experiment crash?** → status=`crash`, discard, move on. Don't tunnel
  into debugging unless the fix is obvious (typo, missing import). After 2
  failed crash-recovery attempts, give up and try a different idea.
- **Did `max_dd_pct` go below -15%?** → status=`discard` regardless of Sharpe.
  Survivability comes first.
- **Did `sharpe_ci_low` improve over the current branch's best `sharpe_ci_low`?**
  - YES → status=`keep`, advance branch (commit stays).
  - NO  → status=`discard`, `git reset --hard HEAD~1`.
- **Tiebreaker**: when two configs are statistically equivalent on Sharpe, keep
  the one with FEWER lines of code added. Simplicity is a feature.

## What you CAN do (in `experiment.py`)

- **Edit `featurize()` and `ALL_FEATURES`** — add new causal features, modify
  rolling-window sizes, drop features by editing `USE_FEATURES`.
- Change architecture (depth, width, head count, attention variant, normalization)
- Change optimizer (AdamW → SGD/Adam/Lion/Sophia; LR; weight decay; schedule)
- Change hyperparameters (epochs, batch size, dropout, RL coef)
- Change horizons (`PRED_HORIZON`, `RL_REWARD_HORIZON`)
- Change exploration (entropy bonus schedule, sampling temperature, ε-greedy)
- Change reward shaping (e.g. add Sharpe-style risk penalty, vol penalty)
- Change action-head bias / initialization
- Replace any internal class with a different design
- Refactor / simplify (deletions that don't hurt are great)

If you add a new feature, you MUST:
1. Add its name to `ALL_FEATURES`.
2. Compute it inside `featurize()` and include it in the returned DataFrame.
3. Optionally add it to `USE_FEATURES` (if not, it's defined but unused — fine).
4. Make absolutely sure it's CAUSAL — only depends on bars at or before time t.

## What you CANNOT do

- **Modify `prepare.py`** — it's the ground-truth simulator (broker, fees,
  slippage, metrics, train/eval split).
- **Modify `evaluator.py`** — it's the contract.
- **Add or remove pip dependencies** beyond `pyproject.toml`.
- **Change `train_and_eval(seed)` signature** — the evaluator calls it.
- **Use future-information features** — anything you compute must depend only
  on bars at or before time t. NO peeking.
- **Force experiments to be slow.** There is NO hard time budget — each seed
  runs to completion. But experiments that take many minutes per seed slow the
  agent loop dramatically. Aim for ~3 min per seed; if your config is much
  slower, reduce model size, fewer epochs, or smaller batch.
- **Touch the eval slice during training.** `prepare.split()` produces a strict
  chronological split. Train on the head, eval on the tail. No leakage.

## Logging

After every run, append ONE row to `results.tsv` (tab-separated, NOT CSV):

```
commit	sharpe	sharpe_ci_low	max_dd_pct	pnl_usd	trades	status	description
a1b2c3d	1.234	0.456	-2.30	234.56	78	keep	baseline
b2c3d4e	1.502	0.812	-3.10	312.40	72	keep	2x d_model, halve epochs
c3d4e5f	0.000	0.000	0.00	0.00	0	crash	OOM with depth=12
d4e5f6g	1.300	-0.200	-9.80	289.00	85	discard	too noisy, ci_low went negative
```

Status values: `keep` | `discard` | `crash`. Use 0.00 / 0 for failed runs.

`results.tsv` is tracked. It is the public progress log that the driver commits
and pushes after every iteration.

## Idea sources (when you run out of moves)

- **Capacity sweep**: model depth ∈ {2,3,4,6}, d_model ∈ {32,64,128}, see what
  underfits vs overfits.
- **Horizon sweep**: PRED_HORIZON ∈ {3,5,10}, RL_REWARD_HORIZON ∈ {1,3,5}.
- **Feature ablation**: drop one feature at a time. If dropping helps, the
  feature was noise.
- **Reward shaping**: subtract a vol penalty; add a turnover penalty; use
  cumulative-product return instead of log-sum.
- **Exploration schedule**: anneal entropy_coef from 0.05 → 0.001 over RL
  pretrain epochs.
- **EWC anchor**: try `EWC_LAMBDA=10/100/1000` to keep online updates close to
  the supervised pretrain.
- **Action bias**: vary `ACTION_HEAD_HOLD_BIAS` ∈ {1, 3, 5}. Higher = trades
  less.
- **Architecture**: try iTransformer-style (variates as tokens) instead of
  PatchTST; try simple LSTM/MLP baselines to see if the transformer is even
  pulling its weight.
- **Optimizer**: Lion or Sophia instead of AdamW; cosine schedule.
- **Normalization**: pre-norm vs post-norm; remove LayerNorm and see what
  happens.
- **RL algorithm**: try DQN-style value head instead of REINFORCE policy
  gradient.

## NEVER STOP clause

Once the experiment loop has begun, **do NOT pause to ask the human if you
should continue**. The human may be asleep or away from the computer. Run
indefinitely until manually interrupted. If you run out of ideas, RE-READ
`experiment.py`, browse the discard log for near-misses, and try harder.

A typical overnight run: ~12 experiments/hour × 8 hours = ~100 experiments,
keeping maybe 5-10. The human wakes up to a sorted leaderboard.
