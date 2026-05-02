# Iteration log

Every autoresearch iteration writes a full report here. Most recent first.

| status | iter | description | sharpe | PnL |
|---|---|---|---:|---:|
| 🟢 | [iter 071 — beaa5e6](iter_071_beaa5e6.md) | exp71: top5 ranking horizons (4,8,10)→(3,4) — shorter, less noisy | **+1.549** | **$+4797.33** (+9.595%) |
| 🟢 | [iter 070 — fa4ab20](iter_070_fa4ab20.md) | exp70: concentration sweep — add top1/top3 to profile suite | **+1.295** | **$+2944.08** (+5.888%) |
| 🟢 | [iter 069 — 4ec6ecb](iter_069_4ec6ecb.md) | exp69: canonical = top5_picker (exp68 proved it beats SPY across seeds) | **+1.295** | **$+2944.08** (+5.888%) |
| 🔴 | [iter 068 — e94504e](iter_068_e94504e.md) | exp68: FRESH full pretrain (no cache) — ranking loss actually engaged this time | **+1.037** | **$+2275.95** (+4.552%) |
| 🔴 | [iter 067 — 5467e68](iter_067_5467e68.md) | exp67: N_SEEDS=3 multi-seed validation (fair CI vs prior best) | **+0.818** | **$+1540.17** (+3.080%) |
| 🔴 | [iter 066 — 06076ef](iter_066_06076ef.md) | exp66 SPEEDUP: precompute predictions (validate same numbers as exp65, faster wa | **+1.037** | **$+2275.95** (+4.552%) |
| 🔴 | [iter 065 — c091321](iter_065_c091321.md) | exp65: relax DD floor -10→-15 (SPY-realistic) — should KEEP exp64's +1.04 sharpe | **+1.037** | **$+2275.95** (+4.552%) |
| 🔴 | [iter 064 — e3199bb](iter_064_e3199bb.md) | exp64: top20_picker = canonical strategy (proves exp63 alpha by changing the bro | **+1.037** | **$+2275.95** (+4.552%) |
| 🔴 | [iter 063 — 2c0ab38](iter_063_2c0ab38.md) | exp63: cross-sectional ranking loss + per-timestep standardization (research-bac | **-2.359** | **$-3586.84** (-7.174%) |
| 🔴 | [iter 062 — 6fb35d3](iter_062_6fb35d3.md) | exp62: rank-percentile=0.90 on active profiles (top-decile) | **-1.189** | **$-158.00** (-0.316%) |
| 🔴 | [iter 061 — 7db4e02](iter_061_7db4e02.md) | exp61: bigger model + 4 active profiles + 3 passive top-N pickers + SPY benchmar | **-1.189** | **$-158.00** (-0.316%) |
| 🔴 | [iter 060 — 8cad106](iter_060_8cad106.md) | exp60: multi-trader-profile comparison (intraday/intraweek/intramonth/longterm v | **-1.189** | **$-158.00** (-0.316%) |
| 🔴 | [iter 059 — b20af41](iter_059_b20af41.md) | exp59: foundation reset — 11 horizons, n_layers=4, 3 epochs, 2yr train, 95 syms | **-2.121** | **$-575.39** (-1.151%) |
| 🔴 | [iter 058 — 0307051](iter_058_0307051.md) | exp58: bigger model + 95-sym + MPS + N_SEEDS=1 | **+0.425** | **$+617.19** (+1.234%) |
| 🔴 | [iter 057 — 2d9ea93](iter_057_2d9ea93.md) | exp57: volume-aware slippage + liquidity gate | **+0.734** | **$+1286.82** (+2.574%) |
| 🔴 | [iter 056 — 2ce550f](iter_056_2ce550f.md) | exp56: extended universe (~95 names) + VIX removed + full retrain | **+0.796** | **$+1404.38** (+2.809%) |
| 🔴 | [iter 055 — 56c1960](iter_055_56c1960.md) | exp55: 1-hour min hold per position (no slot cap, BUY > 0, SWAP > 0.20) | **+1.092** | **$+2122.31** (+4.245%) |
| 🔴 | [iter 054 — 91b0b60](iter_054_91b0b60.md) | exp54: BUY_THRESHOLD=0.5 (model picks any high-conviction names) + SWAP disabled | **+1.080** | **$+2391.43** (+4.783%) |
| 🔴 | [iter 053 — e243765](iter_053_e243765.md) | exp53: SWAP_MARGIN 0.15→1.0 + cached pretrain (5min iteration) | **+1.080** | **$+2391.43** (+4.783%) |
| 🔴 | [iter 052 — 09119c6](iter_052_09119c6.md) | exp52: WEIGHTED_MAX_CONCURRENT=5 — force selectivity, fix buy-and-hold bug | **-57.870** | **$-44800.91** (-89.602%) |
| 🟢 | [iter 051 — 2f8bd0b](iter_051_2f8bd0b.md) | exp51: SPY-alpha reward (coef=0.5) | **+1.629** | **$+3597.60** (+7.195%) |
| 🟢 | [iter 050 — 29c9eaf](iter_050_29c9eaf.md) | exp50: ENTROPY_COEF 0.005 + SWAP_MARGIN 0.15 + holdout eval enabled | **+1.626** | **$+3505.58** (+7.011%) |
| 🔴 | [iter 049 — e9a95a5](iter_049_e9a95a5.md) | exp49: SWAP_MARGIN 0.20→0.15 at cap=0.50 (more rotations) | **+1.548** | **$+3442.85** (+6.886%) |
| 🔴 | [iter 048 — 37994af](iter_048_37994af.md) | exp48: SWAP + cap 0.50→0.55 | **+1.510** | **$+3135.88** (+6.272%) |
| 🟢 | [iter 047 — 6e143de · exp47](iter_047_6e143de.md) | SWAP pass + cap 0.65 → 0.50 — bring DD floor under -10% on all seeds | **+1.535** | **$+3,260.33** (+6.521%) |
| 🟢 | [iter 042 — 8988cee · exp42](iter_042_8988cee.md) | cap 0.95 → 0.80 — bring DD safely under floor | **+0.988** | **$+2,557.00** (+5.114%) |

← [back to project README](../README.md)
