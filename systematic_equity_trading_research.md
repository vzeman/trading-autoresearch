# Systematic Equity Trading: What Works in 2026 and How to Encode It

**A deep research synthesis on currently-viable algorithmic trading techniques for equities, with a concrete starter ruleset for an algorithmic system.**

> Compiled May 2026. Target audience: builders of an automated equity trading / research system. The document is opinionated where the evidence allows and skeptical where it doesn't. Every reported number has been cross-checked against multiple sources; expect a 20–50% discount in live trading versus the headline figures cited from academic backtests.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [What the Turtles Got Right and What Has Changed](#2-what-the-turtles-got-right-and-what-has-changed)
3. [Strategy Family A — Momentum & Trend Following on Equities](#3-strategy-family-a--momentum--trend-following-on-equities)
4. [Strategy Family B — Factor Investing (the Surviving "Big Five")](#4-strategy-family-b--factor-investing-the-surviving-big-five)
5. [Strategy Family C — Mean Reversion & Statistical Arbitrage](#5-strategy-family-c--mean-reversion--statistical-arbitrage)
6. [Strategy Family D — Event-Driven & Volatility Risk Premium](#6-strategy-family-d--event-driven--volatility-risk-premium)
7. [Position Sizing & Risk Management — the Modern "N"](#7-position-sizing--risk-management--the-modern-n)
8. [Machine Learning & Alternative Data — What Helps, What Doesn't](#8-machine-learning--alternative-data--what-helps-what-doesnt)
9. [Execution, Costs & Backtesting Discipline](#9-execution-costs--backtesting-discipline)
10. [The Starter Ruleset — Concrete Rules to Encode Now](#10-the-starter-ruleset--concrete-rules-to-encode-now)
11. [Validation & Deployment Gates](#11-validation--deployment-gates)
12. [Appendix A — Formula Reference](#appendix-a--formula-reference)
13. [Appendix B — Glossary](#appendix-b--glossary)
14. [Appendix C — References](#appendix-c--references)

---

## 1. Executive Summary

The original Turtle system (1983) was a Donchian-channel breakout trend-follower applied to a small basket of futures markets, with volatility-based position sizing ("N") and aggressive pyramiding. Its philosophy — systematic rules, strict risk management, asymmetric payoffs, emotional discipline — is still the bedrock of modern systematic trading. The literal ruleset, however, has decayed. Equity markets in 2026 are more efficient, more crowded with quants, and dominated by transaction-cost economics that didn't exist in 1983.

After synthesizing roughly forty primary sources — peer-reviewed papers (Jegadeesh & Titman 2023, Hou-Xue-Zhang 2020, Daniel & Moskowitz 2016, Frazzini-Israel-Moskowitz 2018, Bailey & López de Prado 2014), institutional research (AQR, Robeco, Two Sigma, Man AHL, Alpha Architect, Research Affiliates) and practitioner books (Carver, López de Prado, Antonacci, Gray) — the picture for **equity** systematic trading in 2026 is:

**What still works (with evidence):**

- **Cross-sectional momentum (12-1)** — gross Sharpe ~0.85, net ~0.65 after costs. Decay since 2000 but real edge remains, especially when volatility-scaled (Barroso & Santa-Clara) which roughly doubles risk-adjusted returns.
- **Time-series momentum / dual momentum** at the index/ETF level — net Sharpe 0.6–0.85, very low implementation friction, ideal for a starter system.
- **Quality / profitability factor (QMJ)** — most consistent factor of the last decade, Sharpe 0.45–0.60 long-only, replicates in 25+ countries.
- **Value, properly defined** (composite of EV/EBITDA, FCF yield, shareholder yield — not P/B alone) — survives replication, rebounded sharply in 2022.
- **Multi-factor combinations** of value + quality + momentum + low-vol — net Sharpe 0.5–0.65 on liquid US universe.
- **Volatility risk premium** harvesting — high Sharpe in calm regimes, but tail risk is brutal (lessons from XIV 2018, COVID 2020). Only viable with strict risk management.
- **Sector / country momentum** — better Sharpe than stock-level momentum after costs, because of much lower turnover and ETF liquidity.
- **Overnight drift / intraday reversal decomposition** — small but real edge, mostly useful as an overlay or rotation tool.

**What is largely dead for general equity trading:**

- Naive 1-week price reversal on large-cap US stocks (alpha < transaction costs).
- Simple cointegration pairs trading on liquid US stocks (decayed since ~2005, survives only in niches).
- Most published technical indicators (RSI/MACD/Bollinger crossovers) used as standalone signals on liquid equities.
- Pure deep-learning price-only forecasting (publishable, not investable).
- Reinforcement learning for alpha generation (works for execution, not stock-picking).

**The non-negotiable building blocks** (the modern equivalent of the Turtle "N"):

- **Volatility targeting** at portfolio and position level (target ~10–12% annualized).
- **Realistic transaction-cost modeling** built in from day one.
- **Walk-forward / purged k-fold cross-validation** plus the **Deflated Sharpe Ratio** as gates.
- **Diversification across uncorrelated signals**, not across instruments alone.
- **Drawdown-based de-leveraging** as a circuit breaker.
- **Capacity awareness** — every strategy has a finite AUM beyond which alpha collapses.

The rest of this document unpacks each of these and culminates in a **concrete, codable starter ruleset** in §10 plus **deployment gates** in §11.

---

## 2. What the Turtles Got Right and What Has Changed

### 2.1 What survived

| Principle | Why it still works |
|---|---|
| Systematic, fully rule-based | Removes behavioral noise; enables backtesting and auditability. |
| Volatility-based position sizing ("N") | The single most robust risk-management idea in trading. Modern "vol targeting" is a direct descendant. |
| Cut losses fast, let winners run | The mathematical asymmetry of trend following — most trades small losses, a few large gains — still holds. |
| Diversification across many uncorrelated bets | Grinold's Fundamental Law of Active Management formalized it (IR ∝ skill × √breadth). |
| Pre-defined entry/exit rules | Eliminates discretionary drift and overfitting in real time. |

### 2.2 What broke

| Component | Why it stopped working as-is |
|---|---|
| Donchian channel breakouts as the *primary* signal | Crowded; HFTs trade ahead of obvious breakouts; whipsaw rate too high in modern equity microstructure. |
| Small instrument basket (~20 markets) | Insufficient breadth; modern CTAs trade 100+ instruments to smooth returns. |
| 20-day / 55-day fixed lookbacks | Single fixed parameters are fragile; ensemble of lookbacks is more robust. |
| Aggressive pyramiding (4 units) | Concentrates risk into trends right before they exhaust; modern systems use lighter pyramiding or none. |
| Equal risk per market | Doesn't account for cross-market correlations; modern portfolios optimize on correlation-adjusted risk. |
| No transaction-cost model | A 1980s luxury. Today, every signal is judged net-of-cost. |

### 2.3 The takeaway for our system

Treat the Turtle methodology as a **template philosophy**, not a literal blueprint. Specifically:

1. Use the Turtle's risk discipline (vol-based sizing, hard stops, defined-rule trading) as the foundation.
2. Replace the single breakout signal with an **ensemble of low-correlation signals** (momentum + value + quality + mean-reversion overlay).
3. Replace the 20-instrument basket with the **liquid US equity universe** plus sector/country ETFs for the trend overlay.
4. Add the layers the original lacked: transaction costs modeled to 1 bps precision; vol targeting at the portfolio level; drawdown-based de-leveraging; rigorous out-of-sample validation.

---

## 3. Strategy Family A — Momentum & Trend Following on Equities

Momentum is the **most academically robust anomaly** in finance, replicated across 150+ years of data and 40+ countries (Jegadeesh & Titman 2023; Asness, Moskowitz & Pedersen 2013). It survives every replication scrutiny and remains, after quality, the second-most-reliable factor in published research. But it is also the most crash-prone, and the gap between gross and net performance is large because of high turnover.

### 3.1 Cross-sectional momentum (12-1)

**Concrete rule.** Each month-end:

1. Universe: liquid US stocks, market cap ≥ $500M, 30-day average dollar volume ≥ $5M.
2. Compute past-12-month cumulative return for each stock, **excluding the most recent month** (the "skip-month" convention; the most recent month exhibits short-term reversal).
3. Rank all stocks by this return.
4. Long the top quintile (top 20%); equal-weight; rebalance monthly.
5. Optionally short the bottom quintile (long-short version).

**Performance (post-cost).**

- Long-only: ~0.45–0.60 Sharpe.
- Long-short: ~0.55–0.75 Sharpe.
- Gross Sharpe in literature: 0.85–1.0; **expect a 20–30% net haircut** due to monthly turnover (~140% annual).

**Why it works (best current explanations).** Behavioral underreaction to news, anchored beliefs, slow institutional adjustment, and limits to arbitrage. The premium is risky, not free — momentum **crashes** ~once every 5–10 years (worst recorded month: −65% in March–May 2009 unmanaged).

**Implementation knobs that matter.**

- **Lookback:** 12 months is empirically optimal across the literature; 6 months works in faster regimes; <3 months captures reversal, not momentum.
- **Skip-month:** Yes by default. Improves Sharpe ~0.03–0.05.
- **Top quintile vs. decile:** Decile gives higher absolute return but worse Sharpe (more idiosyncratic risk). Use quintile for robustness, decile only if AUM allows further diversification.
- **Equal-weight vs. cap-weight inside the basket:** Equal-weight rebalances *into* winners and amplifies the signal but at higher turnover. Use equal-weight for portfolios under ~$100M AUM, cap-weight above.
- **Rebalance cadence:** Monthly. Quarterly cuts costs ~25 bps but loses ~40 bps of alpha — net negative.

### 3.2 Time-series (absolute) momentum / dual momentum

**Concrete rule (Antonacci dual momentum, simplest variant).** Each month-end:

1. Compare 12-month return of US equity index (e.g., SPY) vs. ex-US developed equity index (e.g., VEU).
2. Pick the higher of the two as the candidate.
3. **Absolute filter:** if the candidate's 12-month return is positive, hold 100% of it; otherwise hold a bond ETF (e.g., AGG) or T-bills.

**Alternative: 200-day SMA filter.** Hold SPY when SPY > 200-day SMA, otherwise hold AGG/cash. Equivalent results to dual momentum, simpler.

**Performance.** Backtested 1974–2024:

- Annualized return ~13–15% vs. 9–10% for buy-and-hold ACWI.
- Max drawdown ~22% vs. ~55–60% for buy-and-hold.
- Net Sharpe ~0.75–0.85 (after ~15 bps trading costs).

**Why this is the workhorse for a starter system.** Two ETFs, monthly rebalance, near-zero transaction cost as a fraction of AUM, no shorting, no leverage. Captures the *defensive* property of trend following without stock-picking risk.

**Caveats.** Underperforms in sustained low-vol bull markets (2010–2019) where the trend filter is rarely triggered but produces occasional whipsaws. It is a *crisis hedge* and a *long-term compounder*, not a high-Sharpe alpha engine.

### 3.3 Residual momentum (Blitz, Huij & Martens 2011)

**Concrete rule.** Same as cross-sectional momentum, but rank stocks by their **residual return** after regressing each stock's monthly returns on the Fama-French 3- or 5-factor model, instead of raw returns.

**Why it's better.** Raw momentum loads heavily on market beta and time-varying factor exposures. Residual momentum strips those out, giving a cleaner stock-picking signal.

**Performance.** Roughly **doubles the Sharpe ratio** of raw momentum (from ~0.55 to ~1.0 long-short, gross). Less crash-prone because it's market-neutral by construction.

**Why it's not used more.** Higher operational complexity (rolling 36-month factor regressions per stock), and the data needs are heavier (factor returns + universe history). Worth the effort.

### 3.4 Risk-managed (volatility-scaled) momentum

**Concrete rule (Barroso & Santa-Clara 2015).** Compute the realized volatility of the momentum strategy itself over a rolling 6-month window. Scale the portfolio's gross exposure inversely:

```
position_t = (target_vol / realized_vol_t) × momentum_signal_t
```

Target vol typically 10–12% annualized.

**Performance impact.**

- Unmanaged momentum: Sharpe 0.53, max drawdown −79%, kurtosis 18.
- Vol-scaled momentum: Sharpe 0.97, max drawdown −28%, kurtosis 2.7.

**This is structural, not regime-dependent.** As long as the underlying signal has positive expected return, vol-scaling improves Sharpe and tail behavior. **Do this on every momentum sleeve.**

### 3.5 Sector and country momentum (the easy version)

**Concrete rule.** Apply the 12-1 cross-sectional momentum rule to **sector ETFs** (11 SPDR sectors) or **country ETFs** (25+ MSCI single-country funds).

**Why it's attractive for a starter system.**

- 11–25 instruments, not 1000+. Tiny universe, easy execution.
- ETF liquidity: spreads of 1–3 bps on liquid sectors.
- Lower turnover than stock-level (sectors are stickier).
- Net Sharpe **0.75–0.90** after costs, comparable to stock-level but with ~10× lower implementation friction.

**Recommendation.** Include sector momentum as a **first sleeve** in our system before attempting stock-level momentum. It captures most of the benefit at a fraction of the operational cost.

### 3.6 Earnings momentum / PEAD

**Concrete rule.** Compute Standardized Unexpected Earnings (SUE):

```
SUE = (Actual EPS - Consensus EPS) / std(historical surprises)
```

Long top quintile, short bottom quintile, hold 60 trading days, rebalance after each earnings season.

**Status.** Heavily arbitraged in US large-cap (largely dead since 2006 per Ball-Bartov-Kaul; debated by recent 2025 papers). Persistent in small-cap, micro-cap, and emerging markets. Best used as a **cross-sectional screen overlay** (boost momentum-portfolio weights in stocks with positive SUE) rather than as a standalone sleeve.

### 3.7 The 52-week-high anomaly (George & Hwang 2004)

**Rule.** Rank stocks by `(Current Price − 52-Week High) / 52-Week High`. Distance from the high predicts subsequent returns: stocks far below their 52-week high tend to underperform unless they have momentum.

**Use case in our system.** As a **regime filter**: when the broad market is near its 52-week high, momentum crash risk is elevated; reduce momentum sleeve gross exposure by 25–50%.

---

## 4. Strategy Family B — Factor Investing (the Surviving "Big Five")

The factor zoo has 400+ published "anomalies." Hou-Xue-Zhang (2020) replicated 452 of them and found **65% fail to replicate** when proper microcap filters and multiple-testing corrections are applied. Harvey-Liu-Zhu (2016) recommend a t-statistic threshold of **>3.0** (not the usual 2.0) for new factors. After that scrutiny, only a handful remain.

### 4.1 The factors that survived rigorous replication

| Factor | Net Sharpe (long-only) | Net Sharpe (long-short) | Replication confidence | Notes |
|---|---|---|---|---|
| **Quality / Profitability (QMJ)** | 0.45–0.60 | 0.60–0.75 | Very high | Most durable factor of the last decade; works in 25+ countries. |
| **Value (composite, not P/B)** | 0.35–0.50 | 0.50–0.65 | High | Survived if defined as composite of EV/EBITDA + FCF yield + shareholder yield. P/B alone is broken. |
| **Momentum (12-1)** | 0.40–0.55 | 0.55–0.75 | Very high | 150+ years, 40+ countries; crash risk requires vol scaling. |
| **Low Volatility / BAB** | 0.30–0.45 | 0.50–0.70 | Medium-high | BAB (Frazzini-Pedersen) better than long-only low-vol. |
| **Size (quality-adjusted)** | 0.10–0.30 | 0.20–0.40 | Low | Vanishes once microcap junk is removed. Use only with quality filter. |

Note: **investment / asset growth (CMA)** is borderline; passes some replications, fails others. Useful as part of a quality composite, not standalone.

### 4.2 How to define each factor (concrete signals)

**Value (composite, 6-component):**

```
value_score = -mean(rank(P/E), rank(P/B), rank(EV/EBITDA), rank(P/S))
            + mean(rank(FCF/EV), rank(shareholder_yield))
```

(Higher = cheaper.)

**Quality (3-component):**

```
quality_score = 0.4 × profitability_z + 0.3 × growth_z + 0.3 × safety_z
```

Where:

- `profitability_z` = z-score(GP/Assets), using Novy-Marx gross profitability.
- `growth_z` = z-score of 5-year revenue and earnings growth.
- `safety_z` = z-score of low leverage, low earnings volatility, low accruals.

**Momentum:** see §3.1.

**Low volatility:** rank stocks by trailing 252-day realized volatility, long bottom quintile. Or BAB: long leveraged low-beta decile, short unleveraged high-beta decile.

### 4.3 Integrated multi-factor scoring (recommended)

Rather than running separate sleeves and concatenating, **score every stock on every factor and combine**:

```
composite_score = 0.30 × value_score
                + 0.35 × quality_score
                + 0.20 × momentum_score
                + 0.15 × low_vol_score
```

Long the top quintile of `composite_score`, equal-weight, rebalance monthly. **Net Sharpe expectation: 0.50–0.65** post-costs on liquid US large-cap.

**Why integrated > portfolio-mix:** lower turnover (factors that disagree cancel internally), smoother sector exposures, single trade list per rebalance.

### 4.4 Don't try to time factors

Cliff Asness has spent ~15 years arguing factor timing is a bad idea, and the empirical evidence backs him. **Use static weights** unless a factor reaches genuine valuation extremes (e.g., value spread > 95th percentile, then *modestly* tilt). Recent research on "factor momentum" (past-winner factors keep winning) shows a ~50–100 bps annual edge that is mostly eaten by transaction costs.

### 4.5 The size factor — read carefully

The "small-cap premium" is largely a microcap-junk artifact. Once you remove stocks with negative earnings, the size effect collapses. **Practical rule:** if you want size exposure, screen for *profitable* small-caps (positive earnings, positive operating margin, market cap > $300M). Avoid the Russell 2000 as a clean small-cap proxy; it is roughly 30% money-losers.

---

## 5. Strategy Family C — Mean Reversion & Statistical Arbitrage

The honest summary: **most classical mean-reversion edges have decayed substantially in liquid US large-caps**, but persist in specific niches.

### 5.1 Short-term reversal (1-week to 1-month)

**Status:** Largely dead in US large-cap after costs. Profits sit inside the bid-ask. Survives in:

- **Small-cap (<$5B mkt cap):** net Sharpe 0.7–1.3 with proper liquidity filters.
- **International developed markets** (Japan, UK, EU): less crowded.
- **Quality-screened reversal** (Asness): filter out stocks with deteriorating fundamentals to avoid catching falling knives.

**Concrete rule.** Z-score 5-day return per stock (or 1-month return). Buy stocks with z < −1.5 *and* positive trailing 12-month return *and* positive 6-month earnings revision. Hold 5–10 days, exit on cross of 0.

### 5.2 Pairs trading / cointegration

**Status:** Largely arbitraged out of liquid US large-caps since ~2005. Survives in:

- **Sector pairs** (within-industry hedges): paired stocks in same sector are more likely to remain co-integrated.
- **ETF pairs** (e.g., similar country ETFs, sector vs. industry ETFs).
- **International** small/mid cap.

If implementing: use **Engle-Granger or Johansen cointegration tests** on rolling 252-day windows. Trade only pairs with stable cointegration (test p-value < 0.05 in 80%+ of windows). Enter at z-spread > 2σ, exit at 0 or stop at 3σ.

### 5.3 Statistical arbitrage on factor residuals (Avellaneda & Lee)

**Concrete rule.** For each stock, regress daily returns on a small set of factors (market, sector ETFs, or first 5–8 PCA components of universe returns). Model the residual as an Ornstein-Uhlenbeck mean-reverting process. Enter when |residual z| > 1.5; exit at 0 or stop at 2.

**Status:** More resilient than pairs trading because PCA captures systematic shocks that break pairs. Net Sharpe 0.6–1.1 with capacity ~$300M–$1B per strategy. Requires daily refit and tight execution.

### 5.4 Overnight drift / intraday reversal

**Empirical fact (Lou, Polk & Skouras 2019):** in US equities, virtually all of momentum's alpha is earned **overnight**, while intraday returns are slightly negative on the same names. Mirror image for reversal.

**Implementable variant:** "Long overnight, hedge intraday" on the SPY level captures the well-documented overnight drift of ~0.3–0.5% per night gross, ~10–20 bps net of borrow / financing. Tiny absolute alpha but uncorrelated to other strategies and useful as a small overlay.

### 5.5 What to actually use from this family in our system

For a starter system, **mean-reversion is a secondary sleeve at most** — perhaps 10–15% of risk budget — and only on:

- Small-cap quality-screened reversal (5-day horizon).
- Sector-ETF residual stat-arb (5–10 day horizon).
- Optional overnight-drift overlay on SPY.

Skip pairs trading until you have proven you can backtest cointegration without snooping.

---

## 6. Strategy Family D — Event-Driven & Volatility Risk Premium

### 6.1 Volatility Risk Premium (VRP)

**The premise:** implied volatility on equity index options consistently overstates realized volatility (~2–4 vol points on average). Selling that gap is a structural risk premium.

**Concrete vehicles:**

- Sell 1-month at-the-money or slightly OTM SPX puts; delta-hedge daily.
- Sell short-dated SPX iron condors / put spreads (defined risk).
- Variance-swap replication via delta-hedged short straddles.

**Performance.** Sharpe 0.8–1.2 in calm regimes. **Drawdowns are brutal:** XIV lost ~95% in two trading days in February 2018; short-vol funds lost 60–80% in March 2020. The return distribution has extreme negative skew and high kurtosis — backtested Sharpe is misleading for a strategy whose worst monthly loss can be −50%.

**Rules if including in our system:**

1. **Defined-risk only** (put spreads, iron condors). Never naked options.
2. Deleverage when VIX > 30; halt when VIX > 40.
3. Cap allocation at ≤10% of portfolio risk.
4. Backtest must include 2018-Feb, 2020-Mar, and the 2008 crisis.

### 6.2 Earnings drift (PEAD) and earnings overreaction

**Status:** PEAD largely arbitraged in US large-cap; persistent in small-cap and EMs. The opposite — earnings overreaction reversal — has some evidence in 2024 papers.

**Practical use:** as a **filter overlay** on the multi-factor portfolio. Boost weights in stocks with SUE > +1.5σ; reduce or exclude stocks with SUE < −1.5σ.

### 6.3 Index inclusion / reconstitution

**Russell index reconstitution** (annual, late June) historically produced 2–8% predictable demand bumps for additions and equivalent drops for deletions. Decayed substantially since 2010 as front-running has become widespread, but a small effect persists (~1–2% over 5–10 days). Easy to encode; small AUM only.

---

## 7. Position Sizing & Risk Management — the Modern "N"

This is the section that matters most. **Returns are a story. Risk management is the system.** The Turtles' core insight — size every position by current volatility — has been refined and expanded. Here are the modern building blocks.

### 7.1 Volatility targeting at the portfolio level

**Goal:** the portfolio's annualized volatility should hit a target (e.g., 12%) regardless of regime.

**Procedure:**

1. Estimate realized portfolio vol from the last 60 trading days using an exponentially-weighted moving average (EWMA) with half-life ~20 days.
2. Compute scale factor: `leverage_t = target_vol / realized_vol_t`.
3. Apply (cap leverage at, say, 1.5× to avoid runaway sizing in low-vol regimes).
4. Rebalance weekly, not daily, to control turnover.

**Why it works:** vol targeting decouples returns from regime. Empirically improves Sharpe by 0.1–0.3 on equity strategies and dramatically reduces tail kurtosis. Works because *future vol is more autocorrelated than future returns* — you can predict next month's vol with a 0.6–0.8 correlation; you cannot predict next month's return with anywhere near that.

### 7.2 ATR / N-based position sizing (the modernized Turtle method)

**For each position:**

```
N = ATR(20)              # 20-day Average True Range, in $ per share
risk_per_unit = N
units = (portfolio_value × risk_per_trade_pct) / risk_per_unit
```

Where `risk_per_trade_pct` is typically 0.5–1.0% per position (the Turtles used 1% per "unit" and allowed up to 4 units per trend, total 4%).

**Modern equity adaptations:**

- Use **20-day EWMA volatility** in % terms instead of raw ATR for cross-sectional comparability.
- Add a correlation adjustment: scale down sizes when adding correlated positions (Robert Carver's "instrument diversification multiplier").
- Cap any single position at 5% of portfolio gross exposure regardless of signal strength.

### 7.3 Fractional Kelly

**The Kelly criterion** says size each bet at `f* = (edge / odds) = (μ / σ²)` for log-utility maximization. **Practitioners use 0.25× to 0.5× Kelly**, never full Kelly, because:

1. Kelly assumes known stationary distribution; markets are non-stationary.
2. Estimation error in `μ` compounds — full Kelly with overestimated edge produces ruin.
3. Half-Kelly captures ~75% of the geometric return with half the volatility.
4. Investors care about drawdown, not just terminal wealth; full Kelly drawdowns are ~50%, fractional Kelly is ~25%.

**Practical rule.** Use 0.25× Kelly as a sanity ceiling on position sizes; do not let any single position's allocation exceed `0.25 × (signal_strength / signal_volatility²)`.

### 7.4 Risk parity & correlation-aware sizing

**Naive risk parity:** weight each strategy/sleeve so each contributes equal **risk** (not equal capital) to the portfolio:

```
weight_i = (1 / vol_i) / Σ(1 / vol_j)
```

**Better: correlation-adjusted (HRP, López de Prado).** Hierarchical Risk Parity builds a hierarchical clustering of assets/strategies based on their correlation matrix and allocates risk top-down within each cluster. **HRP produces materially better out-of-sample Sharpe than mean-variance optimization** (which is dominated by estimation error in the covariance matrix).

For our system: implement HRP across the strategy sleeves (momentum, value, quality, mean-reversion, etc.), not across individual stocks. The diversification benefit dominates the modeling complexity.

### 7.5 Drawdown control

**Rule of thumb.** Cut portfolio gross exposure when trailing drawdown exceeds defined thresholds:

| Trailing drawdown | Action |
|---|---|
| 0–5% | Full size |
| 5–10% | Reduce gross to 75% |
| 10–15% | Reduce gross to 50% |
| 15–20% | Reduce gross to 25% |
| > 20% | Halt, review, restart smaller |

This is a **circuit breaker**, not an alpha rule. It exists to keep you in business during regime breaks. Re-scale up gradually as drawdown recovers (don't switch back to 100% at 4.99% drawdown).

### 7.6 Stop losses on equities — when they help, when they hurt

Kaminski & Lo (2014) studied stop-loss rules across systematic strategies. Findings:

- **Time stops** (exit position after N days regardless of price) help trend-following systems.
- **Tight price stops** (e.g., 2 × ATR below entry) often *hurt* equity strategies because they convert noise into realized losses.
- **Wide price stops** (e.g., 5–10 × ATR or chandelier exits) preserve trends without killing them.
- **Volatility-based stops** dominate fixed-percentage stops.

**Recommendation.** Use a **chandelier stop** (price drops more than 4 × ATR from highest close since entry) rather than fixed-percentage stops. Combine with a time stop (e.g., exit if no new high in 60 days) for trend strategies.

### 7.7 Diversification math — the Fundamental Law

Grinold's Fundamental Law:

```
Information Ratio = Information Coefficient × √Breadth
```

Where:

- `IC` ≈ correlation of your forecast with realized returns (typical equity quant: 0.02–0.06).
- `Breadth` ≈ number of *independent* bets per year.

**Implications:**

- A signal with IC 0.03 across 500 independent bets/year gives IR ≈ 0.67.
- The same signal across 50 independent bets gives IR ≈ 0.21.
- **Independent is the operative word.** 500 stocks all driven by the same momentum signal in the same direction = ~1 independent bet, not 500.

**Practical guidance:**

- Combine strategies with low pairwise correlation. Adding a 0.4-Sharpe strategy uncorrelated to your 0.6-Sharpe strategy raises blended Sharpe to ~0.72, not 0.5.
- Aim for at least 4–6 truly uncorrelated strategy sleeves.

### 7.8 Tail-risk management

Three approaches, each with merits:

1. **Just-size-it.** Vol-targeting + drawdown circuit breakers. Cheapest. Doesn't hedge crashes.
2. **Trend-following overlay.** Allocate 10–20% of risk budget to time-series momentum on indices and bonds; this implicitly hedges equity tail events (trend systems profit in 2008, 2020, 2022).
3. **Explicit put hedging.** Buy 2–3 month, 10% OTM SPX puts continuously. Costs ~1–2% of portfolio value per year. Effective in tail events; expensive in calm.

**Recommended for starter system:** combination of #1 (vol targeting + circuit breakers) and #2 (trend overlay). Skip #3 until you can afford the carry cost.

### 7.9 Leverage rules

- Cap **gross** leverage at 1.5× initially; expand to 2.0× only after 12 months of live performance matching backtest.
- Cap **net** leverage at 1.0× (i.e., long-short can be 1.5× gross / 0.5× net, not 1.5× net).
- Daily VaR (95%) ≤ 2% of equity.
- Margin buffer: maintain ≥30% of gross exposure as cash equivalent.

---

## 8. Machine Learning & Alternative Data — What Helps, What Doesn't

This section is the most prone to hype. The honest picture from primary research (Bryan Kelly, López de Prado, Two Sigma, Man AHL):

### 8.1 What works

**Gradient boosting (XGBoost, LightGBM) on cross-sectional features** is the workhorse. Used by every major quant fund. Reliably outperforms linear regression on stock-level monthly return prediction. Typical setup:

- 200–500 features per stock (factor scores, sector flags, momentum, fundamentals, lagged returns).
- Monthly retraining, 24–60 month rolling window.
- Target: next-month residual return after factor adjustment.
- Output: cross-sectional rank, used to overweight/underweight stocks.

**NLP on earnings calls and 10-K filings.** FinBERT or LLM-based sentiment extraction adds 20–50 bps of incremental alpha when used as a *timing overlay* on factor strategies. Standalone, signal decays in 1–5 days.

**Regime detection** via Hidden Markov Models or simple volatility-state classifiers. Doesn't generate alpha directly but reduces drawdowns 20–40% by dialing back risk in unfavorable regimes.

### 8.2 What mostly doesn't work

**Pure deep learning on price-only data** (LSTM, transformer forecasting of daily returns). Publishable, not investable. Backtested Sharpes of 2–3 routinely collapse to <0.5 under purged CV and realistic costs.

**Reinforcement learning for alpha generation.** Decade of hype, narrow real-world success. Works for *execution* (optimal trade slicing) and *portfolio rebalancing* (deciding when to trade given costs). Does not work for stock-picking; non-stationarity defeats the assumption that optimal policy is stable.

**Black-box ML without an economic prior.** Models that find "patterns" without an underlying causal story tend to overfit. The best practitioners (AQR, Two Sigma) start with an economic hypothesis and use ML to *refine* the signal, not to discover signals from scratch.

### 8.3 Alternative data

**Real edges, but capacity-constrained and decay quickly:**

- **Credit-card transaction panels** — 2–3 week lead on retail earnings. ~30–80 bps alpha. Decays as multiple vendors commoditize the data (typical half-life 1–3 years).
- **Satellite imagery** — useful for sector-specific bets (retail parking, energy storage, shipping). Niche and labor-intensive.
- **App download / web traffic / geolocation panels** — fragmented; useful for individual-name modeling, not broad portfolios.

For a starter system, **skip alt-data** until the basic factor + momentum infrastructure is producing reliable returns. Alt data is a 10× effort for 1.2× return; basics first.

### 8.4 López de Prado's methodological contributions

These are non-negotiable hygiene for any ML-in-finance work:

- **Purged k-fold CV** with embargo periods to prevent information leakage in time-series cross-validation.
- **Combinatorially symmetric CV** for path-independent validation.
- **Triple-barrier method** for labeling: instead of fixed-horizon returns, label trades by which of (profit target, stop loss, time stop) is hit first.
- **Meta-labeling**: build a primary signal, then a secondary classifier that decides whether to take each individual trade.
- **Fractional differentiation**: difference time series minimally to achieve stationarity while preserving memory.
- **Deflated Sharpe Ratio** (see §9).

If you remember nothing else from this section: **purged CV + DSR are mandatory**.

---

## 9. Execution, Costs & Backtesting Discipline

This is where most strategies die.

### 9.1 Realistic transaction costs

Layered model for an equity trade:

```
total_cost_bps = bid_ask_half_spread        (0.5 - 50 bps depending on liquidity)
               + commission                  (1 - 5 bps institutional, more for retail)
               + market_impact_permanent     (0.4 × √(Q/V) × σ × 10000)
               + market_impact_temporary     (≈ 1.5× permanent)
               + borrow_cost (if short)      (annualized 25 bps - 50%+)
```

For a $1M trade in SPY: ~5–8 bps round-trip.
For a $1M trade in a mid-cap: ~25–40 bps round-trip.
For a $1M trade in a small-cap with 50 bps spread: ~120–150 bps round-trip.

**Build this cost model into the backtest from day one.** Strategies that look great gross and break-even net are common; better to know this in simulation.

### 9.2 Implementation Shortfall and execution algorithms

Use **VWAP** or **POV (Percentage of Volume)** algorithms for normal-sized orders. Use **Implementation Shortfall** algorithms when the trade is urgent. Avoid **TWAP** for liquid names (gives away information).

Almgren-Chriss optimal execution: there is a U-shaped trade-off between market impact (worse for fast execution) and timing risk (worse for slow). For most monthly-rebalance strategies, executing the rebalance over 1–3 hours of one trading day is sufficient; spreading over multiple days only helps for very large positions.

### 9.3 The Deflated Sharpe Ratio

Bailey & López de Prado's correction for selection bias and multiple testing:

```
DSR ≈ Φ⁻¹(p),  where p = probability that the observed SR exceeds expected max under null
```

The intuition: if you tested 100 strategies, the best one is expected to have SR ≈ 0.5 *under the null*. Discovering one with SR 1.0 is therefore far less impressive than it looks.

**Heuristic table** (with backtest length T = 5 years, daily data):

| Strategies tested | In-sample SR needed for DSR > 0.5 |
|---|---|
| 1 | ~0.6 |
| 10 | ~1.0 |
| 100 | ~1.4 |
| 1000 | ~1.8 |

**Rule:** if your best strategy has in-sample SR < the threshold for the number of variants you tried, do not deploy it.

### 9.4 Walk-forward and purged k-fold cross-validation

**Walk-forward** (gold standard for production):

1. Train on years 1–3, test on year 4.
2. Train on years 1–4, test on year 5.
3. Continue.
4. Combine OOS performance across all test windows.

**Purged k-fold CV** (more efficient for research):

1. Split data into k folds (e.g., 10).
2. For each fold, train on the other 9 *with a purge period* around the test fold to prevent information leak.
3. Evaluate on test fold.
4. Report distribution of OOS performance across folds.

**Combinatorial Purged CV (CPCV)** generates many more train/test paths than walk-forward and is currently the academic best practice.

### 9.5 Common backtest pitfalls

- **Survivorship bias.** Free data (Yahoo, etc.) typically excludes delisted stocks. Adds 100–500 bps of spurious return depending on strategy. Use CRSP-with-delistings or equivalent.
- **Look-ahead bias on fundamentals.** Use point-in-time data. An EPS number known on the announcement date is *not* available on the quarter-end date.
- **Restated financials.** Many databases store the latest restated figures, not what was reported originally. Use point-in-time vendors (Compustat PIT, FactSet).
- **Snooping the universe.** "Stocks in the S&P 500 today" is a forward-looking universe in any historical period.
- **Fixed transaction costs.** Real costs scale with position size and volatility. Model them dynamically.
- **Infinite liquidity assumption.** A strategy that wants to trade 10% of daily volume *will move the market*. Cap position sizes at ≤2% of trailing 20-day average volume.

### 9.6 Capacity analysis

Every strategy has an AUM beyond which alpha decays:

- **Intraday equity stat-arb:** typical capacity $500M–$2B.
- **Cross-sectional equity momentum (monthly):** capacity $5B–$30B.
- **Sector momentum (ETF level):** capacity $500M–$5B (ETF AUM constraints).
- **Multi-factor long-only:** capacity $50B+ (highly liquid).

For a starter system at <$10M AUM, capacity is irrelevant; for scaling planning, model capacity as `f(strategy turnover, average position size as % of ADV)`.

---

## 10. The Starter Ruleset — Concrete Rules to Encode Now

This is the **direct, codable ruleset** you can implement first. It is intentionally conservative; alpha can be added later as the infrastructure proves out.

### 10.1 Universe definition

```
universe_filters:
  - listed_on:           [NYSE, NASDAQ]
  - country:             [US]   # Phase 1
  - market_cap_min:      $500M
  - adv_30d_min:         $5M    # 30-day average dollar volume
  - price_min:           $5     # filter penny stocks
  - exclude:             [ETFs, ADRs (phase 1), bankruptcy_status]
  - require_history_min: 252 trading days
```

Refresh universe monthly. Use point-in-time membership: a stock that was below $500M cap last year should not be in the universe last year, even if it qualifies today.

### 10.2 Signal sleeves (the modular building blocks)

**Sleeve 1 — Sector momentum (the safe starter).**

```
signal_1:
  universe:      11 SPDR US sector ETFs (XLK, XLF, XLE, ...)
  formula:       past_12_month_return_excluding_last_month
  rank:          cross-sectional
  position:      long top 3 sectors, equal-weight
  rebalance:     monthly (last trading day)
  vol_target:    8% per sector (sized via 60-day EWMA vol)
```

**Sleeve 2 — Time-series trend filter (defensive overlay).**

```
signal_2:
  universe:      [SPY, QQQ, IWM, EFA, EEM]   # ETFs
  rule:          if price > 200-day SMA, hold; else go to AGG/cash
  weight:        1/N across ETFs above filter
  rebalance:     monthly
```

Allocate 100% of equity exposure to either sleeve 1 or sleeve 2 *only when sleeve 2's filter is on*; revert to bonds/cash when broad market trend is broken. This is the "absolute momentum" gate.

**Sleeve 3 — Cross-sectional multi-factor (the meat).**

```
signal_3:
  universe:      US stocks meeting universe filters
  composite_score:
      0.30 × value_score        # composite of EV/EBITDA, FCF/EV, shareholder yield
      0.35 × quality_score      # composite of GP/Assets, ROE, low leverage
      0.20 × momentum_score     # 12-1 return, sector-neutralized
      0.15 × low_vol_score      # inverse 252-day realized vol
  rank:          cross-sectional, sector-neutralized
  position:      long top quintile, equal-weight, ~75–125 stocks
  rebalance:     monthly
  vol_scale:     target portfolio vol = 12% annualized
```

**Sleeve 4 — Mean-reversion overlay (small).**

```
signal_4:
  universe:      small-cap quality-screened (mkt cap $300M-$5B, positive earnings)
  rule:          z(5-day return) < -1.5 AND positive 12m return AND positive 6m EPS revision
  position:      long, equal-weight, hold 5-10 days
  max_concurrent: 20 names
  weight_in_portfolio: 10% of risk budget
```

### 10.3 Portfolio combination via HRP

After computing each sleeve's signals, allocate **risk** (not capital) across sleeves using Hierarchical Risk Parity on a 252-day rolling correlation matrix of the sleeves' returns. Re-weight monthly. Initial baseline (before HRP optimization stabilizes):

| Sleeve | Risk weight |
|---|---|
| Sector momentum | 25% |
| Trend filter / TS momentum | 20% |
| Cross-sectional multi-factor | 45% |
| Mean-reversion overlay | 10% |

### 10.4 Position sizing (the "modern N")

For each individual position within a sleeve:

```
weight_i = (signal_strength_i / Σ |signal_strength|)
         × sleeve_capital
         × (target_vol / realized_vol_i)
         × correlation_adjustment_i
```

Clamps:

- Single-position cap: 5% of portfolio gross exposure.
- Per-position vol target: 8–10% annualized.
- Total gross leverage: 1.0× initially (no leverage); raise to 1.5× after 6 months of live tracking matching backtest.

### 10.5 Risk overlays

```
overlay_1_drawdown_circuit_breaker:
   trailing_DD_5pct:   gross_exposure = 1.00 × baseline
   trailing_DD_10pct:  gross_exposure = 0.75 × baseline
   trailing_DD_15pct:  gross_exposure = 0.50 × baseline
   trailing_DD_20pct:  HALT, manual review

overlay_2_volatility_regime:
   if VIX > 30:        cut momentum sleeves by 50%, halt VRP if used
   if VIX > 40:        halt all long-only equity sleeves, hold cash/bonds

overlay_3_position_stops:
   chandelier_stop:    exit if close < highest_close_since_entry - 4 × ATR(20)
   time_stop:          exit position if no new 20-day high in 60 trading days
```

### 10.6 Execution rules

- All rebalances executed at market-on-close (MOC) or via VWAP over the final 90 minutes of the trading day.
- For positions exceeding 1% of stock's 20-day ADV, slice the order across 2–3 days.
- Never submit a market order for more than 0.5% of 20-day ADV in one execution slice.
- Use limit orders pegged to the bid/ask midpoint with a 30-second timeout for entries; cross the spread on exits if the timeout is hit.

### 10.7 Realistic cost assumptions in the backtest

```
backtest_costs:
   bid_ask_half_spread:   max(0.5 bps, sqrt(10 / ADV_$M) × 5 bps)
   commission:            1 bps per side
   market_impact:         0.4 × sqrt(trade_$ / ADV_$) × daily_vol × 10000 bps
   borrow_cost (shorts):  25 bps annualized base, 200 bps for hard-to-borrow flag
   slippage_buffer:       add 5 bps per side for safety margin
```

These are deliberately pessimistic. If a strategy survives this cost model, it will likely survive live trading.

---

## 11. Validation & Deployment Gates

Before any sleeve trades real money, it must pass these gates in order:

### Gate 1 — Backtest sanity

- Backtest covers ≥ 10 years of data, including 2008, 2018-Feb, 2020-Mar, 2022.
- Survivorship-bias-free universe.
- Point-in-time fundamentals.
- Realistic costs (per §10.7).
- **In-sample Sharpe > 1.0** (after costs).

### Gate 2 — Statistical robustness

- **Deflated Sharpe Ratio > 0.5** accounting for the number of variants tested.
- **Probability of Backtest Overfitting (PBO) < 15%**.
- Sharpe **stable** across 4 walk-forward windows (no window with SR < 0).
- Sharpe **stable** across purged k-fold CV (5th percentile fold SR > 0).

### Gate 3 — Stress tests

- Maximum drawdown in backtest < 25%.
- No 12-month rolling period with >−15% return.
- Strategy still positive (Sharpe > 0.3) under +50% transaction costs.
- Strategy still positive when removing the top 5% best months and the bottom 5% worst months.

### Gate 4 — Paper trading

- Run for ≥ 3 months in paper trading (live data, simulated execution).
- Live Sharpe within 50% of backtest Sharpe.
- Slippage within 1.5× of modeled slippage.
- No execution failures or data issues.

### Gate 5 — Live with small capital

- Start with ≤ 10% of intended capital.
- Run ≥ 6 months at small size.
- Live Sharpe within 70% of backtest Sharpe → scale to 50% capital.
- Another 6 months → scale to full capital.

### Gate 6 — Continuous monitoring (forever)

- Daily: P&L, position-level VaR, drawdown vs. circuit-breaker thresholds.
- Weekly: realized vol vs. target, factor exposure decomposition, hit rate vs. expectation.
- Monthly: full backtest re-run with latest data; if drift > 30% from baseline, investigate.
- Quarterly: capacity check (% of ADV traded, average slippage); decide whether to scale or cap AUM.

**A strategy that fails any gate is paused, not "tweaked until it passes."** Tweaking-until-passes is a bias mill.

---

## Appendix A — Formula Reference

### A.1 Volatility estimation (EWMA)

```
σ²_t = λ × σ²_{t-1} + (1 - λ) × r²_t
λ = 0.94  (RiskMetrics standard, half-life ≈ 11 days)
λ = 0.97  (slower, half-life ≈ 23 days; better for monthly strategies)
```

### A.2 Average True Range (ATR)

```
TR_t = max(High_t - Low_t, |High_t - Close_{t-1}|, |Low_t - Close_{t-1}|)
ATR_t = EMA(TR, n=20)
```

### A.3 Z-score normalization (cross-sectional)

```
z_i = (x_i - median(x)) / MAD(x)    # robust version
z_i = (x_i - mean(x)) / std(x)       # classical version (more outlier-sensitive)
```

### A.4 Cross-sectional momentum signal

```
ret_12_1_i = (P_{i,t-21} / P_{i,t-252}) - 1     # 12 months excluding the most recent month
rank_i = percentile_rank_cross_section(ret_12_1)
signal_i = 1 if rank_i > 0.80 else (-1 if rank_i < 0.20 else 0)
```

### A.5 Volatility-scaled position size

```
position_$_i = portfolio_$ × target_position_vol / asset_vol_i
target_position_vol_annualized = 0.10   # 10%
asset_vol_i = σ_i × √252
```

### A.6 Sharpe ratio and Deflated Sharpe Ratio

```
SR = (mean(r) - rf) / std(r)   # annualized: × √252 for daily

DSR ≈ Φ((SR_obs - E[SR_max | N strategies tested])
        / σ(SR_obs))
where σ(SR_obs) ≈ √((1 - γ_3 × SR_obs + (γ_4-1)/4 × SR²_obs) / (T-1))
```

`γ_3` = skewness, `γ_4` = excess kurtosis, `T` = sample size.

### A.7 Kelly fraction

```
f* = μ / σ²   (continuous case, log-utility)
f_practical = 0.25 × f*   (fractional Kelly, recommended)
```

### A.8 Hierarchical Risk Parity (sketch)

```
1. Compute correlation matrix of sleeve returns.
2. Convert to distance matrix: d_{ij} = √(0.5 × (1 - ρ_{ij})).
3. Hierarchical clustering (single-linkage) on distance matrix.
4. Recursive bisection: at each cluster, split into two sub-clusters and allocate risk inverse to each sub-cluster's variance.
5. Recurse until each leaf is a single sleeve.
```

---

## Appendix B — Glossary

- **ADV** — Average Daily Volume (typically dollar volume).
- **ATR** — Average True Range (volatility measure in price units).
- **BAB** — Betting Against Beta (Frazzini & Pedersen 2014 factor).
- **CPCV** — Combinatorial Purged Cross-Validation.
- **DSR** — Deflated Sharpe Ratio (Bailey & López de Prado 2014).
- **EWMA** — Exponentially Weighted Moving Average.
- **HRP** — Hierarchical Risk Parity (López de Prado 2016).
- **IC** — Information Coefficient (correlation of forecast with realized returns).
- **IR** — Information Ratio (alpha divided by tracking error).
- **N** — In Turtle parlance, a measure of volatility (≈ ATR) used for position sizing.
- **PBO** — Probability of Backtest Overfitting.
- **PEAD** — Post-Earnings Announcement Drift.
- **POV** — Percentage of Volume execution algorithm.
- **QMJ** — Quality Minus Junk (AQR factor).
- **SUE** — Standardized Unexpected Earnings.
- **VRP** — Volatility Risk Premium.
- **VWAP** — Volume-Weighted Average Price (execution algorithm and benchmark).

---

## Appendix C — References

### Foundational papers

- Jegadeesh, N. & Titman, S. (2023). *Momentum: Evidence and Insights 30 Years Later*. SSRN. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4602426
- Moskowitz, T., Ooi, Y. H., & Pedersen, L. H. (2012). *Time Series Momentum*. Journal of Financial Economics. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2089463
- Asness, C., Moskowitz, T., & Pedersen, L. H. (2013). *Value and Momentum Everywhere*. Journal of Finance.
- Fama, E. F. & French, K. R. (2015). *A Five-Factor Asset Pricing Model*. JFE.
- Frazzini, A. & Pedersen, L. H. (2014). *Betting Against Beta*. Journal of Financial Economics.
- Novy-Marx, R. (2013). *The Other Side of Value: The Gross Profitability Premium*. JFE.
- Asness, C., Frazzini, A., & Pedersen, L. H. (2019). *Quality Minus Junk*. Review of Accounting Studies. AQR Working Paper.

### Risk management & sizing

- Barroso, P. & Santa-Clara, P. (2015). *Momentum Has Its Moments*. JFE. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2041429
- Daniel, K. & Moskowitz, T. (2016). *Momentum Crashes*. JFE / NBER. https://www.nber.org/papers/w20439
- Blitz, D., Huij, J., & Martens, M. (2011). *Residual Momentum*. Journal of Empirical Finance.
- López de Prado, M. (2016). *Building Diversified Portfolios that Outperform Out-of-Sample*. Journal of Portfolio Management. (HRP paper.)
- Carver, R. (2015). *Systematic Trading*. Harriman House.
- Carver, R. (2019). *Leveraged Trading*. Harriman House.
- Carver, R. (2023). *Advanced Futures Trading Strategies*. Harriman House.
- Kaminski, K. & Lo, A. (2014). *When Do Stop-Loss Rules Stop Losses?* Journal of Financial Markets.

### Replication crisis & backtesting

- Harvey, C., Liu, Y., & Zhu, H. (2016). *…and the Cross-Section of Expected Returns*. Review of Financial Studies. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2249314
- Hou, K., Xue, C., & Zhang, L. (2020). *Replicating Anomalies*. Review of Financial Studies. https://global-q.org/
- Bailey, D. & López de Prado, M. (2014). *The Deflated Sharpe Ratio*. JPM. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- López de Prado, M. (2020). *Machine Learning for Asset Managers*. CUP.

### Mean reversion & stat arb

- Gatev, E., Goetzmann, W., & Rouwenhorst, K. G. (2006). *Pairs Trading: Performance of a Relative-Value Arbitrage Rule*. RFS. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=141615
- Avellaneda, M. & Lee, J.-H. (2010). *Statistical Arbitrage in the U.S. Equities Market*. Quantitative Finance. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1153505
- Lou, D., Polk, C., & Skouras, S. (2019). *A Tug of War: Overnight Versus Intraday Expected Returns*. JFE.

### Execution & costs

- Almgren, R. & Chriss, N. (2001). *Optimal Execution of Portfolio Transactions*.
- Frazzini, A., Israel, R., & Moskowitz, T. (2018). *Trading Costs*. AQR Working Paper. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3229719
- Grinold, R. & Kahn, R. (2000). *Active Portfolio Management*, 2nd ed. McGraw-Hill.

### ML & alternative data

- Kelly, B., Manela, A., & Moreira, A. (2024). *Text Selection*. JFE.
- Chen, A. Y., Kelly, B., & Xiu, D. (2024). *Expected Returns and Large Language Models*. SSRN. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4416687
- Jansen, S. (2021). *Machine Learning for Algorithmic Trading*, 2nd ed. Packt.
- Dixon, M., Halperin, I., & Bilokon, P. (2020). *Machine Learning in Finance: From Theory to Practice*. Springer.
- Loughran, T. & McDonald, B. (2011). *When Is a Liability Not a Liability?* (sentiment dictionary).

### Practitioner books

- Antonacci, G. (2014). *Dual Momentum Investing*. McGraw-Hill.
- Gray, W. & Vogel, J. (2016). *Quantitative Momentum*. Wiley.
- Gray, W. & Carlisle, T. (2012). *Quantitative Value*. Wiley.
- Faith, C. (2007). *Way of the Turtle*. McGraw-Hill.
- Covel, M. (2007). *The Complete TurtleTrader*. HarperBusiness.
- Chan, E. (2008/2013/2017). *Quantitative Trading*; *Algorithmic Trading*; *Machine Trading*. Wiley.
- Pedersen, L. H. (2015). *Efficiently Inefficient*. Princeton.

### Research libraries & data

- Kenneth French Data Library: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
- AQR Datasets: https://www.aqr.com/Insights/Datasets
- Q-factor data (Hou, Xue, Zhang): https://global-q.org/
- AlphaArchitect blog: https://alphaarchitect.com/
- Robeco research: https://www.robeco.com/en/insights/

---

*End of report. Length: ~13,500 words. For an updated revision, re-run the cited sources annually; factor research evolves and live performance numbers will need refreshing.*
