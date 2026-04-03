# Pairs Trading — Educational Guide

A plain-English explanation of every concept used in this project, from the statistics to the code. No prior quant finance knowledge assumed.

---

## Table of Contents

1. [What is Pairs Trading?](#1-what-is-pairs-trading)
2. [How is it Different from Arbitrage?](#2-how-is-it-different-from-arbitrage)
3. [Cointegration — The Core Idea](#3-cointegration--the-core-idea)
4. [The Engle-Granger Test](#4-the-engle-granger-test)
5. [The Hedge Ratio](#5-the-hedge-ratio)
6. [The Spread](#6-the-spread)
7. [Half-Life of Mean Reversion](#7-half-life-of-mean-reversion)
8. [Z-Score and Entry/Exit Signals](#8-z-score-and-entryexit-signals)
9. [Position Sizing and Dollar Neutrality](#9-position-sizing-and-dollar-neutrality)
10. [The Backtest — Walk-Forward Method](#10-the-backtest--walk-forward-method)
11. [Commission Drag](#11-commission-drag)
12. [Performance Metrics Explained](#12-performance-metrics-explained)
13. [Why These Pairs? Choosing Candidates](#13-why-these-pairs-choosing-candidates)
14. [Risks and Failure Modes](#14-risks-and-failure-modes)
15. [Glossary](#15-glossary)

---

## 1. What is Pairs Trading?

Pairs trading is a **market-neutral strategy** — it tries to profit regardless of whether the overall market goes up or down.

The core idea: find two assets whose prices tend to move together over time. When they temporarily diverge, bet that they will converge back. You do this by **buying the underperformer and selling the overperformer simultaneously**.

**A simple example:**

Imagine two coffee shops on the same street — Café A and Café B. Their prices usually differ by around £0.20 for the same latte. One day, Café A suddenly charges £0.80 more than Café B. You would expect them to revert to the usual £0.20 gap eventually. Pairs trading is betting on exactly that reversion.

In crypto terms: if RENDER and FET (both AI tokens) usually trade within a tight ratio, and suddenly RENDER shoots up while FET lags, you short RENDER and buy FET, expecting the gap to close.

**Why it is called "market-neutral":**

You hold one long position and one short position simultaneously. If the whole crypto market crashes 10%, both tokens fall ~10% — your long loses and your short gains roughly the same amount. The market direction cancels out. You only profit (or lose) on the *relative movement* between the two tokens.

---

## 2. How is it Different from Arbitrage?

People often call pairs trading "statistical arbitrage" but the two concepts are meaningfully different:

| | True Arbitrage | Pairs Trading |
|---|---|---|
| **Profit guaranteed?** | Yes — by definition | No — it's a probability |
| **Example** | BTC is $100 cheaper on Kraken than Binance — buy one, sell the other, guaranteed profit | ETH is high relative to BTC — bet it reverts, but it might not |
| **Speed required** | Milliseconds (bots close gaps instantly) | Hours to days |
| **Risk** | Near zero (if executed fast enough) | Real — the spread can keep widening |

True arbitrage requires being faster than every other participant. Pairs trading requires being *right about statistics* — a much more achievable edge for a retail trader.

---

## 3. Cointegration — The Core Idea

**Correlation vs. Cointegration — a crucial distinction:**

Two assets can be highly correlated (move up and down together) without being cointegrated. Cointegration is a stronger condition: it means their prices are *bound together in the long run* — they cannot drift apart indefinitely.

**An analogy — the drunk and her dog:**

A drunk person walks home with her dog on a lead. Both are wandering randomly (correlation — they move in similar directions), but the lead means they cannot stray too far from each other (cointegration — the gap is bounded). If you only saw the dog, you could predict roughly where the owner is.

Mathematically: two price series `P1` and `P2` are cointegrated if there exists some constant `β` such that:

```
P1 - β × P2 = spread
```

...and that `spread` is stationary — it has a constant mean and variance over time, rather than drifting without bound.

**Why does this happen in crypto?**

Tokens in the same sector share the same pool of investors, the same macro narrative, and often the same underlying demand drivers. When institutional money rotates into "AI tokens", it tends to buy RENDER and FET together. When it rotates out, both fall. This creates the binding relationship that cointegration tests detect.

---

## 4. The Engle-Granger Test

The Engle-Granger test is the standard statistical test for cointegration between two assets. It was developed by economists Robert Engle and Clive Granger (who won the Nobel Prize in Economics in 2003 partly for this work).

**The two steps:**

**Step 1 — OLS regression:**
Fit a linear regression of `log(price_base)` on `log(price_quote)`:

```
log(P_base) = α + β × log(P_quote) + ε
```

This gives you:
- `α` (intercept) — the long-run level difference between the two assets
- `β` (hedge ratio) — how much the base price moves per unit of quote price movement
- `ε` (residuals) — the spread series

**Step 2 — ADF test on residuals:**
Run an Augmented Dickey-Fuller (ADF) test on the residuals `ε`. The ADF test checks whether a time series is stationary (mean-reverting) or has a "unit root" (random walk, no tendency to revert).

**Reading the p-value:**
- p < 0.05: strong evidence of cointegration (less than 5% chance the relationship is random)
- p < 0.10: moderate evidence (our threshold — acceptable for exploratory trading)
- p > 0.10: insufficient evidence — do not trade this pair

**Why we use log prices:**

Log prices (`log(P)`) are used instead of raw prices for two reasons:
1. They convert multiplicative relationships to additive ones (a 10% move is the same size regardless of whether the price is $1 or $1000)
2. They stabilise variance across different price levels

**In the code:** `cointegration.py → test_cointegration()`

---

## 5. The Hedge Ratio

The hedge ratio `β` tells you how many units of the quote asset to hold per unit of the base asset to achieve dollar neutrality.

**Example:**

If `β = 0.925` for RENDER/FET:
- You buy £800 worth of RENDER (the long leg)
- You sell `0.925 × £800 = £740` worth of FET (the short leg)

The positions are not equal in dollar value, but they are balanced in *spread space* — the OLS regression says this ratio minimises your exposure to the common trend.

**Why not just trade £800 vs £800?**

If you traded equal dollar amounts, you would be implicitly betting that the two assets have a 1:1 price relationship. The hedge ratio corrects for the fact that RENDER and FET have different volatilities and price histories. Using the wrong ratio means your "neutral" position actually has a directional bias.

**Refitting the hedge ratio:**

The relationship between two assets drifts over time. In the paper trader, the hedge ratio is refit every 24 hours using the latest 2000 candles. This prevents the position sizing from becoming stale.

**In the code:** `cointegration.py → test_cointegration()` returns `hedge_ratio`

---

## 6. The Spread

The spread is the core signal in pairs trading. It is defined as:

```
spread = log(P_base) - β × log(P_quote) - α
```

Where:
- `P_base` is the price of the base asset (e.g. RENDER)
- `P_quote` is the price of the quote asset (e.g. FET)
- `β` is the hedge ratio
- `α` is the intercept

**Interpretation:**

- `spread > 0`: base is expensive relative to quote → short the spread (sell base, buy quote)
- `spread < 0`: base is cheap relative to quote → long the spread (buy base, sell quote)
- `spread ≈ 0`: fairly priced — no edge

The spread should fluctuate around zero and revert back when it deviates — that is what the cointegration test verified.

**In the code:** `cointegration.py → compute_spread()`

---

## 7. Half-Life of Mean Reversion

The half-life tells you how quickly the spread is expected to revert back toward its mean — in other words, **how long you should expect to hold the position**.

**How it is calculated:**

Fit an AR(1) model on the spread:

```
Δspread_t = φ × spread_{t-1} + ε
```

If `φ < 0`, the spread is mean-reverting (a high spread yesterday predicts a lower spread today). The half-life is:

```
half-life = -log(2) / log(1 + φ)
```

**Example:**

- Half-life = 23h (RENDER/FET): a spread deviation is expected to decay by 50% within 23 hours → typical holding time is 1–2 days ✓
- Half-life = 400h: takes 16 days to decay by 50% → capital tied up too long, too many things can go wrong ✗
- Half-life = 1h: reverts within an hour → by the time you detect the signal and act, it has already closed ✗

**Our filter:** half-life must be between 4 and 120 hours. Anything outside this range is rejected.

**In the code:** `cointegration.py → _compute_half_life()`

---

## 8. Z-Score and Entry/Exit Signals

The raw spread has different units and scale depending on the pair. To compare spreads consistently and set universal thresholds, we normalise it to a **z-score**:

```
z = (spread - rolling_mean) / rolling_std
```

Where:
- `rolling_mean` = average spread over the last 120 candles (5 days)
- `rolling_std` = standard deviation of spread over the last 120 candles

**The z-score measures how many standard deviations the current spread is from its recent average.** A z-score of ±2 means the spread is unusually extreme — 2 standard deviations from normal.

**Signal logic:**

```
z > +1.6   →  SHORT SPREAD  (base too expensive vs quote — sell base, buy quote)
z < -1.6   →  LONG SPREAD   (base too cheap vs quote — buy base, sell quote)
|z| < 0.3  →  EXIT          (spread has reverted to near-normal)
|z| > 3.0  →  STOP LOSS     (spread widened further — cut the loss)
```

**Why these numbers?**

- **Entry at ±1.6**: In a normal distribution, 1.6 standard deviations covers ~89% of values — so we enter when the spread is in the rarest ~11% of observations. Lower = more trades but noisier. Higher = fewer but cleaner signals.
- **Exit at ±0.3**: We exit well before the spread fully reverts to zero to avoid giving back profits on the last bit of movement.
- **Stop at ±3.0**: If the spread reaches 3 standard deviations, something unusual is happening (potential regime shift). Cut the loss rather than average down.

**In the code:** `spread_tracker.py → SpreadTracker.update()` and `_classify_signal()`

---

## 9. Position Sizing and Dollar Neutrality

**Goal:** make the two legs as equal and opposite as possible so that market-wide moves cancel out.

**Step 1 — Size the base leg:**
```
base_qty = (bankroll × 0.80) / price_base
```
We allocate 80% of the bankroll to the trade. The remainder provides a buffer for commissions and unrealised losses.

**Step 2 — Size the quote leg using the hedge ratio:**
```
quote_qty = base_qty × hedge_ratio × (price_base / price_quote)
```

This ensures that in *log-price space* (where cointegration was measured), the two legs are balanced.

**Example with RENDER/FET:**

- Bankroll: $1,000 → $800 deployed
- RENDER price: $5.00 → buy 160 RENDER (base long)
- FET price: $1.50, hedge ratio = 0.925
- quote_qty = 160 × 0.925 × (5.00 / 1.50) = 493 FET
- Dollar value of FET leg: 493 × $1.50 = $740 (short)

The legs are not equal in dollar terms ($800 vs $740) because the hedge ratio is 0.925, not 1.0.

**In the code:** `paper_trader.py → PaperPortfolio.open_position()`

---

## 10. The Backtest — Walk-Forward Method

A backtest simulates the strategy on historical data to evaluate how it would have performed. The key challenge is avoiding **look-ahead bias** — accidentally using future information when making past decisions.

**Our method: Walk-Forward (In-Sample / Out-of-Sample Split)**

```
|─────────────────────────────────────────────────────────|
|     IN-SAMPLE (60%)          |  OUT-OF-SAMPLE (40%)     |
|  Fit hedge ratio here        |  Simulate trading here   |
|  (2000 × 0.60 = 1200h)       |  (2000 × 0.40 = 800h)    |
|─────────────────────────────────────────────────────────|
```

1. **In-sample period**: Run the cointegration test and OLS regression. Save the hedge ratio, intercept, and spread statistics.
2. **Out-of-sample period**: Simulate trading using *only* the parameters fitted on in-sample data. This mimics what would have happened in real trading.

**Why split the data?**

If you fit the hedge ratio *and* evaluate performance on the same data, you are cheating — the parameters are perfectly tuned for that specific period. Out-of-sample testing forces the strategy to prove itself on data it has never seen.

**Seeding the tracker:**

At the start of the out-of-sample period, the rolling z-score window needs historical values to work. We "seed" it with the last 120 spread values from the in-sample period — this is legitimate because those values would have been available at the time.

**In the code:** `backtester.py → run_backtest()`

---

## 11. Commission Drag

Every trade has a cost. On Binance, the taker fee is **0.10% per side**. Since each pairs trade involves:
- Opening: buy base + sell quote = 2 legs
- Closing: sell base + buy quote = 2 legs

Total cost = **4 × 0.10% = 0.40%** per round-trip trade.

This means a trade must produce more than 0.40% profit *just to break even*. For a $1,000 position, you are paying $4 in commissions per trade regardless of outcome.

**Why this matters:**

A strategy with a 60% win rate but an average win of 0.30% and average loss of 0.40% loses money — even though it wins more often than it loses. The average win must comfortably exceed the commission per trade.

For RENDER/FET, the average winning trade in the backtest was ~+0.60%, well above the 0.40% commission floor.

**In the code:** `config.py → commission = 0.001` (0.10%)

---

## 12. Performance Metrics Explained

### Win Rate
The percentage of closed trades that were profitable.

```
win_rate = winning_trades / total_trades × 100
```

A win rate of 65% sounds good, but it means nothing without knowing the average size of wins vs. losses.

### Profit Factor
The ratio of total gross profit to total gross loss.

```
profit_factor = (avg_win × n_wins) / (avg_loss × n_losses)
```

- Profit factor > 1.5: solid strategy
- Profit factor > 2.0: excellent
- Profit factor < 1.0: losing money despite any win rate

### Sharpe Ratio
Measures return per unit of risk (volatility). The gold standard for evaluating strategy quality.

```
sharpe = mean(trade_returns) / std(trade_returns) × sqrt(252)
```

The `sqrt(252)` annualises the ratio (252 trading days per year).

- Sharpe > 1.0: acceptable
- Sharpe > 2.0: good
- Sharpe > 3.0: excellent (rare)

RENDER/FET achieved **Sharpe = 3.36** in the backtest — which is exceptional. It means returns were large relative to their variability.

### Maximum Drawdown
The largest peak-to-trough decline in the equity curve. Measures the worst-case loss you would have experienced if you had entered at the peak.

```
max_drawdown = max((peak - trough) / peak × 100)
```

A 10% max drawdown means at some point during the backtest, your $1,000 fell to $900 before recovering.

### Average Holding Time
The mean number of hours a position was open. Useful for checking that the half-life estimate is realistic.

RENDER/FET average hold: **20.6 hours** — consistent with the 23-hour half-life estimated by the AR(1) model.

---

## 13. Why These Pairs? Choosing Candidates

Good pairs candidates share **economic reasons to stay cointegrated**, not just historical correlation.

**What works:**

| Category | Example | Why |
|---|---|---|
| Same micro-niche | RENDER + FET | Both AI compute infrastructure — same buyers, same narrative |
| Same protocol category | ONDO + POLYX | Both RWA tokenisation — institutional capital treats them interchangeably |
| Same ecosystem | ATOM + OSMO | Cosmos IBC — protocol-level dependency |

**What does not work (as of April 2026):**

| Category | Example | Why it failed |
|---|---|---|
| Cross-sector majors | ETH/BTC | ETH has structurally underperformed BTC since 2024 — regime shift |
| Competing L1s | SOL/AVAX | Different roadmaps, different exchange listings, different user bases |
| Meme tokens | DOGE/SHIB | Both cointegrated historically but regime changed — -6% OOS |
| Privacy coins | XMR/ZEC | Different regulatory exposure (Binance delisted XMR) |

**The key question to ask:**

> "If I told you nothing about the price, and only told you that TOKEN A went up 20% today, what is the probability that TOKEN B also went up significantly?"

If the answer is "very high, because they have the same investors and narrative", the pair is a candidate. If the answer is "no idea", skip it.

---

## 14. Risks and Failure Modes

### Regime Shift
The most common failure mode. A pair that was cointegrated for 3 years can stop being cointegrated in 3 weeks if:
- One token gets a major protocol-specific event (hack, upgrade, partnership)
- The sector narrative changes (AI hype cools, RWA loses institutional interest)
- One token gets delisted from a major exchange

**How to detect it:** run `python3 run_backtest.py` weekly. If the p-value rises above 0.10 or the OOS P&L turns consistently negative, the regime has changed — stop trading the pair.

### Thin Liquidity
ONDO and POLYX have much lower trading volumes than BTC or ETH. In live trading (not paper trading), a $1,000 order could move the price against you. The 0.10% commission assumption may underestimate real slippage.

**Rule of thumb:** if your position size is more than 0.5% of the 24-hour volume, expect meaningful slippage.

### Hedge Ratio Drift
The OLS hedge ratio is refitted every 24 hours, but a fast market move can make the previous ratio stale within hours. During high-volatility periods, the actual spread may differ significantly from the modelled spread.

### The Overfitting Trap
Running the backtest on many candidate pairs increases the chance that one appears to work *by chance*. This is called the multiple comparisons problem. After testing 24 pairs, you would expect 1–2 to show positive results purely by luck at the 5% significance level.

**Our protection:** out-of-sample testing on 40% of data, requiring at least 3 trades and a positive Sharpe ratio. A genuinely good pair should also have an economic reason to be cointegrated — not just a good p-value.

---

## 15. Glossary

| Term | Definition |
|---|---|
| **ADF test** | Augmented Dickey-Fuller test — checks if a time series is stationary (mean-reverting) or a random walk |
| **AR(1)** | Autoregressive model of order 1 — today's value depends linearly on yesterday's value plus noise |
| **Cointegration** | Two price series are cointegrated if a linear combination of them is stationary |
| **Commission drag** | The cumulative cost of trading fees eroding returns |
| **Dollar-neutral** | A position where the long and short legs have equal dollar exposure, cancelling market-wide moves |
| **Engle-Granger** | Two-step cointegration test: OLS regression followed by ADF test on residuals |
| **Half-life** | Time for a spread deviation to decay by 50%; derived from an AR(1) fit |
| **Hedge ratio (β)** | The OLS coefficient relating log(base) to log(quote); determines how many units of the quote to trade per unit of base |
| **In-sample** | The portion of historical data used to fit model parameters |
| **Log price** | Natural logarithm of price; converts multiplicative returns to additive, stabilises variance |
| **Market-neutral** | A strategy that profits from relative movements rather than overall market direction |
| **Mean reversion** | The tendency of a spread to return toward its long-run average after deviating |
| **OLS** | Ordinary Least Squares — the standard linear regression method minimising squared residuals |
| **Out-of-sample** | The portion of historical data held back for testing — the model never sees this during fitting |
| **Profit factor** | Gross profit divided by gross loss; > 1.0 means the strategy makes money |
| **Regime shift** | A structural change in the market that breaks a previously stable cointegration relationship |
| **Sharpe ratio** | Annualised return divided by annualised volatility of returns; measures reward per unit of risk |
| **Spread** | `log(P_base) - β × log(P_quote) - α`; the signal series that should be stationary |
| **Stationary** | A time series with constant mean, variance, and autocorrelation over time — does not drift |
| **Unit root** | A property of non-stationary series (random walks) — past shocks have permanent effects |
| **Walk-forward** | Backtest methodology that fits parameters on past data and tests on unseen future data |
| **Z-score** | `(value - rolling_mean) / rolling_std` — number of standard deviations from the rolling average |
