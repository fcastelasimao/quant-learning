# A Mathematician's Guide to qframe

*You know linear algebra, probability, and statistics well. You don't know finance. This document explains everything you need to understand what this project is doing, why, and what the numbers mean.*

---

## 1. The Core Problem

We have **daily closing prices** for ~449 individual company stocks (equities) over ~15 years. We want to find a function

```
f(price history up to day t) → score per stock on day t
```

such that stocks with **high scores today tend to outperform** over the next 1–63 days.

This is a supervised learning problem, but with a brutal constraint: **you can only use information available at the time of prediction**. No future data allowed, ever. This is called *no look-ahead bias* and it is the single most important correctness requirement.

---

## 2. What Are We Trading?

**Individual company stocks (equities).** Not ETFs, not options, not futures.

- A **stock** represents partial ownership of a company. When the company does well, the stock price rises; when it does poorly, the price falls.
- The **S&P 500** is an index of the 500 largest US companies by market capitalisation. We use it as our universe because these are the most liquid stocks — easy and cheap to buy and sell in large quantities.
- We are NOT trading the S&P 500 index itself (which you would do via an ETF like SPY). We are trading individual stocks within it, going **long** (buying) stocks we predict will go up and **short** (borrowing-and-selling) stocks we predict will go down.
- This is called a **long-short equity strategy**: we are approximately market-neutral (our profits don't depend on whether the overall market goes up or down — only on whether our *ranking* of stocks is correct).

**Why not ETFs or indices?** Because the signal (IC) measures *relative* stock performance. If all 500 stocks go up by 1% on a day, a factor with IC = 0.03 tells us which ones went up *more* than the others — that's the edge we're capturing. An ETF gives you the average, which has no cross-sectional signal.

---

## 3. The Setup

### Stocks and prices

Think of the price data as a matrix:

- **Rows:** trading days (≈252 per year; markets are closed on weekends and holidays)
- **Columns:** stock tickers (AAPL, MSFT, ...)
- **Values:** closing price on that day

We compute daily **returns** as:

```
r[t, stock] = (price[t] - price[t-1]) / price[t-1]
```

So returns are approximately the percentage change in price each day. Most days, most stocks have returns between −5% and +5%.

### The factor

A *factor* is just a function that assigns a score to every stock on every day, computed from historical price data:

```
factor_score[t, stock] = f(prices up to and including day t)
```

Examples:
- **Momentum**: score = return over the past 12 months. The hypothesis: stocks that went up recently keep going up.
- **Mean reversion**: score = (current price − long-run average) / standard deviation, negated. The hypothesis: stocks far below their average will tend to revert upward.

---

## 4. Spearman Rank Correlation — The Foundation

Before explaining IC, we need to understand the statistical tool it's built on.

### Pearson vs Spearman: why rank correlation?

The **Pearson correlation** measures linear association between two variables *x* and *y*:

```
ρ_Pearson(x, y) = Σ (xᵢ - x̄)(yᵢ - ȳ) / [√Σ(xᵢ-x̄)² · √Σ(yᵢ-ȳ)²]
```

This is sensitive to outliers: if one stock has an extreme return (+50% due to an acquisition), it dominates the whole calculation.

The **Spearman rank correlation** first converts both variables to their *ranks*, then applies Pearson on those ranks:

```
For n stocks, with factor scores x₁,...,xₙ and returns y₁,...,yₙ:

Step 1.  rᵢ = rank of xᵢ among {x₁,...,xₙ}   (1 = lowest score, n = highest)
         sᵢ = rank of yᵢ among {y₁,...,yₙ}   (1 = worst return, n = best)

Step 2.  dᵢ = rᵢ - sᵢ   (difference in ranks)

Step 3.  ρ_Spearman = 1 - 6·Σdᵢ² / [n·(n²-1)]
```

Or equivalently: ρ_Spearman = Pearson(ranks(x), ranks(y)). This is exactly what `scipy.stats.spearmanr` computes.

**Why ranks?** Because we only care about *ordering*, not magnitude. A stock that returned +50% is just "rank 449 (best)" — same as a stock that returned +2% if that happened to be the highest return on a quiet day. This makes the metric stable, bounded in [−1, +1], and robust to extreme values.

### Example with 5 stocks

| Stock | Factor score | Factor rank | Next-day return | Return rank | dᵢ | dᵢ² |
|-------|-------------|-------------|-----------------|------------|-----|------|
| AAPL  | 0.8         | 5 (highest) | +1.2%           | 4          | +1  | 1    |
| MSFT  | 0.6         | 4           | +1.5%           | 5 (best)   | −1  | 1    |
| NVDA  | 0.3         | 3           | +0.2%           | 3          | 0   | 0    |
| AMZN  | −0.1        | 2           | −0.5%           | 2          | 0   | 0    |
| META  | −0.4        | 1 (lowest)  | −1.0%           | 1 (worst)  | 0   | 0    |

Σdᵢ² = 1+1+0+0+0 = 2

ρ = 1 − 6×2 / [5×(25−1)] = 1 − 12/120 = **+0.90** — strong positive signal (high-ranked stocks mostly returned more).

---

## 5. The Key Metric: IC (Information Coefficient)

### What it measures

On each trading day *t*, we have:
1. The factor scores for all stocks (our prediction)
2. The actual returns *h* days later (the truth)

The **Information Coefficient** is the **Spearman rank correlation** between these two:

```
IC(t) = Spearman_corr(factor_scores[t, :], returns[t+h, :])
```

It measures whether stocks we *ranked* high actually *returned* more. Using rank correlation makes it robust to outliers and doesn't assume linearity.

### What IC values mean

| IC | Interpretation |
|----|---------------|
| 0.00 | Completely random — factor has no predictive power |
| 0.01 | Very weak signal — barely above noise |
| **0.015** | **Weak gate threshold** — worth investigating |
| 0.03 | Decent signal — typical for strong academic factors |
| **0.030** | **Pass gate threshold** — consider trading |
| 0.05–0.08 | Excellent — top quant fund territory |
| > 0.10 | Almost certainly a data error or look-ahead bias |

Intuition: IC = 0.03 means that on average, if you rank all 500 stocks by the factor, stocks in the top half outperform stocks in the bottom half by a small but consistent amount. Over thousands of trading days, this compounds to meaningful profit.

### Why such small numbers?

Stock returns are dominated by market-wide noise (the "market return"). Any single factor explains only a tiny fraction of the cross-sectional variance in returns. The key insight is that with daily data across 500 stocks, even IC = 0.015 gives statistical significance over years of OOS data — and it compounds.

---

## 6. ICIR (Information Coefficient Information Ratio)

IC varies day to day. On some days the factor predicts well, on others it's noise. We want a signal that is **consistently** predictive, not just occasionally lucky.

```
ICIR = mean(IC series) / std(IC series)
```

This is exactly the Sharpe ratio applied to the IC time series. It answers: "is this factor reliable, or does it only work sometimes?"

| ICIR | Interpretation |
|------|---------------|
| < 0.15 | Unreliable — even if IC > 0 on average, it's too noisy |
| **0.15** | **Weak gate** |
| **0.40** | **Pass gate** |
| > 1.0 | Very consistent — rare |

**Both IC ≥ 0.015 and ICIR ≥ 0.15 must hold** to pass the weak gate. A factor with high average IC but ICIR = 0.1 is dangerous — it probably passed in-sample by chance.

---

## 7. Slow ICIR (The Honest Metric for Slow Signals)

Standard ICIR is computed on daily IC values. But if a signal only realises over 63 days (e.g., a mean-reversion factor), then consecutive daily ICs are *massively autocorrelated* (yesterday's IC is 62/63 correlated with today's IC, since they share 62 of the same forward-return days). Using autocorrelated samples dramatically inflates the apparent significance.

The fix: use **non-overlapping windows**.

```python
# Instead of IC computed daily:
compute_slow_icir(factor_df, returns_df, horizon=63, oos_start='2018-01-01')
# → Uses non-overlapping 63-day periods. ~25 independent observations per year.
# → Returns: mean_IC_over_these_periods / std_IC_over_these_periods
```

Mathematically this is correct. The daily ICIR was using ~1500 correlated samples and claiming they were independent. `slow_icir_63` uses ~175 truly independent samples.

**Slow ICIR gate thresholds:**
- Weak gate: slow_icir_63 ≥ 0.10
- Pass gate: slow_icir_63 ≥ 0.25

---

## 8. IC Decay

For each factor, we compute IC at multiple forward horizons h = 1, 2, ..., 63 days:

```
IC_decay[h] = mean over all days t of: Spearman_corr(factor[t], returns[t+h])
```

This reveals the **timescale** at which the signal operates:

| IC decay shape | Signal type | Interpretation |
|---------------|-------------|---------------|
| High at h=1, decays to 0 by h=21 | Fast momentum | Trade daily, costs hurt a lot |
| Roughly flat 1–63 days | Persistent factor | Rebalance monthly, costs manageable |
| Low at h=1, peaks at h=63 | Slow mean reversion | Rebalance quarterly, costs negligible |

The `price_level_autocorrelation` factor in this project has IC ≈ 0.002 at h=1 and IC ≈ 0.049 at h=63. **Evaluating it at a 1-day horizon and calling it "no signal" was a measurement error, not a true failure.**

---

## 9. Transaction Costs (Why High IC Isn't Enough)

Every trade costs money. The main components:

### Bid-ask spread
Market makers quote a buy price above and a sell price below the "true" price. The difference is the **spread**. Round-trip cost (buy then sell) ≈ 10–25 bps (1 bp = 0.01%).

### Market impact (Almgren-Chriss model)
Large trades move the price against you. If you trade a fraction *q* of the stock's average daily volume:

```
impact_cost = γ × q^η   (default: γ=30, η=0.6)
```

This is a power law — trading 10× more costs only ~4× more, not 10×.

### Short borrow
To *short* a stock (profit from it going down), you must borrow shares and pay a lending fee: typically ~50 bps/year for S&P 500 stocks, but up to 2000 bps/year for hard-to-borrow names.

### Net IC
```
net_ic = gross_ic − trading_cost_drag − borrow_cost_drag − funding_drag
```

A factor with IC = 0.020 but daily rebalancing at 10% turnover loses ~680 bps/year to costs — the gross IC is meaningless. Net IC is the only metric that matters for actual trading.

**Key numbers:**
| Rebalancing | Turnover | Trading drag/year | Net IC survives? |
|-------------|----------|------------------|-----------------|
| Daily | 5% | ~341 bps | Only if gross IC is strong (>0.025) |
| Monthly | 5% | ~40 bps | Yes for IC ≥ 0.015 |
| Quarterly | 1.4% | ~10 bps | Yes for IC ≥ 0.010 |

---

## 10. Multiple Testing: Why IC ≥ 0.015 Is Not Enough on Its Own

### The problem of testing many factors

Suppose you test 30 different factor formulas. Even if **none of them have any real predictive power**, some will appear to have IC > 0.015 by pure chance. This is the multiple testing problem.

Formally: if each test has a 5% chance of a false positive (p-value < 0.05), and you run 30 independent tests, the probability of at least one false positive is:

```
P(at least one false positive) = 1 - (1-0.05)^30 ≈ 0.785
```

That is, there's a **78% chance** you find something that looks real but is noise. With 30 tests we'd expect about 1–2 false positives even if nothing works.

### The t-statistic

Before applying any correction, we convert IC into a t-statistic — a standardised measure of how many standard errors above zero the mean IC is:

**For fast signals (h=1 day):**
```
t = (mean_IC / std(IC_daily)) × √N_OOS_days
  = IC_Sharpe_annualised / √252 × √N_OOS_days
```
With N = 1760 OOS days: `t ≈ sharpe × 2.64`

**For slow signals (h=63 days, non-overlapping):**
```
t = slow_icir_63 × √N_windows    where N_windows = floor(N_OOS_days / 63)
```
With N_windows = 27: `t ≈ slow_icir_63 × 5.2`

The slow formula uses fewer but truly *independent* observations. The daily formula would give a spuriously large t for slow signals because 63 consecutive daily ICs overlap by 62 days.

### Bonferroni correction (conservative)

The simplest fix: require each individual p-value to be below α/m instead of α, where m is the number of tests. This controls the **Family-Wise Error Rate (FWER)** — the probability of even one false positive.

```
Bonferroni threshold: p_i ≤ α/m   →   t ≥ Φ⁻¹(1 - α/m)
With m=14, α=0.05:  t ≥ 2.69
```

**Problem:** Bonferroni assumes tests are independent. Factor signals are correlated (momentum variants have ρ ≈ 0.88). When tests are positively correlated, Bonferroni is far too conservative — it throws away real signals.

### BHY correction (recommended)

The **Benjamini-Hochberg-Yekutieli (BHY)** procedure controls the **False Discovery Rate (FDR)** — the expected fraction of positive results that are false — and is valid under arbitrary dependence.

**Algorithm:**
1. Sort the m p-values: p_(1) ≤ p_(2) ≤ ... ≤ p_(m)
2. Compute the harmonic correction factor: c(m) = Σᵢ₌₁ᵐ 1/i  (= 1 + 1/2 + 1/3 + ... + 1/m)
3. For each rank k, compute threshold: T_k = k / (m × c(m)) × α
4. Find the largest k such that p_(k) ≤ T_k
5. Reject all hypotheses with p-value ≤ p_(that k)

For m=14 positive-IC factors, c(14) ≈ 3.25, so the BHY threshold corresponds to **t ≥ 3.07**.

### Harvey-Liu-Zhu (2016) rule of thumb

Harvey, Liu & Zhu (2016) proposed a simpler asymptotic rule: require **t ≥ √(2·ln(m))**.

```
m = 14:  t ≥ √(2·ln(14)) = √5.30 ≈ 2.30
m = 100: t ≥ √(2·ln(100)) = √9.21 ≈ 3.03
m = 300: t ≥ √(2·ln(300)) = √11.4 ≈ 3.38
```

HLZ is less conservative than BHY in most cases. We use it as a lower bar.

### Current status (30 factors tested, 449-stock universe)

| Method | t threshold | Factors significant |
|--------|------------|-------------------|
| Unadjusted (naive) | 1.96 | ~6 (but ~1–2 are false) |
| HLZ | 2.30 | 1 (impl_1, 12-1 momentum) |
| **BHY (recommended)** | **3.07** | **0** |

**Honest conclusion:** No factor currently clears the BHY bar. The project is in an exploratory phase — we are building toward the 100–200 test count where the null distribution can be properly estimated.

---

## 11. Walk-Forward Validation

### Why in-sample testing is useless

If you fit any model to a dataset and evaluate it on the same dataset, you will always find a pattern — even if you're fitting noise. With 500 stocks × 3000 days = 1.5M data points and thousands of possible factors to try, **random search will always find something that looks good in-sample**.

### The walk-forward fix

We split time:
- **In-sample (IS): 2010–2017** — factor computation warmup only. No fitting, no parameter selection. The factor formula is fixed.
- **Out-of-sample (OOS): 2018–present** — all reported metrics. The factor has never "seen" this data.

The factor is computed on the full price history (so it has enough warmup data), but IC is only measured from 2018 onwards. This is a **hard temporal split** — no data leakage between training and test.

```
─────────────────────────────────────────────────────
2010                    2018                    2024
│    In-sample (warmup) │    Out-of-sample        │
│    (no fitting here)  │    (IC measured here)   │
─────────────────────────────────────────────────────
```

---

## 12. Temporal Stability: Was the IC Consistent?

Even if a factor's mean OOS IC is positive and its t-stat is high, there is a subtler question: **was the IC consistent across time, or did it only work in one particular market environment?**

### The intuition

Imagine two factors both have mean IC = 0.020 over 2018–2024:
- **Factor A:** IC ≈ 0.020 every year
- **Factor B:** IC = 0.080 in 2020 (COVID crash), IC ≈ −0.005 in all other years

Factor B's mean is high, but it was a one-event fluke. Factor A is genuinely robust.

### How we diagnose this

We split the OOS period into consecutive sub-periods (default: 2-year blocks) and compute the mean IC *within each block*:

| Period | Mean IC | Std IC | ICIR | t-stat (within-period) | N days |
|--------|---------|--------|------|------------------------|--------|
| 2018–2020 | 0.019 | 0.12 | 0.16 | 2.8 | 503 |
| 2020–2022 | 0.022 | 0.13 | 0.17 | 3.0 | 504 |
| 2022–2024 | 0.018 | 0.11 | 0.16 | 2.9 | 504 |

This is called a **temporal stability diagnostic**. It answers: *"Does the factor work consistently, or only in one era?"*

### What it does NOT do

Splitting 1760 IC observations into 5 blocks of 352 each does **not improve the main t-statistic**. The within-block t-stats are actually *weaker* (fewer data points). The correct significance test uses all 1760 daily ICs and applies BHY correction (Section 10).

Use the stability diagnostic for:
- ✅ Building confidence that a factor is structurally real
- ✅ Detecting regime-specific alpha (e.g. only works in bear markets)
- ❌ NOT for claiming significance where the full-period test fails

### Code

```python
from qframe.factor_harness.ic import compute_ic_by_period

period_df = compute_ic_by_period(
    factor_df, returns_df,
    oos_start='2018-01-01',
    period_years=2.0,   # ~2-year blocks → 3 blocks over 2018-2024
    horizon=1,
)
# Returns: period_label, period_start, period_end,
#          mean_ic, std_ic, icir, t_stat (within-period), n_days
```

**Visualisation:** Chart 14 in the notebook shows this as a bar chart with 95% CI error bars and per-period ICIR annotations.

---

## 13. The Gate System

Factors go through sequential gates before being trusted:

| Gate | What it tests | Pass criteria |
|------|--------------|---------------|
| **Weak gate** | Any cross-sectional alpha? | IC ≥ 0.015 AND ICIR ≥ 0.15 |
| **Pass gate** | Strong, consistent alpha | IC ≥ 0.030 AND ICIR ≥ 0.40 |
| **Slow signal** | Slow signal is real | slow_icir_63 ≥ 0.25 |
| **Gate 2** | Regime-conditional alpha | OOS IC difference across market regimes > 0.03 |
| **Gate 3** | Full portfolio Sharpe | OOS Sharpe > 0.5, max drawdown < 25% |
| **Gate 4** | Second market | Same direction on STOXX 600 or crypto |

Gates 0–2 are implemented and passed. Gate 3 (net-of-cost Sharpe) is next (Phase 3).

**Why so many gates?** Each gate eliminates a different class of spurious results:
- Weak gate: eliminates pure noise
- Pass gate: eliminates weak/inconsistent signals
- Gate 2: eliminates factors that only work in one market regime
- Gate 4: eliminates factors specific to US equities (data mining)

---

## 14. The Agentic Pipeline

Instead of manually writing factor code, this project uses a closed-loop AI system:

```
Human sets domain (e.g. "momentum")
        ↓
[Synthesis Agent — Groq Llama 70B]
Reads: academic literature seeds, all prior factor descriptions
Outputs: HypothesisSpec (name, description, mechanism, code sketch)
        ↓
[Implementation Agent — Qwen2.5-Coder:14b via Ollama, LOCAL]
Inputs: HypothesisSpec
Outputs: Python function: factor(prices: DataFrame) → DataFrame
        ↓
[Validation — deterministic Python]
Runs walk-forward harness
Computes IC, ICIR, IC decay, slow ICIR, turnover, net IC
        ↓
[Analysis Agent — Groq/Gemini]
Interprets metrics, issues PASS / FAIL / ERROR verdict
        ↓
[Knowledge Base — SQLite]
Logs everything: hypothesis, code, metrics, verdict
```

Every hypothesis and result is permanently stored. The pipeline reads all prior hypotheses before generating a new one, so it avoids re-testing the same ideas.

---

## 15. Factor Correlations

Two factors can both have IC > 0 but be measuring the same underlying phenomenon. Combining them in a portfolio gives no diversification benefit.

We measure **pairwise Spearman rank correlation** between factor signals in the OOS period:
- ρ ≈ 0: signals are orthogonal — combining them reduces noise
- ρ > 0.7: signals are essentially the same — only keep the better one

**Current finding:** The three best momentum factors (impl_7, impl_10, impl_50) have ρ ≈ 0.80–0.88. They are effectively one signal, not three. `price_level_autocorrelation` has |ρ| < 0.08 with all other factors — it is genuinely independent.

---

## 16. The Knowledge Base Schema

All results live in `knowledge_base/qframe.db` (SQLite). Three main tables:

```
hypotheses
  id, factor_name, description, rationale, mechanism_score (1-5), status

implementations
  id, hypothesis_id → hypotheses, code (Python), git_hash

backtest_results
  id, implementation_id → implementations
  ic, icir, net_ic, sharpe, max_drawdown, turnover
  ic_horizon_1, ic_horizon_5, ic_horizon_21, ic_horizon_63  ← IC at each horizon
  slow_icir_21, slow_icir_63                                ← honest slow ICIR
  ic_decay_json                                             ← full 63-point curve
  passed_gate, gate_level, oos_start, universe
```

Every number you see in the leaderboard came from this database.

---

## 17. Reading the Results

### What to look for in a factor

**A good fast factor (momentum-type):**
- IC@1d > 0.015, ICIR > 0.15
- IC decay: high at h=1, declining by h=21
- Turnover < 5%/day
- Net IC > 0 (positive after costs)

**A good slow factor (mean-reversion type):**
- IC@63d > 0.03, slow_icir_63 > 0.25
- IC decay: flat or *increasing* with horizon
- Turnover < 2%/day (naturally low — only rebalance quarterly)
- Net IC ≈ Gross IC (costs are negligible at quarterly rebalancing)

**Red flags:**
- IC@1d = 0.05+ → probably look-ahead bias
- ICIR negative but IC positive → inconsistent signal, driven by a few lucky periods
- Turnover > 30%/day → impractical and cost-dominated
- IC@63d >> IC@1d but slow_icir_63 < 0.10 → just noise in the tail

### The current leaderboard (96 factors, OOS 2018–2024, 449-stock honest universe)

BHY threshold with m=84 positive-IC factors: t ≥ ~4.0

| Factor | IC | ICIR | t-stat | BHY sig? |
|--------|----|------|--------|---------|
| impl_82 `trend_quality_calmar_ratio` (h=1) | 0.0646 | 0.382 | 10.74 | ✅ |
| impl_92 `calmar_proxy_252` (h=1) | 0.0646 | 0.382 | 10.74 | ✅ (duplicate of impl_82) |
| impl_53 `mean_reversion_factor` (h=63) | 0.0490 | 0.251 | 8.15 | ✅ |
| Momentum cluster (impl_1/7/…) | 0.016–0.017 | 0.166–0.169 | ~2.8 | ❌ (HLZ only) |

**3 BHY-significant factors (2 unique signals).** Phase 1 gate cleared. Phase 2 HSMM regime analysis complete — impl_53 lift = 2.28× in the strong-bull regime, above the 1.5× threshold. Gate 2 passed.

---

## 18. Phase Roadmap

| Phase | Goal | Status |
|-------|------|--------|
| **0** | Factor harness, SQLite KB, unit tests | ✅ Complete |
| **1** | Agentic pipeline — explore factor library | ✅ Complete — 96 factors; 3 BHY-sig |
| **2** | HSMM regime detection — does IC vary by market state? | ✅ Complete — impl_53 lift=2.28× |
| **2.5** | Regime-conditional portfolio construction | ⬜ Not started |
| **3** | Combined impl_82 + impl_53 strategy, net-of-cost Sharpe ≥ 0.5 | ⬜ Next |
| **4** | Crypto extension — does the factor work on BTC/ETH? | ⬜ Future |
| **5** | Cross-lingual signals — Chinese sentiment → Western equities | ⬜ Future |

Gate rule: **Phase N+1 cannot start until Phase N passes validation on two independent markets.** No exceptions.

---

## 19. Glossary

| Term | Plain English meaning |
|------|-----------------------|
| **IC** | Rank correlation between your prediction and what actually happened |
| **ICIR** | Sharpe ratio of the IC — how consistently right you are |
| **slow_icir_63** | Same as ICIR but computed on non-overlapping 63-day periods (statistically honest for slow signals) |
| **IC decay** | How the IC changes as you extend the prediction horizon from 1 to 63 days |
| **factor** | A function that assigns a score to every stock today, based on price history |
| **cross-sectional** | Comparing stocks against each other on the same day (not tracking one stock over time) |
| **OOS** | Out-of-sample: data the model/factor has never "seen" during development |
| **turnover** | Fraction of the portfolio replaced each day (5% means 5% of stocks are bought/sold) |
| **bps** | Basis points: 1 bps = 0.01%. 100 bps = 1%. Used for costs and small returns. |
| **net IC** | IC after subtracting transaction costs |
| **gate** | A pass/fail hurdle a factor must clear before progressing to the next test |
| **HSMM** | Hidden Semi-Markov Model — a state machine for detecting market regimes (bull/bear/crash) |
| **survivorship bias** | The distortion from only having data on stocks that survived — bankrupt companies disappear from the dataset |
| **look-ahead bias** | Using future information to compute today's signal — makes factors appear far better than they are |
