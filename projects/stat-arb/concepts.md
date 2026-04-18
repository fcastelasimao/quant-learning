# Statistical Foundations

This document covers the theory behind cross-sectional equity factor models.
The maths is less exotic than options pricing — it's mostly linear algebra and
statistics — but the empirical craft is harder. Knowing the theory is table stakes;
knowing what goes wrong in practice is what separates good quant researchers.

---

## 1. The Cross-Sectional Framework

### The setup
At each time t, you observe N stocks. Each stock i has:
- A **return** over the next period: `r_{i,t+1}`
- A vector of **characteristics** (signals): `x_{i,t} = (momentum, value, quality, ...)`

The fundamental question: can you predict `r_{i,t+1}` from `x_{i,t}`?

### Cross-sectional vs time-series
- **Time-series**: "will the market go up tomorrow?" (predict aggregate return)
- **Cross-sectional**: "will stock A outperform stock B?" (predict relative return)

Cross-sectional is easier. You don't need to predict market direction — you're
always hedged (long some stocks, short others). The alpha comes from getting
the *ranking* right, not the level.

### The linear factor model
The standard assumption:

```
r_{i,t+1} = alpha_i + beta_{i,1}*f_{1,t+1} + ... + beta_{i,K}*f_{K,t+1} + epsilon_{i,t+1}
```

where:
- `f_k` are common factor returns (market, size, value, momentum, ...)
- `beta_{i,k}` is stock i's loading on factor k
- `epsilon_i` is idiosyncratic return (stock-specific, diversifiable)
- `alpha_i` is the unexplained excess return — what you're looking for

---

## 2. Factor Construction

### The recipe
For any characteristic (e.g. momentum):

1. **Compute the raw signal** for each stock: `signal_i = f(prices_i, fundamentals_i)`
2. **Cross-sectional rank**: rank all N stocks by signal (1 = lowest, N = highest)
3. **Normalise**: convert ranks to z-scores: `z_i = (rank_i - mean) / std`
4. **Form portfolios**: long the top quintile, short the bottom quintile

### Why rank instead of using raw values?
- **Robustness**: ranks are immune to outliers (a stock with 10,000% return doesn't dominate)
- **Comparability**: different signals have different scales; ranks put them on the same footing
- **Non-linearity**: the relationship between signal and return may not be linear

### Specific factors

**Momentum (12-1)**
```
mom_i = price_{i,t-21} / price_{i,t-252} - 1
```
Skip the most recent month because of short-term reversal. The economic story:
investors underreact to news, so past winners continue winning for ~6-12 months.
Crashes in sharp reversals (2009 Q1, 2020 Q2).

**Value (book-to-market)**
```
BM_i = book_value_i / market_cap_i
```
High BM = "cheap" stock. The economic story: cheap stocks are cheap because
they're risky (distress risk, earnings uncertainty). The premium is compensation
for bearing that risk. Or: markets overreact to bad news and underreact to good news.

**Quality (ROE)**
```
ROE_i = net_income_i / shareholders_equity_i
```
High ROE = efficient, profitable company. Quality stocks have stable earnings,
low leverage, high margins. The premium exists because investors overpay for
"exciting" low-quality growth stories and underpay for boring quality.

**Low Volatility**
```
vol_i = std(daily_returns_i[-60:]) * sqrt(252)
```
Low-vol stocks outperform on a risk-adjusted basis (and often on a raw basis too).
This is the "low volatility anomaly" — it shouldn't exist in a rational market
but does because of leverage constraints and lottery preferences.

**Short-Term Reversal**
```
rev_i = price_{i,t} / price_{i,t-5} - 1
```
Stocks that went up this week tend to go down next week (and vice versa).
The economic story: market-making and liquidity provision. Short-term price
pressure from order flow is temporary and reverses.

---

## 3. Signal Evaluation

### Information Coefficient (IC)
The Spearman rank correlation between signal and next-period return:

```
IC_t = corr_rank(signal_t, return_{t+1})
```

Compute this at each rebalancing date. A good single factor has:
- Mean IC: 0.03 - 0.08 (yes, this is small — markets are efficient)
- IC_IR (mean IC / std IC): > 0.5 is good, > 1.0 is exceptional

### The Fundamental Law of Active Management (Grinold & Kahn)
```
IR ~ IC * sqrt(breadth)
```

where:
- **IR**: Information Ratio (Sharpe of the active return)
- **IC**: skill per bet (how good is your signal?)
- **Breadth**: number of independent bets per year

This is why cross-sectional equity works: even with tiny IC (~0.05), if you
trade 500 stocks monthly, breadth = 500 * 12 = 6000, so IR ~ 0.05 * sqrt(6000) ~ 3.9.
In practice, stocks are correlated so effective breadth is lower, but the point stands:
**many small edges beat one big edge**.

### Fama-MacBeth Regression
The gold standard for testing whether a factor is "real":

1. Each month t, run a cross-sectional regression:
   `r_{i,t+1} = gamma_{0,t} + gamma_{1,t}*signal_{i,t} + epsilon_{i,t}`

2. Collect the time series of slopes: `{gamma_{1,t}}`

3. Test whether the average slope is significantly different from zero:
   `t-stat = mean(gamma_1) / (std(gamma_1) / sqrt(T))`

A t-stat > 2 means the factor has a statistically significant premium.
But beware: with many factors tested, you need to adjust for multiple comparisons
(Bonferroni, or better, the Harvey-Liu-Zhu threshold of t > 3.0 for new factors).

### Quintile analysis
Sort stocks into 5 groups by signal. Compute average return per group.
If the factor is real, you should see a **monotonic** relationship:

```
Q1 (low signal)  ->  low return
Q2               ->  below average
Q3               ->  average
Q4               ->  above average
Q5 (high signal) ->  high return
```

If the relationship isn't monotonic, the factor is suspect.

---

## 4. Portfolio Construction

### Long/short formation
Given signal z-scores for N stocks:

**Equal weight quintile:**
- Long: top 20% of stocks, weight = 1/n_long per stock
- Short: bottom 20% of stocks, weight = -1/n_short per stock
- Dollar neutral: total long $ = total short $

**Signal weighted:**
- Weight proportional to z-score: `w_i = z_i / sum(|z_i|)`
- Natural dollar neutral (positive z sums offset negative z sums)
- Gives more weight to high-conviction positions

### Sector neutrality
Without sector neutrality, your "momentum" factor might just be:
"tech went up, energy went down, so go long tech, short energy."
That's a sector bet, not stock selection.

**Sector-neutral construction:**
Within each sector, rank stocks by signal. Go long high-signal, short low-signal
*within the same sector*. Total long and short are equal within each sector.

This isolates pure stock-selection alpha from sector rotation.

### Market neutrality
Beta-neutral: ensure the portfolio has zero market beta.
```
beta_portfolio = sum(w_i * beta_i) = 0
```
Adjust weights so the portfolio doesn't go up or down with the market.

---

## 5. Transaction Costs & Capacity

### Why costs matter
Academic factor returns look great because they assume zero transaction costs.
In reality, trading 500 stocks monthly with 30-50% turnover is expensive.

### Cost model
```
cost_i = half_spread_i + impact_i

half_spread_i = 5 bps (large cap) to 20 bps (small cap)
impact_i = 10 bps * (trade_value_i / ADV_i)^0.5
```

where ADV = average daily volume. This is a simplified square-root impact model.
More sophisticated models use Kyle's lambda or Almgren-Chriss.

### Capacity
The strategy's Sharpe ratio degrades as AUM grows (because your trades have more
market impact). The "capacity" is the AUM at which the Sharpe drops below ~1.

```
Sharpe(AUM) = Sharpe_gross - cost_per_dollar * turnover * AUM / ADV_universe
```

Plot this curve. Most academic factors have capacity $500M-$2B for a single strategy.

### Turnover
```
turnover_t = sum(|w_{i,t} - w_{i,t-1}|) / 2
```

Turnover of 30% means you replace 30% of the portfolio each month.
Momentum has high turnover (~40%/month). Value has low turnover (~10%/month).

---

## 6. Multi-Factor Combination

### Why combine?
Individual factors are noisy (IC ~ 0.05). Combining uncorrelated factors
increases the effective IC:

```
IC_composite ~ sqrt(sum(IC_k^2))    (if factors uncorrelated)
```

With 4 uncorrelated factors each with IC = 0.05:
IC_composite ~ sqrt(4 * 0.0025) = 0.10 — doubled.

### Methods

**Simple average:**
```
composite_i = mean(z_momentum_i, z_value_i, z_quality_i, z_lowvol_i)
```
Hard to beat. No estimation error. Use this as the baseline.

**IC-weighted:**
```
composite_i = sum(IC_k * z_{k,i}) / sum(IC_k)
```
Weight factors by their recent effectiveness. Uses trailing 36-month IC.
Better in theory, introduces estimation risk in practice.

**PCA:**
Run PCA on the factor signal matrix. The first principal component is the
"consensus" signal. Can reveal hidden structure but is harder to interpret.

---

## 7. Performance Attribution

### Fama-French factor model
Regress your portfolio returns on known factor returns:

```
R_p = alpha + beta_mkt*(R_mkt - R_f) + beta_smb*R_SMB + beta_hml*R_HML
      + beta_rmw*R_RMW + beta_cma*R_CMA + epsilon
```

Data source: [Kenneth French's website](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) (free daily/monthly factor returns).

- **alpha > 0**: your strategy generates return beyond what's explained by known factors. This is the holy grail.
- **alpha = 0**: your strategy is just a repackaged loading on known factors. Still useful (as a factor ETF) but not alpha.

### Risk decomposition
Total portfolio variance:
```
Var(R_p) = sum(beta_k^2 * Var(f_k)) + Var(epsilon)
           \_________________________/   \________/
            systematic risk              idiosyncratic risk
```

The goal is high idiosyncratic return (alpha) with low systematic risk (beta).
If your Sharpe is 2.0 but alpha is zero, you're just levered beta.

---

## Reading List

### Essential (read before or during implementation)
1. **Grinold & Kahn, "Active Portfolio Management"** — Chapters 1-6, 14.
   The bible of quantitative equity. Defines IC, breadth, IR, and the fundamental law.
   Dense but essential.

2. **Fama & French (1993), "Common Risk Factors in the Returns on Stocks and Bonds"**
   The three-factor model. Defines market, size, and value factors. Read the original — it's a landmark paper.

3. **Jegadeesh & Titman (1993), "Returns to Buying Winners and Selling Losers"**
   The momentum paper. Documents the 12-1 month effect. Short and clear.

### Recommended (deepen understanding)
4. **Asness, Moskowitz & Pedersen (2013), "Value and Momentum Everywhere"**
   Shows value and momentum work across equities, bonds, currencies, commodities.
   Great evidence that these are real, persistent phenomena.

5. **Harvey, Liu & Zhu (2016), "...and the Cross-Section of Expected Returns"**
   Critical paper: most published factors are false discoveries. Sets the t > 3.0 bar
   for new factors. Read this to understand the replication crisis in finance.

6. **Kakushadze (2016), "101 Formulaic Alphas"**
   Exactly what it says: 101 quantitative signal formulas. Good for breadth, not depth.

### Nice to have (interview depth)
7. **Barra Risk Model Handbook** — the industry-standard factor risk model.
   Used by every large fund. Available online.

8. **Almgren & Chriss (2001), "Optimal Execution of Portfolio Transactions"**
   The standard market impact model. Shows how to optimally trade large orders.
