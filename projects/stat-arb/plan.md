# Implementation Plan

---

## Phase 1 — Universe & Data Pipeline (Week 1)

**Goal:** Get clean, aligned daily data for ~500 US equities.

### 1A: Universe definition
- Start with current S&P 500 constituents (Wikipedia table or `pandas_datareader`)
- For survivorship bias: use the S&P 500 constituents list as of each rebalance date
  (free source: Wikipedia edit history, or use the simpler approach of requiring
  stocks to have existed for the full lookback window)
- Store as a simple CSV: `date, ticker, in_index`

### 1B: Price data
- Fetch daily adjusted close + volume for all tickers via `yf.download(batch)`
- Align all tickers to common trading days (fill NaN for delistings)
- Compute: daily returns, cumulative returns, 21-day rolling volume, market cap proxy
- Store as a single DataFrame (tickers as columns, dates as index)

### 1C: Fundamentals (simplified)
- For value/quality factors, you need accounting data (book value, earnings, ROE)
- Free sources: yfinance `.info` dict, SEC EDGAR bulk download
- Since free data is limited, start with price-based factors only (momentum, vol, reversal)
  and add fundamentals in Phase 3

### 1D: Data quality
- Drop tickers with >20% missing price data
- Flag and investigate extreme daily returns (>30%)
- Ensure no look-ahead bias: all data available on date T must have been observable on date T

**Deliverable:** `data/universe.py`, `data/prices.py` — clean daily price matrix

---

## Phase 2 — Factor Construction & Single-Factor Analysis (Week 1-2)

**Goal:** Build individual alpha signals and measure their predictive power.

### 2A: Momentum factor
- **12-1 month momentum**: cumulative return over months -12 to -2 (skip most recent month)
  `mom_t = price_{t-21} / price_{t-252} - 1`
  The skip avoids short-term reversal contaminating the momentum signal.
- Cross-sectional rank: on each date, rank all stocks by momentum
- Normalise to z-score: `z = (rank - mean(rank)) / std(rank)`

### 2B: Short-term reversal
- 1-week return: `rev_t = price_t / price_{t-5} - 1`
- Contrarian: high recent returns predict low future returns (over 1-2 weeks)
- Same ranking and z-score normalisation

### 2C: Low volatility
- Trailing 60-day realised volatility: `vol = std(daily_returns[-60:]) * sqrt(252)`
- Low vol stocks tend to outperform on a risk-adjusted basis (the "low vol anomaly")
- Note: the signal is **negative** vol — you want to go long low-vol stocks

### 2D: Single-factor evaluation
For each factor, compute:
- **IC (Information Coefficient)**: Spearman rank correlation between signal and next-month return
- **IC time series**: plot IC over time, compute mean IC and IC_IR (mean/std)
- **Quintile returns**: sort stocks into 5 groups by factor, compute average return per group
- **Long-short return**: return of quintile 5 minus quintile 1
- **Turnover**: fraction of positions that change at each rebalance

### 2E: Visualisations
- IC histogram and time series
- Quintile return bar chart (the "factor monotonicity" plot)
- Cumulative long-short return over time
- Turnover time series

**Deliverable:** `factors/momentum.py`, `factors/volatility.py`, `portfolio/analytics.py`

---

## Phase 3 — Value & Quality Factors (Week 2-3)

**Goal:** Add fundamentals-based factors. This is harder due to data limitations.

### 3A: Value factor
- **Earnings yield**: E/P = trailing 12-month earnings / market cap
- **Book-to-market**: book value / market cap
- Data source: yfinance `.info["trailingPE"]`, `.info["priceToBook"]`
- Handle missing data: not all stocks have clean fundamentals

### 3B: Quality factor
- **ROE**: return on equity (net income / shareholders' equity)
- **Debt-to-equity**: total debt / equity
- **Earnings stability**: std of quarterly earnings over last 3 years
- Composite quality score: z-score average of individual quality metrics

### 3C: Analyse each
Same evaluation pipeline as Phase 2: IC, quintile returns, turnover.
Value and quality factors typically have lower turnover than momentum
(fundamentals change slowly) but lower IC (noisier signal).

**Deliverable:** `factors/value.py`, `factors/quality.py`, `data/fundamentals.py`

---

## Phase 4 — Portfolio Construction & Backtesting (Week 3)

**Goal:** Turn signals into portfolios and backtest with realistic assumptions.

### 4A: Portfolio formation
- **Quintile long/short**: long top 20%, short bottom 20%, equal weight within each leg
- **Z-score weighted**: weight proportional to signal strength (higher conviction = bigger position)
- **Sector neutral**: within each GICS sector, go long high-signal stocks and short low-signal
  stocks. This removes sector bets — the portfolio is purely stock-selection alpha.
- **Dollar neutral**: total long notional = total short notional

### 4B: Backtest engine
- Monthly rebalancing (first trading day of each month)
- Track: portfolio weights, returns, turnover, gross/net exposure
- Transaction cost model: `cost = half_spread + impact`
  - Half spread: 5bps for large-cap, 15bps for small-cap
  - Linear impact: 10bps * (trade_size / ADV)^0.5
- Apply costs at each rebalance

### 4C: Performance metrics
- Total return, CAGR, Sharpe, max drawdown, Calmar (reuse from all-weather)
- But also factor-specific metrics:
  - **IC_IR**: mean(IC) / std(IC) — the "Sharpe of the signal"
  - **Hit rate**: fraction of months with positive L/S return
  - **Max DD duration**: how long does the worst drawdown last?
  - **Turnover-adjusted Sharpe**: Sharpe after realistic costs

### 4D: Capacity analysis
- Vary the assumed AUM from $10M to $1B
- At each size, compute the Sharpe after impact costs
- Plot Sharpe vs AUM — the curve shows where the strategy "breaks"
- Most academic factors have capacity ~$500M-$2B

**Deliverable:** `portfolio/construction.py`, `portfolio/backtest.py`

---

## Phase 5 — Multi-Factor Model (Week 4)

**Goal:** Combine factors into a composite signal that's stronger than any individual.

### 5A: Simple combination
- Z-score average: `composite = mean(z_momentum, z_value, z_quality, z_lowvol)`
- This is the baseline — surprisingly hard to beat

### 5B: IC-weighted combination
- Weight each factor by its trailing IC_IR
- Recalculate weights monthly using expanding window (no look-ahead)
- Gives more weight to factors that have been working recently

### 5C: Fama-MacBeth regression
- Each month: regress cross-section of next-month returns on all factor signals
- Collect the regression slopes (factor premia) over time
- Average slope = expected factor premium
- t-statistic of average slope = statistical significance

### 5D: Out-of-sample test
- Train combination weights on 2006-2016
- Test on 2016-2026
- Compare: does the combination beat the best individual factor OOS?

### 5E: Factor correlation analysis
- Compute pairwise correlation between factor signals
- PCA on the factor cross-section: how many independent sources of alpha?
- If factors are correlated, the diversification benefit of combining is small

**Deliverable:** `factors/composite.py`, OOS comparison plots

---

## Phase 6 — Attribution & Polish (Week 4-5)

**Goal:** Decompose returns, understand what's driving performance.

### 6A: Return attribution
- Regress portfolio returns on known factor returns (Fama-French 5 factors)
  `R_portfolio = alpha + beta_mkt*R_mkt + beta_smb*R_SMB + beta_hml*R_HML + ...`
- Alpha = the intercept. This is the "unexplained" return — the value you add.
- If alpha is zero, your "signal" is just loading on known factors

### 6B: Risk attribution
- What fraction of portfolio variance comes from market risk vs factor risk vs idiosyncratic risk?
- Variance decomposition using the factor model covariance structure
- This tells you: "I'm taking 40% market risk, 30% factor risk, 30% idiosyncratic risk"

### 6C: Regime analysis
- Split the sample into regimes: bull/bear markets, high/low volatility
- Does the factor work in all regimes or only some?
- Momentum famously crashes in bear-to-bull reversals (2009, 2020)

### 6D: Code quality & documentation
- Type hints, docstrings, tests
- README with summary results table
- Prepare to explain: why these factors? what's the economic intuition?
  why would the premium persist? what's the capacity constraint?

**Deliverable:** `portfolio/attribution.py`, final results, clean code
