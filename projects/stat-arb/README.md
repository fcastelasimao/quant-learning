# Cross-Sectional Equity Factor Model & Statistical Arbitrage

A from-scratch implementation of equity factor research: factor construction, alpha signal evaluation, long/short portfolio formation, and realistic backtesting with transaction costs.

> **This is a research framework, not a live trading system.** It demonstrates the full alpha research pipeline that systematic equity funds use.

---

## What This Project Covers

### 1. Universe & Data Pipeline
- Fetch daily prices and fundamentals for S&P 500 constituents
- Handle survivorship bias (use historical index membership, not today's list)
- Compute derived features: returns, volume, market cap, book-to-market, etc.

### 2. Factor Construction
- **Momentum**: 12-month return skipping the most recent month (Jegadeesh & Titman)
- **Value**: book-to-market ratio, earnings yield
- **Quality**: ROE, debt-to-equity, earnings stability
- **Low Volatility**: trailing realised vol, idiosyncratic vol (residual from market model)
- **Short-Term Reversal**: 1-week return (contrarian signal)
- Each factor: rank stocks cross-sectionally, normalise to z-scores

### 3. Alpha Signal Evaluation
- **Information Coefficient (IC)**: rank correlation between signal and next-period return
- **IC decay**: how quickly does the signal lose predictive power over time?
- **Turnover**: how many positions change each rebalance? (determines feasibility)
- **Fama-MacBeth regression**: cross-sectional regression each period, time-series average of slopes
- **Factor return analysis**: long top quintile, short bottom quintile, measure spread return

### 4. Portfolio Construction
- **Equal-weight long/short**: long top quintile, short bottom quintile
- **Signal-weighted**: weight proportional to z-score
- **Risk-neutral**: sector-neutral and market-neutral construction
- **Transaction costs**: realistic slippage model (linear impact, bid-ask spread)
- **Capacity analysis**: how much AUM before the strategy breaks?

### 5. Multi-Factor Combination
- Combine individual factors into a composite alpha signal
- Methods: simple z-score average, regression-based (fitted IC weights), PCA
- Out-of-sample test: train combination weights on first half, evaluate on second half

### 6. Performance Attribution
- Decompose returns into factor exposures (market, size, value, momentum)
- Compute alpha: the residual return after removing known factor loadings
- Risk attribution: what fraction of portfolio variance comes from each factor?

---

## Project Structure

```
stat-arb/
├── data/
│   ├── universe.py           S&P 500 universe with survivorship handling
│   ├── prices.py             Daily price/volume data via yfinance
│   └── fundamentals.py       Accounting data (earnings, book value, ROE)
├── factors/
│   ├── momentum.py           12-1 month momentum, short-term reversal
│   ├── value.py              Book-to-market, earnings yield
│   ├── quality.py            ROE, leverage, earnings stability
│   ├── volatility.py         Realised vol, idiosyncratic vol
│   └── composite.py          Multi-factor combination methods
├── portfolio/
│   ├── construction.py       Long/short formation (quintile, z-score, neutral)
│   ├── backtest.py           Realistic backtest engine with costs
│   ├── analytics.py          IC, turnover, Fama-MacBeth, factor returns
│   └── attribution.py        Return and risk decomposition
├── tests/
│   ├── test_factors.py
│   ├── test_portfolio.py
│   └── test_analytics.py
├── notebooks/                Exploration and visualisation
├── results/                  Generated output (.gitignore'd)
├── config.py                 Universe, date ranges, cost assumptions
├── plan.md                   Phase-by-phase implementation plan
├── concepts.md               Statistical foundations and learning guide
├── requirements.txt
└── README.md
```

## Installation

```bash
cd quant-learning/projects/stat-arb
conda create -n statarb python=3.11
conda activate statarb
pip install -r requirements.txt
```

## Known Limitations

- Fundamentals data via free APIs is limited (yfinance, SEC EDGAR)
- No intraday data — daily rebalancing only
- S&P 500 survivorship bias partially addressed but not perfectly
- No shorting costs modelled (borrow fee, locate availability)
- Transaction cost model is simplified (no market impact function)

## References

See `concepts.md` for the full statistical treatment and reading list.
