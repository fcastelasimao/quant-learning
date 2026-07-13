# One-Month Quant Finance Plan For ETF Strategy Research

Educational note: this plan is for research skill-building, not investment advice.

## Operating Rules

- Time budget: 8-10 focused hours per week from May 11 to June 7, 2026.
- Main goal: improve judgment around ETF buy/hold/sell research, especially TQQQ/QQQ.
- Research rule: tune on train, validate on val, and keep test frozen until the strategy is final.
- Weekly rhythm: 2 hours reading, 4-5 hours repo work, 1-2 hours writing the deliverable, 1 hour review.
- Default target: each week ends with a small artifact that would survive critique by a quant-minded reader.

## Week 1: ETF Mechanics, Returns, And Risk

Learn:
- Simple returns, log returns, compounding, volatility, drawdown, CAGR, Sharpe, turnover, and transaction costs.
- ETF mechanics: tracking, liquidity, spreads, expense ratios, and daily rebalancing.
- Leveraged ETF behavior: TQQQ targets roughly 3x daily QQQ returns, not 3x multi-day QQQ returns.

Repo work:
- Read `README.md`, `METRICS.md`, and the latest backtest report output.
- Trace how `src/backtest.py` computes returns, costs, drawdowns, and benchmark comparisons.
- Compare QQQ close-to-close returns with TQQQ returns over calm and volatile windows.

Deliverable:
- Write a one-page note explaining why QQQ signals can be cleaner than TQQQ signals, and why high volatility can damage leveraged ETF returns.

Suggested prompts:
- What part of TQQQ return is intended leverage, and what part is path dependence?
- Does a high CAGR matter if the drawdown duration is psychologically or operationally unbearable?
- Which costs are explicit in this repo, and which real-world costs are still simplified?

## Week 2: Probability And Statistics For Trading Signals

Learn:
- Expectation, variance, covariance, correlation, sampling error, confidence intervals, hypothesis tests, p-values, and multiple testing.
- Time-series hazards: autocorrelation, non-IID returns, regime changes, volatility clustering, lookahead bias, and survivorship bias.

Repo work:
- Classify every metric by family using `src/metrics.py`.
- Run the signal credibility diagnostics:

```bash
cd src
python -m signal_diagnostics --split train --horizon 5
python -m signal_diagnostics --split val --horizon 5
python -m signal_diagnostics --compare-train-val --horizon 5
```

Optional CSV export:

```bash
cd src
python -m signal_diagnostics --compare-train-val --horizon 5 --output ../outputs/signal_credibility_5d.csv
```

Deliverable:
- Produce a ranked table of signals by credibility, not just raw performance.

Suggested prompts:
- Does a signal work in both train and validation, or only where it was discovered?
- Is the signal direction stable, or does it flip?
- Are bullish and bearish votes both represented, or is the result driven by a tiny sample?

## Week 3: Strategy Design And Backtesting Discipline

Learn:
- Train/validation/test separation, walk-forward testing, transaction costs, slippage, benchmark choice, exposure-adjusted performance, and overfitting.

Repo work:
- Study `src/strategy.py` until you can explain `score`, `tau`, `alpha`, `p_buy`, `p_hold`, and `p_sell`.
- Study `src/backtest.py` until you can explain the one-day position shift and why it matters.
- Run train and val experiments only. Do not use the test split for iteration.

Deliverable:
- Complete the backtest checklist below before trusting any new result.

Backtest checklist:
- Causality: every signal at time `t` uses only information available by time `t`.
- Fill assumption: the strategy decides at close `t` and fills at open `t+1`.
- Costs: turnover and position changes include explicit costs; any missing costs are documented.
- Benchmarks: compare against TQQQ buy-and-hold and QQQ buy-and-hold.
- Risk: report max drawdown, drawdown duration, exposure, turnover, and trade count.
- Validation: parameters are selected on train and evaluated on val without re-tuning.
- Robustness: performance is not explained by one tiny period, one metric, or one lucky trade.

## Week 4: One Research-Grade Strategy Improvement

Choose one hypothesis:
- Improve vote weighting.
- Add volatility-aware exposure.
- Improve hold/sell logic.
- Calibrate probabilities.
- Remove weak or noisy metrics.

Rules:
- Change one hypothesis family at a time.
- Avoid broad parameter searches.
- Evaluate train first, then validation.
- Freeze the test set until the strategy is final.

Deliverable:
- Write a short research memo with this structure:
  - Hypothesis.
  - Motivation.
  - Method.
  - Results.
  - Failure modes.
  - Decision.

Acceptance criteria:
- The memo includes CAGR, Sharpe, max drawdown, drawdown duration, exposure, turnover, trade count, validation degradation, and comparison to QQQ/TQQQ buy-and-hold.
- The decision says whether to keep, revise, or reject the idea.

## Core Reading

- SEC ETF bulletin: https://www.investor.gov/additional-resources/news-alerts/alerts-bulletins/investor-bulletin-exchange-traded-funds-etfs
- SEC leveraged and inverse ETF bulletin: https://www.investor.gov/introduction-investing/general-resources/news-alerts/alerts-bulletins/investor-alerts/sec
- FINRA leveraged/inverse ETP note: https://www.finra.org/investors/insights/lowdown-leveraged-and-inverse-exchange-traded-products
- MIT 18.05 probability/statistics refresher: https://openlearninglibrary.mit.edu/courses/course-v1%3AMITx%2B18.05r_10%2B2022_Summer/about
- Ruppert and Matteson, Statistics and Data Analysis for Financial Engineering: https://link.springer.com/book/10.1007/978-1-4939-2614-5
- Moskowitz, Ooi, and Pedersen, Time Series Momentum: https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID2089463_code753937.pdf?abstractid=2089463&mirid=1
- Bailey et al., The Probability of Backtest Overfitting: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253
- Andrew Lo, The Statistics of Sharpe Ratios: https://rpc.cfainstitute.org/research/financial-analysts-journal/2002/the-statistics-of-sharpe-ratios

