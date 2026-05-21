# Day 1 - Returns, Equity Curves, And Drawdowns

## Purpose

Day 1 builds the base layer for almost every quant strategy.

By the end of the session, you should be able to explain:

- What a price series is.
- What a return series is.
- Why returns are the object we usually model.
- How returns compound into an equity curve.
- What a drawdown is.
- Why max drawdown is often more emotionally and practically important than volatility.
- How small implementation choices can change results.

This is intentionally basic. The point is to make the foundation solid enough
that later strategy work has somewhere to stand.

## Mentor Assessment

Strengths of this Day 1 plan:

- It starts from quantities used in every strategy.
- It combines concept, code, visualization, and writing.
- It keeps the code small enough to fully understand.
- It introduces risk before alpha, which is the right instinct for trading.
- It creates a reusable diagnostic pattern for future strategies.

Weaknesses and risks:

- Daily price data is simpler than 15-minute data, so it hides intraday execution problems.
- Adjusted-close data can make ETF handling look cleaner than it is.
- A beautiful equity curve can still be statistically meaningless.
- Metrics can become decorative if we do not connect them to decisions.
- Python frustration can crowd out the actual financial idea.

Improvements built into the plan:

- Use daily data first, then explicitly list what changes for 15-minute trading.
- Write the formula before using a library function.
- Produce one small script or notebook, not a framework.
- Finish with a memo, not just charts.
- Keep a "questions for later" section so we do not chase every rabbit hole.

## Materials

Primary references:

- Ernest Chan, *Quantitative Trading*, chapters on backtesting and performance measurement.
- Marcos Lopez de Prado, *Advances in Financial Machine Learning*, read later for research hygiene; do not start here on Day 1.
- Existing local guide: `projects/all-weather/learning_guide.md`, especially the sections on backtests and metrics.

Useful online references to look up when needed:

- Portfolio Visualizer glossary for metrics.
- Investopedia for plain-English ETF and metric definitions.
- pandas documentation for `pct_change`, `cumprod`, `rolling`, and time series indexing.

Do not over-read. Day 1 is about owning five objects: prices, returns, equity,
peaks, drawdowns.

## Data Choice

Use 3 to 5 liquid ETFs:

- `SPY`: US large-cap equities.
- `QQQ`: growth/technology-heavy equities.
- `TLT`: long-duration US Treasuries.
- `GLD`: gold.
- `IWM` or `EFA`: small-cap US or international equities.

Start with daily adjusted prices. Intraday data comes later.

Why adjusted prices:

- Dividends matter.
- Splits and distributions distort raw prices.
- Total-return-like behavior is closer to what an investor experienced.

Caveat:

- Adjusted prices are not directly tradable prices. This matters more later,
  especially for intraday trading and execution assumptions.

## Four-Hour Schedule

### 0:00-0:15 - Setup And Goal

Write the session goal in your own words:

> "Today I am learning how a price history becomes a return stream, how a
> return stream becomes an equity curve, and how an equity curve reveals losses
> through drawdowns."

Open a scratch file or notebook.

Choose the ETF list.

Decide whether you will reuse code from `projects/all-weather` or write a fresh
minimal version. For Day 1, fresh and minimal is usually better.

### 0:15-0:45 - Concept Block

Definitions to write by hand:

Price:

```text
P_t = observed adjusted price at time t
```

Simple return:

```text
r_t = P_t / P_{t-1} - 1
```

Log return:

```text
ell_t = log(P_t / P_{t-1})
```

Equity curve for one asset:

```text
E_t = E_0 * product_{s <= t} (1 + r_s)
```

Running peak:

```text
M_t = max(E_0, E_1, ..., E_t)
```

Drawdown:

```text
D_t = E_t / M_t - 1
```

Max drawdown:

```text
min_t D_t
```

Plain-English questions:

- Why do we model returns instead of prices?
- Why does compounding make the order of returns matter?
- Why can two strategies with the same average return feel very different?
- Why is max drawdown path-dependent?

Expected understanding:

- A return is a relative change, not a dollar change.
- An equity curve is the cumulative result of reinvesting returns.
- Drawdown measures pain from the previous high, not loss from the starting point.

### 0:45-1:30 - Code Reading

Read these files lightly:

- `projects/all-weather/backtest.py`
- `projects/all-weather/portfolio.py`
- `projects/all-weather/validation.py`
- `projects/all-weather/tests/test_stats.py`

Do not try to understand every line.

Look specifically for:

- Where returns are computed.
- Where portfolio value is updated.
- Where performance statistics are computed.
- Whether drawdown appears explicitly.
- Whether costs or rebalancing appear.

Write down:

- One function you understand.
- One function you partly understand.
- One function or line that is unclear.

### 1:30-2:15 - Tiny Implementation

Write a small script or notebook that does only this:

1. Load ETF prices.
2. Compute daily simple returns.
3. Compute cumulative equity curves.
4. Compute rolling peaks.
5. Compute drawdowns.
6. Print summary metrics.
7. Plot equity and drawdown.

Target functions:

```python
def simple_returns(prices):
    ...

def equity_curve(returns, initial_value=1.0):
    ...

def drawdowns(equity):
    ...

def max_drawdown(equity):
    ...
```

Keep the functions small. No classes. No optimizer. No strategy engine.

Important implementation choices to notice:

- The first return is missing because there is no previous price.
- A return of `0.01` means 1 percent, not 1 dollar.
- `cumprod` compounds multiplicatively.
- Drawdowns are zero at new highs and negative below prior highs.
- Max drawdown is usually reported as a negative number or its absolute value.
  Pick one convention and state it.

### 2:15-2:45 - Manual Check

Before trusting the code, verify a toy example by hand.

Example:

```text
Prices: 100, 110, 99, 120
Returns: 10%, -10%, 21.2121%
Equity from 1.0: 1.0, 1.1, 0.99, 1.2
Running peak: 1.0, 1.1, 1.1, 1.2
Drawdown: 0%, 0%, -10%, 0%
Max drawdown: -10%
```

If your function does not reproduce this, debug before moving on.

### 2:45-3:15 - Experiment

Run the code on the selected ETFs.

Produce:

- A price plot.
- An equity curve plot.
- A drawdown plot.
- A summary table.

Minimum summary metrics:

- Total return.
- Annualized return.
- Annualized volatility.
- Sharpe ratio, with risk-free rate set to zero for now.
- Max drawdown.

Questions to answer:

- Which ETF had the highest total return?
- Which had the worst drawdown?
- Which looked easiest to hold emotionally?
- Did the highest-return ETF also have the best drawdown behavior?
- How different are the conclusions if you look only at return vs return and drawdown?

### 3:15-3:45 - Research Memo

Write a short memo with these sections:

```text
# Day 1 Memo - Returns, Equity Curves, And Drawdowns

## Question

## Data

## Method

## Results

## Interpretation

## Caveats

## Questions For Later
```

Keep it under one page.

The memo should be understandable without looking at the code.

### 3:45-4:00 - Explain It Back

Answer from memory:

1. What is the difference between a price and a return?
2. Why do returns compound with multiplication rather than addition?
3. What exactly is an equity curve?
4. What does a drawdown measure?
5. Why can max drawdown be more useful than volatility?
6. Where could lookahead bias enter this simple workflow?
7. What changes when the data frequency becomes 15 minutes?

If any answer is fuzzy, mark it in `progress.md`.

## What Changes For 15-Minute Trading

Daily data hides several problems that matter for the startup context:

- The bid/ask spread matters more.
- Slippage matters more.
- The open and close behave differently from the middle of the day.
- Missing bars are more common.
- A signal computed using the current bar close cannot trade at that same close
  unless the execution assumption is explicitly justified.
- Turnover can explode.
- Small per-trade costs can dominate the strategy.

Day 1 deliberately ignores most of this. We first learn the objects cleanly,
then we make the setup more realistic.

## Acceptance Criteria

Day 1 is complete when:

- You can define price, simple return, equity curve, peak, drawdown, and max drawdown.
- You have a small working implementation.
- The toy example matches by hand.
- You have produced plots and a summary table for several ETFs.
- You have written the memo.
- You can explain the caveats without prompting.

## Stop Conditions

Stop and ask for mentor review if:

- You cannot explain what a line of code is doing.
- Your toy example does not match the expected drawdown.
- The data source changes prices in a way you do not understand.
- You are tempted to add a strategy before the metrics are clear.

