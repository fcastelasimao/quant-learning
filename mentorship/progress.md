# Quant Mentorship Progress

Use this as the living checklist. The goal is not to complete boxes quickly.
The goal is to make each box honest.

Scoring:

- 0: I do not know this.
- 1: I recognize the words.
- 2: I can use it with guidance.
- 3: I can explain and apply it independently.

## Week 1 - Research Foundations

### Day 1 - Returns, Equity Curves, Drawdowns

Status: Not started

Checklist:

- [ ] I can define simple return.
- [ ] I can define log return.
- [ ] I can explain why returns are usually modeled instead of prices.
- [ ] I can compute an equity curve from returns.
- [ ] I can compute running peaks.
- [ ] I can compute drawdowns.
- [ ] I can compute max drawdown.
- [ ] I can reproduce the toy example by hand.
- [ ] I can explain the difference between volatility and drawdown.
- [ ] I have produced ETF plots and a summary table.
- [ ] I have written the Day 1 memo.

Explain it back:

```text
What is a return?

Why does compounding use multiplication?

What is a drawdown?

Why does max drawdown depend on the path?

What would change if this were 15-minute ETF data?
```

Open questions:

```text

```

Mentor review notes:

```text

```

### Day 2 - Volatility, Sharpe, And Rolling Windows

Status: Planned

Core questions:

- What is volatility estimating?
- What assumptions are hidden in annualization?
- Why can Sharpe be misleading?
- How do rolling windows change interpretation?

### Day 3 - Transaction Costs And Slippage

Status: Planned

Core questions:

- How do costs enter returns?
- Why do small costs matter at high turnover?
- What is the difference between commission, spread, and slippage?
- How can a strategy die after costs?

### Day 4 - Backtest Timing And Lookahead Bias

Status: Planned

Core questions:

- When is information known?
- When is the signal computed?
- When can the trade be executed?
- What does it mean to shift positions by one bar?

### Day 5 - First Research Memo Review

Status: Planned

Core questions:

- Can the result be explained without code?
- Are assumptions explicit?
- Are caveats honest?
- What should be tested next?

## Reference Queue

Use references as tools, not as a syllabus to drown in.

Primary:

- Ernest Chan, *Quantitative Trading*.
- Existing local `projects/all-weather/learning_guide.md`.
- pandas documentation for time series operations.

Secondary:

- Marcos Lopez de Prado, *Advances in Financial Machine Learning*.
- Grinold and Kahn, *Active Portfolio Management*.
- Meucci, *Risk and Asset Allocation*.
- Tsay, *Analysis of Financial Time Series*.

Later, if relevant:

- Hull, *Options, Futures, and Other Derivatives*.
- Baxter and Rennie, *Financial Calculus*.

## Weekly Self-Assessment

At the end of each week, answer:

```text
What can I now explain that was vague before?

What code can I now read without fear?

What assumptions did I discover matter more than I expected?

Where did I still rely on trust rather than understanding?

What should we slow down on next week?
```

