# ETF Strategy Research Memo

## Hypothesis

State one testable idea. Keep it narrow enough that a bad result is informative.

Example: volatility-aware exposure improves validation drawdown without destroying CAGR.

## Motivation

Explain the finance intuition and the statistical reason this might work.

Include:
- Which ETF behavior or market regime the idea targets.
- Which current metric, vote, or strategy rule it changes.
- Why this should generalize beyond the train period.

## Method

Describe exactly what changed.

Include:
- Inputs used.
- Parameters selected on train.
- Fill rule and cost assumptions.
- Train/val windows used.
- What was deliberately left unchanged.

## Results

| Split | CAGR | Sharpe | MaxDD % | DD Duration | Exposure % | Turnover | Trades | vs TQQQ B&H |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train |  |  |  |  |  |  |  |  |
| Val |  |  |  |  |  |  |  |  |

Also report:
- Best benchmark comparison.
- Worst drawdown window.
- Whether validation degradation is acceptable.

## Failure Modes

List the ways this could be fooling you.

Check:
- Too few trades.
- One crisis period explains most of the result.
- Performance disappears after costs.
- Signal direction flips between train and validation.
- The test implicitly used future data.
- The idea is just a proxy for market beta or exposure.

## Decision

Choose one:
- Keep: the idea improves validation quality and has a plausible mechanism.
- Revise: there is a useful hint, but the implementation is too fragile.
- Reject: the evidence is weak, unstable, or not worth the added complexity.

Decision:

Next action:

