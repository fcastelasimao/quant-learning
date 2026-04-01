# Wave Rider Research Goal

## Mission

Build a strategy that can survive rocky markets, reduce drawdowns, and
outperform simple passive benchmarks on a risk-adjusted basis.

This is a research goal, not a promise of profit.

## What We Are Trying To Build

A dynamic cross-asset allocation system that:
- participates when trends are healthy
- cuts risk when markets become unstable
- can hold substantial cash when no strong opportunity exists
- is simple enough to understand, test, and trust

## What We Are Not Trying To Build

- an all-weather clone
- a prediction engine for macro events
- a highly parameterised machine-learning strategy
- a strategy that depends on a single market regime continuing
- a system that must always be fully invested

## Desired Outcome

Compared with sensible baselines, Wave Rider should aim to:
- improve drawdown control
- keep return degradation acceptable during bad regimes
- capture enough upside during trending periods
- remain understandable after all risk rules are added

## Minimum Standard For Credibility

Before we trust the strategy, it should show:
- a correct and testable backtest engine
- realistic transaction costs and turnover reporting
- performance versus simple benchmarks
- robustness across nearby parameter settings
- walk-forward or out-of-sample validation

The first two items are now in place in the codebase. The remaining items are
still research work, not solved problems.

## Practical Framing

The realistic target is not "the best possible strategy for the future market."
The realistic target is "a disciplined strategy with a reasonable chance of
holding up better than passive investing when markets become hostile."

## Current Status

Wave Rider now has:
- a runnable backtest engine
- a first baseline strategy implementation
- benchmark support
- transaction cost tracking
- walk-forward reporting
- unit tests around core mechanics

What it does not have yet:
- validated out-of-sample evidence
- a final live universe decision
- robustness proof across parameter ranges
- enough research to trust real capital to it
