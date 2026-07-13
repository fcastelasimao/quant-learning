# Week 1 Note: QQQ Signals And TQQQ Path Dependence

## Claim

Write the central claim in one sentence.

Example: QQQ can be a cleaner signal source than TQQQ because it measures the underlying Nasdaq-100 state without the extra noise created by daily leverage, volatility drag, and path dependence.

## Mechanics

Simple return:

```text
r_t = P_t / P_{t-1} - 1
```

Log return:

```text
g_t = log(P_t / P_{t-1})
```

Approximate leveraged ETF drag term:

```text
drag ≈ 0.5 * L * (L - 1) * sigma^2
```

where `L = 3` for TQQQ and `sigma` is realized volatility of the underlying.

## Why QQQ Can Be Cleaner

- QQQ is the underlying exposure.
- TQQQ embeds daily leverage and rebalancing effects.
- TQQQ return depends on the path, not just the start and end point.
- High volatility can hurt TQQQ even when QQQ has no clear directional trend.

## Evidence From This Repo

Fill this with screenshots, tables, or notes from `METRICS.md`, the indicator workbench, or backtest outputs.

Observations:
- 
- 
- 

## What I Still Do Not Trust

List assumptions or simplifications that need care before relying on the result.

- 
- 
- 

