# Funding Rate Arbitrage

Delta-neutral strategy that captures the funding rate spread between perpetual futures and spot positions on crypto exchanges.

## Status

Planned. Not yet started.

## Concept

When perpetual futures trade at a premium to spot, longs pay shorts a periodic funding rate. A delta-neutral position (long spot + short perp) captures this yield without directional exposure.

Key research questions:
- Which exchanges and pairs offer the most consistent funding rate premium?
- What is the realistic yield after exchange fees, slippage, and withdrawal costs?
- How does the strategy behave during funding rate inversions (negative funding)?
