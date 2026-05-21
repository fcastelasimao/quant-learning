# Quant Mentorship Track

This folder is the working home for a four-week quant finance readiness sprint.
The goal is not to collect impressive material. The goal is to become useful,
careful, and clear in systematic ETF research.

## Mentor Framing

Current profile:

- Strong mathematical maturity from a PhD.
- Quant finance concepts are mostly familiar by name, but not yet owned.
- Python reading is more comfortable than Python writing.
- Prior projects sometimes grew faster than understanding.
- Likely target context: ETF strategies trading every 15 minutes.
- Available time: roughly 4 hours per day.

The training should therefore emphasize:

1. Market mechanics: what is actually being traded.
2. Statistical judgment: whether evidence is real or noise.
3. Backtesting hygiene: avoiding accidental cheating.
4. Readable Python: small functions, transparent assumptions, reviewable output.
5. Research communication: short memos that separate hypothesis, evidence, and caveats.

## Repo Decision

For now, keep this track inside `quant-learning`.

Reasons:

- The repo already contains useful prior work, books, and experiments.
- We can compare new clean work against older projects without copying context.
- A new repo would create overhead before the learning loop is stable.

When to create a separate repo:

- The training code becomes reusable as a clean quant research template.
- The structure stabilizes.
- You want to show it as a portfolio artifact.
- Old experiments begin to distract from current work.

Until then, this folder is the clean control center.

## Operating Rules

- Small files beat giant generated systems.
- Every concept must be connected to an executable artifact.
- Every artifact must be explainable in plain language.
- Every backtest must state timing, costs, and assumptions.
- We do not move on because a chart looks good. We move on when the mechanism is understood.

## Daily Loop

Each four-hour session should follow this shape:

1. Concept block: 45 minutes.
2. Code-reading or tiny implementation: 90 minutes.
3. Experiment: 60 minutes.
4. Research memo: 45 minutes.

The memo is not optional. It is how vague understanding becomes durable.

## First Milestone

Day 1 focuses on returns, equity curves, and drawdowns.

Read:

- [Day 1 plan](day-01-returns-equity-drawdowns.md)
- [Progress tracker](progress.md)

