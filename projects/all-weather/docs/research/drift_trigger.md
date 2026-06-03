# Drift-trigger rebalancing (D.15)

**Status:** implemented in the engine; default unchanged. `RebalancePolicy` in
`engine/backtest.py`. Tests: `tests/test_rebalance_policy.py`,
`tests/test_backtest_golden.py`.

## Question

The production engine rebalanced to target weights **every month,
unconditionally**. Under realistic US tax (D.16) that realizes taxable gains
12×/year whether or not the portfolio has actually drifted. Can a drift-based
trigger — only trade when an asset's weight strays far enough from target —
defer tax and improve risk-adjusted returns?

The closed `failed_strategies/weekly_rebalance/` verdict rejected threshold
rebalancing, but that was under **transaction-cost-only** modelling. Tax changes
the trade-off, so the question was reopened (handoff decision §3.5).

## What was built

A `RebalancePolicy` value object with four modes:

| Mode | Trigger |
|---|---|
| `monthly_unconditional` (**default**) | rebalance every month |
| `drift_relative(pct)` | any asset's |weight − target| > `pct` × target |
| `drift_absolute(pp)` | any asset's |weight − target| > `pp` (percentage points) |
| `monthly_check_then_drift(pct)` | same as `drift_relative` in a month-end engine |

`should_rebalance(current_weights, target_weights)` is the single behavioural
entry point. When any asset breaches, **all** assets are restored to target
(full-on-breach, matching `research/rebalance_thresholds.py`).

## Zero-regression discipline

`run_backtest` gained a `rebalance_policy=None` parameter (last positional →
defaults to `monthly_unconditional`). The default path is **byte-identical** to
the pre-refactor engine:

- A golden fixture was captured from the **unmodified** engine
  (`tests/data/backtest_golden.csv`, synthetic seeded 6-asset series, 228 rows,
  sha256 `64c5753…`).
- The refactored default reproduces it exactly (hash MATCH).
- Full suite green (308 passed) after the change.

Transaction cost is now only charged on months that actually trade (a no-op for
the default; correct for drift modes).

## Note on the month-end engine

Because the engine resamples to month-end before iterating,
`monthly_check_then_drift` is **behaviourally identical** to `drift_relative`
here. It is kept as a distinct named policy so manifests/artifacts can record
intent; the two would diverge only in a finer-grained (daily) engine.

## Not done (deferred)

The D.15 spec also suggested folding `research/rebalance_thresholds.py`'s
threshold logic into the engine. **Not done** — that module has a richer
`hybrid` mode and full diagnostic CSV emission with its own passing test
contract, and the D.18 sweep calls `run_tax_aware_backtest(..., rebalance_policy=...)`
directly without needing it. Revisit under F.26 / H.35.

## Result

See `docs/research/tax_threshold_sweep.md` for the head-to-head verdict. Short
version: under US tax, **every** drift policy beat monthly rebalancing on Calmar
across all three OOS windows.
