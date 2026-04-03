# Research Log

---

## Project goal

Backtesting and validation engine for risk-balanced portfolios.
Primary metric: Calmar ratio. Positioned against Bridgewater's ALLW ETF.

## Phase 1-8 (2026-03-18 to 2026-03-20)
Explored 8 universes. TIP replaces LQD. 6asset_tip_gsg as best universe.

## Phase 9 — DE optimiser (2026-03-21 to 2026-03-23)
26 experiments, all fail vs manual. Gate 1 closed.

## Phase 10 — RP foundation (2026-03-23 to 2026-03-25)
Rf=0.035. Daily MDD. RP diagnostic: TLT 2x overweighted, TIP 2x underweighted.
RP 5yr OOS Calmar 0.512 vs manual 0.403.

## Phase 11 — Validation (2026-03-25 to 2026-03-26)

### Data integrity: OOS results in strategies.json corrected.

### RP multi-window: Gate 2 PASSED
| Split | Manual | RP | Improvement |
|---|---|---|---|
| 2020 | 0.406 | 0.480 | +18% |
| 2018 | 0.417 | 0.462 | +11% |
| 2022 | 0.345 | 0.385 | +12% |

### Universe scan: 15 ETFs, 16k subsets. Confirms 6-asset universe.

### Overlay grid: 126 combos, 3-split OOS. Does not add value. Closed.

### ALLW comparison (monthly rebalanced, fee-adjusted, Mar 2025-Mar 2026)
| Metric | rpavg | ALLW |
|---|---|---|
| CAGR | 16.05% | 17.23% |
| Max DD | -5.74% | -8.79% |
| Calmar | 2.797 | 1.961 |
| Ulcer | 1.134 | 1.845 |

## Phase 12 — Rolling RP + rebalancing frequency (current)

### Rolling RP concept
Instead of static weights computed once, recompute RP weights quarterly
from trailing 5-year covariance. Weights adapt to structural shifts
(e.g. rising rate regime post-2021) without predicting regimes.

### Weekly rebalancing
Test DATA_FREQUENCY="W" with transaction costs. Compare to monthly.
Hypothesis: weekly won't improve Calmar after costs because RP weights
don't drift meaningfully in one week.

---

## Open questions
- Does rolling RP beat or match static RP across 3 OOS windows?
- Does weekly rebalancing improve Calmar after transaction costs?
- Brand name, FCA compliance, GBP/EUR adjustment