# Research Log

---

## Project goal

Backtesting and validation engine for risk-balanced portfolios. Primary metric: Calmar.
Product positioned against Bridgewater's ALLW ETF ($1B+ AUM, 0.85% fees, ~2x leverage).

---

## Phase 1-8 (2026-03-18 to 2026-03-20)

8 asset universe experiments. Key: TIP replaces LQD, Calmar as primary metric,
6asset_tip_gsg as best universe, 8asset demoted.

## Phase 9 — DE optimiser (2026-03-21 to 2026-03-23)

26 experiments with fixed DE. All fail vs manual (0.403). Root cause: regime mismatch.
Gate 1 closed permanently.

## Phase 10A — Fixes (2026-03-23)

Rf=0.035, daily MDD, RP diagnostic function added.

## Phase 10B-D — Risk parity discovery (2026-03-25)

TLT 2x overweighted vs RP, TIP 2x underweighted. Window-stable across 3/5/10yr.
RP 5yr OOS Calmar 0.512 vs manual 0.403 on 2020-split (+27%).

## Phase 11 — Validation sprint (2026-03-25 to 2026-03-26)

### Data integrity audit
OOS numbers in strategies.json were wrong (used DE weights, not manual). Corrected.

### RP multi-window validation — Gate 2 PASSED

| Split | Manual | RP | Improvement |
|---|---|---|---|
| 2020 | 0.406 | 0.480 | +18% |
| 2018 | 0.417 | 0.462 | +11% |
| 2022 | 0.345 | 0.385 | +12% |

Production weights (averaged): SPY 13%, QQQ 11%, TLT 19%, TIP 33%, GLD 14%, GSG 10%.

### Universe scan
15 ETFs, 16,415 subsets, scored by diversification ratio under RP weights.
With min 4% vol and max 40% single-asset filters applied.
Result: TLT, GLD, GSG in all top-20. TIP absent (covariance-redundant in normal regimes
but essential in rate shocks — proven by OOS). Confirms 6-asset universe.

### ALLW comparison (RP weights, live ETFs, fee-adjusted)
CAGR 15.75% vs 15.64% (tied). Max DD -5.66% vs -8.79% (36% less drawdown).
Calmar 2.782 vs 1.779 (+56%). Caveat: 1 year of data only, rising-rate environment
favours unlevered strategy. ALLW's 2x leverage would help in falling rates.

### SPY overlay — Gate 3 CLOSED
126 parameter combinations grid-searched on IS. 110/126 beat baseline on IS.
Best params (d=15, th=12%, rp=100%) tested on 3 OOS splits:

| Split | Baseline | Overlay | Result |
|---|---|---|---|
| 2020 | 0.452 | 0.458 | +1.3% ✓ |
| 2018 | 0.450 | 0.456 | +1.3% ✓ |
| 2022 | 0.359 | 0.340 | -5.3% ✗ |

Marginal gains on 2/3 splits, meaningful loss on hardest split. Not worth the complexity.
Root cause: re-entry timing — by the time D1>0 AND D2>0, recovery is already 5-10% in.

---

## Current state (2026-03-26)

Research phase substantially complete. Production allocation decided.
Next steps: full-period backtest with production weights, paper trading,
blog post for demand validation, FCA consultation.

## Open questions

- Brand name
- FCA compliance for "rebalancing instructions"
- Rolling RP (quarterly recompute) — Phase 12 feature
- Rebalancing mismatch (backtest monthly vs live 5% threshold)
- GBP/EUR adjusted backtest