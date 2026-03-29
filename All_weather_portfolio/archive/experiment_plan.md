# Experiment Plan — Phase 10D+ (post-RP-diagnostic)

## Status: Phase 9 + 10C/10D complete. RP framework established.

**⚠️ This document reflects Phase 3 planning. Much of Workstream C is now
resolved. See session_handoff.md and research_log.md for current state.**

### Summary of resolved questions (as of 2026-03-25)
- ✅ Workstream C (DE repair + validation): CLOSED. DE confirmed to fail across
  26 experiments even after all four bug fixes. Gate 1 closed. Do not re-open.
- ✅ Workstream A (Phase 10A engineering): Rf fix, daily MDD, RP diagnostic done.
- ✅ Phase 10C: RP 5yr weights on 6asset — OOS Calmar 0.512 vs manual 0.403 (+27%).
- ✅ Phase 10D: RP 5yr weights on 7asset_vnq — OOS Calmar 0.459 vs manual 0.465 (flat).

### Open questions for Opus session (highest priority)
1. Static RP vs rolling RP (monthly covariance update)?
2. Hybrid (50/50 blend of manual + RP)? Captures IS stability + OOS resilience?
3. Should RP replace manual as Tier 1 production strategy?
4. RP + overlay combination: complementary or conflicting?
5. RP-informed bounds for DE tactical layer — is a hybrid approach worth designing?

---

Three parallel workstreams:
- **Workstream A:** Engineering fixes (code correctness, RP implementation)
- **Workstream B:** Market experiments (demand validation, content)
- **Workstream C:** ~~Optimiser validation~~ **CLOSED — pivoted to risk parity**

---

## Blocking experiments — Workstream B (market)

### B0: ALLW head-to-head comparison ← #1 PRIORITY
Fetch ALLW daily prices. Run 6asset_tip_gsg and 7asset_tip_gsg_vnq over
the same period. Compare CAGR, MaxDD, Calmar, Ulcer, Sortino. Compute
fee-adjusted comparison (ALLW 0.85% vs DIY ~0.12%).

**Deliverables:** metrics table, drawdown overlay chart, fee drag chart.
Data feeds blog post (C1) and landing page (C3).

### B1: Iran war stress test (with ALLW)
full_backtest 2025-01-01 to 2026-03-21. All four candidates + 60/40 +
SPY + ALLW. Break out per-asset contribution.

### B2: GBP/EUR-adjusted backtest
Fetch GBPUSD=X and EURUSD=X. Convert portfolio values. Re-run stats.
Required before non-US product claims.

---

## Blocking experiments — Workstream A (engineering)

### B3: Weekly vs monthly max drawdown comparison
Run DATA_FREQUENCY = "W" for all four candidates. If gap > 3pp, switch
to daily MDD for all reporting.

### B4: Threshold-based rebalancing comparison
After fixing backtest engine: threshold=0.0 vs threshold=0.05. Compare
Calmar, MaxDD, trade count.

---

## Blocking experiments — Workstream C (optimiser repair) — ✅ CLOSED

### ~~C0: Implement the four optimiser fixes~~ ✅ DONE

All four changes are in `optimiser.py` (plus config for bounds):

**Fix 1: Per-asset bounds** (15 min)
Replace uniform `[OPT_MIN_WEIGHT, OPT_MAX_WEIGHT]` with per-asset
`ASSET_BOUNDS` dict. For 6asset_tip_gsg:
```
SPY: [0.05, 0.20], QQQ: [0.05, 0.20], TLT: [0.20, 0.45],
TIP: [0.05, 0.20], GLD: [0.05, 0.20], GSG: [0.05, 0.15]
```
Pass as `bounds` to `differential_evolution()`.

**Fix 2: Martin ratio objective** (30 min)
In `_score_allocation()`, replace:
  `return -compute_calmar(cagr, mdd)`
with:
  `ulcer = compute_ulcer_index(series)`
  `return -(cagr / ulcer) if ulcer > 0 else -cagr`

This gives a smooth, all-data-point objective instead of single-worst-point.
Import `compute_ulcer_index` from backtest.

**Fix 3: Remove normalisation** (1 hour)
Replace `w_norm = w / w.sum()` in `de_objective()` with N-1
parameterisation: optimise first N-1 weights, compute Nth as
`max(min_bound, 1.0 - sum(w[:N-1]))`. Adjust DE bounds accordingly.

**Fix 4: Project inside loop** (30 min)
Move `_project_weights()` call inside `de_objective()`, before scoring.
Every candidate DE evaluates is already feasible. Remove post-convergence
projection.

### ~~C1–C3: Walk-forward validation experiments~~ ✅ SUPERSEDED

Phase 9 ran 26 experiments with the fully fixed optimiser. All failed.
WF median is not a reliable predictor of OOS performance (retracted).
Split2022 made results worse. Gate 1 is closed.

**The correct next workstream is risk parity, not further DE experimentation.**

---

## Paper trading setup (after blocking experiments)

### G1-G4: Same as before (4 strategies, live tickers, 3-month success criteria)

**Change:** Track ALLW as 5th benchmark. If C2 passes, add DE-optimised
variant of 6asset_tip_gsg as 5th paper trading strategy.

---

## Content experiments (Track B, ongoing)

### D1: Blog post "ALLW vs DIY: Is the 0.85% worth it?"
### D2: Blog post "How our portfolio handled the Iran war"
### D3: Landing page
### D4: Reddit regime heatmap post

---

## Robustness experiments (parallel with paper trading)

### R1: Regime-conditional performance (4 hours)
### R2: Bootstrap confidence intervals (3 hours)
### R3: Drawdown decomposition (2 hours)
### R4: Complete 3-window validation for Tier 2/3 (2 hours)
### R5: Fee drag projection (1 hour)
### R6: VNQ + DJP combined universe

---

## Decision gates

- **Gate 1** (after B0): Can we compete with ALLW on risk-adjusted terms?
- **Gate 2** (after D1+D3): Is there organic demand? (>100 signups)
- **Gate 3** (after 3 months paper trading): Model matches reality? (<1% tracking error)
- **Gate 4** (after C2): Does fixed DE beat/match manual? → Pro tier feature decision.

---

## Principles (updated Phase 10D — 2026-03-25)

1. GSG/DJP non-negotiable. TIP essential. QQQ non-negotiable.
2. ~~Manual allocation beats DE optimiser.~~ **CONFIRMED PERMANENTLY.**
   26 experiments with the fully fixed DE. Zero beat manual 0.403. Gate 1 closed.
3. Three OOS windows required for production promotion.
4. ~~WF median > 0.6 = HIGH reliability.~~ **RETRACTED (Phase 9).**
5. ~~2022 in IS training improves robustness.~~ **RETRACTED (Phase 9).**
6. Compare strategies on same window length only.
7. ALLW is the primary competitive benchmark, not 60/40.
8. Every experiment should produce both research data AND content.
9. Validate demand before building product.
10. GBP/EUR adjustment required for non-US product claims.
11. Cannot use "All Weather" in the product name.
12. **Use Martin ratio (CAGR/Ulcer) for reporting. Calmar as primary metric.**
13. **Per-asset bounds, not uniform bounds.**
14. **Risk parity is the correct optimisation framework.**
    RP 5yr: +27% OOS Calmar on 6asset. Flat on 7asset (multiple over-concentrations cancel).
    Next: rolling RP, hybrid blend, Opus session to design production implementation.

---

## Rejected experiments (do not repeat)

| Experiment | OOS Calmar | Reason |
|-----------|-----------|--------|
| 8asset_agg_replaces_qqq | 0.198 | Only result below 60/40 |
| 8asset_lqd_replaces_shy | 0.291 | LQD fails in rate shocks |
| 7asset_vnq_reits (no TIP+GSG) | 0.287 | REITs without inflation anchor |
| 7asset_6tip_plus_ief | 0.310 | Duration overlap |
| 6asset_duration_ladder | 0.329 | Worst overfitting |
| 6asset_intl_equity (EFA) | 0.300 | Lowest CAGR, no diversification |