# Experiment Plan — Phase 3 (post-optimiser-analysis)

## Status: Phase 2 complete. ALLW discovered. Optimiser bugs identified.

Phase 2 validated four strategies. Market validation revealed ALLW as the
primary competitor. Optimiser root-cause analysis revealed that DE's
"failure" to beat manual allocation was caused by four setup bugs, not
by the strategy or the optimiser itself. Phase 3 adds optimiser repair
as a high-priority engineering workstream.

Three parallel workstreams:
- **Workstream A:** Engineering fixes (code correctness, optimiser repair)
- **Workstream B:** Market experiments (demand validation, content)
- **Workstream C:** Optimiser validation (does DE add value once fixed?)

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

## Blocking experiments — Workstream C (optimiser repair)

### C0: Implement the four optimiser fixes

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

### C1: Validate fixes with single-window optimisation
**Purpose:** Quick sanity check that fixed DE produces sensible weights.

**Method:** Run `optimise` mode on IS period (2006-2020) for 6asset_tip_gsg
with the four fixes applied. Compare optimised Calmar to manual Calmar.
The optimised result should be equal or better (since the manual allocation
is now inside the search space).

**If optimised IS Calmar < manual IS Calmar:** Something is wrong with
the fix implementation. Debug before proceeding to C2.

### C2: Re-run walk-forward with fixed optimiser ← KEY EXPERIMENT
**Purpose:** Determine whether DE adds value once the setup bugs are fixed.
This is the experiment that decides whether "optimised allocation" becomes
a product feature or stays diagnostic-only.

**Method:** Run walk-forward for 6asset_tip_gsg with all four fixes.
Use the same WF parameters: train 5yr, test 2yr, step 2yr.

**Decision rule:**
| WF median | opt beats orig | Verdict |
|-----------|---------------|---------|
| ≥ 0.6 | ≥ 50% windows | "Optimised" is a validated product feature |
| 0.3-0.6 | any | Improved but not production-grade. Keep as diagnostic. |
| < 0.3 | any | DE genuinely doesn't generalise. Finding confirmed properly. |

If the first row is achieved, add "DE-optimised allocation" as a Pro tier
feature in the product. This is a genuine differentiator — ALLW uses
Bridgewater's proprietary model; we offer transparent, inspectable
optimisation the user can verify.

### C3: Walk-forward with Martin ratio for all four strategies
**Purpose:** If C2 passes for 6asset_tip_gsg, validate across all four
paper trading candidates.

**Method:** Run WF for 7asset_tip_gsg_vnq, 7asset_tip_djp, 5asset_dalio
with the same fixed optimiser. Compare WF metrics to manual baseline.

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

## Principles (updated post-optimiser-analysis)

1. GSG/DJP non-negotiable. TIP essential. QQQ non-negotiable.
2. ~~Manual allocation beats DE optimiser.~~ **RETRACTED.** Previous DE
   "failure" was caused by setup bugs (wrong bounds, bad objective,
   normalisation distortion, post-hoc projection). Pending re-evaluation
   after fixes (experiment C2). Until C2 is complete, use manual
   allocation for production but do not treat DE as permanently broken.
3. Three OOS windows required for production promotion.
4. WF median > 0.6 = HIGH reliability. Use median, not mean.
5. 2022-style shocks recur ~once/decade. Stress test mandatory.
6. Compare strategies on same window length only.
7. ALLW is the primary competitive benchmark, not 60/40.
8. Every experiment should produce both research data AND content.
9. Validate demand before building product.
10. GBP/EUR adjustment required for non-US product claims.
11. Cannot use "All Weather" in the product name.
12. **Use Martin ratio (CAGR/Ulcer) for optimisation, Calmar for reporting.**
    Calmar is a useful summary metric but a terrible optimisation objective.
13. **Per-asset bounds, not uniform bounds.** Different assets have different
    roles and require different weight ranges.

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