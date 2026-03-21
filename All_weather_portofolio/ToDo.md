# All Weather Portfolio — TODO & Action Plan

Last updated: 2026-03-21 (post-optimiser-analysis)

Two parallel tracks run simultaneously:
- **Track A (Engineering):** Code fixes, optimiser repair, experiments
- **Track B (Market):** Demand validation, content, positioning

Priority key:
🔴 = Blocking (this week). 🟡 = Before live capital. 🟢 = Before product launch. 🔵 = Product roadmap.

---

## 🔴 TRACK B — Market validation (this week, in parallel with Track A)

### B0.1: ALLW head-to-head comparison ⏱ 3 hours
Bridgewater's ALLW ETF ($1B+ AUM, 0.85% expense, ~2x leverage) is now
the customer's real alternative. Fetch ALLW prices, compare to 6asset_tip_gsg
over the same period. This is both research and the foundation of all marketing.

### B0.2: Write the ALLW comparison blog post ⏱ 4 hours
"ALLW vs DIY All Weather: Is the 0.85% worth it?"
Demand validation test. Success metric: >50 Reddit upvotes, >10 signups.

### B0.3: Choose a brand name ⏱ decision
Cannot use "All Weather" (Bridgewater trademark). Pick a name, check domain.

### B0.4: Build a landing page ⏱ 3 hours
Single page: ALLW comparison chart, value prop, email signup. Vercel/Netlify.
Success metric: 100 signups in 4 weeks.

### B0.5: User research interviews ⏱ ongoing
10-15 conversations with DIY All Weather implementors. What do they use?
What frustrates them? Would they pay? Do they know about ALLW?

---

## 🔴 TRACK A — Engineering fixes (this week)

### A0.1: Update config.py to validated strategy ⏱ 10 min
Replace 8-asset demoted allocation with 6asset_tip_gsg.

### A0.2: Fix rebalancing mismatch ⏱ 2 hours
Add `rebalance_threshold` parameter to `run_backtest()`. Default to 0.0.
Run comparison: threshold=0.0 vs threshold=0.05.

### A0.3: Fix max drawdown to use daily data ⏱ 1.5 hours
Add `compute_max_drawdown_daily()`. Report both monthly and daily MDD.

### A0.4: Add data validation after yfinance download ⏱ 30 min
Assert no all-NaN columns, no returns > ±30%, no negative prices.

### A0.5: Fix _Tee to capture stderr ⏱ 5 min

### A0.6: Create strategies.json registry ⏱ 45 min
Already templated. Add `load_strategy()` to config.py.

### A0.7: Fix optimiser — per-asset bounds ⏱ 15 min ← NEW
**The problem:** OPT_MAX_WEIGHT = 0.25 globally, but every validated
strategy has TLT at 25-40%. The optimiser cannot search where the good
allocations live. This is the primary reason DE "fails."

**Fix:** Replace `OPT_MIN_WEIGHT / OPT_MAX_WEIGHT` with `ASSET_BOUNDS`:
```python
ASSET_BOUNDS = {
    "SPY": (0.05, 0.20),
    "QQQ": (0.05, 0.20),
    "TLT": (0.20, 0.45),
    "TIP": (0.05, 0.20),
    "GLD": (0.05, 0.20),
    "GSG": (0.05, 0.15),
}
```
Pass per-asset bounds to DE and random search. The asset class caps
still prevent absurd group concentrations.

### A0.8: Fix optimiser — switch to Martin ratio objective ⏱ 30 min ← NEW
**The problem:** Calmar = CAGR / |MaxDD| depends on a single worst data
point. The landscape is discontinuous: tiny weight changes shift which
month is worst, causing objective jumps. DE overfits to one event that
won't recur in the test period.

**Fix:** Replace `-compute_calmar(cagr, mdd)` with `-cagr / ulcer_index`
(the Martin ratio). Ulcer Index uses ALL drawdown data across ALL time
steps. It's smooth, penalises both depth and duration, and captures
the same risk-aversion intent without single-point fragility.

One-line change in `_score_allocation()`, plus add `compute_ulcer_index`
to the import. Calmar remains a reporting metric — it just stops being
the optimisation objective.

### A0.9: Fix optimiser — remove normalisation from DE ⏱ 1 hour ← NEW
**The problem:** `de_objective()` does `w_norm = w / w.sum()` which warps
the search space. Two nearby raw-weight points can map to very different
normalised weights. DE can't learn the gradient correctly.

**Fix:** Use N-1 parameterisation: optimise weights for first N-1 assets,
compute the Nth as `1.0 - sum(w[:N-1])`. Enforce `w[N-1] >= min_weight`
via penalty. Or use Dirichlet parameterisation. Either way, the mapping
from DE's internal space to actual weights becomes linear.

### A0.10: Fix optimiser — project inside DE loop ⏱ 30 min ← NEW
**The problem:** After DE converges, `_project_weights()` clips and
renormalises the result, moving it away from what DE actually optimised.
DE never explored the neighbourhood of the projected weights.

**Fix:** Move `_project_weights()` inside `de_objective()`, before
`_score_allocation()`. Every candidate DE evaluates is already feasible.
DE learns the actual landscape, not a distorted version.

---

## 🔴 Critical experiments (this week)

### E1: ALLW comparison (= B0.1)
Head-to-head: 6asset_tip_gsg vs ALLW. See B0.1 above.

### E2: Iran war stress test ⏱ 1 hour
full_backtest 2025-01-01 to 2026-03-21. All candidates + 60/40 + SPY + ALLW.

### E3: GBP-adjusted backtest ⏱ 2 hours
Convert all values to GBP and EUR. Required for non-US product claims.

### E4: Weekly vs monthly max drawdown comparison ⏱ 1 hour

### E5: Re-run walk-forward with fixed optimiser ⏱ 2 hours ← NEW
**Purpose:** Test whether the optimiser fixes (A0.7-A0.10) change the
walk-forward verdict. If DE now beats or matches manual allocation,
"optimised weights" becomes a Pro tier product feature.

**Method:** Run walk-forward for 6asset_tip_gsg with:
- Per-asset bounds (A0.7)
- Martin ratio objective (A0.8)
- Fixed parameterisation (A0.9)
- In-loop projection (A0.10)

Compare WF overfit ratio and "opt beats original" count to the old results.

**Decision rule:**
- If WF median ≥ 0.6 AND opt beats original in ≥ 50% of windows →
  "Optimised allocation" becomes a validated product feature.
- If WF median 0.3-0.6 → optimiser improves but still not production-grade.
  Use as diagnostic only (current status, but now evidence-based).
- If WF median < 0.3 → optimiser genuinely doesn't add value beyond
  manual allocation. Confirmed finding, not a setup artefact.

---

## 🟡 Before live capital (next 2 weeks)

### A1.1: Add risk-free rate to Sharpe and Sortino ⏱ 30 min
### A1.2: Strip benchmark from _score_allocation() ⏱ 1.5 hours
### A1.3: Add unit tests for newer stat functions ⏱ 1 hour
### A1.4: Resolve MiFID II broker access ⏱ research task
### A1.5: Fractional share support in rebalancing ⏱ 1 hour
### A1.6: Add ALLW as permanent 5th benchmark ⏱ 1 hour

---

## 🟡 Analysis & robustness (next month)

### A2.1: Regime-conditional performance ⏱ 4 hours
### A2.2: Bootstrap confidence intervals ⏱ 3 hours
### A2.3: Drawdown decomposition ⏱ 2 hours
### A2.4: Inflation-adjusted returns ⏱ 1 hour
### A2.5: Correlation heatmap per regime ⏱ 2 hours
### A2.6: Complete 3-window validation for Tier 2/3 ⏱ 2 hours
### A2.7: Fee drag visualisation (ALLW vs DIY) ⏱ 1 hour

---

## 🟢 Before product launch

### P4.1: Visualisation suite (see visualisation_strategy_v2.md)
### P4.2: Trademark and brand name
### P4.3: FCA/regulatory compliance review
### P4.4: Competitive positioning document (center on ALLW)
### P4.5: Minimum portfolio size analysis

---

## 🔵 Product roadmap

### P5.1: Web interface (FastAPI + React)
### P5.2: User accounts and data persistence
### P5.3: Monthly rebalancing reminders
### P5.4: Trend-following overlay (Pro tier)
### P5.5: Referral mechanism
### P5.6: Advisor tier
### P5.7: Optimised allocation (Pro tier) ← NEW
If E5 validates the fixed optimiser, offer DE-tuned weights per risk
tier as a Pro feature. Updated periodically as new regime data accumulates.
This is a genuine differentiation — ALLW uses Bridgewater's proprietary
model, we offer transparent, validated optimisation the user can inspect.

---

## ✅ Completed

- [x] Modular 9-file architecture, StrategyStats, 9 stat helpers
- [x] Four optimisation methods with asset class caps
- [x] Walk-forward validation, IS/OOS enforcement, 60/40 benchmark
- [x] Excel master log (229 rows), transaction cost + tax modelling
- [x] Total return pricing, GitHub repo, 53 passing tests
- [x] run_experiment.py batch runner (23 universes)
- [x] All four strategies validated across multiple OOS windows
- [x] 8asset_manual demoted, ETF substitutions confirmed
- [x] Market validation completed (ALLW discovery, competitive landscape)
- [x] strategies.json template created
- [x] Optimiser root-cause analysis: 4 bugs identified, fixes specified

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