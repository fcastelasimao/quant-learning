# Research Log — All Weather Portfolio Engine

A running record of decisions, experiments, findings, and open questions.
Updated after each session.

---

## Project Goal

Build a backtesting, optimisation, and validation engine for risk-balanced
portfolio strategies. Primary objective: capital preservation with competitive
risk-adjusted returns. Primary metric: Calmar ratio (CAGR / max drawdown).

Longer-term: paper trade validated strategies, then deploy live capital,
then sell as a product — positioned as a transparent, unlevered, customisable
alternative to Bridgewater's ALLW ETF for self-directed global investors.

---

## Phase 1 — Initial Architecture and 5-Asset Experiments

### Decision: Replace LQD with TIP
LQD and TLT correlated in rate shocks. TIP provides inflation-linked
mechanism instead. Significant 2022 stress test improvement.

### Decision: Calmar ratio as primary optimisation objective
Sharpe penalises upside volatility. Calmar captures return per unit of
maximum loss. (Note: later revised — Calmar is good for reporting but bad
for optimisation. See Phase 8.)

### Finding: 2022 is the structural constraint
Simultaneous bond/equity/gold losses. Universe expansion (commodities) is
the right lever, not weight tuning.

---

## Phase 2 — Architecture Improvements

### Decision: Strict IS/OOS methodology (2026-03-19)
Three-date boundary. All optimisation confined to IS. Enforced architecturally.

### Decision: Total return prices (2026-03-20)
TLT price return: -5%. Total return: +79%. Made explicit via PRICING_MODEL.

### Decision: Extended metrics (2026-03-19)
Added Avg Drawdown, Max DD Duration, Avg Recovery, Ulcer Index, Sortino.

### Decision: Transaction cost standard (2026-03-20)
0.001 standard. Robust at all levels through 0.5%.

---

## Phase 3 — 8-Asset Experiments

### Finding: DE optimiser does not reliably beat manual allocation (2026-03-19)
Optimiser beat manual in only 1-3 of 4 walk-forward windows.
**⚠️ RETRACTED in Phase 8:** This finding was caused by setup bugs in the
optimiser, not by the optimiser or strategy being fundamentally limited.
See Phase 8 for full root-cause analysis.

### Finding: 8-asset manual allocation validated then demoted (2026-03-19)
OOS Calmar 0.441. Later demoted: 29% Calmar drop in 2022 stress.

---

## Phase 4 — Infrastructure (2026-03-20)

run_experiment.py, curate_master_log.py, results_dashboard.py.

---

## Phase 5 — 23-Universe Experiment Study (2026-03-20 to 2026-03-21)

### Group A: 6asset_tip_gsg — 3-window validation
OOS Calmar: 0.476 / 0.473 / 0.405. All beat 60/40. MaxDD stable at ~-20%.

### Group B: New asset class experiments
- 7asset_tip_gsg_vnq: OOS 0.465. TIP+GSG anchor neutralises VNQ rate risk.
- 8asset_agg_replaces_qqq: IS 0.626 → OOS 0.198. Only failure below 60/40.

### Group E: 7-asset robustness
7asset_tip_gsg_vnq: Calmar range 0.042 across 3 windows. Most regime-stable.

### Group F: 2022 stress tests
8asset_manual worst (-29% Calmar drop). 7asset_tip_gsg_vnq best (-7.7%).

### WF finding: 2022 in training improves robustness

---

## Phase 6 — Critical Review (2026-03-21)

### Identified gaps (code)
- Rebalancing mismatch (backtest unconditional vs live 5% threshold)
- Monthly max drawdown understates true drawdowns
- No data validation after yfinance download
- Benchmark computation waste in optimiser
- config.py defaults to demoted allocation
- _Tee doesn't capture stderr

### Identified gaps (strategy)
- No GBP/EUR adjustment (largest unmodelled risk)
- Sharpe/Sortino assume Rf = 0
- No regime-conditional performance analysis
- US-only equity exposure
- No confidence intervals

### Identified gaps (macro)
- Iran war (Feb-Mar 2026): oil $100-112, 20% supply disrupted, Fed paused

---

## Phase 7 — ALLW Discovery and Market Validation (2026-03-21)

### Critical finding: Bridgewater launched the ALLW ETF
March 2025. $1.17B AUM. 0.85% expense. ~2x leverage on bonds. 99% turnover.

### Impact
- ALLW is now the primary competitor and benchmark, not 60/40
- Cannot use "All Weather" commercially (trademark)
- Product positioning: transparent, unlevered, customisable, cheaper

### ALLW's weakness in current macro
~2x leveraged bonds are a headwind with Fed paused and inflation rising.
Our unlevered strategies with commodity exposure should outperform during
exactly the type of crisis that attracts defensive investors.

### Market validation
$10-14B robo-advisory market growing 25-30% CAGR. Four customer segments
identified. Pricing must be cheaper than ALLW ($850/yr on $100k).
Content-led GTM: ALLW comparison blog post tests demand.

---

## Phase 8 — Optimiser Root-Cause Analysis (2026-03-21)

### Context
The Phase 3 finding — "DE optimiser does not reliably beat manual
allocation" — was accepted as a fundamental result: manual allocation
wins, DE is diagnostic only. This conclusion was embedded into every
document and principle in the project. Phase 8 challenged that conclusion.

### Root-cause: four compounding setup bugs

**Bug 1: Search space excludes the best allocations.**

Every validated strategy has TLT at 25-40%:
- 6asset_tip_gsg: TLT = 30% (exceeds 25% cap by 5pp)
- 7asset_tip_djp: TLT = 27% (exceeds by 2pp)
- 5asset_dalio: TLT = 40%, SPY = 30% (both exceed by 15pp and 5pp)

But `OPT_MAX_WEIGHT = 0.25`. DE is restricted to a box that does not
contain any of the allocations it's being compared against. It cannot
find them because it's not allowed to look there. The walk-forward
"failure" was guaranteed by construction.

**Bug 2: Normalisation distorts the search space.**

`de_objective()` normalises weights: `w_norm = w / w.sum()`. This creates
a non-linear mapping from DE's raw search space to the actual objective
landscape. Two points that DE considers "nearby" can produce very
different normalised weights. At the extremes (all weights at min or all
at max), normalisation collapses to equal weight regardless of input.
DE can't learn the gradient correctly in this warped space.

**Bug 3: Calmar is a bad optimisation objective.**

Calmar = CAGR / |MaxDD|. MaxDD depends on a SINGLE worst data point
in the entire time series. This creates:
- **Discontinuity:** Moving one weight by 0.001 can change which month
  is the worst, causing a discrete jump in the objective.
- **Overfitting:** The optimiser finds weights that minimise one specific
  event's impact. When the test period has a different worst event, the
  weights fail.
- **Flat regions:** In most of the weight space, MaxDD doesn't change
  (same month is still worst), so the objective moves only via CAGR.
  Then suddenly a threshold is crossed and the objective jumps.

The Ulcer Index captures the same risk concept (drawdown depth × duration)
but uses ALL time steps, producing a smooth landscape. The Martin ratio
(CAGR / Ulcer) is the drop-in replacement.

**Bug 4: Post-convergence projection.**

After DE converges, `_project_weights()` clips and renormalises the
result. The projected weights can differ significantly from what DE
optimised. DE never explored the neighbourhood of the final output.

### Interaction between bugs

The bugs compound each other:
- Bug 1 (wrong bounds) prevents finding good allocations.
- Bug 2 (normalisation) prevents DE from navigating efficiently even
  within the (too-small) feasible region.
- Bug 3 (Calmar objective) makes the landscape too rugged for any
  optimiser to navigate reliably.
- Bug 4 (post-hoc projection) distorts whatever result DE manages to find.

With all four active simultaneously, DE's apparent "failure" is expected.
The surprise would have been if it worked.

### Specified fixes

| Fix | Change | Effort | Impact |
|-----|--------|--------|--------|
| Per-asset bounds | Replace uniform [0.05, 0.25] with per-asset dict | 15 min | Critical — opens the search space |
| Martin ratio objective | Replace -Calmar with -CAGR/Ulcer in `_score_allocation()` | 30 min | Critical — smooth landscape |
| Remove normalisation | N-1 parameterisation or Dirichlet in `de_objective()` | 1 hour | High — correct gradient signal |
| Project inside loop | Move `_project_weights()` inside `de_objective()` | 30 min | Medium — ensures feasibility |

### Implication: "manual beats DE" is retracted

The principle "manual allocation beats DE optimiser; DE is diagnostic only"
is retracted as of Phase 8. It was based on experimental evidence from a
fundamentally flawed setup. The correct statement is:

**"Under the previous configuration (uniform bounds, Calmar objective,
normalised search, post-hoc projection), DE did not beat manual allocation.
Four setup bugs were identified that individually and jointly explain this
result. Pending re-evaluation after fixes are implemented (experiment C2)."**

If the fixed optimiser passes walk-forward validation, this reopens
"optimised allocation" as a product feature — potentially the most
powerful Pro tier differentiator, since it would be transparent optimisation
the user can inspect and verify, unlike ALLW's opaque Bridgewater model.

### Principle updates

- ~~"Manual allocation beats DE optimiser."~~ → **RETRACTED.** Pending
  re-evaluation after C2.
- **NEW:** "Use Martin ratio (CAGR/Ulcer) for optimisation, Calmar for
  reporting." Calmar is a useful summary metric but a terrible
  optimisation objective due to single-point dependency.
- **NEW:** "Use per-asset bounds, not uniform bounds." Different assets
  have different roles and need different weight ranges.

---

## Consolidated Principles (updated Phase 8)

1. **GSG/DJP is non-negotiable.**
2. **TIP inflation bonds are essential.**
3. **QQQ (growth equity) is non-negotiable.**
4. ~~Manual allocation beats DE optimiser.~~ **RETRACTED (Phase 8).** Pending
   re-evaluation. Use manual for production until C2 validates fixed DE.
5. **Three OOS windows required for production promotion.**
6. **WF median > 0.6 = HIGH reliability. Use median, not mean.**
7. **Including 2022 in IS training improves WF reliability.**
8. **Calmar is biased against short OOS windows.**
9. **2022-style shocks recur ~once/decade.**
10. **TQQQ is incompatible with All Weather.**
11. **ALLW is the primary competitive benchmark, not 60/40.**
12. **Every experiment should produce both research data AND content.**
13. **Validate demand before building product.**
14. **Cannot use "All Weather" in the product name.**
15. **Use Martin ratio for optimisation, Calmar for reporting.**
16. **Use per-asset bounds, not uniform bounds.**

---

## Open Questions

**Active (highest priority):**
- Do our strategies beat ALLW on risk-adjusted terms? (B0)
- Does the fixed optimiser pass walk-forward validation? (C2) — if yes,
  this is a major product feature
- Is there paying demand for a DIY ALLW alternative? (content test)
- Brand name?

**Active (engineering):**
- Weekly vs monthly MDD gap? (B3)
- GBP/EUR Calmar impact? (B2)
- Threshold rebalancing tracking error? (B4)
- Bootstrap CI width? (R2)

**Answered:**
- ✅ Robust to transaction costs
- ✅ ETF substitutions confirmed
- ✅ All four strategies validated OOS
- ✅ 8asset_manual demoted
- ✅ REITs work only with TIP+GSG anchor
- ✅ ALLW exists ($1B+ AUM, 0.85%, ~2x leverage)
- ✅ DE "failure" was setup bugs, not fundamental limitation (Phase 8)

---

## Current Status (2026-03-21)

**Paper trading candidates:** 4 strategies (6asset_tip_gsg, 7asset_tip_gsg_vnq,
7asset_tip_djp, 5asset_dalio)

**Primary competitive benchmark:** ALLW ETF

**Blocking items:** Config fix, optimiser repair (4 bugs), ALLW comparison,
Iran stress test, GBP backtest, optimiser WF re-validation

**Master log:** 229 rows