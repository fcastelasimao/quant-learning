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
**⚠️ PARTIALLY REINSTATED in Phase 9:** After fixing the four bugs, the
optimiser still fails to beat manual allocation for the 2020-split OOS window.
The failure is now attributed to regime mismatch (IS 2006-2020 does not contain
a rate shock), not setup bugs. See Phase 9 for full analysis.

### Finding: 8-asset manual allocation validated then demoted (2026-03-19)
OOS Calmar 0.441. Later demoted: 29% Calmar drop in 2022 stress.

---

## Phase 4 — Infrastructure (2026-03-20)

run_experiment.py, curate_master_log.py, results_dashboard.py.

---

## Phase 5 — 23-Universe Experiment Study (2026-03-20 to 2026-03-21)

### Group A: 6asset_tip_gsg — 3-window validation
OOS Calmar: 0.476 / 0.473 / 0.405. All beat 60/40. MaxDD stable at ~-20%.
Note: these results used manual allocation and the broken optimiser.
Re-validation with fixed DE is Phase 9.

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

### Identified gaps (code) — all now fixed
- ✅ config.py defaults to demoted allocation → fixed to 6asset_tip_gsg
- ✅ _Tee doesn't capture stderr → fixed
- ✅ No data validation after yfinance download → fixed
- ✅ Optimiser bugs (4 identified) → fixed
- ⏳ Rebalancing mismatch (backtest unconditional vs live 5% threshold)
- ⏳ Monthly max drawdown understates true drawdowns

### Identified gaps (strategy) — open
- No GBP/EUR adjustment (largest unmodelled risk for non-US customers)
- Sharpe/Sortino assume Rf = 0 (Fed at 3.5-3.75%)
- No regime-conditional performance analysis
- US-only equity exposure
- No confidence intervals

### Identified gaps (macro)
- Iran war (Feb 2026 ongoing): oil $108-112, 20% supply disrupted, Fed paused
- Live stagflationary stress test — commodity allocations performing well

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
Unlevered strategies with commodity exposure outperform during exactly
the type of crisis that attracts defensive investors.

### Market validation
$10-14B robo-advisory market growing 25-30% CAGR. Four customer segments
identified. Pricing must be cheaper than ALLW ($850/yr on $100k).
Content-led GTM: ALLW comparison blog post tests demand.

---

## Phase 8 — Optimiser Root-Cause Analysis (2026-03-21)

### Four compounding setup bugs identified and fixed

**Bug 1 (fixed): Search space excluded good allocations.**
OPT_MAX_WEIGHT = 0.25 globally, but validated strategies need TLT at 30-40%.
Fix: per-asset ASSET_BOUNDS dict with TLT: (0.20, 0.45).

**Bug 2 (fixed): Normalisation distorted DE search space.**
w / w.sum() in de_objective warped the gradient signal.
Fix: projection inside de_objective, no normalisation.

**Bug 3 (fixed): Calmar is a bad optimisation objective.**
Single worst data point → discontinuous landscape → overfitting to one event.
Fix: Martin ratio (CAGR / Ulcer Index) as DE objective. Calmar retained
for reporting only.

**Bug 4 (fixed): Post-convergence projection distorted result.**
_project_weights() called after DE converged, moving result to unexplored space.
Fix: projection moved inside de_objective loop.

### Additional fix: master log column order and Martin ratio
- METRIC_COLS reordered: Calmar and Martin first, then CAGR/risk, then detail
- Martin ratio added to StrategyStats, compute_stats(), master log
- META_COLS reordered: config detail columns moved right
- Master log auto-archives on first run (master_log_archive_phase1.xlsx)

---

## Phase 9 — Full Re-run with Fixed Optimiser (2026-03-22)

### Setup
33 experiments run with fixed DE (per-asset bounds, Martin objective,
in-loop projection, no normalisation). popsize=15, maxiter=400.
OOS window: 2020-01-01 to 2026-03-22 (dynamic BACKTEST_END = today).

### Critical finding: DE still fails to beat manual for 2020-split OOS

Every experiment shows the same pattern without exception:
- IS optimised Calmar: 0.46–0.76 (optimiser improves IS reliably)
- OOS optimised Calmar: 0.16–0.35 (all collapse in OOS)
- Manual allocation OOS: 0.403 (beats every optimised result)

The optimiser is working correctly as a mathematical optimiser. It finds
the true global maximum of the Martin ratio on the 2006-2020 IS data.
That maximum is a bond-heavy allocation (TLT at or near 40% cap) because
every stress event in 2006-2020 saw rates fall and bonds rally.
The 2022 rate shock (TLT -30%) is in OOS, not IS. The optimiser was never
penalised for bond concentration during training.

### This is a data problem, not an optimiser problem

The four bugs were real and are fixed. But fixing them revealed a deeper
structural issue: the IS period 2006-2020 is a single macro regime
(falling rates, bonds rally in every crisis). The OOS period 2020-2026
contains a regime change. No optimiser can generalise across regime
changes it has never seen.

### Principle update: "manual beats DE" partially reinstated

**Revised principle:** "For the 2020-split OOS window, manual allocation
beats DE-optimised weights. The fix to include 2022 in IS training
(split2022 experiments) is the correct response. Pending split2022 OOS
results to determine whether DE adds value when trained on a regime-diverse
IS period."

### Notable results from Phase 9 run (2020-split OOS)

| Strategy | IS Manual | IS Opt | OOS Opt | Note |
|---|---|---|---|---|
| 6asset_tip_gsg_manual | — | — | 0.403 | Manual baseline, best result |
| 7asset_no_iwd | 0.287 | 0.584 | 0.349 | Best optimised OOS |
| 8asset_efa_replaces_iwd | 0.309 | 0.527 | 0.333 | EFA adds modest value |
| 7asset_no_gsg | 0.591 | 0.650 | 0.318 | No commodities, suspicious |
| 6asset_intl_equity | 0.233 | 0.461 | 0.308 | Previously rejected, still weak |
| 6asset_duration_ladder | 0.623 | 0.762 | 0.253 | High IS → worst OOS collapse |
| 6asset_tip_gsg | 0.367 | 0.582 | 0.218 | Optimised weights fail OOS |
| 8asset_tip_replaces_shy | 0.333 | 0.700 | 0.161 | Worst result |

### Key observation: IS optimised Calmar is inversely correlated with OOS

Higher IS optimisation = worse OOS generalisation. The duration_ladder
achieves the highest IS Calmar (0.762) and the second-worst OOS (0.253).
This is a textbook overfit signature. More degrees of freedom + richer
IS landscape = more overfit risk.

### ETF substitution finding (backtest vs live ETFs)

Compared backtest ETFs (SPY/QQQ/GLD/GSG) vs live ETFs (IVV/QQQM/GLDM/GSG)
over the ALLW window (March 2025 to March 2026). Key findings:
- With GSG in both: gap is negligible (+0.04% to +0.07% CAGR, live slightly
  better due to lower expense ratios on GLDM/QQQM)
- PDBC (live commodity ETF) caused ~0.8% CAGR drag vs GSG in this window
- Cause: PDBC's active management diverged from GSG during the Iran war oil
  spike. GSG is ~70% energy and tracked the crude price spike directly.
- Decision: use GSG for backtesting and ALLW comparison. Recommend GSG or
  PDBC to customers with clear disclosure of the difference.

### ALLW live comparison (March 2025 to March 2026, daily data)

| Strategy | CAGR | Max DD | Calmar | Vol | Worst Month |
|---|---|---|---|---|---|
| 6asset_tip_gsg (gross) | 17.47% | -6.70% | 2.608 | 10.07% | -3.97% |
| 6asset_tip_gsg (fee-adj) | 17.33% | -6.70% | 2.586 | 10.07% | -3.97% |
| ALLW (gross) | 16.62% | -8.78% | 1.894 | 12.60% | -6.97% |
| ALLW (fee-adj 0.85%) | 15.64% | -8.79% | 1.779 | 12.60% | -7.01% |
| SPY | 14.28% | -13.72% | 1.041 | 18.99% | -5.20% |
| 60/40 | 8.28% | -8.79% | 0.942 | 12.59% | -5.19% |

6asset_tip_gsg beats ALLW on every risk-adjusted metric. The Iran war
environment (oil shock, paused Fed, inflation) favours commodity-weighted
unlevered strategies over ALLW's ~2x leveraged bond exposure.
Caveat: 12 months of data in a favourable macro environment — not a
general claim of superiority.

### New scripts added

- compare_allw.py: head-to-head vs ALLW, $10k growth chart, fee drag chart,
  Excel export (results/allw_comparison_YYYYMMDD.xlsx)
- crisis_analysis.py: per-crisis regime analysis across 6 historical events,
  Excel summary, PNG charts (to be completed)

### New infrastructure decisions

- BACKTEST_END = date.today() (dynamic, no manual updating)
- Single chart output (1200×675, 16:9) — reddit/twitter dual format removed
- $10,000 starting value for all growth charts (retail legibility)
- Spot-check and ETF validation experiments moved to separate
  VALIDATION_CHECKS list in run_experiment.py (not run by default)
- Data source: yfinance retained. Revisit when first paying customer
  requires production reliability SLA.

---

## Phase 9 Final Results (2026-03-23) — 33 Experiments Complete

### Gate 1 closed: DE does not beat manual. Confirmed definitively.

26 DE-optimised experiments. Zero beat the manual baseline of 0.403.
Best optimised OOS: 7asset_no_iwd at 0.349 — 13% below manual.
This is not noise. This is a confirmed finding across every universe,
every IS/OOS split, and every variant of the fixed optimiser.

### The split2022 experiments are the worst results in the entire table

This was supposed to be the fix. It made things worse.

| Experiment | IS Opt | WF median | OOS |
|---|---|---|---|
| 6asset_tip_gsg_split2022 | 0.639 | 0.709 | 0.129 — worst result |
| 7asset_tip_gsg_vnq_split2022 | 0.588 | 0.408 | 0.140 |
| 5asset_dalio_split2022 | 0.678 | 2.000 | 0.140 |
| 7asset_tip_djp_split2022 | 0.623 | 1.212 | 0.165 |
| 8asset_manual_split2022 | 0.618 | 0.745 | 0.323 |

Why: the OOS window for split2022 is 2022-2026, starting at the worst
possible moment. The optimiser learns to survive the rate shock — but
the OOS window begins after the shock and enters the recovery/Iran war
period. The weights optimised for 2022 survival are wrong for 2022-2026.

`5asset_dalio_split2022` WF median = 2.000 (capped), OOS = 0.140.
Perfect walk-forward generalisation. Complete OOS failure. The most
extreme overfit signature in the entire dataset.

### WF median is not a reliable predictor of OOS performance

Principle 6 ("WF median > 0.6 = HIGH reliability") is retracted.

Evidence: the experiments with the highest WF medians produce the worst
OOS results consistently:
- 5asset_dalio_split2022: WF median 2.000, OOS 0.140
- 7asset_tip_djp_split2022: WF median 1.212, OOS 0.165
- 5asset_dalio_original: WF median 1.133, OOS 0.188
- 6asset_tip_gsg_split2022: WF median 0.709, OOS 0.129

High WF median is at best uncorrelated with OOS performance, at worst
inversely correlated. The WF median is elevated when the training and
test windows within the IS period happen to share similar characteristics.
This says nothing about whether those weights survive a genuinely different
OOS regime.

### Principle 7 retracted: 2022 in IS training does not improve OOS

Split2022 was the primary proposed fix. All split2022 OOS results are
below their standard-split counterparts. Including 2022 in training
produces weights optimised for a regime change — but the OOS window
begins after that regime change has played out. The fix assumes the
OOS environment resembles the IS environment. It does not.

### 7asset_tip_djp retracted as "best WF generaliser"

Previously flagged (Phase 5) as the strategy with the best walk-forward
robustness. Phase 9 WF median: 0.998. Phase 9 OOS Calmar: 0.247.
Another high-WF / low-OOS failure. The previous claim is retracted.

### Notable finding: 8asset_djp_replaces_gsg

OOS 0.347, WF median 0.803 — highest WF median of any non-split
experiment, second-best optimised OOS. DJP (balanced 35/35/30
energy/metals/agriculture) replacing GSG (70% energy) produces better
OOS stability. Commodity diversification away from pure energy reduces
optimiser overfit sensitivity. DJP may be preferable to GSG in the
risk parity framework.

### Complete OOS ranking — Phase 9 (26 non-spot-check experiments)

| Rank | Experiment | IS Opt | WF med | OOS |
|---|---|---|---|---|
| — | 6asset_tip_gsg (manual) | — | — | **0.403** ← baseline |
| 1 | 7asset_no_iwd | 0.584 | 0.712 | 0.349 |
| 2 | 8asset_djp_replaces_gsg | 0.568 | 0.803 | 0.347 |
| 3 | 8asset_efa_replaces_iwd | 0.527 | 0.641 | 0.333 |
| 4 | 8asset_manual_split2022 | 0.618 | 0.745 | 0.323 |
| 5 | 7asset_no_gsg | 0.650 | 0.703 | 0.318 |
| … | … | … | … | … |
| 26 | 6asset_tip_gsg_split2022 | 0.639 | 0.709 | 0.129 |

### Definitive conclusion

Return-based DE optimisation (Martin ratio objective) does not add value
over principled manual allocation for this ETF universe. Confirmed across
26 experiments with the fully fixed optimiser (correct bounds, smooth
objective, in-loop projection, no normalisation distortion). The failure
is structural: the IS period always contains a regime the optimiser
over-learns, and the OOS period always contains a different regime that
punishes that over-learning. This cannot be fixed by adjusting the IS/OOS
split, fixing bugs, or increasing DE search intensity.

**The correct response is to change the optimisation framework entirely.**
Risk contribution equalisation (risk parity) is the appropriate foundation.
It derives weights from structural properties of the assets (volatilities,
correlations) rather than from a historical return path. It does not
overfit to a regime. It is what Dalio actually uses.

---

## Phase 10 — Optimisation Strategy Rethink and Publishing (2026-03-22)

### Core concern: manual weights lack a principled derivation

The four validated strategies use manually chosen weights. These were
informed by the All Weather philosophy and refined through 229 experiments,
but were not derived from any systematic, repeatable process. This is a
problem for two reasons:

1. **Product credibility:** "I chose these weights by hand" is not a
   defensible methodology for a product that competes with Bridgewater.
2. **Generalisability:** weights that validated well historically may be
   coincidentally correct, not structurally correct. They cannot be
   systematically updated as market conditions change.

### Risk contribution equalisation — the missing foundation

Ray Dalio's actual All Weather methodology uses risk parity, not
return-based optimisation. The current engine implements the wrong
mathematical framework for the strategy it is trying to build.

**Definition:** Risk contribution equalisation requires every asset to
contribute equally to total portfolio volatility. For N assets:

    PRC_i = w_i · (Σw)_i / (wᵀ Σ w) = 1/N  for all i

where Σ is the covariance matrix, (Σw)_i is the marginal risk contribution
of asset i, and PRC_i is the percentage risk contribution.

This is solved numerically (no closed form). The objective is:

    minimise  Σᵢ Σⱼ (TRC_i - TRC_j)²

which is convex and smooth — well-suited to gradient-based methods.

**Why this is better than return-based DE:**
- Derived from structural properties (volatilities, correlations), not
  a historical return path
- Relatively stable over time — covariance structure changes slowly
  compared to return sequences
- Does not overfit to a specific regime
- Principled and defensible: "every asset takes equal risk" is a
  first-principles argument, not a data-mining result

**Why this is not a complete solution:**
- Requires leverage to achieve equity-like returns (Bridgewater uses ~2x)
- Unlevered risk parity is bond-heavy and lower-returning than 60/40
  in bull equity markets
- Covariance matrix estimation is sensitive to window length
- Correlations spike in crises — the 2022 rate shock saw bonds and equities
  become positively correlated, breaking the risk parity assumption

### Proposed hybrid approach (to discuss with Opus)

Use risk parity to derive the *structural baseline* weights (the starting
point grounded in first principles), then use DE to make *tactical
adjustments* around those weights within tight bounds.

This separates the problem into two layers:
1. **Structural layer:** risk parity sets the asset class proportions
   based on volatility and correlation. Repeatable, principled, updatable.
2. **Tactical layer:** DE searches for small improvements within
   ±5-10% of the risk parity weights. Limited degrees of freedom
   reduces overfit risk dramatically.

The tactical layer's objective could remain Martin ratio, but the search
space is now constrained to a neighbourhood of a principled starting point
rather than the full weight simplex. This is likely to generalise better
across regimes because the structural layer already handles the regime-
invariant part of the problem.

### Alternative: use risk parity as the optimisation objective directly

Instead of Martin ratio, replace the DE objective with:

    minimise  Σᵢ Σⱼ (TRC_i - TRC_j)²

This makes DE solve the risk parity problem directly. The result is
weights derived from first principles, computed by a robust optimiser,
within the per-asset bounds already defined. No historical return
path is involved in the weight derivation at all.

**Decision:** Defer to Opus session. Bring Phase 9 results table,
the overfit pattern finding (high IS Calmar predicts OOS collapse),
and the risk parity math. The central question: should the objective
function change entirely, or should risk parity be used as a prior?

### LinkedIn and content publishing — timing and legal

**Recommendation: wait 2-3 weeks before posting.**

Reasons to wait:
- ~~split2022 OOS result pending~~ — **resolved: split2022 makes things worse,
  not better. This no longer blocks publishing.**
- Sharpe/Sortino assume Rf = 0 — will be challenged by finance readers
- Daily MDD fix pending — numbers slightly overstated
- No brand name or landing page — nowhere to direct interest
- Risk parity framework needs to be designed before publishing methodology
  claims (the current methodology is "manual weights" which is not defensible)

When to post: after Rf fix, daily MDD fix, brand name decided, minimal
landing page live. Risk parity design does not need to be complete before
publishing the ALLW comparison — the comparison is based on returns, not
methodology claims.

**Legal position on Bridgewater comparison:**

Direct factual comparison is legal in both UK and US (comparative
advertising law). Requirements:
- Claims must be accurate and methodology disclosed
- No implied affiliation with Bridgewater or State Street
- No forward-looking performance claims
- Past performance disclaimer required
- Do not characterise ALLW as a bad product — frame as alternative
  for a different investor type

Required disclaimer: "ALLW is a registered trademark of Bridgewater
Associates / State Street. This is an independent analysis with no
affiliation to either firm."

**Strategic framing:** Never attack Bridgewater. Lead with numbers.
Calmar 2.59 vs 1.78 speaks for itself. "ALLW is excellent for investors
who want single-ticker simplicity. This is for investors who want
transparency, lower fees, and the ability to inspect the methodology."

**LinkedIn audience strategy:**
- Technical readers (quant researchers, PMs, fintech founders) care
  about methodology — lead with IS/OOS discipline and walk-forward
  validation, not headline returns
- Retail readers care about the $10k chart and worst-month comparison
- Post methodology content first (builds credibility), then results
  content (drives engagement), then product content (drives signups)

**GitHub sharing for job applications:**
- Make repo public but remove results/ folder (master log, experiment
  summaries, performance numbers — this is the real IP)
- Remove strategies.json (validated allocations with OOS results)
- Remove or sanitise research_log.md and session_handoff.md
- Keep all engine code — this is what interviewers want to see
- Describe as "quantitative portfolio research tool with strict
  out-of-sample validation" — do not mention ALLW or commercial intent
- Risk of IP theft is low: building a product from the code requires
  compliance, customer acquisition, and capital deployment. The window
  of competitive advantage is 12-18 months — timeline pressure is real
  but the threat is execution speed, not code theft.

---

## Consolidated Principles (updated Phase 10)

1. **GSG/DJP is non-negotiable for commodity exposure.**
   DJP (balanced commodities) shows better OOS stability than GSG (energy-heavy)
   in optimised context. Prefer DJP for risk parity framework.
2. **TIP inflation bonds are essential.**
3. **QQQ (growth equity) is non-negotiable.**
4. **Manual allocation beats DE. Confirmed definitively across 26 experiments.**
   Return-based DE optimisation does not add value over principled manual
   allocation. The correct framework is risk contribution equalisation.
   Do not invest further engineering time in Martin ratio DE.
5. **Three OOS windows required for production promotion.**
6. ~~**WF median > 0.6 = HIGH reliability.**~~ **RETRACTED (Phase 9).**
   WF median is not predictive of OOS performance. High WF median is at best
   uncorrelated, at worst inversely correlated with OOS Calmar in this dataset.
   Do not use WF median as a reliability signal.
7. ~~**Including 2022 in IS training improves WF reliability.**~~ **RETRACTED (Phase 9).**
   Split2022 experiments are the worst OOS results in the entire table.
   The fix assumes OOS resembles IS. It does not.
8. **Calmar is biased against short OOS windows.**
9. **2022-style shocks recur ~once/decade.**
10. **TQQQ is incompatible with All Weather.**
11. **ALLW is the primary competitive benchmark, not 60/40.**
12. **Every experiment should produce both research data AND content.**
13. **Validate demand before building product.**
14. **Cannot use "All Weather" in the product name.**
15. **Use Martin ratio for reporting only. Do not use for optimisation.**
    Martin ratio DE is confirmed to overfit. Calmar also retained for reporting.
16. **Use per-asset bounds, not uniform bounds.**
    Relevant for future risk parity implementation.
17. **High IS optimised Calmar (or Martin) is a red flag, not a green light.**
    IS opt > 0.7 consistently predicts OOS collapse. Confirmed across 26 experiments.
18. **Backtest with GSG; recommend GSG or PDBC to customers with disclosure.**
19. **DO NOT run spot-check experiments in standard runs.**
20. **Manual weights are not principled — risk parity is the correct foundation.**
    Risk contribution equalisation derives weights from structural asset properties,
    not historical return paths. This is what Dalio actually uses. Pending Opus
    session to design the implementation.
21. **Do not publish results until: Rf fixed, daily MDD fixed, brand name
    decided, landing page live.** Split2022 result no longer a blocker.
22. **7asset_tip_djp is NOT the best WF generaliser. RETRACTED (Phase 9).**
    WF median 0.998 in Phase 9, OOS Calmar 0.247. High WF median is unreliable.

---

## Open Questions

**Active (highest priority):**
- ~~Does split2022 DE beat manual?~~ **ANSWERED: No. Split2022 is the worst
  result. DE confirmed to fail across all splits and universes.**
- Should the optimisation objective change to risk contribution equalisation?
  **(Opus session required — this is now the most important open question)**
- Is there paying demand for a DIY ALLW alternative? (content test)
- Brand name?
- FCA compliance review — "rebalancing instructions" may constitute
  regulated advice under FSMA 2000. Must resolve before any UK product launch.

**Active (strategy/methodology):**
- Risk parity implementation: covariance matrix estimation window
  (1yr vs 3yr vs 5yr sensitivity)
- Leverage question: unlevered risk parity is bond-heavy. Commodity tilt
  is the current answer — validate explicitly with risk parity weights
- DJP vs GSG: DJP shows better OOS stability in Phase 9. Should DJP
  replace GSG as the primary commodity vehicle?
- Hybrid approach: risk parity structural layer + constrained DE tactical
  layer — is this worth implementing or does it reintroduce overfit risk?

**Active (engineering):**
- Rebalancing mismatch: backtest monthly unconditional vs live 5% threshold
- Daily vs monthly MDD gap (monthly understates true drawdowns)
- GBP/EUR adjusted backtest for non-US product claims
- Rf = 0 in Sharpe/Sortino (Fed at 3.5-3.75% — inflates ratios)
- Crisis analysis script completion (crisis_analysis.py)
- Spot-check experiments → VALIDATION_CHECKS list

**Answered:**
- ✅ Robust to transaction costs
- ✅ ETF substitutions confirmed (IVV≈SPY, GLDM≈GLD, QQQM≈QQQ)
- ✅ PDBC diverges from GSG in energy-shock windows — use GSG for backtest
- ✅ All four strategies validated OOS (manual weights)
- ✅ 8asset_manual demoted
- ✅ REITs work only with TIP+GSG anchor
- ✅ ALLW exists ($1B+ AUM, 0.85%, ~2x leverage)
- ✅ Four DE bugs identified and fixed
- ✅ DE confirmed to fail across all 26 experiments — regime mismatch is
  structural, not fixable by optimiser tuning or IS/OOS split changes
- ✅ Split2022 makes OOS worse, not better — retracted as a fix strategy
- ✅ WF median is not a reliable OOS predictor — retracted as reliability signal
- ✅ 6asset_tip_gsg beats ALLW on all risk metrics over March 2025–March 2026
- ✅ yfinance sufficient for research phase; revisit at first paying customer
- ✅ Bridgewater comparison is legal — factual, with disclaimer, no attack
- ✅ GitHub sharing safe if results/ and strategies.json excluded
- ✅ Publishing blockers: Rf fix, MDD fix, brand name, landing page
  (split2022 result no longer a blocker)

---

## Phase 10A — Foundation Fixes (2026-03-23)

All four blocking code issues fixed. All 55 tests passing.

### Fix 1: Sharpe/Sortino Rf=0 assumption
- Added `RISK_FREE_RATE = 0.035` to config.py (Fed funds ~3.5-3.75%)
- `compute_sharpe(rf_annual=0.0)` and `compute_sortino(rf_annual=0.0)` now subtract monthly Rf
- `compute_stats()` passes `config.RISK_FREE_RATE` automatically
- All future runs report correct Sharpe/Sortino. Prior runs in archive are pre-fix.
- Set `RISK_FREE_RATE = 0.0` in config.py to reproduce pre-fix numbers exactly.

### Fix 2: Daily MDD
- Added `compute_max_drawdown_daily(prices, allocation)` to backtest.py
- Simulates same monthly rebalancing but tracks portfolio value at every trading day
- New `StrategyStats.max_drawdown_daily` field (0.0 when daily prices not provided)
- `compute_stats(backtest, prices=prices, allocation=allocation)` optional signature
- Main.py and run_experiment.py (OOS + full_backtest steps) updated to pass daily prices
- New column `Max_DD_Daily (%)` in master log (AW_R strategy only; others remain 0.0)

### Fix 3: WF threshold comments retracted
- validation.py overfit_ratio thresholds (0.8/0.6) marked as unreliable per Phase 9
- WF framework retained as diagnostic; interpretation guidance updated in code

### Fix 4: Stale baseline
- run_experiment.py: `BASELINE_NAME = "6asset_tip_gsg_manual"`, `BASELINE_OOS_CAL = 0.403`
- Old 8asset_manual_v1 baseline (0.441 OOS) was demoted; now removed from output

### Risk parity diagnostic added
- `compute_risk_parity_weights(prices, tickers, estimation_years=5)` added to optimiser.py
- Uses SLSQP to minimise sum of squared pairwise risk contribution differences
- Standalone diagnostic — run before designing the full risk parity experiment phase
- **Next step:** run this on 6asset_tip_gsg to compare RP weights vs manual weights

### Updated CLAUDE.md
- Role section added: operate as quant finance researcher + software engineer

---

## Current Status (2026-03-23)

**Experiments completed:** 33 universes, Phase 9 complete
**Master log:** master_log_phase9_full.xlsx (229+ rows)
**DE optimisation:** confirmed failed across all experiments — do not repeat

**Production strategy (manual weights, confirmed):**
- 6asset_tip_gsg: OOS 0.403 (manual) — primary
- 7asset_tip_gsg_vnq: OOS pending re-evaluation (manual weights only)
- 7asset_tip_djp: previously claimed best WF — retracted, OOS 0.247
- 5asset_dalio: OOS 0.188 (optimised) — conservative anchor

**Next phase:** Risk parity framework design
**Primary competitive benchmark:** ALLW ETF
**ALLW comparison:** completed, 6asset beats ALLW on all risk metrics
**Crisis analysis:** script briefed, not yet implemented

**Blocking items for content launch:**
- ~~Rf fix (Sharpe/Sortino assume Rf=0)~~ **DONE**
- ~~Daily MDD fix~~ **DONE**
- Brand name decision
- Landing page

**Blocking items for next experiment phase:**
- Run risk parity diagnostic on 6asset_tip_gsg (compare RP vs manual weights)
- Design risk parity experiment methodology (covariance window sensitivity)
- Implement risk parity optimiser (replace Martin ratio objective in DE)
- Re-validate production strategies with risk-parity-derived weights