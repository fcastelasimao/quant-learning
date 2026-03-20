# Research Log — All Weather Portfolio Tracker

A running record of decisions made, experiments run, findings discovered,
and questions that remain open. Updated after each meaningful session.

---

## Project Goal

Build a backtesting, optimisation, and validation tool for a Ray Dalio-style
All Weather Portfolio. Primary objective: capital preservation with competitive
risk-adjusted returns. Primary metric: Calmar ratio (CAGR / max drawdown).
Longer-term goal: paper trade, then live trade, then sell as a product.

---

## Phase 1 — Initial Architecture and 5-Asset Experiments

### Decision: Replace LQD with TIP
**Date:** Early project (pre-2026-03-17)
**Context:** Original 6-asset universe included LQD (corporate bonds).
**Finding:** LQD and TLT are highly correlated in rate-shock environments.
Holding both doubles bond exposure without adding diversification. In 2022,
both fell together. TIP (inflation-linked bonds) provides a different
mechanism of protection — its value is directly tied to inflation rather
than just duration.
**Decision:** Replace LQD with TIP permanently.
**Outcome:** Significant improvement in 2022 stress test performance.

### Decision: Calmar ratio as primary optimisation objective
**Date:** Early project
**Context:** Considered Sharpe, Sortino, and Calmar as objectives.
**Reasoning:** Sharpe penalises upside and downside volatility equally, which
is wrong from a capital preservation perspective. Sortino is better but harder
to compute gradients for. Calmar directly captures the risk we care about:
how much return do we earn per unit of maximum loss?
**Decision:** Calmar as primary metric throughout. Sortino tracked as secondary.

### Finding: 5-asset TIP allocation validated
**Allocation:** SPY 14.2%, QQQ 20.3%, TLT 30%, TIP 14.2%, GLD 21.3%
**IS (2004-2020):** CAGR 8.86%, MaxDD -15.83%, Calmar 0.560
**OOS (2020-2026):** CAGR 8.72%, MaxDD -23.09%, Calmar 0.378
**Full (2004-2026):** CAGR 8.99%, MaxDD -23.09%, Calmar 0.389
**Notes:** Beats 60/40 on all risk-adjusted metrics over 21 years. The DE
optimiser found GLD 42.7% and QQQ 30.3% concentration — identified as
backtest-fitted rather than robust. Walk-forward validated with mean overfit
ratio 0.750.

### Finding: 2022 is the structural weak spot
**Date:** Identified during 5-asset experiments
**Finding:** Any allocation combining bonds and equities struggled in 2022
when rates rose at the fastest pace in 40 years. Simultaneous losses in
bonds, equities, and gold. No weight configuration within the 5-asset universe
could fully avoid this.
**Conclusion:** Universe expansion is the right lever, not weight optimisation.
Adding a commodity ETF (GSG) was identified as the primary fix — commodities
surged in 2022 when everything else fell.

---

## Phase 2 — Architecture Improvements

### Decision: Strict IS/OOS methodology
**Date:** 2026-03-19
**Context:** Original codebase ran optimisation on the full period including
what would become the test set.
**Decision:** Enforce three-date boundary: BACKTEST_START, OOS_START,
BACKTEST_END. All optimisation confined to IS period. OOS_START = 2020-01-01.
**Rationale:** Every time you see an OOS result and adjust something, you leak
information from the test set. The methodology has to be enforced architecturally
not just by discipline.
**Implementation:** RUN_MODE controls which date slice each mode sees.
oos_evaluate is the only mode that touches 2020-2026 data.

### Decision: Total return prices
**Date:** 2026-03-20
**Context:** Suspected backtest was using price returns only.
**Investigation:** Verification script showed TLT price return 2006-2026: -5%.
TLT total return: +79%. The entire positive return from holding TLT comes
from coupon income — invisible in price-only data.
**Surprise finding:** yfinance uses auto_adjust=True by default in versions
>= 0.2.x, meaning ALL historical runs in this project were already using
total return data. The "price_return" label on old master log rows is
misleading — they are total return.
**Decision:** Make this explicit via PRICING_MODEL parameter. Keep
auto_adjust=True as default (total_return). The price_return option is
available for comparison but is not the recommended setting.

### Decision: Transaction costs and tax drag modelling
**Date:** 2026-03-20
**Parameters added:** TRANSACTION_COST_PCT (default 0.0), TAX_DRAG_PCT (default 0.0)
**Rationale:** Monthly rebalancing has real-world costs. For UK investors:
bid-ask spread ~0.05%, FX conversion ~0.5-1.5% if buying USD ETFs with GBP.
Tax drag is zero inside an ISA/SIPP. This strategy is best implemented in a
tax-sheltered account.
**Note:** Tax model is a blunt annual drag — not a proper CGT calculation
on realised gains. Sufficient for sensitivity analysis but not for precise
tax planning.

### Decision: 5 new performance metrics
**Date:** 2026-03-19
**Added:** Avg Drawdown, Max DD Duration, Avg Recovery Time, Ulcer Index,
Sortino Ratio
**Rationale:** Max drawdown alone is dangerously incomplete. It tells you
the single worst loss but nothing about how long you stay underwater, how
often you lose, or what a typical bad period looks like.
**Key insight:** Ulcer Index penalises portfolios that spend a long time
moderately underwater, not just the single worst point. This captures the
2022 experience better than max drawdown alone.

---

## Phase 3 — 7-Asset and 8-Asset Experiments

### Finding: 7-asset uncapped DE — cautionary tale
**Allocation:** TLT 30%, SHY 30%, gold near floor, equity near floor
**IS Calmar:** 0.753 (excellent)
**OOS Calmar:** 0.252 (collapsed)
**Lesson:** The optimiser pushed 60% into bonds because the 2004-2020 IS
period was a bond bull market. This worked beautifully in-sample and
catastrophically out-of-sample when 2022 arrived. A high IS Calmar with
no caps is not a signal of a good strategy — it is a signal of overfitting
to the specific macro regime in the training data.

### Finding: 7-asset with 50% bond cap — previous best
**Allocation:** TLT 30%, QQQ 19.7%, GLD 22.3%, SHY 7%, others ~7%
**IS Calmar:** 0.583
**OOS (2020-2026) Calmar:** 0.384
**Full Calmar:** 0.400
**Status:** Previous best before 8-asset experiments. Superseded by
8-asset manual allocation.

### Decision: 8-asset macro-informed allocation
**Date:** 2026-03-19
**Motivation:** Two concerns drove the expansion:
1. Gold at all-time highs — reduce from 22% to 15%
2. Iran oil shock pushing oil toward $100/barrel — need commodity hedge
   for stagflation scenario
**New assets added:** IWD (value equity), IEF (intermediate bonds),
SHY (short-term anchor), GSG (commodities)
**Rationale per asset:**
- IWD: lower-beta equity alternative to QQQ, value factor diversification
- IEF: two-rung bond duration ladder alongside TLT
- SHY: stability anchor, near-cash, minimal rate sensitivity
- GSG: direct commodity exposure — surged in 2022 when bonds/equities fell

### Finding: DE optimiser does not reliably beat manual allocation
**Date:** 2026-03-19
**Context:** Ran walk-forward with multiple cap configurations and seeds.
**Result:** Optimiser beat manual allocation in only 1-3 out of 4 windows
across all tested configurations.
**Root cause:** The 2006-2020 IS period contains two very different regimes —
the 2008 crisis and the 2010-2019 low-volatility bull market. These pull
the optimiser in opposite directions. Window 3 (2010-2017) is structurally
pathological — any Calmar-maximising optimiser trained on it finds a
tech-heavy momentum allocation that fails in the subsequent period.
**Decision:** Use manual allocation for live implementation. DE optimiser
retained as a diagnostic tool, not a production tool.

### Finding: 8-asset manual allocation validated OOS
**Date:** 2026-03-19
**Allocation:** SPY 10%, QQQ 15%, IWD 10%, TLT 25%, IEF 10%, SHY 5%,
GLD 15%, GSG 10%
**IS (2006-2020):** CAGR 7.13%, MaxDD -21.23%, Calmar 0.336
**OOS (2020-2026):** CAGR 8.11%, MaxDD -18.38%, Calmar 0.441
**Full (2006-2026):** CAGR 7.51%, MaxDD -21.23%, Calmar 0.354
**vs 60/40 OOS:** Calmar 0.441 vs 0.263 — clear margin
**Notes:** OOS Calmar higher than IS Calmar (overfit ratio 1.31) — genuine
positive surprise. Beats previous best (7-asset 50% cap: OOS Calmar 0.384).
The 8-asset version is the current recommended allocation.

### Decision: AW Rebalanced vs Buy-and-Hold framing
**Date:** 2026-03-19
**Finding:** B&H consistently beats rebalanced on Calmar in trending markets.
Monthly rebalancing sells winners in a bull market.
**Reframing:** The correct comparison is AW_R vs 60/40, not AW_R vs B&H.
On OOS Calmar: AW_R 0.441 vs 60/40 0.263 — decisive.
**Rationale for rebalancing despite lower Calmar:** After 15 years of a gold
rally, a B&H portfolio may be 40% gold — an allocation no rational investor
would choose intentionally and one that is extremely vulnerable to a gold
correction. The rebalanced version maintains a stable, explainable allocation.

---

## Phase 4 — Code Quality and Pre-Paper-Trading

### Code audit findings (2026-03-20)
**Branch:** audit/code-audit, merged to main as v1.1-post-audit
**Key bugs fixed:**
- plotting.py: crash when 60/40 absent (4 locations)
- optimiser.py: np.random.seed() reset global state — walk-forward windows
  were getting identical random sequences. Fixed with default_rng().
- validation.py: negative train Calmar produced misleading overfit ratio
- validation.py: mean overfit ratio distorted by outlier windows. Now clamped
  at 2.0 with median reported alongside mean.
- config.py: no validation that WF_STEP_YEARS >= WF_TEST_YEARS
- portfolio.py: KeyError on manually edited holdings JSON
- data.py: global warning suppression affected entire process
**Financial correctness:** All clear. Rebalancing logic, 60/40 benchmark,
transaction costs, tax drag, walk-forward data leakage — all verified correct.
**Tests added:** 25 new tests covering 5 previously untested stat helpers.
53 tests total, all passing.

### Decision: run_experiment.py automation
**Date:** 2026-03-20
**Rationale:** 10 universe experiments × 6 steps × manual config editing =
60+ config changes with high error risk. Automation eliminates human error,
enables overnight runs, and produces a clean summary table at the end.
**Design:** Full workflow automated (backtest → optimise → walk_forward →
oos_evaluate → full_backtest). Human confirmation gate before oos_evaluate
preserves the quality checkpoint that prevents OOS contamination.
**Status:** Built and running.

---

## Open Questions

- Does GSG actually contribute to 2022 protection, or does the optimiser
  minimise it because the 2014-2016 commodity crash dominates the IS period?
  → Phase 3 experiments will answer this (7asset_no_gsg experiment).

- Does IWD (value equity) contribute anything beyond what SPY provides?
  The optimiser consistently minimised it. Is this a signal or a regime artefact?
  → Phase 3 experiments will answer this (7asset_no_iwd experiment).

- Is geographic diversification (EFA) more valuable than domestic value (IWD)
  given current US equity concentration?
  → Phase 3 experiments will answer this (8asset_efa_replaces_iwd experiment).

- What is the minimum portfolio size below which monthly rebalancing costs
  erode the advantage over buy-and-hold?
  → Need sensitivity analysis with TRANSACTION_COST_PCT.

- Does the Dalio original allocation (30/40/15/7.5/7.5) actually underperform
  all the engineered versions, or has 20 years of methodology added little value?
  → Phase 3 experiments will answer this (5asset_dalio_original experiment).

---

## Current Status (2026-03-20)

**Active work:** Phase 3 asset universe experiments running via run_experiment.py.
10 universes across 5/6/7/8-asset configurations. Results will determine which
universes progress to OOS evaluation.

**Current validated allocation:** 8-asset manual (OOS Calmar 0.441)

**Next decision point:** After Phase 3, select the best 2-3 universes and
determine whether to proceed to paper trading or continue optimising.

**Baseline for comparison:**
- IS Calmar (manual): 0.336
- OOS Calmar: 0.441
- 60/40 OOS Calmar: 0.263