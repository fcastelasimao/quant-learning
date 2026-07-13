# Capacity Model — Phase 2: Almgren adoption + two-door capacity

> **Status: PAUSED (2026-07-07)** in favor of Path A (execution track) — see ROADMAP.md. Stage 0
> was completed as S12 (`research/12_stage0_forks`): 0a = multi-day (Phase 2 stays), 0b = Almgren
> permanent term rejected for the ETFs, small residual carried; Stage A.5 downgraded to a bounded
> check. Any GATE decision must present the two-model envelope (√-law Y-band vs Almgren) as the
> uncertainty band.

## Context

The `slippage/` library answers the **capacity question** for TQQQ/SQQQ 15-min strategies: *up to
what per-trade (and per-session) size can we trade before market impact hurts us, and how do we
execute at scale to minimise it?* Impact is the binding uncertainty, and today it rests on a
borrowed `Y ∈ [0.3, 1.0]` √-law constant — a ~10× band that is the width of the entire answer.

The Almgren, Thum, Hauptmann & Li (2005) *Direct Estimation of Equity Market Impact* paper is the
empirical source that constant was loosely drawn from. It publishes **tightly-calibrated
coefficients** (γ=0.314, η=0.142), a **linear permanent + 3/5 temporary** decomposition, and it
**rejects the √-law** (β=½) in favour of β=3/5. Adopting it directly (a) collapses the band toward
a justified center, (b) fixes a mild under-statement of cost at large size, and (c) upgrades the
model structure. This phase adopts Almgren, anchors it with free tick data, models permanent-impact
accumulation, and (conditionally) models the ETF creation/redemption door.

**This plan branches.** Two upstream unknowns (Stage 0) can each delete downstream stages, so they
are resolved *before* any code.

## Ground rules (apply to every stage)

- **Env:** `~/opt/anaconda3/envs/quant/bin/python`. Run from `volume_research/`.
- **Tests after every stage:** `~/opt/anaconda3/envs/quant/bin/python -m pytest tests/ -q` (85
  currently green — keep them green; defaults must not change existing results).
- **Evidence tiers — label every number** in findings/plots as one of:
  **Measured** (spread S11, λ Stage C) · **Calibrated/adopted** (Almgren — golden-testable) ·
  **Modeled/no-ground-truth** (accumulation, creation). The last tier gets a wider, explicitly-softer
  band + mandatory sensitivity analysis.
- **Reproducibility:** any external snapshot (shares outstanding, ADV, creation fee) is persisted
  **with its as-of date** into the stage's `results/` and echoed in findings.
- **Research-folder numbering** continues from the existing `research/11_cs_validation/`
  (folders are +2 offset from BUILD_LOG "Stage N" — keep BUILD_LOG stage numbers continuing from 13).
- **New research stage layout** mirrors existing ones: `research/NN_name/build_NN_name.py` +
  `findings_NN_name.md` + gitignored `results/`. Follow `research/11_cs_validation/` as the template
  (path bootstrap, `quantcore.config`, Alpaca fetch pattern).

---

## STAGE 0 — Resolve the two forks (NO CODE beyond a small probe; do first)

Outputs are two decisions that gate everything below.

### 0a — Alpha horizon (gates whether Phase 2 exists)
- **Action (user + measurement):** determine whether the 15-min edge is captured intraday
  (minutes–hours) or accumulated over days. Measure decay from the backtest in
  `../TQQQ_SQQQ_analysis/` (autocorrelation of signed post-signal returns) and/or confirm with the team.
- **Branch:** intraday-only ⇒ **cut Phase 2 (Stages B, E)** — creation/redemption is a batch,
  hours-to-overnight, creation-unit-discrete process and is temporally inaccessible for an intraday
  signal; the screen answer is the whole story. Multi-day accumulation ⇒ Phase 2 lives.

### 0b — ETF permanent impact (gates the permanent term in A and the size of A.5)
- **Reasoning to resolve:** an arbitrage-pinned ETF's *price* tracks NAV, so its permanent impact is
  the **underlying's** (~0 for NDX at your sizes), not its own `Θ/V`. Almgren's permanent term is a
  single-stock float proxy and may be *conceptually wrong* for TQQQ, not merely extrapolated.
- **Empirical check (small probe, reuse Alpaca `/trades`):** pull tick data around the largest TQQQ
  prints and test whether price *permanently* deviates from the moving underlying/NAV proxy (QQQ×k).
- **Branch:** permanent≈0 ⇒ Stage A **omits the `Θ/V` permanent term** (cost = spread + temporary),
  and **A.5 shrinks from a stage to a bounded check**. permanent≠0 ⇒ keep the full model.

**Deliverable:** `research/12_stage0_forks/findings_12_stage0_forks.md` recording both decisions
with evidence. **Do not start Stage A until 0b is decided.**

---

## PHASE 1 — the executable screen-door answer

### STAGE A — Adopt Almgren  ·  tier: Calibrated/adopted

**Files:** `slippage/impact.py`, `slippage/cost.py`, `slippage/model.py`, `slippage/calibrate.py`,
`slippage/__init__.py`; `tests/test_impact.py`, `tests/test_cost.py`, `tests/test_model.py`,
`tests/test_calibrate.py`; `research/13_almgren_adopt/`.

**Model (adopt verbatim):**
```
Temporary (the cost):  J_temp = η · σ · participation^β         η=0.142, β=0.6
Permanent (if 0b≠0):   I      = γ · σ · (X/V) · (Θ/V)^0.25       γ=0.314, α=1, δ=0.25
```
Key bridge (already true in the code): `slippage_cost` uses `impact = Y·σ·participation^β` where
`participation = (notional/ADV)·(day/horizon)` — this **is** Almgren's `X/(V·T)`. So temporary
impact is a **drop-in**: replace `Y·σ·participation^0.5` with `η·σ·participation^0.6`.
Permanent uses order-size-vs-ADV `X/V = notional/adv_usd` and `Θ/V = market_cap/adv_usd`.

**Changes:**
1. `impact.py`: add `almgren_temporary(participation, sigma_bps, *, eta=0.142, beta=0.6)` and (if
   0b≠0) `almgren_permanent(notional, adv_usd, market_cap_usd, sigma_bps, *, gamma=0.314, delta=0.25)`.
   Named module constants `ETA, BETA, GAMMA, ALPHA=1, DELTA` with a paper citation. Keep existing
   `impact_bps`/`capacity` (the √-law) intact for the before/after comparison.
2. `cost.py`: add an `impact_model: str = "sqrt"` param to `slippage_cost` (and thread through
   `optimal_participation`, `capacity_at_horizon`). `"almgren"` routes impact to the temporary term
   (+ permanent as a separate returned key if 0b≠0). Add a `market_cap_usd` field to `MarketParams`
   (default `None`; required only for `impact_model="almgren"` when permanent is on).
3. `model.py`: thread `impact_model` through `roundtrip`/`roundtrip_optimal`/`roundtrip_band`.
   **Specify round-trip accounting explicitly:** `RT_expected = spread(×2) + J_temp(×2) [+ I_perm
   per 0b]`, permanent parts **added, not netted**. Ensure **no spread double-count** (Almgren J is
   mid-to-mid; total = `half_spread + J`).
4. `calibrate.py`: add `market_cap_usd` (or `shares_outstanding`) to `Calibration`/`MarketParams`,
   sourced from **FMP** (`/profile` or `/quote` `sharesOutstanding × price`) with an as-of date;
   pass-through only (never fit).
5. `__init__.py`: export the new impact functions.

**Tests (the correctness proof):**
- **Golden test** in `tests/test_impact.py` reproducing paper Table 3 to <1 bp:
  - IBM (V=6.561M sh, Θ=1728M sh, σ=1.57%): `J = 32 / 25 / 18` bp at `T = 0.1 / 0.2 / 0.5`.
  - DRI (V=1.929M sh, Θ=168M sh, σ=2.26%): `J = 43 / 32 / 23` bp at the same T.
  - Permanent: IBM `I≈20` bp, DRI `I≈22` bp.
- `impact_model="sqrt"` path unchanged (regression guard on existing numbers).
- No-double-count assertion (total = spread + J).

**Research:** `build_13_almgren_adopt.py` re-runs the 04→07 capacity chain under `impact_model=
"almgren"`, emits a **before/after capacity table** (see GATE for the σ constraint), and defines the
new band = Monte-Carlo over published std-errors (γ±0.041, η±0.006) × an ETF-extrapolation
multiplier `m` supplied by Stage C. `findings_13` documents the movement + reconciliation of old
`Y∈[0.3,1.0]` vs `γ,η`. **Default stays `"sqrt"` until the GATE decision** (reversible, mirrors how
EDGE was added opt-in).

**Done when:** golden test passes <1 bp; sqrt path unchanged; before/after table produced.

### STAGE C — Near-field tick anchor (boxed, 1 stage)  ·  tier: Measured

**Files:** `research/14_near_field_lambda/` only (no library change unless a clean helper emerges).

**Changes:** using the free Alpaca `/trades` + `/quotes` (delayed-SIP, >15 min; pattern from
`research/11_cs_validation/build_11...py`, add pagination via `next_page_token`):
1. `fetch_trades()` (paginated) alongside the existing `fetch_quotes()`.
2. Sign trades (Lee-Ready / tick rule); estimate **Kyle's λ** (price move per signed volume) and the
   **effective spread**. **Control bid-ask-bounce bias** — sample at fixed volume/clock intervals
   (or a Hasbrouck-style VAR), do **not** regress raw high-frequency mid-changes.
3. **Validate the estimator on QQQ** (known-tiny) before trusting TQQQ/SQQQ.
4. Compare implied small-order impact to Almgren-at-small-participation → emit the extrapolation
   multiplier `m` for Stage A's band. Also supply 0b's empirical permanent check.

**Scope box:** near-field boundary + σ-scaling only; label `m` a **weak prior** on far-field
transfer (ticks cannot see meta-orders). No microstructure sprawl.

**Done when:** λ + CI reported; QQQ sanity check passes; `m` emitted; findings_14 written.

### GATE (after A + C)
Proceed only if **all**: (1) golden test passed; (2) `participation`/V·T mapping confirmed; (3) the
before/after capacity table holds **σ identical** across `sqrt` and `almgren` (only the impact model
changes — so movement isn't σ bookkeeping). **Decision:** flip the library default to
`impact_model="almgren"` if warranted; if the band collapses and capacity barely moves, de-prioritise
Phase 2.

### STAGE A.5 — Permanent-impact accumulation (W5)  ·  tier: Modeled
Sized by 0b. If permanent≈0 ⇒ a bounded one-shot check in `build_13` and a note. Else a dedicated
`research/15_permanent_accumulation/`: sum persistent `I` across a session under the strategy's
**real trade-direction autocorrelation** (from the backtest trade sequence — signs are observable
even at retail size); report **per-trade vs aggregate session capacity**, treating non-decaying
permanent as an **upper bound** plus a decayed-permanent sensitivity. Mandatory sensitivity table.
**Done when:** aggregate-vs-per-trade capacity reported with a decay sensitivity band.

### STAGE D — Screen execution schedule, re-grounded  ·  tier: Calibrated
**Files:** `slippage/plan.py`, `slippage/cost.py` (`optimal_participation`, `capacity_at_horizon`),
`tests/test_plan.py`, `tests/test_cost.py`, `tests/test_cost_aware_sizing.py`;
`research/16_schedule_regrounded/` (re-runs Stages 05/06/07 outputs under Almgren).
**Changes:** thread `impact_model="almgren"` (β=0.6, η) through `plan_execution` and the optimizers;
add a **volume-cap guardrail** (never exceed a configurable % of interval volume — Zipline's idea);
in findings/plots **shade the >10%-participation region as extrapolation** (Almgren's validated
domain is ≤10% ADV). **Done when:** planner + optimisers coherent under Almgren; extrapolation region
marked; existing λ-grid / POV outputs regenerated.

**→ End Phase 1 = the capacity answer for how you trade today.**

---

## PHASE 2 — the creation-door what-if (ONLY if 0a = multi-day)

### STAGE B — Creation door, lower envelope, crossover  ·  tier: Modeled (what-if)
**Files:** `research/17_creation_door/` (+ optionally a small `creation_cost()` helper in a new
`slippage/creation.py` if it's reused; otherwise keep it in the build script).
**Changes:**
1. Screen door = Phase 1 result.
2. Pull QQQ + NDX-future (NQ) ADV + σ (FMP).
3. Fetch TQQQ/SQQQ **creation-unit fee** from the ProShares SAI (flag if not machine-readable; use a
   documented assumption otherwise).
4. `creation_cost(N) = fee% + almgren_temporary(hedge on NDX)` with **leverage `L=3` first-class**:
   `hedge_notional = L·N`, and use **NDX's ~1× σ** on the NDX side. Add a test asserting
   `σ_TQQQ ≈ L·σ_NDX`.
5. Capacity = `min(screen, creation)` vs size; report the **crossover $** per ticker. QQQ (L=1) as an
   out-of-sample sanity check.
6. If leveraged-ETF creation (cash-create + swaps) can't be pinned, deliver a **bounded range**
   (`fee ≤ C ≤ fee + 3×-NDX-impact`), not a false-precision line. Flag **swap financing** as a
   separate out-of-model line, and restate the **access + horizon caveats** (what-if, not executable
   via Alpaca).
**Done when:** lower-envelope curve + crossover per ticker, all assumptions itemized and tiered.

### STAGE E — Generalise to the class  ·  tier: Modeled
`research/18_class/`: apply the pipeline to SPXL/UPRO/SPXS etc., parametrized by **per-name
underlying liquidity**. **Sub-scope the "creation is cheap" claim to large-cap-index underlyings
(SPX/NDX)**; treat small-cap-underlying leveraged ETFs (TNA/TZA on RUT) as a separate regime or
exclude. **Done when:** comparative capacity + crossover table for the sub-scoped class.

---

## Verification (end-to-end)

1. **Unit/regression:** `pytest tests/ -q` stays green throughout; the **golden test** (Almgren
   Table 3, <1 bp) is the load-bearing correctness check for Stage A.
2. **σ-controlled before/after** at the GATE proves the capacity move is the coefficient change.
3. **Reproduce each stage:** `python research/NN_name/build_NN_name.py` regenerates its `results/`
   and matches its findings.
4. **Data-tier sanity:** QQQ acts as the out-of-sample check in Stages C (near-field λ) and B
   (creation door) — the model must reproduce QQQ's known-large capacity / known-tiny spread.
5. **Docs:** update `BUILD_LOG.md` (new stages), `FINDINGS.md` (revised capacity headline + tier
   labels), `HANDOFF.md`, and `slippage/README.md` (new `impact_model` param).

## Open dependencies / decisions the user owns
- **0a alpha horizon** — needs the backtest measurement and/or a team answer; determines whether
  Phase 2 is built at all. Default to the conservative screen-only branch if unavailable.
- **Library default** — whether `impact_model="almgren"` becomes the default is a GATE decision, not
  pre-decided here (keeps the change reversible, mirrors the EDGE opt-in precedent).
