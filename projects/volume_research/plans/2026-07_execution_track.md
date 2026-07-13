# BUILD PLAN — predictor + scheduler tracks, capacity add-ons

**For:** an implementation agent (Sonnet). **Date:** 2026-07-08. **Status: ACTIVE.**
Supersedes the A1–A4 bullet stub. The paused Almgren plan (`plans/2026-07_almgren_adoption.md`)
is partially folded in here as stage C01.

## Context (read first)

This project replaced a flat 20 bps backtest cost with a size-aware slippage model (stages
S01–S12, see ROADMAP.md). The work now splits into three tracks matching the client's three asks:

- **Predictor** — given an order and the market state, predict fill-price slippage ("I set an
  order at X and filled at X + diff — predict diff").
- **Scheduler** — given a large order and the market state, produce the child-order schedule
  ("execute large volumes in small chunks over time, depending on market volume").
- **Capacity** — from what size does impact become a problem (mostly answered by S04–S07; two
  add-ons remain).

Key prior results this plan builds on: half-spread ~0.7–1 bp (S01, NBBO-validated in S11); timing
risk std ≈ 17.6·t^0.45 bps TQQQ (S03); √-law impact with adopted Y ∈ [0.3,1.0] (S04); live fills
show a +14 bps limit-chase entry drag, n=15 (S08); the edge is multi-day — median hold ~20 h,
sub-2.4 h trades are net losers, a 30–60 min entry delay forfeits only ~11–26% of the eventual
move (S12); ETF permanent impact ≈ negligible, Almgren's permanent term rejected (S12).

## Ground rules (every stage)

- **Env:** `~/opt/anaconda3/envs/quant/bin/python`, run from `volume_research/`.
  `python -m pytest tests/ -q` green after every stage (85 now; grow, never break).
- **Stage layout:** `research/<track>/NN_name/build_NN_name.py` + `findings_NN_name.md` +
  gitignored `results/`. Follow `research/11_cs_validation/` as the template for structure, path
  bootstrap, DB access (`quantcore.config`), and the Alpaca fetch pattern.
- **Findings style:** mirror the existing files — Headline / Strategy & mathematics / Numbers /
  Caveats / Reproduce. Plain math (no raw LaTeX). Label every number with its evidence tier:
  **Measured / Calibrated-adopted / Modeled**. Impact numbers are always **bands**, never points.
- **Docs upkeep:** after each stage, add a BUILD_LOG entry (canonical ID) and update the ROADMAP
  status table. One git commit per stage (or logical slice); do not push.
- **Data:** ticker DBs at `/Users/franciscosimao/Documents/QuantFinance/data/DB_<TKR>_historical_data.db`
  (tables `candles_1min`, `candles_15min`, `candles_1d`); trade logs at
  `../TQQQ_SQQQ_analysis/full_history_canonical/TRADES_{SYM}_full_history.csv` (2,343 TQQQ /
  1,930 SQQQ; columns include `entry_time`, `decision_price`, `avg_order_price`, `exit_time`,
  `exit_reason`, `stop_price_entry`, `regime_entry`, `hold_days`); live Alpaca cycle logs in
  `LIVE_alpaca_cycle_from_20260522_20260624/` (parser exists in
  `research/08_live_validation/build_08_live_validation.py`).
- **Out of scope:** the live trading engine (separate repo). Where a result implies an engine
  change (entry order type, timeout), say so in findings and stop.

## Canonical IDs and folder convention (new)

New work lives in **track folders**, numbered within the track:

```
research/predictor/01_market_state/      → ID P01
research/scheduler/01_alpha_decay/       → ID E01
research/capacity/01_almgren_envelope/   → ID C01
```

Legacy stages S01–S12 **stay where they are** (their findings contain reproduce-paths; they are
shared foundations, mapped in ROADMAP.md). Update ROADMAP.md's table with three new track
sections using these IDs.

---

## Phase 0 — conventions and the rename (do first)

### 0.1 — Track folders + ROADMAP

Create `research/predictor/`, `research/scheduler/`, `research/capacity/`. Add the ID scheme
above to ROADMAP.md ("Canonical numbering" section) and stub rows for the stages below with
status "planned".

### 0.2 — Rename the cost API to `expected_slippage_bps`

Owner decision (final): the generic pre-trade cost estimate is renamed for clarity — "cost"
confused a non-specialist reader.

| Old | New |
|---|---|
| `slippage/cost.py::slippage_cost(...)` | `expected_slippage_bps(...)` |
| `RoundTripCost.expected_cost_bps` (model.py) | `RoundTripCost.expected_slippage_bps` |
| any `expected_cost*` field on `ExecutionPlan` (plan.py) | `expected_slippage*` |

Rules: grep for every public name whose meaning is "the mean pre-trade cost estimate" and apply
the same convention; update **all** callers (`slippage/`, `tests/`, `research/*/build_*.py`) and
the docs that quote code names (`slippage/README.md`, README.md, FINDINGS.md if applicable). No
back-compat aliases (internal project). Add one docstring sentence everywhere renamed:
*"Expected (mean) slippage: spread + impact (+ signed entry drag). Timing risk is a separate
variance channel and is deliberately NOT included in this number."*
Do not regenerate legacy `results/` (outputs are numerically unchanged); tests green is the check.

---

## Phase 1 — Predictor track

### P01 — Market state (`research/predictor/01_market_state`)

The client's "depending on the state of the market / market volume" input. From `candles_1min` /
`candles_15min` (recent ~2y unless stated):

1. **Intraday volume profile:** mean and median share of the day's volume per 15-min bin
   (expect the U-shape); day-of-week effect; dispersion per bin. TQQQ, SQQQ, QQQ.
2. **Interval-volume predictability:** predict the next 15-min bar's volume from
   (time-of-day profile × trailing-day activity EWMA). Report out-of-sample R² and the p10/p20
   thin-tape quantiles per bin (the "thin tape" detector).
3. **Volatility nowcast:** trailing realized σ from 1-min returns, mapped onto the existing
   calm/normal/stress tercile definition (S03's `vol_regime`) so regimes stay consistent.
4. **Spread state:** time-of-day spread curve. For live-facing use, measure directly from Alpaca
   SIP NBBO (pattern from `research/11_cs_validation`); CS-15min stays as the historical/fallback
   estimate (S11 showed CS reads ~20–25% low — fine for a floor, but prefer measured where
   available).

**Deliverable:** a small `slippage/state.py` — `MarketState` dataclass +
`estimate_state(ts, data) → MarketState` (expected interval volume, thin-quantile volume, σ_now,
regime, spread_bps) — plus findings + tests (synthetic recovery; sanity: open bin > midday bin
for both volume share and spread).

### P02 — Chase simulation (`research/predictor/02_chase_simulation`)

Replace the n=15 live estimate of the +14 bps limit-chase drag with a full-history simulation,
and answer "if we still chase, what timeout?". At every historical entry signal (TRADES CSVs,
entry_time + decision_price) simulate on 1-min bars:

- **Style A — cross at decision:** cost = half_spread(state) + within-minute drift proxy.
- **Style B — limit at decision price, timeout T, then cross:** for T ∈ {1, 2, 5, 10, 15, 30,
  60 min}: filled at the limit if the bar low trades ≤ limit within T (run both `≤` and `<` as
  a sensitivity); otherwise cross at the price at decision+T. Cost = fill vs decision, + spread
  where crossed.

Outputs: entry-cost distribution per style/T (mean, σ, p90), overall and by regime and
time-of-day; the **optimal timeout curve** (expected cost vs T); reconciliation vs the live
+14 bps (S08) and vs the ~2 bps average 15-min drift (S12) — the gap is the adverse-selection
component, quantify it. Tier: **Modeled** (1-min bars can't see intraminute queue dynamics — no
queue-position model; state the assumption prominently).

**Done when:** the cross-vs-chase decision and the optimal-T curve are stated with full-history
statistics, and the S08 live numbers sit inside the simulated distribution.

### P03 — The predictor (`research/predictor/03_predictor` + `slippage/predict.py`)

Package the client's "predict diff" function:

```
predict_slippage(notional_usd, side, order_type, state: MarketState, *, latency_min=...)
  → {mean_bps, p50, p90, p95, components: {spread, impact_band, drag, timing_sigma}}
```

- spread from `state` (measured NBBO when supplied — the live override); impact from the √-law
  band (envelope with Almgren after C01 lands — leave a hook); drag from P02 keyed by
  `order_type` ("cross" ≈ 0, "limit_chase(T)" from the P02 curve); timing quantiles from
  σ(latency) per S03, regime-scaled.
- Quantiles combine the mean components with the timing distribution (leptokurtic — use the
  empirical S03 quantiles per horizon, not a Gaussian).
- Tests: components match their source stages at reference points; band ordering (p50 < p90 <
  p95); order-type monotonicity (chase ≥ cross in mean).
- Findings: worked examples table ($100k / $1M / $10M × calm/normal/stress × cross/chase).

### P04 — Recurring TCA monitor (`research/predictor/04_tca_monitor` + `scripts/tca_monitor.py`)

Generalize the S08 parser into a rerunnable monitor:

- Input: the live-log directory (idempotent — keeps a processed-files ledger; appends to a
  running `fills.csv`).
- Per fill: realized slippage vs `predict_slippage` prediction under the logged conditions;
  rolling 20-fill mean and σ per side/order-type; **alert** (nonzero exit code + printed summary)
  when the rolling mean leaves the model's 2σ band.
- One command: `python scripts/tca_monitor.py <log_dir>`. Findings document the baseline run over
  the existing 572 logs and the alert thresholds. How/when to schedule it recurrently is the
  owner's choice — document, don't implement scheduling.

---

## Phase 2 — Scheduler track

### E01 — Alpha-decay curve (`research/scheduler/01_alpha_decay`)

Fix the S12-0a soft spot and produce the scheduler's key input:

- Recompute the "fraction of the eventual move forfeited by an entry delay h" table
  **conditioned on the P&L-bearing trades (hold ≥ 1 day)**, plus a P&L-weighted variant (the
  current table mixes hold lengths and its 4h/1d rows are dominated by short holds).
- Fit a smooth `g(h)` = expected fraction of per-trade edge forfeited by delaying entry h
  (bootstrap CI), per symbol.
- **Deliverable:** `g(h)` as a callable (keep it in the stage folder or `slippage/state.py` —
  wherever E03 can import it cleanly) + findings.

### E02 — Interruption risk (`research/scheduler/02_interruption_risk`)

The two mid-fill hazards: the trailing stop fires, or the signal flips, while the order is still
being worked. Both are measurable from the existing logs — no live experiment needed for v1:

- From TRADES CSVs: the hazard curve **P(trade exits within h of entry)**, overall, by regime,
  and by `exit_reason` (TRAIL_STOP vs others). (Context: p25 hold ≈ 2.8 h — roughly a quarter of
  trades are gone within ~3 h, so this is not a tail concern.)
- A simple interruption-cost model: if interrupted at filled-fraction φ, the residue is either
  cancelled (forfeit the remaining edge) or the exit itself must now be executed at the same
  participation cost. Parametrize both; keep it simple and label **Modeled**.
- **Design for later experiments (owner requirement):** the deliverable is
  `interruption_hazard(h, state)` + `interruption_cost(h, φ, mode)`, and E04's replay harness
  must accept an **event stream** (timestamps of stop-fires / signal-flips) so someone can later
  feed backtest or live events instead of the historical averages. Leave that interface ready and
  documented.

### E03 — The scheduler (`slippage/schedule.py` + `research/scheduler/03_scheduler`)

The centerpiece. Extends (does not replace) `plan_execution`:

```
schedule_order(notional_usd, side, state: MarketState, *, edge_bps, pov_cap=0.10, mode=...)
  → Schedule: slices [(time_offset, child_notional, order_style)], horizon h*,
              expected_slippage_bps band, alpha_forfeit_bps, interruption summary, assumptions
```

- **Horizon choice:** h* = argmin over h of
  `expected_slippage(notional, h, state) + edge_bps · g(h) + E[interruption_cost(h)]`
  — impact saved by going slower vs alpha forfeited (E01) vs interruption risk (E02). This
  replaces the hard 15-min pin everywhere in scheduler-land (owner decision, backed by S12:
  the edge is multi-day; the profitable trades hold ≥ 1 day).
- **Slice allocation:** across h*, child sizes proportional to P01's predicted interval volume
  (VWAP-shaped), each capped at `pov_cap` of the predicted interval volume (guardrail; default
  10%, configurable).
- **Order style per child:** from P02's result (expected: "cross"; carry the limit+timeout
  option through anyway).
- Tests: monotonicity (bigger order → longer h*, more slices; thinner predicted volume → smaller
  children); small-order limit degenerates to a single cross ≈ `plan_execution` agreement;
  g ≡ 0 and no interruption → pure cost minimization; POV cap never violated.
- Findings: worked schedules ($1M / $5M / $20M TQQQ, normal vs stress), each with the cost band
  and what a naive 15-min single fill would have cost instead.

### E04 — Historical replay harness (`research/scheduler/04_replay`)

Prove the scheduler on history and leave the experiment rig behind:

- Replay `schedule_order` at every historical entry (1-min bars): simulated schedule cost vs
  (a) single 15-min fill baseline, (b) day-VWAP baseline. Interruptions replayed from the actual
  trade exits (the event-stream interface from E02).
- Output: realized-cost-vs-size table (sweep notional $100k → $20M), by regime; findings with the
  headline "what the scheduler saves at each size".
- This harness is deliberately the "code ready for someone to do backtest/live experiments"
  artifact: document its event-stream input and how to point it at new data.

---

## Phase 3 — Capacity track add-ons

### C01 — Almgren temporary + two-model envelope (`research/capacity/01_almgren_envelope`)

The useful core of the paused Almgren plan, post-S12 (permanent term dropped — wrong mechanism
for elastic-supply ETFs):

- `slippage/impact.py`: add `almgren_temporary(participation, sigma_bps, *, eta=0.142, beta=0.6)`
  with named constants + paper citation. Keep the √-law functions untouched.
- **Golden test** (tests/test_impact.py): reproduce Almgren-Thum-Hauptmann-Li Table 3 temporary
  impact to <1 bp — IBM (V=6.561M sh, Θ=1728M sh, σ=1.57%/day): J = 32/25/18 bps at
  T = 0.1/0.2/0.5 days; DRI (V=1.929M sh, σ=2.26%): J = 43/32/23 bps at the same T.
- Thread an `impact_model` parameter ("sqrt" default | "almgren") through
  `expected_slippage_bps` → `CostModel` → `predict_slippage`; add an `"envelope"` reporting mode:
  band = min/max across {√-law at Y=0.3, Y=1.0, Almgren temporary}.
- Findings: before/after capacity table with **σ held identical** across models (so the movement
  is the coefficient/exponent choice, nothing else) + a reconciliation paragraph (expect Almgren
  to sit near/below the √-law's optimistic Y=0.3 edge at typical participations).
- **The library default stays "sqrt". Flipping it is an owner (GATE) decision — do not decide.**

### C02 — Capacity curve refresh (`research/capacity/02_capacity_refresh`) — run LAST

Re-run the S06 capacity chain with everything learned:

1. **Envelope band** (C01) instead of Y-band alone.
2. **Added metrics** alongside Sharpe: return-on-AUM, max drawdown, Sortino (all from the daily
   P&L series build_06 already constructs). Present Sharpe + return-on-AUM + maxDD; keep the
   table readable.
3. **Edge-sensitivity row:** capacity thresholds recomputed at 50% of the backtest gross edge
   (the gross Sharpe 4.9 may be in-sample-flattered; under the √-law, halving the edge quarters
   capacity — the reader should see that).
4. **Execution basis:** if E03 is done, cost each trade at the scheduler's chosen horizon instead
   of the 15-min pin (the honest basis after S12); otherwise run with the pin and a footnote.

Findings update FINDINGS.md's headline table (keep the old table as "previous, 15-min-pinned"
for continuity).

---

## Order of work and dependencies

```
0.1, 0.2                            (conventions + rename; fast)
P01, P02, E01, E02                  (independent, all local-data; any order / parallel)
P03   ← P01, P02
P04   ← P03
C01                                 (independent; anytime)
E03   ← P01, P03, E01, E02
E04   ← E03
C02   ← C01 (+ E03 preferred)       (run last)
```

## Owner decisions (flag, don't decide)

- C01 GATE: whether "almgren"/"envelope" becomes the default impact model.
- POV cap default (10% proposed) and any urgency/λ defaults in `schedule_order`.
- Adopting the P02 timeout/cross recommendation in the **live engine** (separate repo).
- When/how to run the TCA monitor recurrently.

## Acceptance (end-to-end)

1. All tests green; every stage's `build_*.py` regenerates its `results/` and matches findings.
2. `predict_slippage` answers the client's question: for a given order and state, a mean and
   quantiles, with components labeled by evidence tier.
3. `schedule_order` produces a full child-order schedule for a $5M TQQQ order that a reader can
   sanity-check by hand from the findings.
4. `scripts/tca_monitor.py` runs over the existing 572 logs and reproduces S08's +14 bps buy
   drag as its baseline.
5. ROADMAP.md, BUILD_LOG.md, FINDINGS.md, HANDOFF.md, slippage/README.md all updated; canonical
   IDs used throughout.
