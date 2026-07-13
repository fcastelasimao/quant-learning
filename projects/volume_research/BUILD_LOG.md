# BUILD_LOG — volume_research (slippage / market-impact)

**Numbering note:** canonical stage IDs are the `research/` folder numbers (table in
[ROADMAP.md](ROADMAP.md)). Entries below keep their historical titles ("Block N" / "Stage N");
each heading is now prefixed with its canonical ID.

Working mode: **normal fast-mode.** Claude writes the code; this log tracks decisions, blocks,
and open questions so the work stays legible across sessions.

---

## Decisions
*One line each. The choices we've locked in and why, so future-you doesn't re-litigate them.*

- Spread estimator = **Corwin–Schultz (2012)** on **15-min bars** (daily-bar CS is unusable for
  these ultra-liquid ETFs — spread is swamped by the daily range; signed mean goes negative).
- **Aggregate signed, clamp the aggregate** (not per pair) — per-pair clamping biases the mean
  badly upward.
- Intraday pairs are **within-session only** (no overnight-boundary pairing).
- Env = **quant** conda env; estimator lives in `slippage/spread.py` (start of the cost library).

## Open questions
*Things we deferred or haven't resolved. Pull from here when picking the next block.*

- Stress-year spreads (2020/2022 ≈ 3.5 bps) conflate genuine widening with CS's volatility bias —
  worth a TAQ or 1-min cross-check if the spread term ever becomes binding (it currently isn't).

### Closed questions

- **Can CS run on 15-min bars (consecutive *bars*, not days)?** Yes — within-session pairing works.
  (Block 1.)
- **What bar resolution for CS?** Transition study (1/5/10/15-min, all resampled from 1-min): the
  signed mean climbs monotonically −1.48 → −0.46 → +0.30 → +0.77 and **never plateaus**. Below ~10-min
  CS is broken (negative, discreteness/bounce dominates); threshold is volatility-dependent (QQQ
  recovers at 5-min, TQQQ/SQQQ need 10-min+). CS pins the *magnitude* (sub-bp to ~1 bp half), not the
  exact level. **Use native 15-min** (resampled-15 ≈ native-15, consistency confirmed). Don't expect a
  precise level from CS — would need Roll / Abdi–Ranaldo, but spread isn't binding so not worth it.

---

## Blocks

### AUDIT FIXES #1 + #2 — stretched-exp g(h) + objective double-count ✅ DONE (2026-07-09)
*Landed the first two of the three 2026-07-09 audit findings (fix #3, the E04 caveat, is separate).*

- **Fix #1 — E01 `g(h)` is now a stretched exponential** `1-exp(-(h/τ)^k)` (was a single
  exponential, k≡1). Refit in `build_01_alpha_decay.py` (τ and k jointly, k∈[0.3,6], 500-boot CI
  on both). New constants baked into `slippage/alpha_decay.py`: TQQQ (τ=256.9, k=2.480), SQQQ
  (τ=251.7, k=2.723); pnl-weighted TQQQ (256.2, 1.636), SQQQ (236.5, 1.233). The single-exp
  overcharged the 15-120 min band 2.5-3.5x (g(15m)=5.1% fitted vs 1.6% empirical); the stretched
  form nails the knee (g(120m)=14.0% vs 13.8%). Residual: it slightly *under*-charges below ~60min
  (g(60m)=2.7% vs 6.9%), tiny in bps and biases h* slightly long, not short — the opposite,
  smaller direction. Test reference points updated in `tests/test_alpha_decay.py`.
- **Fix #2 — removed the alpha double-count in the horizon objective.** `schedule.py::_objective`
  charged `edge·g(h)` unconditionally PLUS the interruption term, but `interruption_cost` (cancel
  mode) already prices the filled part's `g(h)` on the interrupted branch. Clean expectation:
  `edge·[(1-hazard)·g(h) + hazard·interruption_cost(h)]`. Factored a shared helper
  `schedule.alpha_interruption_bps(h, state, edge, mode)` returning the two channels, and routed
  `_objective`, `schedule_order`'s reporting, **and** the two research builds that re-derived the
  same formula (E03's naive baseline, C02's `cost_at_scheduler_horizon`) through it — so the
  decomposition can't drift (this exact double-count trap recurred 3x, per the handoff warning).
  `Schedule.alpha_forfeit_bps` now reports the completion-branch `edge·(1-hazard)·g(h)`.
- **Downstream reruns:** both fixes bias the old numbers conservative (anti-scheduler), so with
  them fixed h* extends further and the scheduler wins more. E03: worked-example improvement grew
  from +0.9→+48.7 bps to **+3.7→+58.5 bps** (h*=45-120 vs 25-105 min). C02: `alpha+interrupt`
  add-on roughly halved (TQQQ $30M: 20.1→11.1 bps), so central Sharpe-zero AUM ~doubled, **~$30M →
  ~$65M** (TQQQ; SQQQ ~$37M), envelope ~$13M-$300M+. Updated findings_01/03/C02, FINDINGS.md,
  ROADMAP.md, docs/briefing_2026-07.md. All 171 tests still pass.

### C02 — Capacity curve refresh ✅ DONE (RAN LAST, per plan)
*Status: complete. See `research/capacity/02_capacity_refresh/findings_02_capacity_refresh.md`.*

- **Goal:** re-run S06's capacity chain with everything learned: envelope band (C01), added risk
  metrics, an edge-sensitivity row, and the scheduler's chosen horizon (E03) instead of the
  15-min pin — the final stage of `plans/2026-07_execution_track.md`.
- **What I wrote:** `research/capacity/02_capacity_refresh/build_02_capacity_refresh.py`
  (envelope = min/max{sqrt Y=0.3, Y=1.0, Almgren} at `h*`; Sortino/maxDD/return-on-AUM from the
  full daily net-P&L path, not just mean/var; 50%-edge sensitivity column; `h*` from
  `schedule.py::_choose_horizon` using each ticker's own **measured** average trade edge) +
  findings; updated `FINDINGS.md`'s headline section (new table + explicit "previous,
  15-min-pinned" table kept for continuity, per instruction).
- **Bug caught and fixed before the headline was reported:** the first pass costed trades at h*
  using **execution cost alone**, silently banking the benefit of a longer horizon without
  paying the alpha-forfeiture/interruption cost that horizon actually incurs — exactly the
  mistake E04 exists to catch. Fixed to charge the full `execution + alpha + interruption` total
  (matching `schedule.py`'s own `_objective`) at every AUM/band edge.
- **Result:** capacity extends **moderately, not dramatically** — TQQQ's Sharpe-zero AUM moves
  from ~$14M (S06) to ~$30–100M depending on envelope edge, and roughly halves again under the
  50%-edge sensitivity. The envelope band is wide (C01's Almgren/√-law divergence carries
  through). No new library code (pure research composition); 171 tests unchanged. **This
  completes the entire 12-stage plan.**

### E04 — Historical replay harness ✅ DONE
*Status: complete. See `research/scheduler/04_replay/findings_04_replay.md`.*

- **Goal:** prove `schedule_order` (E03) on real history, replaying interruptions from actual
  trade exits (the event-stream interface E02's docstring promised) instead of the modeled
  average hazard, and leave a reusable replay artifact behind.
- **What I wrote:** `research/scheduler/04_replay/build_04_replay.py` — every historical
  TQQQ/SQQQ entry (2,315 / 1,904 aligned trades) × a $100k-$20M notional sweep (16,876
  trade×size events), real `MarketState` construction (real trailing-120min sigma nowcast, real
  15-min bin volume, real regime), schedule replayed against real 1-min closes, interrupted at
  the trade's *actual* exit_time; two real-data baselines (naive 15-min single fill, day-VWAP).
- **Result — qualifies E03's optimism:** at $100k-$1M the scheduler clearly wins (TQQQ 8.3-15.4
  bps vs 21.4/18.6 bps baselines, 94%+ filled). At $5M fill rates start dropping (~79%) as longer
  horizons meet real trailing-stop exits. **At $20M TQQQ, scheduled cost (32.2 bps, only 60%
  filled) is *worse* than both baselines** — E03's worked examples assumed full completion; real
  interruption events bite harder than the modeled average hazard implied at large size. Also
  surfaced an unexplained pattern (fill rate lowest in calm, highest in stress — opposite the
  naive expectation) flagged for follow-up, not force-fit into a narrative. No new library module
  (pure replay script); 171 tests unchanged.

### E03 — The scheduler: schedule_order() ✅ DONE
*Status: complete. See `research/scheduler/03_scheduler/findings_03_scheduler.md`.*

- **Goal:** the centerpiece — extend `plan_execution` from picking a participation rate at a
  fixed cadence to picking the **horizon itself**, trading off execution cost (P01-P03) against
  alpha-forfeiture (E01) and interruption risk (E02), then slicing VWAP-shaped and POV-capped.
- **What I wrote:** `slippage/schedule.py` (`schedule_order()` → `Schedule`/`ScheduleSlice`;
  `h* = argmin_h [predict_slippage(...).mean_bps + edge_bps·g(h) + hazard(h)·interruption_cost(h,
  0.5, mode)·edge_bps]`; VWAP-shaped waterfall slice allocation respecting `pov_cap`, extending
  the schedule past h* if needed to fit, `feasible=False` beyond a hard bin cap);
  `research/scheduler/03_scheduler/build_03_scheduler.py` (worked schedules: TQQQ $1M/$5M/$20M ×
  normal/stress vs a naive 15-min single fill, on a like-for-like total-objective basis) +
  findings; `tests/test_schedule.py` (12 tests: monotonicity, small-order degeneracy, `edge_bps=0`
  → pure cost minimization, POV cap never violated, infeasibility, slice-sum consistency).
- **Result:** scheduling never makes the total objective worse (h=15 is itself in the search
  grid) and the benefit scales sharply with size/stress — from +0.9 bps ($1M/normal) to
  +48.7 bps ($20M/stress). Caught and fixed a comparison bug while drafting findings: comparing
  only raw execution-cost savings against alpha-forfeiture (omitting the naive baseline's own
  small alpha/interruption terms) is not apples-to-apples — fixed to compare full totals.
  159 → 171 tests.

### C01 — Almgren temporary + two-model envelope ✅ DONE
*Status: complete. See `research/capacity/01_almgren_envelope/findings_01_almgren_envelope.md`.*

- **Goal:** the useful core of the paused Almgren-adoption plan post-S12 (permanent term
  dropped): implement the temporary-impact term, verify it against the paper exactly, thread
  `impact_model` through the pipeline, add an envelope reporting mode — **without** flipping the
  library default (owner/GATE decision).
- **What I wrote:** `slippage/impact.py::almgren_permanent`/`almgren_temporary` (named, cited
  constants); golden test in `tests/test_impact.py` (all 6 IBM/DRI Table-3 reference points, <1bp);
  threaded `impact_model=("sqrt"|"almgren")` through `cost.py::expected_slippage_bps`/
  `optimal_participation` → `model.py::CostModel.roundtrip`/`roundtrip_optimal`; implemented the
  `"almgren"`/`"envelope"` branches in `predict.py` (removing the P03 `NotImplementedError`
  placeholder); `research/capacity/01_almgren_envelope/build_01_almgren_envelope.py` (before/after
  cost + capacity tables, σ/ADV/spread held identical) + findings; updated `slippage/README.md`'s
  API table (also backfilling P03/E01/E02's additions, not previously documented there).
- **Result:** golden test exact on all 6 points. At TQQQ/SQQQ's actual σ/ADV and the strategy's
  forced 15-min cadence, Almgren predicts **~7.4× more capacity** than even sqrt's optimistic
  Y=0.3 edge — both models are extrapolating outside their fitted domains here (Almgren: large-cap
  single stocks, multi-hour horizons, ≤10% ADV; sqrt: Y never fit to any data), so this is the
  honest width of the uncertainty, not a preference for either model. **Library default stays
  "sqrt"; not flipped.** 140 → 159 tests (golden test ×6 parametrized + almgren/envelope +
  cost.py/model.py impact_model coverage).

### E02 — Interruption risk: hazard curve + cost model ✅ DONE
*Status: complete. See `research/scheduler/02_interruption_risk/findings_02_interruption_risk.md`.*

- **Goal:** measure the mid-fill interruption hazard (trailing stop / signal flip) and give the
  scheduler a simple cost model, with an event-stream interface left ready for E04.
- **What I wrote:** `slippage/interruption.py` (`interruption_hazard(h_min, state)` — baked
  empirical hazard curves, overall + regime-conditioned; `interruption_cost(h_min, phi, mode,
  symbol)` — "cancel" vs "complete_now", both composing with E01's g(h)); `research/scheduler/
  02_interruption_risk/build_02_interruption_risk.py` (hazard curve from TRADES CSVs' hold_days,
  by regime, by exit_reason) + findings; `tests/test_interruption.py` (13 tests).
- **Result:** p25 hold ≈ 2.7-2.8h (matches the plan's own context note); regime barely moves the
  hazard (<5pp spread, no clean ordering — unlike cost-magnitude measurements elsewhere);
  >99.9% of exits are TRAIL_STOP (no separately-observable "signal flip" in this dataset, flagged
  prominently). 127 → 140 tests.

### E01 — Alpha-decay curve: g(h) ✅ DONE
*Status: complete. See `research/scheduler/01_alpha_decay/findings_01_alpha_decay.md`.*

- **Goal:** fix S12-0a's soft spot (its longer-delay rows mixed all hold lengths, dominated by
  trades too short to have survived to that horizon) and produce the scheduler's alpha-forfeiture
  input.
- **What I wrote:** `slippage/alpha_decay.py::alpha_forfeit_frac(h_min, symbol, pnl_weighted=)` —
  pure, baked per-symbol τ constants; `research/scheduler/01_alpha_decay/build_01_alpha_decay.py`
  (recomputes S12's delay-fraction table conditioned on hold ≥ 1 day, adds a pnl-weighted
  variant, fits `g(h)=1-exp(-h/τ)` via `scipy.optimize.curve_fit` with a 500-resample bootstrap
  CI) + `findings_01_alpha_decay.md`; `tests/test_alpha_decay.py` (9 tests: boundary conditions,
  monotonicity, matches the fitted reference points, per-symbol distinctness).
- **Result:** flat through the first hour (2-7% forfeited), a sharp knee at 2-4h (14%→56%),
  saturated by end-of-day (96-101%) — conditioning on the P&L-bearing subset (hold≥1d) cleanly
  fixed S12's mixing artifact. τ ≈ 288 min (TQQQ) / 274 min (SQQQ), tight bootstrap CI, pnl-weighted
  variant close to unweighted (not driven by a few outsized trades). 118 → 127 tests.

### P04 — Recurring TCA monitor ✅ DONE
*Status: complete. See `research/predictor/04_tca_monitor/findings_04_tca_monitor.md`.*

- **Goal:** generalize S08's one-off parse into a rerunnable, idempotent monitor comparing
  realized vs. `predict_slippage`-predicted fills, with drift alerting.
- **What I wrote:** `scripts/tca_monitor.py` (idempotent ledger + appended `fills.csv`;
  per-fill realized-vs-predicted residual; rolling-window 2σ drift check, nonzero exit code on
  breach) + `research/predictor/04_tca_monitor/findings_04_tca_monitor.md`;
  `tests/test_tca_monitor.py` (9 tests: parsing, idempotency, drift-alert logic on synthetic
  data with independent per-group detection).
- **Result:** the baseline run over the same 572 logs S08 used **independently reproduces**
  S08's +14.2 bps buy-limit realized mean exactly, and the residual against the monitor's own
  prediction (+5.3 bps) matches P02's separately-measured adverse-selection gap (5.5 bps) — two
  independently-built stages landed on the same number. Drift check correctly skips every group
  (only 10/18/2 fills, below the 20-fill window) rather than false-triggering on a small sample;
  idempotency confirmed live and by test. 109 → 118 tests.

### P03 — The predictor: predict_slippage() ✅ DONE
*Status: complete. See `research/predictor/03_predictor/findings_03_predictor.md`.*

- **Goal:** the client's "I set an order at X and filled at X + diff — predict diff" function.
- **What I wrote:** `slippage/predict.py` (`predict_slippage(notional_usd, side, order_type,
  state, price, latency_min=15)` — composes P01's `MarketState`, P02's measured chase-cost curve,
  and S03's empirical non-Gaussian timing quantiles into `{mean_bps, p50/p90/p95_bps,
  components}`, pure, no IO); `research/predictor/03_predictor/build_03_predictor.py` (worked
  examples: $100k/$1M/$10M × calm/normal/stress × cross/chase, TQQQ) + `findings_03_predictor.md`;
  `tests/test_predict.py` (15 tests: each component matches its source stage at a reference
  point, p50<p90<p95 always, chase≥cross in mean).
- **Result:** timing dominates at every size (a calm $100k cross has mean 2.8 bps but p95
  59 bps); chase costs more than cross in every grid cell (P02's finding surfaced at the
  predictor level); impact only bites at $10M+ (15-20 bps, comparable to the chase drag itself).
  Timing quantile multipliers (p90/p95 = 0.98x/1.44x sigma) are measured directly from S03's own
  method — notably *below* the Gaussian's 1.28x/1.64x at p90 (thin shoulder) despite the
  documented "fat tails" (which bite further out than p95). `impact_model="almgren"` raises
  `NotImplementedError` — an explicit hook for C01, not yet landed. 94 → 109 tests.

### P02 — Chase simulation: cross vs chase, full history ✅ DONE
*Status: complete. See `research/predictor/02_chase_simulation/findings_02_chase_simulation.md`.*

- **Goal:** replace S08's n=15 live limit-chase estimate (+14.2 bps) with a full-history
  simulation, and answer "if we still chase, what timeout?"
- **Data bug found and fixed:** the TRADES CSVs' own `decision_price` (FMP-sourced) is not on the
  same split-adjustment basis as this project's Alpaca-sourced DB candles (~2.0x ratio for TQQQ,
  ~0.2x for SQQQ across most of history, converging only near 2026) — mixing them produced
  thousand-bps nonsense. Fixed by using only the DB's own 1-min bar open as "decision price"
  (S12's 0a convention). Also fixed a fill-window bug: a bar's low is tautologically ≤ its own
  open, so the wait window must start strictly after the decision bar or every limit "fills"
  instantly.
- **What I wrote:** `research/predictor/02_chase_simulation/build_02_chase_simulation.py`
  (simulates Style A cross-now vs Style B passive-limit-then-cross at T ∈
  {1,2,5,10,15,30,60 min} on every historical entry) + `findings_02_chase_simulation.md`.
- **Result:** crossing beats chasing at every T (TQQQ 2.88 vs 8.66 bps @T=15; SQQQ 4.48 vs
  13.22 bps) — confirms S08's live conclusion at 150x the sample size. S08's live +14.2 bps sits
  inside the simulated distribution (mean±1σ) for both tickers. Style-B cost is non-monotone in
  T (rises then plateaus — fill-rate and adverse-drift effects fight each other). Clean monotonic
  calm<normal<stress regime ordering. No new tests (build script only, reuses tested library
  functions); 94 tests unchanged.

### P01 — Market state: volume profile, predictability, vol nowcast, spread state ✅ DONE
*Status: complete. See `research/predictor/01_market_state/findings_01_market_state.md`.*

- **Goal:** the client's "depending on the state of the market / market volume" input — package it
  behind one pure function the predictor (P03) and scheduler (E03) can call.
- **What I wrote:** `slippage/state.py` (`MarketState`, `VolumeProfile`, `SpreadCurve`,
  `VolRegimeBounds`, `estimate_state()` — pure, no IO);
  `research/predictor/01_market_state/build_01_market_state.py` (measures the three reference
  curves from real 15-min/1-min/daily bars) + `findings_01_market_state.md`;
  `tests/test_state.py` (synthetic recovery + a DB-guarded real-data sanity check, skips cleanly
  without the DBs, mirroring `test_calibrate_live.py`). Exported from `slippage/__init__.py`.
- **Result:** U-shaped volume profile confirmed (open bin **4.2x** the midday lull, TQQQ/SQQQ);
  bin-volume predictability is real but modest (OOS R² 0.09–0.24, simple time-of-day×EWMA
  baseline); the intraday vol nowcast agrees with S03's daily regime label ~51% of the time
  (well above ~33% chance, and expected to be well below 100% — different time horizons by
  design); the spread-state curve **reproduces S01's published intraday numbers exactly**
  (2.53/2.76/1.19 bps at the open for TQQQ/SQQQ/QQQ). 94 tests (85 → 94).

- **Does the spread change over time / by regime, and does the ramp plateau by 30-min?**
  (`diagnose_resolution.py`, sweep 1–30 min × 5-yr window × vol regime.) (a) **No plateau** — the
  estimate keeps rising through 30-min. (b) **Time variation is volatility-driven**, not secular
  tightening (2020–2024 highest; 2010–2014 negative = CS fails on thin early data, don't use
  pre-2015). (c) **Clear monotonic vol-regime ordering** stress>normal>calm, but conflated with CS's
  vol bias. (d) **Robustness:** across all resolutions/eras/regimes the half-spread tops out ~3.7 bps,
  never reaching the flat 5 bps.

### S12 — Almgren-adoption plan, Stage 0: resolve the two forks (was "Stage 14") ✅ DONE
*Status: complete. See `research/12_stage0_forks/findings_12_stage0_forks.md`.*

- **Goal:** before adopting Almgren et al. (2005)'s calibrated impact coefficients, resolve two
  upstream unknowns that each gate/delete a downstream stage: (0a) is the strategy's edge intraday
  or multi-day (does the creation/redemption door even have time to fire)? (0b) is Almgren's
  single-stock `(Θ/V)^{1/4}` permanent-impact term even the right mechanism for an arbitrage-pinned,
  elastic-supply ETF?
- **What I wrote:** `research/12_stage0_forks/build_12_stage0_forks.py` — 0a: forward-return decay
  + delay-cost + P&L-by-hold-bucket on real 15-min data anchored at the `TQQQ_SQQQ_analysis`
  backtest's entry times; 0b: event study on real Alpaca `/trades` tick data (excess return vs a
  3×QQQ proxy, short vs long horizon around large prints) + `findings_12`.
- **Result 0a — multi-day, not intraday.** Forward returns grow monotonically through 10 days (no
  spike-then-fade); a 30–60 min entry delay gives up only ~11–26% of a trade's eventual move;
  trades held **<2.4h are net losers** on both symbols (−10%/−32% of total P&L) while **1–5+ day
  holds carry ~75–90% of the profit**. **Phase 2 (creation door) stays in the plan.**
- **Result 0b — partial confirmation, not a hard zero.** Reasoning: AP creation/redemption arbitrages
  away any TQQQ screen deviation from `3×NAV` within a trading day — Almgren's fixed-float permanent
  term is the wrong mechanism for an elastic-supply ETF. Empirically (11k+ large TQQQ prints, real
  ticks): only ~28% of the initial excess-return reverts by 30 min, leaving a small but
  statistically nonzero residual (~0.26 bps) — two orders of magnitude below Almgren's single-stock
  prediction but not literally zero. **Stage A omits Θ/V on TQQQ's own float; carries a tiny
  near-field-anchored permanent term instead (tier: Modeled/no-ground-truth). A.5 (accumulation)
  downgraded from a full stage to a bounded check.**

### S11 — CS-15min accuracy validation + calibrate NaN guard (was "Stage 13") ✅ DONE
*Status: complete. See `research/11_cs_validation/findings_11_cs_validation.md`.*

- **Goal:** is CS-15min's ~0.7–1 bp *accurate* (not just self-consistent)? And harden `calibrate`
  against a NaN footgun found while reviewing (`max(nan,0.0)==nan` silently poisons MarketParams).
- **What I wrote:** `research/11_cs_validation/build_11_cs_validation.py` (Tier 1: tick-floor +
  1-min-vs-15-min CS; Tier 2: real **Alpaca SIP NBBO** quotes, time-weighted, vs CS) + `findings_11`;
  `calibrate._aggregate_half_spread` (raises on empty/non-finite, warns on high-NaN-fraction) +
  6 guard tests.
- **Result — CS-15min validated against real NBBO:** TQQQ 0.74 vs SIP 0.89, SQQQ 1.00 vs 1.32
  (within ~25%, conservatively low → accurate for the traded names); CS **overstates QQQ ~2×**
  (0.72 vs 0.35 — high-price bias, immaterial: QQQ not traded, overstatement is conservative).
  1-min CS collapses to 0 → confirms 15-min is the only working resolution. Spread stays a small
  **non-binding** floor. 79 → 85 tests.
- **Note:** SIP access confirmed on the live Alpaca account (real NBBO, not just free IEX).

### S10 — EDGE spread estimator (opt-in) (was "Stage 12") ✅ DONE
*Status: complete. See `research/10_edge_vs_cs/findings_10_edge_vs_cs.md`.*

- **Goal:** a colleague pasted a (broken, non-EDGE) `edge_spread_estimator`. Implement the *real*
  EDGE (Ardia–Guidotti–Kroencke 2024) and decide whether it should replace Corwin–Schultz.
- **What I wrote:** faithful numpy `edge()` + per-session `edge_intraday()` in `slippage/spread.py`
  (matches the authors' `bidask.edge` **exactly**, diff=0); `calibrate(spread_method=…)` (default
  **"cs"**); tests (`test_spread.py` EDGE recovery + faithfulness-vs-`bidask` + intraday;
  `test_calibrate.py` default-cs / edge-runs / rejects-bad); gate probe
  `research/10_edge_vs_cs/build_10_edge_vs_cs.py` + `findings_10`. Answered the `_overnight_shift`
  inline `Q:` in its docstring.
- **Decision (gate caught it):** **keep CS default; EDGE is opt-in.** EDGE is unreliable at 15-min
  — underpowered per-session (~0) and overnight-gap sensitive across sessions (0→10 bps depending
  on aggregation). Its domain is daily bars / long gap-free samples. Spread is a non-binding floor
  here anyway → **no re-baseline** of the capacity chain. 69 → 79 tests.

### S09 — `calibrate()` validation + σ mean/median (was "Stage 11") ✅ DONE
*Status: complete. See `research/09_calibrate_validation/findings_09_calibrate_validation.md`.*

- **Goal:** confirm `calibrate()` reproduces the published findings from the *real* DBs (its one
  path exercised only by synthetic frames), not just recover synthetic inputs.
- **What I wrote:** probe `research/09_calibrate_validation/build_09_calibrate_validation.py`
  (loads real daily+15-min, checks vs findings_04/01, writes `calibration_check.csv`); guarded test
  `tests/test_calibrate_live.py` (5 checks × 3 tickers, `skipif` no DB); `findings_09`.
- **Result:** $ADV / σ_stress / half-spread reproduce to <1%. The lone mismatch — normal σ +9.2% —
  was a **mean-vs-median summary choice** on the right-skewed rolling-vol series, not a bug.
- **Decision (both):** report **both** σ summaries. `calibrate()` now returns `sigma_daily_bps`
  (mean, the expected-cost headline, used by `normal`) **and** `sigma_daily_median_bps` (typical-day
  sensitivity). Re-baselined build_04/05/06/07 to lead with the mean; findings show both. Capacity
  central ~$17M → ~$14M (TQQQ), edge-gone ~$25M → ~$14M — still well inside the 12× Y-band, so the
  verdict is unchanged. 54 → 69 tests.

### L02 — Execution planner (was "Stage 10") ✅ DONE
*Status: complete. See `slippage/README.md` ("Planning an execution").*

- **Goal:** the function the client asked for — input a trade, output how fast / how to slice.
- **What I wrote:** `slippage/plan.py` (`plan_execution` + `ExecutionPlan`), exported in `__init__`;
  `tests/test_plan.py` (5 passing).
- **Behaviour:** A–C optimal participation (via `optimal_participation`) capped at the 15-min
  cadence; returns POV, horizon, slice plan, expected cost (Y-band), timing 1σ, and a `feasible`
  flag (False if the order needs >100% of cadence volume). $5M → ~6% POV / ~6 min / 6 slices /
  ~43 bps. **Pre-trade planner, not a live router** (the broker still slices/routes).
- **Caveat surfaced:** the √-law over-extrapolates impact for small fast clips — cost number is
  meaningful from ~$1M up; at retail "just cross." Documented in plan.py + README + findings_08.

### S08 — Live-fill validation (was "Stage 9") ✅ DONE
*Status: complete. See `research/08_live_validation/findings_08_live_validation.md`.*

- **Goal:** validate the model against the live Alpaca logs (572 cycles, 2026-05-22→06-24).
- **What I wrote:** `research/08_live_validation/build_08_live_validation.py` (log parser + analysis
  + plot). No new tests (parser over external logs).
- **Key results:** 30 fills (15 buy / 15 sell) + 1 unfilled limit sell, $79k–181k. **Buy slippage
  mean +14 bps (1σ 35), sell ≈0.** Three findings: (1) magnitude is timing-scale, no impact signal
  at retail (√-law over-extrapolates impact for small clips — don't read it literally); (2) buy 1σ
  35 sits between the 1-min/5-min model timing → chase latency a few min; (3) **buy mean is +14 bps
  adverse, not 0 → W3 resolved: this is momentum, delay is a SIGNED drag.** Sells (market) clean;
  1 limit sell never filled = live opportunity cost (W3 price-guard).
- **Decision:** apply the timing term as a *signed adverse* entry drag for this momentum strategy.
  Live fills validate spread+timing only (too small for impact/Y); sanity check, not calibration.

### L01 — Package the slippage library (was "Block 8") ✅ DONE
*Status: complete. See `slippage/README.md`.*

- **Goal:** make `slippage/` droppable into a colleague's backtest — facade, calibration, guide,
  packaging.
- **What I wrote:** `slippage/model.py` (`CostModel` facade + `RoundTripCost`: `roundtrip`,
  `roundtrip_optimal`, `roundtrip_band`, component toggles); `slippage/calibrate.py`
  (`calibrate(daily, intraday_15min) → Calibration(normal, stress)`, takes DataFrames — library
  stays pure); `slippage/README.md` (integration guide + sizing recipe + caveats); `pyproject.toml`
  (numpy+pandas, `pip install -e .`); tests `test_model.py` (6) + `test_calibrate.py` (4).
- **Refactor:** build_06/07 now import `CostModel` (single-sourced the round-trip formula);
  re-ran — **output byte-identical**, findings unchanged. **42 tests passing.**
- **Decisions (from user):** cost-aware sizing ships as a **README recipe**, not core API (library =
  cost model only); refactored the committed build scripts; `pyproject` + README packaging.

### S07 — Cost-aware sizing with λ (was "Block 7") ✅ DONE
*Status: complete. See `research/07_cost_aware_sizing/findings_07_cost_aware_sizing.md`.*

- **Goal:** choose per-trade deployed fraction `f*` maximising mean–variance utility on
  return-on-AUM; integrate with the signal's `1 − p_severe`.
- **What I wrote:** `research/07_cost_aware_sizing/build_07_cost_aware_sizing.py` (reuses Stage-6
  loaders), `tests/test_cost_aware_sizing.py` (4 passing).
- **Key result:** uniform `f` leaves Sharpe unchanged → cost-aware sizing protects the
  per-deployed-$ edge by *not over-trading*; the cost-optimal **trade size saturates** (~$8M TQQQ /
  ~$4M SQQQ) and `f*` falls as 1/AUM. Net Sharpe holds flat (~1.5/~0.9) vs Stage 6's all-in
  collapse; **return-on-AUM decay is the true capacity cost.** Higher λ → smaller f*, higher Sharpe,
  lower return-on-AUM. `Q*` stays ≪ $50M so W6 stops binding.
- **Decision:** compose with signal as `final = min(1−p_severe, f*)` — two independent shrink
  factors, the binding one wins. λ here is the *sizing* mean-variance knob (O(1)–O(20)), distinct
  scale from the Stage-6 *execution* λ; present as a grid {0,5,20}, not a pinned value.

### S06 — Net-Sharpe-vs-AUM capacity curve (was "Block 6") ✅ DONE  (the headline)
*Status: complete. See `research/06_capacity_curve/findings_06_capacity_curve.md`.*

- **Goal:** the headline curve — re-charge the strategy's *gross* returns with the size-aware cost
  across AUM $100k→$1B.
- **What I wrote:** `research/06_capacity_curve/build_06_capacity_curve.py`,
  `tests/test_capacity.py` (5 passing). Gross returns reconstructed from
  `decision_price→exit_decision_price` of the TQQQ/SQQQ canonical logs (verified the baked 20.2 bps).
- **Inputs:** gross Sharpe **TQQQ 4.92 / SQQQ 2.81**; turnover **169 / 140 rt/yr**; per-trade
  notional ≈ 0.95·AUM (all-in).
- **Key result (no-λ, 15-min-pinned, central Y):** TQQQ keeps ~80% gross Sharpe to ~$1M, half to
  **~$4–5M**, edge gone by ~$25M; SQQQ ~half by ~$2.5M, gone by ~$10M. Wide Y-band + stress ~halves.
  **With-λ view:** for a ~1-day-hold strategy *patient execution wins* (Sharpe highest at λ=0) —
  paying impact to suppress timing variance is self-defeating when timing ≪ daily P&L variance.
- **Decision:** View 1 (15-min-pinned, expected-cost, λ-free) is the **operative** capacity curve;
  View 2 (λ-chosen execution speed) is the mechanism illustration (it unrealistically lets fills
  stretch past the 15-min cadence). Present λ as a grid {0,1,3} + Y-band, never a single line.

### S05 — Size-aware cost function (was "Block 5") ✅ DONE
*Status: complete. See `research/05_cost_function/findings_05_cost_function.md`.*

- **Goal:** compose Blocks 1+3+4 into `cost(Q, urgency)` with `t=(Q/ADV)/participation·day`.
- **What I wrote:** `slippage/cost.py` (MarketParams + slippage_cost + optimal_participation +
  capacity_at_horizon, pure), `tests/test_cost.py` (Block-4 consistency + trade-off, 6 passing),
  `build_05_cost_function.py`.
- **Key result / correction:** Block 4's capacity assumed day-long execution; the strategy's 15-min
  cadence forces fast trading → **real per-trade capacity ~26× lower** (TQQQ ~$3.9M at 15-min/25bps
  vs $101M over a day). Same $17M order: 10 bps impact over a day (340 bps timing) vs 51 bps impact in
  15 min. Almgren–Chriss trade-off optimum ≈ 9% POV for $10M TQQQ (impact≈timing≈51 bps).
- **Decision:** impact = expected drag (return), timing = risk (variance) — different channels, don't
  sum. Capacity-by-horizon (expected cost) is λ-free; optimal-execution bps are λ-dependent (Stage 7).

### S04 — Market impact (√-law) + capacity (was "Block 4") ✅ DONE
*Status: complete. See `research/04_impact_capacity/findings_04_impact_capacity.md`.*

- **Goal:** model impact `I=Y·σ·(Q/V)^β` and the capacity it implies; the headline curve.
- **What I wrote:** `slippage/impact.py` (law + capacity inversion, pure), `tests/test_impact.py`
  (math self-consistency, reproduces plan example, 6 passing), `build_04_impact_capacity.py`.
- **Key result:** capacity at 10 bps impact (central Y=0.5, band [0.3,1.0]): **TQQQ ~$17M normal /
  $4M stress**, SQQQ ~$10M / $2M, QQQ ~$800M / $170M. TQQQ $17M reproduces the plan's $18M example
  from measured σ≈339 bps & $ADV≈$4.9B.
- **Decision / caveat:** Y is **adopted from literature, NOT fitted** (not identifiable from OHLC) —
  capacity is a band, not a line. Single-name √-law valid to ~$50M for 3× ETFs (above that the
  underlying QQQ/futures binds — W6). Combined picture: small size → delay-risk-dominated, large size
  → impact-dominated, spread a ~1 bp floor throughout.

### S03 — Delay / timing cost (was "Block 3") ✅ DONE
*Status: complete. See `research/03_delay_cost/findings_03_delay_cost.md`.*

- **Goal:** measure the price drift between a 15-min decision and a fill k minutes later.
- **What I wrote:** `slippage/delay.py` (forward-return / timing-risk, pure), `tests/test_delay.py`
  (√-time recovery, 4 passing), `research/03_delay_cost/build_03_delay_cost.py`.
- **Result:** timing risk **dwarfs spread ~17×** — TQQQ/SQQQ 1σ = 17 bps at 1-min delay, 36 at 5-min,
  59 at 15-min (QQQ ~⅓). Mean≈0 (a **risk**, not a drag — inflates variance/tails). √t holds, slightly
  sub-linear (mild mean-reversion). Strong clean regime split (stress 15-min ≈ 83 bps).
- **Decision:** delay term for `cost(Q)` ≈ `σ_1min·√k` (σ_1min ≈ 17 bps TQQQ/SQQQ, 5.5 QQQ), entered
  as variance not mean until momentum/mean-reversion sign is resolved (SLIPPAGE_PLAN Weakness 3).

### S01 — Bid-ask spread estimate (Corwin–Schultz) (was "Block 1") ✅ DONE
*Status: complete. See `research/01_spread_estimate/findings_01_spread_estimate.md`.*

- **Goal:** measure the bid-ask spread floor for TQQQ/SQQQ/QQQ from OHLC, replacing the flat 5 bps.
- **What I wrote:** `slippage/spread.py` (CS estimator, pure/vectorized), `tests/test_spread.py`
  (synthetic recovery, 7 passing), `research/01_spread_estimate/build_01_spread_estimate.py`.
- **Sanity-check:** synthetic test recovers known 5/10/20 bps spreads in the mean within 20%;
  real-data magnitudes realistic + expected intraday shape + 2020/2022 stress spikes.
- **Result:** half-spread ~0.7–1.0 bps mean, ~2.5 bps at open, ~3.5 bps in stress — **all under the
  flat 5 bps**. Spread is a small near-constant floor; capacity will be driven by impact, not spread.

---

## Glossary (in your own words)
*Define each term the first time you meet it — spread, half-spread, impact, slippage,
implementation shortfall, participation rate, ADV. Forces articulation.*

-
