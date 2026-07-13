# HANDOFF — volume_research (slippage / market-impact)

**Last updated:** 2026-07-09

## State

**Latest session (2026-07-09, continued): briefing refresh + code audit.**

- **`docs/briefing_2026-07.md` refreshed to post-fix state.** E04 numbers updated (TQQQ $20M
  fill 60%→**42%**, cost crossover now $100k–$1M only; scheduler loses the measurable-cost edge
  from ~$5M up), added an **Audit pass (2026-07-09)** section (fixes #1–#3 + open finding #4) and
  two design caveats: the scheduler is a **pre-trade planner with no live re-steer** (sizes all
  slices once from *forecast* volume, can breach POV cap if realized volume comes in light), and
  **slice bin-width is hand-pinned at 15 min** (a 5/10-min sweep is a candidate, matters only once
  POV binds).
- **Code audit (`/debug`) — 171 tests pass, no headline bug, but 5 latent issues found** (see
  Next steps "Latent code-audit findings"). Two are on the **production path** and should be fixed
  before wiring `schedule_order` into the live stack with a real `VolumeProfile`. None affect
  current research numbers (all runs use `profile=None`, `order_style="cross"`, modest horizons).
- Conceptual Q&A only (no code changes): the five objective terms, interruption definition, the
  E03-vs-E04 reconciliation (E03 charges impact + assumes full fill; E04 charges drift+spread
  only + real interruptions), 14-vs-22 bps, and limit-vs-cross differentiation (aggressiveness vs
  NBBO at submission; verified after the fact by the maker/taker flag — the +14 bps mechanism).

**Prior session (2026-07-09): independent audit of the 12-stage build, plots everywhere, client
briefing.**

- **The audit verdict: the build is sound.** All 171 tests pass; every hand-checked number
  reproduced exactly (E03's $5M worked cell re-derived from first principles; the Almgren golden
  test pins all six Table-3 points; the TCA monitor independently reproduces S08's +14.2 bps and
  its +5.3 bps residual matches P02's 5.5 bps adverse-selection gap). Three substantive findings
  were identified; **fixes #1 and #2 landed in a later 2026-07-09 session** (stretched-exp g(h) +
  objective double-count), #3 (E04 caveat) still open, plus a **new finding #4** the owner
  surfaced (alpha charged at terminal g(h*) not fill-averaged g). All bias the same (conservative,
  anti-scheduler) direction, so E04's "$20M reversal" should still be read as *unresolved*, not
  "scheduler loses". See Next steps 1-3b.
- **Every research folder now has a plot.** Added `make_plot()` (house dark style, reads only
  the stage's `results/` CSVs) to S09/S10/S11/S12/P03/E03/E04/C01/C02 builds +
  `scripts/tca_monitor.py` (refreshed each run). S11/S12-0b/E04 plots regenerate from CSV
  without network/re-replay. The plotting pass caught a real error: P04's findings table had
  transposed group counts (said 10 buy-limit / 18 sell-market; data says **15 / 13**) — fixed
  in `findings_04_tca_monitor.md` (means were always correct).
- **`docs/briefing_2026-07.md`** — meeting prep for the client (2026-07-09): the three questions
  (predictor / scheduler / capacity), implementation steps, owner decisions, what not to
  overclaim, QQQ ~50× depth explained (5× volume × ~9× from 1/σ²).

**`plans/2026-07_execution_track.md`'s full 12-stage build is DONE** (all three phases,
committed, 171 tests passing). Canonical numbering: `research/` folder number = stage ID
(S01…S12 legacy + P01-P04 predictor, E01-E04 scheduler, C01-C02 capacity); `ROADMAP.md` is the
master map with links to every findings doc — read it first.

**Phase 0 (conventions):** track folders created; `slippage_cost` renamed to
`expected_slippage_bps` throughout (pure rename, behavior-neutral).

**Phase 1 — Predictor track (P01-P04), answers "predict the fill diff":**
- **P01** `slippage/state.py::estimate_state` → `MarketState` (volume profile, vol nowcast,
  regime, spread curve — reproduces S01's spread numbers exactly).
- **P02** `research/predictor/02_chase_simulation`: full-history cross-vs-chase simulation.
  Found and fixed a real bug: the TRADES CSVs' `decision_price` isn't on the same
  split-adjustment basis as the DB candles (~2x/~5x off) — fixed by using the DB's own price.
- **P03** `slippage/predict.py::predict_slippage` — the actual predictor: mean + non-Gaussian
  p50/p90/p95 band, itemized components.
- **P04** `scripts/tca_monitor.py` — idempotent recurring monitor; its baseline run
  independently reproduces S08's +14.2 bps exactly, and its residual (+5.3 bps) matches P02's
  separately-measured adverse-selection gap.

**Phase 2 — Scheduler track (E01-E04), the child-order scheduler:**
- **E01** `slippage/alpha_decay.py::alpha_forfeit_frac` — g(h), fixed S12-0a's hold-length
  mixing bug (conditioned on hold≥1 day).
- **E02** `slippage/interruption.py` — hazard curve + cancel/complete_now cost model; regime
  barely moves the hazard (unlike every cost-magnitude measurement elsewhere).
- **E03** `slippage/schedule.py::schedule_order` — the centerpiece: picks the horizon h*
  trading off execution cost vs. alpha-forfeiture vs. interruption risk, then slices
  VWAP-shaped/POV-capped. Worked examples: +0.9 to +48.7 bps vs. a naive 15-min fill.
  **Caught a comparison bug while drafting findings** (partial-terms comparison, not
  apples-to-apples) — fixed.
- **E04** `research/scheduler/04_replay` — replays the scheduler against **real** historical
  entries and **real** interruption events (not the modeled average hazard). **Qualifies E03**:
  wins clearly at $100k-$1M, but at $20M TQQQ the scheduled cost is *worse* than both baselines
  (only 60% filled) — real interruptions bite harder than the model's average at large size.

**Phase 3 — Capacity track (C01-C02):**
- **C01** `slippage/impact.py::almgren_temporary`/`almgren_permanent` — Almgren et al. (2005),
  golden-tested exact against the paper's Table 3. Threaded `impact_model="sqrt"|"almgren"`
  through `cost.py`→`model.py`→`predict.py` (+ `"envelope"` in predict.py). At TQQQ/SQQQ's own
  σ/ADV, Almgren predicts **~7.4x more capacity** than sqrt's optimistic edge — both models are
  extrapolating outside their fitted domains here. **Library default stays "sqrt" — not
  flipped**, an explicit owner/GATE decision.
- **C02** `research/capacity/02_capacity_refresh` — re-ran S06 with the envelope band +
  scheduler horizon + full alpha/interruption cost (not execution alone — **caught and fixed
  the same class of bug as E03/E04 warn about**, one more time, before reporting). Capacity
  extends **moderately**: TQQQ Sharpe-zero AUM moves from ~$14M (old, 15-min-pinned) to
  ~$30-100M depending on envelope edge. `FINDINGS.md`'s headline section now shows both tables
  (new + old, kept for continuity per instruction).

Working tree is clean except an unrelated `all-weather/live/daily_snapshot.py` edit (not part of
this project — left untouched). Nothing pushed. (Full chronological history, including the
pre-execution-track sessions this summarizes, lives in `BUILD_LOG.md` — not duplicated here.)

## Next steps (priority order)

**Fixes from the 2026-07-09 audit (technical, small, do before relying on the scheduler):**

1. **✅ DONE (2026-07-09, later session).** Refit E01's g(h) as a stretched exponential
   `g(h) = 1 - exp(-(h/tau)^k)`. New constants in `slippage/alpha_decay.py`: TQQQ (τ=256.9,
   k=2.480), SQQQ (τ=251.7, k=2.723). No longer overcharges the 15-120 min band; residual is a
   small *under*-charge below ~60 min (biases h* slightly long, opposite direction). E03/C02
   rerun and all findings/ROADMAP/FINDINGS/briefing updated. 171 tests pass.
2. **✅ DONE (2026-07-09, later session).** Fixed the `_objective` double-count. Alpha term is now
   `edge*(1-hazard)*g(h)`; factored a shared `schedule.alpha_interruption_bps()` helper and routed
   `_objective`, `schedule_order` reporting, **and** E03's naive baseline + C02's addon through it
   (all three had re-derived the same double-count). E03 improvement grew to +3.7→+58.5 bps; C02
   central Sharpe-zero AUM ~doubled to ~$65M (TQQQ) / ~$37M (SQQQ).
3. **✅ DONE (2026-07-09, later session).** Added the "impact NOT charged" caveat to E04's
   findings, and re-ran E04 against the post-fix scheduler. Post-fix horizons are longer, so fill
   rates dropped (TQQQ $20M 60%→42%) and the cost crossover moved earlier (scheduled beats naive
   only at $100k–$1M now). The decision-grade $20M finding remains the fill-rate collapse, now
   explicitly framed against the no-impact caveat. E04 numbers/plot/ROADMAP row refreshed.
3b. **Finding #4 — the alpha term charges terminal g(h*), not the fill-averaged g (STILL OPEN,
   logged 2026-07-09).** `schedule.py` slices the order over `[0, h*]` but charges alpha as
   `g(h*)` — i.e. as if the whole order filled at the terminal instant. A real sliced fill is
   partly in the market throughout, so its true forfeiture is the fill-time-weighted average of
   `g` over `[0, h*]`, which is materially less than `g(h*)` because g is rising/convex there
   (even-fill estimate: ~2.1% vs g(90)=7.2% at h*=90 → ~3x overcharge). Same direction/family as
   fix #1: overcharges alpha, biases h* short. Note the internal inconsistency it exposes — the
   *execution* term already rewards spreading (lower participation over a longer window), while
   the *alpha* term pays the full terminal delay cost. Fix = replace `alpha_forfeit_frac(h*)` with
   an integral of g over the actual slice schedule; it interacts with the interruption term's
   `phi` fill-fraction assumption, and would shift E03/C02 h* longer / more scheduler benefit.
   Judgment call (what benchmark the alpha term is measured against), not a pure bug — surfaced by
   the owner. NB: the underlying TRADES log has instant fills only (one `entry_time`,
   `avg_order_price`=decision_price+flat 5bps), so there is no execution-duration data to fit
   against — this is a modeling choice, not fittable from our data.

**Latent code-audit findings (2026-07-09 continued) — none affect current research numbers,
but fix the two production-path ones before live wiring:**

1a. **`_bin_volume` walks wall-clock minutes past the session close (`schedule.py:120`) — PRODUCTION
   PATH.** `bin_ts = state.ts + timedelta(15*i min)` has no session awareness; once a schedule
   spans past 16:00, `session_bin_label` yields labels not in `bin_share_mean`, so every
   after-close bin silently falls back to the mean share (VWAP shaping → flat) and it never rolls
   to next day's 09:30. Bites only when a real `VolumeProfile` is supplied *and* h* crosses the
   close. Fix: step over trading bins, not raw minutes.
1b. **`schedule_order` prices the plan as `"cross"` even when `order_style="limit_chase"`
   (`schedule.py:190`) — PRODUCTION PATH.** Slices are tagged limit_chase but `expected_slippage_bps`
   omits the P02 chase-drag (cross ⇒ drag 0). Latent because order_style defaults to cross. Fix:
   thread `order_style` into the `predict_slippage` calls (cost + `_objective`).
1c. **Hazard-grid 1d/2d knots on an inconsistent time axis (`interruption.py:32`).** 6.5h/13h map
   cleanly to 390/780 min, but "1d"/"2d" (calendar per findings_02) sit at 1560/3120 min — neither
   calendar (1440/2880) nor trading minutes. `np.interp` in the 780–3120 region is on a mislabeled
   x-axis. Low severity (h* rarely lands there). Fix: decide trading- vs calendar-minutes and place
   the knots consistently.
1d. **`capacity_at_horizon` never clamps participation ≤ 1 (`cost.py:113`).** A generous budget can
   yield `part > 1`, reporting a capacity that requires >100% of window volume as feasible. Fix:
   `min(part, 1.0)` or a feasibility flag.
1e. **Doc mismatch (not code):** `build_03_scheduler.py:37` uses `EDGE_BPS=50` ("illustrative")
   vs briefing line ~137 citing "~107 bps." Pick one placeholder story.

All four owner decisions flagged throughout the build, still open:

4. **GATE: flip the library's default `impact_model`?** Currently "sqrt". C01 found Almgren
   implies ~7.4x more capacity at TQQQ/SQQQ's actual parameters — both models are extrapolating
   outside their fitted domains, so this is a genuine judgment call, not a data question.
5. **`edge_bps` calibration** — E03/E04's worked examples used an illustrative 50 bps; C02 used
   each ticker's own measured average trade return (107.1/103.9 bps) for the capacity refresh.
   Decide which is the right number for live scheduling decisions (likely the measured one,
   re-measured periodically).
6. **`pov_cap` default** (currently 10%) and any `schedule_order` urgency/λ-equivalent defaults —
   not tuned against real fill data yet.
7. **When/how to run `scripts/tca_monitor.py` recurrently** (cron, launchd, CI) — deliberately
   left undecided; the script only implements the one-shot check.

Secondary (not blocking): **bound `Y`/η from live fills** as they accumulate past retail size —
still not possible at current trade sizes.

## Open questions

- **λ (risk aversion)** — *resolved by design:* not pinned to one value; presented as a **grid**
  (execution λ {0,1,3} in Stage 6, sizing λ {0,5,20} in Stage 7). It's a preference input from the
  strategy owner, not data-fittable; the curve *shapes* are robust to the exact value.
- **Strategy type: momentum vs mean-reversion (W3)** — *FULLY RESOLVED & IMPLEMENTED:* momentum;
  signed +14 bps adverse entry drag now in the model (`CostModel.entry_drag_bps`) and the fix (cross
  entries) in the planner (`plan_execution` → `cross_entry`). Nothing left to do here except
  re-measuring the ~14 bps as more live fills accumulate.
- **Permanent impact / repeated-trade accumulation (W5)** not modelled — current impact is single-order
  temporary; aggregate session capacity is lower.
- **Y band is irreducible from OHLC** — bound later with live fills (realized `avg_order_price −
  decision_price` vs model) as they accumulate.
- **`literature/` (63 MB PDFs)** — decide: keep out of git (current), gitignore, or store externally.

## Decisions (audit + plots session, 2026-07-09)

- **Every research stage's `build_*.py` regenerates a `results/*.png`** via a `make_plot()` that
  reads only the stage's own results CSVs — so plots can be regenerated without re-running heavy
  or network-dependent stages (S11/S12-0b fetch from Alpaca; E04 re-replays for ~2.5 min). New
  stages should follow the same pattern (house dark style, copied per script — research folders
  aren't packages).
- **The three audit findings (Next steps 1-3) are acknowledged, not yet fixed** — do not
  re-litigate whether they're real; the numbers are in this handoff. E04's $20M cost comparison
  is not decision-grade until #3's caveat lands (fill-rate collapse is the valid finding).

## Decisions (Phase 1-3 build, 2026-07-08)

- **`schedule_order`'s interruption default is `mode="cancel"`** — the more realistic default for
  an entry schedule (a firing exit stop means the position is being closed, so an unfilled entry
  residue should be cancelled, not chased further).
- **`predict_slippage`'s `"envelope"` band never changes the point/mean estimate** — it only
  widens the reported low/high (sqrt Y=0.5 stays the mean even under `impact_model="envelope"`).
  Same convention reused in C02's capacity refresh.
- **Every "total cost" number composing the scheduler must include alpha-forfeiture +
  interruption, not execution cost alone** — this exact mistake was made and caught twice (once
  drafting E03's findings, once in C02's first pass) before being fixed. If extending this work,
  assume the same trap exists anywhere a horizon is chosen and re-costed.
- **`edge_bps` is always caller/context-supplied, never fitted** — E03/E04 used an illustrative
  50 bps for worked examples; C02 used each ticker's own measured average trade return for the
  capacity headline. Match the source to the consumer's precision needs.
- Spread = 15-min Corwin–Schultz, within-session pairs, **clamp the aggregate not per pair**.
- Delay = 1-min forward return, within-session; a **risk** (std), not an expected cost.
- Impact = √-law with **Y adopted from literature (band), never fitted from our data**; report capacity
  as a band; single-name ceiling ~$50M for 3× ETFs.
- Cost model: **impact = mean drag, timing = variance** — different channels, don't sum. Capacity-by-
  horizon (expected cost) is λ-free; optimal-execution bps are λ-dependent.
- **Normal σ is reported as a pair (mean + median).** The rolling-vol series is right-skewed, so one
  "normal" scalar is a summary choice. **Headline = mean** (`calibrate().sigma_daily_bps`, used by
  `normal`) — the expected-cost / time-average moment (average cost over many trades scales with
  E[σ]; cost is linear in σ). **Median** (`sigma_daily_median_bps`) is the typical-day sensitivity.
  Stress stays the separate p90. The ~9% σ gap → ~16% capacity, well inside the 12× Y-band, so the
  verdict is unchanged. build_04/05/06/07 lead with the mean; findings show both (findings_09).
- Env = **quant** conda env; `slippage/` package = the library. Working mode = **fast-mode**.
- **Signed entry drag lives on `CostModel`, not `MarketParams`** — it's strategy/execution-style
  dependent (momentum + limit-chase), not a property of the instrument. Charged **once per round trip**
  (entry only; clean market exits ≈0), a **mean** cost on top of the symmetric timing **variance**
  (different moments of the same decision→fill move — not a double-count). Not fittable from OHLC, so
  `calibrate` never sets it; caller/live-fill input. Default 0 = symmetric assumption.
- **Cross-vs-rest is a differential decision** — impact is ~common to a marketable cross and a passive
  limit chase, so it cancels; the choice is spread-you-cross vs drift-you-chase. `plan_execution`
  therefore compares `entry_drag_bps` to **`half_spread`**, deliberately NOT to `expected_cost_bps`
  (its √-law impact over-extrapolates at retail, which the live fills contradict).
- **The capacity curve assumes efficient execution** — the flat, size-independent entry drag is a
  *current-P&L* item (fix by crossing), **not** baked into the net-Sharpe-vs-AUM headline; it would only
  dent retail Sharpe, not the impact-bound ceiling.
