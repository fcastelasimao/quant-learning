# Slippage project — briefing for the client meeting (2026-07-09)

One page of talking points, then the detail behind each. Every number cites the stage that
produced it (`research/<folder>/findings_*.md` has the full method and caveats).

---

## The 30-second version

3 goals: **predict the fill-price surprise** ("I set an order at X and filled at
X + diff"), **tell us the best way to execute large volumes depending on the state of the
market**, and **how big can we get**. All three are now answered with working, tested code
(171 tests) calibrated on 13 years of TQQQ/SQQQ data plus our own live fills:

1. **The "X + diff" mystery is solved and fixable today.** The diff wasn't bad luck — our live
   engine rests limit orders while the momentum signal pushes the price away, so we paid
   **+14 bps per buy entry**. Simply crossing the spread instead costs ~3–4.5 bps. That's
   **~10 bps saved per entry, implementable immediately** — the single most valuable line in
   this project.
2. **There is now a pre-trade predictor and an execution scheduler.** `predict_slippage()`
   returns the expected cost *and* the tail (p90/p95) for any order given the current market
   state; `schedule_order()` turns a large order into timed child orders. On a full historical
   replay, the scheduler beats a naive single fill at sizes up to ~$1M; from ~$5M up the
   measurable-cost advantage is gone and real-exit fill-rate collapse dominates — honest caveats
   apply (below).
3. **Capacity is real but finite.** For the 3× ETFs the edge is largely gone somewhere between
   **~$10M and ~$100M AUM** depending on which impact model you trust — that band is the honest
   uncertainty, not indecision. QQQ itself is ~50× deeper (5× the daily volume, a third of the
   volatility: a 15-min order at the same 25 bps budget can be ~$150M in QQQ vs ~$3M in TQQQ),
   so the unlevered instrument is the escape hatch if the strategy outgrows the 3× ETFs.

---

## Question 1 — "Predict the diff" (the predictor)

### What we found

- **The diff does not come from market noise.** Our 30 live fills
  ([research/08_live_validation](../research/08_live_validation/findings_08_live_validation.md)):
  buys entered with resting limit orders slipped **+14.2 bps** on average; sells sent as market
  orders slipped **−0.3 bps** (i.e. nothing). Same market, same engine — the only difference is
  order type. Mechanism: a momentum signal fires *because* the price is rising, so a passive
  limit at the decision price fills mostly when the trade is about to go badly and chases when
  it's going well (adverse selection).
- **Confirmed on 4,000+ simulated entries, not just 15 live ones.**
  [research/predictor/02_chase_simulation](../research/predictor/02_chase_simulation/findings_02_chase_simulation.md)
  replayed both entry styles at every historical entry signal: crossing costs **2.9 bps (TQQQ) /
  4.5 bps (SQQQ)**; resting a limit with *any* timeout from 1 to 60 minutes costs **6.7–13.2 bps**.
  Chasing never wins, and it gets worse in stressed markets (13–20 bps).
- **The predictable part of slippage is small; the *risk* is not.** The expected cost of a
  typical order is a few bps, but the p95 outcome at 15-minute latency is **~60–180 bps**
  depending on volatility regime — driven by timing risk (mean zero, so it's invisible in
  averages but it is exactly the "substantial diff" experience). The predictor reports both.

### The deliverable

- **`slippage/predict.py::predict_slippage(notional, side, order_type, state, price)`** →
  expected slippage + p50/p90/p95, itemized into spread / impact / chase-drag / timing.
  Worked grid in [research/predictor/03_predictor](../research/predictor/03_predictor/findings_03_predictor.md).
- **`slippage/state.py::estimate_state()`** — the "state of the market" input: intraday volume
  profile (the open bin carries 4× the midday volume), volatility nowcast, live NBBO spread.
  ([research/predictor/01_market_state](../research/predictor/01_market_state/findings_01_market_state.md))
- **`scripts/tca_monitor.py`** — a rerunnable monitor that compares every new live fill against
  the prediction and alerts if the model drifts. Its baseline run over our 572 logs reproduces
  the +14.2 bps independently.
  ([research/predictor/04_tca_monitor](../research/predictor/04_tca_monitor/findings_04_tca_monitor.md))

---

## Question 2 — "Best way to execute large volumes, depending on the market" (the scheduler)

### What we found

- **The trade-off is real and measurable.** Executing faster costs impact (you push the price);
  executing slower forfeits alpha (the signal's edge decays with a ~4.8-hour half-life —
  [research/scheduler/01_alpha_decay](../research/scheduler/01_alpha_decay/findings_01_alpha_decay.md))
  and risks interruption (a quarter of our positions exit within ~2.7 hours —
  [research/scheduler/02_interruption_risk](../research/scheduler/02_interruption_risk/findings_02_interruption_risk.md)).
- **`slippage/schedule.py::schedule_order()`** balances all three and outputs a concrete plan:
  the horizon, the child orders (sized to predicted volume, capped at 10% participation per
  bin), and the expected cost with a band.
  ([research/scheduler/03_scheduler](../research/scheduler/03_scheduler/findings_03_scheduler.md))
- **Replayed against real history** (every entry 2013–2026, real interruptions,
  [research/scheduler/04_replay](../research/scheduler/04_replay/findings_04_replay.md)):
  at $100k–$1M the scheduler clearly beats a naive single fill (TQQQ 14.5–18.7 bps vs 21.4 bps)
  and a day-VWAP benchmark; **from $5M up the measurable-cost edge is gone (TQQQ ~23 bps ≥ naive
  21.4) and the schedule gets cut short by real exits — only ~42% fills at $20M.** Note the
  replay charges *drift + spread only, not impact*, so naive/VWAP are flat across size; E03 (which
  *does* charge impact) is the optimistic bracket and E04 the pessimistic one. We report both
  honestly rather than hiding it — very large orders remain the open frontier.
- Note there is no broker algo to lean on: Alpaca's retail API has no VWAP/TWAP service —
  **we are the execution algo**, which is why this scheduler exists.

---

## Question 3 — "How big can we get" (capacity)

### What we found

- **Cost grows with the square root of size** (industry-standard impact law), so capacity is a
  curve, not a cliff. Calibration: our σ and volume, coefficient adopted from the literature —
  it is *not measurable from our own price data* (we'd need to know what prices would have done
  without our trades), so it's carried as a band, never a point.
  ([research/04_impact_capacity](../research/04_impact_capacity/findings_04_impact_capacity.md),
  [05](../research/05_cost_function/findings_05_cost_function.md),
  [06](../research/06_capacity_curve/findings_06_capacity_curve.md))
- **Headline (TQQQ):** net Sharpe holds ~80% of gross to ~$1M AUM, half by ~$15–20M, and crosses
  zero at **~$65M central** (vs ~$14M under the original 15-min execution) with a wide envelope of
  **~$13M (pessimistic) to $300M+ (optimistic)** once the scheduler's smarter horizons are used
  and all costs charged. SQQQ is roughly half that (central ~$37M). Above ~$50M the ETF itself
  stops being the binding constraint (the underlying QQQ book is), so numbers past that are
  flagged invalid — and the central crossing sits right at that ceiling.
  ([research/capacity/02_capacity_refresh](../research/capacity/02_capacity_refresh/findings_02_capacity_refresh.md))
- **The band is wide and that's the truthful answer.** An alternative published model (Almgren
  2005, ported and verified against the paper exactly —
  [research/capacity/01_almgren_envelope](../research/capacity/01_almgren_envelope/findings_01_almgren_envelope.md))
  predicts ~7× more capacity than our conservative default. Both models are extrapolating
  outside their fitted domain here. We report the **envelope** of the two and flag the choice
  of default as a management decision, not a technical one.
- **Sizing rule, not just a ceiling:** past ~$7M (TQQQ) the optimal behavior is "trade a fixed
  dollar size, deploy the fraction that implies" — capacity becomes a return-on-AUM decay, not
  a Sharpe collapse. ([research/07_cost_aware_sizing](../research/07_cost_aware_sizing/findings_07_cost_aware_sizing.md))

---

## How to implement this — concrete next steps

**Do now (config-level change, ~10 bps per entry):**
1. Switch momentum entry orders from resting limits to **crossing the spread** (marketable
   orders). Cost of crossing: ~1 bp. Cost of chasing (measured live *and* in simulation):
   8–14 bps. This is the project's cheapest win.

**Do this month (wire into the trading stack):**
2. Call **`predict_slippage()` before sending any sizable order** — it gives the expected cost
   and the p95 so position-level decisions can price execution in.
3. Run **`scripts/tca_monitor.py`** on a schedule (daily cron is enough — it's idempotent).
   It builds the live evidence base and alarms if reality drifts from the model. Decision
   needed: who owns the cadence.
4. For orders above ~$1M, generate the plan with **`schedule_order()`** (measured up to $5M;
   supply the strategy's real per-trade edge — currently ~107 bps — instead of the placeholder).

**Decisions that belong to you (flagged in [HANDOFF.md](../HANDOFF.md), deliberately not made by the research):**
5. Which impact model is the reporting default — conservative sqrt-law, the more optimistic
   Almgren, or the envelope band (recommended: envelope for reporting, sqrt for sizing).
6. The participation cap per slice (10% default) and the alert threshold / cadence of the TCA
   monitor.
7. Whether capacity planning uses the 15-min-pinned or scheduler-based numbers.

**What I'd caution against saying we know:**
- Exact capacity to one number — the honest range at 25 bps/round-trip is wide (~12× across
  models) because impact coefficients cannot be fitted from our own OHLC data.
- Scheduler performance at $20M+ — the replay shows real interruptions bite there; treat that
  size as unproven until we have live/paper evidence at scale.
- Anything about instruments other than TQQQ/SQQQ/QQQ — every constant is calibrated to these.
- **The scheduler is a *pre-trade planner*, not a live executor.** It sizes all child orders
  once, up front, from *forecast* volume and never re-plans. If realized volume comes in light,
  the static plan can breach the 10% POV cap in reality. A live re-scheduler (re-solve the
  remaining allocation each bin from realized volume) is genuine future work.

---

## Audit pass (2026-07-09) — what changed since the last cut

A correctness review of the scheduler math produced four items; three are fixed, one is open,
and all four pushed in the *conservative* (anti-scheduler) direction — so fixing them made the
scheduler look **better** in the modeled views, which is the honest direction of the correction:

1. **Alpha-decay curve refit (fixed).** E01's `g(h)` is now a stretched exponential
   `1−exp(−(h/τ)^k)` fitted jointly, replacing a single exponential that over-charged short delays.
2. **Double-count removed (fixed).** The interruption term was charging alpha twice; the objective
   now routes completion and interruption through one shared `alpha_interruption_bps()` helper
   (also fixed the two copies of the bug in the E03 baseline and C02). Post-fix the scheduler
   picks longer horizons and wins more in the modeled views (E03 +3.7 to +58.5 bps; C02 central
   Sharpe-zero ~$30M → ~$65M).
3. **E04 impact caveat documented (fixed).** The replay charges drift + spread only, *not* impact
   — now stated explicitly wherever E04 numbers appear, since it's why E04 and E03 look opposed
   at size (E03 charges impact and favors spreading; E04 doesn't and is punished by real exits).
4. **Fill-averaged `g` — OPEN.** The scheduler charges alpha at the *terminal* `g(h*)` but the
   order fills gradually over `[0, h*]`, so the true forfeiture is the *average* g over the fill,
   ~3× smaller. Still conservative (same safe direction), logged as an open modeling task — the
   scheduler's modeled edge is if anything *understated* today.

A knob worth a sensitivity sweep (not yet done): **slice bin width** is hard-pinned at 15 min
(matching the strategy's decision bar); 5/10-min bins would sharpen VWAP shaping but need a
finer volume profile to mean anything. Expected to matter only once POV binds (large size).

---

## Where everything lives

| Layer | Location |
|---|---|
| Narrative summary of all results | [FINDINGS.md](../FINDINGS.md) |
| Stage map (S01–S12, P01–P04, E01–E04, C01–C02) | [ROADMAP.md](../ROADMAP.md) |
| The library (pure, tested, importable) | `slippage/` — `predict.py`, `schedule.py`, `state.py`, `cost.py`, `model.py`, `plan.py` |
| Per-stage evidence (method, numbers, caveats, plots) | `research/<stage>/findings_*.md` + `results/*.png` |
| Live-fill monitor | `scripts/tca_monitor.py` |
| Open decisions & state | [HANDOFF.md](../HANDOFF.md) |

Every stage's `build_*.py` regenerates its numbers and plot from scratch; `pytest tests/`
(171 tests) verifies the library, including a golden test reproducing the Almgren paper's own
published examples.
