# ROADMAP — volume_research

The master map: what the project answers, what's been done, what hasn't, and the two paths
forward. Read this before BUILD_LOG.md (chronology) or any individual `findings_*.md` (detail).

## The question

This project replaces the backtest's flat, size-blind **20 bps** round-trip cost with a
**size-aware slippage model** for the TQQQ/SQQQ 15-min decision strategies. It answers three
things: (a) what a trade really costs as a function of **size, execution speed, and market
state**; (b) how big the strategy can trade before market impact eats the edge (the **capacity**
question); (c) how to **execute** a large order to minimize that cost. The client's framing of the
live ask is narrower and more operational: an **execution scheduler**, a **fill-price-deviation
predictor**, and a clear answer to **"from what size does impact become a problem."**

## What has been done

**Canonical numbering: a stage's ID is its `research/` folder number.** Work with no research
folder gets an "L" (library) ID. Old labels ("Block N" / "Stage N") survive only in parentheses —
see BUILD_LOG.md for the full history under the old names.

| ID | Findings | Old label | What it is | Status |
|---|---|---|---|---|
| S00 | [docs/history/SLIPPAGE_PLAN.md](docs/history/SLIPPAGE_PLAN.md) | "Stage 0 audit" | Audit: backtest charges flat 5/15 bps, size-blind | Done |
| S01 | [research/01_spread_estimate](research/01_spread_estimate/findings_01_spread_estimate.md) | Block 1 | Corwin–Schultz spread: half-spread ~0.7–1 bp, ≤3.5 stress | Done |
| S02 | — (folder never created) | plan "Stage 1" | Fill-timing assumption check; absorbed into S08 live validation | Skipped |
| S03 | [research/03_delay_cost](research/03_delay_cost/findings_03_delay_cost.md) | Block 3 | Timing risk: std ≈ 17.6·t^0.45 bps (TQQQ), mean ≈ 0 | Done |
| S04 | [research/04_impact_capacity](research/04_impact_capacity/findings_04_impact_capacity.md) | Block 4 | √-law impact, Y ∈ [0.3,1.0] adopted; capacity ~$14M TQQQ @10 bps day-exec | Done |
| S05 | [research/05_cost_function](research/05_cost_function/findings_05_cost_function.md) | Block 5 | cost(Q, participation); 15-min pin cuts capacity ~26× | Done |
| S06 | [research/06_capacity_curve](research/06_capacity_curve/findings_06_capacity_curve.md) | Block 6 | Net Sharpe vs AUM: TQQQ half-edge ~$4M, gone ~$14M | Done |
| S07 | [research/07_cost_aware_sizing](research/07_cost_aware_sizing/findings_07_cost_aware_sizing.md) | Block 7 | Optimal trade size saturates (~$6.8M TQQQ); f* ~ 1/AUM | Done |
| L01 | [slippage/README.md](slippage/README.md) | Block 8 | CostModel facade, calibrate(), pyproject, README | Done |
| S08 | [research/08_live_validation](research/08_live_validation/findings_08_live_validation.md) | Stage 9 | 30 live fills: buy +14 bps limit-chase drag (n=15, provisional); sells clean | Done |
| L02 | [slippage/plan.py](slippage/plan.py) | Stage 10 | plan_execution(): POV / horizon / slices / cross-vs-rest | Done |
| S09 | [research/09_calibrate_validation](research/09_calibrate_validation/findings_09_calibrate_validation.md) | Stage 11 | calibrate() reproduces findings from real DBs <1%; σ mean+median pair | Done |
| S10 | [research/10_edge_vs_cs](research/10_edge_vs_cs/findings_10_edge_vs_cs.md) | Stage 12 | EDGE estimator: faithful impl, rejected as default (unreliable at 15-min) | Done |
| S11 | [research/11_cs_validation](research/11_cs_validation/findings_11_cs_validation.md) | Stage 13 | CS-15min vs real SIP NBBO: TQQQ 0.74 vs 0.89 (~20–25% low, OK) | Done |
| S12 | [research/12_stage0_forks](research/12_stage0_forks/findings_12_stage0_forks.md) | plan "Stage 0" | Forks: alpha horizon = multi-day; ETF permanent impact ≈ tiny residual, Almgren perm term rejected | Done |

Next research folder number: **13**.

**Track IDs (new, 2026-07-08):** work under `plans/2026-07_execution_track.md` lives in per-track
folders numbered within the track — `research/predictor/NN_*` → **PNN**, `research/scheduler/NN_*`
→ **ENN**, `research/capacity/NN_*` → **CNN**. Legacy S01–S12 stay where they are (shared
foundations). Status legend: planned → in progress → Done.

| ID | Folder | What it is | Status |
|---|---|---|---|
| P01 | [research/predictor/01_market_state](research/predictor/01_market_state/findings_01_market_state.md) | MarketState: volume profile (open bin 4.2x midday), OOS R² 0.09–0.24, vol nowcast, spread curve matches S01 | Done |
| P02 | [research/predictor/02_chase_simulation](research/predictor/02_chase_simulation/findings_02_chase_simulation.md) | Cross (2.9-4.5bps) beats chase (6.7-13.2bps) at every T; S08 live sits in the sim distribution | Done |
| P03 | [research/predictor/03_predictor](research/predictor/03_predictor/findings_03_predictor.md) | `predict_slippage()`: mean+p50/p90/p95, composes P01+P02+S03; timing dominates at every size | Done |
| P04 | [research/predictor/04_tca_monitor](research/predictor/04_tca_monitor/findings_04_tca_monitor.md) | scripts/tca_monitor.py: reproduces S08's +14.2bps exactly; residual (5.3bps) matches P02's gap | Done |
| E01 | [research/scheduler/01_alpha_decay](research/scheduler/01_alpha_decay/findings_01_alpha_decay.md) | g(h)=1-exp(-(h/tau)^k) stretched-exp, tau~255min k~2.5 both tickers; flat<1h, knee 2-4h, saturated by 1d | Done |
| E02 | [research/scheduler/02_interruption_risk](research/scheduler/02_interruption_risk/findings_02_interruption_risk.md) | Hazard: p25 hold~2.7h, regime barely matters; cancel/complete_now cost model | Done |
| E03 | [research/scheduler/03_scheduler](research/scheduler/03_scheduler/findings_03_scheduler.md) | `schedule_order()`: +3.7bps($1M/normal) to +58.5bps($20M/stress) vs naive 15-min fill | Done |
| E04 | [research/scheduler/04_replay](research/scheduler/04_replay/findings_04_replay.md) | Real replay (no impact charged): wins $100k-$1M, loses by $5M; $20M TQQQ only 42% filled — fill-rate collapse is the finding | Done |
| C01 | [research/capacity/01_almgren_envelope](research/capacity/01_almgren_envelope/findings_01_almgren_envelope.md) | Golden test exact; Almgren gives ~7.4x sqrt(Y=0.3)'s capacity — both extrapolating; default unchanged | Done |
| C02 | [research/capacity/02_capacity_refresh](research/capacity/02_capacity_refresh/findings_02_capacity_refresh.md) | TQQQ Sharpe-zero AUM: ~$14M -> ~$65M central (envelope $13M-$300M+), scheduler horizon + full cost | Done |

## The storyline, one paragraph per phase

**Measure (S01, S03).** First measured the two components that don't need a model: spread and
timing risk. Spread (Corwin–Schultz on 15-min bars) is tiny — ~0.7–1 bp half, up to ~3.5 bps in
stress — and stays under the flat cost the backtest already charges. Timing risk (forward-return
std from decision to fill) scales as ~17.6·t^0.45 bps for TQQQ with mean ≈ 0: not a drag, but a
real variance driver at the 3× ETFs' vol.

**Model (S04–S07).** With spread and timing pinned, the impact term was the missing piece. Adopted
the √-law (`I = Y·σ·√(Q/V)`, Y ∈ [0.3, 1.0] from the literature, not fit) to get a first capacity
number, composed it with timing into a size-aware `cost(Q, participation)`, found the 15-min fill
pin cuts real capacity ~26× versus a full trading day, and turned that into the headline
net-Sharpe-vs-AUM capacity curve (TQQQ keeps half its edge to ~$4M, gone by ~$14M) plus a
cost-aware sizing rule that saturates trade size rather than sizing linearly with AUM.

**Validate & package (L01–S12).** Packaged the pieces into a small `CostModel` + `calibrate()`
library any backtest can drop in, validated it against 30 real live fills (revealing a momentum
entry-drag the model didn't have, since fixed), checked `calibrate()` reproduces the research
findings from real data to <1%, evaluated and rejected the EDGE spread estimator as unreliable at
15-min resolution, validated the Corwin–Schultz spread against real SIP NBBO quotes, and — most
recently — resolved the two forks that gate the next (paused) phase of capacity refinement:
the strategy's edge is multi-day, and Almgren's single-stock permanent-impact term doesn't fit an
arbitrage-pinned ETF.

## What has NOT been done

- **Volume-state modeling** — intraday volume-profile shape, interval-volume predictability. This
  is the client's "depending on market volume" — currently `calibrate()` uses static ADV, not a
  state-conditional volume forecast.
- **A packaged `predict_slippage(order, state)`** with order-type distinction (market vs limit vs
  algo) — the library composes cost components but doesn't expose a single state-conditional
  predictor function.
- **The execution scheduler** — a child-order schedule that consumes a volume profile + the S12
  alpha-decay curve to actually slice an order over time. `plan.py::plan_execution` recommends a
  POV/horizon; it doesn't emit a schedule.
- **A recurring live-TCA monitor** — S08 was a one-off parse of 30 fills, not a running comparison
  of realized vs predicted slippage.
- **Almgren coefficient adoption** — planned (`plans/2026-07_almgren_adoption.md`), paused.
- **The creation/redemption door** (ETF primary-market execution) — planned in the same paused
  plan, Phase 2.
- **Y calibration from our own fills** — impossible until trade size grows enough to carry an
  impact signal (retail size doesn't).

## Paths forward

**Path A — Execution track (ACTIVE, client-aligned).** A1 volume-state modeling → A2 slippage
predictor packaging → A3 scheduler → A4 recurring TCA monitor. Specs to be written; see
HANDOFF.md for next-session detail. See `plans/2026-07_execution_track.md` for the current bullet
list (detailed spec: TBD with owner).

**Path B — Capacity refinement (PAUSED).** Almgren coefficient adoption per
`plans/2026-07_almgren_adoption.md`, already amended by S12's findings (permanent term dropped for
the ETF case, the accumulation stage (A.5) shrunk to a bounded check, Phase 2 — the creation door —
kept alive since the alpha horizon came back multi-day). **Standing review caveat:** any future GATE
decision to flip the library's default impact model must present the **two-model envelope**
(√-law Y-band vs Almgren temporary-only) as the honest uncertainty, since the switch alone moves
capacity ~4–6×.

## Backlog (small known fixes)

- Recompute S12's 0a delay-fraction table conditioned on hold ≥ 1 day — the current table mixes
  hold lengths, so its 4h/1d rows are misleading.
- Add a "capacity at 50% of backtest edge" sensitivity row to S06 — gross Sharpe 4.9 may be
  in-sample-inflated.
- Re-measure the +14 bps entry drag as live fills accumulate (currently n=15, SE ≈ 9 bps).
- Consider renaming the public cost API for clarity (code change — needs owner).
- Relax the hard 15-min fill pin in S05/S06 using the S12 alpha-decay curve (research change —
  owner decision).
- Switch live-facing spread to direct NBBO measurement (SIP access confirmed in S11), keep CS for
  pre-quote history.
