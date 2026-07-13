# Findings E03 — The scheduler: worked schedules vs a naive 15-min single fill

**Date:** 2026-07-08 (numbers refreshed 2026-07-09 after audit fixes #1+#2) · **Instrument:** TQQQ
· **Question:** does replacing the hard 15-min single-fill pin with a horizon that trades off
execution cost against alpha-forfeiture (E01) and interruption risk (E02) actually help?
**Answer: yes, and the benefit grows sharply with size and stress — from +3.7 bps at $1M/normal to
+58.5 bps at $20M/stress — while staying honest that the scheduler never chooses to go slower
unless the total (execution + alpha + interruption) objective actually improves, by construction.**

> **2026-07-09 audit refresh:** these numbers supersede the original 2026-07-08 run. Two fixes
> landed: (1) E01's `g(h)` is now a **stretched exponential** `1-exp(-(h/τ)^k)` (k≈2.5), which no
> longer overcharges alpha-forfeiture 2.5-3.5x in the 15-120 min band; (2) the horizon objective
> no longer **double-counts** the completion-branch `g(h)` — alpha is now `edge·(1-hazard)·g(h)`.
> Both bias the old numbers conservative (anti-scheduler); with them fixed the scheduler extends
> horizons further (h*=45-120 min vs 25-105) and the benefit is larger across the board.

## Headline

`schedule_order()` never makes things worse in expectation: because the naive 15-min horizon is
itself one point in the search grid, the chosen `h*` can only match or beat it on the **total**
objective (execution + alpha-forfeiture + interruption). What the worked examples show is *how
much* better, and where the benefit concentrates: **small orders move the needle modestly** ($1M
normal: +3.7 bps, h* extends to 45 min) **while large/stressed orders benefit enormously**
($20M stress: +58.5 bps, h* extends to 120 min) — exactly the shape you'd expect, since execution
cost is convex in size (participation-driven) while alpha-forfeiture and interruption risk grow
only roughly linearly with the small delay needed to fix it.

## Strategy and mathematics

Six worked cells: TQQQ, notional ∈ {$1M, $5M, $20M} × regime ∈ {normal, stress}. `edge_bps=50`
throughout — an **illustrative** per-trade edge (this library doesn't measure the strategy's own
edge; the owner should supply the real, measured number). Regime parameters match S06/C01's
`MARKET` dict exactly (σ 370/510 bps, ADV $4.9B/$2.9B); interval volume uses `ADV_shares / 26`
bins/day as a flat "typical bin" proxy (no `VolumeProfile` loaded in this worked-example stage —
see caveats).

For each cell, two numbers are compared **on the same total-objective basis** (execution cost +
`edge_bps·g(h)` + `hazard(h)·interruption_cost(h,·)·edge_bps`, all at their respective horizons):
the **scheduled** total (at `h*`) and the **naive** total (execution + alpha + interruption, all
evaluated at the fixed 15-min cadence). This is the correct comparison — comparing only raw
execution-cost savings against the alpha-forfeiture, as an earlier draft of this stage did, is
**not** apples-to-apples, since the naive baseline also forfeits some alpha and carries some
interruption risk at h=15 (now tiny under the stretched-exp fit: g(15)≈0.1%, hazard(15)≈1%).

## A worked schedule, by hand: $5M TQQQ, normal regime

Before the aggregate table, the actual slice plan for one cell — verifiable by hand from the
`Schedule` object directly. Interval volume is flat (no `VolumeProfile` loaded, see caveats), so
`h*=75 min` forces `min_bins = ⌈75/15⌉ = 5` bins; with `pov_cap=0.10` each bin's cap
(≈$18.9M) dwarfs the order, so the waterfall splits the $5M **exactly evenly**, one fifth per bin:

| time offset | child notional | order style |
|---|--:|---|
| 0 min | $1,000,000 | cross |
| 15 min | $1,000,000 | cross |
| 30 min | $1,000,000 | cross |
| 45 min | $1,000,000 | cross |
| 60 min | $1,000,000 | cross |

Sum = $5,000,000 (feasible, no cancellation). `expected_slippage_bps = 14.22` (band 8.8–27.7). By
hand: `g(75min) = 1-exp(-(75/256.9)^2.480) = 0.0461`; `hazard(75min,normal) = 0.121`;
`alpha_forfeit_bps = edge_bps·(1-hazard)·g = 50·0.879·0.0461 ≈ 2.03`; `interruption_bps =
edge_bps·hazard·interruption_cost(75,0.5,"cancel") = 50·0.121·(0.5·0.0461+0.5) ≈ 3.16` — matching
the aggregate table's $5M/normal row below exactly (14.22 + 2.03 + 3.16 = 19.41 bps total).

## Numbers

**Normal regime** (tier: **Modeled**, composing E01/E02/P01-P03's measured/fitted pieces):

| notional | h* (min) | slices | scheduled total | naive (h=15) total | improvement |
|---|--:|--:|--:|--:|--:|
| $1M | 45 | 3 | 10.87 bps | 14.53 bps | **+3.66 bps** |
| $5M | 75 | 5 | 19.41 bps | 31.19 bps | **+11.78 bps** |
| $20M | 90 | 6 | 32.24 bps | 61.32 bps | **+29.09 bps** |

**Stress regime:**

| notional | h* (min) | slices | scheduled total | naive (h=15) total | improvement |
|---|--:|--:|--:|--:|--:|
| $1M | 59 | 4 | 16.63 bps | 25.10 bps | **+8.47 bps** |
| $5M | 90 | 6 | 29.97 bps | 54.95 bps | **+24.98 bps** |
| $20M | 120 | 8 | 50.43 bps | 108.94 bps | **+58.51 bps** |

All six cells: `feasible=True`, and the improvement is **positive in every cell** (guaranteed by
construction, since h=15 is itself in the search grid — this is a sanity property, not a finding
per se; the finding is the *magnitude* and how it scales).

**Decomposition at $20M/stress** (h*=120 min): execution cost falls from 108.7 bps (naive) to
38.9 bps (scheduled) — a **69.8 bps** saving — at the cost of 5.5 bps more alpha forfeited and
5.8 bps more interruption risk than the naive baseline's own (near-zero) versions of those terms;
net +58.5 bps. The execution-cost saving dominates by a wide margin at this size — and dominates
*more* than in the pre-fix numbers, because the corrected alpha term is much cheaper.

## Caveats

- **`edge_bps=50` is illustrative**, not measured — every alpha-forfeiture and interruption-cost
  number in this stage scales linearly with it. A real deployment must supply the strategy's
  actual measured edge.
- **No `VolumeProfile` loaded** — slices are flat at a constant "typical bin" volume
  (`ADV/26`), not P01's actual U-shaped intraday curve. A schedule that happens to span the
  market open or the midday lull would look different (and the POV cap would bind harder or
  softer) with the real profile; `schedule_order(volume_profile=...)` supports this, just not
  exercised here.
- **`phi=0.5` in the interruption term** (the assumed average fill-fraction at a random
  interruption time) is a documented simplification in `schedule.py`, not measured.
- **This is a demonstration on representative parameters, not a historical replay** — E04 runs
  the scheduler against every real historical entry and real interruption events.

## Reproduce

```
/Users/franciscosimao/opt/anaconda3/envs/quant/bin/python \
    research/scheduler/03_scheduler/build_03_scheduler.py
/Users/franciscosimao/opt/anaconda3/envs/quant/bin/python -m pytest tests/test_schedule.py -v
```
Outputs in `results/` (gitignored): `summary_{regime}_{notional}.csv`, `all_slices.csv`,
`scheduler_worked_examples.png`.
