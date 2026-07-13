# Findings E04 — Historical replay: what the scheduler actually saves

**Date:** 2026-07-08 (numbers refreshed 2026-07-09 after audit fixes #1+#2) · **Tickers:** TQQQ,
SQQQ · **Question:** does `schedule_order`'s benefit (E03, shown on representative parameters
against a *modeled* average hazard) survive being replayed against **real** historical entries and
**real** interruption events? **Answer: yes at small size — meaningfully so — but the benefit
shrinks and, by $5M+, reverses. E03's theoretical demonstration was too optimistic at large size
because it evaluated the schedule against the modeled *average* hazard, not the realized fact that
a specific trade's real exit often arrives before a long schedule finishes — and it charged market
impact, which this replay does not (see the load-bearing caveat below).**

> **2026-07-09 audit refresh:** re-run after fixes #1 (stretched-exp g(h)) and #2 (objective
> double-count). Because the corrected alpha term is cheaper, the scheduler now chooses **longer**
> horizons, which in the real replay means **lower fill rates** (TQQQ $20M: 60%→42%) and an
> **earlier cost crossover** (TQQQ scheduled now beats naive only at $100k–$1M, loses by $5M).
> The qualitative finding is unchanged and if anything sharper: longer horizons are more exposed
> to real interruptions. The two baselines (naive 21.35 / VWAP 18.60 bps for TQQQ) are unchanged —
> they don't depend on the scheduler.

## Headline

At **$100k–$1M**, the scheduler clearly wins: TQQQ 14.5–18.7 bps (scheduled, ≥94% filled) vs
21.4 bps (naive 15-min) and 18.6 bps (day-VWAP); SQQQ similarly (22.3–26.0 vs 33.2 / 36.0). At
**$5M**, the scheduler's realized cost on the filled portion (TQQQ 23.1 bps) is already *worse*
than the naive baseline (21.4), and fill rates have dropped to ~70%. **At $20M, TQQQ's scheduled
cost (27.4 bps, only 42% filled) is worse than both baselines** (naive 21.4, VWAP 18.6) — the
single clearest, most important finding of this stage: **E03's optimistic worked examples assumed
full completion AND charged impact; real interruption events bite harder than the modeled average
hazard implied** (a large order's optimal horizon grows long enough that a specific trade's actual,
often much shorter, hold time cuts it off before it finishes), **while this replay charges no
impact at all** (so the naive baseline stays flat at 21.4 bps across every size — see caveat).

## Strategy and mathematics

Every historical entry (TQQQ 2,315 / SQQQ 1,904 aligned trades) × a notional sweep
($100k/$1M/$5M/$20M) = **9,260 + 7,616 = 16,876** (trade × size) replay events. For each: a
`MarketState` built from **real** data at that entry (decision price = DB open, per P02's
price-basis fix; σ = a real trailing-120-min nowcast via `sigma_now_bps` on actual 1-min returns;
interval volume = the real 15-min bar's own volume; regime = S03's daily label; spread = the
findings_01 representative constant) — then `schedule_order` is called for real, and its slices
are **replayed against real 1-min closes** for the executed cost.

**The event-stream interface, exercised for real:** each trade's *actual* `exit_time` is the
interruption event — any slice scheduled after it is cancelled, exactly the design E02's module
docstring promised ("a caller can later feed real backtest or live events instead of the
[modeled] averages"). This is that promise kept: E03 used the modeled hazard; E04 uses ground
truth.

Two baselines, also priced off real 1-min data: a **naive 15-min single fill** (price at
entry+15min + spread) and a **day-VWAP** (volume-weighted average close over the rest of the
session + spread) — the "just be patient with a standard algo" benchmark.

**Reporting convention:** `scheduled_bps` is the cost on the **filled portion only** (paired with
`fill_rate`), not blended with a cancellation penalty for the unfilled residue — deliberately, so
a reader sees both numbers rather than a single figure that hides which one moved. The baselines
always fill 100% by construction (one atomic execution), so `fill_rate < 100%` on the scheduled
side is itself part of the honest comparison, not an artifact to explain away.

## Numbers

**By notional** (tier: **Measured** cost realization, **Modeled** schedule construction):

| symbol | notional | n | fill rate | scheduled (filled) | naive 15-min | day-VWAP |
|---|--:|--:|--:|--:|--:|--:|
| TQQQ | $100k | 2,315 | 100% | **14.53** | 21.35 | 18.60 |
| TQQQ | $1M | 2,315 | 94% | **18.72** | 21.35 | 18.60 |
| TQQQ | $5M | 2,315 | 69% | 23.09 | 21.35 | **18.60** |
| TQQQ | $20M | 2,315 | 42% | 27.38 | **21.35** | **18.60** |
| SQQQ | $100k | 1,904 | 99% | **22.33** | 33.19 | 35.96 |
| SQQQ | $1M | 1,904 | 94% | **26.02** | 33.19 | 35.96 |
| SQQQ | $5M | 1,904 | 70% | **30.01** | 33.19 | 35.96 |
| SQQQ | $20M | 1,904 | 39% | 33.26 | **33.19** | 35.96 |

(bold = the cheapest of the three at that cell.) Post-fix, TQQQ crosses over earlier — scheduled
beats naive only at $100k–$1M, loses from $5M on. SQQQ stays nominally cheapest on the *filled
portion* up to $5M, but note its fill rate collapses to 39% at $20M, so that "win" is on less than
half the order. **Both baselines are flat across size because no impact is charged (see caveats) —
the honest large-size comparison is the fill-rate collapse, not the cost line.**

**By regime** (all sizes pooled) — fill rate is *lowest in calm* (TQQQ 69%, SQQQ 66%) and
*highest in stress* (TQQQ 87%, SQQQ 85%), the **opposite** of the naive expectation that stress
would interrupt schedules more. Not explained by E02's own hazard curve (which showed no clean
regime ordering) — flagged as a real, observed pattern in this replay, not a hypothesis this
stage can confirm the mechanism for; worth a dedicated follow-up rather than a forced narrative.

## Caveats

- **Market impact is NOT charged in this replay — realized cost is drift + spread only** (audit
  fix #3, 2026-07-09). Each slice is priced at the real 1-min close minus decision price, plus the
  representative half-spread; there is no participation-driven impact term. This is why **both
  baselines are identical at every size** (naive 21.35 / VWAP 18.60 bps for TQQQ, flat from $100k
  to $20M) — a real $20M single fill would carry large impact the naive number omits. **Therefore
  the large-size cost *comparison* overreaches: the decision-grade finding at $20M is the fill-rate
  collapse (42%), not "scheduled 27.4 > naive 21.4."** E03 (which *does* charge impact) and E04
  (which does not, but adds real interruptions) bracket the truth; neither alone is the whole
  large-size picture.
- **This is the load-bearing finding that qualifies E03.** Anyone using `schedule_order` at
  $5M+ should read fill_rate alongside the cost number — a low fill rate means the "savings"
  are real only on the executed fraction, and the residual is being cancelled (E02's default
  mode), not banked.
- **`edge_bps=50` is still illustrative** (as in E03) — the schedule construction (hence h* and
  the resulting fill-rate exposure) scales with it.
- **spread is a constant, not a live curve** — every realized-cost number here carries findings_01's
  representative half-spread, not a time-of-day-varying one.
- **The calm/stress fill-rate pattern is observed, not explained** — a genuine open question for
  follow-up, not resolved here.
- **This harness is the reusable artifact** — point `load_trades`/`load_1min`/`load_15min_volume`
  at new data and swap `exit_time` for a real event stream (live stop-fires, signal-flips) to
  replay against a different history or a forward-looking paper-trading log.

## Reproduce

```
/Users/franciscosimao/opt/anaconda3/envs/quant/bin/python \
    research/scheduler/04_replay/build_04_replay.py
```
Runtime ~2.5 min (full history, both tickers, 4-size sweep — no sampling). Outputs in `results/`
(gitignored): `replay_events.csv` (every trade x size event), `summary_by_size.csv`,
`summary_by_regime.csv`, `replay_summary.png`.
