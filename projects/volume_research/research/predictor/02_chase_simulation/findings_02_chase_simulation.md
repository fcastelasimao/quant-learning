# Findings P02 — Chase simulation: cross vs chase, full history

**Date:** 2026-07-08 · **Tickers:** TQQQ, SQQQ · **Question:** replace S08's n=15 live estimate of
the limit-chase entry drag (+14.2 bps) with a full-history simulation, and answer "if we still
chase, what timeout?" **Answer: crossing is cheaper than chasing at every timeout tested — S08's
live number sits inside the simulated distribution for both tickers, and the gap to it shrinks
to under 1 bp for SQQQ.**

## Headline

Simulating **every** historical entry signal (2,343 TQQQ / 1,930 SQQQ) on 1-min bars: **Style A
(cross immediately)** costs **2.88 bps (TQQQ) / 4.48 bps (SQQQ)** — close to the half-spread plus a
small latency drift. **Style B (rest a passive limit, timeout T, then cross)** costs **more, not
less**, at every T from 1 to 60 minutes — 6.7–8.7 bps (TQQQ), 9.1–13.2 bps (SQQQ) — because the
price that runs away from an unfilled limit costs more than the spread saved by filling passively.
**This confirms S08's live conclusion (cross beats chase) with 150x the sample size**, and the
live +14.2 bps figure sits comfortably inside the simulated distribution (mean ± 1σ) for both
tickers.

## Strategy and mathematics

A data-integrity finding came first and reshaped the method: **the TRADES CSVs' own
`decision_price` column (FMP-sourced) is not on the same split-adjustment basis as this project's
Alpaca-sourced DB candles** — a direct check found a stable **~2.0x ratio for TQQQ and ~0.2x for
SQQQ** across nearly the entire 2013–2024 history, converging to ~1.0x only near 2026 (consistent
with differing cumulative split adjustments between the two vendors). Mixing them as price levels
silently produced nonsense (thousands of bps). Fix: **use only the DB's own price series** —
"decision price" here is the DB's 1-min bar **open** at `entry_time`, exactly the convention S12's
0a analysis already used successfully.

Two entry styles, simulated at every historical entry:

- **Style A — cross now:** `cost = half_spread(bin) + (close_of_decision_bar - open_of_decision_bar) / open · 1e4`
  — the spread you cross, plus a small drift proxy for the fact you can't literally trade at the
  exact instant of the signal (you still eat the rest of that first minute).
- **Style B — passive limit at the decision price, timeout T, then cross:** filled (cost = 0, a
  maker fill, no spread) if the running minimum **low** over the **T minutes strictly after** the
  decision bar trades at or below the limit; otherwise cross at the price T minutes out:
  `cost = (price_T - decision_price)/decision_price · 1e4 + half_spread(bin at T)`. Both `<=` and
  `<` were run as a fill-boundary sensitivity (results reported use `<=`; `<` is materially
  identical, in the results CSVs). **The wait window deliberately excludes the decision bar
  itself** — a bar's low is tautologically ≤ its own open, so including it would make every limit
  "fill" instantly regardless of what actually happens afterward.

`half_spread(bin)` reuses P01's CS-15min intraday spread curve. Regime labels reproduce S03's
exact 20-day tercile split (same method as P01, S12).

## Numbers

**Overall, T = 15 min** (tier: **Modeled**):

| ticker | Style A (cross) | Style B (chase, T=15) | fill rate | S08 live | sim within S08's 1σ? |
|---|--:|--:|--:|--:|---|
| TQQQ | 2.88 bps | 8.66 bps | 85.2% | +14.2 bps (n=15, 1σ=35) | yes (mean±1σ = [-18.8, 36.1]) |
| SQQQ | 4.48 bps | 13.22 bps | 83.7% | +14.2 bps (pooled) | yes (mean±1σ = [-28.1, 54.5]) |

**Style B cost is not monotone in T** — it *rises* from T=1 to ~T=15 then plateaus (TQQQ:
6.67 → 6.75 → 7.50 → 8.05 → 8.66 → 8.32 → 8.44 bps at T=1/2/5/10/15/30/60). Two effects pull in
opposite directions: a longer wait raises the **fill rate** (63% → 92%, cheaper on average) but
also raises the **adverse drift paid when unfilled** (price has more room to run away before you
're forced to cross), and the second effect dominates until ~15 minutes. This *is* the genuine
execution-side trade-off — E01/E03 add the **alpha-forfeiture** side (the cost of delaying entry
at all) on top of it.

**By volatility regime, T=15** (tier: **Modeled**) — clean monotonic ordering, both tickers:

| ticker | calm | normal | stress |
|---|--:|--:|--:|
| TQQQ | 6.82 bps | 8.39 bps | 13.19 bps |
| SQQQ | 9.03 bps | 11.07 bps | 19.60 bps |

Chasing gets markedly more expensive in stress — consistent with wider spreads and faster-moving
prices both working against a resting limit at once.

**Reconciliation vs. S08 (adverse-selection gap):** TQQQ sim@15min = 8.66 bps vs. S08's pooled
+14.2 bps → **gap 5.5 bps**; SQQQ sim@15min = 13.22 bps vs. S08 → **gap 1.0 bps**. The gap is the
part of the live drag this simulation *doesn't* capture — most plausibly the live strategy's
actual behavior (repricing/re-chasing an unfilled limit rather than resting it statically for a
fixed T, and n=15 sampling noise given S08's own 1σ of 35 bps). SQQQ's near-zero gap and TQQQ's
larger one both remain **inside** the live estimate's own uncertainty band.

## Caveats

- **Tier: Modeled, and prominently so** — 1-min bars cannot see intraminute queue dynamics. There
  is no queue-position model: a "fill" here means the market traded through the limit price
  sometime in that bar, not that a specific resting order at a specific queue position would have
  been filled. This is the standard, acknowledged limitation of any bar-data execution simulation.
- **S08's +14.2 bps is pooled across TQQQ and SQQQ** (n=15 total); this stage reports per-ticker,
  so the "reconciliation" comparison is against a blended number, not a same-ticker one — flagged,
  not fixable without a larger live sample.
- **The `<=` vs `<` fill-boundary sensitivity** is in the results CSVs (`cost_B_T{n}_lt_bps`
  columns) but not tabulated above — differences were immaterial at this bar resolution.
- **No true "optimal timeout" is claimed** — the plan's own E01 stage exists to add the
  alpha-forfeiture penalty that turns this execution-only curve into a real speed-vs-cost
  trade-off; P02 deliberately reports only the raw execution-cost side.

## Reproduce

```
/Users/franciscosimao/opt/anaconda3/envs/quant/bin/python \
    research/predictor/02_chase_simulation/build_02_chase_simulation.py
```
Outputs in `results/` (gitignored): `chase_sim_events_{SYM}.csv` (per-trade, all T/sensitivity
columns), `chase_sim_summary_{SYM}.csv`, `chase_sim_by_regime_{SYM}.csv`,
`chase_sim_by_bin_{SYM}.csv`, `chase_simulation.png`.
