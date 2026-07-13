# Findings 01 — Bid-ask spread estimate (Corwin–Schultz)

**Date:** 2026-06-24 · **Tickers:** TQQQ, SQQQ, QQQ · **Method:** Corwin–Schultz (2012) on
15-min bars, within-session pairs, full history 2010→2026.

> **Validated (findings_11, Stage 13):** CS-15min was later checked against real Alpaca **SIP
> NBBO** quotes — accurate for TQQQ/SQQQ (0.74 vs 0.89, 1.00 vs 1.32 bp, conservatively low); it
> overstates QQQ ~2× (high-price bias, immaterial — QQQ untraded). EDGE was evaluated (findings_10)
> and kept as opt-in: unreliable for these ultra-liquid names at any resolution.

## Headline

The real half-spread for our instruments is **~0.7–1.0 bps on average**, ~**2.5–2.8 bps at the
open**, sub-bp midday, spiking to ~**3.5 bps in stress years (2020, 2022)**. **All of this sits under
the flat 5 bps entry the backtests charge** — so the spread component is *not* the binding cost, and
the current flat assumption is conservative (our Sharpes are not flattered by an optimistic spread).
The interesting cost lives at scale (impact), not in the spread.

## Strategy, mathematics

### The strategy (why this estimator)

We need the **bid-ask spread** — the cost of crossing the book at one instant — but we have **no
quote data**, only OHLC candles. The trick (Corwin–Schultz 2012) is that a candle's **high is
struck at the ask and its low at the bid**, so the *observed* high–low range carries two things
mixed together: the genuine price diffusion over the bar (**volatility**, which grows with the
length of the interval) and a fixed **spread** inflation (which does *not* grow with the interval).
That different scaling with time is the whole lever. Compare the squared log-range of **two single
bars** against the squared log-range of the **one combined two-bar window**: volatility contributes
in proportion to elapsed time (so it doubles), while the spread contributes the same way regardless.
Two observations, two unknowns (σ and S) → solve for the spread without ever seeing a quote.

### The estimator (as implemented in `slippage/spread.py`)

For each consecutive pair of bars `t`, `t+1` with highs `H` and lows `L`:

```
β = ln(H_t /L_t )² + ln(H_t₊₁/L_t₊₁)²        # the two single-bar squared log-ranges
γ = ln(H₂/L₂)²,  H₂=max(H_t,H_t₊₁), L₂=min(L_t,L_t₊₁)   # the combined two-bar window
                                              # k = (√2 − 1)² = 3 − 2√2 ≈ 0.1716
α = (√(2β) − √β)/k  −  √(γ/k)
S = 2·(e^α − 1)/(e^α + 1)                     # proportional (round-trip) spread
```

- `β` isolates **2 bars' worth of volatility + 2 spread imprints**; `γ` carries **2 bars' worth of
  volatility + 1 spread imprint** (the combined window is hit by the spread once). The `α`
  combination is algebraically arranged so the volatility terms cancel and only the spread survives;
  the logistic-style map `S = 2(e^α−1)/(e^α+1)` turns it back into a proportional spread.
- **`S` is the round-trip proportional spread.** A marketable order crosses **half** of it per side,
  so the per-trade cost floor is `half_spread = S/2`, and `×1e4` expresses it in **bps**
  (`half_spread_bps` in the library). Every number in the tables below is this `S/2·1e4`.

### Assumptions and where they bite

CS assumes a **continuous, driftless diffusion** within each bar and that high=ask, low=bid. Two
consequences we hit directly:
- When a bar's genuine diffusion range is tiny (too-fine bars, or very liquid names), `√(2β)−√β`
  shrinks toward the `√(γ/k)` term and **`α`, hence `S`, can go negative** — the estimator breaks
  (discreteness / bid-ask bounce, not diffusion, drives the range). This is the resolution floor
  discussed in *Methodological decisions* — sub-10-min bars give negative signed means for the 3× ETFs.
- A **drift** across the two bars (e.g. an overnight gap) inflates `γ` → reads as extra volatility →
  distorts `S`. Hence the **overnight adjustment** (shift bar `t+1`'s range so it meets bar `t`'s,
  removing the gap return) for daily bars, and **within-session-only pairing** intraday so no pair
  ever straddles the overnight boundary.

### From per-pair `S` to the reported numbers

The per-pair `S` is **unbiased only in the mean of the signed values** — individual estimates are
noisy and right-skewed (hence median > mean in the table). So we average the **signed** per-pair
`S`, **clamp the aggregate at 0** (clamping each pair first biases the mean badly upward — see
*Methodological decisions* §2), then take `S/2·1e4`. Time-of-day and by-year figures are the same
computation restricted to those bars; percentiles describe the estimator's noisy tail, not a
literal spread level.

## Numbers

Overall mean half-spread, 15-min CS (bps):

| ticker | n (pairs) | mean | median | p90 | p95 |
|---|---|---|---|---|---|
| TQQQ | 102,176 | 0.74 | 3.55* | 25.0 | 35.0 |
| SQQQ | 102,014 | 1.00 | 3.76* | 25.5 | 35.2 |
| QQQ  |  95,271 | 0.72 | 1.57* |  8.7 | 12.1 |

\* median > mean because per-pair estimates are noisy and right-skewed; the **mean of the signed
estimates is the CS point estimate** (clamped at 0), not the median. Percentiles describe the
estimator's noisy tail, not a literal spread.

Intraday (time-of-day) half-spread, bps:

| | open 09:45 | midday | close 15:45 |
|---|---|---|---|
| TQQQ | 2.53 | ~0.6 | 0.74 |
| SQQQ | 2.76 | ~1.0 | 0.96 |
| QQQ  | 1.19 | ~0.65 | 0.69 |

Wide at the open, tightening through the day — the textbook microstructure pattern. QQQ (most liquid)
is tightest; the leveraged ETFs are slightly wider. Magnitudes are realistic for a penny-wide spread
on these names, which is the main external sanity check (no quote data available to compare directly).

By year (TQQQ mean half-spread, bps): ~0 in 2010–2013, rising to ~1.4 in 2018–2019, **3.5 in 2020**,
2.3 in 2021, **3.2 in 2022**, easing to ~1.7 by 2025–2026. Clear regime structure.

## Methodological decisions

1. **Resolution matters, and CS has no plateau — it pins the magnitude, not the exact level.** CS is a
   *range/volatility-scaling* estimator: it needs genuine diffusion range per bar. Too fine breaks it;
   too coarse loses the spread in the range. Resampling all intraday bars from 1-min (one clean
   source), the TQQQ signed-mean half-spread climbs **monotonically** with bar size:

   | resolution | signed mean (bps) | 2023+ signed | %neg | zero-range |
   |---|---|---|---|---|
   | 1-min  | −1.48 | −0.03 | 50% | 5.1% |
   | 5-min  | −0.46 | +0.91 | 44% | 0.7% |
   | 10-min | +0.30 | +1.65 | 42% | 0.2% |
   | 15-min | +0.77 | +2.18 | 41% | 0.06% |
   | daily  | <0 (pure noise) | — | — | — |

   - **Below ~10-min the estimator is broken** (negative signed mean) — within-bar price barely moves,
     so discreteness / the bid-ask bounce dominates and violates CS's continuous-price assumption (the
     opposite of "finer = better"). The threshold is **volatility-dependent**: calm QQQ recovers by
     5-min (0.34 bps); the 3× TQQQ/SQQQ need 10-min+.
   - **The estimate never plateaus** — it keeps rising with coarseness. So CS does *not* sharply
     identify the spread level for these ultra-liquid names; it gives the order of magnitude
     (sub-bp to ~1 bp half typical), not a precise number. A bounce-based estimator (Roll 1984 /
     Abdi–Ranaldo) would be the tool to pin it — out of scope, and unnecessary since spread isn't the
     binding cost.
   - Resampled-15min (0.77) ≈ native-15min (0.74) — consistency check passed.
   - **Operating choice: native 15-min** — the coarsest clean intraday resolution, a mildly
     conservative point on the ramp. Daily is noise; 1–5 min are broken.
2. **Aggregation:** CS is unbiased only when **signed** per-pair estimates are averaged and the
   **aggregate** is clamped at zero. Clamping each pair *then* averaging biases the mean badly upward
   (an early version reported a spurious 43 bps for TQQQ). Fixed.
3. **Intraday pairing is within-session only** — consecutive 15-min bars are never paired across the
   overnight boundary (resolves the BUILD_LOG open question; avoids overnight-gap contamination).
4. **Overnight adjustment** is on for daily, off for intraday (no overnight gap inside a session).

## Resolution / time / regime diagnostic (`diagnose_resolution.py`)

Swept 1→30 min (open-aligned bins resampled from 1-min) and sliced by 5-year window and
volatility regime (terciles of trailing 20-day realized vol). Signed means, bps:

**By ticker — no plateau:** the ramp keeps rising through 30-min for all three; it never settles.
TQQQ −1.48 (1m) → 0.77 (15m) → 1.73 (30m); SQQQ similar; QQQ flatter (−0.02 → 1.04) and crosses
zero earlier (calmer/more liquid). Confirms CS does not identify a spread *level* here — coarser
always gives more.

**By 5-year window (TQQQ) — time variation is volatility-driven, not secular tightening:**

| res | 2010–2014 | 2015–2019 | 2020–2024 | 2025–2026 |
|----|----|----|----|----|
| 15m | −1.66 | 0.82 | 2.76 | 1.87 |
| 30m | −0.62 | 1.56 | 3.74 | 3.27 |

2020–2024 is *highest* (COVID + 2022 bear), not lowest — so the era effect tracks each period's
volatility, not a clean liquidity-improvement trend. **2010–2014 is negative at every resolution —
CS fails outright on the thin early history; do not use pre-2015 levels.**

**By volatility regime (TQQQ) — clean monotonic ordering** stress > normal > calm at every
resolution (15m: calm 0.10 / normal 0.62 / stress 1.72; 30m: 0.57 / 1.83 / 2.89). Real and
directional, but **conflates genuine stress widening with CS's upward volatility bias** — both push
the same way, so read it as "spreads are wider in stress," not as a calibrated stress multiple.

**Robustness of the headline:** across *all* resolutions, eras, and regimes, the CS half-spread tops
out around **~3.7 bps** (2020–2024, 30-min — a coarse, upward-biased cell). It never reaches the flat
**5 bps**. So "spread is small and the flat 5 bps is conservative" holds in every slice, not just on
average.

## Caveats

- **CS is volatility-sensitive.** Part of the 2020/2022 "widening" is the estimator's known positive
  bias under high volatility, not purely genuine spread widening. Treat stress-year levels as an
  **upper bound** on the spread, not a clean measurement.
- **No quote-data ground truth.** We validate by (a) a synthetic recovery test (`tests/test_spread.py`,
  recovers a known spread in the mean) and (b) realistic magnitudes + the expected intraday shape. A
  TAQ cross-check would be stronger but is out of scope.
- **Early years (2010–2013) read ~0**, likely thin/coarse early 15-min data rather than truly zero
  spread. Don't lean on pre-2014 levels.

## Implication for the cost model

The spread floor for these instruments is **< 3 bps half (one-way) in all but stress, < 1 bp typical**.
Use the **time-of-day curve** (open ≈ 2.5 bps, midday ≈ 0.7 bps) as the spread term in `cost(Q)`, and a
stress level of ~3.5 bps for the low-liquidity scenario. Spread is a small, near-constant floor — the
capacity story will be driven by the **impact** term (next stages), not this.

## Reproduce

```
/Users/.../envs/quant/bin/python research/01_spread_estimate/build_01_spread_estimate.py
/Users/.../envs/quant/bin/python -m pytest tests/test_spread.py -v
```
Outputs in `results/` (gitignored): `cs_spread_daily.csv`, `cs_spread_summary.csv`,
`cs_spread_intraday_tod.csv`, `cs_spread.png`.
