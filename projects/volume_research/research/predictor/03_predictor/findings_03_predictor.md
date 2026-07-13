# Findings P03 — The predictor: `predict_slippage()`

**Date:** 2026-07-08 · **Question:** the client's "I set an order at X and filled at X + diff —
predict diff." **Answer: `slippage/predict.py::predict_slippage(notional_usd, side, order_type,
state, price)` — one function combining P01's market state, P02's measured chase cost, and S03's
empirical (non-Gaussian) timing quantiles into a mean + p50/p90/p95 band, itemized by component.**

## Headline

Composing three already-measured pieces — no new measurement in this stage. The worked examples
below confirm the project's standing headline at the predictor level: **timing risk (the p90/p95
tail) dwarfs the mean** — even a calm-regime, $100k cross has a mean of 2.8 bps but a p95 of
59 bps, because a single 15-min fill delay carries ~17–59 bps of symmetric risk (S03) regardless
of size. **Chasing costs more than crossing in every cell of the grid** (P02's finding, now
exposed at the predictor level), and **impact only starts to matter at $10M+** — at $100k it's a
fraction of a bp; at $10M it's 15–20 bps, comparable to or larger than the chase drag itself.

## Strategy and mathematics

```
predict_slippage(notional_usd, side, order_type, state, price, latency_min=15)
  -> {mean_bps, p50_bps, p90_bps, p95_bps, components: {...}}
```

- **`spread_bps`** = `state.spread_bps` for `order_type="cross"`; **0** for `"limit_chase"` — the
  P02 chase-cost curve is a blended empirical mean (fill-at-limit=0 mixed with
  cross-if-unfilled=drift+spread) that already contains the effective spread paid on unfilled
  attempts, so reporting it again separately would double-count it.
- **`impact_band_bps`** = `impact.py::impact_bps()` at `Y ∈ {0.3, 0.5, 1.0}` (the adopted √-law
  band, unchanged), with participation computed from `notional_usd` against
  `state.expected_interval_volume × price`, scaled from the 15-min forecast window to the
  requested `latency_min`.
- **`drag_bps`** = 0 for `"cross"`; the **P02 chase-cost curve** (linearly interpolated between
  the measured T ∈ {1,2,5,10,15,30,60 min} points, findings_02) for `"limit_chase"`. Symbols
  outside {TQQQ, SQQQ} fall back to TQQQ's curve (documented, not fitted).
- **`timing_sigma_bps`** = `state.sigma_now_bps` scaled to `latency_min` via the same √t
  convention as `MarketParams.sigma_1min_bps` elsewhere in the library.
- **`mean_bps`** = `spread + impact(Y=0.5) + drag` — timing is **excluded** from the mean (it's a
  variance channel, mean ≈ 0, per S03/the library-wide convention).
- **`p50/p90/p95`** = `mean_bps + multiplier × timing_sigma_bps`, where the multipliers
  (0 / 0.98 / 1.44) are **empirical, not Gaussian** — measured directly from S03's own
  `delay_costs_bps`/`timing_risk_bps` machinery at the 15-min horizon on real TQQQ 1-min data
  (checked stable across 1–60 min). A Gaussian would use 0 / 1.28 / 1.64 instead — the measured
  p90 multiplier is *below* the Gaussian value (thinner shoulder) while p95 is close, the
  leptokurtic "sharp peak, fat far-tail" shape S03 already documented.

## Numbers

**Worked examples, TQQQ, `latency_min=15`** (tier: mean = **Calibrated/Modeled composite**;
p90/p95 = **Modeled**, non-Gaussian empirical quantiles):

| regime (σ_now) | size | cross mean | chase mean | cross p95 | chase p95 |
|---|--:|--:|--:|--:|--:|
| calm (200 bps) | $100k | 2.8 | 10.7 | 59.3 | 67.2 |
| calm | $1M | 7.3 | 15.2 | 63.8 | 71.7 |
| calm | $10M | 21.6 | 29.5 | 78.0 | 86.0 |
| normal (300 bps) | $100k | 3.9 | 11.8 | 88.6 | 96.5 |
| normal | $1M | 10.6 | 18.5 | 95.3 | 103.3 |
| normal | $10M | 32.0 | 39.9 | 116.7 | 124.6 |
| stress (450 bps) | $100k | 5.4 | 13.3 | 132.5 | 140.4 |
| stress | $1M | 15.5 | 23.5 | 142.6 | 150.6 |
| stress | $10M | 47.6 | 55.5 | 174.6 | 182.6 |

Full grid (all size × regime × order-type cells, plus p50/p90 and every component) in
`results/predictor_worked_examples.csv`.

**Sanity checks confirmed by tests** (`tests/test_predict.py`, 15 tests): each component matches
its source stage exactly at a reference point (impact vs. `impact_bps()`, drag vs. the P02
T=15 TQQQ number, timing vs. the √t formula); `p50 < p90 < p95` always; chase's mean ≥ cross's
mean for both TQQQ and SQQQ synthetic states.

## Caveats

- **No new measurement** — this stage is pure composition. Every number traces to P01, P02, or
  S03; nothing here should be read as an independent validation of those stages.
- **Chase-drag curve is TQQQ/SQQQ-specific**, linearly interpolated, with an undocumented
  (flagged) TQQQ fallback for any other symbol — do not trust `predict_slippage` on a name
  outside those two without recalibrating P02 for it.
- **Timing quantile multipliers are fixed constants** (measured once at the 15-min horizon,
  TQQQ), not recomputed per-call — a documented simplification (checked roughly horizon-stable
  1–60 min; SQQQ runs a few % higher at p95, using TQQQ's slightly more conservative values for
  both).
- **`impact_model="almgren"` raises `NotImplementedError`** — a named hook for C01, which hasn't
  landed yet. Only `"sqrt"` (the adopted Y-band) works today.
- **`side="sell"` is accepted but not separately calibrated** — S08/P02 only measured buy-side
  chase drag (TQQQ/SQQQ are long-only books); sells are assumed clean per S08's own finding, but
  that's an assumption carried through, not a sell-specific measurement.

## Reproduce

```
/Users/franciscosimao/opt/anaconda3/envs/quant/bin/python \
    research/predictor/03_predictor/build_03_predictor.py
/Users/franciscosimao/opt/anaconda3/envs/quant/bin/python -m pytest tests/test_predict.py -v
```
Outputs in `results/` (gitignored): `predictor_worked_examples.csv`,
`predictor_worked_examples.png`.
