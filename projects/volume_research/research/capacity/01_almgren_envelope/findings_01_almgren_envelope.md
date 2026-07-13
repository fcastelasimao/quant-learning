# Findings C01 — Almgren temporary + two-model envelope

**Date:** 2026-07-08 · **Tickers:** TQQQ, SQQQ · **Question:** the useful core of the paused
Almgren-adoption plan, post-S12 (permanent term dropped): implement Almgren's temporary-impact
term, verify it against the paper exactly, and compare it to the adopted sqrt-law with **σ held
identical** so any movement is the coefficient/exponent choice alone. **Answer: the port is
exact (golden test <1 bp on all 6 Table-3 reference points); at TQQQ/SQQQ's actual σ/ADV and the
strategy's forced 15-min cadence, Almgren predicts ~7.4× more capacity than even the sqrt-law's
optimistic Y=0.3 edge — a real divergence, not noise, and both models are extrapolating outside
their comfortable domains here. The library default stays "sqrt"; flipping it is an owner
decision, not made by this stage.**

## Headline

Two results, cleanly separated:

1. **The port is correct.** `almgren_temporary`/`almgren_permanent` reproduce Almgren, Thum,
   Hauptmann & Li (2005)'s published IBM/DRI worked examples (Table 3) to within 1 bp on all 6
   reference points — this is not in question.
2. **Applied to TQQQ/SQQQ's own σ/ADV at the strategy's 15-min forced cadence, the two models
   diverge sharply** — Almgren predicts *less* impact than sqrt's most optimistic setting
   (Y=0.3), by roughly 7.4×, not merely "near or below" it as originally anticipated. This is a
   genuine extrapolation gap, not a bug (the golden test proves the formula is implemented
   correctly) — explained below.

## Strategy and mathematics

**The port** (`slippage/impact.py`):
```
Permanent:  I = gamma . sigma . (X/V) . (Theta/V)^delta      gamma=0.314, delta=0.25
Temporary:  K = eta . sigma . sign(p) . |p|^beta               eta=0.142,  beta=0.6
Realized:   J = I/2 + K
```
`almgren_permanent` exists **only** to reproduce the paper's realized cost `J` in the golden
test — it is not used anywhere in this project's own pipeline (S12: the wrong mechanism for an
arbitrage-pinned, elastic-supply ETF). `almgren_temporary` is the one threaded through
`expected_slippage_bps` → `CostModel` → `predict_slippage` via `impact_model="almgren"`; a third
mode, `impact_model="envelope"`, reports `band = min/max{sqrt@Y=0.3, sqrt@Y=1.0, Almgren point}`
while leaving the reported *mean* unchanged (still sqrt Y=0.5) — envelope widens the honest
uncertainty band, it does not silently move the point estimate.

**Why the two models diverge so much here — both are extrapolating.** Almgren's `beta=0.6 > 0.5`
means, for participation `p < 1` (`p^0.6 < p^0.5`), the temporary term grows more slowly with
size than the sqrt-law — and `eta=0.142` is itself well below the sqrt-law's Y-band floor
(0.3). Combined, at the **small, short-horizon** regime this strategy operates in (the 15-min
cadence forces `T ≈ 0.038` days — far below Almgren's own `T ∈ [0.1, 0.5]` day test range, and
TQQQ/SQQQ order sizes here are a small fraction of ADV, unlike the paper's institutional-block
sample), Almgren's fitted coefficients — measured on **large-cap single stocks at multi-hour
institutional-block horizons** — simply don't have direct empirical support. **Both models are
extrapolating outside their comfortable domain**: the sqrt-law's Y is a generic literature prior
never fit to any data; Almgren's η/β are fit to data, but not this data, this size regime, or
this horizon.

## Numbers

**Round-trip cost (bps), σ/ADV/spread FIXED across models** (tier: **Calibrated/adopted** for
sqrt's shape, **Calibrated/adopted** for Almgren's coefficients — both extrapolated to this
regime, see caveats):

| notional | TQQQ sqrt(Y=0.3) | TQQQ sqrt(Y=1.0) | TQQQ Almgren | SQQQ sqrt(Y=0.3) | SQQQ sqrt(Y=1.0) | SQQQ Almgren |
|---|--:|--:|--:|--:|--:|--:|
| $100k | 6.6 | 18.5 | 2.6 | 8.8 | 24.7 | 3.6 |
| $1M | 17.7 | 55.4 | 6.0 | 23.5 | 73.7 | 8.4 |
| $5M | 37.6 | 122.0 | 13.4 | 50.1 | 162.3 | 18.7 |
| $10M | 52.6 | 171.9 | 19.5 | 70.0 | 228.7 | 27.4 |
| $30M | 90.1 | 296.7 | 36.4 | 119.8 | 394.7 | 51.1 |
| $100M | 163.2 | 540.5 | 73.3 | 217.1 | 718.9 | 103.1 |

Almgren sits **below sqrt(Y=0.3) at every size tested**, not just "near" it.

**Capacity at a 25 bps round-trip budget** (tier: **Modeled**, numeric bisection —
`capacity_at_horizon` is sqrt-only closed-form; this stage adds a general numeric version for a
fair cross-model comparison):

| ticker | sqrt(Y=0.3) | sqrt(Y=0.5, central) | sqrt(Y=1.0) | Almgren | Almgren / sqrt(Y=0.3) |
|---|--:|--:|--:|--:|--:|
| TQQQ | $2.1M | $762k | $190k | $15.6M | **7.35×** |
| SQQQ | $1.1M | $412k | $103k | $8.5M | **7.42×** |

## Caveats

- **The library default stays "sqrt". This finding does NOT flip it.** Whether to trust Almgren's
  ~7× larger capacity number over the sqrt-law's is exactly the "which extrapolation do you
  trust" judgment call this stage flags for the owner (per plans/2026-07_execution_track.md's
  explicit "do not decide" instruction) — it is not resolved here.
- **Neither model has direct empirical support at this size/horizon.** Almgren's own paper caps
  at ≤10% of ADV over T ∈ [0.1, 0.5] *days*; the sqrt-law's Y was never fit to any data at all.
  The 7× gap is the honest **width of that uncertainty**, not a preference for either model.
- **`envelope` mode is the recommended reporting default going forward** for exactly this reason
  — it surfaces the full sqrt-vs-Almgren spread rather than picking a winner.
- **`almgren_permanent` is unused in the pipeline** — golden-test-only, per S12's finding that its
  single-stock float mechanism doesn't apply to TQQQ/SQQQ.
- **This stage does not re-run the full S06 capacity curve** — that's C02, which runs last and
  will present the envelope band alongside the existing Y-band headline.

## Reproduce

```
/Users/franciscosimao/opt/anaconda3/envs/quant/bin/python \
    research/capacity/01_almgren_envelope/build_01_almgren_envelope.py
/Users/franciscosimao/opt/anaconda3/envs/quant/bin/python -m pytest tests/test_impact.py tests/test_cost.py tests/test_model.py tests/test_predict.py -v
```
Outputs in `results/` (gitignored): `cost_table_{SYM}.csv`, `capacity_table_{SYM}.csv`,
`almgren_envelope.png`.
