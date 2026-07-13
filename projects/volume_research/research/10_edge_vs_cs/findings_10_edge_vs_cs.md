# Findings 10 — EDGE vs Corwin–Schultz (Stage 12)

**Date:** 2026-07-03 · **Tickers:** TQQQ, SQQQ, QQQ · **Question:** should the EDGE estimator
(Ardia, Guidotti & Kroencke, *JFE* 2024) replace Corwin–Schultz as the library's spread estimator?
**Answer: no — not at our 15-min resolution.** EDGE is added to the library as a faithful, opt-in
estimator; **CS stays the default.**

## Why we looked

A colleague-supplied `edge_spread_estimator` (from Gemini) turned out to be **broken** (it crashed)
and **not** EDGE — invented constant, no cross-bar moment conditions, no discreteness correction.
The *real* EDGE is a genuinely better estimator in principle (uses all four OHLC prices vs CS's
high/low; asymptotically unbiased under sparse/discrete trading; GMM-optimal). So we ported it
faithfully and ran a data-verification gate before adopting it (per the CLAUDE.md mandate).

## The port is correct

`slippage.spread.edge()` is a line-for-line numpy port of the authors' reference
`bidask.edge()` and matches it **exactly** (diff = 0 across signed/unsigned, random samples;
`tests/test_spread.py::test_edge_matches_bidask_reference`). On a high-power synthetic bid-ask
bounce it recovers a known spread to **~2%**. So what follows is not an implementation problem.

## The gate result — EDGE is unreliable at 15-min

Aggregate one-way half-spread on the real 15-min bars, by method:

| ticker | CS (current default) | EDGE per-session¹ | EDGE whole-sample² | EDGE rolling(26)² |
|---|---:|---:|---:|---:|
| TQQQ | 0.74 bp | ~0.0 | 7.9 | 10.2 |
| SQQQ | 1.00 bp | ~1.8 | 7.9 | 10.4 |
| QQQ  | 0.72 bp | ~0.0 | 2.7 | 3.3 |

¹ respects session boundaries (correct for intraday). ² ignores them.

**The estimate swings from 0 to 10 bps depending purely on how EDGE is aggregated** — it is not a
usable number for our pipeline. Two compounding problems:

1. **Underpowered per session.** With ~26 bars/day and a sub-bp true spread swamped by 15 minutes
   of volatility, the per-session EDGE estimate scatters symmetrically around zero (session p90
   ≈ 4 bps, but the mean / pooled-s² ≈ 0). Robust across two poolings (signed-mean and pooled-s²).
2. **Overnight-gap contamination.** Applying EDGE across session boundaries (whole-sample or
   rolling) lets the prev-close→open overnight jump enter its `o − c₁` moment. For 3× ETFs that gap
   is huge, inflating the estimate to a spurious ~8–10 bps.

EDGE's designed domain is **daily bars / long gap-free samples** (the paper; crypto minute data
works because it is 24/7 with no gaps). Equity 15-min — overnight gaps plus a sub-bp spread — is
its worst case.

**Daily bars don't rescue it *for our names* either.** Running EDGE on daily OHLC (its home turf)
over the 2y window gives **TQQQ 75 bp, SQQQ 81 bp, QQQ 26 bp** half-spread — ~100× too high. The
reason is the flip side of the same problem: a 3× ETF's daily range is enormous (~340 bps of vol),
so the true ~1 bp spread is a rounding error inside it and the estimator has no signal. **The rule
isn't "use EDGE on daily" or "use EDGE on QQQ" — it's use EDGE when the spread is a non-trivial
fraction of the bar's range**, i.e. genuinely wider-spread / less-liquid instruments (small/mid
caps, illiquid ETFs, crypto). None of TQQQ/SQQQ/QQQ qualify at any resolution — for these,
15-min CS is the only estimator that yields a sane number (validated in findings_11).

## Decision

- **CS remains the default** (`calibrate(spread_method="cs")`). It is stable (~0.7–1 bp), validated
  (findings_01), and — critically — **the spread is a small, non-binding floor** dominated by
  impact + timing, so nothing about the capacity headline depends on refining it. The live
  market-order sells (~0 bps, findings_08) corroborate a tiny spread.
- **EDGE ships as an opt-in** (`calibrate(spread_method="edge")`, plus `edge()` / `edge_intraday()`
  exported) with a docstring caveat: it's for daily-bar / long gap-free use, underpowered at 15-min.
- **No re-baseline** of findings_01/04/05/06/07 — the capacity chain is untouched.

## Answering the related code question (`_overnight_shift`)

The CS `_overnight_shift` is the textbook CS overnight correction — a *bias-removal* step (an
overnight gap inflates the `gamma` range term, misread as volatility → overpriced spread), not a
price model. In our pipeline it is already **inert**: `corwin_schultz_intraday` runs with
`overnight_adjust=False` (within-session pairs only). Its docstring now says so. EDGE, notably,
would have *needed* an analogous guard — its cross-session `o − c₁` term is exactly the overnight
contamination seen above.

## Reproduce

```
/Users/.../envs/quant/bin/python research/10_edge_vs_cs/build_10_edge_vs_cs.py
/Users/.../envs/quant/bin/python -m pytest tests/test_spread.py -q
```
Outputs in `results/` (gitignored): `edge_vs_cs.csv`, `edge_vs_cs.png`.
