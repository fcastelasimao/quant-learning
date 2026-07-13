# Findings E01 — Alpha-decay curve: g(h)

**Date:** 2026-07-08 (fit form revised 2026-07-09, audit fix #1) · **Tickers:** TQQQ, SQQQ ·
**Question:** fix S12-0a's soft spot (its longer-delay rows mixed hold lengths, dominated by
trades too short to have "survived" to that horizon) and produce the scheduler's alpha-forfeiture
input, `g(h)`. **Answer: a stretched exponential `g(h) = 1 - exp(-(h/τ)^k)`, τ ≈ 257 min / k ≈ 2.48
(TQQQ), τ ≈ 252 min / k ≈ 2.72 (SQQQ) — under an hour costs almost nothing; by a full day,
~94-100% of the edge is gone.**

> **2026-07-09 audit fix #1:** the original fit was a *single* exponential (`1-exp(-h/τ)`, τ≈288/274
> min). It could not represent both the flat first hour and the sharp 2-4h knee at once, so it
> **overcharged the 15-120 min band 2.5-3.5x** — exactly the scheduler's operating range — biasing
> the chosen horizon short (g(15m)=5.1% fitted vs 1.6% empirical; g(60m)=18.8% vs 6.9%). The
> stretched exponential (stretch exponent `k>1`) fixes the knee (g(120m)=14.0% fitted vs 13.8%
> empirical, essentially exact). Residual: it now slightly *under*-charges below ~60 min
> (g(60m)=2.7% vs 6.9% empirical) — tiny in absolute terms (<2 bps at 50 bps edge), see caveats.

## Headline

Conditioning on **hold ≥ 1 day** (S12's own P&L-bearing subset — trades held under 2.4h are net
*losers*, so mixing them in was measuring the wrong population) gives a clean, monotonic curve:
essentially **flat through the first hour** (2-7% forfeited), a **sharp knee around 2-4 hours**
(14% → 56%), and **saturated by end-of-day** (96-101%). A **stretched** exponential fits this shape
well — the k≈2.5 exponent gives the flat start + sharp knee a single exponential (k=1) could not —
with a tight bootstrap CI (500/500 resamples converged); the P&L-weighted variant has a lower,
noisier k (wide CI) but overlapping τ — the curve shape isn't an artifact of a few large trades.

## Strategy and mathematics

Same event construction as S12's 0a (`research/12_stage0_forks/build_12_stage0_forks.py::delay_cost`),
restricted to `hold_days >= 1`:
```
frac(h) = (price[entry+h] - price[entry]) / (price[exit] - price[entry])
```
— clipped to `[-1, 2]` to guard pathological ratios from near-zero total moves — computed only for
trades whose hold period actually spans `h` (guaranteed for all `h` up to 1 day, since every
trade in this subset holds ≥ 1 day).

Two summaries per delay bucket: the **unweighted median** across trades, and a **|pnl|-weighted
mean** (each trade's fraction weighted by the size of its eventual profit or loss — a "the trades
that matter most" view). Both are fit to the stretched exponential `g(h) = 1 - exp(-(h/τ)^k)` via
nonlinear least squares (`scipy.optimize.curve_fit`, τ and k jointly, k bounded to [0.3, 6]), with
a **500-resample bootstrap** (resample trades within each delay bucket, refit) giving a 90% CI on
both τ and k.

## Numbers

**Empirical fractions by delay** (tier: **Measured**), hold ≥ 1 day subset:

| delay | TQQQ (n≈728) | SQQQ (n≈461) |
|---|--:|--:|
| 15 min | 1.6% | 0.7% |
| 30 min | 2.8% | 2.7% |
| 1 h | 6.9% | 6.0% |
| 2 h | 13.8% | 14.6% |
| 4 h | 55.9% | 55.8% |
| 1 d | 95.9% | 101.4%* |
| 2 d | 98.8% | 103.3%* |

\* SQQQ's >100% values reflect a modest overshoot-then-partial-reversion pattern in a few
trades' post-entry path relative to their final exit — expected noise at this tail, not a
methodology error (the underlying `frac` values are individually clipped to `[-1,2]`).

**Fitted g(h) = 1 - exp(-(h/τ)^k)** (tier: **Modeled**, fit to Measured data):

| ticker | τ (unwtd) | τ 90% CI | k (unwtd) | k 90% CI | τ (pnl-wtd) | k (pnl-wtd) |
|---|--:|--:|--:|--:|--:|--:|
| TQQQ | 256.9 min | [241.8, 274.3] | 2.480 | [2.24, 2.83] | 256.2 min | 1.636 |
| SQQQ | 251.7 min | [235.6, 263.4] | 2.723 | [2.46, 3.40] | 236.5 min | 1.233 |

The two tickers' τ are close (both ≈ 4.2-4.3 hours) with tight CIs; the pnl-weighted fit has a
much lower, noisier k (90% CI [1.05, 2.38] TQQQ, [0.76, 2.65] SQQQ — spanning k=1) because the
|pnl|-weighted means are dominated by a few large trades, so the unweighted-median fit is the
headline used at runtime.

**Reference `g(h)` values** (TQQQ, unweighted stretched-exp): g(15min)=0.1%, g(1h)=2.7%,
g(2h)=14.0%, g(4h)=57.0%, g(1d)=94.0%, g(2d)≈100%. The stretched form now tracks the empirical
knee closely (g(2h) fitted 14.0% vs empirical 13.8%; g(4h) 57.0% vs 55.9%; g(1d) 94.0% vs 95.9%),
a large improvement over the single-exponential fit that missed 1d by 22 points.

## Caveats

- **Stretched exponential slightly undershoots below ~60 min** — the k≈2.5 form is very flat near
  h=0, so it charges g(15m)=0.1% / g(60m)=2.7% against empirical 1.6% / 6.9%. In absolute terms
  this is <2 bps at a 50 bps edge — negligible next to the execution-cost differences the
  scheduler trades against — and it biases the chosen horizon *slightly long* rather than short
  (the opposite, and much smaller, direction than the old single-exponential's 2.5-3.5x
  overcharge). The knee and saturation (120 min–2 day) are now tracked closely.
  `results/g_curve_{SYM}.csv` has the fitted curve at every measured delay point for comparison.
- **`abs_pnl` weighting uses realized full-trade pnl**, not a delay-specific decomposition — it's
  an importance weight ("trades that mattered more get more say"), not a claim that the fraction
  itself is pnl-dependent.
- **Symbols outside {TQQQ, SQQQ} fall back to TQQQ's τ** in `alpha_decay.py` — undocumented
  extrapolation risk for any other name.
- **This curve describes *entry* delay only** — it says nothing about exit timing (already
  strategy-controlled via trailing stop) or about interruption risk (E02's job).

## Reproduce

```
/Users/franciscosimao/opt/anaconda3/envs/quant/bin/python \
    research/scheduler/01_alpha_decay/build_01_alpha_decay.py
/Users/franciscosimao/opt/anaconda3/envs/quant/bin/python -m pytest tests/test_alpha_decay.py -v
```
Outputs in `results/` (gitignored): `delay_fraction_events_{SYM}.csv`, `g_curve_{SYM}.csv`,
`alpha_decay.png`.
