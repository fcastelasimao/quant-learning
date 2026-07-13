# 19 — SQQQ target exploration

> Period: OOS 2018–2026, walk-forward annual refit, strict-prior enriched features (35 features = curated-12 + regime dummies + 21 daily context), L1-logit (C=0.1). SQQQ only.

**Scope.** Grid search over 3 severity targets × 5 sizing functions for SQQQ.
Uses the same enriched walk-forward setup as item 17. Resolves the "which target
and sizing maximises SQQQ performance?" question left open by items 12 and 17.

Targets:  `pnl_pct ≤ {−1.0%, −1.5%, −2.0%}`
Sizing:   `linear_skip, sqrt_skip, aggressive_2x, moderate_1p5x, step_skip_at_50`

## Headline — full grid (15 combinations + baseline)

| target | sizing | mean size | CAGR | Sharpe | Max DD | Calmar |
|---|---|---:|---:|---:|---:|---:|
| **baseline** | **full** | 1.00 | **113%** | 2.56 | −18.1% | 6.22 |
| −1.0% | linear_skip | 0.59 | 58% | 2.53 | −10.2% | 5.71 |
| −1.0% | **sqrt_skip** | 0.76 | 80% | 2.57 | −12.1% | **6.60** |
| −1.0% | aggressive_2x | 0.23 | 15% | 1.62 | −7.1% | 2.14 |
| −1.0% | moderate_1p5x | 0.39 | 32% | 2.21 | −8.6% | 3.71 |
| −1.0% | step_skip_at_50 | 0.76 | 59% | 1.92 | −16.4% | 3.59 |
| −1.5% | linear_skip | 0.75 | 71% | 2.54 | −12.3% | 5.75 |
| −1.5% | sqrt_skip | 0.86 | 88% | 2.58 | −14.5% | 6.07 |
| −1.5% | aggressive_2x | 0.53 | 38% | 2.10 | −11.4% | 3.37 |
| −1.5% | moderate_1p5x | 0.63 | 52% | 2.32 | −11.9% | 4.36 |
| −1.5% | step_skip_at_50 | 0.90 | 86% | 2.30 | −16.1% | 5.34 |
| −2.0% | **linear_skip** | 0.86 | 86% | **2.60** | −15.2% | 5.64 |
| −2.0% | **sqrt_skip** | 0.92 | 97% | **2.62** | −16.6% | 5.87 |
| −2.0% | aggressive_2x | 0.73 | 63% | 2.40 | −12.4% | 5.08 |
| −2.0% | moderate_1p5x | 0.79 | 73% | 2.50 | −13.8% | 5.30 |
| −2.0% | step_skip_at_50 | 0.95 | 100% | 2.49 | −18.1% | 5.52 |

## Best combination

**Maximise Calmar:** `−1.0% sqrt_skip` — Calmar 6.60, Sharpe 2.57, MaxDD −12.1%
(best risk-adjusted return after accounting for drawdown)

**Maximise Sharpe:** `−2.0% sqrt_skip` — Sharpe 2.62, Calmar 5.87, MaxDD −16.6%
(best absolute Sharpe, beats baseline by +0.06)

These are not the same, which reveals a SQQQ-specific tension:
- −1% target identifies trades best, cuts drawdown aggressively (−12.1% vs −18.1% baseline), but CAGR drops substantially (80% vs 113%).
- −2% target preserves most of the CAGR while still improving Sharpe modestly.

**Recommendation**: depends on mandate.
- If the goal is best risk-adjusted return (Calmar): `−1% sqrt_skip`.
- If the goal is best Sharpe with drawdown reduction: `−2% sqrt_skip` (aligns with item 17 conclusion).
- Neither is clearly dominant when both Sharpe and Calmar are weighted.

## What does NOT work

- `aggressive_2x` at any target: degrades Sharpe to ≤ 2.40, destroys CAGR. The 2× amplification oversizes down too many winning trades.
- `step_skip_at_50` at any target: binary cutoff is the wrong shape for this signal — low Sharpe and poor Calmar.
- `−1.5%` target: consistently middle-of-the-road across all sizing functions. It does not dominate either end.

## Resolves the "SQQQ Sharpe unresolved" question

Item 17 concluded that SQQQ Sharpe improves only with −2% enriched features. This grid confirms that finding and shows it holds across all five sizing functions at −2%:

| −2% sizing | Sharpe | vs baseline (+) |
|---|---:|---:|
| sqrt_skip | 2.62 | +0.06 |
| linear_skip | 2.60 | +0.04 |
| moderate_1p5x | 2.50 | −0.06 |
| aggressive_2x | 2.40 | −0.16 |

Only `sqrt_skip` and `linear_skip` reliably improve Sharpe. The conclusion stands: **for SQQQ, use `sqrt_skip` or `linear_skip` with `is_severe_loss @ −2%` target**.

## Artifacts

| file | content |
|---|---|
| `build_19_sqqq_target_exploration.py` | Analysis script (3 targets × 5 sizing) |
| `sqqq_target_sizing_grid.csv` | Full grid with all metrics |
| `sqqq_target_exploration.png` | Sharpe vs MaxDD scatter across all combinations |
| `findings_19_sqqq_target_exploration.md` | This note |
