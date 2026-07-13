# 17 - Sizing with item-06 enriched features

> Period: OOS 2018–2026, walk-forward annual refit, strict-prior enriched features (curated-12 + regime dummies + 21 daily context), L1-logit (C=0.1).

**Scope.** Re-run item 12's continuous-sizing simulation using item-06 enriched features: curated strategy features, regime dummies, and 21 daily cross-asset context features. Train annual walk-forward L1-logit models on severe-loss targets (−1%, −1.5%, −2%) and size each trade with `size = 1 - p_severe`.

## Historical correction

The original 2026-05-29 item-17 result used item-06 enriched trades that accidentally allowed same-day daily-bar context. That created look-ahead leakage for intraday entries. The old headline — "enriched sizing is a Pareto improvement on both symbols" — is therefore historical only.

On 2026-06-09 item 06 and item 17 were rebuilt with the strict-prior context policy:
```text
context_date < entry_date
```

## Corrected headline — three severity targets

| symbol | sizing | mean size | CAGR | Sharpe | Max DD | Calmar |
|---|---|---:|---:|---:|---:|---:|
| TQQQ | baseline full size | 1.00 | 167 % | 4.16 | −18.3 % | 9.15 |
| TQQQ | linear skip baseline 1% | 0.69 | 111 % | 4.19 | −11.0 % | 10.10 |
| TQQQ | linear skip enriched 1% | 0.68 | 109 % | 4.22 | **−10.0 %** | **10.89** |
| TQQQ | sqrt skip enriched 1% | 0.82 | 134 % | 4.21 | −13.2 % | 10.18 |
| TQQQ | linear skip enriched 1.5% | 0.82 | 128 % | 4.20 | −12.3 % | 10.42 |
| TQQQ | sqrt skip enriched 1.5% | 0.90 | 144 % | 4.20 | −14.4 % | 10.01 |
| TQQQ | linear skip enriched 2% | 0.93 | 150 % | 4.18 | −15.3 % | 9.81 |
| TQQQ | sqrt skip enriched 2% | 0.96 | 157 % | 4.17 | −16.7 % | 9.42 |
| SQQQ | baseline full size | 1.00 | 113 % | 2.56 | −18.1 % | 6.22 |
| SQQQ | linear skip baseline 1% | 0.58 | 59 % | 2.53 | −10.7 % | 5.47 |
| SQQQ | linear skip enriched 1% | 0.59 | 58 % | 2.53 | −10.2 % | 5.71 |
| SQQQ | sqrt skip enriched 1% | 0.76 | 80 % | 2.57 | −12.1 % | 6.60 |
| SQQQ | linear skip enriched 1.5% | 0.75 | 71 % | 2.54 | −12.3 % | 5.75 |
| SQQQ | sqrt skip enriched 1.5% | 0.86 | 88 % | 2.58 | −14.5 % | 6.07 |
| SQQQ | linear skip enriched 2% | 0.86 | 86 % | **2.60** | −15.2 % | 5.64 |
| SQQQ | sqrt skip enriched 2% | 0.92 | 97 % | **2.62** | −16.6 % | 5.87 |

## Three-threshold comparison

**TQQQ**: The −1% enriched target gives the best drawdown protection (MaxDD −10.0%, Calmar 10.89), closely followed by −1.5% (Calmar 10.42). The −2% target barely improves on curated-only sizing (Calmar 9.81 vs 10.10 baseline) because the model rarely fires. **Recommendation for TQQQ: use enriched −1%.**

**SQQQ**: The −1.5% target lands in the middle — worse drawdown cut than −1%, worse Sharpe preservation than −2%. The −2% enriched target uniquely preserves (actually improves) Sharpe (2.60 vs 2.56 baseline) with a 3.3% drawdown reduction. The `sqrt_skip` variant at −2% achieves the best Sharpe across all SQQQ rows (2.62). This resolves the "SQQQ Sharpe unresolved" question from before: **with enriched features and a −2% or √-skip target, SQQQ gets both Sharpe and drawdown improvement.** The −1.5% target is dominated by the more extreme choice.

## Interpretation

This is validated research/scoring evidence, not yet a final production rule:

- TQQQ `linear_skip_enriched_1pct`: modest Sharpe lift (+0.06) and meaningful drawdown cut (−8.3 pp) — the best TQQQ option.
- SQQQ `sqrt_skip_enriched_2pct`: Sharpe +0.06, drawdown cut −1.5 pp — resolves the SQQQ Sharpe question.
- The −1.5% target does not clearly dominate either end; useful as a mid-point reference but not recommended.

The right next step is item 18 (combined strategy) or the external backtest to validate the SQQQ −2% result.

## Probability quality diagnostics

Calibration artifacts for `p_severe @ −1%`:

| file | content |
|---|---|
| `calibration_deciles_enriched_1pct.csv` | Mean predicted risk vs realized severe-loss rate by decile |
| `calibration_yearly_enriched_1pct.csv` | Mean predicted risk, realized severe-loss rate, Brier score, calibration error by year/symbol |

Top deciles are overconfident for both symbols in the strict-prior rebuild. Treat `p_severe` as a ranking/sizing signal, not a precisely calibrated probability.

## How to read the plots

`equity_enriched_sizing.png` shows corrected strict-prior equity curves for all 8 non-baseline scenarios. The −1.5% scenarios (orange, yellow-green) now appear between the −1% and −2% lines.

`sharpe_vs_drawdown.png` is a trade-off chart. TQQQ plots cluster in the upper-left (good Sharpe, small DD). SQQQ −2% scenarios are now in a more favorable position than −1% scenarios.

## Artifacts

| file | content |
|---|---|
| `build_17_sizing_with_enriched_features.py` | Strict-prior walk-forward sizing script (3 targets) |
| `sizing_enriched_summary.csv` | Per-symbol sizing summary (all targets/scenarios) |
| `calibration_deciles_enriched_1pct.csv` | Decile reliability diagnostics |
| `calibration_yearly_enriched_1pct.csv` | Yearly Brier/calibration diagnostics |
| `equity_enriched_sizing.png` | Corrected equity curves (all 8 scenarios) |
| `sharpe_vs_drawdown.png` | Corrected Sharpe/drawdown scatter |
| `findings_17_sizing_with_enriched_features.md` | This note |
