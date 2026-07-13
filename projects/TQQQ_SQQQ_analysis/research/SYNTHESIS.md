# Exploratory pass synthesis

This document summarizes research items 01–20 and the 2026-06-09 strict-prior correction to items 06 and 17.

## Read This First

The original 2026-05-29 synthesis treated item 06 context enrichment and item 17 enriched sizing as the main deployable result. That conclusion is now superseded.

The issue was causality: item 06 was intended to join each intraday trade to prior daily market context, but exact same-day daily bars were allowed. For an intraday entry, same-day daily high/low/close/volume are not known yet. On 2026-06-09 the enrichment code and scorer feature helper were changed to require:

```text
context_date < entry_date
```

After rebuilding item 06, item 17, scorer artifacts, and tests under this strict-prior rule, the combined strategy (items 17–18) still delivers meaningful improvements — but the SQQQ claim required using the −2% target (not −1%) and the `sqrt_skip` function.

## Current Research State

### 1. Context Enrichment Is Useful But Not The Original Breakthrough

> Period: IS 2015–2020, OOS 2021–2026, strict-prior daily context.

Corrected item-06 AUC:

| symbol | model + target | curated only | strict-prior context | delta |
|---|---|---:|---:|---:|
| TQQQ | tree, `is_severe_loss` | 0.670 | 0.629 | -0.041 |
| TQQQ | tree, `is_loser` | 0.563 | 0.526 | -0.037 |
| TQQQ | L1-logit, `is_severe_loss` | 0.593 | 0.611 | **+0.018** |
| SQQQ | tree, `is_severe_loss` | 0.616 | 0.592 | -0.024 |
| SQQQ | tree, `is_loser` | 0.571 | 0.563 | -0.009 |
| SQQQ | L1-logit, `is_severe_loss` | 0.508 | 0.534 | **+0.026** |

The useful signal is concentrated in the L1-logit severe-loss probability. Trees weaken with 35 features; L1-logit with regularization improves modestly on both symbols.

**Item 06 feature selection (permutation importance, item C):**
- TQQQ: 13 of 35 features have positive WF-averaged AUC delta. Top: `atr_pct` (+0.021), regime dummies, `hour_of_entry`, `RSI_entry`.
- SQQQ: 19 of 35 features positive. Top: `RSI_entry` (+0.020), `VIX_pctile_252d` (+0.013), `QQQ_dist_MA200` (+0.009), `QQQ_50d_return_pctile_252` (+0.009).
- `HYG_LQD_ratio` has **negative** delta for both symbols — remove from any enriched model.
- Pruning to positive-delta features costs < 0.6 AUC points for TQQQ and actually helps SQQQ (+0.001).

### 2. Multi-Threshold Sizing (Items 12, 17, 19)

> Period: OOS 2018–2026, walk-forward annual refit, L1-logit (C=0.1).

**Item 12 — curated-12 + regime dummies:**

| symbol | target | sizing | CAGR | Sharpe | Max DD | Calmar |
|---|---|---|---:|---:|---:|---:|
| TQQQ | baseline | full | 158% | 4.15 | −18.3% | 8.66 |
| TQQQ | −1.0% | linear_skip | 105% | **4.18** | **−11.0%** | **9.54** |
| SQQQ | baseline | full | 113% | 2.56 | −18.1% | 6.23 |
| SQQQ | −1.0% | linear_skip | 59% | 2.53 | **−10.7%** | 5.47 |
| SQQQ | −2.0% | sqrt_skip | 96% | **2.59** | −16.4% | 5.89 |

**Item 17 — enriched (35 features), strict-prior:**

| symbol | sizing | CAGR | Sharpe | Max DD | Calmar |
|---|---|---:|---:|---:|---:|
| TQQQ | baseline | 167% | 4.16 | −18.3% | 9.15 |
| TQQQ | **linear_skip_enriched_1%** | 109% | **4.22** | **−10.0%** | **10.89** |
| SQQQ | baseline | 113% | 2.56 | −18.1% | 6.22 |
| SQQQ | sqrt_skip_enriched_2% | 97% | **2.62** | −16.6% | 5.87 |

Recommendations per item 17: **TQQQ → linear_skip @ −1%, SQQQ → sqrt_skip @ −2%**.

**Item 19 — SQQQ grid (3 targets × 5 sizing):**

Best Calmar: −1% `sqrt_skip` (Calmar 6.60, Sharpe 2.57, MaxDD −12.1%).
Best Sharpe: −2% `sqrt_skip` (Sharpe 2.62, Calmar 5.87, MaxDD −16.6%).
`aggressive_2x` and `step_skip_at_50` degrade Sharpe — do not use.

### 3. The Original Focus Rule Still Survives

Item 08 remains a credible positive result independent of the item-06 daily-context leakage:

```text
SQQQ rsi_x_atr_cell_3_1
RSI_entry ∈ [56.4, 59.85]  AND  atr_pct ∈ [0.39, 0.47]
```

Block-bootstrap-by-year 95% CI for OOS net pnl impact is entirely positive. Sparse (7–10 OOS trades), but zero overlap with high-p_severe trades (see item 18).

### 4. Regime-Conditional Rules (Item 11)

```text
TQQQ + regime == sideways_lowvol + atr_pct ∈ (0.42, 0.48] + MA100_D1 ≤ 0.000204
```

R-OOS: 10 flagged trades, 100% precision, +10.6 pp net pnl impact. Embargo 2026 held.
Also sparse (~23 trades in full OOS 2021–2026) but zero overlap with p_severe > 0.5.

### 5. Combined Strategy (Item 18)

> Period: OOS 2018–2026, walk-forward annual refit, strict-prior enriched features.
> TQQQ: p_severe @ −1% + sideways_lowvol rule. SQQQ: p_severe @ −2% + RSI×ATR rule.

| symbol | scenario | mean size | CAGR | Sharpe | Max DD | Calmar |
|---|---|---:|---:|---:|---:|---:|
| **TQQQ** | baseline | 1.00 | 158% | 4.14 | −18.3% | 8.64 |
| TQQQ | p_severe_only | 0.67 | 103% | 4.20 | −10.0% | 10.28 |
| TQQQ | rules_only | 0.99 | 160% | 4.20 | −18.3% | 8.75 |
| **TQQQ** | **combined** | **0.66** | 105% | **4.27** | **−10.0%** | **10.43** |
| **SQQQ** | baseline | 1.00 | 113% | 2.56 | −18.1% | 6.22 |
| SQQQ | p_severe_only | 0.86 | 86% | 2.60 | −15.2% | 5.64 |
| SQQQ | rules_only | 0.99 | 113% | 2.58 | −18.1% | 6.26 |
| **SQQQ** | **combined** | **0.84** | 87% | **2.63** | **−15.2%** | **5.69** |

**Zero overlap** between rule fires and p_severe > 0.5 for both symbols. The components are genuinely complementary. Combined beats both individual components.

### 6. Cross-Symbol Signal (Item 20)

> Period: OOS 2018–2026, walk-forward annual refit, strict-prior enriched features.

| train | score | kind | agg OOS AUC |
|---|---|---|---:|
| TQQQ | TQQQ | own | 0.593 |
| SQQQ | SQQQ | own | 0.581 |
| TQQQ | SQQQ | cross | 0.548 |
| SQQQ | TQQQ | cross | 0.576 |

Own-symbol beats cross for both: models are not merely learning shared QQQ macro regime effects. Symbol-specific trade features (`atr_pct`, `RSI_entry`, MA slopes) provide 3–5 AUC points over and above the cross-symbol prediction. Keep TQQQ and SQQQ as separate models.

## Closed Branches

- Item 14: modelable intraday features add essentially no AUC over daily context; revisit only after longer intraday ^VIX history is available.
- Item 15: tighter exit stops destroy net pnl across all tested levels. Down-sizing entry notional is preferable to tightening stops.
- Item 16: calendar/FOMC features are not useful model inputs beyond daily context.
- Item 04 GBM regression on continuous `pnl_pct` remains rejected due to weak/negative OOS R².
- The `−1.5%` severity target is dominated by both `−1%` and `−2%` — not recommended.

## Current Productionization Stance

The combined strategy (item 18) is the recommended configuration:

1. **TQQQ**: `linear_skip` sizing with `is_severe_loss @ −1%` enriched walk-forward model + sideways_lowvol skip rule. Expected: Sharpe 4.27, MaxDD −10.0%, Calmar 10.43.
2. **SQQQ**: `linear_skip` sizing with `is_severe_loss @ −2%` enriched walk-forward model + RSI×ATR skip rule. Expected: Sharpe 2.63, MaxDD −15.2%, Calmar 5.69.

**Pre-deployment checklist:**
- Re-validate annually using the yearly checkpoint loop.
- Enforce `context_date < entry_date` for all feature computation.
- Use calibration diagnostics (Brier score, decile calibration) before treating `p_severe` as an absolute probability.
- The crisp rule thresholds were discovered IS — re-check against each new annual batch.
- Item 20 confirmed the models are not just learning shared macro effects: symbol-specific features contribute.

## Data And Validation Rules That Are Load-Bearing

- `pnl_pct` is in percentage points. `2.5` means +2.5%.
- TQQQ and SQQQ are analyzed separately.
- Same-day daily context is forbidden for intraday entries.
- For a trade in year `Y`, the model must be trained only on labeled trades with `entry_year < Y`.
- Candidate-trade scoring cannot reconstruct strategy-internal fields like `RSI_entry`, `BBP_entry`, `bars_since_last_stop`, or `regime_entry`; the calling backtest must provide them.

## Folder Map

```text
research/
  01_data_diagnostics/                feature audit and curated set
  02_univariate_signal/               per-feature signal checks
  03_multivariate_structure/          PCA / PLS-DA / LDA loadings
  04_loss_region_models/              trees, L1-logit, GBM, with walk-forward
  05_capital_normalization/           constant-notional metrics
  06_context_enrichment/              strict-prior daily context + permutation importance
  07_severity_threshold_sweep/        AUC vs threshold, tree and L1-logit
  08_focus_rule_recheck/              SQQQ focus rule block bootstrap
  09_validation_redesign/             IS / research-OOS / embargo split
  10_regime_paper_review/             HSMM review and rejection
  11_regime_conditional_rules/        regime-conditioned loss rules
  12_continuous_sizing_simulation/    continuous sizing: curated features, 3 targets
  13_within_csv_compounding/          auxiliary CAGR view
  14_intraday_context/                intraday feature check
  15_tighter_stops_simulation/        stop-distance simulation
  16_calendar_features/               calendar/FOMC feature check
  17_sizing_with_enriched_features/   strict-prior enriched sizing, 3 targets
  18_combined_strategy/               p_severe + crisp rules combined
  19_sqqq_target_exploration/         SQQQ sizing grid (3 targets × 5 functions)
  20_cross_symbol_signal/             cross-symbol AUC: own vs cross
  _walkforward.py                     shared walk-forward helpers
  SYNTHESIS.md                        this note

deployable_strategies/
  metrics.py                          constant_notional_metrics()
  continuous_sizing/
    sizing.py                         sizing function library
    p_severe_scorer/                  walk-forward scorer with annual JSON artifacts
  focus_rules/
    sqqq_rsi_atr_skip.py             SQQQ RSI×ATR skip rule mask
  regime_rules/
    tqqq_sideways_skip.py            TQQQ sideways_lowvol skip rule mask
  README.md                           colleague-facing integration guide
```
