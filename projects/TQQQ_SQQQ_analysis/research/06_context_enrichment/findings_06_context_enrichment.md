# 06 - Context enrichment

> Period: IS 2015–2020, OOS 2021–2026, strict-prior daily context (context_date < entry_date).

**Scope.** For each regime-labeled trade, join pre-entry daily market context from QQQ / SPY / ^VIX / ^VIX3M / HYG / LQD / ^TNX / ^IRX, then re-fit the depth-4 tree and L1-logit models on `is_severe_loss` and `is_loser`.

## Historical correction

The original 2026-05-29 item-06 run was documented as "prior business day" context, but the implementation used `merge_asof(..., allow_exact_matches=True)`. For intraday trades this allowed the same calendar day's completed daily bar to enter the features. That is look-ahead leakage because the high/low/close/volume of the entry date is not known at entry time.

On 2026-06-09 item 06 was rebuilt with strict-prior joining:

```text
context_date < entry_date
```

Exact same-day matches are now disallowed in both the research builder and the scorer feature helper. The FMP SQLite database is also opened in immutable read-only mode so the enrichment step cannot mutate source market data.

Validation after the rebuild:

| symbol | strict-prior rows | violations |
|---|---:|---:|
| TQQQ | 100.0% | 0 |
| SQQQ | 100.0% | 0 |

## Corrected headline

Strict-prior context still helps the linear probability model, but the original "main AUC unlock" was overstated. The tree models weaken under the causal join, and the L1-logit lift is modest.

| symbol | model + target | curated_12 only | strict-prior context | delta |
|---|---|---:|---:|---:|
| TQQQ | tree, `is_severe_loss` | 0.670 | 0.629 | -0.041 |
| TQQQ | tree, `is_loser` | 0.563 | 0.526 | -0.037 |
| TQQQ | L1-logit, `is_severe_loss` | 0.593 | 0.611 | +0.018 |
| SQQQ | tree, `is_severe_loss` | 0.616 | 0.592 | -0.024 |
| SQQQ | tree, `is_loser` | 0.571 | 0.563 | -0.009 |
| SQQQ | L1-logit, `is_severe_loss` | 0.508 | 0.534 | +0.026 |

## Interpretation

The deployable insight is narrower than the original note claimed:

- Daily cross-asset context is still directionally useful for the enriched L1-logit severe-loss probability.
- The effect is not large enough, by itself, to justify a production sizing rule without downstream validation.
- Any finding, artifact, or README text that relied on the old same-day AUC values around 0.70 should be treated as historical and superseded.

The most plausible useful features remain volatility, QQQ state, credit stress, and curve context: `VIX_pctile_252d`, `VIX_level`, `QQQ_drawdown_60d`, `QQQ_50d_return_pctile_252`, `HYG_LQD_ratio`, and `yield_curve_slope`.

## Consequence for item 17

Item 17 was rebuilt from these strict-prior enriched trades. The corrected sizing result is no longer a clean two-symbol Pareto improvement: TQQQ still improves modestly, while SQQQ mostly gets drawdown reduction without better Sharpe.

## How to read the plot

`headline_auc_compare.png` compares curated features vs strict-prior daily context. The corrected visual should no longer show large context gains across all model/target pairs. The important bars are the L1-logit severe-loss bars: small gains for both symbols, not the prior large jump.

## Feature selection via permutation importance

Walk-forward permutation importance on the enriched L1-logit (35 features = 12 curated + 2 regime dummies + 21 context). For each year Y, train on [2013..Y-1], score on year Y, shuffle each feature 10 times. Average AUC drop across 11 WF folds (2015–2025). Positive delta = feature helps (AUC drops when shuffled).

**TQQQ — 13 of 35 features have positive mean AUC delta**

| rank | feature | mean AUC Δ | std |
|---:|---|---:|---:|
| 1 | atr_pct | +0.021 | 0.039 |
| 2 | regime_sideways_lowvol | +0.008 | 0.017 |
| 3 | regime_chop_highvol | +0.008 | 0.017 |
| 4 | hour_of_entry | +0.007 | 0.025 |
| 5 | RSI_entry | +0.007 | 0.028 |
| 6 | QQQ_gap_overnight | +0.004 | 0.007 |
| 7 | dist_to_MA20 | +0.003 | 0.021 |
| 8 | bars_since_last_stop | +0.002 | 0.004 |
| 9 | QQQ_realized_vol_20d | +0.001 | 0.035 |
| 10 | QQQ_dist_MA200 | +0.001 | 0.004 |
| 11 | QQQ_50d_return_pctile_252 | +0.000 | 0.002 |
| 12 | HYG_5d_change | +0.000 | 0.001 |
| 13 | TNX_5d_change | +0.000 | 0.002 |
| 14–35 | (22 features, mostly SPY, VIX non-pctile, credit ratio, MA slopes) | ≤ 0 | — |

**SQQQ — 19 of 35 features have positive mean AUC delta**

| rank | feature | mean AUC Δ | std |
|---:|---|---:|---:|
| 1 | RSI_entry | +0.020 | 0.047 |
| 2 | VIX_pctile_252d | +0.013 | 0.017 |
| 3 | QQQ_dist_MA200 | +0.009 | 0.013 |
| 4 | QQQ_50d_return_pctile_252 | +0.009 | 0.006 |
| 5 | QQQ_dist_high_20d | +0.006 | 0.013 |
| 6 | dist_to_MA20 | +0.006 | 0.014 |
| 7 | QQQ_realized_vol_20d | +0.004 | 0.008 |
| 8 | QQQ_50d_return | +0.003 | 0.010 |
| 9 | atr_pct | +0.003 | 0.006 |
| 10 | VIX_level | +0.002 | 0.004 |
| 11 | MA20_D5 | +0.001 | 0.004 |
| 12 | yield_curve_slope | +0.001 | 0.005 |
| 13–19 | (QQQ_drawdown_5d, SPY_RSI_14, hour_of_entry, dist_to_MA50, bars_since_last_stop, regime_sideways_lowvol, BBP_entry) | ≈ 0 | — |
| 20–35 | (16 features, regime_chop_highvol, HYG_LQD_ratio, HYG_5d_change, TNX_5d_change, QQQ_gap_overnight, …) | ≤ 0 | — |

**Pruned vs full AUC**

| symbol | n kept | n total | full AUC | pruned AUC | Δ AUC |
|---|---:|---:|---:|---:|---:|
| TQQQ | 13 | 35 | 0.611 | 0.605 | −0.006 |
| SQQQ | 19 | 35 | 0.534 | 0.535 | **+0.001** |

Pruning costs at most 0.6 AUC points (TQQQ) for a 63% feature reduction, and is essentially free for SQQQ.

**Key patterns:**

- TQQQ: strategy features dominate — `atr_pct` alone contributes 0.021 Δ, roughly 3× the next feature. Only 6 context features survive (vs 7 curated + regime dummies). Enrichment helps at the margin.
- SQQQ: broader context sensitivity — 10 of 19 kept features are context. `VIX_pctile_252d` (market stress regime) is the key macro signal. Regime dummies barely help SQQQ (consistent with item 11 showing regime rules mainly improve TQQQ).
- `QQQ_dist_MA200` and `QQQ_50d_return_pctile_252` are positive for both symbols — the bubble-proxy percentile signal survives the strict-prior rebuild.
- **Correction to the prior interpretation:** `HYG_LQD_ratio` has _negative_ mean AUC delta for both symbols. It should not be listed as a "useful feature" — remove from any future enriched model.

**Pruned feature sets (use for downstream sizing in preference to all 35):**

TQQQ-13: `atr_pct`, `regime_sideways_lowvol`, `regime_chop_highvol`, `hour_of_entry`, `RSI_entry`, `QQQ_gap_overnight`, `dist_to_MA20`, `bars_since_last_stop`, `QQQ_realized_vol_20d`, `QQQ_dist_MA200`, `QQQ_50d_return_pctile_252`, `HYG_5d_change`, `TNX_5d_change`

SQQQ-19: `RSI_entry`, `VIX_pctile_252d`, `QQQ_dist_MA200`, `QQQ_50d_return_pctile_252`, `QQQ_dist_high_20d`, `dist_to_MA20`, `QQQ_realized_vol_20d`, `QQQ_50d_return`, `atr_pct`, `VIX_level`, `MA20_D5`, `yield_curve_slope`, `QQQ_drawdown_5d`, `SPY_RSI_14`, `hour_of_entry`, `dist_to_MA50`, `bars_since_last_stop`, `regime_sideways_lowvol`, `BBP_entry`

## Artifacts

| file | content |
|---|---|
| `build_06_context_enrichment.py` | Strict-prior enrichment, model comparison, and permutation importance script |
| `context_daily_features.csv` | 21-feature daily context series |
| `enriched_trades_<sym>.csv` | Per-trade enriched canonical with `context_date < entry_date` |
| `tree_leaves_enriched_<sym>.csv` | Tree leaves with context flag |
| `headline_auc_compare.csv` | Corrected strict-prior AUC table |
| `headline_auc_compare.png` | Corrected AUC comparison chart |
| `permutation_importance_TQQQ.csv` | WF permutation importance, all 35 features, TQQQ |
| `permutation_importance_SQQQ.csv` | WF permutation importance, all 35 features, SQQQ |
| `pruned_feature_set_TQQQ.csv` | 13 positive-delta features for TQQQ |
| `pruned_feature_set_SQQQ.csv` | 19 positive-delta features for SQQQ |
| `pruned_vs_full_auc.csv` | Full vs pruned model AUC comparison per symbol |
| `findings_06_context_enrichment.md` | This note |
