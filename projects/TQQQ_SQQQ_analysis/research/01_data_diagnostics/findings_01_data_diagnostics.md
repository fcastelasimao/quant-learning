# 01 — Data diagnostics and multicollinearity audit

> Period: Regime-labeled subset 2015–2026.

**Scope.** Rows with `regime_entry` non-null. TQQQ 1,782 of 2,343 (76.1 %), SQQQ 1,511 of 1,930 (78.3 %). Run script: `build.py`.

## What survived after filtering

| symbol | bull | chop_highvol | sideways_lowvol | total |
|---|---:|---:|---:|---:|
| TQQQ | 329 | 763 | 690 | **1,782** |
| SQQQ | 386 | 597 | 528 | **1,511** |

`sample_sizes.csv` adds outcome columns. Two things worth noting:

1. **Regime *does* differentiate outcomes for TQQQ.** Bull regime has loser rate 25 % and mean +2.46 pp; chop_highvol and sideways_lowvol are both around loser rate 47 % and mean +0.5–0.8 pp. This is meaningful asymmetry and is a real candidate signal for item 4.
2. **For SQQQ the loser rate is flat across regimes (50–55 %).** Mean pnl varies (bull +1.87 vs sideways +0.51) but the *classification* of a trade as loser doesn't shift much by regime. SQQQ's edge is in the magnitude of winners, not in fewer losers.

## IS / OOS window under the regime filter

`year_sizes.csv`. **Earliest regime-labeled year is 2015 for both symbols** — the 2013-2014 trades are precisely the unlabeled rows we just dropped. The hardcoded `IS_END = 2020-12-31` from `archive/full_history_feature_scan.py` still applies, but the effective IS window for this analysis is **2015–2020 (6 years)** and OOS is **2021–2026 (5.4 years, 2026 partial)**.

| | TQQQ IS | TQQQ OOS | SQQQ IS | SQQQ OOS |
|---|---:|---:|---:|---:|
| bull | 181 | 148 | 208 | 178 |
| chop_highvol | 374 | 389 | 282 | 315 |
| sideways_lowvol | 372 | 318 | 293 | 235 |

`regime_x_year.csv` has the full breakdown. Two warnings for walk-forward design:

- **TQQQ 2022 has 23 trades total**, only 19 in chop_highvol and 4 in bull. SQQQ 2017 bull has 4. Several other regime-year cells are <10. Expanding-window WF stepping yearly will produce some evaluation windows where a regime-conditioned metric is essentially unmeasurable. We should report metrics only when `n ≥ 20` per regime-year and aggregate the thin years.
- **Some regime-year cells are empty** (TQQQ 2020 sideways_lowvol, SQQQ 2020 sideways_lowvol). The regime classifier never labeled those bars.

## Missingness

`missingness.csv`. Only the MA-slope features (`MA20_D1`, `MA50_D3`, …) have any missing values, and at most 13 of 1,511 rows (0.9 %). Drop these rows in models that consume those columns; no imputation needed.

## Multicollinearity is severe

`high_corr_pairs.csv` lists every Spearman |ρ| ≥ 0.85. 30 pairs for TQQQ, 28 for SQQQ. Headline patterns:

- **Raw price-level features are all colinear.** `MA20`, `MA50`, `MA100`, `atr`, `high_water_mark_entry` all sit at |ρ| > 0.99 with each other for TQQQ. These features encode "the underlying went up since 2015" and carry essentially zero information about a specific trade. They must not enter any model.
- **Rolling-percentile features add almost nothing.** `RSI_entry` vs `RSI_entry_roll_pctile_252` = 0.996. Same for BBP, atr_pct, volume_ratio (all > 0.93). The 252-trade rolling normalization didn't decorrelate from the raw version because the raw distributions of these indicators are already roughly stationary on this dataset. Keep the raw version per pair, drop the percentile.
- **MA slopes are inter-collinear within an MA.** `MA20_D1 / D3 / D5` all > 0.92. `MA50_D1 / D3 / D5` all > 0.96. Keep one per MA — preferably the longest lookback (`MA20_D5`, `MA50_D5`) since shorter ones are noisier.
- **`log_volume_ratio` and `volume_ratio` are mechanically identical** by Spearman (rank-preserving log transform). `log_volume_ratio` is preferred for any model that assumes approximately Gaussian features.
- **TQQQ has a soft entanglement**: `BBP_entry ↔ dist_to_MA20` (ρ = 0.94) and `RSI_entry ↔ dist_to_MA50` (ρ = 0.88). These are mechanical (BBP encodes position relative to a moving average; RSI tracks normalized momentum that correlates with trend distance). Keep both for now since the entanglement isn't perfect and they may carry different shape in tails — revisit in item 2.

## Decorrelated feature set proposal

The script-generated `decorrelated_feature_set.csv` picked representatives by `|Spearman(feature, pnl_pct)|`. That heuristic picked some **raw price-level features as cluster reps** (TQQQ cluster 1 → `MA50`), which we know are spurious. Override the auto-pick with this interpretability-first set (13 numeric + 1 categorical):

| feature | role |
|---|---|
| `atr_pct` | normalized volatility |
| `RSI_entry` | momentum |
| `BBP_entry` | Bollinger Band position |
| `dist_to_MA20` | short-horizon trend |
| `dist_to_MA50` | medium-horizon trend |
| `dist_to_MA100` | long-horizon trend |
| `MA20_D5` | short-horizon slope |
| `MA50_D5` | medium-horizon slope |
| `MA100_D1` | long-horizon slope (only lookback available) |
| `log_volume_ratio` | volume |
| `bars_since_last_stop` | recovery time since prior stop |
| `hour_of_entry` | session clock |
| `is_bullish_c1c2` | last-2-candle direction state |
| `regime_entry` | categorical (treat separately in models) |

Dropped on grounds of multicollinearity or zero-information level: `MA20`, `MA50`, `MA100`, `atr`, `high_water_mark_entry`, `volume_cur`, `volume_ratio` (use log), `MA20_D1`, `MA20_D3`, `MA50_D1`, `MA50_D3`, `RSI_entry_roll_pctile_252`, `BBP_entry_roll_pctile_252`, `atr_pct_roll_pctile_252`, `volume_ratio_roll_pctile_252`. That's **15 features dropped**, 13 kept, no information loss except in the percentile rankings (whose information overlap with raw versions is > 93 %).

## Sanity

- All regime-labeled rows have regime in `{bull, chop_highvol, sideways_lowvol}`. No nulls slipped through.
- `pnl_pct` median is 0.74 (TQQQ) / −0.78 (SQQQ) — single-digit percentage points, not double-scaled.
- Earliest year confirmed 2015 across both symbols.

## What to use this for next

- **Item 02 (univariate signal):** run on the full 28-feature list (keep redundancy visible there — we want a complete ranking) but flag the cluster representative for each feature.
- **Items 03–04 (multivariate, models):** use the 13-feature curated set above. Multicollinearity wrecks linear / PLS / logistic; trees don't care but interpretation is cleaner with the smaller set.
- **Walk-forward design:** require `n ≥ 20` per regime-year cell when reporting regime-conditioned metrics. Plan a fallback aggregation for 2017 SQQQ bull / 2022 TQQQ / similar thin cells.

## How to read the plots

**`corr_heatmap_<sym>.png`** — 28×28 Spearman correlation matrix, features reordered by hierarchical clustering so similar features sit next to each other.

- **Color scale**: red = positive ρ, blue = negative ρ, white = ~0. Saturation = magnitude. The diagonal is always 1.0 (a feature with itself).
- **What to look for**: **dark-red blocks along the diagonal** = clusters of mutually redundant features. The big dark-red block containing `MA20/MA50/MA100/atr/high_water_mark_entry` is the price-level cluster (ρ > 0.99). Smaller blocks are the MA-slope family, the volume family, the rolling-percentile pairs.
- **Why it matters**: anything inside a dark block carries the same information. Picking one representative per block is the multicollinearity collapse from 28 → 13 features.
- **Off-diagonal blue patches** = features that anti-correlate (e.g. SQQQ `volume_cur` vs MA levels at ρ ≈ −0.94). Anti-correlation is just correlation with sign flipped — same redundancy story.

## Artifacts in this folder

| file | purpose |
|---|---|
| `build.py` | the analysis script |
| `sample_sizes.csv` | per (symbol, regime) sample size + baseline outcomes |
| `year_sizes.csv` | per (symbol, year) sample size + baseline outcomes |
| `regime_x_year.csv` | full (symbol, year, regime) breakdown |
| `missingness.csv` | per-feature missingness on the filtered subset |
| `corr_pearson_<sym>.csv` / `corr_spearman_<sym>.csv` | 28×28 correlation matrices |
| `high_corr_pairs.csv` | all pairs with |Spearman| ≥ 0.85 |
| `feature_clusters.csv` | hierarchical-clustering output (long form) |
| `decorrelated_feature_set.csv` | one row per cluster with auto-picked representative |
| `findings.md` | this note |
