# 03 — Multivariate structure

> Period: IS 2015–2020, regime-labeled subset. Curated 12 numeric features + 2 regime dummies.

**Scope.** Regime-labeled subset, curated 12 numeric features + 2 regime dummies (`bull` is reference). Standardized via `StandardScaler`. Three orthogonal views: PCA (unsupervised), PLS-DA (supervised on `is_loser`), LDA (supervised, single discriminant).

## Headlines

1. **No dominant axis.** PC1 explains 36 % (TQQQ) / 34 % (SQQQ); PC1+PC2 cumulative reaches 55 % / 50 %. Need 7–8 PCs for 90 % variance. The feature space is genuinely high-dimensional — no quick PCA-based dimensionality reduction.
2. **PC1 is a "market trend" axis** — `dist_to_MA50`, `dist_to_MA100`, MA slopes, and `RSI_entry` all load positively. Symbol-agnostic.
3. **PC2 is "extended / volatile entry"** — `atr_pct`, `BBP_entry`, `dist_to_MA20`, `log_volume_ratio` load positively; `hour_of_entry` loads negatively.
4. **PC3 is regime separation** — the two regime dummies dominate. Confirms that the regime label is approximately orthogonal to the trend features (it carries its own structure rather than being reducible to them).
5. **Losers DO separate from winners in PLS / LDA space — but weakly.** Standardized class-mean separation along PLS1: **−0.29 (TQQQ), +0.31 (SQQQ)**. LDA single-discriminant separation: **0.43 (TQQQ), 0.39 (SQQQ)**. For reference, > 1.0 is clean separation; < 0.3 is "barely useful for filtering". We're in the borderline zone. See `projection_TQQQ.png` / `projection_SQQQ.png`.
6. **The two symbols have a different loser-region story.** TQQQ's LDA leads with the **regime dummies** (chop_highvol +0.41, sideways_lowvol +0.39) — non-bull regimes are the loser zone. SQQQ's LDA has near-zero regime coefficients (0.015) and leads with `dist_to_MA100` (+0.40) and `hour_of_entry` (+0.31). **For SQQQ, regime is not the lever; hour-of-day and trend extension are.** Mirrors item 02.

## What PLS1 says (the supervised first axis)

| symbol | top + loadings on PLS1 | top − loadings on PLS1 | interpretation |
|---|---|---|---|
| TQQQ | atr_pct (+0.49) | dist_to_MA100 (−0.49), MA50_D5 (−0.49), MA100_D1 (−0.44), dist_to_MA50 (−0.42), MA20_D5 (−0.40), RSI_entry (−0.31) | losers = **high volatility + weak/negative trend**. Bull market = no loser. |
| SQQQ | hour_of_entry (+0.40) | dist_to_MA20 (−0.48), RSI_entry (−0.43), atr_pct (−0.42), BBP_entry (−0.39), dist_to_MA50 (−0.36) | losers = **late in session + extended (high indicator values)**. Regime does not contribute. |

The sign convention: PLS1 score is positive on winner direction for TQQQ (std_diff < 0 → losers below winners) and on loser direction for SQQQ (std_diff > 0 → losers above winners). The qualitative reading is the one in the table — read the loadings, not the absolute sign.

## What LDA says (single-discriminant rank by |coef|)

TQQQ top 5: `regime_chop_highvol` (+0.41), `regime_sideways_lowvol` (+0.39), `dist_to_MA100` (−0.21), `dist_to_MA20` (−0.16), `RSI_entry` (+0.14).

SQQQ top 5: `dist_to_MA100` (+0.40), `hour_of_entry` (+0.31), `MA50_D5` (−0.16), `MA100_D1` (−0.16), `atr_pct` (−0.14).

Notable: **TQQQ atr_pct LDA coef is only −0.13**, despite atr_pct being the #1 univariate signal for severe losses. LDA's small coefficient does not mean atr_pct is unimportant — it means the **regime dummies already absorb the volatility-vs-trend distinction** because non-bull regimes are exactly the high-volatility regimes. This is a multicollinearity-after-encoding artifact and motivates trying a model that doesn't share variance across colinear inputs (i.e. trees, item 04).

## Implications for item 04

- **Linear projections give 0.3–0.4 std-diff separation.** A linear classifier (L1-logistic) will not give us strong AUC. Expect AUC ≈ 0.60–0.65, in line with the best single-feature.
- **The two symbols need different feature emphasis.** TQQQ models should be allowed to lean on the regime dummies; SQQQ models should lean on `hour_of_entry` and `dist_to_MA*`. A non-linear model (depth-4 tree, GBM) should pick this up automatically, but it's worth confirming in item 04's importance rankings.
- **PCA components are interpretable** and could be used as engineered features (PC1 = trend, PC2 = extension/volatility), but with so few components capturing variance, this is probably not a productive route. Stick to the curated 12 features.

## How to read the plots

**`projection_<sym>.png`** — two side-by-side scatterplots: PCA (left) and PLS-DA (right). Each dot is a trade.

- **Left panel (PCA)**: x = PC1 score, y = PC2 score. PCA is **unsupervised** — the axes are chosen to maximize total feature variance, with no knowledge of `is_loser`. Winners and losers will only separate here if loss-prone trades happen to live in a high-variance direction.
- **Right panel (PLS-DA)**: x = PLS1 score, y = PLS2 score. PLS-DA is **supervised** — the axes are chosen to maximize covariance with `is_loser`. If the data has any linear separability, the PLS1 axis will pick it up.
- **Colors**: **blue** = winner (`pnl_pct ≥ 0`), **red** = loser. Heavy overlap is expected — the AUC of a linear classifier here is in the 0.55-0.65 range.
- **What to look for**:
  - In PLS-DA, look at the **horizontal distance** between the blue and red clouds along PLS1 — that's the supervised separation. We measured it as 0.29–0.31 standardized units in the table above; visually it's a slight blue-on-one-side, red-on-the-other tendency, with massive overlap.
  - In PCA, expect to see no obvious color separation — confirming PC1/PC2 are about feature variance, not loss prediction.
- **Why it matters**: visual confirmation that this dataset is not linearly separable by any simple 2D projection. A nonlinear classifier (tree, GBM) is needed to do better — but item 04 shows even those only get to AUC 0.65, so the structural limit is real and not just a 2D-projection artifact.

## Artifacts

| file | content |
|---|---|
| `build.py` | the analysis script |
| `pca_variance_explained_<sym>.csv` | scree numbers per symbol |
| `pca_loadings_<sym>.csv` | per-feature loadings on PC1…PCk |
| `pls_loadings_<sym>.csv` | per-feature loadings on PLS1, PLS2, PLS3 |
| `lda_coefs_<sym>.csv` | per-feature single-discriminant coefficient |
| `class_separation_summary.csv` | std-diff between winner and loser class means on PLS1/PLS2/LDA |
| `projection_<sym>.png` | 2-panel scatter — PC1 vs PC2 and PLS1 vs PLS2, colored by is_loser |
| `findings.md` | this note |
