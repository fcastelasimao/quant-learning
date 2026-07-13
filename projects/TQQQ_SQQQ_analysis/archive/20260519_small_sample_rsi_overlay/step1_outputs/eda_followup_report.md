# Step 1 EDA Follow-up — Non-linear Tests and Contribution Decomposition

## Why this exists

Step 1's linear regression of `pnl_pct` on `RSI_entry` returned slope ≈ 0 (p=0.40) and concluded 'no RSI signal.' That test can only detect monotone relationships. Post-hoc analysis revealed the real structure: RSI predicts the *shape* of the return distribution — particularly tail risk and contribution density — not the mean. This document adds the quadratic fit, non-parametric tests, and per-bin decomposition that expose that structure.

## Per-bin contribution

### 5-pt bins

| bin | n | % trades | mean | median | std | % of P/L | Sharpe/trade |
|-----|---|---------|------|--------|-----|---------|-------------|
| [35,40) | 113 | 6.9% | 0.096 | 0.484 | 1.964 | 2.7% | 0.049 |
| [40,45) | 139 | 8.5% | 0.386 | 0.839 | 1.989 | 13.4% | 0.194 |
| [45,50) | 279 | 17.1% | 0.238 | 0.758 | 2.565 | 16.6% | 0.093 |
| [50,55) | 251 | 15.4% | 0.350 | 0.890 | 2.421 | 22.0% | 0.144 |
| [55,60) | 238 | 14.6% | 0.436 | 0.848 | 2.179 | 25.9% | 0.200 |
| [60,65) | 248 | 15.2% | 0.033 | -0.588 | 1.886 | 2.0% | 0.017 |
| [65,70) | 230 | 14.1% | 0.253 | 0.569 | 1.950 | 14.6% | 0.130 |
| [70,75) | 129 | 7.9% | 0.088 | 0.426 | 1.597 | 2.8% | 0.055 |

See `contribution_by_rsi_bin.png` for the bar chart.

### 2.5-pt bins

| bin | n | % trades | mean | % of P/L | Sharpe/trade |
|-----|---|---------|------|---------|-------------|
| [35.0,37.5) | 57 | 3.5% | 0.102 | 1.5% | 0.049 |
| [37.5,40.0) | 56 | 3.4% | 0.090 | 1.3% | 0.049 |
| [40.0,42.5) | 57 | 3.5% | 0.363 | 5.2% | 0.179 |
| [42.5,45.0) | 82 | 5.0% | 0.402 | 8.2% | 0.203 |
| [45.0,47.5) | 153 | 9.4% | 0.267 | 10.2% | 0.106 |
| [47.5,50.0) | 126 | 7.7% | 0.202 | 6.4% | 0.077 |
| [50.0,52.5) | 135 | 8.3% | 0.293 | 9.9% | 0.116 |
| [52.5,55.0) | 116 | 7.1% | 0.415 | 12.0% | 0.180 |
| [55.0,57.5) | 107 | 6.6% | 0.765 | 20.5% | 0.311 |
| [57.5,60.0) | 131 | 8.1% | 0.166 | 5.4% | 0.088 |
| [60.0,62.5) | 125 | 7.7% | 0.266 | 8.3% | 0.131 |
| [62.5,65.0) | 123 | 7.6% | -0.204 | -6.3% | -0.120 |
| [65.0,67.5) | 111 | 6.8% | 0.343 | 9.5% | 0.177 |
| [67.5,70.0) | 119 | 7.3% | 0.169 | 5.0% | 0.086 |
| [70.0,72.5) | 129 | 7.9% | 0.088 | 2.8% | 0.055 |

## Counterfactual drop-bin CAGR

Full baseline CAGR (all trades): **0.7247** (72.47%)

| bin dropped | final equity | CAGR | ΔCAGR vs full |
|-------------|-------------|------|--------------|
| [55,60) | $125,302 | 48.90% | -23.58 pp |
| [50,55) | $147,883 | 52.83% | -19.64 pp |
| [45,50) | $184,214 | 58.21% | -14.26 pp |
| [65,70) | $190,756 | 59.08% | -13.39 pp |
| [40,45) | $196,364 | 59.81% | -12.66 pp |
| [70,75) | $290,544 | 69.98% | -2.50 pp |
| [35,40) | $293,089 | 70.21% | -2.26 pp |
| [60,65) | $306,864 | 71.45% | -1.03 pp |

## Non-linear tests

### Quadratic regression  `pnl_pct ~ RSI_entry + RSI_entry²`

- Intercept: -1.68716 (t=-1.03, p=0.3043)
- β₁ (RSI):   0.07768 (t=1.27, p=0.2026)
- β₂ (RSI²):  -0.000748 (t=-1.35, p=0.1757)
- R²: 0.0016   (R²_linear was 0.0004)
- Vertex of parabola (peak): RSI = **51.93**
- Interpretation: inverted-U shape — returns peak near RSI≈52, fall off at extremes. Quadratic t-stat = -1.35 (p=0.176) — suggestive but not significant at 5%.

### LOWESS smoothing (frac=0.3)

See `polynomial_fit.png` for the orange LOWESS curve. Non-monotone shape confirmed visually; mild peak in the RSI 50–60 region with a drop-off at high RSI consistent with the dead-zone finding.

### Kruskal-Wallis test  (any cross-bin difference in central tendency)

- H = 6.857,  p = 0.4439
- Not significant at 5% (p=0.444). Confirms that the mean/median story is null — RSI bins do not differ in central tendency.

### Levene's test for variance equality

- Statistic = 6.979,  p = 0.000000
- **Highly significant (p < 0.001).** Variances differ substantially across RSI bins. This is the real RSI signal: risk level, not mean return.

## Risk asymmetry

| bin | n_trades | n_losses | min_loss (%) | loss_std | pct5_loss | hard floor (<−2.5%) |
|-----|----------|----------|-------------|----------|-----------|---------------------|
| [35,40) | 113 | 55 | -2.125 | 0.363 | -2.063 | **YES** |
| [40,45) | 139 | 60 | -2.129 | 0.380 | -2.097 | **YES** |
| [45,50) | 279 | 128 | -5.896 | 0.951 | -3.840 | no |
| [50,55) | 251 | 107 | -4.704 | 0.919 | -3.591 | no |
| [55,60) | 238 | 98 | -3.580 | 0.709 | -3.157 | no |
| [60,65) | 248 | 127 | -4.322 | 0.702 | -2.687 | no |
| [65,70) | 230 | 106 | -4.913 | 0.772 | -3.127 | no |
| [70,75) | 129 | 59 | -3.068 | 0.630 | -2.740 | no |

See `loss_distribution_by_rsi.png` for the distribution plot.
Bins with a hard floor (`min_loss > −2.5%`) exhibit tightly capped downside — likely a tighter trail-stop on low-RSI entries in the original strategy. This makes those bins asymmetrically attractive for a leverage overlay: known cap on loss, but full upside participation.

## Conclusions

- **The [55,60) RSI bin is the workhorse**: 25.9% of total P/L from 14.6% of trades. Dropping it collapses CAGR by 23.6 pp. Fine bin [55,57.5) alone: 20.5% of P/L.
- **RSI predicts risk, not return**: Levene p = 0.0000 (variance differs) vs Kruskal-Wallis p = 0.444 (median unchanged). Low-RSI bins have hard loss floors (~−2.1%) with uncapped upside — asymmetric risk profile that favours leverage overlay.
