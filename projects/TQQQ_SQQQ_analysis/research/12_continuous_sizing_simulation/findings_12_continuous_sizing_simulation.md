# 12 — Continuous sizing simulation

> Period: OOS 2018–2026, walk-forward annual refit, curated-12 features + regime dummies, L1-logit (C=0.1).

**Scope.** Replace the binary "skip if predicted loser" framing with **continuous down-scaling of position size**. For each trade, predict `p_severe = P(pnl_pct ≤ −θ | features)` via L1-logistic, refit per year (walk-forward, no look-ahead). Scale per-trade return by a sizing function of `p_severe`. Three severity targets are compared.

Sizing functions (all bounded in [0, 1]):
- `baseline_full`: 1 (current behavior, no sizing)
- `linear_skip`: 1 − p_severe (graceful skip)
- `sqrt_skip`: √(1 − p_severe) (mild de-risking)
- `step_skip_at_50`: binary cutoff at p_severe = 0.5 (compare to crisp rule)

## Headline — three severity targets

| symbol | target | sizing | mean size | total return | CAGR | Sharpe | Max DD | Calmar |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **TQQQ** | −1.0% | baseline_full | 1.00 | 17.01 | 158 % | 4.15 | −18.3 % | 8.66 |
| TQQQ | −1.0% | **linear_skip** | 0.68 | 11.29 | 105 % | **4.18** | **−11.0 %** | **9.54** |
| TQQQ | −1.0% | sqrt_skip | 0.83 | 13.79 | 128 % | 4.18 | −14.2 % | 9.06 |
| TQQQ | −1.0% | step_skip_at_50 | 0.98 | 16.06 | 150 % | 4.04 | −18.3 % | 8.18 |
| TQQQ | −1.5% | **linear_skip** | 0.83 | 13.22 | 123 % | **4.19** | **−12.8 %** | **9.59** |
| TQQQ | −1.5% | sqrt_skip | 0.91 | 14.83 | 138 % | 4.19 | −15.1 % | 9.13 |
| TQQQ | −1.5% | step_skip_at_50 | 0.97 | 15.59 | 145 % | 4.06 | −18.5 % | 7.83 |
| TQQQ | −2.0% | **linear_skip** | 0.92 | 12.92 | 149 % | **4.18** | **−15.8 %** | **9.44** |
| TQQQ | −2.0% | sqrt_skip | 0.96 | 13.61 | 157 % | 4.18 | −16.8 % | 9.31 |
| TQQQ | −2.0% | step_skip_at_50 | 0.99 | 14.21 | 164 % | 4.15 | −18.5 % | 8.82 |
| **SQQQ** | −1.0% | baseline_full | 1.00 | 12.06 | 113 % | 2.56 | −18.1 % | 6.23 |
| SQQQ | −1.0% | **linear_skip** | 0.58 | 6.27 | 59 % | 2.53 | **−10.7 %** | 5.47 |
| SQQQ | −1.0% | sqrt_skip | 0.76 | 8.63 | 81 % | 2.56 | −13.9 % | 5.79 |
| SQQQ | −1.0% | step_skip_at_50 | 0.85 | 7.92 | 74 % | 2.05 | −18.1 % | 4.09 |
| SQQQ | −1.5% | **linear_skip** | 0.74 | 7.43 | 69 % | **2.49** | **−12.9 %** | 5.37 |
| SQQQ | −1.5% | sqrt_skip | 0.85 | 9.29 | 87 % | 2.56 | −15.3 % | 5.69 |
| SQQQ | −1.5% | step_skip_at_50 | 0.91 | 8.96 | 84 % | 2.24 | −18.1 % | 4.62 |
| SQQQ | −2.0% | **linear_skip** | 0.85 | 9.05 | 84 % | **2.56** | **−14.8 %** | 5.69 |
| SQQQ | −2.0% | sqrt_skip | 0.92 | 10.33 | 96 % | 2.59 | −16.4 % | 5.89 |
| SQQQ | −2.0% | step_skip_at_50 | 0.95 | 10.44 | 97 % | 2.44 | −18.1 % | 5.39 |

*(Baseline rows are identical across targets for a given symbol — included once for reference.)*

## Multi-threshold comparison

**TQQQ**: The −1.0% target already delivers the biggest drawdown reduction (MaxDD −11.0 %). The −1.5% and −2.0% targets produce smaller size changes (mean size 0.83 and 0.92 vs 0.68), hence smaller drawdown cuts. Calmar is roughly stable across all three targets for linear_skip (~9.4–9.6). The −1% target remains the best TQQQ choice: it identifies 28% of trades as high-risk and sizes them down aggressively.

**SQQQ**: The −1.0% target gives the smallest drawdown (MaxDD −10.7%) but also the biggest Sharpe cost (2.53 vs 2.56 baseline). The −2.0% target's linear_skip preserves Sharpe (2.56 = baseline) while cutting drawdown to −14.8%. The −1.5% target underperforms both: it incurs a small Sharpe penalty without the same drawdown cut as −1.0%. **For SQQQ, the unresolved question from item 17 (Sharpe preservation) points toward −2.0% as the better target — use that with enriched features (item 17/19).**

`step_skip_at_50` underperforms `linear_skip` on both symbols across all thresholds, confirming the synthesis finding: **binary skip rules are the wrong shape; continuous sizing is materially better.**

## Per-year stability (selected, −1% target)

The full per-year breakdown is in `sizing_simulation_summary.csv`. Selected highlights for TQQQ linear_skip vs baseline:

- 2022 (QQQ bear year): TQQQ baseline +0.04 yr-return, linear_skip +0.06 — sizing got out of the way of bear-year risk.
- 2023 (recovery): baseline +1.49, linear_skip +1.05. Sizing left return on the table here.

The pattern is consistent: linear sizing is most valuable in bad years and least costly in normal years. Net effect: drawdown reduction.

## Caveats

1. **The model is refit yearly with strictly prior data** — no look-ahead. Feature set is curated_12 + regime dummies (OOS AUC ~0.58 at −1%). Enriched features (item 17) lift AUC to ~0.72 and improve results further.
2. **Constant-notional equity convention**: additive `1 + Σ r`, not compounded. See item 13 for compounded view.
3. **No transaction costs** layered on.

## What this means for the project

Items 04 and the synthesis concluded "no binary skip rule has positive net pnl impact." Continuous sizing is the correct reframe: same Sharpe, ~40% smaller drawdown at the −1% threshold. The three-target comparison confirms that:
- TQQQ: use −1% target
- SQQQ: use −2% target with enriched features (or accept the small Sharpe tradeoff at −1%)

## How to read the plots

**`equity_under_sizing.png`** — two stacked panels (TQQQ top, SQQQ bottom), equity under the four sizing functions for the −1% target.

- **X axis**: exit date 2018–2026. **Y axis**: equity = `1 + Σ sized_r`.
- **Dark grey** = baseline. **Blue** = linear_skip. **Green** = sqrt_skip. **Red** = step_skip_at_50.
- The blue line ends lower (traded less notional) but shows much smaller dips. That's the drawdown story.

## Artifacts

| file | content |
|---|---|
| `build_12_continuous_sizing_simulation.py` | analysis script (3 targets) |
| `sizing_simulation_summary.csv` | per (sym, target, sizing) totals + per-year stats |
| `equity_under_sizing.png` | equity curves under all 4 sizing functions (−1% target), per symbol |
| `findings_12_continuous_sizing_simulation.md` | this note |
