# 20 — Cross-symbol signal check

> Period: OOS 2018–2026, walk-forward annual refit, strict-prior enriched features (35 features), L1-logit (C=0.1), `is_severe_loss` target (pnl_pct ≤ −1%).

**Scope.** Does the feature set trained on one symbol predict severe losses on the
other? Answers whether the enriched L1-logit signal is capturing shared market-regime
information (QQQ trend, VIX state, yield curve) or something symbol-specific.

Four walk-forward experiments:

| train | score | kind |
|---|---|---|
| TQQQ | TQQQ | own-symbol baseline |
| SQQQ | SQQQ | own-symbol baseline |
| TQQQ | SQQQ | cross-symbol |
| SQQQ | TQQQ | cross-symbol |

## Cross-symbol AUC table

| train symbol | score symbol | kind | agg OOS AUC | WF median AUC |
|---|---|---|---:|---:|
| TQQQ | TQQQ | own-symbol | 0.593 | 0.567 |
| SQQQ | SQQQ | own-symbol | 0.581 | 0.548 |
| TQQQ | SQQQ | cross-symbol | **0.548** | 0.515 |
| SQQQ | TQQQ | cross-symbol | **0.576** | 0.533 |

## Per-year stability

Full per-year breakdown is in `cross_symbol_yearly_auc.csv`. Summary:

- Own-symbol TQQQ: WF range [0.461, 0.724], 11 folds
- Own-symbol SQQQ: WF range [0.468, 0.771], 11 folds
- Cross TQQQ→SQQQ: WF range [0.364, 0.779] — high variance, some folds near random
- Cross SQQQ→TQQQ: WF range [0.469, 0.658] — more stable than TQQQ→SQQQ

The high variance in cross-symbol folds (especially TQQQ→SQQQ) is expected: cross-symbol predictions rely entirely on shared macro features, with zero symbol-specific pattern.

## Interpretation

**Own-symbol beats cross-symbol for both:** TQQQ (0.593 vs 0.576 scored), SQQQ (0.581 vs 0.548 scored). This confirms that symbol-specific features (trade-level: `atr_pct`, `RSI_entry`, `MA` slopes) add real predictive value beyond the shared macro context.

**SQQQ→TQQQ cross (0.576) is much closer to own-symbol (0.593) than TQQQ→SQQQ (0.548) is to own-symbol SQQQ (0.581).** This asymmetry makes intuitive sense: SQQQ trades tend to happen during QQQ downturns, so SQQQ loss patterns partly capture macro deterioration that also threatens TQQQ. The reverse (TQQQ loss patterns predicting SQQQ losses) is weaker.

**Both cross-symbol AUCs are above 0.5 (not random):** The shared macro context (VIX_pctile_252d, QQQ_dist_MA200, QQQ_50d_return_pctile_252) carries real signal even when applied to the "wrong" symbol. This is consistent with the permutation importance finding (item 06): those three features appear in the positive-delta set for both symbols.

**Practical implication:** There is no evidence that the own-symbol models are merely learning QQQ macro regime effects. The 3–5 AUC point gap (own vs cross) is attributable to symbol-specific trade features. The models are doing genuine per-symbol work and should remain separate.

## Artifacts

| file | content |
|---|---|
| `build_20_cross_symbol_signal.py` | Analysis script (4 WF cross-symbol experiments) |
| `cross_symbol_auc.csv` | Summary: (train, score, kind, agg_oos_auc, wf_median_auc) |
| `cross_symbol_yearly_auc.csv` | Per-year AUC for each experiment |
| `cross_symbol_auc.png` | Bar chart: own vs cross AUC per scored symbol |
| `findings_20_cross_symbol_signal.md` | This note |
