# 04 — Loss-region models

> Period: IS 2015–2020, OOS 2021–2026, WF 2017–2025. Curated 12 numeric features + 2 regime dummies.

**Scope.** Curated 12 numeric features + 2 regime dummies (`bull` is reference). IS = 2015-2020, OOS = 2021-2026. Three model families, all hyperparameters fixed:

- `DecisionTreeClassifier` depth 4, min_samples_leaf 30 — fit on **`is_loser`** and on **`is_severe_loss`** (two separate trees).
- `LogisticRegression` L1 penalty, C = 0.1, scaled features — fit on both binary targets.
- `GradientBoostingRegressor` depth 3, n_estimators 200, learning_rate 0.05 — fit on **`pnl_pct`** (continuous).

Walk-forward (expanding window, yearly step) runs each model family from train-end-year ∈ {2017…2025} → evaluated on the following year, giving 9 windows per symbol.

## Headlines

1. **`is_severe_loss` is genuinely predictable; `is_loser` and `pnl_pct` are barely.** Walk-forward AUC for `tree_severe`: median **0.65 (TQQQ)** and **0.67 (SQQQ)** across 9 yearly windows. AUC for `tree_loser`: median 0.54 / 0.58 — slightly above chance, much noisier year-to-year. GBM on `pnl_pct` has **negative R² in every single window** for both symbols (R² = −0.06 to −2.6) — the continuous target is unfittable with this much data and signal. **Drop the GBM regressor; use the severity classifier.**

2. **L1-logistic is competitive with the tree** for the binary targets. The linear-projection approach (item 3) was right: most of the predictable structure is approximately linear after feature standardization. The tree's depth-4 interactions add maybe 1–2 percentage points of OOS AUC. This matters for productionization — a logistic model is simpler to monitor and calibrate than a tree.

3. **High-precision loser-rules exist but do not produce positive net PnL when used as skip rules.** Of 28 tree leaves across both symbols and both binary targets, **7 leaves are candidate rules** (IS precision ≥ 0.65, n_IS ≥ 30). On OOS yearly evaluation:

   - All 7 have mean precision **+8 to +30 percentage points above a random filter at the same trigger rate** — real signal.
   - **All 7 have negative total net PnL impact over the OOS window** (range −1.2 to −158 pp). The asymmetric pnl distribution (a few big winners coexist with many small losers in the same feature region) defeats the skip-rule framing. Same lesson as the existing `FINDINGS.md` rule that survived OOS: the *one* SQQQ rule that survived previously was the rare exception where IS-frozen-region happened to keep capturing only losers OOS. Once we look broadly, the rule is the norm and the survivor is the outlier.

4. **The "least bad" rule is TQQQ tree_severe `atr_pct ∈ (0.42, 0.48] AND MA100_D1 ≤ 0.0002`.** Total net PnL impact −1.24 pp over 5 OOS years, 3 of 5 years positive. Mean OOS precision 65 % (lift +0.30 above baseline severe-loss rate, +0.30 above random). Economically break-even, statistically real. Not actionable on its own but a confirmed loss-region.

5. **Permutation importance OOS confirms the univariate ranking from item 2.** For TQQQ: `atr_pct` and `MA20_D5` dominate the GBM; `atr_pct` dominates the tree. For SQQQ: `dist_to_MA20` and `atr_pct` dominate, with `hour_of_entry` next — exactly the SQQQ-specific feature that item 02 flagged. The huge permutation values for the SQQQ GBM (40+) are a **diagnostic of overfitting**, not of feature importance — the unpermuted model is so wrong on OOS that destroying its inputs improves error.

6. **Feature importance disagrees with rule structure.** The tree on `is_severe_loss` for TQQQ picks `atr_pct` repeatedly as a split (consistent with its high importance). The tree on `is_loser` for SQQQ picks `hour_of_entry` first — but its rule leaves still cost money OOS. **High importance does not imply economically usable rules** — the model can be "right about what predicts loser-vs-winner" while still failing as a skip strategy because the magnitude distribution is asymmetric.

## What this means for productionization

The core finding is uncomfortable: **the strategy's pnl distribution is such that any region you can identify as "more loser-dense than average" is also "more winner-magnitude-dense than average,"** so binary skip rules cannot improve net PnL on average. This is consistent with the entire existing research pipeline's experience (18 of 19 SQQQ candidates failed OOS, TQQQ has no surviving rule).

Three reframings worth trying in future iterations (these are out of scope for the current exploratory pass; flagging for the synthesis):

- **Continuous position sizing** instead of binary skip. Use the tree-on-severe predicted probability to scale position size down rather than to zero. Trades in high-severe-prob regions become smaller; trades in low-severe-prob regions become larger.
- **Tighter stops, not skips.** A 70-% severe-loss region might respond to a smaller stop-loss without sacrificing the winner tail. This needs a per-trade stop-distance simulation, which we don't have data for here.
- **Regime-conditional rule activation.** Some rules helped in 3–4 years and hurt in 2–3. A meta-rule that activates a sub-rule only when a regime context favors it could net positive — but this multiplies the overfitting risk and requires more data.

## How to read the plots

**`tree_<sym>_<tname>.png`** — full visualization of the depth-4 decision tree fit on IS data, for either `is_loser` or `is_severe_loss`. Four files per symbol pair.

- **Each box is a node.** Top line: the split rule (e.g. `atr_pct ≤ 0.476`). If true, follow left child; if false, follow right child.
- **`samples = NN.N%`**: the fraction of training trades passing through this node.
- **`value = [w, l]`**: the proportion of (winner, loser/severe) at this node. A leaf with `value = [0.25, 0.75]` means 75 % of training trades that landed here were positives.
- **`class = ...`**: the majority class at the node.
- **Color**: orange/red intensity ∝ how strongly the leaf predicts the positive class; blue ∝ how strongly it predicts winner.
- **How to use it**: trace from root to a colored leaf. The conjunction of splits along that path IS the rule. Item 04's candidate rules in `tree_leaves_<sym>.csv` correspond one-to-one with these leaves. **A deep orange leaf with high sample % is a candidate loss region**.

**`wf_auc_line_<sym>.png`** — walk-forward OOS AUC across years for all four classifier models.

- **X axis**: test year (model trained on [2015..year−1], evaluated on `year`).
- **Y axis**: OOS AUC. Dashed line at 0.5 = random.
- **Lines**: tree_loser (blue), tree_severe (red), L1-logit_loser (purple), L1-logit_severe (green).
- **What to look for**: (a) which lines stay consistently above 0.5; (b) which year the lines crash (typically 2022 — TQQQ had only 23 trades, so AUC is noisy); (c) whether tree vs L1 are close. **`tree_severe` (red) staying above ~0.6 across most years is the consistent signal across years.**

**`rule_per_year_heatmap_<sym>.png`** — for each candidate rule (rows) and each OOS year (columns), the net pnl impact of applying that rule as a skip rule that year.

- **Color scale**: green = skipping helped (rule flagged net-negative trades); red = skipping hurt (rule flagged net-positive trades); centered at 0.
- **What to look for**: rules that are **mostly green across years** = robust. Rules that flip color year-to-year = unstable. In our pass-1 most rules have red somewhere — that's why no rule survived unconditional. Item 11 uses regime-conditional gating to fix some of these.

**`calibration_logit_severe_<sym>.png`** — calibration curve for L1-logistic on `is_severe_loss`. Crucial for using the model as a sizing input.

- **X axis**: predicted probability (binned into deciles). **Y axis**: actual fraction of severe losses in that bin.
- **Dashed diagonal**: perfect calibration — "p̂ = 0.3" means 30 % of those trades really were severe.
- **Red line with `n=NN` annotations**: the model's actual calibration. Points BELOW the diagonal = overconfident in predicting severe; ABOVE = underconfident.
- **Why it matters for sizing**: in item 12 we size by `1 − p_severe`. If the model says `p_severe = 0.5` but the real rate is 0.2, we'll under-size profitable trades. A calibration curve close to the diagonal is the condition for the sizing rule to be unbiased.

## Artifacts

| file | content |
|---|---|
| `build.py` | analysis script |
| `headline_model_perf_<sym>.csv` | IS-fit OOS AUC / R² for each model family |
| `logistic_coefs_<sym>.csv` | L1 coefficients for both binary targets, ranked |
| `feature_importances_<sym>.csv` | GBM impurity + 3 permutation-importance columns (GBM, tree-loser, tree-severe), OOS |
| `tree_leaves_<sym>.csv` | every leaf of both trees with IS precision, OOS precision, lift, net-pnl impact, candidate flag, path |
| `wf_rule_eval_<sym>.csv` | per OOS year evaluation of every candidate rule + random-baseline precision |
| `wf_model_eval_<sym>.csv` | per WF window AUC / R² for all three model families |
| `findings.md` | this note |
