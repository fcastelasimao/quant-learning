# Glossary

Cheat sheet for the statistical / ML / finance terms that show up in this
project's `findings.md` files and CSV columns. Entries are intentionally short
(2–4 lines each). Use this when reading; reach for a textbook when designing.

For the meaning of specific *data columns*, see `FEATURE_DICTIONARY.md`. For
load-bearing data semantics (`pnl_pct` scaling, `capital_before` resets, IS/OOS
dates), `FINDINGS.md` § "Data semantics" is authoritative — section 8 below is
just a quick summary.

---

## 1. Statistical association

**Pearson correlation (ρ_P)** — measures **linear** association between two
continuous variables. Range [−1, +1]. 0 means no linear relationship; can be 0
even when a strong nonlinear relationship exists. Sensitive to outliers.

**Spearman correlation (ρ_S)** — Pearson correlation applied to the *ranks* of
the two variables, not their values. Range [−1, +1]. Detects **monotonic**
(linear or nonlinear-but-monotone) association. Robust to outliers because
ranks ignore magnitudes. **Use as the default for this project's features**,
which often have fat tails.

**Mutual information (MI)** — information-theoretic measure of any kind of
statistical dependence between two variables. Unit: nats (or bits). 0 means
independence; large positive means strong association. **Catches nonlinear and
heteroskedastic structure** that Spearman misses (e.g., a feature that controls
the *variance* of `pnl_pct` rather than its sign — see `atr_pct` in item 02).
Unbounded; values are only meaningful within one comparison set.

**Point-biserial correlation** — Pearson correlation when one variable is
binary (0/1) and the other is continuous. Equivalent to standardized mean
difference between the two groups. Not separately reported in this project —
look at `winner_mean_feature` vs `loser_mean_feature` instead.

**Kolmogorov–Smirnov (KS) test** — compares two empirical distributions; the
statistic is the maximum vertical distance between their CDFs. Large statistic
+ small p-value = the two distributions differ. Used here to check whether the
distribution of a feature among winners differs from its distribution among
losers.

**Mann–Whitney U test** — non-parametric "is the median of group A different
from group B" test. Small p-value = yes. Doesn't assume normality. Companion
to KS — KS sees the whole distributional difference; MW focuses on the location.

**p-value** — the probability of seeing data at least as extreme as the
observed, assuming the null hypothesis is true. **Small p ≠ practical
significance**; with N = 1,500 trades, a feature with a true Spearman of
0.05 (useless) gets p < 0.05. Always read effect size + p together.

**Multiple-comparisons correction (Bonferroni / Benjamini–Hochberg)** —
when you test K features against the same target, the chance of one false
positive rises with K. Bonferroni divides the per-test significance threshold
by K (conservative). BH controls the false discovery rate (FDR). Not currently
applied in this project's per-feature scans; flagged because the 28-feature
ranking tables in item 02 have inflated apparent significance.

---

## 2. Classification metrics

**Base rate** — the unconditional fraction of positive cases in the dataset.
For `is_loser` on the regime-labeled subset: 0.43 (TQQQ), 0.54 (SQQQ). For
`is_severe_loss`: 0.31 (TQQQ), 0.42 (SQQQ). The bar to beat for any "predictor".

**ROC curve, AUC** — receiver-operating-characteristic curve plots true positive
rate vs false positive rate as the prediction threshold sweeps. The area under
that curve (AUC, range [0, 1]) is the probability that a random positive case
is ranked higher than a random negative one. **0.5 = random; 1.0 = perfect.**
0.6–0.65 is "weak but real" — typical of what we see here.

**Directional AUC** — `max(AUC, 1 − AUC)`. A feature that ranks negatives
*above* positives (AUC 0.40) is just as informative as one that ranks positives
above negatives (AUC 0.60), only with the sign flipped. Directional AUC erases
the sign so feature rankings can be compared.

**Precision** — fraction of items flagged-positive that are actually positive.
"Of the trades my rule says to skip, what fraction would have been losers?"
**This is what `precision_loser_rate_flagged` columns measure.**

**Recall** — fraction of actually-positive items that get flagged.
"Of all losers, what fraction did my rule catch?"

**Lift** — `precision − base_rate`. How much better than blindly flagging
everything. Item 04 reports `lift_oos_vs_baseline` per rule.

**Random-baseline filter** — a sanity check: a rule that randomly flags trades
at the same trigger rate. If your "smart" rule has the same precision as
random, it has no signal. Item 04's `random_baseline_precision` and
`precision_minus_random` columns implement this.

**Calibrated probability** — a predicted probability `p̂` is calibrated if,
among cases where `p̂ ≈ 0.7`, roughly 70 % are actually positive. L1-logistic
regression produces calibrated probabilities by default; raw decision-tree leaf
fractions are also calibrated by construction (on the training set).

---

## 3. Regression metrics

**R² (coefficient of determination)** — fraction of the target's variance
explained by the model on the evaluation set. Range (−∞, 1]. Positive means
the model beats predicting the target's mean; **negative means it does worse
than predicting the mean** (item 04's GBM on `pnl_pct` lands here).

**MSE / RMSE** — mean (root) squared error. Absolute units, hard to interpret
across targets without context. R² is MSE normalized by the target's variance.

**Residual** — actual minus predicted. Diagnostic plots of residuals expose
where the model is systematically wrong.

**Heteroskedasticity** — the residuals' variance depends on the predicted
value (or on some feature). When `atr_pct` has tiny Spearman with `pnl_pct`
but huge mutual information, the structure is heteroskedastic: `atr_pct`
controls *how spread out* `pnl_pct` is, not its mean. Standard regression
assumes constant residual variance, so it underweights this signal.

---

## 4. Multivariate methods

**Standardization (z-score)** — subtract the mean, divide by the standard
deviation. Puts every feature on a comparable scale. Required for L1-logistic
(so the penalty is applied symmetrically) and for PCA / PLS / LDA.

**Multicollinearity** — two or more predictors are nearly linearly redundant
(|correlation| ≈ 1). Wrecks linear models: coefficients become unstable and
hard to interpret. Trees are immune to this, but interpretation of feature
importance still suffers.

**VIF (variance inflation factor)** — per-feature multicollinearity diagnostic.
VIF = 1 / (1 − R²_j), where R²_j is from regressing feature j on all others.
VIF > 5 = moderate concern; VIF > 10 = drop or aggregate. Not currently
computed in this project — item 01 uses pairwise Spearman + hierarchical
clustering instead, which is sufficient here.

**Hierarchical clustering (on features)** — group features by similarity.
Distance = `1 − |correlation|`. Average-linkage clustering produces a tree; cut
at a chosen distance threshold to get a flat partition. Item 01 cuts at
|Spearman| ≥ 0.85.

**PCA (Principal Component Analysis)** — unsupervised rotation of the
feature space that picks axes (principal components) ranked by how much of the
total variance they explain. PC1 captures the most variance, PC2 the next-most
(orthogonal to PC1), etc. Used here to see if a small number of axes captures
the feature space. **PCA doesn't know about the target** — it can put all the
variance in a component that has nothing to do with `pnl_pct`.

**PLS-DA (Partial Least Squares Discriminant Analysis)** — like PCA but the
rotation is chosen to maximize the covariance of each component with the
**target**. The supervised cousin of PCA. PLS1 is the single linear
combination of features most strongly correlated with `is_loser`.

**LDA (Linear Discriminant Analysis)** — finds the single linear direction
that maximally separates two classes (here: winner vs loser). Different
optimization than PLS-DA but similar in spirit. Coefficients can be read as
"how much does this feature push the prediction toward `is_loser`".

---

## 5. Predictive models

**Decision tree** — recursive binary splits on features. Each leaf corresponds
to a conjunction of inequalities (e.g., `atr_pct > 0.476 AND hour > 12.5 AND
regime_chop_highvol = 1`). Interpretable. Sklearn: `DecisionTreeClassifier` /
`DecisionTreeRegressor`.

**`max_depth`** — limits how deep splits can go. Depth 4 gives at most 16
leaves. Smaller depth = less overfit, less expressive.

**`min_samples_leaf`** — refuses to create a leaf with fewer than N training
samples. Prevents the tree from carving micro-regions that don't generalize.
We use 30 throughout.

**Leaf** — a terminal node in the tree. The path of splits from root to leaf
defines a rule.

**L1-regularized logistic regression** — logistic regression with a penalty on
the **sum of absolute coefficient values**. Pushes weak coefficients to exactly
zero, producing a sparse model. `C` is the inverse of the penalty strength:
small C = more regularization = more sparsity.

**Gradient boosting (GBM)** — fits a sequence of small trees, each correcting
the residuals of the previous. Highly expressive, often the best off-the-shelf
tabular learner — but in this project's small-sample setting it overfits the
continuous `pnl_pct` target badly (negative OOS R² everywhere). Sklearn:
`GradientBoostingRegressor`.

**`learning_rate`** — GBM step size. Smaller = needs more trees (higher
`n_estimators`) but generalizes better. 0.05 with 200 trees is a sensible
default; we use it.

**Subsample** — fraction of training data each tree sees. < 1 introduces
stochasticity that reduces overfit. We use 0.9 for the GBM.

**Feature importance (impurity-based)** — for tree models, how much each
feature reduces training impurity across all splits. Built into sklearn's
`feature_importances_`. **Biased toward high-cardinality features**; can give
high importance to a feature that doesn't help OOS.

**Permutation importance (OOS)** — shuffle one feature's column in the test
set, see how much performance drops. **Directly measures the feature's OOS
contribution**, free of the impurity bias. Sklearn:
`sklearn.inspection.permutation_importance`. Large *positive* permutation
importance for a model with very bad OOS performance is a sign of overfitting,
not signal (item 04's SQQQ GBM).

**SHAP values** — per-prediction additive feature attributions, model-agnostic
when computed with `TreeExplainer` for tree models. Not currently used here
(`shap` not installed in the env). Mentioned because it's the cleanest way to
ask "for THIS trade, which features pushed the prediction toward loser?".

---

## 6. Validation conventions

**In-sample (IS)** — the data used to fit and discover. Numbers from IS are
allowed to be optimistic.

**Out-of-sample (OOS)** — held-out data used only to evaluate. Numbers from
OOS are what you trust. IS-OOS divergence is the standard overfit diagnostic.

**Walk-forward** — instead of one IS/OOS split, run many: train on
[start..Y], evaluate on Y+1. **Expanding window** keeps the training start
fixed and lets the window grow each step. **Rolling window** keeps the
training length fixed and slides it forward. We use expanding-window, yearly
step.

**IS-frozen rule** — a rule whose feature thresholds are chosen on IS data and
never updated. Evaluated as-is on OOS. This is the existing pipeline's pattern
and matches what we do in item 04's `wf_rule_eval`.

**Bootstrap confidence interval** — resample the OOS-flagged trades with
replacement, recompute the metric, repeat ~1,000 times, report the 2.5 / 97.5
percentile of the resulting distribution. Tells you how much of the observed
metric is sample noise. Used by the existing focus-rule validation.

---

## 7. Performance metrics

**Compounded equity vs constant notional**:
- **Compounded** — each trade's return scales next trade's capital base:
  `equity_t = equity_{t−1} × (1 + r_t)`. Geometric growth. CAGR is its natural
  rate. **The existing pipeline uses this, and it is inflated by the six-CSV
  `capital_before` resets** (see `FINDINGS.md` data finding #2).
- **Constant notional** — each trade contributes `pnl_pct / 100` to a fixed
  notional base, no compounding: `equity_t = 1 + Σ r_i`. Arithmetic growth.
  Item 05 uses this. The right convention when "what % was made per year on a
  fixed base" is the question.

**CAGR (Compound Annual Growth Rate)** — annualized geometric return:
`(final_equity / initial_equity)^(1/years) − 1`. Belongs with compounded
equity. **Under constant notional, replaced by `annualized_arith_return` =
total_return / years.**

**Sharpe ratio (daily)** — `mean(daily_return) / std(daily_return) × √252`.
Risk-adjusted return per unit of total volatility. Scale-invariant —
compounding vs constant-notional gives nearly the same Sharpe (item 05
verified this). Quote Sharpe across runs without correction.

**Sortino ratio (daily)** — same as Sharpe but uses only the standard
deviation of *negative* returns in the denominator. Punishes downside
volatility only. Also scale-invariant.

**Max drawdown (Max DD)** — the largest peak-to-trough decline in equity.
**Slightly worse under constant notional** because additive equity exposes
drawdowns at full depth; compounded equity hides them when running equity is
low.

**Calmar ratio** — annualized return divided by |max DD|. Belongs to whichever
equity convention CAGR uses. Item 05 reports both old (compounded) and new
(constant-notional) Calmar.

**Ulcer index** — RMS of drawdowns over time, in percent. Punishes both
depth and *duration* of drawdowns. Smaller is better. Cross-run absolute
comparison requires the same equity convention.

**VaR (Value at Risk, 95 %)** — the 5th percentile of `pnl_pct`. "95 % of
trades do at least this well." Reported in `traditional_metrics_baseline.csv`.

**CVaR (Conditional VaR, 95 %)** — mean of trades worse than the VaR
threshold. "When it does badly, how badly does it do on average?"

**Win rate** — `1 − loser_rate`. Fraction of trades with `pnl_pct ≥ 0`.

**Profit factor** — sum of winning trades' PnL divided by absolute sum of
losing trades' PnL. Above 1 = profitable on net.

**Expectancy** — average `pnl_pct` per trade.

---

## 8. Project conventions (summary)

For the authoritative version, see `FINDINGS.md` § "Data semantics" — this
section is a quick reference, not the source of truth.

- **`pnl_pct` is in percentage points**, not fractions. `2.5` means +2.5 %.
  Divide by 100 to use in any return formula.
- **TQQQ and SQQQ are always analyzed separately** — never pool.
- **In-sample window**: `entry_time ≤ 2020-12-31` (hardcoded `IS_END`). OOS is
  2021–2026.
- **Effective IS window for regime-labeled subset**: 2015–2020 (the 2013–2014
  rows are exactly the unlabeled ones — see `research/01_data_diagnostics/findings.md`).
- **`capital_before` resets** across the six source CSVs because each is its
  own backtest with its own $10k starting capital. The compounded
  `total_return_chain` and CAGR in old run folders are artificially inflated.
  Use constant-notional metrics (item 05) for absolute equity figures.
- **Loser target**: `pnl_pct < 0`. **Severe-loss target**: `pnl_pct ≤ −1` (one
  percentage point or worse).
- **Asymmetric pnl distribution**: SQQQ's median trade loses money; the
  strategy makes money via a few large winners. This is why binary skip rules
  fail OOS (item 04) — flagging losers also flags the winners that compensate
  them.
