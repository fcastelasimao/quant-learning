# Item 06 — Tree overfitting diagnostic

Walk-forward IS vs OOS AUC for depth-4 tree and L1-logit on `is_severe_loss`.
Each fold trains on years < Y, evaluates on year Y.
A large mean IS–OOS gap (mean_gap) relative to L1-logit indicates overfitting.

## Summary table

symbol    model  mean_auc_train  mean_auc_test  mean_gap  std_gap  max_gap
  TQQQ     tree           0.793          0.630     0.163    0.105    0.425
  TQQQ l1_logit           0.684          0.584     0.100    0.094    0.235
  SQQQ     tree           0.811          0.677     0.134    0.079    0.271
  SQQQ l1_logit           0.695          0.571     0.124    0.080    0.235

## Interpretation

- **TQQQ**: tree mean gap = 0.163 vs L1 mean gap = 0.100 — gap is modest.
- **SQQQ**: tree mean gap = 0.134 vs L1 mean gap = 0.124 — gap is modest.

## Artifacts

| file | content |
|---|---|
| `overfit_diagnostic.csv` | Per-fold IS and OOS AUC for both models |
| `overfit_summary.csv` | Mean/std/max gap per symbol × model |
| `overfit_diagnostic.png` | IS vs OOS AUC line plot per fold |
| `findings_06_tree_overfit.md` | This note |