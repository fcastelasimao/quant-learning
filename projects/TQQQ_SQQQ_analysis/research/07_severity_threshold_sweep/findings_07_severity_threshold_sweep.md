# 07 — Severity threshold sweep

> Period: IS 2015–2020, OOS 2021–2026, WF 2017–2025. Curated 13 features (12 numeric + regime dummies). Depth-4 tree + L1-logit (C=0.1).

**Scope.** For each (symbol, side, threshold), fit a depth-4 tree (class-balanced) AND an L1-logistic regression on the binary target and report WF-median AUC across 2018–2026 evaluation years. Curated 12 features + regime dummies. Targets:

- Loss side: `pnl_pct ≤ θ`, θ ∈ {−0.25, −0.5, −1.0, −1.5, −2.0, −3.0}
- Win side: `pnl_pct ≥ θ`, θ ∈ {+0.5, +1.0, +1.5, +2.0, +3.0}

## Headline — extreme losses are dramatically more predictable than ordinary ones

WF-median OOS AUC by loss threshold (tree and L1-logit):

| threshold | TQQQ tree | TQQQ L1-logit | SQQQ tree | SQQQ L1-logit |
|---:|---:|---:|---:|---:|
| −0.25 | 0.54 | 0.60 | 0.58 | 0.60 |
| −0.50 | 0.54 | 0.60 | 0.58 | 0.60 |
| **−1.0** | **0.65** | **0.58** | **0.64** | **0.54** |  ← current project threshold
| −1.5 | 0.82 | 0.76 | 0.83 | 0.73 |
| **−2.0** | **0.91** | **0.89** | **0.90** | **0.86** |
| **−3.0** | **0.97** | **0.97** | **0.88** | **0.96** |

**The monotonic rise holds for both model families.** Tree wins at −1% and −1.5% on both symbols (depth-4 tree partitions well on this moderate imbalance). L1-logit closes the gap at −2% and exceeds tree at −3% (fewer positives makes the smooth logistic boundary more sample-efficient than a greedy tree).

Sample-size caveat at extreme thresholds (TQQQ had only 14 trades ≤ −3 % in OOS), which is why **−2 % is the practical sweet spot**: high AUC (~0.90 tree / ~0.89 logit), enough positives to bootstrap (82 TQQQ OOS, 125 SQQQ OOS), reasonable trigger rate.

## L1-logit comparison

L1-logit underperforms tree at mild thresholds (−1%, −1.5%) by 5–10 AUC points. This is not surprising: depth-4 trees naturally fit the non-linear severity boundary better when there are enough positives. However:

- At −2% and −3%, L1-logit is competitive with tree.
- **L1-logit handles enriched feature sets more gracefully than depth-4 trees** (see item 06 — enriched tree AUC drops while enriched L1-logit AUC rises). So for item 12/17 sizing, where we add 21 daily context features, L1-logit is the correct choice.
- For curated-only binary skip rules, tree still wins at the −1% threshold.

## Win side — much weaker

| threshold | TQQQ tree | TQQQ L1-logit | SQQQ tree | SQQQ L1-logit |
|---:|---:|---:|---:|---:|
| +0.5 | 0.55 | 0.62 | 0.54 | 0.60 |
| +1.0 | 0.58 | 0.63 | 0.58 | 0.58 |
| +1.5 | 0.57 | 0.63 | 0.60 | 0.59 |
| +2.0 | 0.58 | 0.68 | 0.58 | 0.61 |
| +3.0 | 0.62 | 0.64 | 0.56 | 0.63 |

Win-side signal is materially weaker than loss-side at all thresholds. L1-logit does modestly better (especially TQQQ at +2%), but even 0.68 is far below the 0.90 we get for losses at −2%. **There is no "size up big winners" signal comparable in strength to the "size down severe losses" signal.**

## Implications for downstream items

- **Use `is_severe_loss @ −2%` as the primary target** in all downstream models. AUC distance from random roughly triples (0.65 − 0.5 = 0.15 → 0.91 − 0.5 = 0.41 for tree).
- For **enriched sizing** (items 12/17): use L1-logit. At −1% it's weaker than tree on curated features, but it scales to 35 enriched features while tree overfits.
- Win-side predictability not worth pursuing further with this feature set.

## Top-decile lift

For the −2 % target (tree):
- TQQQ: 37 % severe-loss rate in flagged decile vs 9.6 % baseline (lift +0.28)
- SQQQ: 44 % vs 17 % baseline (lift +0.27)

Useful for a sizing rule: down-size positions in the top 10 % of predicted-severe-loss space.

## How to read the plots

**`severity_sweep_auc.png`** — two-panel chart (loss side left, win side right). Each symbol now shows two lines: solid for tree, dashed for L1-logit.

- **X axis**: threshold value (more negative = more extreme loss on the left panel).
- **Y axis**: walk-forward median OOS AUC across the 9 WF windows.
- **Shaded band** (tree only): WF min-to-max range.
- **Dashed horizontal at 0.5** = random.
- **Key pattern on left**: both model lines rise steeply as threshold moves from −0.5 to −2.0. The lines converge at −2% and cross at −3%.

## Artifacts

| file | content |
|---|---|
| `build_07_severity_threshold_sweep.py` | analysis script (tree + L1-logit) |
| `severity_sweep.csv` | per (sym, side, threshold, model) WF stats |
| `severity_sweep_auc.png` | WF AUC vs threshold, both sides, both models |
| `findings_07_severity_threshold_sweep.md` | this note |
