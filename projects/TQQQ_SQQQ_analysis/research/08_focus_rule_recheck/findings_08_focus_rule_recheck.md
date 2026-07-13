# 08 — Focus rule (`SQQQ rsi_x_atr_cell_3_1`) re-check

> Period: OOS 2021–2026, regime-labeled SQQQ subset (1,511 trades).

**Rule.** SQQQ trades where `RSI_entry ∈ [56.4, 59.85]` AND `atr_pct ∈ [0.39, 0.47]`.

**Setup.** Re-evaluate on the regime-labeled subset (dropping the 22 % of SQQQ rows without regime). Per-OOS-year. Per-regime. Naive bootstrap vs **block bootstrap by year** (corrects for year-clustering of flagged trades). Random-baseline filter at the same trigger rate. EMBARGO (2026) reported separately under item 09's split.

## Headline — the rule holds under harder tests

| metric | full_data (old `FINDINGS.md`) | regime-labeled (this check) |
|---|---:|---:|
| OOS n_flagged | 8 | **7** (one trade was unlabeled-regime; correctly dropped) |
| OOS precision (loser rate) | 0.875 | **0.857** |
| OOS precision (severe loss) | n/a | **0.714** (5 of 7 are severe) |
| OOS net pnl impact | +6.35 | **+5.26** |
| OOS naive bootstrap CI (95 %) | [+2.43, +8.45] | **[+1.41, +7.33]** (still entirely positive) |
| **OOS block-bootstrap-by-year CI (95 %)** | not previously computed | **[+2.16, +8.36]** (still entirely positive) |
| Random same-rate net impact (median, 500 iters) | not computed | **−5.90** (rule beats random by ~+11 pp) |

**The rule survives the block bootstrap.** This was the test I was most skeptical it would pass — block bootstrap by year correctly accounts for the fact that 5 of 7 flagged trades came from 2024-2026. Even when the bootstrap resamples *which years' trades to include*, the 2.5 % percentile is **+2.16 pp**, well above zero. The wins are not a single-year fluke.

## Per-year breakdown (regime-labeled OOS)

| year | n_total | n_flagged | precision_loser | precision_severe | mean_flagged_pnl_pct |
|---:|---:|---:|---:|---:|---:|
| 2021 | 121 | 2 | 0.50 | 0.00 | −0.02 |
| 2022 | 149 | 0 | — | — | — |
| 2023 | 110 | 0 | — | — | — |
| 2024 | 136 | 2 | **1.00** | **1.00** | **−1.07** |
| 2025 | 146 | 1 | **1.00** | **1.00** | −1.03 |
| 2026 | 66 | 2 | **1.00** | **1.00** | −1.02 |

2022 and 2023 had zero triggers (the regime didn't enter the rule's feature region). 2024-2026 had 5/5 precision. 2021 was the only mediocre year (one of two trades was actually a small winner).

## Per-regime breakdown (regime-labeled OOS)

| regime | n_total | n_flagged | precision_loser | precision_severe | mean_flagged_pnl_pct |
|---|---:|---:|---:|---:|---:|
| bull | 178 | 0 | — | — | — |
| chop_highvol | 315 | 3 | **1.00** | **1.00** | −1.03 |
| sideways_lowvol | 235 | 4 | 0.75 | 0.50 | −0.54 |

**Zero triggers in bull regime** — the rule's feature window doesn't overlap with bull-regime conditions. The wins concentrate in chop_highvol (3/3 severe losses) with some in sideways_lowvol. This is consistent with item 11's regime-conditional finding.

## What changed vs the original `FINDINGS.md` claim

- Original: OOS precision 87.5 %, 8 trades.
- Re-check: OOS precision 85.7 %, 7 trades (regime-labeled subset).
- One previously-counted trade in `FINDINGS.md` was in an unlabeled-regime period. Removing it doesn't change the conclusion.
- The block-bootstrap CI is **the new credibility check that wasn't done before**. It passes.

## Updated reading of the rule

Previously I argued in pre-execution discussion that "the bootstrap CI [+2.43, +8.45] is structurally misleading because 6 of 8 trades came from 2024-2026 — plausibly the rule got lucky in one recent regime." **That hypothesis is wrong.** The block bootstrap explicitly corrects for year-clustering and the CI is still entirely positive. The 2024-2026 concentration reflects the rule's feature window simply not firing in 2022-2023, not a luck artifact.

**The rule is real signal.** It's small in absolute trade count (7 OOS triggers across 5 years = ~1.4/year), but it has high precision, large net pnl impact relative to its size, and survives the harder statistical test.

## Open question

Why does the rule fire in chop_highvol with perfect precision but rarely in sideways_lowvol? The feature window (RSI ∈ [56.4, 59.85] AND atr_pct ∈ [0.39, 0.47]) might mechanically map to a specific subregion of the chop_highvol regime — the moderate-RSI, moderate-vol corner. Could check by inspecting the joint (regime × RSI × atr_pct) distribution. Out of scope here.

## How to read the plots

**`yearly_triggers.png`** — vertical bar chart of how often the focus rule fired each OOS year, annotated with realized precision.

- **X axis**: OOS year (2021..2026).
- **Y axis**: number of trades flagged by the rule that year.
- **Annotation above each bar**: realized loser-precision among those flagged trades. `prec=100%` means every flagged trade in that year was a loser; `n=0` means the rule didn't fire that year.
- **What to look for**: (a) **zero-trigger years (2022, 2023)** — the feature window simply didn't get hit. Doesn't reflect a "miss"; it's a feature of how the rule is shaped. (b) **2024-2026 cluster** with perfect precision — 5 trades, all losers. (c) Whether the precision number is *spread* across years or *concentrated* in one. Spread = robust signal. Concentrated = the rule got one lucky regime.
- **Why it matters for this project**: pre-execution I'd argued the bootstrap CI was misleading because 6 of 8 trades came from 2024-2026. The bar chart makes that concentration visible at a glance. **The block-bootstrap-by-year test (item 08's headline) explicitly corrects for that concentration and the rule still survives.**

## Artifacts

| file | content |
|---|---|
| `build_08_focus_rule_recheck.py` | analysis script |
| `focus_rule_summary.csv` | per (scope, period/year/regime) summary |
| `focus_rule_bootstrap.csv` | naive vs block bootstrap CI + random baseline |
| `oos_flagged_trades.csv` | the actual 7 OOS-flagged trades for inspection |
| `yearly_triggers.png` | bar chart of yearly trigger count + precision |
| `findings_08_focus_rule_recheck.md` | this note |
