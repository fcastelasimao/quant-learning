# 09 — Validation redesign

> Period: IS 2015–2020, R-OOS 2021–2025, Embargo 2026.

**Scope.** Define the validation convention that supersedes the single-cutoff IS=2015-2020 / OOS=2021-2026 split used in items 01-05.

## New convention

| split | years | role |
|---|---|---|
| **IS** | 2015 – 2020 | training / rule discovery |
| **RESEARCH_OOS** | 2021 – 2025 | OOS that research can iterate against |
| **EMBARGO** | 2026 (partial) | embargoed holdout; never seen during discovery; final number for any rule reported here ONCE |

Plus expanding-window walk-forward inside IS+RESEARCH_OOS for per-year stability checks.

The embargoed holdout is the key addition. As of this pass, 2026 is partial (~66-81 trades per symbol) but uncontaminated by any modeling iteration. Future passes that touch 2026 should clearly state they're using it as embargo and report numbers once.

## Demonstration on the SQQQ focus rule

| split | n_total | n_flagged | precision_loser | precision_severe | net_pnl_impact |
|---|---:|---:|---:|---:|---:|
| IS (2015-2020) | 783 | 13 | 0.92 | 0.15 | +3.99 |
| RESEARCH_OOS (2021-2025) | 662 | 5 | 0.80 | 0.60 | +3.22 |
| **EMBARGO (2026)** | 66 | 2 | **1.00** | **1.00** | **+2.04** |

The rule passes its first true-embargo test (n=2 is tiny, but both flagged trades were severe losses). The IS → RESEARCH_OOS → EMBARGO progression is monotone improving in precision-severe, which is a healthy pattern (the rule wasn't trained to maximize severe-rate; it just happens to do that in production).

**Practical note**: 2026 is only ~5 months in (the trade data ends mid-May 2026). When the rest of 2026 fills in, this embargo number will become much more informative. Until then, treat the 2026 result as a *direction* not a *magnitude*.

## How other items use this

| direction | uses split for |
|---|---|
| 04 (loss-region models) | Trained on IS, evaluated on RESEARCH_OOS+EMBARGO unified as "OOS". Did NOT separate EMBARGO. Should be patched in a future pass — but the conclusions don't substantially change. |
| 08 (focus rule recheck) | First item to apply EMBARGO separately. The focus rule passes. |
| 11 (regime-conditional rules) | RESEARCH_OOS used for ranking meta-rules; EMBARGO reported once per surviving meta-rule, never as a selection criterion. |
| 12 (sizing simulation) | Walk-forward refit per year, so no IS/OOS split needed — every year's prediction was made from data strictly prior to it. |

## Artifacts

| file | content |
|---|---|
| `build_09_validation_redesign.py` | analysis script |
| `validation_splits.csv` | the 3-row definition table |
| `focus_rule_under_new_splits.csv` | demo applied to the focus rule |
| `findings_09_validation_redesign.md` | this note |
