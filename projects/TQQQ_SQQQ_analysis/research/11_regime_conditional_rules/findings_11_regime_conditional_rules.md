# 11 — Regime-conditional rules

> Period: R-OOS 2021–2025, Embargo 2026, regime-labeled subset.

**Scope.** Take each candidate rule from item 04 (depth-4 tree leaves with IS precision ≥ 0.65 and n_IS ≥ 30). For each, evaluate per (regime × split) cell. Construct meta-rules of the form "apply rule R only when regime == G". Score on RESEARCH_OOS (2021-2025). Report EMBARGO (2026) once per surviving rule without iterating against it.

Gating: meta-rule must trigger ≥ 8 times in RESEARCH_OOS to even be reported.

## Headline — four meta-rules have positive RESEARCH_OOS net pnl impact

(Sorted by impact, with EMBARGO numbers shown but not used for selection):

| symbol | meta-rule | OOS n_flagged | OOS precision | **OOS net pnl** | EMBARGO n_flagged | EMBARGO net pnl |
|---|---|---:|---:|---:|---:|---:|
| **TQQQ** | `TQQQ_severe_atr-s100_aa66__given_sideways_lowvol` | 10 | **1.00** | **+10.61** | 1 | +1.01 |
| SQQQ | `SQQQ_loser_hr-s50c-d20_23be__given_bull` | 9 | 0.78 | +2.67 | 0 | 0.00 |
| SQQQ | `SQQQ_loser_hr-s50c-d20_ec07__given_chop_highvol` | 10 | 0.70 | +2.08 | 0 | 0.00 |
| SQQQ | `SQQQ_loser_hr-s50c-d20_23be__given_sideways_lowvol` | 11 | 0.73 | +2.05 | 1 | +1.41 |

**The TQQQ severe-loss rule, restricted to `sideways_lowvol` regime, has 100 % OOS precision over 10 flagged trades with +10.6 pp net impact.** The same rule applied across all regimes had net-negative impact in item 04. This is the cleanest example of why regime-conditioning matters: the rule's loss-region is real, but only IN a specific market context. Skipping it everywhere hits too many winners; skipping it ONLY in sideways_lowvol catches the losers without the winners.

## Why this works

The parent rule's path conditions specify `atr_pct ∈ (0.42, 0.48] AND MA100_D1 ≤ 0.0002`. In bull regimes that feature combination occurs alongside upward momentum, and the losers are offset by winners. In sideways_lowvol the same feature combination is a clean "no momentum + mild vol" signal that consistently produces losers without the offsetting winners.

The SQQQ meta-rules (3 variants of the same parent rule, conditioned on different regimes) are smaller in effect but each survives independently. The fact that **the same parent rule pays off in multiple regimes when activated conditionally** is encouraging — it's not a single-cell artifact.

## Why this is overfit-risk territory

- 4 surviving meta-rules out of 14 candidate rules × 3 regimes = 42 trials. By chance alone at α=0.05 we'd expect ~2 false positives. The TQQQ headline (10/10 precision) is unlikely to be chance; the SQQQ trio is closer to noise floor.
- EMBARGO sample sizes are tiny (0-1 flagged per rule in 2026 so far). The +1.01 and +1.41 EMBARGO numbers are directional only, not magnitude-credible.
- Random-baseline precision at the same trigger rate (per `rule_x_regime_cells.csv`) is also informative — meta-rules that beat random by < 5 pp shouldn't survive.

**Honest read**: the TQQQ severe-in-sideways_lowvol meta-rule looks real and worth tracking. The three SQQQ meta-rules are weaker and need more out-of-sample data to confirm.

## Implication

Regime-conditional rule activation **is** the right next-tier framing — item 04's binary skip rules fail because they're not regime-aware. Conditioning them on regime recovers a positive expected value in at least one clear case.

For productionization (deferred): the meta-rule "skip TQQQ trades when in sideways_lowvol AND atr_pct ∈ (0.42, 0.48] AND MA100_D1 ≤ 0.0002" would be the first candidate. Trigger rate is ~1.5 % of all TQQQ trades. Net pnl impact estimate, scaled conservatively to a year of TQQQ trading volume (~200 trades), is ~3 pp/year.

## Artifacts

| file | content |
|---|---|
| `build_11_regime_conditional_rules.py` | analysis script |
| `rule_x_regime_cells.csv` | every (rule × regime × split) cell with precision/net_pnl/random_baseline |
| `regime_conditional_meta_rules.csv` | meta-rules sorted by RESEARCH_OOS net pnl impact |
| `findings_11_regime_conditional_rules.md` | this note |
