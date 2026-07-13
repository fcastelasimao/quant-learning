# 18 — Combined strategy simulation

> Period: OOS 2018–2026, walk-forward annual refit, strict-prior enriched features. Combined = p_severe sizing (L1-logit) + focus rule (SQQQ) + regime rule (TQQQ). Targets: TQQQ @ −1%, SQQQ @ −2% (per item 17 recommendation).

**Scope.** Test whether the three independently validated components compose
additively or redundantly: (A) continuous p_severe sizing, (B) the SQQQ RSI×ATR
focus rule (item 08), (C) the TQQQ sideways_lowvol regime rule (item 11).

## Headline comparison

| symbol | scenario | mean size | CAGR | Sharpe | Max DD | Calmar |
|---|---|---:|---:|---:|---:|---:|
| **TQQQ** | baseline_full | 1.00 | 158% | 4.14 | −18.3% | 8.64 |
| TQQQ | p_severe_only | 0.67 | 103% | 4.20 | **−10.0%** | 10.28 |
| TQQQ | rules_only | 0.99 | 160% | 4.20 | −18.3% | 8.75 |
| TQQQ | **combined** | **0.66** | 105% | **4.27** | **−10.0%** | **10.43** |
| **SQQQ** | baseline_full | 1.00 | 113% | 2.56 | −18.1% | 6.22 |
| SQQQ | p_severe_only | 0.86 | 86% | 2.60 | −15.2% | 5.64 |
| SQQQ | rules_only | 0.99 | 113% | 2.58 | −18.1% | 6.26 |
| SQQQ | **combined** | **0.84** | 87% | **2.63** | **−15.2%** | **5.69** |

## Component attribution

| symbol | n trades | rule fires | p_severe > 0.5 | overlap |
|---|---:|---:|---:|---:|
| TQQQ | 1,629 | 23 (1.4%) | 86 (5.3%) | **0** |
| SQQQ | 1,332 | 19 (1.4%) | 63 (4.7%) | **0** |

**Zero overlap in both symbols.** The crisp rule fires on a completely different
set of trades than the high-p_severe trades. The components are complementary,
not redundant.

## Interpretation

**Combined beats both components individually for both symbols:**

- TQQQ combined Sharpe **4.27** vs p_severe_only 4.20 (+0.07) and rules_only 4.20 (+0.07).
- SQQQ combined Sharpe **2.63** vs p_severe_only 2.60 (+0.03) and rules_only 2.58 (+0.05).
- Max drawdown: driven entirely by p_severe (rules_only has same MaxDD as baseline — 23/19 skipped trades are too few to move the portfolio max-drawdown meaningfully).

**Rules_only contribution is small but additive:**

The crisp rules fire on only 1.4% of trades. This is by design — they target narrow high-precision cells (item 11 TQQQ: 10 trades, 100% precision; item 08 SQQQ: ~7 trades, ~86% precision). The contribution to combined is the marginal Sharpe lift (+0.07 TQQQ, +0.05 SQQQ) on top of what p_severe already captures.

**p_severe is the load-bearing component.** It handles 5% of trades and delivers the full drawdown reduction. The crisp rules are an incremental Sharpe improvement on top.

**The combined strategy is the recommended deployment configuration:**
- TQQQ: `combined` (p_severe linear_skip @ −1% + sideways_lowvol rule) — Sharpe 4.27, MaxDD −10.0%, Calmar 10.43
- SQQQ: `combined` (p_severe linear_skip @ −2% + RSI×ATR rule) — Sharpe 2.63, MaxDD −15.2%, Calmar 5.69

## Caveats

1. **Rules are OOS-evaluated but the crisp thresholds came from IS rule discovery.** Treat the exact thresholds as subject to drift. Re-check against each annual checkpoint.
2. **p_severe walk-forward uses enriched features (35 cols).** Requires daily cross-asset data at trade entry time. The deployment pipeline must enforce `context_date < entry_date`.
3. **Zero overlap is a good sign, not guaranteed to persist.** In a different regime, high-VIX periods could cause the focus-rule cell to also have elevated p_severe.

## How to read the plots

`combined_equity_curves.png`: grey = baseline, blue = p_severe_only, green = rules_only, red = combined. Both panels show the combined (red) tracking the p_severe (blue) curve closely but with slightly higher equity — the rule contribution is visible as small divergences.

## Artifacts

| file | content |
|---|---|
| `build_18_combined_strategy.py` | Analysis script |
| `combined_strategy_summary.csv` | Per (symbol, scenario) metrics |
| `component_attribution.csv` | Rule fire counts, p_severe > 0.5 counts, overlap |
| `combined_equity_curves.png` | Per-symbol equity curves for all 4 scenarios |
| `findings_18_combined_strategy.md` | This note |
