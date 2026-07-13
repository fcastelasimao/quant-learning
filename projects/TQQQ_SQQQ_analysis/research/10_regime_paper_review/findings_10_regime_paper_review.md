# 10 — Review of Zakamulin (2022) 5-state HSMM

**Paper.** Zakamulin, V. "Not all bull and bear markets are alike: insights from a five-state hidden semi-Markov model." *Risk Management* 25:5 (2022). Local copy: `/Users/franciscosimao/Documents/QuantFinance/personal_projects/resources/papers/s41283-022-00112-y.pdf`.

## What the paper does

Fits a hidden **semi**-Markov model (HSMM, generalizing HMM by allowing arbitrary state-duration distributions) to monthly S&P 500 capital-gain returns Jan 1957 – Dec 2020. Tries J ∈ {2..7} states. AIC favors J=3. Author overrides AIC with a novel "% correctly decoded states" criterion (the fraction of decoded states whose realized mean return sign matches the estimated state mean's sign) and picks J=5. Five labelled states emerge:

| State | μ (ann.) | σ (ann.) | Avg dur (months) | Share | Interpretation |
|---|---:|---:|---:|---:|---|
| 1 | +16 % | 6.7 % | 17 | 19 % | low-vol bull |
| 2 | −3 % | 12.8 % | 17 | 33 % | regular bear |
| 3 | +27 % | 12.7 % | 10 | 12 % | high-vol bull |
| 4 | −30 % | 25.3 % | 3 | 7 % | crash / correction |
| 5 | +20 % | 13.8 % | 17 | 29 % | bubble |

Stylized findings: bubbles always end in a bear (correction or crash), 56 % of bears escalate to crashes, V-shaped recovery after crashes is real. Asset-allocation case study shows the 5-state model reduces risk 30-35 % vs buy-and-hold while raising Sharpe 35-50 %.

## Critical assessment — should we implement?

**No, not as a direct HSMM fit.** Three blocking reasons:

1. **Frequency mismatch.** Monthly returns vs our 15-min trade strategy. The 5 states have **average durations of 3–17 months** — that's the *backdrop* of thousands of our trades, not a per-trade regime. Joining a 17-month state label to a 15-min entry timestamp gives every trade in a 17-month window the same label, which is the same information content as one tag per quarter. Not informative at our resolution.

2. **Wrong underlying.** S&P 500, not QQQ. NASDAQ-100 has different regime structure (more concentrated in tech, more rate-sensitive, different bubble dynamics). The 1999-2000 dot-com bubble that the paper highlights would look qualitatively different in QQQ-only fits.

3. **Methodological fragility + cherry-picked criterion.**
   - The author admits the HSMM is "often more art than science" — likelihood surface has multiple local maxima, parameters are sensitive to starting values and sample boundaries.
   - **AIC picks 3 states, not 5.** The novel "% correctly decoded states" criterion is admitted to be not "rigorously founded" and is essentially a self-fulfilling metric (it rewards models where the decoded state means have the same sign as the mean return over that period — easier to satisfy with more states because each state is shorter and noise averages out less).
   - The 5-state choice gives 93 % decoded states vs 91 % for J=2 — a 2-percentage-point improvement that's plausibly within the noise of the fitting procedure.

## What's worth borrowing — the conceptual taxonomy, NOT the model

The paper's five conceptual buckets *do* carve up the data in a useful way:

- **Low-vol bull** ≠ **high-vol bull** ≠ **bubble**. The strategy probably faces different risk-reward in each (item 04 already showed atr_pct controls severe-loss probability; the paper's distinction maps onto our atr_pct + VIX state).
- **Regular bear** ≠ **crash**. A 60-day slow drift down has different mean-reversion behavior than a 3-month vertical crash.

These distinctions can be captured **as continuous features** at *our* frequency (daily, on QQQ), without fitting an HSMM:

| Paper concept | Feature equivalent in item 06 |
|---|---|
| low-vol bull | `QQQ_dist_MA200 > 0 AND VIX_pctile_252d < 0.4` (composite) |
| high-vol bull | `QQQ_dist_MA200 > 0 AND VIX_pctile_252d > 0.6` (composite) |
| bubble | `QQQ_50d_return_pctile_252 > 0.9` (parabolic acceleration) |
| regular bear | `QQQ_drawdown_60d < -0.10 AND QQQ_drawdown_5d > -0.05` (slow drift) |
| crash | `QQQ_drawdown_5d < -0.10` (short, sharp) |

**Item 06's context-enrichment pass already captures all of these continuous features.** The depth-4 tree can build the same conceptual distinctions automatically through splits on these variables, without us prescribing the regime structure ex-ante.

## Decision

**Skip the HSMM. The paper's contribution to our project is its conceptual vocabulary**, which item 06 captures more granularly through continuous QQQ-derived features. If after item 06 we find that the tree learns splits that visibly correspond to bubble / crash regions, we'll know the taxonomy mattered — but we won't have paid the cost of fitting a fragile, monthly, SP500-trained, multi-modal model.

The decision could revisit if:
1. After fresh data arrives (item 06 + future passes), context features still fail to help (AUC doesn't move). Then a coarse regime label *might* add orthogonal information.
2. We move from "predict each trade" to "predict the trading regime should be active." Then a monthly-state label could matter as a meta-rule. Currently not in scope.

## Artifacts

| file | content |
|---|---|
| `findings_10_regime_paper_review.md` | this note |

(no `build_*.py` — this is a methodology decision, not a computation.)
