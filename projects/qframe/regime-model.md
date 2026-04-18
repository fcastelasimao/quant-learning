# Regime Model

*Read when working on HSMM implementation, Hurst exponent, velocity metrics, or regime-conditional factor allocation.*

---

## Five-State Hidden Semi-Markov Model (HSMM)

**Reference:** Zakamulin (2023), "Not all bull and bear markets are alike", Risk Management.

The HSMM differs from a standard HMM in that state duration is modelled explicitly — states have a built-in duration distribution rather than the geometric duration implied by constant transition probabilities. This captures the empirical finding that bull and bear markets have characteristic durations.

### Five States

| State | Economic label | Return characteristics |
|---|---|---|
| 1 | Low-vol bull | High mean, low variance, long duration |
| 2 | High-vol bull | Positive mean, elevated variance |
| 3 | Bubble | Very high mean, low variance, short duration |
| 4 | Regular bear | Negative mean, moderate variance |
| 5 | Crash | Strongly negative mean, high variance, very short duration |

### Implementation

```python
# Primary library: hmmlearn
from hmmlearn import hmm

# For HSMM with explicit duration distributions: pomegranate
# pip install pomegranate
from pomegranate.hmm import DenseHMM
```

**Key parameters to fit:**
- State means and variances (Gaussian emissions)
- Transition matrix (sparse — not all transitions are allowed)
- Duration distributions (Poisson or Negative Binomial per state)

**Fitting:** Baum-Welch EM algorithm. Run multiple initialisations with different random seeds. Keep the run with the highest log-likelihood.

**Output at each timestep:** Posterior probability vector $\pi_t \in \Delta^4$ (probabilities over 5 states summing to 1).

---

## Implementation Status

**Phase 2 is complete. All modules are implemented in `src/qframe/regime/`.** Gate 2 passed (2026-04-17).

| Module | Status | Notes |
|--------|--------|-------|
| `hsmm.py` — `RegimeHSMM` | ✅ Complete | `[r, r²]` features; canonical state ordering (ascending mean); walk-forward via `fit_rolling()` |
| `hurst.py` — `HurstEstimator` | ✅ Complete | DFA-based rolling Hurst; NumPy 2.0 compatible |
| `velocity.py` — `RegimeVelocity` | ✅ Complete | KL-divergence velocity + EWM smoothing |
| `analyzer.py` — `RegimeICAnalyzer` | ✅ Complete | Integrates all three; `fit()`, `regime_ic_decomposition()`, `unconditional_vs_conditional()`, `regime_weights()` |
| `src/qframe/viz/charts.py` Charts 16–19 | ✅ Complete | `plot_regime_timeline`, `plot_regime_ic`, `plot_velocity`, `plot_hurst_rolling` |
| `notebooks/phase2_regime_analysis.ipynb` | ✅ Complete | Sections A–D with explanatory markdown + equity curve |

### Phase 2 Results (OOS 2018–2024, 449-stock universe)

5-state HSMM fitted walk-forward (504-day window, 63-day step, refit every quarter).

**Regime characterisation:**

| State | Label | % Time | Ann Return | Ann Vol | Sharpe |
|-------|-------|--------|------------|---------|--------|
| 0 | Bear/stress | 17.7% | 3.4% | 18.5% | 0.19 |
| 1 | Neutral | 24.8% | 19.3% | 23.9% | 0.81 |
| 2 | **Strong bull / low-vol** | 12.5% | **43.8%** | 14.7% | **2.98** |
| 3 | Choppy bear | 23.2% | 3.6% | 19.5% | 0.18 |
| 4 | Bull | 21.8% | 21.1% | 19.8% | 1.06 |

**IC decomposition:**

| Factor | Uncond IC | Best-state IC | Lift | Verdict |
|--------|-----------|---------------|------|---------|
| impl_82 Calmar h=1 | 0.0646 | 0.082 (state 2) | 1.27× | NO-GO — use unconditionally |
| impl_53 mean-rev h=63 | 0.0238 | 0.054 (state 2) | **2.28×** | ✅ GO — regime-condition this factor |

Gate prerequisite was at least 1 factor BHY-significant in Phase 1. **Cleared: 3 BHY-significant factors** (impl_82 t=10.74, impl_53 t=8.15, impl_92 t=10.74).

---

## Two Distinct Notions of "Velocity" in This Project

**1. Regime transition velocity (this document)**
The rate at which the HSMM's belief about the current market state changes. Computed from the state posterior distribution π_t (a 5-vector of probabilities). The recommended metric is KL divergence between consecutive posteriors. This is a Phase 2 signal — it requires the HSMM to be built first.

**2. Price velocity as a factor (Phase 1, can be tested now)**
The rate of change of price or returns as a cross-sectional predictor. For example:
- `price_acceleration`: second derivative of log-price (does momentum have momentum?)
- `return_rate_of_change`: (return_t − return_{t-lag}) / return_{t-lag}
- `momentum_decay_rate`: slope of the IC decay curve as a signal in itself

These are **Phase 1 factors** — they can be submitted to the pipeline immediately via:
```bash
./run_pipeline.sh --domain momentum --n 5
# Synthesis agent will explore price velocity variants if prompted with
# "focus on rate-of-change, acceleration, or derivative-based momentum signals"
```

---

## Regime Velocity Metrics

Five candidates for the "velocity" of regime change in discrete time. All are computed from $\pi_t$ and should be tested as separate features.

| Metric | Formula | Notes |
|---|---|---|
| First-order difference | $v_t^{(1)} = \pi_t - \pi_{t-1}$ | Vector. Direct and interpretable. |
| Second-order (acceleration) | $a_t = \pi_t - 2\pi_{t-1} + \pi_{t-2}$ | Vector. Rate of change of the change. |
| EW smoothed | $v_t^{(\alpha)} = (1-\alpha)\sum_s \alpha^s(\pi_{t-s} - \pi_{t-s-1})$ | Scalar per state. Reduces noise. |
| Geodesic (Fisher) | $v_t^{\text{geo}} = \|\log(\pi_t/\pi_{t-1})\|_{\pi_{t-1}}$ | Scalar. Weights small-prob changes more. |
| KL divergence | $D_{KL}(\pi_t \| \pi_{t-1}) = \sum_k \pi_t(k) \log \frac{\pi_t(k)}{\pi_{t-1}(k)}$ | Scalar. Most principled. Use as primary. |

**Recommended primary:** KL divergence. Zero = no belief update. Large values = sharp regime signal arrived. Directly interpretable as information gained.

```python
import numpy as np

def kl_velocity(pi_t: np.ndarray, pi_prev: np.ndarray, eps: float = 1e-8) -> float:
    """KL divergence from pi_prev to pi_t. Scalar regime velocity."""
    pi_t = np.clip(pi_t, eps, 1)
    pi_prev = np.clip(pi_prev, eps, 1)
    return float(np.sum(pi_t * np.log(pi_t / pi_prev)))
```

---

## Rolling Hurst Exponent

**Method:** Detrended Fluctuation Analysis (DFA). More robust than R/S for financial data.

**Interpretation:**
- $H > 0.6$: trending, persistent — overweight momentum
- $H < 0.4$: mean-reverting — overweight value and contrarian
- $H \in [0.4, 0.6]$: near-critical zone — reduce factor exposure, favour quality and low-vol

**Connection to phase transitions:** $H = 0.5$ is the critical point analogous to a phase boundary in statistical physics. Departures from this critical point indicate which regime the market is in.

```python
# Option 1: nolds library
import nolds
H = nolds.dfa(returns_series)

# Option 2: manual DFA (implement as learning exercise)
def compute_hurst_dfa(series: np.ndarray, window: int = 250) -> float:
    """Rolling DFA Hurst exponent on the last `window` observations."""
    # Detrend in blocks, compute fluctuation function, fit power law
    ...
```

**Rolling computation:** 250-day expanding or rolling window. Update daily. Computationally trivial.

---

## Factor Allocation Function

The complete factor weight function incorporating all regime signals:

$$\mathbf{w}_t^{\text{factors}} = g\!\left(\hat{\pi}_t,\; v_t^{\text{KL}},\; H_t,\; \text{regime}_t\right)$$

Where:
- $\hat{\pi}_t$ — HSMM posterior (5-vector)
- $v_t^{\text{KL}}$ — KL divergence velocity (scalar)
- $H_t$ — rolling Hurst exponent (scalar)
- `regime_t` — most probable state (integer 1–5, derived from $\hat{\pi}_t$)

**Implementation approach:** Start with a simple linear rule based on the most probable regime state. Add continuous weighting by posterior only after Gate 1 passes. Add velocity and Hurst after Gate 2 passes.

---

## Three-Signal Fusion (Phase 2 target state)

| Signal | Source | Role |
|---|---|---|
| HSMM posterior $\hat{\pi}_t$ | Historical returns | Primary regime label |
| KL velocity $v_t^{\text{KL}}$ | Rate of change of $\hat{\pi}_t$ | Transition early warning |
| Hurst exponent $H_t$ | Rolling DFA | Trending vs. mean-reverting classification |
| LLM semantic label | News/text data | Sanity check on statistical regime label |

The LLM semantic layer is Phase 2 Gate 2+. Do not add it before the statistical signals are validated.

---

## Crisis Failure Modes and Mitigations

*Added 2026-04-18. These are the four known failure modes of the dollar-neutral long/short strategies
in tail-risk / geopolitical shock scenarios. Each has a concrete mitigation, ranging from already-
implemented to Phase 3 prerequisite.*

### Background: How Dollar-Neutral L/S Makes Money

A dollar-neutral long/short portfolio has beta ≈ 0 by construction ($1 long per $1 short). It captures
**cross-sectional alpha** — the spread between longs (high-factor-score stocks) and shorts
(low-factor-score stocks). When the market moves ±3%, both books move roughly equally and cancel.
The return comes entirely from whether the factor correctly ranked stocks relative to each other.

IC = 0.0646 (impl_82) means: the factor's cross-sectional rank ordering has a 6.46% Spearman
correlation with next-day relative returns — small but consistent. This is the source of the edge.
It is unrelated to market direction, which is why the strategy has low market correlation in normal
conditions.

---

### Failure Mode 1 — Factor Crash (CRITICAL RISK)

**What happens:** When hedge funds face redemptions or margin calls, they liquidate their most liquid
long/short books simultaneously. Every quant fund selling the same "quality/momentum" longs creates a
**factor crash**: your long book falls as others dump the same names; your short book is squeezed as
those names rally (crowded shorts get expensive). The 2007 quant meltdown (−15% to −25% in a week
followed by snapback) is the canonical example. The OOS period 2018–2024 does not contain an event
of this severity.

**Why impl_82 (Calmar ratio) is specifically exposed:** Calmar is essentially a momentum-quality
measure. Momentum factors have among the highest AUM crowding in systematic equity — there is
significant overlap with how other L/S funds are positioned.

**Mitigations:**
1. **Pairwise correlation circuit breaker** — if 5-day rolling correlation between impl_82 and the
   equal-weight S&P 500 spikes above 0.5 (from its normal ~0.0), reduce gross exposure by 50%.
   Implementable today using existing `regime_weights()` infrastructure.
2. **Factor spread monitor** — track the daily return spread between top and bottom decile of
   impl_82 ranking. If the 21-day rolling mean spread turns negative, pause trading.
3. **Diversification across uncorrelated domains** — the cure for crowding is uncorrelation. Adding
   a BHY-significant `volatility` or `value` factor would reduce overlap with momentum quant books.
   Continue Phase 1 runs across all 5 domains specifically for this purpose.
4. **Gross exposure cap** — never exceed 2× NAV gross exposure (current backtest assumes 1×). At 1×,
   a 15% factor crash is a 15% drawdown; at 2× leverage it is 30%.

**Implementation priority:** Pairwise correlation circuit breaker is a **Phase 2.5 addition** — can
be added as a single conditional in the `regime_weights()` multiplier calculation.

---

### Failure Mode 2 — HSMM Lag (MEDIUM RISK)

**What happens:** The HSMM uses 504-day rolling windows re-fitted every 63 days. In a sudden
geopolitical shock (oil spike, nuclear escalation), the regime posterior takes 5–15 trading days to
meaningfully shift. During this lag window, the `regime_weights()` multiplier remains near 1.0
while the actual environment may warrant 0.2–0.3 exposure. The model is calibrated to detect
regimes that persist for months, not shocks that resolve in hours.

**Mitigations:**
1. **Fast emergency brake** — add a secondary rapid-response indicator that does not depend on the
   HSMM: if 1-day market return < −3% OR 5-day VIX-equivalent (rolling return std) > 95th percentile
   of historical 5-day stds, immediately clip the `regime_weights()` output to max 0.5.
   This is a single-parameter rule, no model fitting required.
2. **Velocity amplification in the multiplier** — the `regime_weights()` already applies a velocity
   downscale (`velocity_scale_down=0.5` when velocity > 90th percentile). Reduce this to 0.3 and lower
   the threshold percentile to 80 to make it more responsive.
3. **Intraday re-evaluation** (Phase 3) — current HSMM is daily. A faster HSMM fitted on hourly
   returns would detect regime shifts 5–10× faster, at the cost of more noise.

**Implementation priority:** Fast emergency brake is a **Phase 2.5 addition** to `analyzer.py`'s
`regime_weights()`. Velocity parameter tuning can be done in `phase25_portfolio.ipynb`.

---

### Failure Mode 3 — Short Borrow Execution (LOW-MEDIUM RISK)

**What happens:** In risk-off events, prime brokers tighten short availability and raise borrow rates,
often with 24-hour notice. The `DEFAULT_COST_PARAMS` assumes `short_borrow_bps_annual=50`, which is
a normal-market assumption for liquid large-caps. In a crisis, popular shorts can cost 500–2000 bps/year
(10–40× more). The strategy's net IC would deteriorate significantly on the short book.

**Mitigations:**
1. **Universe filter** — exclude stocks with market cap < $5B and short interest / float > 20%
   from the short book. These are the most borrow-constrained names. Apply this as a `min_mktcap`
   filter in the price data loading step.
2. **Regime-conditional cost params** — in crash/drawdown regimes (states 3–4 in the fitted model),
   switch to `AGGRESSIVE_COST_PARAMS` (`short_borrow_bps_annual=150`) for net IC computation.
   This gives a conservative stress test of strategy profitability.
3. **Position concentration cap** — cap any single short at 0.5% of gross book. This limits damage
   from any one name becoming a squeeze target.

**Implementation priority:** Regime-conditional cost params is a **Phase 2.5 stress test** that can
be added to `phase25_portfolio.ipynb` Section E.

---

### Failure Mode 4 — Correlation Collapse (MEDIUM RISK)

**What happens:** The diversification benefit between impl_82 (Calmar) and impl_53 (mean-reversion)
is estimated on normal market conditions. In tail events, cross-factor correlations spike toward 1.0 as
forced de-risking affects all names simultaneously. The two-factor blend may behave as a single factor
with no diversification in the exact scenario where diversification is most needed.

The OOS correlation between impl_82 and impl_53 is estimated from 1762 days that include COVID and
Ukraine but not a true quant meltdown. This estimate may be optimistic.

**Mitigations:**
1. **Stress-test the blend** — in `phase25_portfolio.ipynb`, add a scenario where all factor-factor
   correlations are doubled. Re-compute the combined strategy Sharpe and max drawdown under this
   "stressed correlation" assumption. Gate 3 should pass under stressed correlations too.
2. **Regime-conditional gross exposure** — in crash regimes (state 4 highest posterior), reduce
   combined gross exposure to 50%. The existing `regime_weights()` multiplier handles this if
   `ic_by_state` values for the combined signal are set to 0.0 for crash states.
3. **Monitor realized factor correlation rolling** — add a chart to `phase25_portfolio.ipynb` showing
   rolling 63-day correlation between impl_82 and impl_53. If correlation exceeds 0.6 in backtest,
   the diversification story needs revisiting.
4. **Add a third uncorrelated BHY factor** — the ultimate hedge against correlation collapse is a
   factor from a genuinely different information domain (e.g., a volatility or quality signal). This
   should be the primary outcome of Phase 1 continuation runs across all 5 domains.

**Implementation priority:** Stress correlation scenario is a **Phase 2.5 stress test** in the
notebook. A third uncorrelated BHY factor is a **Phase 1 continuation goal**.

---

## Per-Regime IC-Weighted Blending (Phase 2.5b Design)

*Decision: NOT Phase 3 (Phase 3 = crypto extension). This is a Phase 2.5 enhancement.*
*Approved by user 2026-04-18. To be implemented after Gate 3 passes with the base strategy.*

### The Problem with Current Blending

The current Phase 2.5 approach uses fixed IC-proportional weights × soft multiplier:

```python
# Fixed weights (never updated):
w_82 = IC_82 / (IC_82 + IC_53) ≈ 0.57
w_53 = IC_53 / (IC_82 + IC_53) ≈ 0.43

# Soft multiplier applied separately per factor
combined = w_82 * mult_82 * factor_82 + w_53 * mult_53 * factor_53
```

This is suboptimal: when the HSMM says we are in regime 2 (strong bull, IC_53=0.054), the blend
weight for impl_53 should be higher than 0.43. But the weights are static.

### The Proposed Solution: Posterior-Weighted IC Blending

At each date *t*, given the HSMM posterior π_t(s):

$$w_i(t) = \frac{\sum_s \pi_t(s) \cdot \text{IC}_i(s)}{\sum_j \sum_s \pi_t(s) \cdot \text{IC}_j(s)}$$

Where IC_i(s) is the IC of factor *i* in regime state *s* (from Phase 2 `regime_ic_decomposition()`).

**Properties of this formula:**
- When π_t assigns full probability to state 2 (impl_53's best state), w_53 → ~0.65 (up from 0.43)
- When regime is ambiguous (uniform posterior), weights fall back to IC-proportional → identical to
  current approach
- No new parameters to fit — uses only quantities already computed in Phase 2
- Continuous, no hard switching → minimal turnover impact

**Why this avoids the overfitting problems:**
- No parameters are fitted on OOS data — IC_i(s) values come directly from the Phase 2 decomposition
- Blending is fully determined by the posterior-weighted average of Phase 2 outputs
- The only "free choice" is using Phase 2 by_state IC values, which are already validated OOS

**Why NOT hard strategy selection:**
- Hard switching (use only impl_82 when state < 3, only impl_53 when state ≥ 3) would require
  fitting a threshold — one free parameter
- It also creates large one-day position changes when the state switches, amplifying turnover
- The soft weighting above achieves the same economic goal with zero additional parameters

### Alternative Overfitting-Resistant Approaches

1. **Shrinkage toward equal weights** — blend the IC-weighted blending above with equal-weighted:
   `w_final = λ × w_IC_posterior + (1-λ) × (1/N)` where λ = 0.5 (no fitting needed)
   Useful when there are few BHY factors (N=2–3); prevents extreme concentration.

2. **Walk-forward blend calibration** — rather than using full Phase 2 OOS IC_i(s) values, use
   a rolling 252-day window to estimate them. Re-estimate blend weights each quarter. This is a
   genuinely walk-forward extension but requires more code and will have high estimation variance
   with short regime windows.

3. **Bayesian shrinkage** — model per-regime IC as drawn from a Normal(μ_global, σ²) prior where
   μ_global = unconditional IC. Posterior estimate: regress per-regime IC toward unconditional
   by `n_days_in_regime / (n_days_in_regime + 100)`. Fully principled, no tuning.

### Implementation Plan (after Gate 3)

1. Add `regime_blend_weights(bhy_factor_dfs, decomps, proba_df)` function to `analyzer.py`
   - Input: dict of factor DFs, dict of RegimeDecomposition, posterior probability DataFrame
   - Output: time-indexed DataFrame of per-factor blend weights (dates × factors)
2. Use in `phase25_portfolio.ipynb` as an alternative blending cell (compare vs fixed weights)
3. Chart: show how blend weights change over time (stacked area chart by factor)
4. Gate 3+ check: does posterior blending improve Sharpe and reduce max drawdown vs fixed blend?

---

## Why 63 Days? (Reference Window)

The 63-day (≈ 1 quarter) window appears in multiple places:
- **HSMM re-fit step:** `hsmm_step=63` — HSMM state posteriors updated every quarter
- **ICIR rolling window:** `icir_window=63` in `WalkForwardValidator`
- **Slow ICIR computation:** `slow_icir_63` uses non-overlapping 63-day periods
- **impl_53 natural horizon:** IC builds monotonically to its peak at exactly 63 days

**Rationale:**
1. Aligns with the fiscal quarter — the natural rhythm of institutional portfolio reviews,
   earnings announcements, and analyst forecast revisions
2. Long enough to smooth daily IC noise (63 independent observations per quarter)
3. Short enough to be responsive to structural changes (4 updates per year)
4. impl_53's IC peak at 63 days is a post-hoc confirmation, not the reason the window was chosen

**Should it be changed?**

For the HSMM re-fit step: **no**. Regimes persist for weeks to months; a 21-day step would add
computation and refit noise without capturing faster regime transitions. A 126-day step would
miss regime changes in narrow windows.

For the ICIR rolling window: **context-dependent**.
- impl_82 (fast signal, h=1): a 21-day ICIR window would be more responsive to recent IC quality.
- impl_53 (slow signal, h=63): 63 days is the *minimum* meaningful window — fewer than 63 periods
  at h=63 days provides fewer than 1 non-overlapping observation. Do not reduce.
- For a combined strategy with both: keep 63 days as the standard to ensure both factors are
  evaluated on a comparable basis.

**Recommendation:** Do not change 63 days as the standard. If there is a concern about the
fast signal's responsiveness, add a secondary 21-day ICIR series alongside the standard 63-day
series for diagnostic purposes only (do not gate on it).
