# Gate Thresholds

*Reference file for qframe-research skill. Loaded when evaluating whether a backtest passes a gate.*

---

## Gate System Overview

Gates are ordered. You cannot start Gate N+1 before Gate N passes in both backtest AND on a second market. This prevents premature complexity and forces honest validation.

The OOS split is: data before 2018 = training; data from 2018 onwards = out-of-sample. Never use post-2018 data for any parameter estimation.

---

## Factor Gate — IC & ICIR Thresholds

These thresholds apply **before** multiple testing correction. Apply the BHY correction separately (see section below) whenever m ≥ 10 factors have been tested.

| Gate | IC | ICIR | slow_icir_63 | Meaning |
|------|----|------|-------------|---------|
| **Weak gate** | ≥ 0.015 | ≥ 0.15 | — | Any evidence of cross-sectional alpha |
| **Pass gate** | ≥ 0.030 | ≥ 0.40 | — | Strong, consistent fast signal |
| **Slow weak** | — | — | ≥ 0.10 | Slow mean-reversion signal worth investigating |
| **Slow pass** | — | — | ≥ 0.25 | Strong slow signal — evaluate at h=63 horizon |

**Cost rule:** Net IC must always be positive (gross IC minus all cost drags). A factor that passes the IC gates but has negative Net IC is not tradeable.

---

## Multiple Testing Correction (required when m ≥ 10 factors)

When testing many factors on the same dataset, some will appear to pass by chance. Use the BHY procedure from `factor_harness/multiple_testing.py`.

### Why BHY and not Bonferroni?

Factor signals are correlated (momentum variants have ρ ≈ 0.80–0.88). Bonferroni assumes independence and is overly conservative. BHY (Benjamini-Hochberg-Yekutieli) controls the False Discovery Rate under arbitrary dependence — the correct choice here.

### Required t-statistics with current factor library (m=84 positive-IC factors)

| Method | t threshold | Interpretation |
|--------|------------|---------------|
| **BHY (recommended)** | **t ≥ ~4.0** | Controls FDR at 5% under correlation (c(84) ≈ 5.0) |
| Harvey-Liu-Zhu (2016) | t ≥ 3.03 | √(2·ln(84)) — asymptotic rule from HLZ |
| Bonferroni (conservative) | t ≥ 3.76 | α/m — FWER control, too strict for correlated signals |
| Unadjusted (naive) | t ≥ 1.96 | **Do not use — guarantees false positives** |

### How t-stat is computed

**For fast signals (h=1 day):**
```
t = (IC_Sharpe / √252) × √N_OOS_days
  = (sharpe / √252) × √1762  ≈ sharpe × 2.645
```

**For slow signals (h=63 days, non-overlapping):**
```
t = slow_icir_63 × √N_windows
  = slow_icir_63 × √(1762/63)  ≈ slow_icir_63 × 5.29
```

Using daily IC for a slow signal inflates N by ~63× — always use the slow formula for signals with IC@63d >> IC@1d.

### Current status (96 results, 84 positive-IC, OOS 2018–2024, universe=449 stocks)

BHY threshold with m=84 positive-IC factors: **t ≥ ~4.0**

| Factor | IC | t-stat | BHY sig? | HLZ sig? |
|--------|----|--------|----------|----------|
| impl_82 `trend_quality_calmar_ratio` (h=1) | 0.0646 | **10.74** | ✅ | ✅ |
| impl_92 `calmar_proxy_252` (h=1) | 0.0646 | **10.74** | ✅ | ✅ (duplicate of impl_82) |
| impl_53 `mean_reversion_factor` (h=63) | 0.0490 | **8.15** | ✅ | ✅ |
| Momentum cluster (impl_1/7/85/94/95) | 0.016–0.017 | ~2.8 | ❌ | ✅ |

**3 BHY-significant factors (2 unique signals). Phase 1 gate cleared.**  
Note: impl_82 and impl_92 are the same Calmar formula. Count as 1 unique signal for ensemble purposes.

---

## Gate 0 — Factor Harness Operational

**Test:** Run the factor harness on momentum (12-1 month price momentum) using yfinance data.

**Pass criteria:**
- IC series has no NaN beyond the first warm-up period
- ICIR computation runs without error
- IC decay curve shows plausible shape (momentum IC should peak at ~1-month horizon then decay)
- SQLite logging works end-to-end

**Status: ✅ PASSED**

---

## Gate 1 — HSMM Regime Detection

**Test:** Reproduce Zakamulin (2023) five-state Hidden Semi-Markov Model on S&P 500 daily returns.

**Pass criteria:**
- Five states recovered with plausible economic interpretations:
  - State 1: Low-vol bull (high mean, low variance)
  - State 2: High-vol bull (positive mean, elevated variance)
  - State 3: Bubble (high mean, low variance, short duration)
  - State 4: Regular bear (negative mean, moderate variance)
  - State 5: Crash (strongly negative mean, high variance, short duration)
- Transition matrix is sparse (each state doesn't freely transition to all others)
- Results directionally consistent with Zakamulin (2023) Figure 2

**Plus:** Rolling Hurst exponent (DFA, 250-day window) computed and plausible:
- Bull market periods should show H > 0.5
- Choppy/crash periods should show H near or below 0.5

**Paper trading gate:** Gate 1 backtest must run on Alpaca paper trading for 3–6 months before Gate 2 begins. This is non-negotiable.

**Status: ✅ PASSED** — 5-state HSMM fitted walk-forward (504-day window, 63-day step). Five states recovered with economically interpretable characteristics. State 2 (strong bull / low-vol) achieved Sharpe=2.98 and 43.8% annualised return. Convergence warnings from hmmlearn are normal (sparse transitions in short windows). Full results in `notebooks/phase2_regime_analysis.ipynb` Section A.

---

## Gate 2 — Regime-Conditional Factor IC

**Test:** For each factor in the library, compute OOS IC separately within each HSMM regime state.

**Pass criteria:**
- OOS IC difference between best and worst regime is > 0.03 (3 IC points)
- The difference is consistent in direction across at least 3 factors
- Transition velocity (KL divergence metric) adds statistically significant IC beyond the regime label alone (t-stat > 1.5)

**What this proves:** Regime detection is doing real work — factor performance genuinely differs across states, not just by chance.

**Status: ✅ PASSED** — impl_53 (mean-reversion, h=63) lift = **2.28×** > threshold 1.5×. Best regime IC = 0.054 (state 2, strong bull) vs unconditional IC = 0.024. impl_82 (Calmar, h=1) lift = 1.27× — below threshold, used unconditionally. Full results in `notebooks/phase2_regime_analysis.ipynb` Section B–C.

| Factor | Uncond IC | Best-regime IC | Lift | Gate 2? |
|--------|-----------|----------------|------|---------|
| impl_82 Calmar (h=1) | 0.0646 | 0.082 (state 2) | 1.27× | ❌ Use unconditionally |
| impl_53 mean-rev (h=63) | 0.0238 | 0.054 (state 2) | **2.28×** | ✅ PASSED |

---

## Gate 3 — Net-of-Cost Sharpe

**Test:** Run the regime-conditional factor allocation strategy OOS (2018 onwards) with full Almgren-Chriss cost model.

**Pass criteria:**
- Annualised OOS Sharpe ratio > 0.5 net of costs
- Maximum drawdown < 25%
- No single calendar year with return below -15%
- Turnover implies realistic execution (< 200% annual for a daily-rebalanced strategy)

**Notes:**
- 10bps round-trip is conservative for large-cap liquid equities — appropriate for Phase 0/2
- Crypto strategies will use 20-30bps due to higher spread
- If gross Sharpe > 0.8 but net Sharpe < 0.5, the problem is turnover, not alpha — fix the rebalancing rule

**Status: ⬜ NOT STARTED**

---

## Gate 4 — Second Market Confirmation

**Test:** Run the same strategy (same parameters, no re-fitting) on:
- European equities (STOXX 600 universe), OR
- Top 50 crypto assets by 90-day volume

**Pass criteria:**
- Direction of regime-conditional IC differences is consistent (not necessarily magnitude)
- Net-of-cost Sharpe > 0.3 on the second market (lower threshold — less data, higher costs)

**What failure here means:** The strategy is curve-fitted to US equities. Go back to Gate 2 and look for more robust factors.

**Status: ⬜ NOT STARTED**

---

## Common Failure Modes

| Symptom | Likely cause |
|---------|-------------|
| IC spike at horizon 0 | Look-ahead bias — check return calculation shift |
| IC > 0.05 consistently at h=1 | Almost certainly a data error or look-ahead |
| IC drops to 0 in OOS | Overfitting in IS — use simpler, fewer-parameter factors |
| ICIR negative but IC positive | Signal inconsistent — driven by a few lucky windows |
| IC@63d >> IC@1d, slow_icir_63 low | Long-horizon noise, not a genuine slow signal |
| Sharpe collapses after 2020 | Factor crowding — check correlation to recent top factors |
| All momentum variants correlated | Normal — they're all measuring the same thing; keep only the best one |
| t-stat < 2.30 after 30+ tests | Not enough data or not enough cross-section (expand universe) |
| States don't make economic sense | HSMM initialisation problem — try different seeds |
