"""
Multiple testing correction for factor research.

When you test many factors on the same dataset, some will appear to pass
the IC gate by pure chance. This module implements the Benjamini-Hochberg-
Yekutieli (BHY) correction, which controls the False Discovery Rate (FDR)
under arbitrary dependence — appropriate here because factor signals are
correlated with each other.

Reference: Harvey, Liu & Zhu (2016), "...and the Cross-Section of Expected
Returns", Review of Financial Studies. They argue a t-statistic of at least
3.0 should be required for a single new factor test, and proportionally more
for multiple tests from the same data-mining exercise.

Usage:
    from qframe.factor_harness.multiple_testing import correct_ic_pvalues

    from qframe.knowledge_base.db import KnowledgeBase
    kb = KnowledgeBase('knowledge_base/qframe.db')
    results = kb.get_all_results()

    corrected = correct_ic_pvalues(results, alpha=0.05, n_oos_days=1762)
    print(corrected[['factor_name', 'ic', 't_stat', 'p_raw', 'p_bhy', 'bhy_significant']])
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats


# OOS period 2018-01-01 to 2024-12-30 ≈ 1762 trading days
_DEFAULT_OOS_DAYS = 1762


def compute_t_stat(ic: float, sharpe: float, n_oos_days: int = _DEFAULT_OOS_DAYS) -> float:
    """
    Compute the t-statistic for a mean IC estimate.

    SHARPE CONVENTION:
        sharpe_annual  = ICIR × √252   ← what the DB stores as `sharpe`; used HERE
        sharpe_per_period = ICIR        ← used by deflated_sharpe_ratio(); NOT here

    The IC Sharpe ratio is: sharpe_annual = (mean_IC / std_IC_daily) × √252
    So:  mean_IC / std_IC_daily = sharpe_annual / √252

    The t-stat for mean IC over N days:
        t = (mean_IC / std_IC_daily) × √N = (sharpe_annual / √252) × √N

    Args:
        ic:          Mean IC over OOS period.
        sharpe:      Annualised IC Sharpe ratio (= ICIR × √252).  This is the value
                     stored as `sharpe` in the knowledge-base `backtest_results` table.
        n_oos_days:  Number of OOS trading days.

    Returns:
        float: t-statistic (one-sided; positive = signal is positive)
    """
    if not math.isfinite(ic) or not math.isfinite(sharpe) or ic == 0:
        return 0.0
    return (sharpe / math.sqrt(252)) * math.sqrt(n_oos_days)


def compute_p_value(t_stat: float, n_oos_days: int = _DEFAULT_OOS_DAYS) -> float:
    """
    One-tailed p-value for H0: IC ≤ 0  vs  H1: IC > 0.

    Uses t-distribution with (n_oos_days - 1) degrees of freedom.

    Args:
        t_stat:      t-statistic from compute_t_stat().
        n_oos_days:  Degrees of freedom = n_oos_days - 1.

    Returns:
        float: p-value in [0, 1]. Smaller = stronger evidence IC > 0.
    """
    if not math.isfinite(t_stat):
        return 1.0
    return float(stats.t.sf(t_stat, df=n_oos_days - 1))  # sf = 1 - cdf (upper tail)


def bhy_correction(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Benjamini-Hochberg-Yekutieli (BHY) procedure for False Discovery Rate
    control under arbitrary (including positive) dependence.

    BHY is less conservative than Bonferroni and appropriate when factor
    signals are correlated (which they are — momentum variants are highly
    correlated).

    The correction factor c(m) = Σ_{i=1}^{m} 1/i (harmonic number) accounts
    for the dependence structure. Under independence, c(m) = 1 (standard BH).

    Args:
        p_values:  Array of raw p-values, one per test.
        alpha:     Desired FDR level (default 0.05 = 5% false discovery rate).

    Returns:
        Boolean array of the same length: True = significant after correction.
    """
    m = len(p_values)
    if m == 0:
        return np.array([], dtype=bool)

    # BHY correction factor (harmonic number)
    c_m = sum(1.0 / i for i in range(1, m + 1))

    # Sort p-values (keep track of original indices)
    order = np.argsort(p_values)
    sorted_p = p_values[order]

    # BHY threshold for rank k (1-indexed): p_(k) ≤ k / (m × c(m)) × α
    thresholds = np.arange(1, m + 1) / (m * c_m) * alpha

    # Find largest k where sorted_p[k-1] ≤ threshold[k-1]
    significant_sorted = sorted_p <= thresholds

    # All tests up to the last significant one are rejected
    if not significant_sorted.any():
        return np.zeros(m, dtype=bool)

    last_sig = np.where(significant_sorted)[0][-1]
    significant_sorted[:last_sig + 1] = True

    # Map back to original order
    result = np.zeros(m, dtype=bool)
    result[order] = significant_sorted
    return result


def bonferroni_correction(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Conservative Bonferroni correction: reject H_i if p_i ≤ α / m.
    Controls family-wise error rate (FWER) but is very conservative for
    correlated tests.

    Use BHY in practice — Bonferroni is shown here for comparison.
    """
    m = len(p_values)
    return p_values <= (alpha / m)


def compute_slow_t_stat(
    slow_icir: float,
    n_oos_days: int = _DEFAULT_OOS_DAYS,
    horizon: int = 63,
) -> float:
    """
    Correct t-statistic for a slow signal evaluated at non-overlapping windows.

    For a slow signal with holding period `horizon` days, the number of
    independent IC observations is N_windows = floor(n_oos_days / horizon).
    The t-stat is simply: slow_icir × √N_windows.

    This is the honest statistic — the daily t-stat formula is NOT valid
    for slow signals because consecutive daily ICs overlap by (horizon-1) days.

    Args:
        slow_icir:   slow_icir_63 or slow_icir_21 from the DB.
        n_oos_days:  Number of OOS trading days.
        horizon:     Holding period in days (21 or 63).

    Returns:
        float: t-statistic with (N_windows - 1) degrees of freedom.
    """
    if not math.isfinite(slow_icir):
        return 0.0
    n_windows = n_oos_days // horizon
    return slow_icir * math.sqrt(n_windows)


def correct_ic_pvalues(
    results: list[dict],
    alpha: float = 0.05,
    n_oos_days: int = _DEFAULT_OOS_DAYS,
    min_ic: float = 0.0,
) -> pd.DataFrame:
    """
    Apply BHY multiple testing correction to all backtest results.

    Filters to results with positive IC (we only test H1: IC > 0), then
    computes t-statistics, raw p-values, and BHY-corrected significance.

    Args:
        results:     List of dicts from kb.get_all_results().
        alpha:       FDR level for BHY (default 0.05).
        n_oos_days:  Number of OOS trading days (default 1762 for 2018–2024).
        min_ic:      Minimum IC to include in test (default 0.0 — all positive IC).

    Returns:
        pd.DataFrame sorted by IC descending, with columns:
            factor_name, ic, icir, sharpe, t_stat, p_raw, p_bhy,
            bonferroni_sig, bhy_significant, hlz_t3_sig
    """
    # Filter to positive-IC results with the required fields
    valid = [
        r for r in results
        if r.get("ic") is not None
        and r.get("sharpe") is not None
        and r["ic"] > min_ic
        and math.isfinite(r["ic"])
        and math.isfinite(r["sharpe"])
    ]

    if not valid:
        return pd.DataFrame()

    df = pd.DataFrame([{
        "result_id":   r.get("id"),
        "factor_name": r.get("factor_name") or f"impl_{r.get('implementation_id')}",
        "ic":          r["ic"],
        "icir":        r.get("icir"),
        "sharpe":      r["sharpe"],
        "slow_icir_63": r.get("slow_icir_63"),
        "passed_gate_raw": r.get("passed_gate", 0),
    } for r in valid])

    # Compute t-stats and p-values
    # For slow signals (slow_icir_63 available and positive), use the
    # non-overlapping t-stat which is the honest estimate for slow signals.
    # Note: use pd.notna() rather than a truthiness check — 0.0 is falsy
    # but would still correctly fall through to the fast formula via > 0.
    def _t_stat_row(row):
        _s63 = row.get("slow_icir_63")
        if _s63 is not None and pd.notna(_s63) and _s63 > 0:
            return compute_slow_t_stat(_s63, n_oos_days, horizon=63)
        return compute_t_stat(row["ic"], row["sharpe"], n_oos_days)

    df["t_stat"] = df.apply(_t_stat_row, axis=1)
    df["p_raw"] = df["t_stat"].apply(lambda t: compute_p_value(t, n_oos_days))

    # Multiple testing corrections
    p_arr = df["p_raw"].values

    df["bonferroni_sig"] = bonferroni_correction(p_arr, alpha)
    df["bhy_significant"] = bhy_correction(p_arr, alpha)

    # Harvey-Liu-Zhu (2016) threshold: t ≥ 3.0 for any single factor,
    # higher when many factors are tested. With m tests, HLZ recommend
    # t ≥ √(2 × ln(m)) as the asymptotic correct threshold.
    # For m=1, ln(1)=0, yielding t≥0 which is nonsensical — HLZ explicitly
    # state t≥3.0 as the floor for any single new factor test.
    m = len(df)
    hlz_t_threshold = max(3.0, math.sqrt(2 * math.log(m))) if m >= 2 else 3.0
    df["hlz_t_threshold"] = round(hlz_t_threshold, 3)
    df["hlz_sig"] = df["t_stat"] >= hlz_t_threshold

    # Adjusted IC threshold: what IC would be needed to pass BHY at current m?
    # t_needed = stats.t.isf(alpha / (m * c_m), df=n_oos_days - 1)  (approx)
    c_m = sum(1.0 / i for i in range(1, m + 1))
    t_bhy_threshold = stats.t.isf((alpha / (m * c_m)), df=n_oos_days - 1)
    df["bhy_t_threshold"] = round(float(t_bhy_threshold), 3)

    return df.sort_values("ic", ascending=False).reset_index(drop=True)


def deflated_sharpe_ratio(
    sharpe_obs: float,
    n_trials: int,
    t: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """
    Deflated Sharpe Ratio (DSR) — probability that the observed Sharpe Ratio
    is greater than the expected maximum Sharpe Ratio from independent trials.

    Reference: Bailey & López de Prado (2014), "The Deflated Sharpe Ratio:
    Correcting for Selection Bias, Backtest Overfitting, and Non-Normality".
    Journal of Portfolio Management 40(5).

    The DSR answers: "What fraction of the observed Sharpe is genuine after
    accounting for the fact that we searched over `n_trials` strategies?"

    Formula:
        SR_0* = E[max SR] ≈ ((1 − γ) × Z^{−1}(1 − 1/n) + γ × Z^{−1}(1 − 1/(n×e))) × √(1/T)
        where γ = Euler-Mascheroni constant ≈ 0.5772

        DSR = Φ( (SR_obs − SR_0*) × √T
                  × √(1 − ŝkew×SR_obs + (kurt−1)/4 × SR_obs²) )

    ⚠️  SHARPE CONVENTION — CRITICAL:
        `sharpe_obs` MUST be the **per-period (daily) SR = ICIR** (= mean_IC / std_IC).
        Do NOT pass in the annualised SR (= ICIR × √252).  The formula already
        multiplies by √T internally; passing the annualised figure inflates DSR by √252.
        Quick check: if your IC Sharpe is displayed as ~4.0 on the leaderboard
        (annualised), divide by √252 ≈ 15.87 before calling this function → ~0.25.

    Args:
        sharpe_obs:  Per-period (daily) Sharpe Ratio = ICIR = mean_IC / std_IC_daily.
                     **NOT the annualised SR** (do not multiply by √252 before calling).
        n_trials:    Number of strategies tried (= number of backtest results).
        t:           Number of independent observations (= OOS trading days).
        skewness:    Skewness of the strategy return series (default 0).
        kurtosis:    Kurtosis of the strategy return series (default 3 = Gaussian).

    Returns:
        float in [0, 1]: probability that SR_obs exceeds the selection-inflated
        expected maximum SR.  DSR > 0.95 is the conventional significance threshold.
        Returns NaN if inputs are degenerate.
    """
    if n_trials < 1 or t < 2 or not math.isfinite(sharpe_obs):
        return float("nan")

    n = max(n_trials, 2)
    euler_gamma = 0.5772156649

    # Expected maximum SR under independent trials (Bailey & López de Prado eq. 8)
    # Use the analytic approximation for the expected max of n half-normal variates
    z1 = stats.norm.ppf(1.0 - 1.0 / n)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n * math.e))
    sr_0_star = ((1.0 - euler_gamma) * z1 + euler_gamma * z2) / math.sqrt(t)

    # Non-normality adjustment (equation 9 in the paper)
    # Reduces SR by the amount attributable to fat tails / skew
    adj = math.sqrt(1.0 - skewness * sharpe_obs + (kurtosis - 1.0) / 4.0 * sharpe_obs ** 2)
    if adj <= 0:
        return float("nan")

    # DSR
    dsr_z = (sharpe_obs - sr_0_star) * math.sqrt(t) * adj
    return float(stats.norm.cdf(dsr_z))


def bhy_t_threshold(
    m: int,
    alpha: float = 0.05,
    n_oos_days: int = _DEFAULT_OOS_DAYS,
) -> float:
    """
    Return the minimum t-statistic required for significance under BHY correction,
    given that ``m`` positive-IC factors have been tested so far (including the
    current one).

    This is the online version of the BHY threshold used in the runtime gate:
    after each factor test, m increments by 1 so the bar rises monotonically.

    Formula: t_BHY = t-distribution quantile at level α / (m × c(m)), one-tailed.
    where c(m) = Σ_{i=1}^{m} 1/i (harmonic number for arbitrary dependence).

    Args:
        m:           Number of positive-IC tests (including the current one).
                     Must be ≥ 1.
        alpha:       FDR level (default 0.05).
        n_oos_days:  OOS trading days for the t-distribution df (default 1762).

    Returns:
        float: t-stat threshold.  Factor passes BHY gate if |t| ≥ this value.
    """
    m = max(1, int(m))
    c_m = sum(1.0 / i for i in range(1, m + 1))
    level = alpha / (m * c_m)
    return float(stats.t.isf(level, df=n_oos_days - 1))


def print_correction_summary(
    corrected_df: pd.DataFrame,
    n_oos_days: int = _DEFAULT_OOS_DAYS,
    alpha: float = 0.05,
) -> None:
    """Pretty-print the multiple testing correction results."""
    if corrected_df.empty:
        print("No results to display.")
        return

    m = len(corrected_df)
    c_m = sum(1.0 / i for i in range(1, m + 1))
    bhy_t = corrected_df["bhy_t_threshold"].iloc[0] if not corrected_df.empty else float("nan")
    hlz_t = corrected_df["hlz_t_threshold"].iloc[0] if not corrected_df.empty else float("nan")

    # Corrected IC threshold (approximate — assumes sharpe ∝ IC)
    # IC_needed ≈ bhy_t × √252 / √N_OOS × avg_std_IC
    print(f"\n{'='*70}")
    print(f"Multiple Testing Correction  (m={m} positive-IC factors, α={alpha})")
    print(f"{'='*70}")
    print(f"  BHY (FDR control):       t ≥ {bhy_t:.2f}  required for significance")
    print(f"  Harvey-Liu-Zhu (2016):   t ≥ {hlz_t:.2f}  (= √(2·ln({m})))")
    print(f"  Bonferroni (FWER):       t ≥ {float(stats.t.isf(alpha/m, df=n_oos_days-1)):.2f}  (most conservative)")
    print()

    cols = ["factor_name", "ic", "t_stat", "p_raw", "bhy_significant", "hlz_sig"]
    available = [c for c in cols if c in corrected_df.columns]
    print(corrected_df[available].to_string(index=False, float_format="{:.4f}".format))

    n_bhy = corrected_df["bhy_significant"].sum()
    n_hlz = corrected_df["hlz_sig"].sum()
    print(f"\n  Significant after BHY: {n_bhy}/{m}")
    print(f"  Significant after HLZ: {n_hlz}/{m}")
    print(f"{'='*70}\n")
