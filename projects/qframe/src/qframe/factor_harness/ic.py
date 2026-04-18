"""
Information Coefficient (IC) computation.

All functions operate cross-sectionally: at each date, we rank assets by their
factor value and compute the Spearman rank correlation with forward returns.

Conventions:
- factor_df : pd.DataFrame, shape (dates x tickers), factor values
- returns_df: pd.DataFrame, shape (dates x tickers), simple returns
- horizon   : int, number of trading days forward to compute IC
- All DataFrames must be sorted by date (ascending) before being passed in.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from qframe.factor_harness import DEFAULT_OOS_START


# ---------------------------------------------------------------------------
# Cross-sectional IC
# ---------------------------------------------------------------------------

def compute_ic(
    factor_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    horizon: int = 1,
    min_stocks: int = 10,
    oos_start: str | None = None,
) -> pd.Series:
    """
    Compute the cross-sectional rank IC series (vectorized implementation).

    At each date t, IC(t) = Spearman(factor[t], returns[t → t+horizon]).
    Forward returns are aligned so no look-ahead bias is introduced.

    Uses a vectorized pandas rank approach (Spearman = Pearson of ranks),
    computing all dates simultaneously rather than one spearmanr call per date.
    This is ~50–100× faster than a per-date scipy loop on large universes.

    Args:
        factor_df:  (dates x tickers) factor values. NaN means stock excluded.
        returns_df: (dates x tickers) simple daily returns.
        horizon:    forward period in trading days.
        min_stocks: minimum number of valid stocks required to compute IC.
                    Returns NaN for that date if fewer are available.
        oos_start:  if provided, only compute IC from this date onwards.
                    Forward returns are still built from the full returns_df
                    to avoid boundary NaN artefacts; only the output is sliced.

    Returns:
        pd.Series indexed by date, IC value at each date.
        Dates where IC cannot be computed (insufficient stocks, NaN forward
        returns) are NaN — not dropped — so the caller can see the warm-up.
    """
    # Align columns and index
    shared_tickers = factor_df.columns.intersection(returns_df.columns)
    factor_df = factor_df[shared_tickers]
    returns_df = returns_df[shared_tickers]

    shared_dates = factor_df.index.intersection(returns_df.index)
    factor_df = factor_df.loc[shared_dates]
    returns_df = returns_df.loc[shared_dates]

    # Build forward returns on the full index first (avoids boundary NaNs if
    # oos_start is near the start of the data).
    # fwd_ret[t] = (1 + r[t+1]) * ... * (1 + r[t+horizon]) - 1
    log_ret = np.log1p(returns_df)
    fwd_log = log_ret.rolling(horizon, min_periods=horizon).sum().shift(-horizon)
    fwd_ret = np.expm1(fwd_log)

    # Optionally restrict evaluation to OOS dates
    if oos_start is not None:
        factor_df = factor_df.loc[oos_start:]
        fwd_ret = fwd_ret.loc[oos_start:]

    # --- Vectorized Spearman = Pearson of cross-sectional ranks ---------------
    # Mark positions where both factor and forward return are available
    valid = factor_df.notna() & fwd_ret.notna()
    n_valid = valid.sum(axis=1)

    # Mask invalid entries so they are excluded from ranking and correlation
    f = factor_df.where(valid)
    r = fwd_ret.where(valid)

    # Cross-sectional rank per date; NaN entries remain NaN (na_option='keep')
    f_rank = f.rank(axis=1, method="average", na_option="keep")
    r_rank = r.rank(axis=1, method="average", na_option="keep")

    # Pearson of ranks — skipna=True means NaN stocks are excluded from sums
    f_mean = f_rank.mean(axis=1)
    r_mean = r_rank.mean(axis=1)
    f_c = f_rank.sub(f_mean, axis=0)
    r_c = r_rank.sub(r_mean, axis=0)

    num   = (f_c * r_c).sum(axis=1)
    denom = np.sqrt((f_c ** 2).sum(axis=1) * (r_c ** 2).sum(axis=1))

    ic = (num / denom).rename(f"IC_h{horizon}")
    ic[n_valid < min_stocks] = np.nan
    ic[denom < 1e-10] = np.nan

    return ic


# ---------------------------------------------------------------------------
# ICIR
# ---------------------------------------------------------------------------

def compute_icir(
    ic_series: pd.Series,
    window: int = 63,
) -> pd.Series:
    """
    Rolling Information Coefficient Information Ratio.

    ICIR(t) = mean(IC[t-window+1 : t]) / std(IC[t-window+1 : t])

    Args:
        ic_series: pd.Series of IC values (NaN-safe — NaNs are dropped per window).
        window:    rolling window in trading days (default 63 ≈ 1 quarter).

    Returns:
        pd.Series of ICIR, same index as ic_series.
    """
    # Require a full window before reporting ICIR — avoids misleadingly extreme
    # values (and visible spikes in charts) during the first window warm-up period.
    roll_mean = ic_series.rolling(window, min_periods=window).mean()
    roll_std = ic_series.rolling(window, min_periods=window).std(ddof=1)
    icir = roll_mean / roll_std
    icir.name = f"ICIR_w{window}"
    return icir


# ---------------------------------------------------------------------------
# IC decay curve
# ---------------------------------------------------------------------------

def compute_ic_decay(
    factor_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    horizons: list[int] | None = None,
    min_stocks: int = 10,
    oos_start: str | None = None,
) -> pd.DataFrame:
    """
    Compute mean IC at multiple forward horizons (IC decay curve).

    Why the default horizons [1,5,10,21,63]?
    These correspond to natural holding periods (1d, 1w, 2w, 1m, 1q) and are
    logarithmically spaced. You CAN pass any set of horizons — e.g. range(1,64) —
    to get a smooth daily curve. Important caveat: cumulative returns at horizons
    h and h+1 overlap by h/(h+1) days, so adjacent IC values are highly correlated
    and NOT statistically independent. Use the dense curve for visualisation only;
    for formal significance testing use the pre-specified sparse horizons.

    Args:
        factor_df:  (dates x tickers) factor values.
        returns_df: (dates x tickers) simple daily returns.
        horizons:   list of forward periods in trading days.
                    Default: [1, 5, 10, 21, 63].
                    Pass list(range(1, 64)) for a smooth daily curve.
        min_stocks: forwarded to compute_ic.
        oos_start:  if provided, only evaluate IC from this date onwards.
                    Strongly recommended when called from the batch harness —
                    avoids computing IC on in-sample dates that aren't used.

    Returns:
        pd.DataFrame with columns ['horizon', 'mean_ic', 'icir', 'n_obs'].
        One row per horizon. Index = horizon.
    """
    if horizons is None:
        horizons = [1, 5, 10, 21, 63]

    rows = []
    for h in horizons:
        ic_series = compute_ic(
            factor_df, returns_df, horizon=h,
            min_stocks=min_stocks, oos_start=oos_start,
        )
        valid = ic_series.dropna()
        rows.append({
            "horizon": h,
            "mean_ic": valid.mean() if len(valid) > 0 else np.nan,
            "icir": valid.mean() / valid.std(ddof=1) if len(valid) > 1 else np.nan,
            "n_obs": len(valid),
        })

    return pd.DataFrame(rows).set_index("horizon")


def compute_slow_icir(
    factor_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    horizon: int = 63,
    oos_start: str = DEFAULT_OOS_START,
    min_periods: int = 8,
    min_stocks: int = 10,
) -> float:
    """
    Compute ICIR using non-overlapping `horizon`-day return windows.

    The standard ICIR uses daily IC which is heavily auto-correlated for slow
    signals (mean-reversion, skewness). This function uses only non-overlapping
    observations, giving an unbiased estimate at the factor's natural holding period.

    Example: for horizon=63, we get ~40 non-overlapping quarterly IC observations
    over a 7-year OOS period — small sample but statistically honest.

    Args:
        factor_df:   (dates x tickers) factor values.
        returns_df:  (dates x tickers) simple daily returns.
        horizon:     holding period in days (e.g. 21 for monthly, 63 for quarterly).
        oos_start:   OOS evaluation start date.
        min_periods: minimum non-overlapping observations for a valid ICIR estimate.
        min_stocks:  minimum valid stocks per observation date (should match the
                     value used in the main harness to keep thresholds consistent).

    Returns:
        float ICIR. Returns np.nan if insufficient data.
    """
    oos_factor = factor_df.loc[oos_start:]
    oos_dates = oos_factor.index

    if len(oos_dates) < horizon * min_periods:
        return np.nan

    # Select non-overlapping observation dates: every `horizon` steps
    sample_dates = oos_dates[::horizon]

    # Build forward returns at exactly `horizon` days — same NaN-propagation
    # logic as compute_ic: no fillna(0), require all horizon days present.
    log_ret = np.log1p(returns_df)
    fwd_log = log_ret.rolling(horizon, min_periods=horizon).sum().shift(-horizon)
    fwd_ret = np.expm1(fwd_log)

    ic_values = []
    for date in sample_dates:
        if date not in factor_df.index or date not in fwd_ret.index:
            continue
        f = factor_df.loc[date]
        r = fwd_ret.loc[date]
        mask = f.notna() & r.notna()
        if mask.sum() < min_stocks:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr, _ = spearmanr(f[mask], r[mask])
        ic_values.append(float(corr))

    if len(ic_values) < min_periods:
        return np.nan

    arr = np.array(ic_values)
    std = arr.std(ddof=1)
    return float(arr.mean() / std) if std > 0 else np.nan


# ---------------------------------------------------------------------------
# IC by calendar period (temporal stability diagnostic)
# ---------------------------------------------------------------------------

def compute_ic_by_period(
    factor_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    oos_start: str = DEFAULT_OOS_START,
    period_years: float = 2.0,
    horizon: int = 1,
    min_stocks: int = 10,
) -> pd.DataFrame:
    """
    Compute mean IC, std IC, ICIR, and t-stat for each consecutive OOS sub-period.

    This is a **temporal stability diagnostic**, not an alternative significance
    test.  It answers: "was the factor's IC consistent across all calendar
    periods, or did it only work in one particular market environment?"

    A factor that has IC ≈ 0.015 in every 2-year block is far more trustworthy
    than one with IC = 0.06 in 2018-2020 and IC = -0.01 in 2022-2024, even if
    the full-period average is the same.

    Does NOT improve the main t-statistic (splitting 1760 independent
    daily ICs into 5 blocks of 352 each reduces power). Use the full-period
    t-stat from multiple_testing.py for significance; use this for stability.

    Args:
        factor_df:    (dates x tickers) factor values.
        returns_df:   (dates x tickers) simple daily returns.
        oos_start:    OOS evaluation start date (same as walk-forward split).
        period_years: length of each sub-period in years (default 2).
        horizon:      forward return horizon in days.
        min_stocks:   passed to compute_ic.

    Returns:
        pd.DataFrame with columns:
            period_label  – e.g. "2018–2020"
            period_start  – pd.Timestamp
            period_end    – pd.Timestamp
            mean_ic       – mean IC over the period
            std_ic        – std of daily IC
            icir          – mean_ic / std_ic
            t_stat        – icir × √n_days  (within-period t, for relative comparison)
            n_days        – number of valid IC observations
    """
    # Full OOS IC series
    ic_series = compute_ic(factor_df, returns_df, horizon=horizon, min_stocks=min_stocks)
    ic_oos = ic_series.loc[oos_start:].dropna()

    if len(ic_oos) == 0:
        return pd.DataFrame()

    # Split into consecutive blocks of ~period_years × 252 trading days
    days_per_period = int(period_years * 252)
    dates = ic_oos.index
    rows = []

    start_idx = 0
    while start_idx < len(dates):
        end_idx = min(start_idx + days_per_period, len(dates))
        block = ic_oos.iloc[start_idx:end_idx]
        if len(block) < 20:          # skip tiny trailing block
            break

        mean_ic = float(block.mean())
        std_ic  = float(block.std(ddof=1))
        icir    = mean_ic / std_ic if std_ic > 0 else np.nan
        t_stat  = icir * np.sqrt(len(block)) if np.isfinite(icir) else np.nan

        start_yr = block.index[0].year
        end_yr   = block.index[-1].year
        label    = f"{start_yr}–{end_yr}" if start_yr != end_yr else str(start_yr)

        rows.append({
            "period_label": label,
            "period_start": block.index[0],
            "period_end":   block.index[-1],
            "mean_ic":      mean_ic,
            "std_ic":       std_ic,
            "icir":         icir,
            "t_stat":       t_stat,
            "n_days":       len(block),
        })
        start_idx = end_idx

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# IC half-life (exponential decay fit)
# ---------------------------------------------------------------------------

def estimate_ic_halflife(decay_df: pd.DataFrame) -> float:
    """
    Fit a simple exponential decay to the IC decay curve and return the
    estimated half-life in days.

    Args:
        decay_df: output of compute_ic_decay — index is horizon, column 'mean_ic'.

    Returns:
        Half-life in days. Returns np.nan if fit fails.
    """
    from scipy.optimize import curve_fit

    horizons = decay_df.index.values.astype(float)
    ic_vals = decay_df["mean_ic"].values

    mask = np.isfinite(ic_vals) & np.isfinite(horizons)
    if mask.sum() < 3:
        return np.nan

    def exp_decay(x, a, lam):
        return a * np.exp(-lam * x)

    try:
        ic0 = ic_vals[mask][0]
        p0 = [ic0, 0.01]
        popt, _ = curve_fit(exp_decay, horizons[mask], ic_vals[mask], p0=p0, maxfev=2000)
        lam = popt[1]
        if lam <= 0:
            return np.nan
        return float(np.log(2) / lam)
    except Exception:
        return np.nan
