"""
Hurst Exponent via Detrended Fluctuation Analysis (DFA).

DFA is preferred over the classical R/S method because R/S is biased for short
time series and inflates H estimates in the presence of short-range correlations.
DFA removes polynomial trends at each scale, making it robust for typical equity
return series lengths (100–3000 observations).

Interpretation:
    H > 0.5  — trending / persistent (momentum regime)
    H ≈ 0.5  — random walk (no exploitable autocorrelation)
    H < 0.5  — mean-reverting / anti-persistent

References:
    Peng et al. (1994) "Mosaic organization of DNA nucleotides", Physical Review E.
    Kantelhardt et al. (2001) "Detecting long-range correlations with DFA".
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# DFA core function
# ---------------------------------------------------------------------------

def _dfa_single(y: np.ndarray, scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute DFA fluctuation function F(s) for a profile array.

    Args:
        y:      Profile (cumulative sum of demeaned signal), 1-D float array.
        scales: 1-D array of window sizes (int values).

    Returns:
        Tuple (valid_scales, F_values) — only scales where at least one full
        non-overlapping window fits are included.
    """
    n = len(y)
    valid_scales: list[int] = []
    F_values: list[float] = []

    for s in scales:
        s = int(s)
        n_windows = n // s
        if n_windows < 1:
            continue

        residuals_sq: list[float] = []
        for w in range(n_windows):
            seg = y[w * s : (w + 1) * s]
            x = np.arange(s, dtype=float)
            # Fit linear trend (order-1 polynomial)
            coeffs = np.polyfit(x, seg, 1)
            trend = np.polyval(coeffs, x)
            residuals_sq.append(np.mean((seg - trend) ** 2))

        F = np.sqrt(np.mean(residuals_sq))
        valid_scales.append(s)
        F_values.append(F)

    return np.array(valid_scales, dtype=float), np.array(F_values, dtype=float)


# ---------------------------------------------------------------------------
# HurstEstimator class
# ---------------------------------------------------------------------------

class HurstEstimator:
    """
    Hurst exponent estimator using Detrended Fluctuation Analysis (DFA).

    DFA algorithm:
        1. Compute profile Y[k] = cumsum(r - mean(r))
        2. For each scale s in `scales`, split Y into non-overlapping windows
        3. Fit a linear trend in each window and compute residuals
        4. F(s) = sqrt(mean squared residual across all windows)
        5. Regress log(F(s)) on log(s) — slope is the Hurst exponent H

    Args:
        scales:      Sequence of window sizes to use (int). Defaults to powers
                     of 2 from 4 to 256.  Larger range gives more stable H
                     estimates but requires longer series.
        min_periods: Minimum number of observations required to compute H.
                     Returns NaN if the series is shorter.
    """

    _DEFAULT_SCALES = np.array([4, 8, 16, 32, 64, 128, 256], dtype=int)

    def __init__(
        self,
        scales: np.ndarray | list[int] | None = None,
        min_periods: int = 200,
    ) -> None:
        self.scales: np.ndarray = (
            np.asarray(scales, dtype=int)
            if scales is not None
            else self._DEFAULT_SCALES.copy()
        )
        self.min_periods = min_periods

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, returns: pd.Series) -> float:
        """
        Compute the Hurst exponent for a single return series.

        Args:
            returns: pd.Series of returns (any frequency). Must be numeric,
                     NaN values are dropped before computation.

        Returns:
            Hurst exponent H ∈ (0, 1), or np.nan if insufficient data.
        """
        r = returns.dropna().values.astype(float)
        if len(r) < self.min_periods:
            return np.nan
        return self._compute_hurst(r)

    def fit_rolling(self, returns: pd.Series, window: int = 252) -> pd.Series:
        """
        Compute rolling Hurst exponent with a fixed lookback window.

        Each date t uses the `window` observations ending at t (inclusive),
        so there is no look-ahead bias.

        Args:
            returns: pd.Series of returns, DatetimeIndex recommended.
            window:  Number of periods in each rolling window (default 252 ≈ 1 year).

        Returns:
            pd.Series of Hurst exponents, same index as `returns`.
            Values before the first full window (first `window - 1` dates) are NaN.
        """
        r_arr = returns.values.astype(float)
        n = len(r_arr)
        result = np.full(n, np.nan, dtype=float)

        for t in range(window - 1, n):
            seg = r_arr[t - window + 1 : t + 1]
            if np.isnan(seg).any():
                # Drop NaNs within window — if too few remain, skip
                seg = seg[~np.isnan(seg)]
            if len(seg) < self.min_periods:
                continue
            result[t] = self._compute_hurst(seg)

        return pd.Series(result, index=returns.index, name="hurst_dfa")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_hurst(self, r: np.ndarray) -> float:
        """
        Core DFA computation on a clean (NaN-free) 1-D return array.

        Returns:
            H estimate as float, or np.nan if regression fails / < 2 valid scales.
        """
        # Build profile: cumulative sum of demeaned returns
        y = np.cumsum(r - r.mean())

        # Restrict scales to those where at least one window fits
        usable_scales = self.scales[self.scales <= len(y) // 2]
        if len(usable_scales) < 2:
            return np.nan

        valid_scales, F = _dfa_single(y, usable_scales)
        if len(valid_scales) < 2:
            return np.nan

        # Guard against zero or negative F values before log
        pos_mask = F > 0
        if pos_mask.sum() < 2:
            return np.nan

        log_s = np.log(valid_scales[pos_mask])
        log_F = np.log(F[pos_mask])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coeffs = np.polyfit(log_s, log_F, 1)

        return float(coeffs[0])  # slope = H
