"""
Regime transition velocity metrics.

These metrics quantify HOW FAST the regime is changing, not which regime we
are currently in.  High velocity signals an environment where the model is
rapidly shifting probability mass across states — often a precursor to or
co-incident with regime breaks.

Two primary metrics are provided:

    kl_divergence_velocity  — KL(π_t ‖ π_{t-window})
        Measures the "informational distance" between today's state
        distribution and the distribution `window` days ago.  KL divergence
        is asymmetric and sensitive to probability mass concentration.

    first_order_velocity    — ‖π_t - π_{t-window}‖₁
        L1 norm of the raw change in state-probability vector.  Simpler
        and more interpretable (bounded in [0, 2]), but less sensitive to
        changes in the tails of the distribution.

A smoothing utility ``smoothed_velocity`` reduces noise via an exponentially
weighted moving average with a configurable half-life.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# KL Divergence Velocity
# ---------------------------------------------------------------------------

def kl_divergence_velocity(
    proba_df: pd.DataFrame,
    window: int = 21,
) -> pd.Series:
    """
    KL divergence between today's state distribution and `window` days ago.

    High KL = fast regime change = high transition velocity.

    Formally:
        KL(P ‖ Q) = Σ_s  P(s) · log( P(s) / Q(s) )

    where P = π_t (today), Q = π_{t-window} (window days ago).
    An epsilon=1e-10 is added to all probabilities before taking logs to
    avoid log(0) or division-by-zero when a state has zero posterior mass.

    Args:
        proba_df: pd.DataFrame of shape (dates, n_states) from
                  ``RegimeHSMM.predict_proba``.  Rows should sum to 1.
        window:   Number of days to look back when computing the reference
                  distribution Q (default 21 ≈ 1 month).

    Returns:
        pd.Series indexed by the same dates as ``proba_df``.
        The first `window` values are NaN (no reference distribution yet).
        All valid values are non-negative (KL divergence ≥ 0).
    """
    eps = 1e-10
    arr = proba_df.values.astype(float)
    n = len(arr)
    result = np.full(n, np.nan, dtype=float)

    for t in range(window, n):
        P = arr[t] + eps          # today
        Q = arr[t - window] + eps  # window days ago
        # Normalise after adding epsilon so they remain distributions
        P = P / P.sum()
        Q = Q / Q.sum()
        kl = float(np.sum(P * np.log(P / Q)))
        result[t] = max(kl, 0.0)  # clip tiny negative due to floating-point

    return pd.Series(result, index=proba_df.index, name=f"kl_velocity_w{window}")


# ---------------------------------------------------------------------------
# First-Order Velocity
# ---------------------------------------------------------------------------

def first_order_velocity(
    proba_df: pd.DataFrame,
    window: int = 5,
) -> pd.Series:
    """
    L1 norm of the change in state probability vector over `window` days.

    Formally:
        v_t = Σ_s |π_t(s) - π_{t-window}(s)|

    This is bounded in [0, 2] for valid probability vectors (since the L1
    distance between two probability simplices is at most 2).

    Args:
        proba_df: pd.DataFrame of shape (dates, n_states) from
                  ``RegimeHSMM.predict_proba``.
        window:   Number of days over which to measure the change (default 5).

    Returns:
        pd.Series indexed by the same dates as ``proba_df``.
        The first `window` values are NaN.
        Valid values lie in [0, 2].
    """
    arr = proba_df.values.astype(float)
    n = len(arr)
    result = np.full(n, np.nan, dtype=float)

    for t in range(window, n):
        result[t] = float(np.sum(np.abs(arr[t] - arr[t - window])))

    return pd.Series(result, index=proba_df.index, name=f"l1_velocity_w{window}")


# ---------------------------------------------------------------------------
# Smoothed Velocity
# ---------------------------------------------------------------------------

def smoothed_velocity(
    velocity: pd.Series,
    halflife: int = 21,
) -> pd.Series:
    """
    Exponentially weighted moving average of a velocity series.

    Reduces the noise inherent in day-to-day fluctuations while preserving
    the medium-term trend in regime transition speed.

    Args:
        velocity: pd.Series of raw velocity values (from
                  ``kl_divergence_velocity`` or ``first_order_velocity``).
        halflife: EWM half-life in days (default 21 ≈ 1 month).
                  Higher = more smoothing, slower response.

    Returns:
        pd.Series of smoothed velocity, same index as `velocity`.
        NaN values at the start (inherited from `velocity`) are preserved.
    """
    smoothed = velocity.ewm(halflife=halflife, min_periods=1).mean()
    # Re-apply NaN mask from the raw series so we don't produce spurious
    # smoothed values before the first valid observation.
    smoothed[velocity.isna()] = np.nan
    smoothed.name = f"{velocity.name}_ema{halflife}" if velocity.name else f"smoothed_ema{halflife}"
    return smoothed
