"""
Statistical cointegration analysis for candidate pairs.

Uses the Engle-Granger two-step method:
  1. OLS regression of log(base) on log(quote) to find the hedge ratio.
  2. ADF test on the OLS residuals (spread) to test for stationarity.
     If the spread is stationary, the pair is cointegrated.

Also computes the half-life of mean reversion via an AR(1) fit on the spread,
which tells us how quickly divergences are expected to revert — a key filter
for whether a signal is actually tradeable within our holding period limits.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import coint, adfuller
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    logger.error("statsmodels not installed. Run: pip install statsmodels")

from config import STRATEGY
from models import Candle, CointegrationResult


def _extract_log_closes(candles: list[Candle]) -> np.ndarray:
    closes = np.array([c.close for c in candles], dtype=float)
    return np.log(closes)


def _compute_half_life(spread: np.ndarray) -> float:
    """
    Estimate half-life of mean reversion from AR(1) fit on the spread.

    Fits: Δspread_t = phi * spread_{t-1} + epsilon
    Half-life = -log(2) / log(1 + phi)

    A negative phi means the spread is mean-reverting. A very small phi
    (close to 0) means very slow reversion (large half-life).
    """
    spread_lag = spread[:-1]
    spread_diff = np.diff(spread)
    spread_lag_c = sm.add_constant(spread_lag)
    try:
        model = sm.OLS(spread_diff, spread_lag_c).fit()
        phi = model.params[1]
        if phi >= 0:
            return float("inf")  # Not mean-reverting
        return -math.log(2) / math.log(1 + phi)
    except Exception:
        return float("inf")


def test_cointegration(
    base_candles: list[Candle],
    quote_candles: list[Candle],
    base_symbol: str,
    quote_symbol: str,
) -> Optional[CointegrationResult]:
    """
    Run the Engle-Granger cointegration test on a pair.

    Returns a CointegrationResult with is_cointegrated=True if the pair
    passes the p-value threshold and has a tradeable half-life.
    Returns None if statsmodels is not available.
    """
    if not HAS_STATSMODELS:
        return None

    if len(base_candles) != len(quote_candles):
        min_len = min(len(base_candles), len(quote_candles))
        base_candles = base_candles[-min_len:]
        quote_candles = quote_candles[-min_len:]

    log_base = _extract_log_closes(base_candles)
    log_quote = _extract_log_closes(quote_candles)

    # Step 1: OLS regression — log(base) = alpha + beta * log(quote) + epsilon
    X = sm.add_constant(log_quote)
    try:
        ols = sm.OLS(log_base, X).fit()
    except Exception as e:
        logger.warning(f"OLS failed for {base_symbol}/{quote_symbol}: {e}")
        return None

    intercept = ols.params[0]
    hedge_ratio = ols.params[1]
    spread = log_base - hedge_ratio * log_quote - intercept

    # Step 2: ADF test on the spread (residuals)
    try:
        _, p_value, _ = coint(log_base, log_quote)
    except Exception as e:
        logger.warning(f"Cointegration test failed for {base_symbol}/{quote_symbol}: {e}")
        return None

    # Step 3: Half-life
    half_life = _compute_half_life(spread)

    is_cointegrated = (
        p_value < STRATEGY.coint_pvalue_threshold
        and STRATEGY.min_half_life_hours <= half_life <= STRATEGY.max_half_life_hours
    )

    result = CointegrationResult(
        base=base_symbol,
        quote=quote_symbol,
        is_cointegrated=is_cointegrated,
        p_value=round(p_value, 4),
        hedge_ratio=round(hedge_ratio, 6),
        intercept=round(intercept, 6),
        half_life_hours=round(half_life, 1) if not math.isinf(half_life) else 9999.0,
        spread_mean=round(float(np.mean(spread)), 6),
        spread_std=round(float(np.std(spread)), 6),
    )

    status = "COINTEGRATED" if is_cointegrated else "rejected"
    logger.info(
        f"{base_symbol}/{quote_symbol}: {status} | "
        f"p={p_value:.4f} | hedge={hedge_ratio:.4f} | "
        f"half_life={result.half_life_hours:.1f}h"
    )
    return result


def compute_spread(
    base_candles: list[Candle],
    quote_candles: list[Candle],
    hedge_ratio: float,
    intercept: float,
) -> np.ndarray:
    """Compute the spread series given a fitted hedge ratio and intercept."""
    log_base = _extract_log_closes(base_candles)
    log_quote = _extract_log_closes(quote_candles)
    return log_base - hedge_ratio * log_quote - intercept


def rolling_zscore(spread: np.ndarray, window: int) -> np.ndarray:
    """
    Compute rolling z-score of the spread.

    For each point t, z_t = (spread_t - mean(spread[t-window:t])) / std(spread[t-window:t]).
    The first `window` values are NaN.
    """
    n = len(spread)
    z = np.full(n, np.nan)
    for i in range(window, n):
        window_slice = spread[i - window:i]
        mu = np.mean(window_slice)
        sigma = np.std(window_slice)
        if sigma > 1e-10:
            z[i] = (spread[i] - mu) / sigma
    return z
