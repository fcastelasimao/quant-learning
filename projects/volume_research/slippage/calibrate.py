"""Calibrate MarketParams (σ, ADV, half-spread) for a ticker from its own OHLCV.

This is the only part of the library that touches data — and it takes DataFrames, not a DB
handle, so the library stays portable (the caller brings their loader). It encodes the
measurement choices locked in Blocks 1 & 4:

  - σ      : 20-day rolling realized vol of daily returns. The rolling series is right-skewed,
             so we return two central summaries and let the caller choose:
               * `sigma_daily_bps` = window **mean** — the expected-cost / time-average moment.
                 Impact cost per day ≈ Y·σ_t·√(Q/V), so the average cost over many trades scales
                 with mean(σ_t); this is the default for a cost model, and what `normal` uses.
               * `sigma_daily_median_bps` = window **median** — the typical-day moment (cleaner
                 regime split, since the p90 stress figure already carries the tail).
             stress = 90th percentile (vol clusters; stress is when you most want to trade).
  - $ADV   : daily (volume × close), window mean; thin = 10th percentile (low-volume day).
  - spread : 15-min Corwin–Schultz half-spread, **aggregate clamped at 0** (not per-pair —
             per-pair clamping biases the mean up).

Returns both a `normal` and a `stress` MarketParams so the capacity work can show both.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .cost import MarketParams
from .spread import corwin_schultz_intraday, edge_intraday, half_spread_bps

WINDOW_DAYS = 504        # ~2 trading years
VOL_LOOKBACK = 20        # days in the realized-vol window
STRESS_Q = 0.90          # σ stress percentile
THIN_Q = 0.10            # ADV thin percentile
NAN_WARN_FRAC = 0.50     # warn if more than this fraction of spread estimates are NaN


def _aggregate_half_spread(s: pd.Series, method: str) -> float:
    """Aggregate the per-window spread series into one half-spread (bps), robustly.

    The clamp `max(..., 0)` floors negative estimates at zero — but it does NOT protect
    against NaN (`max(nan, 0.0)` returns nan), so an empty/all-NaN series would silently
    poison `MarketParams.half_spread_bps`. Guard that explicitly: a high NaN fraction warns,
    and a non-finite aggregate raises rather than flowing a nan into the cost model.
    """
    hs = half_spread_bps(s)
    n = int(len(hs))
    n_valid = int(np.isfinite(hs).sum()) if n else 0
    if n == 0 or n_valid == 0:
        raise ValueError(
            f"spread_method={method!r} produced no valid spread estimates "
            f"({n_valid}/{n} finite) — check the intraday data (need ≥3 bars/session for EDGE, "
            f"≥2 for CS; finite, positive high/low[/open/close])."
        )
    if (n - n_valid) / n > NAN_WARN_FRAC:
        warnings.warn(
            f"spread_method={method!r}: {n - n_valid}/{n} spread estimates are NaN "
            f"({100 * (n - n_valid) / n:.0f}%) — the half-spread may be unreliable.",
            stacklevel=2,
        )
    half = float(max(np.nanmean(hs), 0.0))
    if not np.isfinite(half):
        raise ValueError(f"spread_method={method!r} aggregated to a non-finite half-spread.")
    return half


@dataclass
class Calibration:
    """Calibrated inputs for one instrument, normal and stress regimes."""
    normal: MarketParams
    stress: MarketParams
    sigma_daily_bps: float          # window mean — expected-cost moment (default, used by `normal`)
    sigma_daily_median_bps: float   # window median — typical-day moment (sensitivity)
    sigma_stress_bps: float
    adv_usd: float
    adv_thin_usd: float
    half_spread_bps: float


def calibrate(daily: pd.DataFrame, intraday_15min: pd.DataFrame, *,
              window_days: int = WINDOW_DAYS, minutes_per_day: int = 390,
              spread_method: str = "cs") -> Calibration:
    """Measure MarketParams from a ticker's daily OHLCV and 15-min OHLC.

    Parameters
    ----------
    daily : DataFrame
        Daily bars with 'close' and 'volume' columns, chronological. The most recent
        `window_days` rows are used.
    intraday_15min : DataFrame
        15-min bars, DatetimeIndex. For `spread_method="cs"` (default) only 'high'/'low' are
        needed; for `spread_method="edge"` also pass 'open'/'close'. Applied within each session.
    window_days : int
        Daily lookback for σ and ADV (default ~2y).
    spread_method : {"cs", "edge"}
        Spread estimator. **"cs" (default)** = Corwin–Schultz (the validated 15-min floor).
        "edge" = the EDGE estimator (Ardia–Guidotti–Kroencke) — available but **not recommended
        at 15-min**: it is underpowered per-session and overnight-gap–sensitive at this resolution
        (see findings_10). Its domain is daily bars / long gap-free samples.

    Returns
    -------
    Calibration
        `.normal` and `.stress` MarketParams plus the raw measured scalars.
    """
    d = daily.tail(window_days)
    ret = d["close"].pct_change()
    realized = ret.rolling(VOL_LOOKBACK).std().dropna()
    sigma = float(realized.mean() * 1e4)            # expected-cost moment (default)
    sigma_median = float(realized.median() * 1e4)   # typical-day moment (sensitivity)
    sigma_stress = float(realized.quantile(STRESS_Q) * 1e4)

    dollar_vol = d["volume"] * d["close"]
    adv = float(dollar_vol.mean())
    adv_thin = float(dollar_vol.quantile(THIN_Q))

    if spread_method == "cs":
        s = corwin_schultz_intraday(intraday_15min[["high", "low"]], clamp_negative=False)
    elif spread_method == "edge":
        s = edge_intraday(intraday_15min, clamp_negative=False)
    else:
        raise ValueError(f"spread_method must be 'cs' or 'edge', got {spread_method!r}")
    half_spread = _aggregate_half_spread(s, spread_method)

    normal = MarketParams(sigma_daily_bps=sigma, adv_usd=adv,
                          half_spread_bps=half_spread, minutes_per_day=minutes_per_day)
    stress = MarketParams(sigma_daily_bps=sigma_stress, adv_usd=adv_thin,
                          half_spread_bps=half_spread, minutes_per_day=minutes_per_day)
    return Calibration(normal=normal, stress=stress,
                       sigma_daily_bps=sigma, sigma_daily_median_bps=sigma_median,
                       sigma_stress_bps=sigma_stress,
                       adv_usd=adv, adv_thin_usd=adv_thin, half_spread_bps=half_spread)
