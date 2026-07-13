"""MarketState — the client's "depending on the state of the market / market volume" input.

Three reference curves, measured once from history by `research/predictor/01_market_state`
(intraday volume-share profile, intraday spread curve, volatility-regime tercile bounds), plus
one runtime function, `estimate_state`, that combines them with a live snapshot (a trailing
window of recent 1-min returns, the day's trailing volume EWMA, and an optional live spread
override) into a `MarketState` for one `(ticker, timestamp)`.

Pure functions / small frozen dataclasses; no IO. The reference curves are plain dicts keyed by
"HH:MM" bin-start labels (session bins, 09:30..15:45) so they round-trip through CSV/JSON easily.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VolumeProfile:
    """Per-15-min-bin historical share of the day's volume (fraction, sums to ~1 over a session).

    `bin_share_mean` is the point estimate used for the volume forecast; `bin_share_p10`/`p20`
    are the thin-tape floors (how low a bin's share can plausibly go), used to flag "thin tape"
    risk separately from the central forecast.
    """
    bin_share_mean: dict[str, float]
    bin_share_p10: dict[str, float]
    bin_share_p20: dict[str, float]


@dataclass(frozen=True)
class SpreadCurve:
    """Per-15-min-bin half-spread (bps), historical (CS-15min) or live-measured (NBBO)."""
    bin_half_spread_bps: dict[str, float]


@dataclass(frozen=True)
class VolRegimeBounds:
    """calm/normal/stress tercile cutpoints on trailing-20-day realized daily vol, in bps.

    Mirrors `research/03_delay_cost/build_03_delay_cost.py::vol_regime` exactly (same 20-day
    window, same full-sample tercile split) so a regime label here means the same thing as a
    regime label in S03's tables.
    """
    q1_bps: float
    q2_bps: float


@dataclass
class MarketState:
    """The predictor/scheduler's market-state input for one (ticker, timestamp)."""
    ts: pd.Timestamp
    symbol: str
    bin_label: str                  # "HH:MM" — which 15-min session bin `ts` falls in
    expected_interval_volume: float  # forecast shares in this 15-min bin
    thin_volume_p10: float          # p10/p20 floors for the same bin (thin-tape detector)
    thin_volume_p20: float
    sigma_now_bps: float            # trailing realized vol, annualized to daily-bps-equivalent units
    regime: str                     # "calm" | "normal" | "stress"
    spread_bps: float               # half-spread for this bin (live override if supplied, else curve)


def session_bin_label(ts: pd.Timestamp, *, bin_minutes: int = 15) -> str:
    """Map a timestamp to its 15-min session-bin label, e.g. 09:37 -> "09:30"."""
    floored_minute = (ts.minute // bin_minutes) * bin_minutes
    return f"{ts.hour:02d}:{floored_minute:02d}"


def sigma_now_bps(recent_1min_returns: pd.Series, *, minutes_per_day: int = 390) -> float:
    """Trailing realized vol from 1-min log returns, scaled to a daily-bps-equivalent.

    `recent_1min_returns` need not span a full day — scaling by sqrt(minutes_per_day / n)
    annualizes whatever trailing window is supplied to the same daily-vol units used
    throughout `calibrate.py` / `MarketParams.sigma_daily_bps`.
    """
    r = np.asarray(recent_1min_returns, float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return float("nan")
    std_1min = float(np.std(r, ddof=1))
    return std_1min * np.sqrt(minutes_per_day) * 1e4


def classify_regime(sigma_bps: float, bounds: VolRegimeBounds) -> str:
    """calm/normal/stress from a sigma reading against the fixed S03-style tercile bounds."""
    if not np.isfinite(sigma_bps):
        return "normal"   # neutral fallback when the trailing window is too short to estimate
    if sigma_bps <= bounds.q1_bps:
        return "calm"
    if sigma_bps <= bounds.q2_bps:
        return "normal"
    return "stress"


def estimate_state(ts: pd.Timestamp, symbol: str, *, recent_1min_returns: pd.Series,
                   trailing_daily_volume: float, profile: VolumeProfile,
                   spread_curve: SpreadCurve, vol_bounds: VolRegimeBounds,
                   live_spread_bps: float | None = None,
                   minutes_per_day: int = 390) -> MarketState:
    """Combine the reference curves + a live snapshot into one `MarketState`.

    Parameters
    ----------
    ts : the decision timestamp (session-local).
    recent_1min_returns : a trailing window of 1-min log returns ending at (or just before) `ts`
        — as short or long as the caller has available; longer windows are more stable, shorter
        windows react faster to a regime shift intraday.
    trailing_daily_volume : an EWMA (or any smoothed estimate) of recent daily total volume,
        evaluated strictly before `ts`'s session — the multiplier applied to the bin-share
        profile to get an absolute volume forecast.
    live_spread_bps : if supplied (e.g. a fresh Alpaca NBBO read), overrides the historical
        `spread_curve` for this bin — the live-facing preference noted in the predictor plan.
    """
    label = session_bin_label(ts)
    share = profile.bin_share_mean.get(label, float("nan"))
    p10 = profile.bin_share_p10.get(label, float("nan"))
    p20 = profile.bin_share_p20.get(label, float("nan"))
    sigma = sigma_now_bps(recent_1min_returns, minutes_per_day=minutes_per_day)
    regime = classify_regime(sigma, vol_bounds)
    spread = (live_spread_bps if live_spread_bps is not None
              else spread_curve.bin_half_spread_bps.get(label, float("nan")))
    return MarketState(
        ts=ts, symbol=symbol, bin_label=label,
        expected_interval_volume=share * trailing_daily_volume,
        thin_volume_p10=p10 * trailing_daily_volume,
        thin_volume_p20=p20 * trailing_daily_volume,
        sigma_now_bps=sigma, regime=regime, spread_bps=spread,
    )
