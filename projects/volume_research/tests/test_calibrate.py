"""Tests for calibrate() — recover known σ / ADV / spread from synthetic OHLCV."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slippage import MarketParams, calibrate  # noqa: E402
from slippage.calibrate import _aggregate_half_spread  # noqa: E402


def _synthetic_daily(sigma_true=0.03, n=600, price0=100.0, vol=1e6, seed=0):
    """Daily bars with a known per-day return vol and a fixed $ADV (price·volume)."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, sigma_true, n)
    close = price0 * np.exp(np.cumsum(rets))
    idx = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame({"close": close, "volume": vol}, index=idx)


def _synthetic_15min(s_true=0.002, n_days=60, bars=26, steps=120, seed=1):
    """15-min OHLC across sessions with a known proportional spread baked in.

    Mirrors the validated generator in test_spread.py: a continuous Brownian path chopped
    into bars (true high/low), then inflated by half the known spread (ask) / deflated (bid).
    Open/close (a bid-ask bounce at a random side) are included so the EDGE path is testable;
    the high/low columns are unchanged, so the Corwin–Schultz tests are unaffected.
    """
    rng = np.random.default_rng(seed)
    rows, idx = [], []
    half = s_true / 2.0
    for d in range(n_days):
        day = pd.Timestamp("2022-01-03") + pd.Timedelta(days=d)
        logp = np.cumsum(rng.normal(0.0, 0.0008, bars * steps)).reshape(bars, steps)
        hi = np.exp(logp.max(axis=1)) * (1.0 + half)
        lo = np.exp(logp.min(axis=1)) * (1.0 - half)
        op = np.exp(logp[:, 0]) * (1.0 + rng.choice([-1, 1], bars) * half)
        cl = np.exp(logp[:, -1]) * (1.0 + rng.choice([-1, 1], bars) * half)
        for b in range(bars):
            idx.append(day + pd.Timedelta(minutes=570 + 15 * b))  # 09:30 + 15·b
            rows.append((op[b], hi[b], lo[b], cl[b]))
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"],
                        index=pd.DatetimeIndex(idx))


def test_adv_is_exact():
    daily = _synthetic_daily(vol=2e6, price0=50.0)
    cal = calibrate(daily, _synthetic_15min())
    assert cal.adv_usd == pytest.approx((daily["volume"] * daily["close"]).tail(504).mean())


def test_sigma_in_ballpark():
    cal = calibrate(_synthetic_daily(sigma_true=0.03), _synthetic_15min())
    assert cal.sigma_daily_bps == pytest.approx(0.03 * 1e4, rel=0.20)


def test_half_spread_recovered():
    cal = calibrate(_synthetic_daily(), _synthetic_15min(s_true=0.002))
    # s_true=0.002 proportional -> half-spread = 10 bps; CS is noisy, allow a wide band.
    assert cal.half_spread_bps == pytest.approx(10.0, rel=0.40)


def test_stress_is_worse_than_normal():
    cal = calibrate(_synthetic_daily(), _synthetic_15min())
    assert cal.sigma_stress_bps >= cal.sigma_daily_bps
    assert cal.adv_thin_usd <= cal.adv_usd
    assert isinstance(cal.normal, MarketParams) and isinstance(cal.stress, MarketParams)


def test_default_spread_method_is_cs():
    """Default must stay CS (EDGE is opt-in; see findings_10)."""
    default = calibrate(_synthetic_daily(), _synthetic_15min(s_true=0.002))
    cs = calibrate(_synthetic_daily(), _synthetic_15min(s_true=0.002), spread_method="cs")
    assert default.half_spread_bps == pytest.approx(cs.half_spread_bps)


def test_edge_spread_method_runs_and_is_finite():
    """The EDGE path is wired and returns a finite, non-negative half-spread. It may read low
    at 15-min (underpowered per-session) — that's the documented limitation, so we don't assert
    recovery here, only that the option works."""
    cal = calibrate(_synthetic_daily(), _synthetic_15min(s_true=0.002), spread_method="edge")
    assert np.isfinite(cal.half_spread_bps) and cal.half_spread_bps >= 0.0


def test_calibrate_rejects_unknown_spread_method():
    with pytest.raises(ValueError):
        calibrate(_synthetic_daily(), _synthetic_15min(), spread_method="bogus")


# --- NaN / data-robustness guard (_aggregate_half_spread) --------------------------------

def test_aggregate_half_spread_normal():
    # S=0.0002 -> 1.0 bps half; S=0.0001 -> 0.5 bps half; mean = 0.75.
    assert _aggregate_half_spread(pd.Series([0.0002, 0.0001]), "cs") == pytest.approx(0.75)


def test_aggregate_half_spread_all_nan_raises():
    """The footgun: max(nan, 0.0) == nan. Must raise, not silently poison MarketParams."""
    with pytest.raises(ValueError):
        _aggregate_half_spread(pd.Series([np.nan, np.nan, np.nan]), "cs")


def test_aggregate_half_spread_empty_raises():
    with pytest.raises(ValueError):
        _aggregate_half_spread(pd.Series([], dtype=float), "edge")


def test_aggregate_half_spread_negative_floored_not_nan():
    # Negative estimates floor at 0 (not NaN); a valid all-negative series -> 0.0, no raise.
    assert _aggregate_half_spread(pd.Series([-0.0002, -0.0001]), "cs") == 0.0


def test_aggregate_half_spread_warns_on_high_nan_fraction():
    s = pd.Series([0.0002] + [np.nan] * 4)  # 80% NaN
    with pytest.warns(UserWarning):
        val = _aggregate_half_spread(s, "edge")
    assert val == pytest.approx(1.0)  # mean of the one finite estimate


def test_aggregate_half_spread_no_warn_on_low_nan_fraction():
    import warnings as _w
    s = pd.Series([0.0002, 0.0001, np.nan])  # 33% NaN
    with _w.catch_warnings():
        _w.simplefilter("error")  # any warning becomes an error
        assert _aggregate_half_spread(s, "cs") == pytest.approx(0.75)
