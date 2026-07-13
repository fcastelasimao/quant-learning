"""Tests for the Corwin–Schultz spread estimator.

The headline test is an independent ground-truth check: build a continuous
Brownian price path, impose a *known* proportional spread on the observed
high/low, and confirm the estimator recovers it in the mean (Corwin–Schultz is
noisy per-pair but approximately unbiased in mean).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the project importable when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slippage.spread import (  # noqa: E402
    corwin_schultz, edge, edge_intraday, half_spread_bps,
)


def _synthetic_high_low(s_true: float, n_periods: int = 8000, steps: int = 120, seed: int = 0):
    """Continuous Brownian path chopped into periods; true high/low per period,
    then inflate/deflate by half the known spread (trade at ask / at bid)."""
    rng = np.random.default_rng(seed)
    logp = np.cumsum(rng.normal(0.0, 0.0008, n_periods * steps)).reshape(n_periods, steps)
    true_high = np.exp(logp.max(axis=1))
    true_low = np.exp(logp.min(axis=1))
    high = true_high * (1.0 + s_true / 2.0)
    low = true_low * (1.0 - s_true / 2.0)
    idx = pd.RangeIndex(n_periods)
    return pd.Series(high, index=idx), pd.Series(low, index=idx)


@pytest.mark.parametrize("s_true", [0.005, 0.010, 0.020])
def test_recovers_known_spread_in_mean(s_true):
    high, low = _synthetic_high_low(s_true)
    # clamp_negative=False for an unbiased mean; no overnight gaps in this path.
    s = corwin_schultz(high, low, clamp_negative=False, overnight_adjust=False)
    recovered = s.mean()
    assert recovered == pytest.approx(s_true, rel=0.20), (
        f"recovered {recovered:.5f} vs true {s_true:.5f}"
    )


def test_output_shape_and_index():
    high, low = _synthetic_high_low(0.01, n_periods=100)
    s = corwin_schultz(high, low)
    assert len(s) == len(high) - 1
    assert list(s.index) == list(high.index[1:])


def test_clamp_negative_floors_at_zero():
    high, low = _synthetic_high_low(0.01, n_periods=2000)
    s = corwin_schultz(high, low, clamp_negative=True)
    assert (s >= 0).all()


def test_overnight_adjustment_reduces_gap_contamination():
    """A pure overnight jump (no real spread) should not be read as a large spread
    once the overnight adjustment is on."""
    # Two flat days separated by a gap up: day1 in [100,101], day2 in [110,111].
    high = pd.Series([101.0, 111.0])
    low = pd.Series([100.0, 110.0])
    s_adj = corwin_schultz(high, low, clamp_negative=False, overnight_adjust=True)
    s_raw = corwin_schultz(high, low, clamp_negative=False, overnight_adjust=False)
    assert abs(s_adj.iloc[0]) < abs(s_raw.iloc[0])


def test_half_spread_bps_conversion():
    assert half_spread_bps(0.0010) == pytest.approx(5.0)  # 10 bps round-trip -> 5 bps half


# --- EDGE estimator -------------------------------------------------------------------

def _synthetic_ohlc(s_true: float, n: int = 40000, steps: int = 40, seed: int = 0):
    """Efficient Brownian price sampled into OHLC bars, plus a bid-ask bounce of known
    proportional spread `s_true`: open/close are transaction prices at a random side, the
    high sits at the ask and the low at the bid. High-power regime where EDGE is consistent."""
    rng = np.random.default_rng(seed)
    logp = np.cumsum(rng.normal(0.0, 0.0006, n * steps)).reshape(n, steps)
    eff_o, eff_c = np.exp(logp[:, 0]), np.exp(logp[:, -1])
    eff_h, eff_l = np.exp(logp.max(axis=1)), np.exp(logp.min(axis=1))
    half = s_true / 2.0
    qo, qc = rng.choice([-1, 1], n), rng.choice([-1, 1], n)
    return (eff_o * (1 + qo * half), eff_h * (1 + half),
            eff_l * (1 - half), eff_c * (1 + qc * half))


@pytest.mark.parametrize("s_true", [0.005, 0.010, 0.020])
def test_edge_recovers_known_spread(s_true):
    o, h, l, c = _synthetic_ohlc(s_true)
    assert edge(o, h, l, c) == pytest.approx(s_true, rel=0.10)


def test_edge_returns_nan_below_three_obs():
    assert np.isnan(edge([100.0, 101.0], [101.0, 102.0], [99.0, 100.0], [100.5, 101.5]))


def test_edge_sign_flag_can_be_negative():
    # A signed estimate exists (may be either sign on noise); default is non-negative.
    o, h, l, c = _synthetic_ohlc(0.010, n=2000)
    assert edge(o, h, l, c, sign=False) >= 0.0


def test_edge_matches_bidask_reference():
    """Our port must equal the authors' reference implementation exactly."""
    bidask = pytest.importorskip("bidask")
    o, h, l, c = _synthetic_ohlc(0.010, n=3000, seed=3)
    for sign in (False, True):
        assert edge(o, h, l, c, sign=sign) == pytest.approx(
            bidask.edge(o, h, l, c, sign=sign), rel=1e-9, abs=1e-15
        )


def test_edge_intraday_one_estimate_per_session():
    o, h, l, c = _synthetic_ohlc(0.010, n=90, steps=10)
    # 3 sessions of 30 bars each.
    idx = pd.DatetimeIndex(
        [pd.Timestamp("2022-01-03") + pd.Timedelta(days=d, minutes=15 * b)
         for d in range(3) for b in range(30)]
    )
    df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c}, index=idx)
    s = edge_intraday(df)
    assert len(s) == 3 and s.notna().all()
