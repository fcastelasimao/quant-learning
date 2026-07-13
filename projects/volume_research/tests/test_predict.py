"""Tests for slippage/predict.py — predict_slippage().

Per the plan's spec: components match their source stages at reference points; band ordering
(p50 < p90 < p95); order-type monotonicity (chase >= cross in mean).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slippage.impact import impact_bps  # noqa: E402
from slippage.predict import (  # noqa: E402
    predict_slippage, _chase_drag_bps, _timing_sigma_bps,
    Y_BAND, P90_TIMING_MULT, P95_TIMING_MULT,
)
from slippage.state import (  # noqa: E402
    MarketState, VolumeProfile, SpreadCurve, VolRegimeBounds, estimate_state,
)


def _state(symbol="TQQQ", sigma_bps=300.0, spread_bps=0.74, bin_share=0.05):
    rng = np.random.default_rng(0)
    profile = VolumeProfile(bin_share_mean={"10:00": bin_share}, bin_share_p10={"10:00": bin_share / 2},
                            bin_share_p20={"10:00": bin_share * 0.7})
    spread = SpreadCurve(bin_half_spread_bps={"10:00": spread_bps})
    bounds = VolRegimeBounds(q1_bps=250.0, q2_bps=358.0)
    # scale the synthetic return window's std to land near sigma_bps
    per_min_std = sigma_bps / np.sqrt(390) / 1e4
    returns = pd.Series(rng.normal(0, per_min_std, 300))
    return estimate_state(pd.Timestamp("2026-01-05 10:05:00"), symbol,
                          recent_1min_returns=returns, trailing_daily_volume=50_000_000.0,
                          profile=profile, spread_curve=spread, vol_bounds=bounds)


# --------------------------------------------------------------------------- validation
def test_rejects_bad_side():
    with pytest.raises(ValueError):
        predict_slippage(1e6, "buyy", "cross", _state(), price=77.0)


def test_rejects_bad_order_type():
    with pytest.raises(ValueError):
        predict_slippage(1e6, "buy", "market", _state(), price=77.0)


def test_rejects_bad_impact_model():
    with pytest.raises(ValueError):
        predict_slippage(1e6, "buy", "cross", _state(), price=77.0, impact_model="linear")


# --------------------------------------------------------------------------- C01: impact models
def test_almgren_model_matches_almgren_temporary_directly():
    from slippage.impact import almgren_temporary, ALMGREN_ETA
    st = _state()
    r = predict_slippage(1e6, "buy", "cross", st, price=77.0, latency_min=15.0,
                         impact_model="almgren")
    window_usd = st.expected_interval_volume * 77.0
    participation = min(1e6 / window_usd, 1.0)
    expected = float(almgren_temporary(participation, st.sigma_now_bps, eta=ALMGREN_ETA))
    assert r["components"]["impact_band_bps"][1] == pytest.approx(expected)


def test_almgren_band_is_narrower_than_sqrt_band():
    # Almgren's band comes from a tight published SE (eta +/- 0.006); sqrt's comes from a much
    # wider adopted Y range (0.3-1.0) -- Almgren should report tighter uncertainty.
    st = _state()
    almgren = predict_slippage(1e6, "buy", "cross", st, price=77.0, impact_model="almgren")
    sqrt_ = predict_slippage(1e6, "buy", "cross", st, price=77.0, impact_model="sqrt")
    a_lo, _, a_hi = almgren["components"]["impact_band_bps"]
    s_lo, _, s_hi = sqrt_["components"]["impact_band_bps"]
    assert (a_hi - a_lo) < (s_hi - s_lo)


def test_envelope_mean_matches_sqrt_default():
    # envelope only widens the reported band; the point/mean estimate is unchanged.
    st = _state()
    envelope = predict_slippage(1e6, "buy", "cross", st, price=77.0, impact_model="envelope")
    sqrt_ = predict_slippage(1e6, "buy", "cross", st, price=77.0, impact_model="sqrt")
    assert envelope["mean_bps"] == pytest.approx(sqrt_["mean_bps"])


def test_envelope_band_spans_sqrt_band_and_almgren_point():
    # envelope = min/max across {sqrt@Y=0.3, sqrt@Y=1.0, Almgren POINT estimate} -- not
    # Almgren's own (tighter, SE-based) band, per the plan's spec.
    from slippage.impact import almgren_temporary, ALMGREN_ETA
    st = _state()
    envelope = predict_slippage(1e6, "buy", "cross", st, price=77.0, impact_model="envelope")
    sqrt_ = predict_slippage(1e6, "buy", "cross", st, price=77.0, impact_model="sqrt")
    window_usd = st.expected_interval_volume * 77.0
    participation = min(1e6 / window_usd, 1.0)
    almgren_point = float(almgren_temporary(participation, st.sigma_now_bps, eta=ALMGREN_ETA))
    e_lo, _, e_hi = envelope["components"]["impact_band_bps"]
    s_lo, _, s_hi = sqrt_["components"]["impact_band_bps"]
    assert e_lo == pytest.approx(min(s_lo, s_hi, almgren_point))
    assert e_hi == pytest.approx(max(s_lo, s_hi, almgren_point))


# --------------------------------------------------------------------------- components match source
def test_impact_band_matches_impact_bps_directly():
    st = _state()
    r = predict_slippage(1_000_000, "buy", "cross", st, price=77.0, latency_min=15.0)
    window_usd = st.expected_interval_volume * 77.0
    expected = tuple(float(impact_bps(1_000_000, window_usd, st.sigma_now_bps, Y=y, beta=0.5))
                     for y in Y_BAND)
    assert r["components"]["impact_band_bps"] == pytest.approx(expected)


def test_spread_component_is_state_spread_for_cross():
    st = _state(spread_bps=1.23)
    r = predict_slippage(1e6, "buy", "cross", st, price=77.0)
    assert r["components"]["spread_bps"] == pytest.approx(1.23)


def test_drag_is_zero_for_cross():
    r = predict_slippage(1e6, "buy", "cross", _state(), price=77.0)
    assert r["components"]["drag_bps"] == 0.0


def test_drag_matches_p02_curve_for_chase():
    st = _state(symbol="TQQQ")
    r = predict_slippage(1e6, "buy", "limit_chase", st, price=77.0, latency_min=15.0)
    assert r["components"]["drag_bps"] == pytest.approx(8.66)   # findings_02, TQQQ, T=15


def test_chase_drag_interpolates_between_measured_timeouts():
    # T=3 sits between the measured T=2 (6.75) and T=5 (7.50) points for TQQQ.
    d = _chase_drag_bps("TQQQ", 3.0)
    assert 6.75 < d < 7.50


def test_unknown_symbol_falls_back_to_tqqq_curve():
    assert _chase_drag_bps("SPXL", 15.0) == pytest.approx(_chase_drag_bps("TQQQ", 15.0))


def test_spread_is_zero_for_chase_to_avoid_double_count():
    r = predict_slippage(1e6, "buy", "limit_chase", _state(spread_bps=5.0), price=77.0)
    assert r["components"]["spread_bps"] == 0.0


def test_timing_sigma_matches_sqrt_t_scaling():
    st = _state(sigma_bps=300.0)
    expected = _timing_sigma_bps(st.sigma_now_bps, 15.0)
    r = predict_slippage(1e6, "buy", "cross", st, price=77.0, latency_min=15.0)
    assert r["components"]["timing_sigma_bps"] == pytest.approx(expected)


# --------------------------------------------------------------------------- quantile band
def test_quantile_band_is_ordered():
    st = _state(sigma_bps=300.0)
    r = predict_slippage(1e6, "buy", "cross", st, price=77.0)
    assert r["p50_bps"] < r["p90_bps"] < r["p95_bps"]


def test_p90_p95_offsets_from_mean_match_the_documented_multipliers():
    st = _state(sigma_bps=300.0)
    r = predict_slippage(1e6, "buy", "cross", st, price=77.0, latency_min=15.0)
    sigma = r["components"]["timing_sigma_bps"]
    assert r["p90_bps"] - r["mean_bps"] == pytest.approx(P90_TIMING_MULT * sigma)
    assert r["p95_bps"] - r["mean_bps"] == pytest.approx(P95_TIMING_MULT * sigma)


# --------------------------------------------------------------------------- monotonicity
def test_chase_costs_at_least_as_much_as_cross_in_mean():
    for sym in ("TQQQ", "SQQQ"):
        st = _state(symbol=sym)
        cross = predict_slippage(1e6, "buy", "cross", st, price=77.0)
        chase = predict_slippage(1e6, "buy", "limit_chase", st, price=77.0)
        assert chase["mean_bps"] >= cross["mean_bps"], sym


def test_bigger_order_has_wider_or_equal_impact_band():
    st = _state()
    small = predict_slippage(1e5, "buy", "cross", st, price=77.0)
    big = predict_slippage(5e7, "buy", "cross", st, price=77.0)
    assert big["components"]["impact_band_bps"][1] >= small["components"]["impact_band_bps"][1]
