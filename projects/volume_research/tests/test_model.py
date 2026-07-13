"""Tests for the CostModel facade — it must compose the building blocks faithfully."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slippage import CostModel, MarketParams  # noqa: E402
from slippage.cost import expected_slippage_bps  # noqa: E402

P = MarketParams(sigma_daily_bps=339.0, adv_usd=4.9e9, half_spread_bps=1.0)


def test_roundtrip_is_twice_the_per_side_expected_slippage():
    m = CostModel(P)
    rt = m.roundtrip(5e6)
    part = (5e6 / P.adv_usd) * (P.minutes_per_day / 15)
    c = expected_slippage_bps(5e6, part, P, Y=0.5)
    assert rt.expected_slippage_bps == pytest.approx(2.0 * c["expected_slippage_bps"])
    assert rt.timing_bps == pytest.approx(np.sqrt(2.0) * c["timing_risk_bps"])


def test_timing_variance_matches_round_trip_sigma():
    rt = CostModel(P).roundtrip(5e6)
    assert rt.timing_var_frac2 == pytest.approx((rt.timing_bps / 1e4) ** 2)


def test_toggles_zero_out_components():
    n = 5e6
    assert CostModel(P, spread=False).roundtrip(n).spread_bps == 0.0
    assert CostModel(P, impact=False).roundtrip(n).impact_bps == 0.0
    assert CostModel(P, delay=False).roundtrip(n).timing_bps == 0.0
    # turning a component off only removes that component
    assert CostModel(P, impact=False).roundtrip(n).spread_bps > 0.0


def test_band_is_monotone_in_Y():
    band = CostModel(P).roundtrip_band(5e6, Ys=(0.3, 0.5, 1.0))
    costs = [band[y].expected_slippage_bps for y in (0.3, 0.5, 1.0)]
    assert costs[0] < costs[1] < costs[2]   # higher Y -> more impact -> more cost


def test_cost_increases_with_notional():
    m = CostModel(P)
    assert m.roundtrip(1e7).expected_slippage_bps > m.roundtrip(1e5).expected_slippage_bps


def test_entry_drag_adds_once_as_mean_drag():
    base = CostModel(P).roundtrip(5e6)
    withd = CostModel(P, entry_drag_bps=14.0).roundtrip(5e6)
    # signed entry drift is charged once (entry only), on top of spread+impact
    assert withd.entry_drag_bps == pytest.approx(14.0)
    assert withd.expected_slippage_bps == pytest.approx(base.expected_slippage_bps + 14.0)
    # it's a mean drag, not variance -> timing is untouched (different moments, no double-count)
    assert withd.timing_bps == pytest.approx(base.timing_bps)


def test_entry_drag_defaults_to_zero():
    assert CostModel(P).roundtrip(5e6).entry_drag_bps == 0.0


def test_entry_drag_rides_the_delay_toggle():
    rt = CostModel(P, entry_drag_bps=14.0, delay=False).roundtrip(5e6)
    assert rt.entry_drag_bps == 0.0
    assert rt.timing_bps == 0.0


def test_optimal_speed_trades_impact_for_timing():
    # higher λ -> faster fill -> more impact, less timing
    slow = CostModel(P).roundtrip_optimal(1e6, lam=0.0)
    fast = CostModel(P).roundtrip_optimal(1e6, lam=3.0)
    assert fast.impact_bps > slow.impact_bps
    assert fast.timing_bps < slow.timing_bps


# --------------------------------------------------------------------------- C01: impact_model
def test_roundtrip_default_impact_model_is_sqrt():
    default = CostModel(P).roundtrip(5e6)
    explicit = CostModel(P).roundtrip(5e6, impact_model="sqrt")
    assert default.impact_bps == pytest.approx(explicit.impact_bps)


def test_roundtrip_almgren_differs_from_sqrt():
    sqrt_rt = CostModel(P).roundtrip(5e6, impact_model="sqrt")
    almgren_rt = CostModel(P).roundtrip(5e6, impact_model="almgren")
    assert sqrt_rt.impact_bps != pytest.approx(almgren_rt.impact_bps)


def test_roundtrip_optimal_respects_impact_model():
    sqrt_rt = CostModel(P).roundtrip_optimal(5e6, impact_model="sqrt")
    almgren_rt = CostModel(P).roundtrip_optimal(5e6, impact_model="almgren")
    assert sqrt_rt.impact_bps != pytest.approx(almgren_rt.impact_bps)
