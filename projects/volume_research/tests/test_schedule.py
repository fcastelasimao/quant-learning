"""Tests for slippage/schedule.py — schedule_order(), the scheduler centerpiece.

Per the plan's spec: monotonicity (bigger order -> longer h*, more slices; thinner predicted
volume -> smaller children); small-order limit degenerates to a single cross; g==0 and no
interruption -> pure cost minimization; POV cap never violated.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slippage.predict import predict_slippage  # noqa: E402
from slippage.schedule import schedule_order, _H_GRID_MIN, _objective  # noqa: E402
from slippage.state import MarketState  # noqa: E402

PRICE = 77.0


def _state(volume=3_000_000.0, sigma=300.0, regime="normal"):
    return MarketState(ts=pd.Timestamp("2026-01-05 10:00:00"), symbol="TQQQ", bin_label="10:00",
                       expected_interval_volume=volume, thin_volume_p10=volume * 0.4,
                       thin_volume_p20=volume * 0.6, sigma_now_bps=sigma, regime=regime,
                       spread_bps=0.74)


# --------------------------------------------------------------------------- monotonicity
def test_bigger_order_gets_longer_or_equal_horizon():
    st = _state()
    small = schedule_order(1e5, "buy", st, price=PRICE, edge_bps=50.0)
    big = schedule_order(2e7, "buy", st, price=PRICE, edge_bps=50.0)
    assert big.horizon_min >= small.horizon_min


def test_bigger_order_gets_more_or_equal_slices():
    st = _state()
    small = schedule_order(1e5, "buy", st, price=PRICE, edge_bps=50.0)
    big = schedule_order(2e7, "buy", st, price=PRICE, edge_bps=50.0)
    assert len(big.slices) >= len(small.slices)


def test_thinner_volume_gives_smaller_children():
    thick = _state(volume=5_000_000.0)
    thin = _state(volume=200_000.0)
    s_thick = schedule_order(5e6, "buy", thick, price=PRICE, edge_bps=50.0)
    s_thin = schedule_order(5e6, "buy", thin, price=PRICE, edge_bps=50.0)
    assert max(sl.child_notional_usd for sl in s_thin.slices) < \
        max(sl.child_notional_usd for sl in s_thick.slices)


# --------------------------------------------------------------------------- small-order limit
def test_small_order_degenerates_to_single_cross():
    st = _state()
    s = schedule_order(1e4, "buy", st, price=PRICE, edge_bps=50.0)
    assert len(s.slices) == 1
    assert s.slices[0].order_style == "cross"
    assert s.slices[0].child_notional_usd == pytest.approx(1e4, rel=1e-6)


def test_small_order_cost_is_same_ballpark_as_predict_slippage_cross():
    # schedule_order sits on predict_slippage internally; at the small-order limit the two
    # systems should broadly agree (same underlying cost engine, not a separate model).
    st = _state()
    s = schedule_order(1e4, "buy", st, price=PRICE, edge_bps=50.0)
    direct = predict_slippage(1e4, "buy", "cross", st, price=PRICE, latency_min=s.horizon_min)
    assert s.expected_slippage_bps == pytest.approx(direct["mean_bps"])


# --------------------------------------------------------------------------- pure cost minimization
def test_zero_edge_gives_pure_cost_minimization():
    # edge_bps=0 zeroes BOTH the alpha-forfeit term (edge_bps * g(h)) AND the interruption term
    # (also scaled by edge_bps) -> the horizon search degenerates to minimizing predict_slippage
    # alone. Verify h* matches a direct grid search of the cost term only.
    st = _state()
    s = schedule_order(5e6, "buy", st, price=PRICE, edge_bps=0.0)
    costs = [predict_slippage(5e6, "buy", "cross", st, price=PRICE, latency_min=h)["mean_bps"]
            for h in _H_GRID_MIN]
    expected_h = float(_H_GRID_MIN[int(np.argmin(costs))])
    assert s.horizon_min == pytest.approx(expected_h)
    assert s.alpha_forfeit_bps == pytest.approx(0.0)
    assert s.interruption_summary["expected_cost_bps"] == pytest.approx(0.0)


def test_objective_reduces_to_cost_alone_at_zero_edge():
    st = _state()
    for h in (15.0, 60.0, 240.0):
        obj = _objective(h, 5e6, "buy", st, PRICE, edge_bps=0.0, mode="cancel")
        cost = predict_slippage(5e6, "buy", "cross", st, price=PRICE, latency_min=h)["mean_bps"]
        assert obj == pytest.approx(cost)


# --------------------------------------------------------------------------- POV cap
def test_pov_cap_never_violated_flat_fallback():
    st = _state(volume=1_000_000.0)   # thin, forces many slices
    pov_cap = 0.10
    s = schedule_order(8e6, "buy", st, price=PRICE, edge_bps=50.0, pov_cap=pov_cap)
    bin_cap_usd = pov_cap * st.expected_interval_volume * PRICE
    for sl in s.slices:
        assert sl.child_notional_usd <= bin_cap_usd + 1e-6


def test_pov_cap_tighter_cap_gives_more_slices():
    st = _state(volume=2_000_000.0)
    loose = schedule_order(5e6, "buy", st, price=PRICE, edge_bps=50.0, pov_cap=0.30)
    tight = schedule_order(5e6, "buy", st, price=PRICE, edge_bps=50.0, pov_cap=0.05)
    assert len(tight.slices) >= len(loose.slices)


def test_infeasible_when_order_dwarfs_all_available_volume():
    st = _state(volume=1_000.0)   # tiny volume, even 200 bins can't absorb a huge order
    s = schedule_order(1e9, "buy", st, price=PRICE, edge_bps=50.0, pov_cap=0.10)
    assert s.feasible is False
    assert s.slices == []
    assert "WARNING" in s.assumptions


# --------------------------------------------------------------------------- basic shape
def test_slices_sum_to_notional_when_feasible():
    st = _state()
    s = schedule_order(5e6, "buy", st, price=PRICE, edge_bps=50.0)
    assert sum(sl.child_notional_usd for sl in s.slices) == pytest.approx(5e6, rel=1e-6)


def test_band_is_ordered():
    st = _state()
    s = schedule_order(5e6, "buy", st, price=PRICE, edge_bps=50.0)
    lo, hi = s.expected_slippage_band_bps
    assert lo <= s.expected_slippage_bps <= hi
