"""Tests for the Stage-5 size-aware cost function (math self-consistency).

No data ground truth (Y is adopted), so we verify: consistency with the Block-4 √-law at
day-execution, the volume↔time identity, and the impact/timing trade-off shape.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slippage.cost import (  # noqa: E402
    MarketParams, expected_slippage_bps, fill_minutes, optimal_participation, capacity_at_horizon,
)
from slippage.impact import impact_bps, capacity, almgren_temporary  # noqa: E402

P = MarketParams(sigma_daily_bps=339.0, adv_usd=4.9e9, half_spread_bps=1.0)


def test_day_execution_matches_block4_sqrt_law():
    # At participation = Q/ADV (work the order over a full day), impact must equal the
    # Block-4 √-law, and the fill time must be one trading day.
    Q = 17e6
    part = Q / P.adv_usd
    c = expected_slippage_bps(Q, part, P)
    assert c["impact_bps"] == pytest.approx(impact_bps(Q, P.adv_usd, P.sigma_daily_bps), rel=1e-9)
    assert c["fill_minutes"] == pytest.approx(P.minutes_per_day, rel=1e-9)


def test_volume_sets_fill_time():
    # t = (Q/ADV)/participation · day
    assert fill_minutes(10e6, 0.10, P) == pytest.approx((10e6 / P.adv_usd) / 0.10 * 390)


def test_faster_more_impact_less_timing():
    slow = expected_slippage_bps(10e6, 0.05, P)
    fast = expected_slippage_bps(10e6, 0.20, P)
    assert fast["impact_bps"] > slow["impact_bps"]        # faster -> more push
    assert fast["timing_risk_bps"] < slow["timing_risk_bps"]  # faster -> less drift


def test_tradeoff_has_interior_optimum():
    # The risk-adjusted cost should be minimised at an interior participation, not at an edge.
    p_star = optimal_participation(20e6, P, lam=1.0)
    grid_edges = (0.002, 0.6)
    assert grid_edges[0] < p_star < grid_edges[1]


def test_capacity_shrinks_with_shorter_horizon():
    # Must-fill-in-15-min capacity << work-over-a-day capacity.
    cap_15 = capacity_at_horizon(25.0, P, exec_minutes=15)
    cap_day = capacity_at_horizon(25.0, P, exec_minutes=390)
    assert cap_15 < cap_day


def test_capacity_at_day_matches_impact_capacity():
    # Over a full day, capacity_at_horizon == the Block-4 inversion on the budget net of spread.
    budget = 25.0
    cap_day = capacity_at_horizon(budget, P, exec_minutes=390)
    expected = capacity(budget - P.half_spread_bps, P.adv_usd, P.sigma_daily_bps)
    assert cap_day == pytest.approx(expected, rel=1e-9)


# --------------------------------------------------------------------------- C01: impact_model
def test_default_impact_model_is_sqrt():
    with_default = expected_slippage_bps(10e6, 0.1, P)
    explicit_sqrt = expected_slippage_bps(10e6, 0.1, P, impact_model="sqrt")
    assert with_default["impact_bps"] == pytest.approx(explicit_sqrt["impact_bps"])


def test_almgren_impact_model_matches_almgren_temporary():
    c = expected_slippage_bps(10e6, 0.1, P, impact_model="almgren")
    expected = almgren_temporary(0.1, P.sigma_daily_bps)
    assert c["impact_bps"] == pytest.approx(expected)


def test_rejects_unknown_impact_model():
    with pytest.raises(ValueError):
        expected_slippage_bps(10e6, 0.1, P, impact_model="linear")


def test_optimal_participation_respects_impact_model():
    p_sqrt = optimal_participation(20e6, P, lam=1.0, impact_model="sqrt")
    p_almgren = optimal_participation(20e6, P, lam=1.0, impact_model="almgren")
    # Both are valid interior optima; just confirm the parameter actually changes the objective
    # (almgren's beta=0.6 > sqrt's default beta=0.5 changes the impact-vs-timing trade-off).
    assert p_sqrt != pytest.approx(p_almgren)
