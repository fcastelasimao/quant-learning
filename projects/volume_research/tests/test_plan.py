"""Tests for plan_execution — the pre-trade execution planner."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slippage import ExecutionPlan, MarketParams, plan_execution  # noqa: E402

P = MarketParams(sigma_daily_bps=339.0, adv_usd=4.9e9, half_spread_bps=0.74)


def test_returns_a_well_formed_plan():
    p = plan_execution(5e6, P, lam=1.0)
    assert isinstance(p, ExecutionPlan)
    assert 0.0 < p.participation <= 1.0
    assert p.horizon_min > 0 and p.n_slices >= 1
    assert p.cost_band_bps[0] < p.expected_slippage_bps < p.cost_band_bps[1]   # central inside Y-band
    assert p.feasible


def test_horizon_respects_the_cadence_cap():
    cap = 15.0
    for n in (1e6, 1e7, 3e7):
        assert plan_execution(n, P, horizon_cap_min=cap).horizon_min <= cap + 1e-6


def test_bigger_order_costs_more():
    assert plan_execution(2e7, P).expected_slippage_bps > plan_execution(1e6, P).expected_slippage_bps


def test_higher_lambda_trades_faster():
    # below the cap, more risk aversion -> higher participation, shorter fill
    slow = plan_execution(1e6, P, lam=0.0)
    fast = plan_execution(1e6, P, lam=3.0)
    assert fast.participation > slow.participation
    assert fast.horizon_min < slow.horizon_min


def test_entry_drag_off_by_default():
    p = plan_execution(5e6, P)
    assert p.entry_drag_bps == 0.0
    assert p.cross_entry is None
    assert "entry:" not in p.note.lower()


def test_entry_drag_is_not_added_to_expected_slippage():
    # the passive-chase drag is an alternative execution style, not additive to the worked cost
    base = plan_execution(5e6, P)
    withd = plan_execution(5e6, P, entry_drag_bps=14.0)
    assert withd.expected_slippage_bps == pytest.approx(base.expected_slippage_bps)


def test_cross_when_drag_exceeds_spread():
    # narrow spread (0.74 bp) << 14 bp chase drift -> cross; recommendation is size-invariant
    for n in (1e5, 5e6, 5e8):
        p = plan_execution(n, P, entry_drag_bps=14.0)
        assert p.cross_entry is True
        assert "cross" in p.note.lower()


def test_rest_limit_when_spread_exceeds_drag():
    wide = MarketParams(sigma_daily_bps=339.0, adv_usd=4.9e9, half_spread_bps=20.0)
    p = plan_execution(5e6, wide, entry_drag_bps=14.0)
    assert p.cross_entry is False
    assert "limit" in p.note.lower()


def test_oversized_order_flagged_infeasible():
    # an order needing >100% of the cadence's volume can't be filled in time
    p = plan_execution(1e10, P, horizon_cap_min=15.0)
    assert not p.feasible
    assert p.participation == 1.0
    assert "too large" in p.note.lower()
