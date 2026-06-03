"""
test_rebalance_policy.py
========================
Tests for the D.15 RebalancePolicy primitive and its wiring into
engine.backtest.run_backtest.

The byte-identical default-behavior guard lives in test_backtest_golden.py.
This file covers:
  * RebalancePolicy.should_rebalance for each mode and its factory guards.
  * That drift modes actually reduce trading vs monthly_unconditional and that
    no transaction cost is charged on no-trade months.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.backtest import RebalancePolicy, run_backtest


# ---------------------------------------------------------------------------
# RebalancePolicy.should_rebalance
# ---------------------------------------------------------------------------

TARGET = {"A": 0.50, "B": 0.30, "C": 0.20}


def test_monthly_unconditional_always_true_regardless_of_weights():
    pol = RebalancePolicy.monthly_unconditional()
    assert pol.should_rebalance(TARGET, TARGET) is True
    assert pol.should_rebalance({"A": 0.99, "B": 0.005, "C": 0.005}, TARGET) is True


def test_monthly_unconditional_is_the_default():
    assert RebalancePolicy().mode == RebalancePolicy.monthly_unconditional().mode


def test_drift_absolute_triggers_on_percentage_points():
    pol = RebalancePolicy.drift_absolute(0.02)  # 2 percentage points
    # B drifts 1.5pp -> no breach
    assert pol.should_rebalance({"A": 0.50, "B": 0.315, "C": 0.185}, TARGET) is False
    # B drifts 2.5pp -> breach
    assert pol.should_rebalance({"A": 0.50, "B": 0.325, "C": 0.175}, TARGET) is True


def test_drift_relative_triggers_on_fraction_of_target():
    pol = RebalancePolicy.drift_relative(0.20)  # 20% of each target weight
    # C target 0.20, 20% of target = 0.04 abs. Drift 0.03 -> no breach.
    assert pol.should_rebalance({"A": 0.50, "B": 0.33, "C": 0.17}, TARGET) is False
    # C drifts 0.05 abs (25% of target) -> breach.
    assert pol.should_rebalance({"A": 0.50, "B": 0.35, "C": 0.15}, TARGET) is True


def test_relative_threshold_is_scaled_per_asset():
    """A small-target asset breaches on a smaller absolute drift than a big one."""
    pol = RebalancePolicy.drift_relative(0.20)
    # Only C moves, by 0.045 abs = 22.5% of its 0.20 target -> breach.
    assert pol.should_rebalance({"A": 0.50, "B": 0.255, "C": 0.245}, TARGET) is True


def test_monthly_check_then_drift_matches_drift_relative_here():
    a = RebalancePolicy.monthly_check_then_drift(0.20)
    b = RebalancePolicy.drift_relative(0.20)
    drifted = {"A": 0.50, "B": 0.35, "C": 0.15}
    assert a.should_rebalance(drifted, TARGET) == b.should_rebalance(drifted, TARGET) is True


@pytest.mark.parametrize("factory", [
    RebalancePolicy.drift_relative,
    RebalancePolicy.drift_absolute,
    RebalancePolicy.monthly_check_then_drift,
])
def test_nonpositive_threshold_rejected(factory):
    with pytest.raises(ValueError):
        factory(0.0)
    with pytest.raises(ValueError):
        factory(-0.1)


def test_unknown_mode_raises():
    pol = RebalancePolicy(mode="bogus")
    with pytest.raises(ValueError, match="Unknown rebalance mode"):
        pol.should_rebalance(TARGET, TARGET)


def test_label():
    assert RebalancePolicy.monthly_unconditional().label == "monthly_unconditional"
    assert RebalancePolicy.drift_relative(0.2).label == "drift_relative(0.2)"
    assert RebalancePolicy.drift_absolute(0.02).label == "drift_absolute(0.02)"


def test_policy_is_frozen():
    pol = RebalancePolicy.monthly_unconditional()
    with pytest.raises((AttributeError, TypeError)):
        pol.mode = "x"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# engine wiring
# ---------------------------------------------------------------------------

def _trending_prices() -> pd.DataFrame:
    """Two assets diverging steadily so any drift policy eventually fires."""
    dates = pd.date_range("2010-01-01", "2015-12-31", freq="B")
    n = len(dates)
    a = 100.0 * np.exp(np.linspace(0, 0.8, n))   # rises ~120%
    b = 100.0 * np.exp(np.linspace(0, -0.2, n))  # falls ~18%
    return pd.DataFrame({"A": a, "B": b}, index=dates)


def test_explicit_monthly_unconditional_equals_default():
    prices = _trending_prices()
    alloc = {"A": 0.6, "B": 0.4}
    base = run_backtest(prices, prices["A"], alloc, 10_000.0,
                        transaction_cost_pct=0.001)
    explicit = run_backtest(prices, prices["A"], alloc, 10_000.0,
                            transaction_cost_pct=0.001,
                            rebalance_policy=RebalancePolicy.monthly_unconditional())
    pd.testing.assert_frame_equal(base, explicit)


def test_drift_policy_changes_outcome_vs_monthly():
    prices = _trending_prices()
    alloc = {"A": 0.6, "B": 0.4}
    monthly = run_backtest(prices, prices["A"], alloc, 10_000.0,
                           transaction_cost_pct=0.001,
                           rebalance_policy=RebalancePolicy.monthly_unconditional())
    drift = run_backtest(prices, prices["A"], alloc, 10_000.0,
                         transaction_cost_pct=0.001,
                         rebalance_policy=RebalancePolicy.drift_relative(0.20))
    # The two equity paths must differ (drift trades less often).
    assert not np.allclose(
        monthly["All Weather Value"].values,
        drift["All Weather Value"].values,
    )


def test_zero_cost_drift_matches_monthly_when_threshold_never_breached():
    """With a huge threshold the drift policy never trades; with zero cost its
    buy-and-hold path equals the engine's Buy & Hold column."""
    prices = _trending_prices()
    alloc = {"A": 0.6, "B": 0.4}
    out = run_backtest(prices, prices["A"], alloc, 10_000.0,
                       transaction_cost_pct=0.0,
                       rebalance_policy=RebalancePolicy.drift_relative(100.0))
    np.testing.assert_allclose(
        out["All Weather Value"].values,
        out["Buy & Hold All Weather"].values,
        rtol=1e-12,
    )


def test_no_cost_charged_on_no_trade_months():
    """A never-firing drift policy with a positive cost must still equal the
    cost-free buy & hold — i.e. cost is only charged when trading."""
    prices = _trending_prices()
    alloc = {"A": 0.6, "B": 0.4}
    out = run_backtest(prices, prices["A"], alloc, 10_000.0,
                       transaction_cost_pct=0.01,  # 1% — large, to be obvious
                       rebalance_policy=RebalancePolicy.drift_relative(100.0))
    np.testing.assert_allclose(
        out["All Weather Value"].values,
        out["Buy & Hold All Weather"].values,
        rtol=1e-12,
    )
