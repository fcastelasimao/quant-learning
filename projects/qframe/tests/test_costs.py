"""
Unit tests for factor_harness/costs.py
"""
import numpy as np
import pandas as pd
import pytest

from qframe.factor_harness.costs import (
    CostParams,
    DEFAULT_COST_PARAMS,
    estimate_cost_bps,
    round_trip_cost_bps,
    compute_turnover,
    net_ic,
)


class TestEstimateCostBps:
    def test_positive_cost(self):
        cost = estimate_cost_bps()
        assert cost > 0

    def test_higher_size_higher_cost(self):
        small = estimate_cost_bps(adv_fraction=0.01)
        large = estimate_cost_bps(adv_fraction=0.50)
        assert large > small

    def test_round_trip_is_double(self):
        one_way = estimate_cost_bps()
        rt = round_trip_cost_bps()
        assert abs(rt - 2 * one_way) < 1e-10

    def test_default_params_reasonable(self):
        # Default spread=10bps + impact should give something in range 10-100 bps
        rt = round_trip_cost_bps()
        assert 10 < rt < 200, f"Round-trip cost {rt:.1f} bps seems unreasonable"


class TestComputeTurnover:
    def test_zero_turnover_constant_weights(self):
        dates = pd.bdate_range("2020-01-01", periods=10)
        w = pd.DataFrame({"A": [0.5] * 10, "B": [0.5] * 10}, index=dates)
        to = compute_turnover(w)
        # After first row (NaN), all subsequent should be 0
        assert to.iloc[1:].abs().max() < 1e-10

    def test_full_turnover_flip(self):
        """Flipping 100% long A short B to long B short A = 100% turnover."""
        dates = pd.bdate_range("2020-01-01", periods=3)
        w = pd.DataFrame({"A": [0.5, -0.5, 0.5], "B": [-0.5, 0.5, -0.5]}, index=dates)
        to = compute_turnover(w)
        # Each flip changes |w_A| + |w_B| by 2 * 0.5 + 2 * 0.5 = 2, half = 1
        assert abs(to.iloc[1] - 1.0) < 1e-10

    def test_first_row_nan(self):
        dates = pd.bdate_range("2020-01-01", periods=5)
        w = pd.DataFrame({"A": [0.5] * 5, "B": [0.5] * 5}, index=dates)
        to = compute_turnover(w)
        assert np.isnan(to.iloc[0])


class TestCostParamsValidation:
    """Tests for CostParams.__post_init__ parameter validation."""

    def test_default_params_valid(self):
        """Default params should not raise."""
        params = CostParams()  # should not raise
        assert params.spread_bps == 10.0

    def test_negative_spread_raises(self):
        with pytest.raises(ValueError, match="spread_bps"):
            CostParams(spread_bps=-1.0)

    def test_zero_spread_allowed(self):
        """spread_bps=0 is valid (zero-cost scenario)."""
        params = CostParams(spread_bps=0.0)
        assert params.spread_bps == 0.0

    def test_eta_zero_raises(self):
        with pytest.raises(ValueError, match="eta"):
            CostParams(eta=0.0)

    def test_eta_above_one_raises(self):
        with pytest.raises(ValueError, match="eta"):
            CostParams(eta=1.5)

    def test_eta_exactly_one_allowed(self):
        """eta=1.0 (linear impact) is valid."""
        params = CostParams(eta=1.0)
        assert params.eta == 1.0

    def test_negative_gamma_raises(self):
        with pytest.raises(ValueError, match="gamma"):
            CostParams(gamma=-0.1)

    def test_negative_borrow_raises(self):
        with pytest.raises(ValueError, match="short_borrow_bps_annual"):
            CostParams(short_borrow_bps_annual=-10.0)

    def test_negative_funding_raises(self):
        with pytest.raises(ValueError, match="funding_cost_bps_annual"):
            CostParams(funding_cost_bps_annual=-5.0)

    def test_zero_adv_fraction_raises(self):
        with pytest.raises(ValueError, match="adv_fraction"):
            CostParams(adv_fraction=0.0)


class TestNetIC:
    def test_net_ic_less_than_gross(self):
        """Net IC should be <= gross IC when there is turnover."""
        dates = pd.bdate_range("2020-01-01", periods=100)
        gross = pd.Series(0.05, index=dates)
        turnover = pd.Series(0.5, index=dates)
        n = net_ic(gross, turnover)
        assert (n <= gross).all()

    def test_zero_turnover_no_drag(self):
        """With zero turnover, net IC is very close to gross IC.

        The cost model charges a small continuous borrow cost even with zero
        turnover (holding short positions costs spread_bps/year ÷ 252/day).
        The difference is tiny (<1e-3) but non-zero. We use atol=1e-3.
        """
        dates = pd.bdate_range("2020-01-01", periods=50)
        gross = pd.Series(0.05, index=dates)
        turnover = pd.Series(0.0, index=dates)
        n = net_ic(gross, turnover)
        np.testing.assert_allclose(n.values, gross.values, atol=1e-3)

    def test_returns_series(self):
        dates = pd.bdate_range("2020-01-01", periods=20)
        gross = pd.Series(0.03, index=dates)
        turnover = pd.Series(0.3, index=dates)
        n = net_ic(gross, turnover)
        assert isinstance(n, pd.Series)
