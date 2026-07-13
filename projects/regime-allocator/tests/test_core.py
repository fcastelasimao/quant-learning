import numpy as np
import pandas as pd
import pytest

from regime_allocator.allocation import (
    momentum_weights,
    regime_allocation,
    apply_max_position,
    apply_turnover_buffer,
    apply_drawdown_guard,
)
from regime_allocator.regimes import fit_hmm, label_regimes, predict_states
from regime_allocator.features import compute_allocation_features, standardize_expanding


class TestMomentumWeights:
    def test_positive_momentum_gets_weight(self):
        dates = pd.date_range("2020-01-01", periods=13, freq="M")
        prices = pd.DataFrame(
            {"A": np.linspace(100, 150, 13), "B": np.linspace(100, 80, 13)},
            index=dates,
        )
        w = momentum_weights(prices, ["A", "B"], lookback=12)
        assert w[0] > 0
        assert w[1] == 0

    def test_sums_to_one(self):
        dates = pd.date_range("2020-01-01", periods=13, freq="M")
        prices = pd.DataFrame(
            {"A": np.linspace(100, 120, 13), "B": np.linspace(100, 130, 13)},
            index=dates,
        )
        w = momentum_weights(prices, ["A", "B"], lookback=12)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_eligible_filter(self):
        dates = pd.date_range("2020-01-01", periods=13, freq="M")
        prices = pd.DataFrame(
            {"A": np.linspace(100, 150, 13), "B": np.linspace(100, 150, 13)},
            index=dates,
        )
        w = momentum_weights(prices, ["A", "B"], lookback=12, eligible={"B"})
        assert w[0] == 0
        assert w[1] > 0


class TestRegimeAllocation:
    def test_full_risk_on(self):
        dates = pd.date_range("2020-01-01", periods=13, freq="M")
        prices = pd.DataFrame(
            {"SPY": np.linspace(100, 150, 13), "TLT": np.linspace(100, 110, 13)},
            index=dates,
        )
        probs = np.array([0.0, 1.0])
        w = regime_allocation(prices, ["SPY", "TLT"], probs, risk_on_idx=1)
        assert w[0] > 0 and w[1] > 0

    def test_full_risk_off_defensive(self):
        dates = pd.date_range("2020-01-01", periods=13, freq="M")
        prices = pd.DataFrame(
            {"SPY": np.linspace(100, 150, 13), "TLT": np.linspace(100, 120, 13)},
            index=dates,
        )
        probs = np.array([1.0, 0.0])
        w = regime_allocation(prices, ["SPY", "TLT"], probs, risk_on_idx=1)
        assert w[1] > w[0]


class TestMaxPosition:
    def test_cap_enforced(self):
        w = np.array([0.6, 0.2, 0.2])
        result = apply_max_position(w, 0.4)
        assert result.max() <= 0.40 + 1e-6
        assert abs(result.sum() - 1.0) < 1e-6


class TestTurnoverBuffer:
    def test_no_change_within_buffer(self):
        current = np.array([0.5, 0.5])
        target = np.array([0.52, 0.48])
        result = apply_turnover_buffer(current, target, buffer=0.03)
        np.testing.assert_allclose(result, current / current.sum(), atol=1e-6)

    def test_rebalances_beyond_buffer(self):
        current = np.array([0.5, 0.5])
        target = np.array([0.7, 0.3])
        result = apply_turnover_buffer(current, target, buffer=0.03)
        np.testing.assert_allclose(result, target / target.sum(), atol=1e-6)

    def test_first_allocation_uses_target(self):
        target = np.array([0.6, 0.4])
        result = apply_turnover_buffer(None, target, buffer=0.03)
        np.testing.assert_allclose(result, target, atol=1e-6)


class TestDrawdownGuard:
    def test_no_trigger_below_threshold(self):
        w = np.array([0.5, 0.5])
        w_out, triggered = apply_drawdown_guard(w, 0.05, 0.25)
        assert not triggered

    def test_trigger_above_threshold(self):
        w = np.array([0.5, 0.5])
        w_out, triggered = apply_drawdown_guard(w, 0.30, 0.25)
        assert triggered
        np.testing.assert_allclose(w_out, w * 0.5)


class TestHMM:
    def test_fit_and_predict(self):
        rng = np.random.RandomState(42)
        data = np.vstack(
            [rng.multivariate_normal([1, 0], np.eye(2), 50),
             rng.multivariate_normal([-1, 0], np.eye(2), 50)]
        )
        model = fit_hmm(data, n_regimes=2)
        states = predict_states(model, data)
        assert len(states) == 100
        assert set(states).issubset({0, 1})

    def test_label_regimes_2_states(self):
        rng = np.random.RandomState(42)
        data = np.vstack(
            [rng.multivariate_normal([2, 0, 0], np.eye(3) * 0.1, 80),
             rng.multivariate_normal([-2, 0, 0], np.eye(3) * 0.1, 80)]
        )
        model = fit_hmm(data, n_regimes=2)
        labels = label_regimes(model)
        assert set(labels.values()) == {"risk_off", "risk_on"}


class TestStandardize:
    def test_expanding_zscore(self):
        rng = np.random.RandomState(42)
        df = pd.DataFrame(
            rng.randn(100, 3), columns=["a", "b", "c"],
            index=pd.date_range("2020-01-01", periods=100),
        )
        result = standardize_expanding(df)
        assert len(result) <= len(df)
        assert not result.isna().any().any()
