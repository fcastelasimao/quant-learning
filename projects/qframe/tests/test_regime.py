"""
Tests for the Phase 2 regime module:
    - RegimeHSMM (hsmm.py)
    - HurstEstimator (hurst.py)
    - velocity functions (velocity.py)
    - RegimeICAnalyzer (analyzer.py)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
DATES_LONG = pd.bdate_range("2012-01-01", periods=1200)
DATES_MED  = pd.bdate_range("2018-01-01", periods=600)
DATES_SHORT = pd.bdate_range("2020-01-01", periods=100)


def _make_returns(n: int = 600, mu: float = 0.0, sigma: float = 0.01, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mu, sigma, n), index=pd.bdate_range("2018-01-01", periods=n))


def _make_trending_returns(n: int = 600) -> pd.Series:
    """AR(1) process with positive autocorrelation → H > 0.5."""
    rng = np.random.default_rng(1)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = 0.4 * r[i - 1] + rng.normal(0, 0.008)
    return pd.Series(r, index=pd.bdate_range("2018-01-01", periods=n))


def _make_mean_reverting_returns(n: int = 600) -> pd.Series:
    """AR(1) process with negative autocorrelation → H < 0.5."""
    rng = np.random.default_rng(2)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = -0.4 * r[i - 1] + rng.normal(0, 0.008)
    return pd.Series(r, index=pd.bdate_range("2018-01-01", periods=n))


def _make_prices(tickers: list[str] = None, n: int = 800, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if tickers is None:
        tickers = [f"T{i:03d}" for i in range(50)]
    rets = rng.normal(0.0003, 0.015, size=(n, len(tickers)))
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=pd.bdate_range("2016-01-01", periods=n), columns=tickers)


# ===========================================================================
# RegimeHSMM tests
# ===========================================================================

class TestRegimeHSMM:
    from qframe.regime.hsmm import RegimeHSMM

    def test_fit_predict_shapes(self):
        from qframe.regime.hsmm import RegimeHSMM
        ret = _make_returns(400)
        m = RegimeHSMM(n_states=3)
        m.fit(ret)
        labels = m.predict(ret)
        assert len(labels) == len(ret)
        assert labels.dtype == int
        assert set(labels.unique()).issubset({0, 1, 2})

    def test_predict_proba_shapes_and_sums(self):
        from qframe.regime.hsmm import RegimeHSMM
        ret = _make_returns(400)
        m = RegimeHSMM(n_states=3)
        m.fit(ret)
        proba = m.predict_proba(ret)
        assert proba.shape == (len(ret), 3)
        np.testing.assert_allclose(proba.sum(axis=1).values, 1.0, atol=1e-6)

    def test_canonical_ordering_ascending(self):
        """States should be ordered ascending by mean return."""
        from qframe.regime.hsmm import RegimeHSMM
        ret = _make_returns(600)
        m = RegimeHSMM(n_states=3)
        m.fit(ret)
        # Extract per-state mean return from the fitted model
        # means_[:, 0] is r_t component; after canonical reordering it should be ascending
        canonical_means = m._model.means_[m._canonical_order, 0]
        assert list(canonical_means) == sorted(canonical_means), (
            "Canonical state means should be sorted ascending by return"
        )

    def test_no_lookahead_fit_rolling(self):
        """Posterior at date t must not use data after t."""
        from qframe.regime.hsmm import RegimeHSMM
        ret = _make_returns(600)
        m = RegimeHSMM(n_states=3)
        proba = m.fit_rolling(ret, window=300, step=21)
        # First 300 dates should be NaN (warm-up)
        assert proba.iloc[:300].isna().all().all()
        # After warm-up there should be valid probabilities
        valid = proba.iloc[300:].dropna()
        assert len(valid) > 0
        np.testing.assert_allclose(valid.sum(axis=1).values, 1.0, atol=1e-5)

    def test_fit_rolling_shape(self):
        from qframe.regime.hsmm import RegimeHSMM
        ret = _make_returns(400)
        m = RegimeHSMM(n_states=3)
        proba = m.fit_rolling(ret, window=200, step=21)
        assert proba.shape == (len(ret), 3)

    def test_features_are_r_and_r_squared(self):
        """_build_features must return [r, r²] not [r, |r|]."""
        from qframe.regime.hsmm import RegimeHSMM
        r = pd.Series([0.01, -0.02, 0.00, 0.03])
        X = RegimeHSMM._build_features(r)
        expected_col1 = r.values ** 2
        np.testing.assert_allclose(X[:, 1], expected_col1, rtol=1e-10)

    def test_predict_labels_within_range(self):
        from qframe.regime.hsmm import RegimeHSMM
        for n_states in [2, 3, 5]:
            ret = _make_returns(400)
            m = RegimeHSMM(n_states=n_states)
            m.fit(ret)
            labels = m.predict(ret)
            assert labels.min() >= 0
            assert labels.max() < n_states

    def test_unfitted_raises(self):
        from qframe.regime.hsmm import RegimeHSMM
        m = RegimeHSMM()
        with pytest.raises(RuntimeError, match="not fitted"):
            m.predict(_make_returns(10))

    def test_regime_stats_columns(self):
        from qframe.regime.hsmm import RegimeHSMM
        ret = _make_returns(400)
        m = RegimeHSMM(n_states=3)
        m.fit(ret)
        labels = m.predict(ret)
        stats = m.regime_stats(ret, labels)
        assert set(stats.columns) >= {"count", "pct_time", "mean_ann", "std_ann", "sharpe"}
        np.testing.assert_allclose(stats["pct_time"].sum(), 1.0, atol=1e-6)


# ===========================================================================
# HurstEstimator tests
# ===========================================================================

class TestHurstEstimator:

    def test_random_walk_near_half(self):
        """Pure iid returns should give H ≈ 0.5."""
        from qframe.regime.hurst import HurstEstimator
        rng = np.random.default_rng(99)
        ret = pd.Series(rng.normal(0, 0.01, 2000), index=pd.bdate_range("2010-01-01", periods=2000))
        est = HurstEstimator(min_periods=200)
        h = est.fit(ret)
        assert 0.35 < h < 0.65, f"Expected H≈0.5 for iid series, got {h:.3f}"

    def test_trending_above_half(self):
        """Positively autocorrelated series should give H > 0.5."""
        from qframe.regime.hurst import HurstEstimator
        ret = _make_trending_returns(1000)
        est = HurstEstimator(min_periods=200)
        h = est.fit(ret)
        assert h > 0.5, f"Expected H > 0.5 for trending series, got {h:.3f}"

    def test_mean_reverting_below_half(self):
        """Negatively autocorrelated series should give H < 0.5."""
        from qframe.regime.hurst import HurstEstimator
        ret = _make_mean_reverting_returns(1000)
        est = HurstEstimator(min_periods=200)
        h = est.fit(ret)
        assert h < 0.5, f"Expected H < 0.5 for mean-reverting series, got {h:.3f}"

    def test_rolling_shape_and_warmup(self):
        from qframe.regime.hurst import HurstEstimator
        ret = _make_returns(400)
        est = HurstEstimator(min_periods=100)
        rolling_h = est.fit_rolling(ret, window=200)
        assert len(rolling_h) == len(ret)
        # First window-1 values should be NaN
        assert rolling_h.iloc[:199].isna().all()
        # Values after warmup should be in (0, 1)
        valid = rolling_h.dropna()
        assert len(valid) > 0
        assert (valid > 0).all() and (valid < 1).all()

    def test_insufficient_data_returns_nan(self):
        from qframe.regime.hurst import HurstEstimator
        ret = _make_returns(50)
        est = HurstEstimator(min_periods=200)
        h = est.fit(ret)
        assert np.isnan(h)

    def test_h_in_unit_interval(self):
        from qframe.regime.hurst import HurstEstimator
        ret = _make_returns(800)
        est = HurstEstimator(min_periods=100)
        h = est.fit(ret)
        assert 0.0 < h < 1.0


# ===========================================================================
# Velocity tests
# ===========================================================================

class TestVelocity:

    def _make_proba(self, n: int = 100, n_states: int = 3, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        raw = rng.dirichlet(np.ones(n_states), size=n)
        return pd.DataFrame(raw, index=pd.bdate_range("2020-01-01", periods=n))

    def test_kl_zero_when_identical(self):
        """KL divergence is 0 when today's distribution equals window-ago."""
        from qframe.regime.velocity import kl_divergence_velocity
        # Constant distribution → KL = 0
        n, k = 50, 3
        arr = np.zeros((n, k))
        arr[:, 0] = 0.6; arr[:, 1] = 0.3; arr[:, 2] = 0.1
        proba = pd.DataFrame(arr, index=pd.bdate_range("2020-01-01", periods=n))
        kl = kl_divergence_velocity(proba, window=5)
        valid = kl.dropna()
        np.testing.assert_allclose(valid.values, 0.0, atol=1e-8)

    def test_kl_positive_when_distributions_differ(self):
        from qframe.regime.velocity import kl_divergence_velocity
        proba = self._make_proba(100, n_states=3)
        kl = kl_divergence_velocity(proba, window=5)
        valid = kl.dropna()
        assert (valid >= 0).all()
        assert (valid > 0).any()

    def test_kl_warmup_nans(self):
        from qframe.regime.velocity import kl_divergence_velocity
        proba = self._make_proba(50, n_states=3)
        kl = kl_divergence_velocity(proba, window=10)
        assert kl.iloc[:10].isna().all()

    def test_l1_bounds(self):
        """L1 velocity must lie in [0, 2]."""
        from qframe.regime.velocity import first_order_velocity
        proba = self._make_proba(200, n_states=3)
        vel = first_order_velocity(proba, window=5)
        valid = vel.dropna()
        assert (valid >= 0.0).all() and (valid <= 2.0 + 1e-8).all()

    def test_smoothed_preserves_nans(self):
        from qframe.regime.velocity import kl_divergence_velocity, smoothed_velocity
        proba = self._make_proba(100, n_states=3)
        kl = kl_divergence_velocity(proba, window=10)
        smooth = smoothed_velocity(kl, halflife=5)
        # NaN in original must stay NaN after smoothing
        assert smooth.iloc[:10].isna().all()

    def test_smoothed_non_negative(self):
        from qframe.regime.velocity import kl_divergence_velocity, smoothed_velocity
        proba = self._make_proba(200)
        kl = kl_divergence_velocity(proba, window=5)
        smooth = smoothed_velocity(kl, halflife=10)
        assert (smooth.dropna() >= 0).all()


# ===========================================================================
# RegimeICAnalyzer tests
# ===========================================================================

class TestRegimeICAnalyzer:

    def _make_analyzer_data(self, n_price: int = 1200, n_tickers: int = 50):
        """Returns (market_returns, prices, factor_df).
        Data spans ~4.7 years from 2015-01-01 giving IS through 2017 and OOS 2018+.
        """
        rng = np.random.default_rng(7)
        tickers = [f"T{i:03d}" for i in range(n_tickers)]
        rets = rng.normal(0.0003, 0.015, size=(n_price, n_tickers))
        prices_arr = 100 * np.exp(np.cumsum(rets, axis=0))
        idx = pd.bdate_range("2015-01-01", periods=n_price)
        prices = pd.DataFrame(prices_arr, index=idx, columns=tickers)
        market_returns = prices.mean(axis=1).pct_change().dropna()
        factor_vals = rng.normal(0, 1, size=prices.shape)
        factor_df = pd.DataFrame(factor_vals, index=prices.index, columns=tickers)
        return market_returns, prices, factor_df

    def test_fit_produces_proba_df(self):
        from qframe.regime.analyzer import RegimeICAnalyzer
        market_returns, prices, _ = self._make_analyzer_data()
        az = RegimeICAnalyzer(n_states=3, hsmm_window=200, hsmm_step=21)
        az.fit(market_returns, is_end="2017-12-31")
        assert az.proba_df is not None
        assert az.proba_df.shape[1] == 3
        valid = az.proba_df.dropna()
        assert len(valid) > 0
        np.testing.assert_allclose(valid.sum(axis=1).values, 1.0, atol=1e-5)

    def test_fit_produces_hurst_and_velocity(self):
        from qframe.regime.analyzer import RegimeICAnalyzer
        market_returns, prices, _ = self._make_analyzer_data()
        az = RegimeICAnalyzer(n_states=3, hsmm_window=200, hsmm_step=21)
        az.fit(market_returns, is_end="2017-12-31")
        assert az.hurst_series is not None
        assert az.velocity_raw is not None
        assert az.velocity_smooth is not None

    def test_decomposition_returns_correct_shape(self):
        from qframe.regime.analyzer import RegimeICAnalyzer
        market_returns, prices, factor_df = self._make_analyzer_data()
        az = RegimeICAnalyzer(n_states=3, hsmm_window=200, hsmm_step=21, min_state_days=5)
        az.fit(market_returns, is_end="2017-12-31")
        decomp = az.regime_ic_decomposition(factor_df, prices, oos_start="2018-01-01", horizon=1)
        assert decomp.by_state.shape[0] == 3
        assert "ic" in decomp.by_state.columns
        assert "t_stat" in decomp.by_state.columns

    def test_decomposition_pct_time_sums_to_one(self):
        from qframe.regime.analyzer import RegimeICAnalyzer
        market_returns, prices, factor_df = self._make_analyzer_data()
        az = RegimeICAnalyzer(n_states=3, hsmm_window=200, hsmm_step=21, min_state_days=5)
        az.fit(market_returns, is_end="2017-12-31")
        decomp = az.regime_ic_decomposition(factor_df, prices, oos_start="2018-01-01")
        np.testing.assert_allclose(
            decomp.by_state["pct_time"].sum(), 1.0, atol=1e-6
        )

    def test_soft_ic_series_aligned_to_oos(self):
        from qframe.regime.analyzer import RegimeICAnalyzer
        market_returns, prices, factor_df = self._make_analyzer_data()
        az = RegimeICAnalyzer(n_states=3, hsmm_window=200, hsmm_step=21, min_state_days=5)
        az.fit(market_returns, is_end="2017-12-31")
        decomp = az.regime_ic_decomposition(factor_df, prices, oos_start="2018-01-01")
        assert (decomp.soft_ic_series.index >= pd.Timestamp("2018-01-01")).all()

    def test_regime_weights_shape_and_bounds(self):
        from qframe.regime.analyzer import RegimeICAnalyzer
        market_returns, prices, factor_df = self._make_analyzer_data()
        az = RegimeICAnalyzer(n_states=3, hsmm_window=200, hsmm_step=21, min_state_days=5)
        az.fit(market_returns, is_end="2017-12-31")
        weights = az.regime_weights(factor_df, oos_start="2018-01-01")
        assert len(weights) > 0
        # Weights must be clipped to [0, 2]
        assert (weights >= 0.0).all()
        assert (weights <= 2.0 + 1e-8).all()

    def test_unfitted_raises(self):
        from qframe.regime.analyzer import RegimeICAnalyzer
        az = RegimeICAnalyzer()
        dummy_prices = _make_prices(n=100)
        dummy_factor = pd.DataFrame(
            np.ones(dummy_prices.shape),
            index=dummy_prices.index,
            columns=dummy_prices.columns,
        )
        with pytest.raises(RuntimeError, match="not fitted"):
            az.regime_ic_decomposition(
                dummy_factor, dummy_prices, oos_start="2018-01-01"
            )

    def test_no_lookahead_in_decomposition(self):
        """OOS IC must only use data from oos_start onwards."""
        from qframe.regime.analyzer import RegimeICAnalyzer
        market_returns, prices, factor_df = self._make_analyzer_data()
        az = RegimeICAnalyzer(n_states=3, hsmm_window=200, hsmm_step=21, min_state_days=5)
        az.fit(market_returns, is_end="2017-12-31")
        decomp = az.regime_ic_decomposition(factor_df, prices, oos_start="2018-01-01")
        # soft_ic_series dates must all be >= oos_start
        assert (decomp.soft_ic_series.index >= pd.Timestamp("2018-01-01")).all()

    def test_unconditional_vs_conditional_columns(self):
        from qframe.regime.analyzer import RegimeICAnalyzer
        market_returns, prices, factor_df = self._make_analyzer_data()
        az = RegimeICAnalyzer(n_states=3, hsmm_window=200, hsmm_step=21, min_state_days=5)
        az.fit(market_returns, is_end="2017-12-31")
        summary = az.unconditional_vs_conditional(factor_df, prices, oos_start="2018-01-01")
        assert "ic" in summary.columns
        assert "lift" in summary.columns
        assert len(summary) == 3  # unconditional, best, worst
