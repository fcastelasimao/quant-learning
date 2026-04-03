"""
Unit tests for the regime detection module.

Tests cover:
  - Feature matrix construction
  - Forward filter (causal, sums to 1)
  - Minimum duration filter
  - RegimeDetector fit + predict round-trip
  - regime_defense_scale mapping
  - BOCPD basic sanity
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime import (
    BOCPD,
    RegimeDetector,
    StudentTHMM,
    apply_min_duration,
    build_feature_matrix,
    forward_filter,
    regime_defense_scale,
)


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_price_series(n: int = 500, seed: int = 42) -> pd.Series:
    """Generate a synthetic daily price series with trending + mean-reverting regimes."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    # Random walk with drift changes to create regime-like behaviour
    returns = np.concatenate([
        rng.normal(0.001, 0.01, n // 3),   # bull
        rng.normal(-0.001, 0.02, n // 3),   # bear
        rng.normal(0.0, 0.008, n - 2 * (n // 3)),  # sideways
    ])
    prices = 100 * np.exp(np.cumsum(returns))
    return pd.Series(prices, index=dates, name="TEST")


def _make_vix_series(price_series: pd.Series, seed: int = 42) -> pd.Series:
    """Generate a synthetic VIX-like series inversely correlated with prices."""
    rng = np.random.RandomState(seed)
    n = len(price_series)
    base = 20 + rng.normal(0, 3, n)
    # VIX tends to spike when prices drop
    returns = price_series.pct_change().fillna(0).values
    vix = base - returns * 500  # negative price returns -> higher VIX
    vix = np.clip(vix, 10, 80)
    return pd.Series(vix, index=price_series.index, name="VIX")


# ── Tests ────────────────────────────────────────────────────────────────

class TestBuildFeatureMatrix:
    def test_returns_correct_columns(self):
        prices = _make_price_series()
        features = build_feature_matrix(prices, window=30)
        expected_cols = {"channel_slope", "channel_position", "channel_width_pct", "ewma_vol", "vol_index_log"}
        assert set(features.columns) == expected_cols

    def test_length_shorter_than_prices(self):
        prices = _make_price_series(200)
        features = build_feature_matrix(prices, window=30)
        # Features are computed on returns (n-1) and need window warmup
        assert len(features) == len(prices) - 1  # returns drop first row

    def test_with_vix(self):
        prices = _make_price_series()
        vix = _make_vix_series(prices)
        features = build_feature_matrix(prices, vix=vix, window=30)
        valid = features.dropna()
        assert len(valid) > 0
        assert not np.any(np.isinf(valid.values))

    def test_without_vix_uses_realised_vol(self):
        prices = _make_price_series()
        features = build_feature_matrix(prices, vix=None, window=30)
        valid = features.dropna()
        assert len(valid) > 0


class TestForwardFilter:
    def test_probabilities_sum_to_one(self):
        prices = _make_price_series(300)
        features = build_feature_matrix(prices, window=30)
        X = features.dropna().values
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = StudentTHMM(n_components=3, n_iter=50, random_state=0)
        model.fit(X_scaled)

        filtered = forward_filter(model, X_scaled)
        row_sums = filtered.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_output_shape(self):
        prices = _make_price_series(300)
        features = build_feature_matrix(prices, window=30)
        X = features.dropna().values
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = StudentTHMM(n_components=3, n_iter=50, random_state=0)
        model.fit(X_scaled)

        filtered = forward_filter(model, X_scaled)
        assert filtered.shape == (len(X_scaled), 3)


class TestMinDuration:
    def test_short_runs_merged(self):
        idx = pd.date_range("2020-01-01", periods=20)
        states = pd.Series(
            ["bull"] * 8 + ["bear"] * 2 + ["bull"] * 10,
            index=idx,
        )
        cleaned = apply_min_duration(states, min_days=5)
        # The 2-day bear run should be merged into bull
        assert (cleaned == "bull").all()

    def test_long_runs_preserved(self):
        idx = pd.date_range("2020-01-01", periods=20)
        states = pd.Series(
            ["bull"] * 10 + ["bear"] * 10,
            index=idx,
        )
        cleaned = apply_min_duration(states, min_days=5)
        assert (cleaned.iloc[:10] == "bull").all()
        assert (cleaned.iloc[10:] == "bear").all()

    def test_handles_nan(self):
        idx = pd.date_range("2020-01-01", periods=10)
        states = pd.Series([np.nan, np.nan] + ["bull"] * 8, index=idx)
        cleaned = apply_min_duration(states, min_days=3)
        assert cleaned.iloc[2:].tolist() == ["bull"] * 8


class TestRegimeDetector:
    def test_fit_predict_round_trip(self):
        prices = _make_price_series(400)
        vix = _make_vix_series(prices)

        detector = RegimeDetector(
            n_states=3,
            channel_window=30,
            n_restarts=5,
            min_duration=3,
        )
        detector.fit(prices.iloc[:300], vix.iloc[:300])
        proba = detector.predict(prices, vix)

        assert not proba.empty
        assert set(proba.columns) == {"bull", "sideways", "bear"}
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_current_regime_label(self):
        prices = _make_price_series(400)
        detector = RegimeDetector(n_states=3, channel_window=30, n_restarts=5)
        detector.fit(prices.iloc[:300])
        label = detector.current_regime_label(prices)
        assert label in {"bull", "sideways", "bear"}


class TestRegimeDefenseScale:
    def test_pure_bull(self):
        probs = pd.Series({"bull": 1.0, "sideways": 0.0, "bear": 0.0})
        assert regime_defense_scale(probs) == 1.0

    def test_pure_bear(self):
        probs = pd.Series({"bull": 0.0, "sideways": 0.0, "bear": 1.0})
        assert regime_defense_scale(probs) == 0.25

    def test_mixed_regime(self):
        probs = pd.Series({"bull": 0.5, "sideways": 0.3, "bear": 0.2})
        expected = 0.5 * 1.0 + 0.3 * 0.75 + 0.2 * 0.25
        assert abs(regime_defense_scale(probs) - expected) < 1e-6

    def test_clipped_to_range(self):
        probs = pd.Series({"bull": 0.0, "sideways": 0.0, "bear": 1.0})
        scale = regime_defense_scale(probs)
        assert 0.25 <= scale <= 1.0


class TestBOCPD:
    def test_output_length(self):
        data = np.random.randn(100)
        bocpd = BOCPD(hazard_rate=1 / 20)
        probs = bocpd.run(data)
        assert len(probs) == 100

    def test_probabilities_in_range(self):
        data = np.random.randn(100)
        bocpd = BOCPD(hazard_rate=1 / 20)
        probs = bocpd.run(data)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_detects_mean_shift(self):
        # Concatenate two different distributions — should detect changepoint
        rng = np.random.RandomState(42)
        data = np.concatenate([rng.normal(0, 1, 50), rng.normal(5, 1, 50)])
        bocpd = BOCPD(hazard_rate=1 / 20)
        probs = bocpd.run(data)
        # Changepoint probability should spike near index 50
        # BOCPD has inherent detection lag — the spike may appear a few
        # observations after the true changepoint at index 50
        peak_idx = np.argmax(probs[45:65]) + 45
        assert 45 <= peak_idx <= 60
