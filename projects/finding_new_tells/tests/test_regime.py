"""Tests for src/fnt/regime.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime import HSMM5, build_regime_features, walk_forward_states


@pytest.fixture(scope="module")
def small_X() -> np.ndarray:
    """300-row 2-feature synthetic data for fast HSMM tests."""
    rng = np.random.default_rng(7)
    n = 300
    # Two alternating regimes: bull (positive mean) and bear (negative mean)
    labels = (np.arange(n) // 60) % 2  # blocks of 60
    X = np.column_stack([
        labels * 0.001 + rng.normal(0, 0.01, n),   # log-ret
        0.15 + labels * 0.05 + rng.normal(0, 0.02, n),  # rv
    ])
    return X


@pytest.fixture(scope="module")
def fitted_model(small_X) -> HSMM5:
    m = HSMM5(n_states=5, max_dur=60, n_iter=20, random_state=0)
    m.fit(small_X)
    return m


# ---------------------------------------------------------------------------
# Basic fit checks
# ---------------------------------------------------------------------------

def test_fit_returns_self(small_X):
    m = HSMM5(n_states=5, max_dur=60, n_iter=5, random_state=1)
    result = m.fit(small_X)
    assert result is m


def test_fitted_params_shapes(fitted_model, small_X):
    N, F = 5, small_X.shape[1]
    assert fitted_model.means_.shape == (N, F)
    assert fitted_model.log_vars_.shape == (N, F)
    assert fitted_model.A_.shape == (N, N)
    assert fitted_model.pi_.shape == (N,)
    assert fitted_model.dur_lambda_.shape == (N,)


def test_transition_matrix_valid(fitted_model):
    A = fitted_model.A_
    # No self-loops (diagonal near zero)
    assert np.all(np.diag(A) < 1e-6)
    # Rows sum to 1
    assert np.allclose(A.sum(axis=1), 1.0, atol=1e-6)


def test_state_labels_assigned(fitted_model):
    assert fitted_model.state_labels_ is not None
    assert len(fitted_model.state_labels_) == 5
    assert fitted_model.state_labels_[0] == "strong_bull"
    assert fitted_model.state_labels_[-1] == "strong_bear"


# ---------------------------------------------------------------------------
# State ordering: monotone in conditional mean return after relabeling
# ---------------------------------------------------------------------------

def test_state_ordering_monotone(fitted_model, small_X):
    """After relabeling, conditional mean return must be non-increasing."""
    states = fitted_model.predict(small_X)
    mean_rets = np.array([
        small_X[states == j, 0].mean() if (states == j).any() else 0.0
        for j in range(5)
    ])
    assert np.all(np.diff(mean_rets) <= 0.01), (
        f"State ordering not monotone: mean_rets = {mean_rets}"
    )


# ---------------------------------------------------------------------------
# predict / predict_proba shape + validity
# ---------------------------------------------------------------------------

def test_predict_shape(fitted_model, small_X):
    s = fitted_model.predict(small_X)
    assert s.shape == (len(small_X),)
    assert s.dtype in (np.int32, np.int64, int)


def test_predict_states_in_range(fitted_model, small_X):
    s = fitted_model.predict(small_X)
    assert np.all((s >= 0) & (s < 5))


def test_predict_proba_shape(fitted_model, small_X):
    p = fitted_model.predict_proba(small_X)
    assert p.shape == (len(small_X), 5)


def test_predict_proba_sums_to_1(fitted_model, small_X):
    p = fitted_model.predict_proba(small_X)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)


def test_predict_proba_non_negative(fitted_model, small_X):
    p = fitted_model.predict_proba(small_X)
    assert np.all(p >= -1e-9)


# ---------------------------------------------------------------------------
# Stability under tiny noise: state sequence must not flip completely
# ---------------------------------------------------------------------------

def test_stability_under_small_noise(fitted_model, small_X):
    """Perturbing inputs by 1e-4 must not flip >30% of state assignments."""
    rng = np.random.default_rng(99)
    X_noisy = small_X + rng.normal(0, 1e-4, small_X.shape)
    s_orig = fitted_model.predict(small_X)
    s_noisy = fitted_model.predict(X_noisy)
    agreement = (s_orig == s_noisy).mean()
    assert agreement > 0.70, f"Only {agreement:.1%} agreement under tiny noise"


# ---------------------------------------------------------------------------
# No-lookahead: refit at t does not use data after t
# ---------------------------------------------------------------------------

def test_walk_forward_no_lookahead(synthetic_panel):
    """Values of state_series up to cutoff must not change when future rows are shuffled."""
    rng = np.random.default_rng(42)
    panel = synthetic_panel.copy()
    cutoff = 600  # must be > min_train_years * 252 (= 5*252 = 1260) — use shorter min

    # Use min_train_years=1 so we actually get states for cutoff=600
    state_series, _ = walk_forward_states(panel, min_train_years=1, n_iter=5, max_dur=60)
    ref_val = state_series.iloc[cutoff]

    # Shuffle rows after cutoff
    panel_shuffled = panel.copy()
    future = panel_shuffled.iloc[cutoff + 1:].values.copy()
    rng.shuffle(future)
    panel_shuffled.iloc[cutoff + 1:] = future

    state_shuffled, _ = walk_forward_states(
        panel_shuffled, min_train_years=1, n_iter=5, max_dur=60
    )
    assert state_shuffled.iloc[cutoff] == ref_val, (
        f"Walk-forward state at cutoff changed after shuffling future: "
        f"{ref_val} → {state_shuffled.iloc[cutoff]}"
    )


# ---------------------------------------------------------------------------
# build_regime_features
# ---------------------------------------------------------------------------

def test_build_regime_features_shape(synthetic_panel):
    X = build_regime_features(synthetic_panel)
    assert X.shape == (len(synthetic_panel), 2)


def test_build_regime_features_no_inf(synthetic_panel):
    X = build_regime_features(synthetic_panel)
    assert np.all(np.isfinite(X))


def test_build_regime_features_missing_qqq():
    panel = pd.DataFrame({"SPY_close": [1.0, 2.0, 3.0]})
    with pytest.raises(KeyError):
        build_regime_features(panel)
