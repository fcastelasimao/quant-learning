"""Tests for src/metrics.py — including the critical lookahead test."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metrics import REGISTRY, compute_all, vote_all


PHYSICS_WATCH_METRICS = {
    "qqq_sign_entropy_20d",
    "qqq_return_entropy_60d",
    "qqq_sample_entropy_60d",
    "qqq_entropy_return_ratio_60d",
    "vix_to_qqq_transfer_entropy_252d",
    "qqq_hurst_126d",
    "qqq_lppl_curvature_126d",
    "rmt_market_mode_126d",
    "rmt_mean_abs_corr_126d",
    "vix_implied_realized_gap_20d",
    "vix_vol_of_vol_20d",
    "cross_asset_herding_alignment_20d",
    "qqq_kelly_fraction_252d",
}


def test_registry_not_empty():
    assert len(REGISTRY) > 10


def test_physics_metrics_start_as_watch_only():
    assert PHYSICS_WATCH_METRICS <= set(REGISTRY)
    for name in PHYSICS_WATCH_METRICS:
        metric = REGISTRY[name]
        assert metric.status == "watch"
        assert metric.family.startswith("physics_")


def test_all_metrics_return_series(synthetic_panel):
    results = compute_all(synthetic_panel)
    for name, s in results.items():
        assert isinstance(s, pd.Series), f"{name}: expected Series, got {type(s)}"


def test_all_series_aligned_to_panel_index(synthetic_panel):
    results = compute_all(synthetic_panel)
    for name, s in results.items():
        if s.empty:
            continue
        assert s.index.equals(synthetic_panel.index) or s.index.isin(synthetic_panel.index).all(), (
            f"{name}: index not aligned to panel"
        )


def test_no_nan_after_warmup(synthetic_panel):
    """After a 300-row warmup, no voting metric should be all-NaN."""
    warmup = 300
    results = compute_all(synthetic_panel)
    for name, m in REGISTRY.items():
        if m.status != "voting":
            continue
        s = results[name]
        if s.empty:
            continue
        tail = s.iloc[warmup:]
        if not tail.empty:
            assert not tail.isna().all(), f"{name}: all-NaN after warmup"


def test_votes_in_valid_set(synthetic_panel):
    votes = vote_all(synthetic_panel)
    for col in votes.columns:
        unique = set(votes[col].dropna().unique())
        assert unique <= {-1, 0, 1}, f"{col}: votes not in {{-1, 0, 1}}: {unique}"


def test_physics_metrics_produce_finite_watch_values_after_warmup(synthetic_panel):
    results = compute_all(synthetic_panel)
    for name in PHYSICS_WATCH_METRICS:
        tail = results[name].iloc[300:].replace([np.inf, -np.inf], np.nan).dropna()
        assert not tail.empty, f"{name}: no finite values after warmup"


def test_vote_dtype_is_int(synthetic_panel):
    votes = vote_all(synthetic_panel)
    for col in votes.columns:
        assert votes[col].dtype in (np.int64, np.int32, int), (
            f"{col}: vote dtype is {votes[col].dtype}"
        )


def test_lookahead_invariant(synthetic_panel):
    """Core correctness test: shuffling data after cutoff t must not change metric values up to t.

    For each voting metric:
      1. Compute on original panel → ref_val at t.
      2. Build a panel where all rows AFTER t are permuted (shuffled across time).
      3. Recompute → must still equal ref_val at t (within float tolerance).
    """
    rng = np.random.default_rng(99)
    cutoff = 500  # index position of the cutoff
    panel = synthetic_panel.copy()

    for name, metric in REGISTRY.items():
        s_orig = metric.compute(panel)
        if s_orig.empty:
            continue
        ref_val = s_orig.iloc[cutoff]
        if np.isnan(ref_val):
            continue  # can't test a NaN reference

        # Shuffle all rows after cutoff
        panel_shuffled = panel.copy()
        future_idx = slice(cutoff + 1, None)
        future_data = panel_shuffled.iloc[future_idx].values.copy()
        rng.shuffle(future_data)
        panel_shuffled.iloc[future_idx] = future_data

        s_shuffled = metric.compute(panel_shuffled)
        if s_shuffled.empty:
            continue

        shuffled_val = s_shuffled.iloc[cutoff]
        assert np.isclose(ref_val, shuffled_val, equal_nan=True), (
            f"LOOKAHEAD DETECTED in {name!r}: "
            f"value at cutoff changed from {ref_val:.6f} to {shuffled_val:.6f} "
            "after shuffling future data"
        )


def test_missing_symbol_returns_empty_or_nan(synthetic_panel):
    """Metrics that require optional symbols must not crash when columns are absent."""
    minimal = synthetic_panel[["TQQQ_close", "TQQQ_open", "TQQQ_high",
                                "TQQQ_low", "TQQQ_volume"]].copy()
    # Should not raise
    for name, m in REGISTRY.items():
        try:
            s = m.compute(minimal)
            assert isinstance(s, pd.Series), f"{name}: expected Series on partial panel"
        except Exception as exc:
            pytest.fail(f"{name}: raised {exc!r} on partial panel")
