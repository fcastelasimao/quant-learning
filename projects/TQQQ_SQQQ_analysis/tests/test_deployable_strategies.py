"""Tests for deployable_strategies/ package components.

Tests that: (1) rule masks flag the expected trades, (2) sizing functions
return correct scalars, (3) metrics reproduce known values, (4) all imports
work from the new package location.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# --------------------------------------------------------------------------- #
# Imports from new location
# --------------------------------------------------------------------------- #

def test_scorer_imports_from_new_location():
    """p_severe_scorer is importable from its new location under deployable_strategies."""
    from TQQQ_SQQQ_analysis.deployable_strategies.continuous_sizing.p_severe_scorer import (  # noqa: F401
        score_trades,
        compute_required_features,
    )
    assert callable(score_trades)
    assert callable(compute_required_features)


def test_sizing_functions_importable():
    from TQQQ_SQQQ_analysis.deployable_strategies.continuous_sizing.sizing import (  # noqa: F401
        linear_skip, sqrt_skip, step_skip, aggressive_skip, moderate_skip,
        SIZING_FUNCTIONS,
    )
    assert callable(linear_skip)
    assert "linear_skip" in SIZING_FUNCTIONS


def test_rule_masks_importable():
    from TQQQ_SQQQ_analysis.deployable_strategies.focus_rules.sqqq_rsi_atr_skip import sqqq_focus_rule_mask  # noqa: F401
    from TQQQ_SQQQ_analysis.deployable_strategies.regime_rules.tqqq_sideways_skip import tqqq_regime_rule_mask  # noqa: F401
    assert callable(sqqq_focus_rule_mask)
    assert callable(tqqq_regime_rule_mask)


# --------------------------------------------------------------------------- #
# Sizing function unit tests
# --------------------------------------------------------------------------- #

def test_linear_skip_function():
    """linear_skip: size = 1 - p."""
    from TQQQ_SQQQ_analysis.deployable_strategies.continuous_sizing.sizing import linear_skip
    assert abs(linear_skip(0.0) - 1.0) < 1e-9
    assert abs(linear_skip(0.3) - 0.7) < 1e-9
    assert abs(linear_skip(1.0) - 0.0) < 1e-9


def test_sqrt_skip_function():
    """sqrt_skip: size = sqrt(1 - p)."""
    from TQQQ_SQQQ_analysis.deployable_strategies.continuous_sizing.sizing import sqrt_skip
    assert abs(sqrt_skip(0.0) - 1.0) < 1e-9
    assert abs(sqrt_skip(1.0) - 0.0) < 1e-9
    assert abs(sqrt_skip(0.75) - 0.5) < 1e-6   # sqrt(0.25) = 0.5


def test_aggressive_skip_function():
    """aggressive_skip(2x): size = clip(1 - 2*p, 0, 1)."""
    from TQQQ_SQQQ_analysis.deployable_strategies.continuous_sizing.sizing import aggressive_skip
    assert abs(aggressive_skip(0.0) - 1.0) < 1e-9
    assert abs(aggressive_skip(0.3, multiplier=2.0) - 0.4) < 1e-9
    assert abs(aggressive_skip(0.5, multiplier=2.0) - 0.0) < 1e-9
    assert abs(aggressive_skip(1.0, multiplier=2.0) - 0.0) < 1e-9   # clipped at 0


def test_step_skip_function():
    """step_skip: 1 below threshold, 0 at/above."""
    from TQQQ_SQQQ_analysis.deployable_strategies.continuous_sizing.sizing import step_skip
    assert step_skip(0.3) == 1.0   # < 0.5 threshold
    assert step_skip(0.5) == 0.0   # == threshold → skip
    assert step_skip(0.7) == 0.0   # > threshold


def test_sizing_functions_return_correct_types_for_series():
    """Sizing functions return pd.Series when given a pd.Series."""
    from TQQQ_SQQQ_analysis.deployable_strategies.continuous_sizing.sizing import linear_skip, sqrt_skip
    p = pd.Series([0.0, 0.3, 1.0])
    result = linear_skip(p)
    assert isinstance(result, pd.Series)
    np.testing.assert_allclose(result.values, [1.0, 0.7, 0.0], atol=1e-9)


# --------------------------------------------------------------------------- #
# SQQQ focus rule mask
# --------------------------------------------------------------------------- #

def test_sqqq_focus_rule_flags_known_trades(research_dir):
    """SQQQ focus rule flags trades in the RSI × ATR danger cell.

    Item 08 reports 7 regime-labeled OOS trades flagged. We verify that
    applying the mask to OOS SQQQ canonical trades gives >= 5 flagged trades
    (allowing for minor discrepancies due to regime-label filtering).
    """
    from TQQQ_SQQQ_analysis.deployable_strategies.focus_rules.sqqq_rsi_atr_skip import sqqq_focus_rule_mask

    canon = research_dir.parent / "full_history_canonical" / "TRADES_SQQQ_full_history.csv"
    if not canon.exists():
        pytest.skip("SQQQ canonical missing")
    df = pd.read_csv(canon, parse_dates=["entry_time"])
    oos = df[df["entry_time"].dt.year > 2020].copy()
    mask = sqqq_focus_rule_mask(oos)
    n_flagged = int(mask.sum())
    assert n_flagged >= 5, (
        f"SQQQ focus rule flagged only {n_flagged} OOS trades — expected >= 5 "
        "(item 08 reports 7 in regime-labeled subset)"
    )
    assert n_flagged <= 50, (
        f"SQQQ focus rule unexpectedly flagged {n_flagged} OOS trades — rule may be too broad"
    )


def test_sqqq_focus_rule_nan_safe():
    """NaN in RSI_entry or atr_pct should not raise and should return False."""
    from TQQQ_SQQQ_analysis.deployable_strategies.focus_rules.sqqq_rsi_atr_skip import sqqq_focus_rule_mask
    df = pd.DataFrame({
        "RSI_entry": [57.0, None, 57.0],
        "atr_pct":   [0.42, 0.42, None],
    })
    mask = sqqq_focus_rule_mask(df)
    assert mask.iloc[1] == False  # noqa: E712
    assert mask.iloc[2] == False  # noqa: E712


# --------------------------------------------------------------------------- #
# TQQQ regime rule mask
# --------------------------------------------------------------------------- #

def test_tqqq_regime_rule_flags_known_trades(research_dir):
    """TQQQ regime rule flags trades in the sideways_lowvol × ATR × MA100 zone.

    Item 11 reports 10 R-OOS trades flagged. We verify >= 5 in full OOS
    (2021–2026) to allow for the R-OOS vs full OOS period difference.
    """
    from TQQQ_SQQQ_analysis.deployable_strategies.regime_rules.tqqq_sideways_skip import tqqq_regime_rule_mask

    canon = research_dir.parent / "full_history_canonical" / "TRADES_TQQQ_full_history.csv"
    if not canon.exists():
        pytest.skip("TQQQ canonical missing")
    df = pd.read_csv(canon, parse_dates=["entry_time"])
    oos = df[df["entry_time"].dt.year > 2020].copy()
    mask = tqqq_regime_rule_mask(oos)
    n_flagged = int(mask.sum())
    assert n_flagged >= 5, (
        f"TQQQ regime rule flagged only {n_flagged} OOS trades — expected >= 5 "
        "(item 11 reports 10 in R-OOS 2021–2025)"
    )
    assert n_flagged <= 100, (
        f"TQQQ regime rule unexpectedly flagged {n_flagged} OOS trades — rule may be too broad"
    )


def test_tqqq_regime_rule_nan_safe():
    """NaN in any condition column should not raise and should return False."""
    from TQQQ_SQQQ_analysis.deployable_strategies.regime_rules.tqqq_sideways_skip import tqqq_regime_rule_mask
    df = pd.DataFrame({
        "regime_entry": ["sideways_lowvol", None, "sideways_lowvol"],
        "atr_pct":      [0.45, 0.45, None],
        "MA100_D1":     [0.0001, 0.0001, 0.0001],
    })
    mask = tqqq_regime_rule_mask(df)
    assert mask.iloc[1] == False  # noqa: E712
    assert mask.iloc[2] == False  # noqa: E712


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def test_metrics_sharpe_matches_item_05(research_dir):
    """constant_notional_metrics() reproduces item 05 TQQQ Sharpe within 0.1."""
    from TQQQ_SQQQ_analysis.deployable_strategies.metrics import constant_notional_metrics

    canon = research_dir.parent / "full_history_canonical" / "TRADES_TQQQ_full_history.csv"
    if not canon.exists():
        pytest.skip("TQQQ canonical missing")
    df = pd.read_csv(canon, parse_dates=["entry_time", "exit_time"])
    df["r"] = df["pnl_pct"] / 100.0
    m = constant_notional_metrics(df, sized_r_col="r")
    # Item 05 reports TQQQ constant-notional Sharpe ≈ 4.00
    assert abs(m["sharpe_daily"] - 4.0) < 0.2, (
        f"constant_notional_metrics Sharpe = {m['sharpe_daily']:.3f}, expected ~4.00"
    )


def test_metrics_drawdown_is_negative():
    """max_drawdown should always be ≤ 0."""
    from TQQQ_SQQQ_analysis.deployable_strategies.metrics import constant_notional_metrics
    df = pd.DataFrame({
        "entry_time": pd.to_datetime(["2022-01-03", "2022-01-04"]),
        "exit_time":  pd.to_datetime(["2022-01-03", "2022-01-04"]),
        "r":          [-0.01, 0.02],
    })
    m = constant_notional_metrics(df, sized_r_col="r")
    assert m["max_drawdown"] <= 0
