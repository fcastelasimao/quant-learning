"""
tests/test_rebalance_thresholds.py
==================================
Unit tests for threshold-based rebalance research helpers.
"""

from __future__ import annotations

import pandas as pd
import pytest

from research.rebalance_thresholds import (
    DriftPolicy,
    run_threshold_comparison,
    rolling_policy_metrics,
    simulate_threshold_rebalance,
)


def _threshold_test_prices() -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    return pd.DataFrame(
        {
            "A": [100.0, 130.0, 130.0, 130.0],
            "B": [100.0, 100.0, 100.0, 100.0],
        },
        index=dates,
    )


def test_hybrid_threshold_uses_floor_or_relative_target_weight():
    policy = DriftPolicy(
        name="hybrid",
        kind="hybrid",
        floor_threshold=0.01,
        relative_threshold=0.20,
    )

    assert policy.threshold_for(0.03) == 0.01
    assert policy.threshold_for(0.20) == pytest.approx(0.04)


def test_threshold_rebalance_stays_fully_invested_and_rebalances_all_assets():
    prices = _threshold_test_prices()
    allocation = {"A": 0.50, "B": 0.50}
    policy = DriftPolicy("absolute_5pp", kind="absolute", absolute_threshold=0.05)

    result = simulate_threshold_rebalance(
        prices,
        allocation,
        policy,
        start_value=10_000.0,
        transaction_cost_pct=0.0,
    )

    assert result["Rebalanced"].sum() == 1
    assert (result["Cash"] == 0.0).all()

    rebalanced_row = result[result["Rebalanced"]].iloc[0]
    assert round(rebalanced_row["A Weight After (%)"], 6) == 50.0
    assert round(rebalanced_row["B Weight After (%)"], 6) == 50.0


def test_threshold_comparison_contains_only_full_rebalance_actions():
    prices = _threshold_test_prices()
    allocation = {"A": 0.50, "B": 0.50}

    summary, values, diagnostics = run_threshold_comparison(
        prices,
        allocation,
        start_value=10_000.0,
        transaction_cost_pct=0.0,
        tax_drag_pct=0.0,
    )

    assert "per_asset" not in set(summary["Rebalance Action"])
    assert "per_asset" not in set(diagnostics["Rebalance Action"])
    assert "absolute_5pp_portfolio | full_on_breach" in values.columns


def test_rolling_policy_metrics_reports_windows_and_activity():
    prices = _threshold_test_prices()
    allocation = {"A": 0.50, "B": 0.50}
    _, _, diagnostics = run_threshold_comparison(
        prices,
        allocation,
        start_value=10_000.0,
        transaction_cost_pct=0.0,
        tax_drag_pct=0.0,
    )

    rolling = rolling_policy_metrics(diagnostics, windows_months=(3,))

    assert {"Date", "Label", "Window", "Rolling Calmar", "Rolling Rebalance Count"} <= set(rolling.columns)
    assert set(rolling["Window"]) == {"3M"}
    policy_rows = rolling[rolling["Label"] == "5pp portfolio"]
    assert not policy_rows.empty
    assert policy_rows["Rolling Rebalance Count"].max() >= 1
