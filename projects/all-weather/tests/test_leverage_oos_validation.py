from datetime import datetime

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("yfinance")

from research.validate_leverage_oos import (
    GLD_EXTENDED_LEVERAGE_GRID,
    build_oos_validation_bundle,
    build_pass_fail_summary,
    select_rules,
    split_pass_flags,
)


EXPECTED_OOS_ARTIFACTS = {
    "manifest.json",
    "price_provenance.json",
    "is_threshold_grid.csv",
    "selected_rules.csv",
    "oos_summary.csv",
    "oos_daily_series.csv",
    "oos_signal_history.csv",
    "oos_overlay_diagnostics.csv",
    "oos_stress_metrics.csv",
    "oos_trade_episodes.csv",
    "pass_fail_summary.csv",
}


def test_oos_validation_bundle_writes_expected_artifacts_and_grids(tmp_path):
    allocation = _allocation()
    bundle = build_oos_validation_bundle(
        prices=_synthetic_prices(),
        strategy_id="synthetic_oos",
        allocation=allocation,
        output_root=tmp_path,
        start_date="2016-01-01",
        end_date="2023-12-31",
        generated_at=datetime(2026, 1, 2, 3, 4, 5),
        entry_grid=(30.0,),
        exit_grid=(50.0,),
        leverage_grid=(0.20,),
        gld_leverage_grid=(0.20, 1.00),
    )

    assert {path.name for path in bundle.iterdir()} == EXPECTED_OOS_ARTIFACTS

    grid = pd.read_csv(bundle / "is_threshold_grid.csv")
    selected = pd.read_csv(bundle / "selected_rules.csv")
    summary = pd.read_csv(bundle / "oos_summary.csv")
    daily = pd.read_csv(bundle / "oos_daily_series.csv")
    signals = pd.read_csv(bundle / "oos_signal_history.csv")
    diagnostics = pd.read_csv(bundle / "oos_overlay_diagnostics.csv")
    stress = pd.read_csv(bundle / "oos_stress_metrics.csv")
    pass_fail = pd.read_csv(bundle / "pass_fail_summary.csv")

    assert set(allocation) <= set(grid["Ticker"])
    assert set(allocation) <= set(summary["Ticker"])
    assert {"2018", "2020", "2022"} <= set(grid["Split"].astype(str))
    assert grid[grid["Ticker"] == "GLD"]["Overlay Weight"].max() == 1.0
    assert grid[grid["Ticker"] != "GLD"]["Overlay Weight"].max() == 0.2
    assert GLD_EXTENDED_LEVERAGE_GRID[-1] == 1.0

    assert {"default_30_50_20", "best_calmar", "robust_calmar_region"} <= set(selected["Selector"])
    assert (pd.to_datetime(selected["IS End Date"]) < pd.to_datetime(selected["OOS Start"])).all()
    assert {"Split", "OOS Start", "Ticker", "Selector"} <= set(summary.columns)
    assert {"Pass Split", "Low Trade Count Flag", "OOS RF Opportunity Cost CAGR (%)"} <= set(summary.columns)
    assert {"Split", "OOS Start", "Ticker", "Selector", "Date", "Value"} <= set(daily.columns)
    assert {"Split", "OOS Start", "Overlay Strategy", "Selector", "Date", "RSI"} <= set(signals.columns)
    assert {"Split", "OOS Start", "Overlay Strategy", "Ticker", "Selector", "Date"} <= set(diagnostics.columns)
    assert {"Split", "OOS Start", "Ticker", "Selector", "Period", "Strategy"} <= set(stress.columns)
    assert {"Ticker", "Selector", "Splits Passed", "Overall Pass"} <= set(pass_fail.columns)


def test_is_grid_dates_are_strictly_before_oos_split(tmp_path):
    """IS threshold grid must be built from pre-split prices only (no lookahead)."""
    split = "2020-01-01"
    bundle = build_oos_validation_bundle(
        prices=_synthetic_prices(),
        strategy_id="lookahead_check",
        allocation=_allocation(),
        output_root=tmp_path,
        start_date="2016-01-01",
        end_date="2023-12-31",
        generated_at=datetime(2026, 1, 1),
        splits=(split,),
        entry_grid=(30.0,),
        exit_grid=(50.0,),
        leverage_grid=(0.20,),
        gld_leverage_grid=(0.20,),
    )

    oos_ts = pd.Timestamp(split)
    grid = pd.read_csv(bundle / "is_threshold_grid.csv")
    is_end = pd.to_datetime(grid["IS End Date"])
    assert (is_end < oos_ts).all(), (
        f"IS grid contains end dates >= OOS split {split}: {grid[is_end >= oos_ts]['IS End Date'].tolist()}"
    )

    selected = pd.read_csv(bundle / "selected_rules.csv")
    sel_is_end = pd.to_datetime(selected["IS End Date"])
    assert (sel_is_end < oos_ts).all()


def test_select_rules_uses_fixed_selectors_and_robust_edge_neighborhood():
    grid = pd.DataFrame({
        "Ticker": ["GLD"] * 4,
        "Strategy": ["s"] * 4,
        "Lookback": [14] * 4,
        "Entry Threshold": [20.0, 20.0, 22.0, 24.0],
        "Exit Threshold": [40.0, 42.0, 42.0, 44.0],
        "Overlay Weight": [0.15, 0.20, 0.20, 0.25],
        "Overlay Weight (%)": [15.0, 20.0, 20.0, 25.0],
        "Active Days (%)": [1.0, 1.0, 1.0, 1.0],
        "Average Overlay Weight (%)": [0.2, 0.2, 0.2, 0.2],
        "Calmar": [0.3, 0.5, 0.4, 0.1],
        "CAGR (%)": [7.0, 7.2, 8.0, 9.0],
        "Max Drawdown (%)": [-20.0, -21.0, -19.0, -18.0],
    })
    base = {"Max Drawdown (%)": -20.0}

    selected = select_rules(grid, base)

    assert {"best_calmar", "best_maxdd_preservation", "best_cagr_with_maxdd_guard", "robust_calmar_region"} <= set(selected["Selector"])
    robust = selected[selected["Selector"] == "robust_calmar_region"].iloc[0]
    assert robust["Robust Neighborhood Size"] >= 2
    assert robust["Robust Avg Calmar"] > 0


def test_pass_fail_logic_allows_low_trade_count_as_caveat_and_summarises_gate():
    base = {"CAGR (%)": 6.0, "Calmar": 0.3, "Max Drawdown (%)": -20.0}
    overlay = {"CAGR (%)": 7.0, "Calmar": 0.4, "Max Drawdown (%)": -20.5}
    rf = {"CAGR (%)": 6.5}

    flags = split_pass_flags(base, overlay, rf, episode_count=1)

    assert flags["Pass Split"] is True
    assert flags["Low Trade Count Flag"] is True
    assert "low-trade-count" in flags["Pass Notes"]

    rows = pd.DataFrame([
        {"Ticker": "GLD", "Selector": "best", "Pass Split": True, "Low Trade Count Flag": True,
         "OOS Calmar Delta": 0.1, "OOS MaxDD Delta (%)": -0.5, "OOS CAGR Delta (%)": 1.0},
        {"Ticker": "GLD", "Selector": "best", "Pass Split": True, "Low Trade Count Flag": False,
         "OOS Calmar Delta": 0.05, "OOS MaxDD Delta (%)": -0.2, "OOS CAGR Delta (%)": 0.7},
        {"Ticker": "GLD", "Selector": "best", "Pass Split": False, "Low Trade Count Flag": False,
         "OOS Calmar Delta": -0.2, "OOS MaxDD Delta (%)": -2.0, "OOS CAGR Delta (%)": -0.4},
    ])
    summary = build_pass_fail_summary(rows).iloc[0]

    assert summary["Splits Passed"] == 2
    assert bool(summary["Overall Pass"]) is True


def _allocation() -> dict[str, float]:
    return {
        "SPY": 0.134,
        "QQQ": 0.103,
        "TLT": 0.175,
        "TIP": 0.348,
        "GLD": 0.142,
        "GSG": 0.098,
    }


def _synthetic_prices() -> pd.DataFrame:
    dates = pd.bdate_range("2016-01-01", "2023-12-31")
    n = len(dates)
    phase = np.arange(n)
    cycle = np.sin(phase / 18) / 220
    shock = np.zeros(n)
    shock[420:450] = -0.014
    shock[450:500] = 0.008
    shock[1040:1070] = -0.018
    shock[1070:1130] = 0.010
    spy_returns = 0.00020 + cycle + shock
    gold_returns = 0.00010 - cycle * 0.5 - shock * 0.25 + np.cos(phase / 23) / 1500

    return pd.DataFrame({
        "SPY": 100 * np.cumprod(1 + spy_returns),
        "QQQ": 100 * np.cumprod(1 + spy_returns * 1.15 + 0.00003),
        "TLT": 100 * np.cumprod(1 + 0.00008 - cycle * 0.4),
        "TIP": 100 * np.cumprod(1 + 0.00006 - cycle * 0.15),
        "GLD": 100 * np.cumprod(1 + gold_returns),
        "GSG": 100 * np.cumprod(1 + 0.00004 + np.sin(phase / 21) / 1200),
    }, index=dates)
