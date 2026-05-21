from datetime import datetime

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("yfinance")

from engine.leverage import (
    OverlaySpec,
    apply_overlay_to_base,
    build_overlay_positions,
    compute_rsi,
    generate_hysteresis_signal,
    selected_window_metrics,
)
from research.build_leverage_comparison_report import build_report_bundle
from research.build_leverage_comparison_report import DEFAULT_ENTRY_GRID, DEFAULT_EXIT_GRID
from research.build_mixed_leverage_report import build_mixed_report_bundle


EXPECTED_LEVERAGE_ARTIFACTS = {
    "manifest.json",
    "price_provenance.json",
    "daily_series.csv",
    "monthly_returns.csv",
    "summary_metrics.csv",
    "calendar_year_metrics.csv",
    "rolling_metrics.csv",
    "drawdown_events.csv",
    "stress_period_metrics.csv",
    "signal_history.csv",
    "overlay_diagnostics.csv",
    "threshold_grid.csv",
    "capital_view.csv",
    "overlay_summary.csv",
    "leverage_summary.csv",
    "yearly_overlay_metrics.csv",
}

EXPECTED_MIXED_LEVERAGE_ARTIFACTS = {
    "manifest.json",
    "price_provenance.json",
    "daily_series.csv",
    "monthly_returns.csv",
    "summary_metrics.csv",
    "stress_period_metrics.csv",
    "overlay_diagnostics.csv",
    "signal_history.csv",
    "position_history.csv",
    "mixed_overlay_summary.csv",
}


def test_rsi_handles_directional_and_flat_paths():
    dates = pd.bdate_range("2020-01-01", periods=30)
    up = pd.Series(np.arange(100.0, 130.0), index=dates)
    down = pd.Series(np.arange(130.0, 100.0, -1.0), index=dates)
    flat = pd.Series(100.0, index=dates)

    assert compute_rsi(up, lookback=14).dropna().iloc[-1] == 100.0
    assert compute_rsi(down, lookback=14).dropna().iloc[-1] == 0.0
    assert compute_rsi(flat, lookback=14).dropna().iloc[-1] == 50.0


def test_hysteresis_enters_and_exits_only_at_thresholds():
    indicator = pd.Series([np.nan, 31.0, 29.0, 35.0, 49.0, 51.0, 28.0])
    signal = generate_hysteresis_signal(indicator, entry_threshold=30.0, exit_threshold=50.0)

    assert signal.tolist() == [0, 0, 1, 1, 1, 0, 1]


def test_default_threshold_grid_is_fine_and_extends_exit_to_70():
    assert DEFAULT_ENTRY_GRID[0] == 20.0
    assert DEFAULT_ENTRY_GRID[-1] == 36.0
    assert DEFAULT_EXIT_GRID[0] == 40.0
    assert DEFAULT_EXIT_GRID[-1] == 70.0
    assert set(np.diff(DEFAULT_ENTRY_GRID)) == {2.0}
    assert set(np.diff(DEFAULT_EXIT_GRID)) == {2.0}


def test_overlay_positions_are_lagged_and_global_cap_is_respected():
    dates = pd.bdate_range("2020-01-01", periods=8)
    prices = pd.DataFrame({
        "SPY": [100, 99, 98, 97, 96, 95, 94, 93],
        "TLT": [100, 99, 98, 97, 96, 95, 94, 93],
    }, index=dates)
    specs = [
        OverlaySpec("SPY", lookback=2, entry_threshold=30, exit_threshold=50, overlay_weight=0.20),
        OverlaySpec("TLT", lookback=2, entry_threshold=30, exit_threshold=50, overlay_weight=0.20),
    ]

    positions, raw, _, _ = build_overlay_positions(prices, specs, global_cap=0.20, execution_lag=1)
    first_active = raw.index[raw["SPY"].eq(1)][0]
    first_active_pos = raw.index.get_loc(first_active)
    next_date = raw.index[first_active_pos + 1]

    assert positions.loc[first_active, "SPY"] == 0.0
    assert positions.loc[next_date, "SPY"] > 0.0
    assert positions.sum(axis=1).max() <= 0.20 + 1e-12
    assert abs(positions.loc[next_date].sum() - 0.20) < 1e-12


def test_overlay_return_math_uses_applied_weight_times_asset_return():
    dates = pd.bdate_range("2020-01-01", periods=8)
    base = pd.Series(100.0, index=dates, name="base")
    prices = pd.DataFrame({"SPY": [100, 99, 98, 97, 96, 97, 98, 99]}, index=dates)
    spec = OverlaySpec("SPY", lookback=2, entry_threshold=30, exit_threshold=50, overlay_weight=0.20)

    result = apply_overlay_to_base(base, prices, [spec], global_cap=0.20)
    expected_overlay = result.positions["SPY"] * prices["SPY"].pct_change().fillna(0.0)
    expected_overlay.index.name = "Date"

    pd.testing.assert_series_equal(
        result.daily_diagnostics["Overlay Return"],
        expected_overlay.rename("Overlay Return"),
        check_names=True,
    )
    pd.testing.assert_series_equal(
        result.daily_diagnostics["Strategy Return"],
        expected_overlay.rename("Strategy Return"),
        check_names=True,
    )


def test_leverage_report_builder_writes_expected_bundle(tmp_path):
    prices = _synthetic_prices()
    allocation = {
        "SPY": 0.134,
        "QQQ": 0.103,
        "TLT": 0.175,
        "TIP": 0.348,
        "GLD": 0.142,
        "GSG": 0.098,
    }

    bundle = build_report_bundle(
        prices=prices,
        strategy_id="synthetic_leverage",
        allocation=allocation,
        output_root=tmp_path,
        start_date="2019-01-01",
        end_date="2021-12-31",
        generated_at=datetime(2026, 1, 2, 3, 4, 5),
        entry_grid=(25.0, 30.0, 35.0),
        exit_grid=(45.0, 50.0, 55.0),
        leverage_grid=(0.15, 0.20, 0.25),
    )

    assert {path.name for path in bundle.iterdir()} == EXPECTED_LEVERAGE_ARTIFACTS

    daily = pd.read_csv(bundle / "daily_series.csv")
    signals = pd.read_csv(bundle / "signal_history.csv")
    grid = pd.read_csv(bundle / "threshold_grid.csv")
    capital = pd.read_csv(bundle / "capital_view.csv")
    overlay_summary = pd.read_csv(bundle / "overlay_summary.csv")
    leverage_summary = pd.read_csv(bundle / "leverage_summary.csv")
    yearly = pd.read_csv(bundle / "yearly_overlay_metrics.csv")

    expected_tickers = set(allocation)
    expected_strategies = {
        f"My Strategy + {ticker} RSI Overlay"
        for ticker in expected_tickers
    }

    assert {"My Strategy (Base)", "S&P 500 (SPY)"} | expected_strategies <= set(daily["Strategy"])
    assert expected_tickers <= set(signals["Ticker"])
    assert {"Date", "Overlay Strategy", "Ticker", "RSI", "Raw Signal", "Applied Overlay Weight"} <= set(signals.columns)
    assert not grid.empty
    assert expected_tickers <= set(grid["Ticker"])
    assert {"Calmar", "Max Drawdown (%)", "Active Days (%)", "Overlay Weight", "Incremental Calmar"} <= set(grid.columns)
    assert expected_tickers <= set(capital["Ticker"])
    assert expected_tickers <= set(overlay_summary["Ticker"])
    assert expected_tickers <= set(leverage_summary["Ticker"])
    assert expected_tickers <= set(yearly["Ticker"])
    valid_threshold_pairs = sum(1 for entry in (25.0, 30.0, 35.0) for exit_ in (45.0, 50.0, 55.0) if exit_ > entry)
    assert len(grid) == len(expected_tickers) * valid_threshold_pairs * 3
    assert set(grid["Overlay Weight"].round(2)) == {0.15, 0.20, 0.25}
    assert "Default 20% 30/50" in set(leverage_summary["Selection"])


def test_mixed_leverage_report_builder_writes_capped_spy_gld_bundle(tmp_path):
    prices = _synthetic_prices()
    allocation = {
        "SPY": 0.134,
        "QQQ": 0.103,
        "TLT": 0.175,
        "TIP": 0.348,
        "GLD": 0.142,
        "GSG": 0.098,
    }

    bundle = build_mixed_report_bundle(
        prices=prices,
        strategy_id="synthetic_mixed_leverage",
        allocation=allocation,
        output_root=tmp_path,
        start_date="2019-01-01",
        end_date="2021-12-31",
        generated_at=datetime(2026, 1, 2, 3, 4, 5),
    )

    assert {path.name for path in bundle.iterdir()} == EXPECTED_MIXED_LEVERAGE_ARTIFACTS

    daily = pd.read_csv(bundle / "daily_series.csv")
    signals = pd.read_csv(bundle / "signal_history.csv")
    diagnostics = pd.read_csv(bundle / "overlay_diagnostics.csv")
    positions = pd.read_csv(bundle / "position_history.csv")
    summary = pd.read_csv(bundle / "mixed_overlay_summary.csv")

    assert "SPY+GLD default 20% total cap" in set(daily["Strategy"])
    assert {"SPY", "GLD"} <= set(signals["Ticker"])
    assert {"Overlay Strategy", "Tickers", "Global Cap", "Overlay Exposure"} <= set(diagnostics.columns)
    assert {"Overlay Strategy", "Ticker", "Applied Overlay Weight"} <= set(positions.columns)
    assert {"Both Active Days (%)", "Average SPY Weight (%)", "Average GLD Weight (%)"} <= set(summary.columns)

    mixed_diag = diagnostics[diagnostics["Overlay Strategy"] == "SPY+GLD default 20% total cap"]
    assert not mixed_diag.empty
    assert mixed_diag["Overlay Exposure"].max() <= 0.20 + 1e-12


def test_leverage_notebook_is_presentation_only():
    notebook = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "notebooks"
        / "leverage_comparison.py"
    )
    source = notebook.read_text()

    forbidden = ["yfinance", "fetch_prices", "build_monthly_rebalanced_series", "apply_overlay_to_base"]
    for token in forbidden:
        assert token not in source

    assert "threshold_grid.csv" in source
    assert "capital_view.csv" in source
    assert "overlay_summary.csv" in source
    assert "leverage_summary.csv" in source
    assert "yearly_overlay_metrics.csv" in source
    assert "pass_fail_summary.csv" in source
    assert "mixed_leverage_sweep" in source
    assert "Broker limit profile" in source
    assert "IBKR Safe" in source
    assert "Strict Pilot" in source
    assert "Research Unrestricted" in source
    assert 'value="Exit x Entry"' in source
    assert "Disciplined SPY+GLD Sweep" in source
    assert "is_sweep_grid.parquet" in source
    assert "walk_forward_summary.csv" in source
    assert "parameter_stability.csv" in source
    assert "sweep_heatmap_tables.csv" in source
    assert "fixed_candidate_walk_forward_summary.csv" in source
    assert "Best Calmar Candidate vs Base" in source
    assert "Calmar-First Candidate Leaderboard" in source
    assert "control only" in source
    assert 'BENCHMARK = "S&P 500 (SPY)"' in source
    assert "PLOT_EXCLUDED = {BENCHMARK}" in source
    assert "Signal ETF" not in source
    assert "Executive Summary" in source
    assert "SPY Deep Dive" in source
    assert "GLD Threshold Grid CSV Rows: Top by Calmar" in source
    assert "SPY Threshold Grid CSV Rows: Top by Calmar" in source
    assert "Selected GLD Rules Above 50% Leverage" in source
    assert "Selected GLD Rules At or Below 50% Leverage" in source
    assert "Selected SPY Rules Above 50% Leverage" not in source
    assert "Selected SPY Rules At or Below 50% Leverage" not in source
    assert "plot_grid_heatmap" in source
    assert "plot_all_etf_rsi_small_multiples" in source
    assert "appendix_threshold_controls" in source
    assert "Inspect ETF" in source
    assert "Exit x Entry" in source
    assert "Leverage %" in source


def test_selected_window_metrics_recomputes_window_from_daily_exports():
    dates = pd.bdate_range("2020-01-01", periods=6)
    daily = pd.DataFrame({
        "Date": list(dates) * 2,
        "Strategy": ["Base"] * 6 + ["Overlay"] * 6,
        "Value": [100, 102, 101, 103, 104, 106, 100, 104, 102, 108, 107, 111],
    })
    diagnostics = pd.DataFrame({
        "Date": dates,
        "Ticker": ["SPY"] * 6,
        "Overlay Strategy": ["Overlay"] * 6,
        "Overlay Return": [0, 0.01, -0.005, 0.015, 0, 0.01],
        "Overlay Exposure": [0, 0.2, 0.2, 0.2, 0, 0],
    })

    out = selected_window_metrics(
        daily,
        diagnostics,
        start=dates[1],
        end=dates[-1],
    )
    overlay = out[out["Strategy"] == "Overlay"].iloc[0]
    base = out[out["Strategy"] == "Base"].iloc[0]

    assert overlay["Observations"] == 5
    assert round(overlay["Total Return (%)"], 4) == round((111 / 104 - 1) * 100, 4)
    assert overlay["Active Days"] == 3
    assert round(overlay["Average Overlay Exposure (%)"], 4) == 12.0
    assert round(overlay["Overlay Return Contribution (%)"], 4) == 3.0
    assert base["Active Days"] == 0


def _synthetic_prices() -> pd.DataFrame:
    dates = pd.bdate_range("2019-01-01", "2021-12-31")
    n = len(dates)
    cycle = np.sin(np.arange(n) / 12) / 180
    selloff = np.zeros(n)
    selloff[90:110] = -0.018
    selloff[110:135] = 0.012
    spy_returns = 0.00025 + cycle + selloff

    return pd.DataFrame({
        "SPY": 100 * np.cumprod(1 + spy_returns),
        "QQQ": 100 * np.cumprod(1 + spy_returns * 1.15 + 0.00005),
        "TLT": 100 * np.cumprod(1 + 0.00010 - cycle * 0.35),
        "TIP": 100 * np.cumprod(1 + 0.00008 - cycle * 0.15),
        "GLD": 100 * np.cumprod(1 + 0.00012 + np.cos(np.arange(n) / 20) / 1000),
        "GSG": 100 * np.cumprod(1 + 0.00005 + np.sin(np.arange(n) / 17) / 1000),
    }, index=dates)
