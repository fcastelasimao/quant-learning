import json
from datetime import datetime

import numpy as np
import pandas as pd

from research.production_validation.build_strategy_comparison_report import build_report_bundle, load_strategy
from research._shared.strategy_plotting import (
    clean_strategy_labels,
    plot_growth,
    plot_risk_diagnostics,
)


EXPECTED_ARTIFACTS = {
    "manifest.json",
    "daily_series.csv",
    "monthly_returns.csv",
    "summary_metrics.csv",
    "calendar_year_metrics.csv",
    "rolling_metrics.csv",
    "drawdown_events.csv",
    "stress_period_metrics.csv",
    "risk_contribution.csv",
    "turnover_costs.csv",
    "leverage_signal_history.csv",
    "leverage_signal_events.csv",
    "price_provenance.json",
}


def _synthetic_bank_pack_prices() -> pd.DataFrame:
    dates = pd.bdate_range("2019-01-01", "2026-03-31")
    n = len(dates)
    cycle = np.sin(np.arange(n) / 45) / 800
    prices = pd.DataFrame({
        "SPY": 100 * np.cumprod(1 + 0.00035 + cycle),
        "QQQ": 100 * np.cumprod(1 + 0.00045 + cycle * 1.2),
        "TLT": 100 * np.cumprod(1 + 0.00010 - cycle * 0.6),
        "TIP": 100 * np.cumprod(1 + 0.00008 - cycle * 0.2),
        "GLD": 100 * np.cumprod(1 + 0.00012 + np.cos(np.arange(n) / 39) / 1200),
        "GSG": 100 * np.cumprod(1 + 0.00005 + np.sin(np.arange(n) / 31) / 1100),
        "JEPQ": np.nan,
        "ALLW": np.nan,
    }, index=dates)
    jepq_mask = prices.index >= pd.Timestamp("2022-05-04")
    prices.loc[jepq_mask, "JEPQ"] = 100 * np.cumprod(1 + 0.00020 + cycle[jepq_mask] * 0.9)
    mask = prices.index >= pd.Timestamp("2025-03-06")
    prices.loc[mask, "ALLW"] = 100 * np.cumprod(1 + 0.00018 + cycle[mask] * 0.4)
    return prices


def test_report_builder_writes_expected_csv_bundle(tmp_path):
    allocation = {
        "SPY": 0.134,
        "QQQ": 0.103,
        "TLT": 0.175,
        "TIP": 0.348,
        "GLD": 0.142,
        "GSG": 0.098,
    }

    bundle = build_report_bundle(
        prices=_synthetic_bank_pack_prices(),
        strategy_id="synthetic_strategy",
        allocation=allocation,
        output_root=tmp_path,
        start_date="2019-01-01",
        end_date="2026-03-31",
        generated_at=datetime(2026, 1, 2, 3, 4, 5),
    )

    assert {path.name for path in bundle.iterdir()} == EXPECTED_ARTIFACTS

    with open(bundle / "manifest.json", "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["strategy_id"] == "synthetic_strategy"
    assert manifest["artifacts"] == sorted(manifest["artifacts"], key=manifest["artifacts"].index)
    assert manifest["leverage_candidate"]["name"] == "SPY 34/42 @ 30% cap"
    assert {candidate["name"] for candidate in manifest["leverage_candidates"]} == {
        "SPY 34/42 @ 30% cap",
        "GLD 32/64 @ 30% cap",
        "SPY 32/42 + GLD 36/52 @ 30% cap",
    }

    daily = pd.read_csv(bundle / "daily_series.csv")
    summary = pd.read_csv(bundle / "summary_metrics.csv")
    calendar = pd.read_csv(bundle / "calendar_year_metrics.csv")
    rolling = pd.read_csv(bundle / "rolling_metrics.csv")
    risk = pd.read_csv(bundle / "risk_contribution.csv")
    events = pd.read_csv(bundle / "leverage_signal_events.csv")

    assert {"Date", "Strategy", "Value", "Indexed Value", "Daily Return (%)", "Drawdown (%)"} <= set(daily.columns)
    assert {"Window", "Strategy", "CAGR (%)", "Max Drawdown (%)", "VaR 5% Daily (%)"} <= set(summary.columns)
    assert {"Year", "Strategy", "Return (%)", "Max Drawdown (%)", "Max DD Duration (days)", "Calmar"} <= set(calendar.columns)
    assert {"Date", "Strategy", "Window", "Rolling CAGR (%)", "Rolling Beta to SPY"} <= set(rolling.columns)
    assert {"Asset", "Weight", "Risk Contribution (%)"} <= set(risk.columns)
    assert not daily[daily["Strategy"] == "My Strategy (DIY)"].empty
    assert not daily[daily["Strategy"] == "SPY 34/42 @ 30% cap"].empty
    assert not daily[daily["Strategy"] == "GLD 32/64 @ 30% cap"].empty
    assert not daily[daily["Strategy"] == "SPY 32/42 + GLD 36/52 @ 30% cap"].empty
    assert not daily[daily["Strategy"] == "S&P 500 (SPY)"].empty
    assert not daily[daily["Strategy"] == "JEPQ (JPM Nasdaq Income)"].empty
    assert not daily[daily["Strategy"] == "ALLW (Bridgewater)"].empty
    assert not summary[summary["Strategy"] == "SPY 34/42 @ 30% cap"].empty
    assert not summary[summary["Strategy"] == "GLD 32/64 @ 30% cap"].empty
    assert not summary[summary["Strategy"] == "SPY 32/42 + GLD 36/52 @ 30% cap"].empty
    assert {"Date", "Ticker", "Event", "Entry Threshold", "Exit Threshold"} <= set(events.columns)
    assert {
        "SPY 34/42 @ 30% cap",
        "GLD 32/64 @ 30% cap",
        "SPY 32/42 + GLD 36/52 @ 30% cap",
    } <= set(events["Candidate"])
    assert {"SPY", "GLD"} <= set(events["Ticker"])
    assert {"Entry", "Exit"} <= set(events["Event"])


def test_strategy_comparison_notebook_is_presentation_only():
    notebook = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "notebooks"
        / "strategy_comparison.py"
    )
    source = notebook.read_text()

    forbidden = ["yfinance", "fetch_prices", "build_daily_series"]
    for token in forbidden:
        assert token not in source

    assert "production_validation" in source
    assert "price_provenance.json" in source
    assert "SPY 34/42 @ 30% cap" in source
    assert "GLD 32/64 @ 30% cap" in source
    assert "SPY 32/42 + GLD 36/52 @ 30% cap" in source
    assert "Full-Grid Top Leaderboard Row" in source
    assert "Full History Summary Metrics" in source
    assert "ALLW Overlap Summary Metrics" in source
    assert "Candidate Summary Metrics" not in source
    assert "mixed_leverage_full_grid_oos" in source
    assert "full_history_scale" in source
    assert "overlap_scale" in source
    assert "growth_strategies" in source
    assert "leverage_signal_events.csv" in source
    assert "Show SPY/GLD leverage entry and exit markers" in source
    assert "show_rebalance_events" in source
    assert "rebalance_events" in source
    assert "tax_summary" in source
    assert "tax_monthly" in source
    assert "regime_comparison" in source
    assert "plot_tax_cost" in source
    assert "plot_regime_comparison" in source
    assert "plot_sweep_heatmap" in source
    assert "tax_regime_comparison.csv" in source
    assert "threshold_sweep_summary.csv" in source


def test_growth_plot_accepts_strategy_and_scale_controls(tmp_path):
    bundle = build_report_bundle(
        prices=_synthetic_bank_pack_prices(),
        strategy_id="synthetic_strategy",
        allocation={
            "SPY": 0.134,
            "QQQ": 0.103,
            "TLT": 0.175,
            "TIP": 0.348,
            "GLD": 0.142,
            "GSG": 0.098,
        },
        output_root=tmp_path,
        start_date="2019-01-01",
        end_date="2026-03-31",
        generated_at=datetime(2026, 1, 2, 3, 4, 5),
    )
    daily = pd.read_csv(bundle / "daily_series.csv", parse_dates=["Date"])
    events = pd.read_csv(bundle / "leverage_signal_events.csv", parse_dates=["Date"])

    fig = plot_growth(
        daily,
        strategies=["My Strategy (DIY)", "SPY 34/42 @ 30% cap"],
        full_history_scale="linear",
        leverage_events=events,
        overlap_scale="log",
        show_leverage_events=True,
    )

    assert fig.axes[0].get_yscale() == "linear"
    assert fig.axes[1].get_yscale() == "log"
    assert len(fig.axes[0].lines) == 2
    assert fig.axes[0].collections


def test_strategy_label_cleanup_and_unknown_plot_colors():
    data = pd.DataFrame({
        "Strategy": [
            "My Strategy (DIY)",
            "SPY only 34/42 @ 30% cap",
            "GLD only 32/64 @ 30% cap",
            "SPY selective + GLD default 25% cap",
            "Unexpected Research Row",
        ],
        "Window": ["ALLW Overlap"] * 5,
        "CAGR (%)": [1, 2, 3, 4, 5],
        "Volatility (%)": [1, 2, 3, 4, 5],
        "Sharpe": [1, 2, 3, 4, 5],
        "Sortino": [1, 2, 3, 4, 5],
        "Calmar": [1, 2, 3, 4, 5],
        "Ulcer Index": [1, 2, 3, 4, 5],
        "Max Drawdown (%)": [-1, -2, -3, -4, -5],
        "Max DD Duration (days)": [1, 2, 3, 4, 5],
        "VaR 5% Daily (%)": [-1, -2, -3, -4, -5],
        "CVaR 5% Daily (%)": [-1, -2, -3, -4, -5],
        "Downside Beta": [1, 2, 3, 4, 5],
        "Up Capture (%)": [1, 2, 3, 4, 5],
    })

    cleaned = clean_strategy_labels(data)

    assert "SPY only 34/42 @ 30% cap" not in set(cleaned["Strategy"])
    assert "GLD only 32/64 @ 30% cap" not in set(cleaned["Strategy"])
    assert "SPY selective + GLD default 25% cap" not in set(cleaned["Strategy"])
    assert "SPY 34/42 @ 30% cap" in set(cleaned["Strategy"])
    assert "GLD 32/64 @ 30% cap" in set(cleaned["Strategy"])

    fig = plot_risk_diagnostics(cleaned)
    assert len(fig.axes) == 12


def test_strategy_report_loader_accepts_baseline_alias():
    canonical = load_strategy("6asset_tip_gsg_rpavg")
    alias = load_strategy("6_asset_rp_baseline")

    assert alias["allocation"] == canonical["allocation"]


def test_data_explorer_defaults_to_adjusted_close():
    notebook = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "notebooks"
        / "data_explorer.py"
    )
    source = notebook.read_text()

    assert 'value="adj_close"' in source
