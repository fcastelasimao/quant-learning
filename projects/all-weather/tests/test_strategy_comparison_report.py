import json
from datetime import datetime

import numpy as np
import pandas as pd

from research.build_strategy_comparison_report import build_report_bundle


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
        "ALLW": np.nan,
    }, index=dates)
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

    daily = pd.read_csv(bundle / "daily_series.csv")
    summary = pd.read_csv(bundle / "summary_metrics.csv")
    calendar = pd.read_csv(bundle / "calendar_year_metrics.csv")
    rolling = pd.read_csv(bundle / "rolling_metrics.csv")
    risk = pd.read_csv(bundle / "risk_contribution.csv")

    assert {"Date", "Strategy", "Value", "Indexed Value", "Daily Return (%)", "Drawdown (%)"} <= set(daily.columns)
    assert {"Window", "Strategy", "CAGR (%)", "Max Drawdown (%)", "VaR 5% Daily (%)"} <= set(summary.columns)
    assert {"Year", "Strategy", "Return (%)", "Max Drawdown (%)", "Max DD Duration (days)", "Calmar"} <= set(calendar.columns)
    assert {"Date", "Strategy", "Window", "Rolling CAGR (%)", "Rolling Beta to SPY"} <= set(rolling.columns)
    assert {"Asset", "Weight", "Risk Contribution (%)"} <= set(risk.columns)
    assert not daily[daily["Strategy"] == "My Strategy (DIY)"].empty
    assert not daily[daily["Strategy"] == "S&P 500 (SPY)"].empty
    assert not daily[daily["Strategy"] == "ALLW (Bridgewater)"].empty


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


def test_data_explorer_defaults_to_adjusted_close():
    notebook = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "notebooks"
        / "data_explorer.py"
    )
    source = notebook.read_text()

    assert 'value="adj_close"' in source
