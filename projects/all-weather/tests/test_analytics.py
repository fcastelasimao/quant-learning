import numpy as np
import pandas as pd

from engine.analytics import (
    calendar_year_metrics,
    drawdown_events,
    drawdown_series,
    max_drawdown_duration,
    risk_contribution,
    rolling_metrics,
    summary_metrics,
    tail_risk_metrics,
    turnover_costs,
)


def test_drawdown_depth_duration_and_recovery_event():
    dates = pd.date_range("2020-01-01", periods=6, freq="B")
    values = pd.Series([100, 120, 90, 80, 110, 125], index=dates)

    dd = drawdown_series(values)
    events = drawdown_events(pd.DataFrame({"Strategy": values}))

    assert round(float(dd.min()), 4) == round((80 / 120 - 1) * 100, 4)
    assert max_drawdown_duration(values) == 3
    assert len(events) == 1
    assert bool(events.iloc[0]["Recovered"])
    assert events.iloc[0]["Peak Date"] == dates[1].date().isoformat()
    assert events.iloc[0]["Trough Date"] == dates[3].date().isoformat()


def test_rolling_metrics_wait_for_full_window_and_beta_corr_are_controlled():
    dates = pd.bdate_range("2020-01-01", periods=270)
    spy_returns = pd.Series([0.001, -0.0005, 0.0008] * 90, index=dates)
    diy_returns = spy_returns * 0.5
    values = pd.DataFrame({
        "My Strategy (DIY)": 100 * (1 + diy_returns).cumprod(),
        "S&P 500 (SPY)": 100 * (1 + spy_returns).cumprod(),
    })

    out = rolling_metrics(values, benchmark="S&P 500 (SPY)", windows=(252,))
    diy = out[out["Strategy"] == "My Strategy (DIY)"]

    assert not diy.empty
    assert diy["Date"].min() == values.index[251]
    assert abs(diy["Rolling Beta to SPY"].dropna().iloc[-1] - 0.5) < 1e-10
    assert abs(diy["Rolling Corr to SPY"].dropna().iloc[-1] - 1.0) < 1e-10


def test_summary_metrics_capture_downside_beta_and_tail_risk():
    dates = pd.bdate_range("2020-01-01", periods=10)
    spy_returns = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01, -0.01, 0.02, -0.02, 0.01, -0.01], index=dates)
    diy_returns = spy_returns * 0.5
    values = pd.DataFrame({
        "My Strategy (DIY)": 100 * (1 + diy_returns).cumprod(),
        "S&P 500 (SPY)": 100 * (1 + spy_returns).cumprod(),
    })

    out = summary_metrics(values, benchmark="S&P 500 (SPY)")
    diy = out[out["Strategy"] == "My Strategy (DIY)"].iloc[0]

    assert abs(diy["Beta to SPY"] - 0.5) < 1e-10
    assert abs(diy["Downside Beta"] - 0.5) < 1e-10
    assert abs(diy["Up Capture (%)"] - 50.0) < 1e-10
    assert abs(diy["Down Capture (%)"] - 50.0) < 1e-10
    assert diy["Worst Day (%)"] < 0
    assert diy["CVaR 5% Daily (%)"] <= diy["VaR 5% Daily (%)"]


def test_tail_risk_and_calendar_year_partial_year_handling():
    dates = pd.bdate_range("2020-06-01", periods=180)
    values = pd.DataFrame({
        "Strategy": pd.Series(np.linspace(100, 120, len(dates)), index=dates)
    })

    tail = tail_risk_metrics(values["Strategy"])
    annual = calendar_year_metrics(values)

    assert "Worst Month (%)" in tail
    assert set(annual["Year"]) == {2020, 2021}
    assert annual.iloc[0]["Start Value"] == 100
    assert "Calmar" in annual.columns


def test_risk_contribution_and_turnover_costs_are_well_formed():
    dates = pd.bdate_range("2020-01-01", periods=90)
    prices = pd.DataFrame({
        "A": 100 * (1.001 ** np.arange(len(dates))),
        "B": 100 * (1.0002 ** np.arange(len(dates))),
    }, index=dates)
    allocation = {"A": 0.6, "B": 0.4}

    risk = risk_contribution(prices, allocation)
    turnover = turnover_costs(prices, allocation, transaction_cost_pct=0.001)

    assert set(risk["Asset"]) == {"A", "B"}
    assert abs(risk["Risk Contribution (%)"].sum() - 100) < 1e-8
    assert not turnover.empty
    assert (turnover["Turnover (%)"] >= 0).all()
    assert turnover["Cumulative Cost"].is_monotonic_increasing
