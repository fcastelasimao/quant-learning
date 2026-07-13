"""
test_daily_tax_backtest.py
==========================
Tests for engine/daily_tax_backtest.run_daily_tax_backtest (L.52).

Four seeded synthetic tests:
  1. Buy-and-hold baseline: no rebalance → value = weighted sum of prices.
  2. 31-day gate: with monthly_unconditional, no rebalance fires before 31 days.
  3. Monthly resample Calmar within 10% of the monthly tax_backtest engine.
  4. FIFO US-tax is charged on a sale: cum_tax > 0 and value < no-tax version.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from engine.backtest import RebalancePolicy
from engine.daily_tax_backtest import run_daily_tax_backtest
from engine.lot_ledger import LotSelector
from engine.stats import compute_calmar, compute_cagr, compute_max_drawdown
from engine.tax import TaxRegime
from engine.tax_backtest import run_tax_aware_backtest


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _prices(n_years: float = 10.0) -> pd.DataFrame:
    """Seeded 6-asset daily price series. Same seed as test_tax_backtest."""
    rng = np.random.default_rng(7)
    n = int(n_years * 252)
    dates = pd.date_range("2008-01-01", periods=n, freq="B")
    cols = {}
    for t, (mu, sd) in {
        "SPY": (0.0003, 0.011),
        "QQQ": (0.0004, 0.014),
        "TLT": (0.0001, 0.009),
        "TIP": (0.00008, 0.004),
        "GLD": (0.00025, 0.010),
        "GSG": (0.00005, 0.013),
    }.items():
        cols[t] = 100.0 * np.exp(np.cumsum(rng.normal(mu, sd, n)))
    return pd.DataFrame(cols, index=dates)


W = {
    "SPY": 0.134, "QQQ": 0.103, "TLT": 0.175,
    "TIP": 0.348, "GLD": 0.142, "GSG": 0.098,
}


def _calmar(monthly_df: pd.DataFrame) -> float:
    """Calmar from a monthly DataFrame with a 'Value' column."""
    vals = monthly_df["Value"].dropna()
    if len(vals) < 2:
        return float("nan")
    years = len(vals) / 12
    cagr = compute_cagr(vals, years)
    mdd = compute_max_drawdown(vals)
    return compute_calmar(cagr, mdd)


# ---------------------------------------------------------------------------
# Test 1: buy-and-hold baseline
# ---------------------------------------------------------------------------

def test_buy_and_hold_no_rebalance():
    """drift_absolute(0.999) never fires → value = weighted buy-and-hold exactly."""
    prices = _prices()
    result = run_daily_tax_backtest(
        prices,
        W,
        regime=TaxRegime.none(),
        rebalance_policy=RebalancePolicy.drift_absolute(0.999),
        transaction_cost_pct=0.0,
        min_rebalance_days=31,
    )

    assert len(result.rebalance_dates) == 0, "Expected zero rebalances"

    first = prices.iloc[0]
    last = prices.iloc[-1]
    expected_final = sum(
        (10_000 * W[t] / float(first[t])) * float(last[t]) for t in W
    )
    actual_final = result.daily_records["Value"].iloc[-1]
    np.testing.assert_allclose(actual_final, expected_final, rtol=1e-4)


# ---------------------------------------------------------------------------
# Test 2: 31-day gate
# ---------------------------------------------------------------------------

def test_gate_blocks_rebalance_before_31_days():
    """monthly_unconditional with 31-day gate: no rebalance fires in first 30 days,
    at least one fires by day 40 on a 60-day series."""
    prices = _prices().iloc[:60]  # ~60 trading days
    result = run_daily_tax_backtest(
        prices,
        W,
        regime=TaxRegime.none(),
        rebalance_policy=RebalancePolicy.monthly_unconditional(),
        transaction_cost_pct=0.0,
        min_rebalance_days=31,
    )

    first_date = prices.index[0].date()
    for d in result.rebalance_dates:
        days_elapsed = (d - first_date).days
        assert days_elapsed >= 31, (
            f"Rebalance fired after only {days_elapsed} days (gate is 31)"
        )

    # At least one rebalance should fire on a 60-day series with 31-day gate
    assert len(result.rebalance_dates) >= 1, "Expected at least one rebalance by day 60"


# ---------------------------------------------------------------------------
# Test 3: monthly Calmar within 10 % of monthly engine
# ---------------------------------------------------------------------------

def test_monthly_calmar_close_to_monthly_engine():
    """daily engine monthly resample Calmar within 10 % of tax_backtest Calmar.

    Both use TaxRegime.none() and zero cost so only timing of rebalances
    (month-end vs every 31 calendar days) differs. Over 10 years the gap
    is small.
    """
    prices = _prices(n_years=10.0)

    daily_result = run_daily_tax_backtest(
        prices,
        W,
        regime=TaxRegime.none(),
        rebalance_policy=RebalancePolicy.monthly_unconditional(),
        transaction_cost_pct=0.0,
        min_rebalance_days=31,
    )
    monthly_result = run_tax_aware_backtest(
        prices,
        W,
        regime=TaxRegime.none(),
        transaction_cost_pct=0.0,
    )

    c_daily = _calmar(daily_result.monthly)
    c_monthly = _calmar(monthly_result.monthly)

    assert not np.isnan(c_daily), "daily Calmar is NaN"
    assert not np.isnan(c_monthly), "monthly Calmar is NaN"
    assert c_monthly > 0, "monthly engine Calmar must be positive"

    rel_diff = abs(c_daily - c_monthly) / abs(c_monthly)
    assert rel_diff < 0.10, (
        f"Calmar gap too large: daily={c_daily:.4f}, monthly={c_monthly:.4f}, "
        f"rel_diff={rel_diff:.2%}"
    )


# ---------------------------------------------------------------------------
# Test 4: FIFO US-tax is charged on sale
# ---------------------------------------------------------------------------

def test_us_tax_charged_on_sale():
    """US regime charges positive cumulative tax; after-tax value < no-tax value."""
    rng = np.random.default_rng(42)
    n = 500  # ~2 years of trading days
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    spy = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n)))
    tlt = 100.0 * np.exp(np.cumsum(rng.normal(0.0000, 0.005, n)))
    prices = pd.DataFrame({"SPY": spy, "TLT": tlt}, index=dates)
    alloc = {"SPY": 0.5, "TLT": 0.5}

    with_tax = run_daily_tax_backtest(
        prices, alloc,
        regime=TaxRegime.us(),
        rebalance_policy=RebalancePolicy.drift_absolute(0.05),
        transaction_cost_pct=0.0,
        min_rebalance_days=31,
    )
    no_tax = run_daily_tax_backtest(
        prices, alloc,
        regime=TaxRegime.none(),
        rebalance_policy=RebalancePolicy.drift_absolute(0.05),
        transaction_cost_pct=0.0,
        min_rebalance_days=31,
    )

    assert with_tax.daily_records["Cumulative Tax Paid"].iloc[-1] > 0.0, (
        "Expected positive cumulative tax under US regime"
    )
    assert len(with_tax.rebalance_dates) > 0, "Expected at least one rebalance"
    assert with_tax.daily_records["Value"].iloc[-1] < no_tax.daily_records["Value"].iloc[-1], (
        "After-tax final value should be lower than no-tax final value"
    )
    # monthly property has expected shape
    assert "Value" in with_tax.monthly.columns
    assert "Cumulative Tax Paid" in with_tax.monthly.columns
    assert len(with_tax.monthly) > 0
