"""
test_tax_backtest.py
====================
Tests for engine/tax_backtest.run_tax_aware_backtest (D.16).

Uses deterministic synthetic prices (no network). The central invariant: with
TaxRegime.none() and zero cost, the tax-aware engine must reproduce the
share-based monthly engine's All Weather value path — proving the rebalance
mechanics agree before any tax is layered on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.backtest import RebalancePolicy, run_backtest
from engine.lot_ledger import LotSelector
from engine.tax import TaxRegime
from engine.tax_backtest import run_tax_aware_backtest


def _prices() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2008-01-01", "2018-12-31", freq="B")
    cols = {}
    for t, (mu, sd) in {
        "SPY": (0.0003, 0.011), "QQQ": (0.0004, 0.014), "TLT": (0.0001, 0.009),
        "TIP": (0.00008, 0.004), "GLD": (0.00025, 0.010), "GSG": (0.00005, 0.013),
    }.items():
        cols[t] = 100.0 * np.exp(np.cumsum(rng.normal(mu, sd, len(dates))))
    return pd.DataFrame(cols, index=dates)


W = {"SPY": 0.134, "QQQ": 0.103, "TLT": 0.175, "TIP": 0.348, "GLD": 0.142, "GSG": 0.098}


# ---------------------------------------------------------------------------
# core cross-check
# ---------------------------------------------------------------------------

def test_none_regime_zero_cost_matches_share_based_engine():
    prices = _prices()
    tax = run_tax_aware_backtest(prices, W, regime=TaxRegime.none(),
                                 transaction_cost_pct=0.0)
    share = run_backtest(prices, prices["SPY"], W, 10_000.0, transaction_cost_pct=0.0)
    # Same months, same value path (rebalance mechanics agree).
    assert len(tax.monthly) == len(share)
    np.testing.assert_allclose(
        tax.monthly["Value"].values,
        share["All Weather Value"].values,
        rtol=1e-6,
    )


def test_none_regime_pays_zero_tax():
    tax = run_tax_aware_backtest(_prices(), W, regime=TaxRegime.none())
    assert tax.monthly["Cumulative Tax Paid"].iloc[-1] == 0.0
    assert (tax.monthly["Period Tax"] == 0.0).all()


# ---------------------------------------------------------------------------
# tax reduces value; selector matters
# ---------------------------------------------------------------------------

def test_us_regime_pays_tax_and_reduces_value():
    prices = _prices()
    none = run_tax_aware_backtest(prices, W, regime=TaxRegime.none())
    us = run_tax_aware_backtest(prices, W, regime=TaxRegime.us())
    assert us.monthly["Cumulative Tax Paid"].iloc[-1] > 0.0
    assert us.monthly["Value"].iloc[-1] < none.monthly["Value"].iloc[-1]


def test_tax_optimal_pays_no_more_tax_than_fifo():
    prices = _prices()
    fifo = run_tax_aware_backtest(prices, W, regime=TaxRegime.us(),
                                  lot_selector=LotSelector.FIFO)
    opt = run_tax_aware_backtest(prices, W, regime=TaxRegime.us(),
                                 lot_selector=LotSelector.TAX_OPTIMAL)
    # tax_optimal is designed to realize no more gain than FIFO over the path
    assert opt.monthly["Value"].iloc[-1] >= fifo.monthly["Value"].iloc[-1] - 1e-6
    assert fifo.selector == "fifo" and opt.selector == "tax_optimal"


# ---------------------------------------------------------------------------
# drift policy reduces trading
# ---------------------------------------------------------------------------

def test_drift_policy_trades_less_than_monthly():
    prices = _prices()
    monthly = run_tax_aware_backtest(prices, W, regime=TaxRegime.us())
    drift = run_tax_aware_backtest(
        prices, W, regime=TaxRegime.us(),
        rebalance_policy=RebalancePolicy.drift_relative(0.25),
    )
    assert drift.monthly["Rebalanced"].sum() < monthly.monthly["Rebalanced"].sum()
    assert drift.policy_label.startswith("drift_relative")


# ---------------------------------------------------------------------------
# dividends
# ---------------------------------------------------------------------------

def _dividends() -> pd.DataFrame:
    from datetime import date
    rows = []
    for yr in range(2008, 2019):
        for mo in (3, 6, 9, 12):
            rows.append({"Ticker": "SPY", "ExDate": date(yr, mo, 15), "Amount": 1.0,
                         "AdjAmount": 1.0, "RecordDate": None, "PaymentDate": None,
                         "DeclarationDate": None})
            rows.append({"Ticker": "TLT", "ExDate": date(yr, mo, 1), "Amount": 0.25,
                         "AdjAmount": 0.25, "RecordDate": None, "PaymentDate": None,
                         "DeclarationDate": None})
    return pd.DataFrame(rows)


def test_dividend_tax_is_charged_under_us_regime():
    prices = _prices()
    no_div = run_tax_aware_backtest(prices, W, regime=TaxRegime.us())
    with_div = run_tax_aware_backtest(prices, W, regime=TaxRegime.us(),
                                      dividends=_dividends())
    assert with_div.monthly["Dividend Tax"].sum() > 0.0
    assert no_div.monthly["Dividend Tax"].sum() == 0.0
    # dividends recorded as income
    assert with_div.monthly["Dividend Income"].sum() > 0.0


def test_dividends_under_none_regime_are_untaxed():
    res = run_tax_aware_backtest(_prices(), W, regime=TaxRegime.none(),
                                 dividends=_dividends())
    assert res.monthly["Dividend Tax"].sum() == 0.0


# ---------------------------------------------------------------------------
# §1256 year-end mark-to-market
# ---------------------------------------------------------------------------

def test_gsg_mark_to_market_fires_at_year_end():
    res = run_tax_aware_backtest(_prices(), W, regime=TaxRegime.us())
    # MTM tax should appear in December rows only
    mtm = res.monthly[res.monthly["MTM Tax"] != 0.0]
    assert not mtm.empty
    assert set(mtm.index.month) <= {12}


def test_no_mtm_under_none_regime():
    res = run_tax_aware_backtest(_prices(), W, regime=TaxRegime.none())
    assert (res.monthly["MTM Tax"] == 0.0).all()


# ---------------------------------------------------------------------------
# artifact shape
# ---------------------------------------------------------------------------

def test_result_artifacts_have_expected_columns():
    res = run_tax_aware_backtest(_prices(), W, regime=TaxRegime.us(),
                                 dividends=_dividends())
    assert {"Value", "Cumulative Tax Paid", "Period Tax",
            "Sale Tax", "Dividend Tax", "MTM Tax"} <= set(res.monthly.columns)
    assert {"Year", "ST_Gain", "LT_Gain", "Dividend_Income",
            "Total_Tax"} <= set(res.tax_summary.columns)
    assert res.regime_name == "us"
    # one summary row per calendar year covered
    assert res.tax_summary["Year"].is_monotonic_increasing


def test_empty_prices_raises():
    empty = pd.DataFrame({t: [] for t in W}, index=pd.DatetimeIndex([]))
    with pytest.raises(ValueError, match="No overlapping monthly data"):
        run_tax_aware_backtest(empty, W, regime=TaxRegime.none())
