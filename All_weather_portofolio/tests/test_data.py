"""
tests/test_data.py
==================
Tests for data-fetching behaviour and cost modelling in run_backtest().

Integration tests (marked @pytest.mark.integration) call yfinance directly
and require a live internet connection. Skip them in offline environments:

    pytest tests/ -m "not integration"

Unit tests use fully synthetic price data and run without any network access.

Run the full suite (requires network):
    pytest tests/test_data.py -v
"""

import numpy as np
import pandas as pd
import pytest
import yfinance as yf

from backtest import run_backtest


# ===========================================================================
# SYNTHETIC DATA HELPERS
# ===========================================================================

def _make_synthetic_prices(start: str = "2010-01-01",
                            end:   str = "2012-12-31") -> pd.DataFrame:
    """
    Generate daily prices for two assets ('A', 'B') and a benchmark ('BENCH').

    Growth rates:
      A     -- 0.10 %/day (fast grower)
      B     -- 0.05 %/day (slow grower)
      BENCH -- 0.08 %/day

    Different growth rates cause weight drift at every rebalancing step, so
    the rebalanced strategy always executes non-trivial trades. This makes
    transaction-cost effects measurable even over short time windows.
    """
    dates = pd.date_range(start, end, freq="B")   # business days only
    n = len(dates)
    return pd.DataFrame({
        "A":     100.0 * (1.001  ** np.arange(n)),
        "B":     100.0 * (1.0005 ** np.arange(n)),
        "BENCH": 100.0 * (1.0008 ** np.arange(n)),
    }, index=dates)


# Simple two-asset allocation used by all unit tests.
# Weights differ (0.6 / 0.4) so drift is visible after just a few months.
_ALLOCATION = {"A": 0.6, "B": 0.4}


# ===========================================================================
# INTEGRATION TESTS  -- require network / yfinance
# ===========================================================================

@pytest.mark.integration
def test_total_return_higher_than_price_return_for_tlt():
    """
    TLT is a long-duration Treasury ETF that pays monthly interest
    distributions. Over a 20-year period (2006-2026) the compounded
    reinvestment of those distributions should make the total-return
    index grow at least 50% more than the price-only index.

    total-return growth  = auto_adjust=True  Close[-1] / Close[0]
    price-return growth  = auto_adjust=False Close[-1] / Close[0]
    (price-return uses the unadjusted Close, i.e. price appreciation only)
    """
    raw_tr = yf.download("TLT", start="2006-01-01", end="2026-01-01",
                         auto_adjust=True,  progress=False)
    raw_pr = yf.download("TLT", start="2006-01-01", end="2026-01-01",
                         auto_adjust=False, progress=False)

    tr_growth = float(raw_tr["Close"].squeeze().iloc[-1]) / float(raw_tr["Close"].squeeze().iloc[0])
    pr_growth = float(raw_pr["Close"].squeeze().iloc[-1]) / float(raw_pr["Close"].squeeze().iloc[0])

    assert tr_growth >= pr_growth * 1.5, (
        f"TLT total-return growth ({tr_growth:.3f}x) should be at least 50% "
        f"higher than price-return growth ({pr_growth:.3f}x). "
        f"If this fails, auto_adjust=True may not be working correctly."
    )


@pytest.mark.integration
def test_gold_total_equals_price_return():
    """
    GLD (SPDR Gold Shares) pays no dividend and has never undergone a stock
    split. Its total-return series should therefore be virtually identical
    to its price-return series -- the relative difference in cumulative
    growth must be less than 0.1%.

    This acts as a negative control: if auto_adjust added spurious returns
    to a non-distributing asset, this test would catch it.
    """
    raw_tr = yf.download("GLD", start="2006-01-01", end="2026-01-01",
                         auto_adjust=True,  progress=False)
    raw_pr = yf.download("GLD", start="2006-01-01", end="2026-01-01",
                         auto_adjust=False, progress=False)

    tr_growth = float(raw_tr["Close"].squeeze().iloc[-1]) / float(raw_tr["Close"].squeeze().iloc[0])
    pr_growth = float(raw_pr["Close"].squeeze().iloc[-1]) / float(raw_pr["Close"].squeeze().iloc[0])

    relative_diff = abs(tr_growth - pr_growth) / pr_growth

    assert relative_diff < 0.001, (
        f"GLD total-return growth ({tr_growth:.6f}x) should be within 0.1% "
        f"of price-return growth ({pr_growth:.6f}x); "
        f"actual relative difference: {relative_diff:.4%}"
    )


# ===========================================================================
# UNIT TESTS  -- fully offline, synthetic data
# ===========================================================================

def test_transaction_costs_reduce_portfolio_value():
    """
    Every monthly rebalancing step subtracts cost from the portfolio. After
    the first row (where both runs start at the same initial value), the
    cost-burdened All Weather portfolio must be worth strictly less than the
    zero-cost baseline on every subsequent row.

    Uses 0.001 (0.1%) per trade -- the project's default realistic estimate.
    """
    df    = _make_synthetic_prices()
    bench  = df["BENCH"]
    prices = df[["A", "B"]]

    result_no_cost   = run_backtest(prices, bench, _ALLOCATION)
    result_with_cost = run_backtest(prices, bench, _ALLOCATION,
                                   transaction_cost_pct=0.001)

    aw_no_cost   = result_no_cost["All Weather Value"].iloc[1:]
    aw_with_cost = result_with_cost["All Weather Value"].iloc[1:]

    assert (aw_with_cost < aw_no_cost).all(), (
        "All Weather Value should be strictly lower on every row after the "
        "first when transaction_cost_pct=0.001 vs 0.0. "
        "Failing rows:\n"
        f"{aw_with_cost[aw_with_cost >= aw_no_cost]}"
    )


def test_zero_costs_matches_baseline():
    """
    Passing explicit zeros (transaction_cost_pct=0.0, tax_drag_pct=0.0)
    must produce results that are bit-for-bit identical to calling
    run_backtest() with no cost parameters (default values).

    This confirms backward compatibility: existing callers that omit cost
    parameters are unaffected by the new feature.
    """
    df    = _make_synthetic_prices()
    bench  = df["BENCH"]
    prices = df[["A", "B"]]

    result_defaults      = run_backtest(prices, bench, _ALLOCATION)
    result_explicit_zero = run_backtest(prices, bench, _ALLOCATION,
                                       transaction_cost_pct=0.0,
                                       tax_drag_pct=0.0)

    pd.testing.assert_frame_equal(
        result_defaults,
        result_explicit_zero,
        check_exact=True,
    )


def test_tax_drag_reduces_portfolio_annually():
    """
    A 10% annual tax drag is applied at the start of each new calendar year.
    From the second year onward, every row of the All Weather Value series
    must be strictly lower in the drag run than in the no-drag baseline.

    Synthetic data spans 2010-2012 so two annual drag events occur
    (Jan 2011 and Jan 2012), making the effect clearly observable.
    """
    df    = _make_synthetic_prices()   # 2010-01-01 to 2012-12-31
    bench  = df["BENCH"]
    prices = df[["A", "B"]]

    result_no_drag   = run_backtest(prices, bench, _ALLOCATION)
    result_with_drag = run_backtest(prices, bench, _ALLOCATION,
                                   tax_drag_pct=0.10)

    # Tax drag first fires in the first month-end date of 2011.
    # Every row from 2011 onward must reflect that reduction.
    mask = result_no_drag.index.year >= 2011
    aw_no_drag   = result_no_drag.loc[mask, "All Weather Value"]
    aw_with_drag = result_with_drag.loc[mask, "All Weather Value"]

    assert (aw_with_drag < aw_no_drag).all(), (
        "All Weather Value should be strictly lower on every row from 2011 "
        "onward when tax_drag_pct=0.10 vs 0.0. "
        "Failing rows:\n"
        f"{aw_with_drag[aw_with_drag >= aw_no_drag]}"
    )


def test_bh_receives_no_transaction_costs():
    """
    The Buy & Hold strategy fixes its share counts at the initial purchase
    and never trades again. Transaction costs are only deducted when trades
    occur, so B&H must be completely unaffected by any transaction_cost_pct.

    Uses a deliberately high cost (1%) to make any unintended effect obvious.

    Also confirms the test is non-trivial: the rebalanced All Weather Value
    must differ between the two runs, proving that costs DO affect trading
    strategies and that only B&H is exempt.
    """
    df    = _make_synthetic_prices()
    bench  = df["BENCH"]
    prices = df[["A", "B"]]

    result_zero_cost = run_backtest(prices, bench, _ALLOCATION)
    result_high_cost = run_backtest(prices, bench, _ALLOCATION,
                                   transaction_cost_pct=0.01)

    pd.testing.assert_series_equal(
        result_zero_cost["Buy & Hold All Weather"],
        result_high_cost["Buy & Hold All Weather"],
        check_names=True,
        obj="Buy & Hold All Weather",
    )

    # Sanity check: confirm that AW_R IS affected, so the test is meaningful
    assert not result_zero_cost["All Weather Value"].equals(
        result_high_cost["All Weather Value"]
    ), (
        "All Weather Value should differ between zero-cost and high-cost runs "
        "-- if it doesn't, transaction costs are not being applied to AW_R."
    )
