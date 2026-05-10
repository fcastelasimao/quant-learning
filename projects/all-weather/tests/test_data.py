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

import sqlite3
import importlib.util
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yfinance as yf

from engine import config
from engine.backtest import run_backtest
from engine.data import fetch_prices_from_fmp_db, get_price_provenance
from engine.stats import compute_calendar_year_metrics


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


# ===========================================================================
# FMP SQLITE DATA LOADER
# ===========================================================================

def _write_fmp_test_db(path, rows):
    with sqlite3.connect(path) as conn:
        conn.execute("""
            create table candles_1d (
                ts integer primary key,
                open real not null,
                high real not null,
                low real not null,
                close real not null,
                volume real not null,
                utc_datetime text not null,
                et_datetime text not null
            )
        """)
        conn.executemany("""
            insert into candles_1d
            (ts, open, high, low, close, volume, utc_datetime, et_datetime)
            values (?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)


def _load_root_fmp_downloader():
    root = Path(__file__).resolve().parents[3]
    path = root / "91_1_XXX_Data_Manager_FMP_single_database_v2_no_api.py"
    spec = importlib.util.spec_from_file_location("fmp_downloader_test_module", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_fetch_prices_from_fmp_db_loads_close_prices(tmp_path):
    rows_a = [
        (1, 10, 11, 9, 10.0, 100, "2020-01-02 05:00:00", "2020-01-02 00:00:00"),
        (2, 11, 12, 10, 11.0, 120, "2020-01-03 05:00:00", "2020-01-03 00:00:00"),
    ]
    rows_b = [
        (1, 20, 21, 19, 20.0, 100, "2020-01-02 05:00:00", "2020-01-02 00:00:00"),
        (2, 21, 22, 20, 21.0, 120, "2020-01-03 05:00:00", "2020-01-03 00:00:00"),
    ]
    _write_fmp_test_db(tmp_path / "DB_A_historical_data.db", rows_a)
    _write_fmp_test_db(tmp_path / "DB_B_historical_data.db", rows_b)

    prices = fetch_prices_from_fmp_db(["A", "B"], "2020-01-01", "2020-01-04",
                                      data_dir=str(tmp_path))

    assert list(prices.columns) == ["A", "B"]
    assert prices.loc[pd.Timestamp("2020-01-03"), "A"] == 11.0
    assert prices.loc[pd.Timestamp("2020-01-03"), "B"] == 21.0
    provenance = get_price_provenance(prices)
    assert provenance["source"] == "fmp_sqlite"
    assert provenance["price_column"] == "close"
    assert provenance["requested_tickers"] == ["A", "B"]


def test_fetch_prices_from_fmp_db_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        fetch_prices_from_fmp_db(["MISSING"], "2020-01-01", "2020-01-04",
                                 data_dir=str(tmp_path))


def test_fetch_prices_from_fmp_db_loads_adj_close(tmp_path):
    path = tmp_path / "DB_A_historical_data.db"
    with sqlite3.connect(path) as conn:
        conn.execute("""
            create table candles_1d (
                ts integer primary key,
                open real not null,
                high real not null,
                low real not null,
                close real not null,
                adj_close real,
                volume real not null,
                utc_datetime text not null,
                et_datetime text not null
            )
        """)
        conn.execute("""
            insert into candles_1d
            (ts, open, high, low, close, adj_close, volume, utc_datetime, et_datetime)
            values (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (1, 10, 11, 9, 10.0, 9.5, 100, "2020-01-02 05:00:00",
              "2020-01-02 00:00:00"))

    prices = fetch_prices_from_fmp_db(["A"], "2020-01-01", "2020-01-04",
                                      data_dir=str(tmp_path),
                                      price_column="adj_close")

    assert prices.loc[pd.Timestamp("2020-01-02"), "A"] == 9.5
    assert get_price_provenance(prices)["price_column"] == "adj_close"


def test_fetch_prices_from_fmp_db_requires_adj_close_column(tmp_path):
    rows = [
        (1, 10, 11, 9, 10.0, 100, "2020-01-02 05:00:00", "2020-01-02 00:00:00"),
    ]
    _write_fmp_test_db(tmp_path / "DB_A_historical_data.db", rows)

    with pytest.raises(ValueError, match="adj_close"):
        fetch_prices_from_fmp_db(["A"], "2020-01-01", "2020-01-04",
                                 data_dir=str(tmp_path),
                                 price_column="adj_close")


def test_config_validates_data_source(monkeypatch):
    monkeypatch.setattr(config, "DATA_SOURCE", "bad_source")
    with pytest.raises(AssertionError):
        config.validate_config()


def test_fmp_downloader_migrates_daily_table_to_adj_close(tmp_path):
    dm = _load_root_fmp_downloader()
    db_path = tmp_path / "DB_A_historical_data.db"
    conn = dm.sqlite_connect(str(db_path))
    try:
        conn.execute("""
            create table candles_1d (
                ts integer primary key,
                open real not null,
                high real not null,
                low real not null,
                close real not null,
                volume real not null,
                utc_datetime text not null,
                et_datetime text not null
            )
        """)
        conn.execute("""
            insert into candles_1d
            (ts, open, high, low, close, volume, utc_datetime, et_datetime)
            values (?, ?, ?, ?, ?, ?, ?, ?)
        """, (1, 10, 11, 9, 10.0, 100, "2020-01-02 05:00:00",
              "2020-01-02 00:00:00"))
        conn.commit()

        dm.ensure_table_and_columns(conn, "candles_1d")
        columns = {row[1] for row in conn.execute("pragma table_info(candles_1d);")}
        row = conn.execute("select close, adj_close from candles_1d where ts=1;").fetchone()
    finally:
        conn.close()

    assert "adj_close" in columns
    assert row == (10.0, None)


def test_fmp_downloader_parses_adj_close(monkeypatch):
    dm = _load_root_fmp_downloader()

    def fake_get_json(url):
        return {
            "historical": [
                {
                    "date": "2020-01-02",
                    "open": 10,
                    "high": 11,
                    "low": 9,
                    "close": 10,
                    "adjClose": 9.5,
                    "volume": 100,
                }
            ]
        }

    monkeypatch.setattr(dm, "http_get_json", fake_get_json)
    df = dm.fetch_daily_range("A", date(2020, 1, 2), date(2020, 1, 2), "key")

    assert list(df.columns) == dm.DAILY_COLUMNS
    assert df.loc[0, "adj_close"] == 9.5


def test_fmp_downloader_upserts_adj_close_without_losing_ohlcv(tmp_path):
    dm = _load_root_fmp_downloader()
    conn = dm.sqlite_connect(str(tmp_path / "DB_A_historical_data.db"))
    try:
        dm.ensure_table_and_columns(conn, "candles_1d")
        conn.execute("""
            insert into candles_1d
            (ts, open, high, low, close, volume, utc_datetime, et_datetime)
            values (?, ?, ?, ?, ?, ?, ?, ?)
        """, (1, 10, 11, 9, 10.0, 100, "2020-01-02 05:00:00",
              "2020-01-02 00:00:00"))
        conn.commit()

        df = pd.DataFrame([{
            "ts": 1,
            "open": 10,
            "high": 11,
            "low": 9,
            "close": 10.0,
            "adj_close": 9.5,
            "volume": 100,
            "utc_datetime": "2020-01-02 05:00:00",
            "et_datetime": "2020-01-02 00:00:00",
        }])
        dm.upsert_df(conn, "candles_1d", df)
        df_missing_adj = df.copy()
        df_missing_adj["close"] = 10.5
        df_missing_adj["adj_close"] = None
        dm.upsert_df(conn, "candles_1d", df_missing_adj)
        row = conn.execute(
            "select open, high, low, close, adj_close, volume from candles_1d where ts=1;"
        ).fetchone()
    finally:
        conn.close()

    assert row == (10.0, 11.0, 9.0, 10.5, 9.5, 100.0)


def test_compute_calendar_year_metrics_includes_pnl():
    idx = pd.to_datetime(["2020-01-31", "2020-12-31", "2021-12-31"])
    backtest = pd.DataFrame({
        "All Weather Value": [10_000.0, 11_000.0, 9_900.0],
        "S&P 500 Value": [10_000.0, 12_000.0, 13_200.0],
    }, index=idx)

    annual = compute_calendar_year_metrics(backtest)
    row = annual[(annual["Year"] == 2021) &
                 (annual["Strategy"] == "All Weather Value")].iloc[0]

    assert row["Start Value ($)"] == 11_000.0
    assert row["End Value ($)"] == 9_900.0
    assert row["PnL ($)"] == -1_100.0
    assert row["Return (%)"] == -10.0
