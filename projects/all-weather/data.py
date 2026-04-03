"""
data.py
=======
Responsible for downloading and cleaning market price data from Yahoo Finance.

Only one public function: fetch_prices().
Nothing in this module should know about allocations, backtests, or portfolios --
it just retrieves and cleans raw price data.
"""

import warnings
from datetime import date
import pandas as pd
import yfinance as yf
import config

warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", message=".*auto_adjust.*")  # yfinance deprecation


def fetch_prices(tickers: list[str],
                 start_date: str,
                 end_date: str) -> pd.DataFrame:
    """
    Download adjusted closing prices for the given tickers
    between start_date and end_date.

    Parameters
    ----------
    tickers    : list of ticker symbols, e.g. ["VTI", "TLT", "GLD"]
    start_date : start date string in "YYYY-MM-DD" format
    end_date   : end date string in "YYYY-MM-DD" format

    Returns
    -------
    pd.DataFrame
        Daily closing prices, one column per ticker, indexed by date.
        Weekend and holiday gaps are forward-filled (see note below).

    Notes
    -----
    Forward-filling (ffill) is the standard approach for daily price data.
    Markets are closed on weekends and public holidays, so those dates have
    no price. Forward-filling means "use the last known closing price until
    a new one is available", which correctly reflects the value of your
    holdings on a non-trading day. Backfilling would introduce lookahead
    bias by using future prices to fill past dates.
    """
    auto_adjust = (config.PRICING_MODEL == "total_return")
    mode_label = ("total return (dividends reinvested)"
                  if auto_adjust else "price return only")
    print(f"Fetching data | {' '.join(tickers)} | "
          f"{start_date} to {end_date} | {mode_label}")

    # If results look identical after switching PRICING_MODEL,
    # clear the yfinance cache: rm -rf ~/.cache/py-yfinance
    raw = yf.download(tickers, start=start_date, end=end_date,
                      progress=False, auto_adjust=auto_adjust)

    # yfinance returns a MultiIndex when downloading multiple tickers
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.dropna(how="all")   # drop rows where ALL tickers are NaN
    prices = prices.ffill()             # forward-fill weekends and holidays

    # ===========================================================================
    # DATA QUALITY CHECKS
    # ===========================================================================

    # Assert no column is entirely NaN (would indicate a bad ticker or date range)
    for ticker in prices.columns:
        if prices[ticker].isna().all():
            raise ValueError(
                f"DATA QUALITY: ticker '{ticker}' has no data at all. "
                f"Check the symbol or the date range."
            )

    # Warn if any single daily return exceeds 30% (likely a data error)
    daily_returns = prices.pct_change()
    for ticker in daily_returns.columns:
        max_return = daily_returns[ticker].abs().max()
        if max_return > 0.30:
            print(
                f"DATA QUALITY WARNING: '{ticker}' has a single-day return of "
                f"{max_return:.1%} -- possible data error or split not adjusted."
            )

    # Assert no negative prices (would indicate corrupted or unadjusted data)
    for ticker in prices.columns:
        if (prices[ticker].dropna() < 0).any():
            raise AssertionError(
                f"DATA QUALITY: ticker '{ticker}' contains negative prices. "
                f"Data is likely corrupted."
            )

    # Warn if the last date in the index is more than 45 calendar days before today.
    # Total return (adjusted) data from yfinance may lag by 30-45 days for some tickers.
    last_date = prices.index[-1].date() if hasattr(prices.index[-1], "date") else prices.index[-1]
    days_stale = (date.today() - last_date).days
    if days_stale > 45:
        print(
            f"DATA QUALITY WARNING: last price date is {last_date} "
            f"({days_stale} calendar days ago). Total return data from yfinance may lag "
            f"by 30-45 days. If lag exceeds 60 days, verify data manually."
        )

    # ===========================================================================

    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        print(f"WARNING: No data found for {missing}. Check ticker symbols.\n")

    return prices
