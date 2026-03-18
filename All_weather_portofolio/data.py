"""
data.py
=======
Responsible for downloading and cleaning market price data from Yahoo Finance.

Only one public function: fetch_prices().
Nothing in this module should know about allocations, backtests, or portfolios --
it just retrieves and cleans raw price data.
"""

import warnings
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")   # suppress yfinance deprecation noise


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
    print(f"Fetching data | {' '.join(tickers)} | {start_date} to {end_date}")

    raw = yf.download(tickers, start=start_date, end=end_date,
                      progress=False, auto_adjust=True)

    # yfinance returns a MultiIndex when downloading multiple tickers
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.dropna(how="all")   # drop rows where ALL tickers are NaN
    prices = prices.ffill()             # forward-fill weekends and holidays

    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        print(f"WARNING: No data found for {missing}. Check ticker symbols.\n")

    return prices
