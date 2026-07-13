import pandas as pd
import yfinance as yf


def download_prices(tickers, start, end=None):
    """Download adjusted close prices. Returns DataFrame with tickers as columns."""
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"][tickers]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers
    prices = prices.dropna(how="all").ffill().dropna()
    return prices


def compute_returns(prices):
    """Daily simple returns."""
    return prices.pct_change().dropna()


def to_monthly(df):
    """Resample to month-end, taking the last available value."""
    return df.resample("M").last().dropna()
