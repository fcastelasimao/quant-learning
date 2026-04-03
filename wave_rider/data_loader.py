from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


def _reshape_download(raw_data: pd.DataFrame, tickers: dict[str, str]) -> pd.DataFrame:
    if isinstance(raw_data.columns, pd.MultiIndex):
        price_field = "Adj Close" if "Adj Close" in raw_data.columns.get_level_values(0) else "Close"
        prices = raw_data[price_field]
    else:
        prices = raw_data

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    ticker_to_asset = {ticker: asset for asset, ticker in tickers.items()}
    prices = prices.rename(columns=ticker_to_asset)

    missing_assets = [asset for asset in tickers if asset not in prices.columns]
    if missing_assets:
        raise ValueError(f"Missing downloaded prices for assets: {missing_assets}")

    ordered = prices[list(tickers.keys())].sort_index().dropna(how="all")
    return ordered


def load_data(
    tickers: dict[str, str],
    start: str,
    cache_file: Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    if cache_file and cache_file.exists() and not refresh:
        cached = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return cached.sort_index()

    raw_data = yf.download(
        list(tickers.values()),
        start=start,
        progress=False,
        auto_adjust=False,
    )
    prices = _reshape_download(raw_data, tickers)

    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(cache_file)

    return prices


def load_vix(
    start: str,
    cache_file: Path | None = None,
    refresh: bool = False,
) -> pd.Series:
    """Download VIX close prices as a pd.Series (for HMM regime detection)."""
    if cache_file and cache_file.exists() and not refresh:
        cached = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return cached.squeeze().sort_index()

    raw = yf.download("^VIX", start=start, progress=False, auto_adjust=False)
    if isinstance(raw.columns, pd.MultiIndex):
        vix = raw["Close"].squeeze()
    else:
        vix = raw["Close"].squeeze()

    vix = vix.dropna().sort_index()
    vix.name = "VIX"

    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        vix.to_csv(cache_file)

    return vix
