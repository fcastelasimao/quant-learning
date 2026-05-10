"""
data.py
=======
Responsible for downloading and cleaning market price data from Yahoo Finance
or repo-local FMP SQLite files.

Only one public function: fetch_prices().
Nothing in this module should know about allocations, backtests, or portfolios --
it just retrieves and cleans raw price data.
"""

from __future__ import annotations

import os
import sqlite3
import warnings
from datetime import date
from typing import Any
import pandas as pd
import yfinance as yf
from . import config

warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", message=".*auto_adjust.*")  # yfinance deprecation

DATA_STALENESS_WARNING_DAYS = 45


def _attach_provenance(prices: pd.DataFrame,
                       *,
                       source: str,
                       tickers: list[str],
                       start_date: str,
                       end_date: str,
                       price_column: str,
                       pricing_model: str | None = None,
                       data_dir: str | None = None) -> pd.DataFrame:
    """Attach reproducibility metadata to a returned price DataFrame."""
    out = prices.copy()
    if out.empty:
        first_date = None
        last_date = None
        missingness: dict[str, float] = {}
    else:
        first_date = out.index[0].date().isoformat()
        last_date = out.index[-1].date().isoformat()
        missingness = {
            column: round(float(out[column].isna().mean()), 6)
            for column in out.columns
        }
    out.attrs["provenance"] = {
        "source": source,
        "requested_tickers": list(tickers),
        "returned_columns": list(out.columns),
        "requested_start": start_date,
        "requested_end": end_date,
        "actual_start": first_date,
        "actual_end": last_date,
        "price_column": price_column,
        "pricing_model": pricing_model,
        "data_dir": data_dir,
        "retrieved_on": date.today().isoformat(),
        "missing_fraction_by_column": missingness,
    }
    return out


def get_price_provenance(prices: pd.DataFrame) -> dict[str, Any]:
    """Return price provenance attached by fetch_prices*, or an empty dict."""
    return dict(prices.attrs.get("provenance", {}))


def _repo_data_dir() -> str:
    """Return the repo-level data directory that holds FMP SQLite files."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo_root = os.path.dirname(os.path.dirname(project_root))
    return os.path.join(repo_root, "data")


def _run_price_quality_checks(prices: pd.DataFrame,
                              tickers: list[str],
                              source_label: str) -> None:
    """Shared sanity checks for downloaded or locally loaded price data."""
    all_nan_cols = prices.columns[prices.isna().all()]
    if len(all_nan_cols) > 0:
        raise ValueError(
            f"DATA QUALITY ({source_label}): tickers {list(all_nan_cols)} have no data at all. "
            f"Check the symbols or the date range."
        )

    suspicious_return_threshold = 0.30
    daily_returns = prices.pct_change()
    max_abs_returns = daily_returns.abs().max()
    suspicious = max_abs_returns[max_abs_returns > suspicious_return_threshold]
    for ticker, max_ret in suspicious.items():
        print(
            f"DATA QUALITY WARNING ({source_label}): '{ticker}' has a single-day return of "
            f"{max_ret:.1%} -- possible data error or split not adjusted."
        )

    neg_cols = prices.columns[(prices.dropna() < 0).any()]
    if len(neg_cols) > 0:
        raise AssertionError(
            f"DATA QUALITY ({source_label}): tickers {list(neg_cols)} contain negative prices. "
            f"Data is likely corrupted."
        )

    last_date = prices.index[-1].date() if hasattr(prices.index[-1], "date") else prices.index[-1]
    days_stale = (date.today() - last_date).days
    if days_stale > DATA_STALENESS_WARNING_DAYS:
        print(
            f"DATA QUALITY WARNING ({source_label}): last price date is {last_date} "
            f"({days_stale} calendar days ago). Verify the data source before using "
            f"these results for live decisions."
        )

    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        print(f"WARNING ({source_label}): No data found for {missing}. Check ticker symbols.\n")


def fetch_prices(tickers: list[str],
                 start_date: str,
                 end_date: str) -> pd.DataFrame:
    """Load prices from the configured data source."""
    if config.DATA_SOURCE == "yfinance":
        return fetch_prices_from_yfinance(tickers, start_date, end_date)
    if config.DATA_SOURCE == "fmp":
        return fetch_prices_from_fmp_db(
            tickers,
            start_date,
            end_date,
            data_dir=config.FMP_DATA_DIR,
            price_column=config.FMP_PRICE_COLUMN,
        )
    raise ValueError(f"Unknown DATA_SOURCE: {config.DATA_SOURCE}")


def fetch_prices_from_yfinance(tickers: list[str],
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

    if prices.empty:
        raise ValueError(
            f"yfinance returned no price rows for {tickers} between "
            f"{start_date} and {end_date}. Check network access and symbols."
        )

    _run_price_quality_checks(prices, tickers, "yfinance")

    return _attach_provenance(
        prices,
        source="yfinance",
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        price_column="Close",
        pricing_model=config.PRICING_MODEL,
    )


def fetch_prices_from_fmp_db(tickers: list[str],
                             start_date: str,
                             end_date: str,
                             data_dir: str | None = None,
                             price_column: str = "close") -> pd.DataFrame:
    """
    Load daily ETF prices from repo-local FMP SQLite candle databases.

    Expected file pattern:
        data/DB_<TICKER>_historical_data.db

    Each DB must contain a candles_1d table with at least:
        utc_datetime, open, high, low, close, volume
    and may also contain:
        adj_close

    Notes
    -----
    `close` is the ordinary FMP daily close. `adj_close` is the dividend-
    adjusted close populated by the FMP downloader's adj_close backfill.
    """
    if data_dir is None:
        data_dir = _repo_data_dir()

    valid_price_columns = {"open", "high", "low", "close", "adj_close"}
    if price_column not in valid_price_columns:
        raise ValueError(
            "price_column must be one of: open, high, low, close, adj_close"
        )

    frames = []
    missing_files = []
    for ticker in tickers:
        path = os.path.join(data_dir, f"DB_{ticker}_historical_data.db")
        if not os.path.exists(path):
            missing_files.append(ticker)
            continue

        uri = f"file:{path}?mode=ro"
        with sqlite3.connect(uri, uri=True) as conn:
            info = conn.execute("PRAGMA table_info(candles_1d);").fetchall()
            columns = {row[1] for row in info}
            if price_column not in columns:
                raise ValueError(
                    f"FMP SQLite file for {ticker} does not contain "
                    f"'{price_column}'. Run the adj_close backfill first "
                    f"if you selected FMP_PRICE_COLUMN='adj_close'."
                )

        query = f"""
            select utc_datetime, {price_column} as price
            from candles_1d
            where date(utc_datetime) >= date(?)
              and date(utc_datetime) < date(?)
            order by utc_datetime
        """
        with sqlite3.connect(uri, uri=True) as conn:
            df = pd.read_sql_query(query, conn, params=(start_date, end_date))

        if df.empty:
            frames.append(pd.Series(dtype=float, name=ticker))
            continue

        if price_column == "adj_close":
            null_fraction = float(df["price"].isna().mean())
            if null_fraction > 0.05:
                raise ValueError(
                    f"FMP adj_close for {ticker} is {null_fraction:.1%} null "
                    f"between {start_date} and {end_date}. Run the "
                    f"adj_close backfill before using this source."
                )

        idx = pd.to_datetime(df["utc_datetime"], utc=True).dt.tz_convert(None).dt.normalize()
        series = pd.Series(df["price"].astype(float).values, index=idx, name=ticker)
        series = series[~series.index.duplicated(keep="last")].sort_index()
        frames.append(series)

    if missing_files:
        raise FileNotFoundError(
            f"Missing FMP DB files for {missing_files} in {data_dir}"
        )
    if not frames:
        raise ValueError("No FMP price series were loaded.")

    prices = pd.concat(frames, axis=1).dropna(how="all").ffill()
    if prices.empty:
        raise ValueError(f"No FMP prices found between {start_date} and {end_date}.")

    _run_price_quality_checks(prices, tickers, "FMP SQLite")
    return _attach_provenance(
        prices,
        source="fmp_sqlite",
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        price_column=price_column,
        pricing_model=None,
        data_dir=data_dir,
    )
