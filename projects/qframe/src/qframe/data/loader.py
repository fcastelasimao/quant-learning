"""
Data loader — all data access goes through here.
Never import yfinance or openbb directly in research code.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from typing import Sequence


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_yfinance(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is not installed. Run: pip install yfinance")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    return raw


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_ohlcv(
    tickers: Sequence[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Load daily OHLCV for a list of tickers.

    Returns a MultiIndex DataFrame with columns (field, ticker):
        Open, High, Low, Close, Volume

    NOTE: Phase 0 uses yfinance directly — survivorship bias acknowledged.
    Phase 2 Gate 1+ must switch to Norgate Data via this same interface.

    Args:
        tickers: list of ticker symbols, e.g. ['AAPL', 'MSFT']
        start:   start date string 'YYYY-MM-DD'
        end:     end date string   'YYYY-MM-DD'

    Returns:
        pd.DataFrame with MultiIndex columns (field, ticker)
    """
    tickers = list(tickers)
    raw = _fetch_yfinance(tickers, start=start, end=end)

    if isinstance(raw.columns, pd.MultiIndex):
        return raw
    # Single ticker — yfinance returns flat columns; wrap into MultiIndex
    raw.columns = pd.MultiIndex.from_tuples(
        [(col, tickers[0]) for col in raw.columns]
    )
    return raw


def load_returns(
    tickers: Sequence[str],
    start: str,
    end: str,
    freq: str = "D",
) -> pd.DataFrame:
    """
    Load simple returns for a list of tickers.

    Args:
        tickers: list of ticker symbols
        start:   start date string 'YYYY-MM-DD'
        end:     end date string   'YYYY-MM-DD'
        freq:    'D' daily (default), 'W' weekly, 'M' monthly

    Returns:
        pd.DataFrame (date x ticker) of simple returns. First row is NaN.
    """
    ohlcv = load_ohlcv(tickers, start=start, end=end)
    close = ohlcv["Close"] if "Close" in ohlcv.columns.get_level_values(0) else ohlcv.xs("Close", axis=1, level=0)
    close = close.sort_index()

    if freq != "D":
        close = close.resample(freq).last()

    returns = close.pct_change()
    returns.columns.name = "ticker"
    return returns


def load_fundamentals(
    tickers: Sequence[str],
    fields: Sequence[str],
) -> pd.DataFrame:
    """
    Load point-in-time fundamental data.

    NOTE: Phase 0 stub — returns empty DataFrame with correct schema.
    Implement with OpenBB / FMP in Phase 1.

    Args:
        tickers: list of ticker symbols
        fields:  list of field names, e.g. ['pe_ratio', 'book_value', 'roe']

    Returns:
        pd.DataFrame with columns (field, ticker) — point-in-time
    """
    raise NotImplementedError(
        "load_fundamentals is not implemented in Phase 0. "
        "Add OpenBB / FMP integration in Phase 1."
    )


def load_sp500_tickers() -> list[str]:
    """
    Return the current S&P 500 ticker list scraped from Wikipedia.

    NOTE: This is NOT point-in-time — it reflects today's constituents.
    Survivorship bias is acknowledged for Phase 0/1 use.
    For point-in-time membership use load_sp500_historical_tickers() below.
    """
    import io
    import urllib.request

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    # Wikipedia returns 403 if no browser-like User-Agent is sent
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req) as resp:
            html = resp.read().decode("utf-8")
        tables = pd.read_html(io.StringIO(html))
        sp500 = tables[0]
        return sp500["Symbol"].str.replace(".", "-", regex=False).tolist()
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch S&P 500 tickers from Wikipedia: {exc}") from exc


def load_sp500_historical_tickers(
    as_of_date: str | None = None,
    cache_path: str | None = None,
) -> list[str]:
    """
    Return S&P 500 tickers that were members as of a given date (best-effort).

    This is a partial survivorship-bias fix using a free public dataset of
    historical S&P 500 additions and removals. It is NOT perfectly accurate —
    for production use, replace with Norgate Data or CRSP point-in-time files.

    Data source: github.com/datasets/s-and-p-500-companies-historical-components
    Falls back to current constituents if the dataset is unavailable.

    Args:
        as_of_date: 'YYYY-MM-DD' — return members as of this date.
                    None = use today's date (same as load_sp500_tickers).
        cache_path: Optional path to a locally cached CSV of historical changes.
                    CSV format: date, ticker, action ('added' or 'removed').

    Returns:
        list of ticker symbols (best-effort point-in-time membership).

    NOTE on survivorship bias:
        Even with this dataset, stocks that were never in the S&P 500 are excluded.
        True unbiased backtesting requires CRSP or Compustat universe with all
        listed NYSE/NASDAQ/AMEX stocks. This is a Phase 2 Gate 1 requirement.
    """
    import io
    import urllib.request
    from pathlib import Path

    if as_of_date is None:
        return load_sp500_tickers()

    target = pd.Timestamp(as_of_date)

    # Try to load from cache or fetch from GitHub
    _GITHUB_CSV = (
        "https://raw.githubusercontent.com/datasets/"
        "s-and-p-500-companies/main/data/constituents-changes.csv"
    )

    changes_df: pd.DataFrame | None = None

    if cache_path and Path(cache_path).exists():
        try:
            changes_df = pd.read_csv(cache_path, parse_dates=["date"])
        except Exception:
            changes_df = None

    if changes_df is None:
        try:
            req = urllib.request.Request(_GITHUB_CSV, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = resp.read().decode("utf-8")
            changes_df = pd.read_csv(io.StringIO(raw), parse_dates=["date"])
            if cache_path:
                Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                changes_df.to_csv(cache_path, index=False)
        except Exception:
            changes_df = None

    if changes_df is None:
        # Fallback: current constituents with warning
        import warnings
        warnings.warn(
            f"Could not fetch historical S&P 500 membership for {as_of_date}. "
            "Falling back to current constituents (survivorship-biased). "
            "For point-in-time data, provide a cache_path CSV or use Norgate/CRSP.",
            stacklevel=2,
        )
        return load_sp500_tickers()

    # Reconstruct membership as of target date
    # Start from current members and work backwards
    current = set(load_sp500_tickers())

    # Apply changes in reverse chronological order
    changes_sorted = changes_df.sort_values("date", ascending=False)
    for _, row in changes_sorted.iterrows():
        if row["date"] <= target:
            break
        ticker = str(row.get("ticker", row.get("symbol", ""))).strip().replace(".", "-")
        action = str(row.get("action", row.get("change", ""))).lower()
        if "add" in action:
            current.discard(ticker)   # added after target → wasn't a member yet
        elif "remov" in action or "delet" in action:
            current.add(ticker)       # removed after target → was still a member

    return sorted(current)


def load_survivorship_free_prices(
    start: str = "2010-01-01",
    end: str = "2024-12-31",
    cache_dir: str = "data/processed",
    min_history_days: int = 252,
) -> pd.DataFrame:
    """
    Load close prices for a survivorship-bias-aware universe.

    For each year in the backtest window, expands the universe to include
    any stock that was a member of the S&P 500 at any point during that year.
    This means stocks that got removed mid-year are still included up to removal.

    NOTE: This is an approximation — true point-in-time data requires CRSP.
    The improvement over load_sp500_tickers() is meaningful but not perfect.

    Args:
        start:             price history start date
        end:               price history end date
        cache_dir:         directory for parquet cache
        min_history_days:  minimum trading days of history required to include stock

    Returns:
        pd.DataFrame (date x ticker) of adjusted close prices
    """
    from pathlib import Path

    cache_path = Path(cache_dir) / "sp500_close_expanded.parquet"
    changes_cache = Path(cache_dir) / "sp500_changes.csv"

    if cache_path.exists():
        print(f"Loading expanded universe prices from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    print("Building survivorship-bias-reduced universe (this may take 5-10 minutes)...")

    # Collect all tickers that were ever in S&P 500 during the backtest window
    all_tickers: set[str] = set(load_sp500_tickers())

    # Add historical members if possible
    for year in range(int(start[:4]), int(end[:4]) + 1):
        try:
            hist = load_sp500_historical_tickers(
                as_of_date=f"{year}-01-01",
                cache_path=str(changes_cache),
            )
            all_tickers.update(hist)
        except Exception:
            pass

    print(f"  Expanded universe: {len(all_tickers)} unique tickers")

    # Fetch prices in batches to avoid yfinance throttling
    import time
    tickers = sorted(all_tickers)
    batch_size = 100
    frames: list[pd.DataFrame] = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i: i + batch_size]
        try:
            ohlcv = load_ohlcv(batch, start=start, end=end)
            close = ohlcv["Close"].sort_index()
            frames.append(close)
            if i + batch_size < len(tickers):
                time.sleep(1)  # gentle throttle
        except Exception as e:
            print(f"  Batch {i//batch_size + 1} failed: {e}")

    if not frames:
        raise RuntimeError("No price data fetched")

    combined = pd.concat(frames, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]

    # Filter: require at least min_history_days of data and ≤20% NaN
    valid = (combined.count() >= min_history_days) & (combined.isna().mean() < 0.20)
    combined = combined.loc[:, valid]
    n_gaps = int(combined.isna().sum().sum())
    if n_gaps > 0:
        warnings.warn(
            f"Forward-filling {n_gaps} NaN price entries (limit=5 days). "
            "This may introduce artificial 0% returns for stocks with data gaps. "
            "Check data quality for affected tickers.",
            UserWarning,
            stacklevel=2,
        )
    combined = combined.ffill(limit=5)

    print(f"  Final universe: {combined.shape[1]} tickers after quality filter")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(cache_path)
    print(f"  Cached to {cache_path}")
    return combined


def load_volume(
    start: str = "2010-01-01",
    end: str = "2024-12-31",
    cache_dir: str = "data/processed",
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load daily trading volume for S&P 500 constituents.

    Volume is required for ADV-based market impact estimation (Phase 3 Gate 0).
    It also enables volume-signal factors (e.g. volume-price trend, amihud illiquidity).

    Data is cached to {cache_dir}/sp500_volume.parquet alongside sp500_close.parquet.
    No extra network requests are made if the cache exists.

    Args:
        start:      history start date
        end:        history end date
        cache_dir:  directory for parquet cache
        tickers:    explicit list of tickers; if None, uses current S&P 500 constituents

    Returns:
        pd.DataFrame (date x ticker) of daily share volume
    """
    from pathlib import Path

    cache_path = Path(cache_dir) / "sp500_volume.parquet"

    if cache_path.exists():
        print(f"Loading volume from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    print("Fetching volume data from yfinance...")
    if tickers is None:
        tickers = load_sp500_tickers()

    raw = _fetch_yfinance(tickers, start=start, end=end)

    if isinstance(raw.columns, pd.MultiIndex):
        volume = raw["Volume"].sort_index()
    else:
        volume = raw[["Volume"]].sort_index()

    # Forward-fill short gaps (max 5 days), consistent with close price loader
    n_gaps = int(volume.isna().sum().sum())
    if n_gaps > 0:
        warnings.warn(
            f"Forward-filling {n_gaps} NaN volume entries (limit=5 days).",
            UserWarning,
            stacklevel=2,
        )
    volume = volume.ffill(limit=5)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    volume.to_parquet(cache_path)
    print(f"  Cached to {cache_path}  shape: {volume.shape}")
    return volume
