"""
Crypto price loader — Binance public OHLCV via ccxt.

No API key required: Binance's public REST endpoint is used for historical
daily OHLCV data. Rate-limited to ~10 requests/second by ccxt automatically.

Usage:
    from qframe.data.crypto import load_binance_close

    prices = load_binance_close(
        symbols=["BTC", "ETH", "SOL"],
        start="2018-01-01",
        end="2024-12-31",
    )
    # Returns (dates x symbols) DataFrame of USDT close prices

Default universe (25 liquid pairs):
    BTC ETH SOL BNB XRP ADA AVAX DOT LINK MATIC UNI ATOM LTC BCH
    ETC DOGE XLM ALGO VET FIL AAVE CRV NEAR TRX EOS
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Default crypto universe  (Binance USDT pairs, broadly liquid since ≤2021)
# ---------------------------------------------------------------------------

DEFAULT_CRYPTO_SYMBOLS: list[str] = [
    "BTC", "ETH", "SOL", "BNB", "XRP",
    "ADA", "AVAX", "DOT", "LINK", "MATIC",
    "UNI", "ATOM", "LTC", "BCH", "ETC",
    "DOGE", "XLM", "ALGO", "VET", "FIL",
    "AAVE", "CRV", "NEAR", "TRX", "EOS",
]

_DEFAULT_CACHE = Path("data/processed/crypto_close.parquet")
_BINANCE_LIMIT = 1000        # max candles per ccxt request
_SLEEP_BETWEEN = 0.12        # seconds between requests (≈8 req/s, well below limit)


def _save_df(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to parquet with CSV fallback (for envs without pyarrow)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path)
    except ImportError:
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path)
        print(f"[crypto] parquet unavailable — saved as CSV: {csv_path}")


def _load_df(path: Path) -> pd.DataFrame | None:
    """Load DataFrame from parquet or CSV fallback."""
    if path.exists():
        try:
            return pd.read_parquet(path)
        except ImportError:
            pass
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        return df
    return None


# ---------------------------------------------------------------------------
# Low-level fetcher
# ---------------------------------------------------------------------------

def _fetch_symbol(
    exchange,
    symbol: str,
    start_ms: int,
    end_ms: int,
    timeframe: str = "1d",
) -> pd.Series:
    """
    Fetch OHLCV for a single symbol from Binance and return the close Series.

    Paginates automatically until end_ms is reached.  Empty if symbol not
    available on Binance or listed after start_ms.
    """
    ohlcvs: list[list] = []
    since = start_ms

    while since < end_ms:
        try:
            batch = exchange.fetch_ohlcv(
                f"{symbol}/USDT",
                timeframe=timeframe,
                since=since,
                limit=_BINANCE_LIMIT,
            )
        except Exception as exc:
            warnings.warn(f"[crypto] {symbol}: fetch error — {exc}", stacklevel=3)
            break

        if not batch:
            break

        ohlcvs.extend(batch)
        last_ts = batch[-1][0]
        if last_ts >= end_ms or len(batch) < _BINANCE_LIMIT:
            break
        since = last_ts + 1
        time.sleep(_SLEEP_BETWEEN)

    if not ohlcvs:
        return pd.Series(dtype=float, name=symbol)

    df = pd.DataFrame(ohlcvs, columns=["ts", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.normalize().dt.tz_localize(None)
    df = df.drop_duplicates("date").set_index("date")["close"].rename(symbol)
    return df.loc[df.index < pd.Timestamp(end_ms, unit="ms")]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_binance_close(
    symbols: Sequence[str] = DEFAULT_CRYPTO_SYMBOLS,
    start: str = "2018-01-01",
    end: str = "2024-12-31",
    timeframe: str = "1d",
    cache_path: str | Path | None = _DEFAULT_CACHE,
    force_refresh: bool = False,
    min_history_days: int = 252,
) -> pd.DataFrame:
    """
    Load daily close prices for crypto symbols from Binance (public API).

    Fetches USDT pairs (e.g. BTC/USDT).  Results are cached to parquet so
    subsequent calls are instant.  Pass ``force_refresh=True`` to re-fetch.

    Args:
        symbols:          List of base asset symbols, e.g. ["BTC", "ETH", "SOL"].
                          Default: DEFAULT_CRYPTO_SYMBOLS (25 liquid pairs).
        start:            History start date 'YYYY-MM-DD'.
        end:              History end date   'YYYY-MM-DD' (inclusive).
        timeframe:        ccxt timeframe string (default '1d' = daily).
        cache_path:       Parquet path for caching; None = no cache.
        force_refresh:    Ignore existing cache and re-fetch.
        min_history_days: Drop symbols with fewer than this many non-NaN days.

    Returns:
        pd.DataFrame (dates × symbols) of USDT close prices, sorted ascending.
        Columns that were listed after `start` contain NaN for missing dates.

    Raises:
        ImportError: if ccxt is not installed.
    """
    try:
        import ccxt
    except ImportError:
        raise ImportError("ccxt is not installed. Run: pip install ccxt")

    # Cache hit
    if cache_path is not None and not force_refresh:
        cp = Path(cache_path)
        cached = _load_df(cp)
        if cached is not None:
            print(f"[crypto] Loading from cache: {cp}")
            cols = [s for s in symbols if s in cached.columns]
            return cached.loc[start:end, cols] if cols else cached.loc[start:end]

    symbols = list(symbols)
    print(f"[crypto] Fetching {len(symbols)} symbols from Binance ({start} → {end})")

    exchange = ccxt.binance({"enableRateLimit": True})

    start_ms = int(pd.Timestamp(start).timestamp() * 1000)
    end_ms   = int((pd.Timestamp(end) + pd.Timedelta(days=1)).timestamp() * 1000)

    series: list[pd.Series] = []
    for i, sym in enumerate(symbols):
        print(f"  [{i+1:2d}/{len(symbols)}] {sym}/USDT ...", end=" ", flush=True)
        s = _fetch_symbol(exchange, sym, start_ms, end_ms, timeframe)
        n = s.notna().sum()
        print(f"{n} days")
        series.append(s)
        time.sleep(_SLEEP_BETWEEN)

    prices = pd.concat(series, axis=1).sort_index()
    prices.index = pd.to_datetime(prices.index)

    # Keep only trading days (remove weekends if any slipped through)
    # Crypto trades 24/7 so all dates should be present — just normalise
    prices = prices.loc[start:end]

    # Drop symbols with insufficient history
    valid = prices.count() >= min_history_days
    n_dropped = (~valid).sum()
    if n_dropped:
        dropped = prices.columns[~valid].tolist()
        warnings.warn(
            f"[crypto] Dropped {n_dropped} symbols with < {min_history_days} days of data: "
            f"{dropped}",
            UserWarning,
            stacklevel=2,
        )
    prices = prices.loc[:, valid]

    print(f"[crypto] Final universe: {prices.shape[1]} symbols × {prices.shape[0]} days")

    # Cache
    if cache_path is not None:
        cp = Path(cache_path)
        _save_df(prices, cp)
        print(f"[crypto] Cached to {cp}")

    return prices


def load_binance_ohlcv(
    symbols: Sequence[str] = DEFAULT_CRYPTO_SYMBOLS,
    start: str = "2018-01-01",
    end: str = "2024-12-31",
) -> dict[str, pd.DataFrame]:
    """
    Load full OHLCV (not just close) for multiple symbols.

    Returns a dict keyed by symbol, each value a (dates × [open,high,low,close,volume])
    DataFrame.  Useful for volume-based factors.  Not cached (use load_binance_close
    for the cached close-only version).

    Args:
        symbols: list of base asset symbols.
        start:   history start date.
        end:     history end date.

    Returns:
        dict[symbol → pd.DataFrame(open, high, low, close, volume)]
    """
    try:
        import ccxt
    except ImportError:
        raise ImportError("ccxt is not installed. Run: pip install ccxt")

    exchange = ccxt.binance({"enableRateLimit": True})
    start_ms = int(pd.Timestamp(start).timestamp() * 1000)
    end_ms   = int((pd.Timestamp(end) + pd.Timedelta(days=1)).timestamp() * 1000)

    result: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        ohlcvs: list[list] = []
        since = start_ms
        while since < end_ms:
            try:
                batch = exchange.fetch_ohlcv(
                    f"{sym}/USDT", timeframe="1d", since=since, limit=_BINANCE_LIMIT
                )
            except Exception as exc:
                warnings.warn(f"[crypto] {sym}: {exc}", stacklevel=2)
                break
            if not batch:
                break
            ohlcvs.extend(batch)
            last_ts = batch[-1][0]
            if last_ts >= end_ms or len(batch) < _BINANCE_LIMIT:
                break
            since = last_ts + 1
            time.sleep(_SLEEP_BETWEEN)

        if ohlcvs:
            df = pd.DataFrame(ohlcvs, columns=["ts", "open", "high", "low", "close", "volume"])
            df["date"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.normalize().dt.tz_localize(None)
            df = df.drop_duplicates("date").set_index("date").drop(columns="ts")
            result[sym] = df.loc[start:end]

    return result
