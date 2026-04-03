"""
Fetch historical and live OHLCV candles from Binance public API.

No API key required — Binance allows public access to klines (candle) data.
Candles are cached to disk so repeated backtest runs don't re-download.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

from config import STRATEGY
from models import Candle

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.binance.com/api/v3/klines"
_MAX_CANDLES_PER_REQUEST = 1000


def _parse_candles(raw: list) -> list[Candle]:
    candles = []
    for row in raw:
        candles.append(Candle(
            timestamp=datetime.fromtimestamp(row[0] / 1000, tz=timezone.utc),
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=float(row[5]),
        ))
    return candles


def fetch_candles(
    symbol: str,
    interval: str = "1h",
    limit: int = 720,
    end_time_ms: Optional[int] = None,
) -> list[Candle]:
    """
    Fetch up to `limit` candles for `symbol` from Binance.

    Handles pagination transparently — Binance caps each response at 1000
    candles, so large requests are split into multiple calls.

    Args:
        symbol:      Binance symbol, e.g. "ETHUSDT"
        interval:    Candle size: "1m", "5m", "1h", "4h", "1d", etc.
        limit:       Total number of candles to return (most recent first).
        end_time_ms: Optional Unix timestamp in milliseconds. If provided,
                     candles are fetched ending at this time.

    Returns:
        List of Candle objects in ascending time order.
    """
    all_candles: list[Candle] = []
    remaining = limit
    current_end = end_time_ms

    while remaining > 0:
        batch = min(remaining, _MAX_CANDLES_PER_REQUEST)
        params: dict = {"symbol": symbol, "interval": interval, "limit": batch}
        if current_end:
            params["endTime"] = current_end

        try:
            resp = requests.get(_BASE_URL, params=params, timeout=10)
            resp.raise_for_status()
            raw = resp.json()
        except requests.RequestException as e:
            logger.error(f"Binance API error for {symbol}: {e}")
            break

        if not raw:
            break

        batch_candles = _parse_candles(raw)
        all_candles = batch_candles + all_candles
        remaining -= len(batch_candles)

        if len(batch_candles) < batch:
            break  # No more historical data available

        # Move window back: next request ends just before the earliest candle
        current_end = int(batch_candles[0].timestamp.timestamp() * 1000) - 1
        time.sleep(0.1)  # Respect rate limits

    logger.info(f"Fetched {len(all_candles)} {interval} candles for {symbol}")
    return all_candles


def fetch_candles_cached(
    symbol: str,
    interval: str = "1h",
    limit: int = 720,
    cache_dir: str = "data/",
    max_cache_age_minutes: int = 60,
) -> list[Candle]:
    """
    Return cached candles if fresh, otherwise fetch from Binance and cache.

    Cache is a simple JSON file keyed by symbol + interval. Stale caches
    (older than max_cache_age_minutes) are refreshed automatically.
    """
    cache_path = Path(cache_dir) / f"{symbol}_{interval}_{limit}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        age_minutes = (time.time() - cache_path.stat().st_mtime) / 60
        if age_minutes < max_cache_age_minutes:
            try:
                with open(cache_path) as f:
                    raw = json.load(f)
                candles = [
                    Candle(
                        timestamp=datetime.fromisoformat(c["timestamp"]),
                        open=c["open"],
                        high=c["high"],
                        low=c["low"],
                        close=c["close"],
                        volume=c["volume"],
                    )
                    for c in raw
                ]
                logger.info(
                    f"Loaded {len(candles)} {interval} candles for {symbol} from cache"
                )
                return candles
            except Exception:
                pass  # Cache corrupt — re-fetch

    candles = fetch_candles(symbol, interval=interval, limit=limit)

    if candles:
        with open(cache_path, "w") as f:
            json.dump(
                [
                    {
                        "timestamp": c.timestamp.isoformat(),
                        "open": c.open,
                        "high": c.high,
                        "low": c.low,
                        "close": c.close,
                        "volume": c.volume,
                    }
                    for c in candles
                ],
                f,
            )
        logger.debug(f"Cached {len(candles)} candles to {cache_path}")

    return candles


def fetch_latest_close(symbol: str, interval: str = "1h") -> Optional[float]:
    """Return the most recent closed candle's close price for a symbol."""
    candles = fetch_candles(symbol, interval=interval, limit=2)
    if len(candles) < 2:
        return None
    return candles[-2].close  # -1 is the still-open candle; -2 is last closed
