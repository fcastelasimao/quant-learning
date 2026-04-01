"""
Bitstamp exchange client for fetching order book data and executing trades.
Uses CCXT for abstraction or direct REST API as a fallback.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, Optional, Tuple

import requests

from config import BITSTAMP_API_KEY, BITSTAMP_API_SECRET

logger = logging.getLogger(__name__)

try:
    import ccxt

    HAVE_CCXT = True
except ImportError:
    HAVE_CCXT = False
    logger.warning("CCXT not installed. Install with: pip install ccxt")


class BitstampClient:
    """Bitstamp exchange client for fetching order books and executing trades."""

    def __init__(self):
        self.name = "Bitstamp"
        self.api_key = BITSTAMP_API_KEY
        self.api_secret = BITSTAMP_API_SECRET
        self.exchange = None
        self._connected = False
        self.session = requests.Session()
        self._initialize()

    def _initialize(self):
        """Initialize Bitstamp exchange connection."""
        if self.api_key == "YOUR_BITSTAMP_API_KEY":
            logger.warning("Bitstamp API key not configured. Using public endpoints only.")
            return

        if HAVE_CCXT:
            try:
                self.exchange = ccxt.bitstamp({
                    "apiKey": self.api_key,
                    "secret": self.api_secret,
                    "enableRateLimit": True,
                })
                self.exchange.fetch_ticker("BTC/GBP")
                self._connected = True
                logger.info("Bitstamp connection successful (CCXT)")
            except Exception as e:
                logger.error(f"Failed to initialize Bitstamp via CCXT: {e}")
                self._connected = False
        else:
            logger.info("Using Bitstamp public endpoints (read-only)")
            self._connected = False

    def get_ticker(self, symbol: str) -> Optional[Dict]:
        """
        Fetch ticker data for a symbol (e.g., 'BTC/GBP').
        Returns: {bid, ask, mid, timestamp}
        """
        try:
            if HAVE_CCXT and self.exchange:
                ticker = self.exchange.fetch_ticker(symbol)
                bid = ticker.get("bid")
                ask = ticker.get("ask")
                if bid is None or ask is None:
                    return None
                return {
                    "exchange": "Bitstamp",
                    "symbol": symbol,
                    "bid": bid,
                    "ask": ask,
                    "mid": (bid + ask) / 2,
                    "timestamp": datetime.utcnow().isoformat(),
                }

            market_symbol = self._market_symbol(symbol)
            url = f"https://www.bitstamp.net/api/v2/ticker/{market_symbol}/"
            resp = self.session.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                bid = float(data.get("bid", 0))
                ask = float(data.get("ask", 0))
                return {
                    "exchange": "Bitstamp",
                    "symbol": symbol,
                    "bid": bid,
                    "ask": ask,
                    "mid": (bid + ask) / 2,
                    "timestamp": datetime.utcnow().isoformat(),
                }
        except Exception as e:
            logger.error(f"Failed to fetch Bitstamp ticker for {symbol}: {e}")
        return None

    def get_order_book(self, symbol: str, depth: int = 10) -> Optional[Dict]:
        """
        Fetch order book snapshot for a symbol.
        Returns: {bids: [[price, quantity], ...], asks: [...]}
        """
        try:
            if HAVE_CCXT and self.exchange:
                order_book = self.exchange.fetch_order_book(symbol, depth)
                return {
                    "exchange": "Bitstamp",
                    "symbol": symbol,
                    "bids": order_book.get("bids", []),
                    "asks": order_book.get("asks", []),
                    "timestamp": datetime.utcnow().isoformat(),
                }

            market_symbol = self._market_symbol(symbol)
            url = f"https://www.bitstamp.net/api/v2/order_book/{market_symbol}/"
            params = {"group": 1}
            resp = self.session.get(url, params=params, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "exchange": "Bitstamp",
                    "symbol": symbol,
                    "bids": data.get("bids", []),
                    "asks": data.get("asks", []),
                    "timestamp": datetime.utcnow().isoformat(),
                }
        except Exception as e:
            logger.error(f"Failed to fetch Bitstamp order book for {symbol}: {e}")
        return None

    def get_top_of_book(self, symbol: str, depth: int = 1) -> Optional[Dict]:
        """Fetch only the best bid/ask levels needed by the simulator."""
        order_book = self.get_order_book(symbol, depth=depth)
        if not order_book:
            return None

        bids = order_book.get("bids") or []
        asks = order_book.get("asks") or []
        if not bids or not asks:
            return None

        best_bid = bids[0]
        best_ask = asks[0]
        bid = float(best_bid[0])
        ask = float(best_ask[0])
        bid_volume = float(best_bid[1])
        ask_volume = float(best_ask[1])
        return {
            "bid": bid,
            "ask": ask,
            "bid_volume": bid_volume,
            "ask_volume": ask_volume,
            "mid": (bid + ask) / 2,
            "timestamp": order_book.get("timestamp", datetime.utcnow().isoformat()),
        }

    def get_best_price(self, symbol: str) -> Optional[Tuple[float, float]]:
        """Get best bid and ask prices."""
        ticker = self.get_ticker(symbol)
        if ticker:
            return (ticker["bid"], ticker["ask"])
        return None

    def test_connection(self) -> bool:
        """Test if connection to Bitstamp is working."""
        try:
            ticker = self.get_ticker("BTC/GBP")
            return ticker is not None
        except Exception as e:
            logger.error(f"Bitstamp connection test failed: {e}")
            return False

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: Optional[float] = None,
    ) -> Optional[Dict]:
        """
        Place a buy or sell order on Bitstamp.
        Returns order dict or None if it fails.
        """
        if not self._connected:
            logger.warning("Bitstamp not connected. Cannot place live orders.")
            return None

        try:
            if HAVE_CCXT and self.exchange:
                if price is None:
                    return self.exchange.create_market_order(symbol, side, quantity)
                return self.exchange.create_limit_order(symbol, side, quantity, price)
        except Exception as e:
            logger.error(f"Failed to place order on Bitstamp: {e}")
        return None

    def _market_symbol(self, symbol: str) -> str:
        """Convert CCXT-style market symbol to Bitstamp REST format."""
        return symbol.replace("/", "").lower()
