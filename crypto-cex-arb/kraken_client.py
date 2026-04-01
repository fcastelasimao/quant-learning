"""
Kraken Exchange client for fetching order book data and executing trades.
Uses CCXT or direct REST API with Kraken's signature scheme.
"""
from __future__ import annotations
import logging
from datetime import datetime
from typing import Optional, Dict, Tuple

import requests

from config import KRAKEN_API_KEY, KRAKEN_API_SECRET

logger = logging.getLogger(__name__)

# Try to use CCXT for cleaner API
try:
    import ccxt
    HAVE_CCXT = True
except ImportError:
    HAVE_CCXT = False
    logger.warning("CCXT not installed. Install with: pip install ccxt")


class KrakenClient:
    """Kraken exchange client for fetching order books and executing trades."""

    def __init__(self):
        self.name = "Kraken"
        self.api_key = KRAKEN_API_KEY
        self.api_secret = KRAKEN_API_SECRET
        self.exchange = None
        self._connected = False
        self.session = requests.Session()
        self._initialize()

    def _initialize(self):
        """Initialize Kraken exchange connection."""
        if self.api_key == "YOUR_KRAKEN_API_KEY":
            logger.warning("Kraken API key not configured. Using public endpoints only.")
            return

        if HAVE_CCXT:
            try:
                self.exchange = ccxt.kraken({
                    "apiKey": self.api_key,
                    "secret": self.api_secret,
                    "enableRateLimit": True,
                })
                # Test connection
                self.exchange.fetch_ticker("BTC/GBP")
                self._connected = True
                logger.info("Kraken connection successful (CCXT)")
            except Exception as e:
                logger.error(f"Failed to initialize Kraken via CCXT: {e}")
                self._connected = False
        else:
            logger.info("Using Kraken public endpoints (read-only)")
            self._connected = False

    def get_ticker(self, symbol: str) -> Optional[Dict]:
        """
        Fetch ticker data for a symbol (e.g., 'BTC/GBP').
        Returns: {bid, ask, mid, timestamp}
        """
        try:
            kraken_symbol = self._normalize_symbol(symbol)

            if HAVE_CCXT and self.exchange:
                ticker = self.exchange.fetch_ticker(kraken_symbol)
                return {
                    "exchange": "Kraken",
                    "symbol": symbol,
                    "bid": ticker.get("bid"),
                    "ask": ticker.get("ask"),
                    "mid": (ticker.get("bid", 0) + ticker.get("ask", 0)) / 2,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            else:
                # Fallback: public REST endpoint
                kraken_pair = kraken_symbol.replace("/", "")
                url = "https://api.kraken.com/0/public/Ticker"
                params = {"pair": kraken_pair}
                resp = self.session.get(url, params=params, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("result"):
                        # Kraken returns result[pair]
                        pair_data = list(data["result"].values())[0]
                        bid = float(pair_data.get("b", [0])[0])
                        ask = float(pair_data.get("a", [0])[0])
                        return {
                            "exchange": "Kraken",
                            "symbol": symbol,
                            "bid": bid,
                            "ask": ask,
                            "mid": (bid + ask) / 2,
                            "timestamp": datetime.utcnow().isoformat(),
                        }
        except Exception as e:
            logger.error(f"Failed to fetch Kraken ticker for {symbol}: {e}")
        return None

    def get_order_book(self, symbol: str, depth: int = 10) -> Optional[Dict]:
        """
        Fetch order book snapshot for a symbol.
        Returns: {bids: [[price, quantity], ...], asks: [...]}
        """
        try:
            kraken_symbol = self._normalize_symbol(symbol)

            if HAVE_CCXT and self.exchange:
                order_book = self.exchange.fetch_order_book(kraken_symbol, depth)
                return {
                    "exchange": "Kraken",
                    "symbol": symbol,
                    "bids": order_book.get("bids", []),
                    "asks": order_book.get("asks", []),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            else:
                # Fallback: public REST endpoint
                kraken_pair = kraken_symbol.replace("/", "")
                url = "https://api.kraken.com/0/public/Depth"
                params = {"pair": kraken_pair, "count": depth}
                resp = self.session.get(url, params=params, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("result"):
                        pair_data = list(data["result"].values())[0]
                        return {
                            "exchange": "Kraken",
                            "symbol": symbol,
                            "bids": pair_data.get("bids", []),
                            "asks": pair_data.get("asks", []),
                            "timestamp": datetime.utcnow().isoformat(),
                        }
        except Exception as e:
            logger.error(f"Failed to fetch Kraken order book for {symbol}: {e}")
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
        """
        Get best bid and ask prices.
        Returns: (bid, ask) or None if fetch fails
        """
        ticker = self.get_ticker(symbol)
        if ticker:
            return (ticker["bid"], ticker["ask"])
        return None

    def get_liquidity_at_price(self, symbol: str, side: str, price: float, depth: int = 50) -> float:
        """
        Calculate cumulative liquidity available at or better than price.
        side: 'buy' or 'sell'
        Returns: total quantity available
        """
        try:
            ob = self.get_order_book(symbol, depth)
            if not ob:
                return 0.0

            if side == "buy":
                quantity = sum(float(q) for p, q in ob.get("asks", []) if float(p) <= price)
            else:
                quantity = sum(float(q) for p, q in ob.get("bids", []) if float(p) >= price)

            return quantity
        except Exception as e:
            logger.error(f"Failed to calculate Kraken liquidity: {e}")
            return 0.0

    def test_connection(self) -> bool:
        """Test if connection to Kraken is working."""
        try:
            ticker = self.get_ticker("BTC/GBP")
            return ticker is not None
        except Exception as e:
            logger.error(f"Kraken connection test failed: {e}")
            return False

    def place_order(self, symbol: str, side: str, quantity: float, price: Optional[float] = None) -> Optional[Dict]:
        """
        Place a buy or sell order on Kraken.
        side: 'buy' or 'sell'
        price: if None, places market order; otherwise limit order
        
        Returns: order dict or None if fails
        Paper trading: This will be mocked to return simulated order
        """
        if not self._connected:
            logger.warning("Kraken not connected. Cannot place live orders.")
            return None

        try:
            if HAVE_CCXT and self.exchange:
                kraken_symbol = self._normalize_symbol(symbol)
                if price is None:
                    order = self.exchange.create_market_order(kraken_symbol, side, quantity)
                else:
                    order = self.exchange.create_limit_order(kraken_symbol, side, quantity, price)
                return order
        except Exception as e:
            logger.error(f"Failed to place order on Kraken: {e}")
        return None

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol for Kraken API."""
        return symbol
