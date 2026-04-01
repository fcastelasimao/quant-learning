"""
Binance Exchange client for fetching order book data and executing trades.
Uses CCXT for abstraction or direct REST API.
"""
from __future__ import annotations
import logging
import os
from datetime import datetime
from typing import Dict, Optional, Tuple

BINANCE_API_KEY = os.environ.get("BINANCE_API_KEY", "YOUR_BINANCE_API_KEY")
BINANCE_API_SECRET = os.environ.get("BINANCE_API_SECRET", "YOUR_BINANCE_API_SECRET")

logger = logging.getLogger(__name__)

# Try to use CCXT for cleaner API, fall back to requests if not available
try:
    import ccxt
    HAVE_CCXT = True
except ImportError:
    HAVE_CCXT = False
    logger.warning("CCXT not installed. Install with: pip install ccxt")


class BinanceClient:
    """Binance exchange client for fetching order books and executing trades."""

    def __init__(self):
        self.name = "Binance"
        self.api_key = BINANCE_API_KEY
        self.api_secret = BINANCE_API_SECRET
        self.exchange = None
        self._connected = False
        self._initialize()

    def _initialize(self):
        """Initialize Binance exchange connection."""
        if self.api_key == "YOUR_BINANCE_API_KEY":
            logger.warning("Binance API key not configured. Using public endpoints only.")
            return

        if HAVE_CCXT:
            try:
                self.exchange = ccxt.binance({
                    "apiKey": self.api_key,
                    "secret": self.api_secret,
                    "enableRateLimit": True,
                })
                # Test connection
                self.exchange.fetch_ticker("BTC/USDT")
                self._connected = True
                logger.info("Binance connection successful (CCXT)")
            except Exception as e:
                logger.error(f"Failed to initialize Binance via CCXT: {e}")
                self._connected = False
        else:
            # Fallback: use public endpoints only
            logger.info("Using Binance public endpoints (read-only)")
            self._connected = False

    def get_ticker(self, symbol: str) -> Optional[Dict]:
        """
        Fetch ticker data for a symbol (e.g., 'BTC/USDT').
        Returns: {bid, ask, mid, timestamp}
        """
        try:
            if HAVE_CCXT and self.exchange:
                ticker = self.exchange.fetch_ticker(symbol)
                return {
                    "exchange": "Binance",
                    "symbol": symbol,
                    "bid": ticker.get("bid"),
                    "ask": ticker.get("ask"),
                    "mid": (ticker.get("bid", 0) + ticker.get("ask", 0)) / 2,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            else:
                # Fallback: public REST endpoint
                import requests
                url = "https://api.binance.com/api/v3/ticker/24hr"
                params = {"symbol": symbol.replace("/", "")}
                resp = requests.get(url, params=params, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    bid = float(data.get("bidPrice", 0))
                    ask = float(data.get("askPrice", 0))
                    return {
                        "exchange": "Binance",
                        "symbol": symbol,
                        "bid": bid,
                        "ask": ask,
                        "mid": (bid + ask) / 2,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
        except Exception as e:
            logger.error(f"Failed to fetch Binance ticker for {symbol}: {e}")
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
                    "exchange": "Binance",
                    "symbol": symbol,
                    "bids": order_book.get("bids", []),
                    "asks": order_book.get("asks", []),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            else:
                # Fallback: public REST endpoint
                import requests
                url = "https://api.binance.com/api/v3/depth"
                params = {"symbol": symbol.replace("/", ""), "limit": depth}
                resp = requests.get(url, params=params, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    return {
                        "exchange": "Binance",
                        "symbol": symbol,
                        "bids": data.get("bids", []),
                        "asks": data.get("asks", []),
                        "timestamp": datetime.utcnow().isoformat(),
                    }
        except Exception as e:
            logger.error(f"Failed to fetch Binance order book for {symbol}: {e}")
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
                # How much sellers will provide at or below price
                quantity = sum(float(q) for p, q in ob.get("asks", []) if float(p) <= price)
            else:
                # How much buyers will provide at or above price
                quantity = sum(float(q) for p, q in ob.get("bids", []) if float(p) >= price)

            return quantity
        except Exception as e:
            logger.error(f"Failed to calculate liquidity: {e}")
            return 0.0

    def test_connection(self) -> bool:
        """Test if connection to Binance is working."""
        try:
            ticker = self.get_ticker("BTC/USDT")
            return ticker is not None
        except Exception as e:
            logger.error(f"Binance connection test failed: {e}")
            return False

    def place_order(self, symbol: str, side: str, quantity: float, price: Optional[float] = None) -> Optional[Dict]:
        """
        Place a buy or sell order on Binance.
        side: 'buy' or 'sell'
        price: if None, places market order; otherwise limit order
        
        Returns: order dict or None if fails
        Paper trading: This will be mocked to return simulated order
        """
        if not self._connected:
            logger.warning("Binance not connected. Cannot place live orders.")
            return None

        try:
            if HAVE_CCXT and self.exchange:
                if price is None:
                    order = self.exchange.create_market_order(symbol, side, quantity)
                else:
                    order = self.exchange.create_limit_order(symbol, side, quantity, price)
                return order
        except Exception as e:
            logger.error(f"Failed to place order on Binance: {e}")
        return None
