"""
Data models used across the system.
Adapted for crypto CEX arbitrage (Bitstamp vs Kraken).
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional
import json
import uuid


class Exchange(str, Enum):
    BITSTAMP = "bitstamp"
    KRAKEN = "kraken"
    BINANCE = "binance"


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class TradeStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass
class PriceSnapshot:
    """Price snapshot for a single trading pair on one exchange."""
    exchange: Exchange
    pair: str          # e.g., "BTC/GBP"
    bid: Optional[float] = None    # Best bid price
    ask: Optional[float] = None    # Best ask price
    bid_volume: Optional[float] = None
    ask_volume: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def mid_price(self) -> Optional[float]:
        if self.bid and self.ask:
            return (self.bid + self.ask) / 2
        return None

    @property
    def spread_pct(self) -> Optional[float]:
        if self.bid and self.ask:
            return ((self.ask - self.bid) / self.bid) * 100
        return None

    @property
    def bid_notional(self) -> float:
        if self.bid and self.bid_volume:
            return self.bid * self.bid_volume
        return 0.0

    @property
    def ask_notional(self) -> float:
        if self.ask and self.ask_volume:
            return self.ask * self.ask_volume
        return 0.0


@dataclass
class ArbOpportunity:
    """A detected cross-CEX arbitrage opportunity."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    pair: str = ""          # e.g., "BTC/GBP"

    # The trade: BUY on one exchange, SELL on the other
    buy_exchange: Exchange = Exchange.BITSTAMP
    buy_price: float = 0.0
    buy_volume_available: float = 0.0

    sell_exchange: Exchange = Exchange.KRAKEN
    sell_price: float = 0.0
    sell_volume_available: float = 0.0

    # Calculated edge (after commissions)
    gross_edge_pct: float = 0.0
    net_edge_pct: float = 0.0

    # Recommended position size
    buy_quantity: float = 0.0
    sell_quantity: float = 0.0
    guaranteed_profit_quote: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


@dataclass
class PaperTrade:
    """A simulated cross-CEX arbitrage trade."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    opportunity_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    pair: str = ""              # e.g., "BTC/GBP"

    # BUY on one exchange
    buy_exchange: Exchange = Exchange.BITSTAMP
    buy_price: float = 0.0
    buy_quantity: float = 0.0

    # SELL on another exchange
    sell_exchange: Exchange = Exchange.KRAKEN
    sell_price: float = 0.0
    sell_quantity: float = 0.0

    expected_profit_quote: float = 0.0
    status: TradeStatus = TradeStatus.OPEN

    # Filled when resolved
    actual_profit_quote: float = 0.0
    resolved_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        if self.resolved_at:
            d["resolved_at"] = self.resolved_at.isoformat()
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class PortfolioState:
    """Current state of the paper trading portfolio."""
    bankroll: float = 1000.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    open_positions: list = field(default_factory=list)  # list of PaperTrade
    closed_positions: list = field(default_factory=list)
    peak_bankroll: float = 1000.0
    max_drawdown: float = 0.0
    stopped_for_day: bool = False      # True if daily loss limit hit

    def update_drawdown(self):
        if self.bankroll > self.peak_bankroll:
            self.peak_bankroll = self.bankroll
        dd = (self.peak_bankroll - self.bankroll) / self.peak_bankroll
        if dd > self.max_drawdown:
            self.max_drawdown = dd
