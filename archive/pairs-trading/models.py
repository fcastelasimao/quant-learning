"""
Data models for the pairs trading system.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class Signal(str, Enum):
    LONG_SPREAD  = "long_spread"   # buy base, sell quote  (spread too low, expect reversion up)
    SHORT_SPREAD = "short_spread"  # sell base, buy quote   (spread too high, expect reversion down)
    EXIT         = "exit"
    NONE         = "none"


@dataclass
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class CointegrationResult:
    """Output of the cointegration test for a candidate pair."""
    base: str                   # e.g. "ETH/USD"
    quote: str                  # e.g. "BTC/USD"
    is_cointegrated: bool
    p_value: float              # Engle-Granger p-value (lower = stronger evidence)
    hedge_ratio: float          # beta from OLS: log(base) = alpha + beta * log(quote)
    intercept: float
    half_life_hours: float      # how quickly the spread mean-reverts
    spread_mean: float
    spread_std: float


@dataclass
class SpreadState:
    """Snapshot of the spread at a single point in time."""
    timestamp: datetime
    price_base: float
    price_quote: float
    spread: float               # log(base) - hedge_ratio * log(quote) - intercept
    z_score: float              # (spread - rolling_mean) / rolling_std
    hedge_ratio: float
    signal: Signal


@dataclass
class PairPosition:
    """An open (or closed) two-leg paper trade."""
    base: str                   # e.g. "ETH/USD"
    quote: str                  # e.g. "BTC/USD"
    direction: Signal           # LONG_SPREAD or SHORT_SPREAD
    entry_z_score: float
    hedge_ratio: float

    # Quantities (always positive; direction determines long/short)
    base_qty: float             # units of base asset traded
    quote_qty: float            # units of quote asset traded

    entry_price_base: float
    entry_price_quote: float
    entry_time: datetime

    # Filled on close
    exit_price_base: Optional[float] = None
    exit_price_quote: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None  # "signal", "stop_loss", "timeout"
    pnl_usd: Optional[float] = None


@dataclass
class BacktestTrade:
    """Completed trade record from the backtester."""
    base: str
    quote: str
    direction: str
    entry_time: datetime
    exit_time: datetime
    holding_hours: float
    entry_z: float
    exit_z: float
    pnl_pct: float
    exit_reason: str


@dataclass
class BacktestResult:
    """Aggregate performance metrics from a backtest run."""
    base: str
    quote: str
    candle_interval: str
    in_sample_candles: int
    out_of_sample_candles: int
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    total_pnl_pct: float
    avg_pnl_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    avg_holding_hours: float
    trades: list[BacktestTrade] = field(default_factory=list)
