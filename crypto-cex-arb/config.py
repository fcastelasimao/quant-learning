"""
Configuration for Cross-CEX Arbitrage Scanner (Crypto Edition).

Paper Trading: Simulates trades without risking real capital.
Live Trading: Actually places orders (use with caution).
"""
from dataclasses import dataclass, field
import os


# ──────────────────────────────────────────────
# BITSTAMP CREDENTIALS
# Get API key: https://www.bitstamp.net/account/security/api/
# ──────────────────────────────────────────────
BITSTAMP_API_KEY = os.environ.get("BITSTAMP_API_KEY", "YOUR_BITSTAMP_API_KEY")
BITSTAMP_API_SECRET = os.environ.get("BITSTAMP_API_SECRET", "YOUR_BITSTAMP_API_SECRET")
BITSTAMP_BASE_URL = "https://www.bitstamp.net"
BITSTAMP_WS_URL = "wss://ws.bitstamp.net"

# ──────────────────────────────────────────────
# KRAKEN CREDENTIALS
# Get API key: https://www.kraken.com/en-us/settings/api
# Create new key, enable Trading & Funds queries
# ──────────────────────────────────────────────
KRAKEN_API_KEY = os.environ.get("KRAKEN_API_KEY", "YOUR_KRAKEN_API_KEY")
KRAKEN_API_SECRET = os.environ.get("KRAKEN_API_SECRET", "YOUR_KRAKEN_API_SECRET")
KRAKEN_BASE_URL = "https://api.kraken.com"
KRAKEN_WS_URL = "wss://ws.kraken.com/v2"

# ──────────────────────────────────────────────
# BINANCE CREDENTIALS
# Get API key: https://www.binance.com/en/my/settings/api-management
# ──────────────────────────────────────────────
BINANCE_API_KEY = os.environ.get("BINANCE_API_KEY", "YOUR_BINANCE_API_KEY")
BINANCE_API_SECRET = os.environ.get("BINANCE_API_SECRET", "YOUR_BINANCE_API_SECRET")
BINANCE_BASE_URL = "https://api.binance.com"

# ──────────────────────────────────────────────
# STRATEGY PARAMETERS
# ──────────────────────────────────────────────

@dataclass
class StrategyConfig:
    quote_currency: str = "USD"

    # EDGE DETECTION
    # Minimum implied edge (%) to trigger a trade.
    # These are NET edge thresholds after commissions and before rebalancing costs.
    paper_min_edge_pct: float = 0.20
    live_min_edge_pct: float = 0.40

    # Exchange commissions (as decimal, not percent)
    # The simulator assumes aggressive execution against top-of-book,
    # so default to taker-style fees.
    bitstamp_commission: float = 0.0030
    kraken_commission: float = 0.0040
    binance_commission: float = 0.0010

    # POSITION SIZING
    initial_bankroll: float = 1000.0
    max_stake_per_trade: float = 20.0
    min_stake_per_trade: float = 1.0

    kelly_fraction: float = 0.25        # Quarter-Kelly for safety
    max_open_positions: int = 5         # Lower for crypto (faster resolution)

    # RISK MANAGEMENT
    max_position_pct: float = 0.08      # Never risk >8% of bankroll per trade
    daily_loss_limit_pct: float = 0.20  # Stop for the day if down 20%
    max_drawdown_pct: float = 0.40      # Hard kill switch at 40% drawdown

    # SCANNING & EXECUTION
    scan_interval_seconds: float = 1.0  # Slightly slower default reduces CPU/network load
    order_timeout_seconds: float = 5.0  # Cancel if not filled in 5s
    order_book_depth: int = 1           # We only consume top-of-book in the simulator
    dashboard_update_interval_scans: int = 10
    snapshot_log_interval_scans: int = 10
    market_data_stale_after_seconds: float = 15.0
    ws_reconnect_delay_seconds: float = 5.0

    # TRADING PAIRS & MARKETS
    # Pairs chosen for USD cross-CEX overlap across Binance, Kraken, and Bitstamp.
    trading_pairs: list = field(default_factory=lambda: [
        "BTC/USD",
        "ETH/USD",
        "XRP/USD",
        "LTC/USD",
        "LINK/USD",
        "BCH/USD",
        "ADA/USD",
        "SOL/USD",
        "DOGE/USD",
        "DOT/USD",
        "MATIC/USD",
    ])

    # Minimum executable top-of-book notional required on both legs.
    min_liquidity_quote: float = 1000.0

    # PAPER TRADING vs LIVE TRADING
    paper_trading: bool = True          # Set False to go live
    paper_trading_duration_hours: int = 24

    # MODE: 'simulation' (synthetic data), 'paper' (real data, no execution), 'live' (real execution)
    mode: str = "simulation"            # Start with simulation, then paper, then live

    # LOG SETTINGS
    log_level: str = "INFO"
    trades_log_path: str = "data/cex_trades.jsonl"
    snapshots_path: str = "data/cex_snapshots.jsonl"
    dashboard_recent_arbs: int = 5

    # NOTIFICATION / ALERTS (optional)
    alert_on_trade: bool = True
    min_profit_for_alert: float = 0.50  # Alert if profit > £0.50

    @property
    def min_edge_pct(self) -> float:
        """Active minimum edge threshold based on running mode."""
        if self.mode == "live":
            return self.live_min_edge_pct
        return self.paper_min_edge_pct


STRATEGY = StrategyConfig()


# ──────────────────────────────────────────────
# EXCHANGE SETTINGS
# ──────────────────────────────────────────────
BITSTAMP_CONFIG = {
    "name": "Bitstamp",
    "api_key": BITSTAMP_API_KEY,
    "api_secret": BITSTAMP_API_SECRET,
    "base_url": BITSTAMP_BASE_URL,
    "commission": STRATEGY.bitstamp_commission,
}

KRAKEN_CONFIG = {
    "name": "Kraken",
    "api_key": KRAKEN_API_KEY,
    "api_secret": KRAKEN_API_SECRET,
    "base_url": KRAKEN_BASE_URL,
    "commission": STRATEGY.kraken_commission,
}

BINANCE_CONFIG = {
    "name": "Binance",
    "api_key": BINANCE_API_KEY,
    "api_secret": BINANCE_API_SECRET,
    "base_url": BINANCE_BASE_URL,
    "commission": STRATEGY.binance_commission,
}

EXCHANGE_CONFIGS = {
    "bitstamp": BITSTAMP_CONFIG,
    "kraken": KRAKEN_CONFIG,
    "binance": BINANCE_CONFIG,
}
