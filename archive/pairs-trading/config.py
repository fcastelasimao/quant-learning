"""
Configuration for the pairs trading system.

All strategy parameters live here. Edit this file to tune the strategy
before running the backtester or paper trader.
"""
from dataclasses import dataclass, field


@dataclass
class PairsConfig:
    # ── CANDIDATE PAIRS ───────────────────────────────────────────────────────
    # Each tuple is (base, quote) using Binance symbols.
    # The spread is: log(base_price) - hedge_ratio * log(quote_price)
    # When the spread is HIGH we short base / long quote (expect base to fall
    # relative to quote). When LOW we do the reverse.
    candidate_pairs: list = field(default_factory=lambda: [
        # === CONFIRMED KEEPERS (baseline) ===
        ("RENDERUSDT", "FETUSDT"),
        ("ONDOUSDT",   "POLYXUSDT"),

        # === NEW CANDIDATES ===
        # AI theme extended — TAO correct symbol
        ("TAOUSDT",    "FETUSDT"),
        ("TAOUSDT",    "RENDERUSDT"),
        # RWA theme extended
        ("ONDOUSDT",   "TRUUSDT"),   # TRU = TrueFi, RWA credit protocol
        ("POLYXUSDT",  "TRUUSDT"),
        # DePIN (decentralised physical infrastructure) — same investor narrative as AI/compute
        ("IOTAUSDT",   "HELIUMUSDT"),
        ("RENDERUSDT", "IOTAUSDT"),
        # Restaking — EigenLayer ecosystem
        ("EIGENUSDT",  "ETHFIUSDT"),
    ])

    # ── DATA ──────────────────────────────────────────────────────────────────
    candle_interval: str = "1h"
    # 2000 x 1h = ~83 days. More data → higher statistical power for cointegration.
    lookback_candles: int = 2000

    # ── COINTEGRATION FILTER ──────────────────────────────────────────────────
    # Engle-Granger p-value must be below this to consider a pair tradeable.
    # 0.10 is less strict than the academic 0.05 but acceptable for exploratory trading.
    coint_pvalue_threshold: float = 0.10
    # Half-life of mean reversion must be in this range (hours).
    # Too short = too noisy. Too long = capital tied up forever.
    min_half_life_hours: float = 4.0
    max_half_life_hours: float = 120.0   # 5 days

    # ── SIGNAL THRESHOLDS ─────────────────────────────────────────────────────
    entry_z_score: float = 1.6       # Enter trade when |z-score| crosses this
    exit_z_score: float = 0.3        # Exit when |z-score| falls back inside this
    stop_loss_z_score: float = 3.0   # Hard stop if spread keeps widening

    # Rolling window for z-score normalisation (in candles).
    # 120 candles = 5 days of context — better estimate of the long-run mean.
    zscore_window: int = 120

    # ── POSITION SIZING ───────────────────────────────────────────────────────
    initial_bankroll: float = 1000.0   # USD
    # Fraction of bankroll allocated to the base leg of each trade.
    position_size_pct: float = 0.80

    # ── RISK MANAGEMENT ───────────────────────────────────────────────────────
    max_holding_hours: int = 120       # Force-exit after 5 days regardless of signal
    max_open_positions: int = 2        # Only trade 2 pairs simultaneously

    # ── EXECUTION COSTS ───────────────────────────────────────────────────────
    commission: float = 0.001          # 0.10% taker fee per leg (Binance)

    # ── BACKTEST SPLIT ────────────────────────────────────────────────────────
    # Fraction of historical data used to fit the hedge ratio (in-sample).
    # The remaining fraction is used to evaluate signals out-of-sample.
    in_sample_fraction: float = 0.60

    # ── PAPER TRADING ─────────────────────────────────────────────────────────
    scan_interval_seconds: float = 300.0   # Check for signals every 5 minutes
    log_path: str = "logs/paper_trades.jsonl"
    snapshot_path: str = "logs/paper_snapshots.jsonl"
    data_cache_path: str = "data/"


STRATEGY = PairsConfig()
