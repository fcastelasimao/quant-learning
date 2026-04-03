from __future__ import annotations

from pathlib import Path


START_DATE = "2010-01-01"
INITIAL_CAPITAL = 10_000.0
TRANSACTION_COST_PCT = 0.001
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.0

UNIVERSE_PRESET = "us_research_core"

UNIVERSE_PRESETS = {
    "us_research_core": {
        "description": "Liquid US ETF research universe with long daily histories.",
        "assets": {
            "GLOBAL_EQ": {"ticker": "VT", "bucket": "equity", "strategy_asset": True},
            "US_LC": {"ticker": "SPY", "bucket": "equity", "strategy_asset": True},
            "US_TECH": {"ticker": "QQQ", "bucket": "equity", "strategy_asset": True},
            "EUROPE": {"ticker": "VGK", "bucket": "equity", "strategy_asset": True},
            "EM": {"ticker": "VWO", "bucket": "equity", "strategy_asset": True},
            "LONG_BOND": {"ticker": "TLT", "bucket": "rates", "strategy_asset": True},
            "INT_BOND": {"ticker": "IEF", "bucket": "rates", "strategy_asset": True},
            "INFL_LINKED": {"ticker": "TIP", "bucket": "rates", "strategy_asset": True},
            "GOLD": {"ticker": "GLD", "bucket": "real_assets", "strategy_asset": True},
            "COMMODITIES": {"ticker": "DBC", "bucket": "real_assets", "strategy_asset": True},
            "ENERGY": {"ticker": "XLE", "bucket": "real_assets", "strategy_asset": True},
            "CASH": {"ticker": "SHY", "bucket": "cash", "strategy_asset": False},
        },
        "benchmarks": {
            "Global_Equity": {"GLOBAL_EQ": 1.0},
            "60_40_Proxy": {"US_LC": 0.60, "INT_BOND": 0.40},
            "Equal_Weight_Basket": {
                "GLOBAL_EQ": 0.10,
                "US_LC": 0.10,
                "US_TECH": 0.10,
                "EUROPE": 0.10,
                "EM": 0.10,
                "LONG_BOND": 0.10,
                "INT_BOND": 0.10,
                "INFL_LINKED": 0.10,
                "GOLD": 0.10,
                "COMMODITIES": 0.10,
            },
            "Cash": {"CASH": 1.0},
        },
    },
    "uk_ucits_candidate": {
        "description": "Candidate UK-listed UCITS mapping for later live implementation.",
        "assets": {
            "GLOBAL_EQ": {"ticker": "VWRL.L", "bucket": "equity", "strategy_asset": True},
            "US_LC": {"ticker": "VUSA.L", "bucket": "equity", "strategy_asset": True},
            "US_TECH": {"ticker": "EQQQ.L", "bucket": "equity", "strategy_asset": True},
            "EUROPE": {"ticker": "VEUR.L", "bucket": "equity", "strategy_asset": True},
            "EM": {"ticker": "EMIM.L", "bucket": "equity", "strategy_asset": True},
            "LONG_BOND": {"ticker": "IDTL.L", "bucket": "rates", "strategy_asset": True},
            "INT_BOND": {"ticker": "AGGH.L", "bucket": "rates", "strategy_asset": True},
            "INFL_LINKED": {"ticker": "INXG.L", "bucket": "rates", "strategy_asset": True},
            "GOLD": {"ticker": "SGLN.L", "bucket": "real_assets", "strategy_asset": True},
            "COMMODITIES": {"ticker": "CMOP.L", "bucket": "real_assets", "strategy_asset": True},
            "ENERGY": {"ticker": "IUES.L", "bucket": "real_assets", "strategy_asset": True},
            "CASH": {"ticker": "CSH2.L", "bucket": "cash", "strategy_asset": False},
        },
        "benchmarks": {
            "Global_Equity": {"GLOBAL_EQ": 1.0},
            "60_40_Proxy": {"US_LC": 0.60, "INT_BOND": 0.40},
            "Equal_Weight_Basket": {
                "GLOBAL_EQ": 0.10,
                "US_LC": 0.10,
                "US_TECH": 0.10,
                "EUROPE": 0.10,
                "EM": 0.10,
                "LONG_BOND": 0.10,
                "INT_BOND": 0.10,
                "INFL_LINKED": 0.10,
                "GOLD": 0.10,
                "COMMODITIES": 0.10,
            },
            "Cash": {"CASH": 1.0},
        },
    },
}

ACTIVE_PRESET = UNIVERSE_PRESETS[UNIVERSE_PRESET]
ACTIVE_UNIVERSE = ACTIVE_PRESET["assets"]
ASSET_TO_TICKER = {asset: meta["ticker"] for asset, meta in ACTIVE_UNIVERSE.items()}
ASSET_BUCKETS = {asset: meta["bucket"] for asset, meta in ACTIVE_UNIVERSE.items()}
STRATEGY_ASSETS = [
    asset for asset, meta in ACTIVE_UNIVERSE.items() if meta.get("strategy_asset", True)
]
BENCHMARKS = ACTIVE_PRESET["benchmarks"]

# Promoted baseline after the latest research pass:
# longer trend windows, weekly rebalance, moderate clustering,
# and no all-weather-style balancing logic.
ABSOLUTE_MOMENTUM_WINDOW = 168
RELATIVE_MOMENTUM_WINDOWS = (84, 168, 252)
RELATIVE_MOMENTUM_WEIGHTS = (0.50, 0.30, 0.20)
VOL_WINDOW = 84

MAX_ASSETS = 4
MAX_WEIGHT = 0.30
MAX_PER_BUCKET = 3
TARGET_VOL = 0.10
REBALANCE_FREQUENCY = 5
NO_TRADE_BAND = 0.02
PARKING_ASSET = "INT_BOND"

# Default baseline has no defense overlay. We keep the old version for testing.
DEFENSE_BREADTH_THRESHOLDS = ((0.0, 1.0),)
ALTERNATE_DEFENSE_BREADTH_THRESHOLDS = (
    (0.60, 1.00),
    (0.40, 0.75),
    (0.20, 0.50),
    (0.00, 0.25),
)

# ── HMM Regime Detection ──────────────────────────────────────────────────
HMM_N_STATES = 3
HMM_CHANNEL_WINDOW = 60
HMM_N_RESTARTS = 30
HMM_MIN_DURATION = 5
HMM_FIXED_DOF = True
HMM_INIT_DOF = 4.0
HMM_FIT_WINDOW = 504  # ~2 years of trading days used to fit the HMM

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RUNS_DIR = RESULTS_DIR / "runs"
CACHE_FILE = DATA_DIR / f"{UNIVERSE_PRESET}_prices.csv"
VIX_CACHE_FILE = DATA_DIR / "vix_prices.csv"

WALKFORWARD_WINDOWS = (
    ("2013_2015", "2013-01-01", "2015-12-31"),
    ("2016_2018", "2016-01-01", "2018-12-31"),
    ("2019_2021", "2019-01-01", "2021-12-31"),
    ("2022_2024", "2022-01-01", "2024-12-31"),
    ("2025_present", "2025-01-01", None),
)
