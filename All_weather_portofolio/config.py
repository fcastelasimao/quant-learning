"""
config.py
=========
Single source of truth for all user parameters.

Every other module imports what it needs from here.
No other module should define configuration values -- they all
read from this file. This means you only ever change one file
to modify the behaviour of the entire project.
"""

from datetime import datetime, date
import json
import os
import numpy as np

# ===========================================================================
# CORE PARAMETERS
# ===========================================================================

INITIAL_PORTFOLIO_VALUE = 10_000    # Starting portfolio value in USD

BACKTEST_START = "2006-01-01"       # format: YYYY-MM-DD
BACKTEST_END   = date.today().strftime("%Y-%m-%d")   # always today
OOS_START      = "2020-01-01"       # IS ends here; OOS begins here
                                    
#RUN_MODE = "optimise"          # BACKTEST_START to OOS_START
#RUN_MODE = "walk_forward"      # BACKTEST_START to OOS_START
#RUN_MODE = "pareto"            # BACKTEST_START to OOS_START
#RUN_MODE = "oos_evaluate"      # OOS_START to BACKTEST_END  
RUN_MODE = "full_backtest"     # BACKTEST_START to BACKTEST_END
#RUN_MODE = "backtest"           # BACKTEST_START to OOS_START

PRICING_MODEL = "total_return"      # dividends and interest reinvested (realistic)
#PRICING_MODEL = "price_return"     # ignores dividends and interest (unrealistic)

REBALANCE_THRESHOLD = 0.05          # Rebalance if any asset drifts > this

HOLDINGS_FILE    = "portfolio_holdings.json"

DATA_FREQUENCY        = "ME"    # "ME" for monthly, "W" for weekly
                                # monthly recommended for live rebalancing
                                # weekly gives more granular backtest data
SHARPE_ANNUALISATION  = 12      # periods per year for Sharpe annualisation
                                # 12 for monthly ("ME"), 52 for weekly ("W")

RISK_FREE_RATE = 0.035          # Annual risk-free rate for Sharpe and Sortino.
                                # US Fed funds ~3.5-3.75% as of March 2026.
                                # Set to 0.0 to reproduce pre-fix baseline numbers.
                                # Update when rates change materially.

BENCHMARK_TICKER = "SPY"            # S&P 500 benchmark for comparison

# ===========================================================================
# COST MODELLING
# ===========================================================================
# Set both to 0.0 to reproduce the pre-cost baseline exactly.
#
# TRANSACTION_COST_PCT: applied to the value of each trade (buy or sell).
#   0.001 = 0.1% per trade. Conservative estimate for a UK retail investor
#   trading USD-denominated ETFs (bid-ask spread + FX conversion).
#   Modern brokers charge no commission on ETFs; the cost is the spread.
#
# TAX_DRAG_PCT: annual drag applied to the total portfolio value to model
#   capital gains tax on realised gains from monthly rebalancing.
#   0.0 for ISA/SIPP (tax-sheltered). 0.05 for a conservative taxable
#   account estimate (assumes ~20% CGT on ~25% of gains realised per year).
#   This is a simplification -- actual CGT depends on individual
#   circumstances, annual allowance usage, and marginal rate.

TRANSACTION_COST_PCT = 0.0      # 0.1% per trade (set 0.0 for no costs)
TAX_DRAG_PCT         = 0.0      # annual drag (set 0.0 for ISA/SIPP)

# Note: results with TRANSACTION_COST_PCT > 0 or TAX_DRAG_PCT > 0
# are NOT comparable to earlier runs where both were implicitly 0.

# ===========================================================================
# TARGET ALLOCATION
# ===========================================================================
# Weights must sum to 1.0 -- the assert below will catch mistakes immediately.
# ETF inception dates (earliest usable backtest start):
#   SPY  -- January 1993        QQQ  -- March 1999
#   TLT  -- July 2002           IEF  -- July 2002
#   SHY  -- July 2002           GLD  -- November 2004
#   IWD  -- January 2000        GSG  -- July 2006  <-- limits start to 2006

# Default: 6asset_tip_gsg (validated Tier 1 Growth).
# See strategies.json for all validated strategies.
TARGET_ALLOCATION = {
    "SPY": 0.15,
    "QQQ": 0.15,
    "TLT": 0.30,
    "TIP": 0.15,
    "GLD": 0.15,
    "GSG": 0.10,
}

assert abs(sum(TARGET_ALLOCATION.values()) - 1.0) < 1e-6, \
    f"TARGET_ALLOCATION weights must sum to 1.0, got {sum(TARGET_ALLOCATION.values()):.4f}"

# ===========================================================================
# OPTIMISER PARAMETERS
# ===========================================================================

# Which optimisation method to use. Options:
#
#   "random"
#       Pure random search. Tries OPT_N_TRIALS random weight combinations and
#       keeps the one with the best Calmar ratio that meets OPT_MIN_CAGR.
#       Simple but inefficient.
#
#   "calmar"
#       Same random search, maximises Calmar = CAGR / |max drawdown|.
#       Balances return and risk in one number. Recommended starting point.
#
#   "differential_evolution"
#       scipy's DE algorithm, maximises Calmar. Smarter than random search --
#       evolves a population of candidates rather than sampling blindly.
#       Does not need gradients so works well with drawdown-based objectives.
#
#   "sharpe_slsqp"                                                          
#       scipy's SLSQP gradient-based optimiser, maximises Sharpe ratio.
#       Fast and deterministic. Works because Sharpe is a smooth function.
#       Does NOT work for max drawdown (discontinuous -- near-zero gradients).

OPT_METHOD     = "differential_evolution"
OPT_MIN_WEIGHT = 0.05               # minimum weight per asset (0.0 to 1.0)
OPT_MAX_WEIGHT = 0.25               # maximum weight per asset (0.0 to 1.0)
OPT_MIN_CAGR   = 0.0                # minimum acceptable CAGR in percent
OPT_N_TRIALS   = 10_000             # random/calmar trials -- higher = better, slower
OPT_RANDOM_SEED = 42                # set to None for different results each run
#OPT_RANDOM_SEED = 123  
#OPT_RANDOM_SEED = 7     

ASSET_CLASS_GROUPS = {
    "stocks":             ["SPY", "QQQ"],
    "long_bonds":         ["TLT"],
    "intermediate_bonds": ["TIP"],
    "gold":               ["GLD"],
    "commodities":        ["GSG"],
}

ASSET_CLASS_MAX_WEIGHT = {
    "stocks":             0.40,
    "long_bonds":         0.40,
    "intermediate_bonds": 0.25,
    "gold":               0.25,
    "commodities":        0.20,
}

# Per-asset weight bounds for the optimiser.
# Takes priority over OPT_MIN_WEIGHT / OPT_MAX_WEIGHT when all tickers are covered.
# Allows asymmetric ranges -- e.g. TLT can reach 45% while stocks stay below 20%.
# Falls back to uniform [OPT_MIN_WEIGHT, OPT_MAX_WEIGHT] if this dict is absent
# or doesn't cover every ticker in the active allocation.
ASSET_BOUNDS = {
    "SPY": (0.05, 0.20),   # Broad US equity
    "QQQ": (0.05, 0.20),   # US tech
    "TLT": (0.20, 0.45),   # Long-term government bonds
    "TIP": (0.05, 0.20),   # Inflation-linked bonds
    "GLD": (0.05, 0.20),   # Gold
    "GSG": (0.05, 0.15),   # Broad commodities
    "VNQ": (0.05, 0.20),   # REITs (7asset_tip_gsg_vnq)
    "IWD": (0.05, 0.15),   # Value equity (7asset_tip_djp)
    "DJP": (0.05, 0.15),   # Diversified commodities (7asset_tip_djp)
    "IEF": (0.10, 0.25),   # Intermediate bonds (5asset_dalio)
}

# ===========================================================================
# PARETO FRONTIER
# ===========================================================================

PARETO_CAGR_RANGE = np.arange(4.0, 14.0, 1.0)  # CAGR targets to sweep (%)

# ===========================================================================
# WALK-FORWARD VALIDATION
# ===========================================================================

WF_TRAIN_YEARS   = 5    # years used to optimise weights
WF_TEST_YEARS    = 2    # years used to evaluate out-of-sample
WF_STEP_YEARS    = 2    # how far to slide the window each step
WF_OPT_METHOD = "differential_evolution"    # "calmar" for speed, "differential_evolution"
                             # for a more faithful test of the DE process
                             # Note: DE walk-forward takes ~3-4x longer

# ===========================================================================
# RUN LABEL  (auto-generated from parameters -- no need to edit manually)
# ===========================================================================

def _build_run_label(price_start: str, price_end: str) -> str:
    """
    Return a self-describing folder-name label for this run.

    Parameters
    ----------
    price_start : the actual start date used for this run (not always BACKTEST_START)
    price_end   : the actual end date used for this run   (not always BACKTEST_END)

    Both are derived in main.py from RUN_MODE + the three date constants so the
    label always matches what was actually run, not the global config dates.
    """
    _method_abbr = {
        "differential_evolution": "de",
        "sharpe_slsqp":           "sharpe",
        "calmar":                 "calmar",
        "random":                 "rnd",
    }
    _freq_abbr = {"ME": "M", "W": "W"}

    n     = len(TARGET_ALLOCATION)
    freq  = _freq_abbr.get(DATA_FREQUENCY, DATA_FREQUENCY)
    start = price_start[:4]
    end   = price_end[:4]

    if RUN_MODE == "backtest":
        return f"backtest_{n}assets_{freq}_{start}_{end}"

    if RUN_MODE == "oos_evaluate":
        return f"oos_evaluate_{n}assets_{freq}_{start}_{end}"

    if RUN_MODE == "full_backtest":
        return f"full_backtest_{n}assets_{freq}_{start}_{end}"

    if RUN_MODE == "optimise":
        method = _method_abbr.get(OPT_METHOD, OPT_METHOD)
        min_w  = int(round(OPT_MIN_WEIGHT * 100))
        max_w  = int(round(OPT_MAX_WEIGHT * 100))
        return f"opt_{n}assets_{method}_w{min_w}_{max_w}_{freq}_{start}_{end}"

    if RUN_MODE == "walk_forward":
        method = _method_abbr.get(WF_OPT_METHOD, WF_OPT_METHOD)
        return (f"wf_{n}assets_{method}"
                f"_tr{WF_TRAIN_YEARS}y_te{WF_TEST_YEARS}y"
                f"_{freq}_{start}_{end}")

    if RUN_MODE == "pareto":
        return f"pareto_{n}assets_{freq}_{start}_{end}"

    return f"run_{n}assets_{freq}_{start}_{end}"  # fallback

# ===========================================================================
# VALIDATION
# ===========================================================================

def validate_config():
    """
    Validate all user parameters at startup.
    Raises AssertionError immediately with a clear message if anything is wrong.
    This prevents silent bugs from bad configuration reaching the backtest.
    """
    assert OPT_MIN_WEIGHT >= 0.0, \
        "OPT_MIN_WEIGHT must be >= 0"
    assert OPT_MAX_WEIGHT <= 1.0, \
        "OPT_MAX_WEIGHT must be <= 1.0"
    assert OPT_MIN_WEIGHT < OPT_MAX_WEIGHT, \
        f"OPT_MIN_WEIGHT ({OPT_MIN_WEIGHT}) must be less than OPT_MAX_WEIGHT ({OPT_MAX_WEIGHT})"
    assert OPT_N_TRIALS > 0, \
        "OPT_N_TRIALS must be a positive integer"
    assert 0.0 <= REBALANCE_THRESHOLD <= 1.0, \
        "REBALANCE_THRESHOLD must be between 0 and 1"
    assert INITIAL_PORTFOLIO_VALUE > 0, \
        "INITIAL_PORTFOLIO_VALUE must be positive"
    assert datetime.strptime(BACKTEST_START, "%Y-%m-%d") < \
           datetime.strptime(OOS_START,      "%Y-%m-%d") < \
           datetime.strptime(BACKTEST_END,   "%Y-%m-%d"), \
        "BACKTEST_START < OOS_START < BACKTEST_END must hold"
    assert RUN_MODE in (
        "backtest", "optimise", "walk_forward",
        "pareto", "oos_evaluate", "full_backtest"
    ), (f"Unknown RUN_MODE: '{RUN_MODE}'. Choose from: backtest, optimise, "
        f"walk_forward, pareto, oos_evaluate, full_backtest")
    assert OPT_METHOD in ("random", "calmar", "differential_evolution", "sharpe_slsqp", "martin"), \
        f"Unknown OPT_METHOD: '{OPT_METHOD}'. " \
        f"Choose from: random, calmar, differential_evolution, sharpe_slsqp, martin"
    assert PRICING_MODEL in ("total_return", "price_return"), \
        f"PRICING_MODEL must be 'total_return' or 'price_return', got '{PRICING_MODEL}'"
    assert 0.0 <= RISK_FREE_RATE <= 0.20, \
        "RISK_FREE_RATE must be between 0 and 0.20"
    assert 0.0 <= TRANSACTION_COST_PCT <= 0.05, \
        "TRANSACTION_COST_PCT must be between 0 and 0.05 (5% is already extreme)"
    assert 0.0 <= TAX_DRAG_PCT <= 0.30, \
        "TAX_DRAG_PCT must be between 0 and 0.30"
    assert (DATA_FREQUENCY == "ME" and SHARPE_ANNUALISATION == 12) or \
       (DATA_FREQUENCY == "W"  and SHARPE_ANNUALISATION == 52), \
    f"DATA_FREQUENCY '{DATA_FREQUENCY}' and " \
    f"SHARPE_ANNUALISATION {SHARPE_ANNUALISATION} are mismatched. " \
    f"Use ME/12 for monthly or W/52 for weekly."
    if RUN_MODE == "walk_forward":
        assert WF_OPT_METHOD in ("random", "calmar", "differential_evolution", "sharpe_slsqp"), \
            f"Unknown WF_OPT_METHOD: '{WF_OPT_METHOD}'. " \
                f"Choose from: random, calmar, differential_evolution, sharpe_slsqp"
        assert WF_TRAIN_YEARS > 0 and WF_TEST_YEARS > 0 and WF_STEP_YEARS > 0, \
        "Walk-forward year parameters must all be positive"
        assert WF_TRAIN_YEARS + WF_TEST_YEARS <= \
           (datetime.strptime(OOS_START,      "%Y-%m-%d") -
            datetime.strptime(BACKTEST_START, "%Y-%m-%d")).days / 365.25, \
        "WF_TRAIN_YEARS + WF_TEST_YEARS exceeds the in-sample period (BACKTEST_START to OOS_START)"
        assert WF_STEP_YEARS >= WF_TEST_YEARS, \
        (f"WF_STEP_YEARS ({WF_STEP_YEARS}) must be >= WF_TEST_YEARS ({WF_TEST_YEARS}) "
         f"to avoid overlapping test windows (correlated results)")


# ===========================================================================
# STRATEGY LOADER
# ===========================================================================

def load_strategy(strategy_id: str) -> dict:
    """
    Load a strategy by ID from strategies.json and return its allocation,
    asset_class_groups, and asset_class_max_weight dicts.

    Parameters
    ----------
    strategy_id : str
        The key in strategies.json, e.g. "6asset_tip_gsg".

    Returns
    -------
    dict with keys:
        "allocation"            : {ticker: weight, ...}
        "asset_class_groups"    : {class: [tickers], ...}
        "asset_class_max_weight": {class: max_weight, ...}

    Raises
    ------
    FileNotFoundError  if strategies.json cannot be found
    KeyError           if strategy_id does not exist in the file
    """
    strategies_path = os.path.join(os.path.dirname(__file__), "strategies.json")
    with open(strategies_path, "r") as f:
        data = json.load(f)

    strategies = data["strategies"]
    if strategy_id not in strategies:
        available = list(strategies.keys())
        raise KeyError(
            f"Strategy '{strategy_id}' not found in strategies.json. "
            f"Available: {available}"
        )

    s = strategies[strategy_id]
    return {
        "allocation":             s["allocation"],
        "asset_class_groups":     s.get("asset_class_groups", {}),
        "asset_class_max_weight": s.get("asset_class_max_weight", {}),
    }
