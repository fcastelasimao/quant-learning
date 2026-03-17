"""
config.py
=========
Single source of truth for all user parameters.

Every other module imports what it needs from here.
No other module should define configuration values -- they all
read from this file. This means you only ever change one file
to modify the behaviour of the entire project.
"""

from datetime import datetime
import numpy as np

# ===========================================================================
# CORE PARAMETERS
# ===========================================================================

INITIAL_PORTFOLIO_VALUE = 10_000    # Starting portfolio value in USD

BACKTEST_START = "2006-01-01"       # format: YYYY-MM-DD
BACKTEST_END   = "2026-01-01"       # format: YYYY-MM-DD

REBALANCE_THRESHOLD = 0.05          # Rebalance if any asset drifts > this
                                    # fraction from its target weight

RUN_LABEL = "original_allweather"   # Used to name the results folder.
                                    # Change before each run to keep results
                                    # organised. Examples:
                                    #   "original_allweather"
                                    #   "no_commodities"
                                    #   "optimised_calmar"

BENCHMARK_TICKER = "SPY"            # S&P 500 benchmark for comparison
HOLDINGS_FILE    = "portfolio_holdings.json"

# ===========================================================================
# TARGET ALLOCATION
# ===========================================================================
# Weights must sum to 1.0 -- the assert below will catch mistakes immediately.
#
# ETF inception dates (earliest usable backtest start):
#   VTI  -- May 2001
#   TLT  -- July 2002
#   IEF  -- July 2002
#   GLD  -- November 2004
#   DJP  -- February 2006  <-- limits backtest to ~2006-06-01
#   LQD  -- July 2002
#   QQQ  -- March 1999

TARGET_ALLOCATION = {
    "SPY":  0.083,  # US Large Cap stocks
    "QQQ":  0.055,  # US Tech stocks
    "TLT":  0.316,  # Long-Term Government Bonds
    "LQD":  0.108,  # Investment Grade Corporate Bonds
    "GLD":  0.295,  # Gold
    "DJP":  0.143,  # Commodities
}

assert abs(sum(TARGET_ALLOCATION.values()) - 1.0) < 1e-6, \
    f"TARGET_ALLOCATION weights must sum to 1.0, got {sum(TARGET_ALLOCATION.values()):.4f}"

# ===========================================================================
# OPTIMISER PARAMETERS
# ===========================================================================

RUN_OPTIMISER = False               # Set to True to run the optimiser

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

OPT_METHOD     = "calmar"
OPT_MIN_WEIGHT = 0.05               # minimum weight per asset (0.0 to 1.0)
OPT_MAX_WEIGHT = 0.60               # maximum weight per asset (0.0 to 1.0)
OPT_MIN_CAGR   = 0.0                # minimum acceptable CAGR in percent
OPT_N_TRIALS   = 2000               # random/calmar trials -- higher = better, slower
OPT_RANDOM_SEED = 42                # set to None for different results each run

# ===========================================================================
# PARETO FRONTIER
# ===========================================================================

RUN_PARETO        = False
PARETO_CAGR_RANGE = np.arange(2.0, 12.0, 1.0)  # CAGR targets to sweep (%)

# ===========================================================================
# WALK-FORWARD VALIDATION
# ===========================================================================

RUN_WALK_FORWARD = False
WF_TRAIN_YEARS   = 8    # years used to optimise weights
WF_TEST_YEARS    = 4    # years used to evaluate out-of-sample
WF_STEP_YEARS    = 4    # how far to slide the window each step


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
           datetime.strptime(BACKTEST_END,   "%Y-%m-%d"), \
        f"BACKTEST_START ({BACKTEST_START}) must be before BACKTEST_END ({BACKTEST_END})"
    assert WF_TRAIN_YEARS > 0 and WF_TEST_YEARS > 0 and WF_STEP_YEARS > 0, \
        "Walk-forward year parameters must all be positive"
    assert WF_TRAIN_YEARS + WF_TEST_YEARS <= \
           (datetime.strptime(BACKTEST_END,   "%Y-%m-%d") -
            datetime.strptime(BACKTEST_START, "%Y-%m-%d")).days / 365.25, \
        "WF_TRAIN_YEARS + WF_TEST_YEARS exceeds the total backtest period"
    assert OPT_METHOD in ("random", "calmar", "differential_evolution", "sharpe_slsqp"), \
        f"Unknown OPT_METHOD: '{OPT_METHOD}'. " \
        f"Choose from: random, calmar, differential_evolution, sharpe_slsqp"
    print("Config validation passed.\n")
