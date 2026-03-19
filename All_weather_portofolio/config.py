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
OOS_START      = "2020-01-01"       # IS ends here; OOS begins here

REBALANCE_THRESHOLD = 0.05          # Rebalance if any asset drifts > this
                                    # fraction from its target weight

DATA_FREQUENCY        = "ME"    # "ME" for monthly, "W" for weekly
                                # monthly recommended for live rebalancing
                                # weekly gives more granular backtest data
SHARPE_ANNUALISATION  = 12      # periods per year for Sharpe annualisation
                                # 12 for monthly ("ME"), 52 for weekly ("W")

BENCHMARK_TICKER = "SPY"            # S&P 500 benchmark for comparison

HOLDINGS_FILE    = "portfolio_holdings.json"

#RUN_MODE = "optimise"          # BACKTEST_START to OOS_START
#RUN_MODE = "walk_forward"      # BACKTEST_START to OOS_START
#RUN_MODE = "pareto"            # BACKTEST_START to OOS_START
#RUN_MODE = "oos_evaluate"      # OOS_START to BACKTEST_END  
RUN_MODE = "full_backtest"     # BACKTEST_START to BACKTEST_END
#RUN_MODE = "backtest"           # BACKTEST_START to OOS_START 

# ===========================================================================
# TARGET ALLOCATION
# ===========================================================================
# Weights must sum to 1.0 -- the assert below will catch mistakes immediately.
# ETF inception dates (earliest usable backtest start):
#   SPY  -- January 1993        QQQ  -- March 1999
#   TLT  -- July 2002           IEF  -- July 2002
#   SHY  -- July 2002           GLD  -- November 2004
#   IWD  -- January 2000        GSG  -- July 2006  <-- limits start to 2006

TARGET_ALLOCATION = {
    "SPY": 0.10,    # Stocks -- broad US equity
    "QQQ": 0.15,    # Stocks -- US tech
    "IWD": 0.10,    # Stocks -- US value equity
    "TLT": 0.25,    # Long bonds -- long-term government
    "IEF": 0.10,    # Intermediate bonds -- 7-10yr government
    "SHY": 0.05,    # Intermediate bonds -- short-term anchor
    "GLD": 0.15,    # Gold
    "GSG": 0.10,    # Commodities -- broad basket
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
    "stocks":             ["SPY", "QQQ", "IWD"],
    "long_bonds":         ["TLT"],
    "intermediate_bonds": ["IEF", "SHY"],
    "gold":               ["GLD"],
    "commodities":        ["GSG"],
}

ASSET_CLASS_MAX_WEIGHT = {
    "stocks":             0.40,   # total stocks cannot exceed 40%
    "long_bonds":         0.40,   # total long bonds cannot exceed 40%
    "intermediate_bonds": 0.25,   # total intermediate bonds cannot exceed 25%
    "gold":               0.20,   # gold cannot exceed 20%
    "commodities":        0.20,   # commodities cannot exceed 20%
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
    assert OPT_METHOD in ("random", "calmar", "differential_evolution", "sharpe_slsqp"), \
        f"Unknown OPT_METHOD: '{OPT_METHOD}'. " \
        f"Choose from: random, calmar, differential_evolution, sharpe_slsqp"
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
