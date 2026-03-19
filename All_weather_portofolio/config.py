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

BACKTEST_START = "2004-01-01"       # format: YYYY-MM-DD
BACKTEST_END   = "2026-01-01"       # format: YYYY-MM-DD

REBALANCE_THRESHOLD = 0.05          # Rebalance if any asset drifts > this
                                    # fraction from its target weight

DATA_FREQUENCY        = "ME"    # "ME" for monthly, "W" for weekly
                                # monthly recommended for live rebalancing
                                # weekly gives more granular backtest data
SHARPE_ANNUALISATION  = 12      # periods per year for Sharpe annualisation
                                # 12 for monthly ("ME"), 52 for weekly ("W")

BENCHMARK_TICKER = "SPY"            # S&P 500 benchmark for comparison

HOLDINGS_FILE    = "portfolio_holdings.json"

#RUN_MODE = "optimise"
#RUN_MODE = "walk_forward"
#RUN_MODE = "pareto"             
RUN_MODE = "backtest_only"

# ===========================================================================
# TARGET ALLOCATION
# ===========================================================================
# Weights must sum to 1.0 -- the assert below will catch mistakes immediately.
# ETF inception dates (earliest usable backtest start):
#   VTI  -- May 2001            DJP  -- February 2006
#   TLT  -- July 2002           LQD  -- July 2002
#   IEF  -- July 2002           QQQ  -- March 1999
#   GLD  -- November 2004

TARGET_ALLOCATION = {
    "SPY":  0.142,  # US Large Cap stocks
    "QQQ":  0.143,  # US Tech stocks
    "TLT":  0.143,    # Long-Term Government Bonds
    "TIP":  0.143,  # Inflation-Protected Bonds
    "GLD":  0.143,  # Gold
    "SHY":  0.143,    # Short-Term Government Bonds (cash proxy)
    "IWD":  0.143,    # US Small Cap stocks§
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
OPT_MAX_WEIGHT = 0.30               # maximum weight per asset (0.0 to 1.0)
OPT_MIN_CAGR   = 0.0                # minimum acceptable CAGR in percent
OPT_N_TRIALS   = 10_000             # random/calmar trials -- higher = better, slower
OPT_RANDOM_SEED = 42                # set to None for different results each run

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

def _build_run_label() -> str:
    """Return a descriptive folder-name label derived from the current parameters."""
    _method_abbr = {
        "differential_evolution": "de",
        "sharpe_slsqp":           "sharpe",
        "calmar":                 "calmar",
        "random":                 "rnd",
    }
    _freq_abbr = {"ME": "M", "W": "W"}

    n       = len(TARGET_ALLOCATION)
    tickers = "-".join(TARGET_ALLOCATION.keys())
    freq    = _freq_abbr.get(DATA_FREQUENCY, DATA_FREQUENCY)
    start   = BACKTEST_START[:4]
    end     = BACKTEST_END[:4]

    if RUN_MODE == "backtest_only":
        return f"bt_{n}assets_{tickers}_{freq}_{start}_{end}"

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


RUN_LABEL = _build_run_label()

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
    assert RUN_MODE in ("optimise", "walk_forward", "pareto", "backtest_only"), \
    f"Unknown RUN_MODE: '{RUN_MODE}'. " \
    f"Choose from: optimise, walk_forward, pareto, backtest_only"
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
           (datetime.strptime(BACKTEST_END,   "%Y-%m-%d") -
            datetime.strptime(BACKTEST_START, "%Y-%m-%d")).days / 365.25, \
        "WF_TRAIN_YEARS + WF_TEST_YEARS exceeds the total backtest period"
