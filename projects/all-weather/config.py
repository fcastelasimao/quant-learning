"""
config.py — Single source of truth for all parameters.
"""

from datetime import datetime, date
import json
import os
import numpy as np

# ---- Core parameters ----

INITIAL_PORTFOLIO_VALUE = 10_000
BACKTEST_START = "2006-01-01"
BACKTEST_END   = date.today().strftime("%Y-%m-%d")
OOS_START      = "2018-01-01"

RUN_MODE = "oos_evaluate"
# Options: backtest, optimise, walk_forward, pareto, oos_evaluate, full_backtest

RUN_TAG = "monthly_2018oos" #run tag

PRICING_MODEL = "total_return"
REBALANCE_THRESHOLD = 0.05
HOLDINGS_FILE = "portfolio_holdings.json"

DATA_FREQUENCY       = "ME"    # "ME" = monthly, "W" = weekly
SHARPE_ANNUALISATION = 12      # must match: 12 for ME, 52 for W

RISK_FREE_RATE = 0.035  # US Fed funds ~3.5-3.75% as of March 2026

BENCHMARK_TICKER = "SPY"

# ---- Rolling RP ----
RP_LOOKBACK_YEARS = 5.0    # covariance estimation window
RP_RECOMPUTE_FREQ = "QS"   # "QS" = quarterly, "MS" = monthly, "YS" = yearly

# ---- Costs ----
TRANSACTION_COST_PCT = 0.001   # 0.001 = 0.1% per trade
TAX_DRAG_PCT         = 0.0   # 0.0 for ISA/SIPP

# ---- Target allocation ----
# Load from strategies.json. Override by setting DEFAULT_STRATEGY.
DEFAULT_STRATEGY = "6asset_tip_gsg_rpavg"

def _load_default_allocation() -> dict[str, float]:
    base_path = os.path.dirname(__file__)
    strategies_path = os.path.join(base_path, "strategies.json")
    example_path = os.path.join(base_path, "strategies.example.json")

    # Check if the private file exists; if not, use the example
    if not os.path.exists(strategies_path):
        if os.path.exists(example_path):
            strategies_path = example_path
        else:
            raise FileNotFoundError("Neither strategies.json nor strategies.example.json found.")

    with open(strategies_path) as f:
        data = json.load(f)
    
    return data["strategies"][DEFAULT_STRATEGY]["allocation"]

TARGET_ALLOCATION = _load_default_allocation()


assert abs(sum(TARGET_ALLOCATION.values()) - 1.0) < 1e-6, \
    f"TARGET_ALLOCATION must sum to 1.0, got {sum(TARGET_ALLOCATION.values()):.4f}"

# ---- Optimiser ----

OPT_METHOD     = "random"
OPT_MIN_WEIGHT = 0.05
OPT_MAX_WEIGHT = 0.25
OPT_MIN_CAGR   = 0.0
OPT_N_TRIALS   = 10_000
OPT_RANDOM_SEED = 42

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

ASSET_BOUNDS = {
    "SPY": (0.05, 0.20),
    "QQQ": (0.05, 0.20),
    "TLT": (0.20, 0.45),
    "TIP": (0.05, 0.20),
    "GLD": (0.05, 0.20),
    "GSG": (0.05, 0.15),
    "VNQ": (0.05, 0.20),
    "IWD": (0.05, 0.15),
    "DJP": (0.05, 0.15),
    "IEF": (0.10, 0.25),
}

RP_MIN_WEIGHT = 0.02  # minimum per-asset weight in risk parity optimisation

# ---- Pareto ----

PARETO_CAGR_RANGE = np.arange(4.0, 14.0, 1.0)

# ---- Asset momentum overlays (parked — default off) ----

ASSET_OVERLAYS = {
    "SPY": {"enabled": False, "threshold": -0.10, "d_window": 5, "reduce_pct": 1.0},
    "TLT": {"enabled": False, "threshold": -0.10, "d_window": 5, "reduce_pct": 1.0},
}

OVERLAY_CASH_RETURN = 0.0  # annual return on cash when overlay exits (0 = no interest)

# ---- Plot lines ----

PLOT_LINES = {
    "buy_and_hold": True,
    "spy":          True,
    "sixty_forty":  True,
}

# ---- Walk-forward ----

WF_OPT_METHOD  = "calmar"
WF_TRAIN_YEARS = 5
WF_TEST_YEARS  = 3
WF_STEP_YEARS  = 3

# ---- Run label (auto-generated) ----

def _build_run_label(price_start: str, price_end: str) -> str:
    _method_abbr = {
        "differential_evolution": "de", "sharpe_slsqp": "sharpe",
        "calmar": "calmar", "random": "rnd",
    }
    _freq_abbr = {"ME": "M", "W": "W"}

    n     = len(TARGET_ALLOCATION)
    freq  = _freq_abbr.get(DATA_FREQUENCY, DATA_FREQUENCY)
    start = price_start[:4]
    end   = price_end[:4]

    active_ov = sorted(t for t, v in ASSET_OVERLAYS.items() if v["enabled"])
    ov_suffix = ("_ov_" + "_".join(active_ov)) if active_ov else ""
    tag_suffix = f"_{RUN_TAG}" if RUN_TAG else ""

    def _label(base: str) -> str:
        return f"{base}{ov_suffix}{tag_suffix}"

    if RUN_MODE == "backtest":
        return _label(f"backtest_{n}assets_{freq}_{start}_{end}")
    if RUN_MODE == "oos_evaluate":
        return _label(f"oos_evaluate_{n}assets_{freq}_{start}_{end}")
    if RUN_MODE == "full_backtest":
        return _label(f"full_backtest_{n}assets_{freq}_{start}_{end}")
    if RUN_MODE == "optimise":
        method = _method_abbr.get(OPT_METHOD, OPT_METHOD)
        min_w  = int(round(OPT_MIN_WEIGHT * 100))
        max_w  = int(round(OPT_MAX_WEIGHT * 100))
        return _label(f"opt_{n}assets_{method}_w{min_w}_{max_w}_{freq}_{start}_{end}")
    if RUN_MODE == "walk_forward":
        method = _method_abbr.get(WF_OPT_METHOD, WF_OPT_METHOD)
        return _label(f"wf_{n}assets_{method}"
                      f"_tr{WF_TRAIN_YEARS}y_te{WF_TEST_YEARS}y"
                      f"_{freq}_{start}_{end}")
    if RUN_MODE == "pareto":
        return _label(f"pareto_{n}assets_{freq}_{start}_{end}")
    return _label(f"run_{n}assets_{freq}_{start}_{end}")

# ---- Validation ----

def validate_config() -> None:
    assert OPT_MIN_WEIGHT >= 0.0
    assert OPT_MAX_WEIGHT <= 1.0
    assert OPT_MIN_WEIGHT < OPT_MAX_WEIGHT
    assert OPT_N_TRIALS > 0
    assert 0.0 <= REBALANCE_THRESHOLD <= 1.0
    assert INITIAL_PORTFOLIO_VALUE > 0
    assert datetime.strptime(BACKTEST_START, "%Y-%m-%d") < \
           datetime.strptime(OOS_START,      "%Y-%m-%d") < \
           datetime.strptime(BACKTEST_END,   "%Y-%m-%d"), \
        "BACKTEST_START < OOS_START < BACKTEST_END must hold"
    assert RUN_MODE in (
        "backtest", "optimise", "walk_forward",
        "pareto", "oos_evaluate", "full_backtest"
    ), f"Unknown RUN_MODE: '{RUN_MODE}'"
    assert OPT_METHOD in ("random", "calmar", "sharpe_slsqp", "martin")
    assert PRICING_MODEL in ("total_return", "price_return")
    assert 0.0 <= RISK_FREE_RATE <= 0.20
    for ticker, ov in ASSET_OVERLAYS.items():
        assert -0.50 <= ov["threshold"] <= 0.50 and ov["threshold"] != 0.0
        assert 1 <= ov["d_window"] <= 60
        assert 0.0 < ov["reduce_pct"] <= 1.0
    assert 0.0 <= TRANSACTION_COST_PCT <= 0.05
    assert 0.0 <= TAX_DRAG_PCT <= 0.30
    assert (DATA_FREQUENCY == "ME" and SHARPE_ANNUALISATION == 12) or \
           (DATA_FREQUENCY == "W"  and SHARPE_ANNUALISATION == 52), \
        f"DATA_FREQUENCY/SHARPE_ANNUALISATION mismatch: {DATA_FREQUENCY}/{SHARPE_ANNUALISATION}"
    if RUN_MODE == "walk_forward":
        assert WF_OPT_METHOD in ("random", "calmar", "sharpe_slsqp")
        assert WF_TRAIN_YEARS > 0 and WF_TEST_YEARS > 0 and WF_STEP_YEARS > 0
        assert WF_TRAIN_YEARS + WF_TEST_YEARS <= \
            (datetime.strptime(OOS_START, "%Y-%m-%d") -
             datetime.strptime(BACKTEST_START, "%Y-%m-%d")).days / 365.25
        assert WF_STEP_YEARS >= WF_TEST_YEARS

# ---- Strategy loader ----

def load_strategy(strategy_id: str) -> dict:
    strategies_path = os.path.join(os.path.dirname(__file__), "strategies.json")
    with open(strategies_path, "r") as f:
        data = json.load(f)
    strategies = data["strategies"]
    if strategy_id not in strategies:
        raise KeyError(f"Strategy '{strategy_id}' not found. Available: {list(strategies.keys())}")
    s = strategies[strategy_id]
    return {
        "allocation":             s["allocation"],
        "asset_class_groups":     s.get("asset_class_groups", {}),
        "asset_class_max_weight": s.get("asset_class_max_weight", {}),
    }