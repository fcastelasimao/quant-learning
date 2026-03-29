"""
Ray Dalio All Weather Portfolio Tracker
========================================
Tracks your portfolio, gives monthly rebalancing instructions,
backtests performance vs two benchmarks:
  1. S&P 500 (SPY)         -- "what if I just bought the market?"
  2. Buy & Hold All Weather -- "what if I set up Dalio's allocation and did nothing?"

Includes four optimisation methods selectable via OPT_METHOD:
  "random"                 -- random search, minimise max drawdown
  "calmar"                 -- random search, maximise Calmar ratio (CAGR / |max drawdown|)
  "differential_evolution" -- scipy DE, maximise Calmar ratio (smarter than random)
  "sharpe_slsqp"           -- gradient-based, maximise Sharpe ratio (fast, deterministic)

Also includes a Pareto frontier plot showing the full CAGR vs drawdown tradeoff curve.


ENVIRONMENT REQUIREMENTS
=============================================================================
Recommended: create a dedicated conda environment before running this script.

   conda create -n allweather python=3.12
   conda activate allweather
   pip install yfinance pandas matplotlib scipy

Minimum versions required:
   python  >= 3.10
   pandas  >= 2.2
"""

# ===========================================================================
# IMPORTS
# ===========================================================================

import json
import os
from typing import Optional
from dataclasses import dataclass
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize, differential_evolution

warnings.filterwarnings("ignore")   # suppress yfinance / pandas deprecation noise

# ===========================================================================
# USER PARAMETERS -- CHANGE THESE
# ===========================================================================

initial_portfolio_value = 10_000    # Starting portfolio value in USD

backtest_start = "2006-01-01"       # format: YYYY-MM-DD
backtest_end   = "2026-01-01"       # format: YYYY-MM-DD

rebalance_threshold = 0.05          # Rebalance if any asset drifts > this % from target

# Label for this run -- used to name the results folder.
run_label = "original_allweather"

# ===========================================================================
# TARGET ALLOCATION
# ===========================================================================

target_allocation = {
    "SPY":  0.083,  # US Large Cap stocks
    "QQQ":  0.055,  # US Tech stocks
    "TLT":  0.316,  # Long-Term Government Bonds
    "LQD":  0.108,  # Investment Grade Corporate Bonds
    "GLD":  0.295,  # Gold
    "DJP":  0.143,  # Commodities
}
assert abs(sum(target_allocation.values()) - 1.0) < 1e-6, \
    f"Weights must sum to 1.0, got {sum(target_allocation.values()):.4f}"

def validate_config():
    """Validate all user parameters at startup to catch mistakes early."""
    assert OPT_MIN_WEIGHT >= 0.0, \
        "OPT_MIN_WEIGHT must be >= 0"
    assert OPT_MAX_WEIGHT <= 1.0, \
        "OPT_MAX_WEIGHT must be <= 1.0"
    assert OPT_MIN_WEIGHT < OPT_MAX_WEIGHT, \
        f"OPT_MIN_WEIGHT ({OPT_MIN_WEIGHT}) must be less than OPT_MAX_WEIGHT ({OPT_MAX_WEIGHT})"
    assert OPT_N_TRIALS > 0, \
        "OPT_N_TRIALS must be a positive integer"
    assert 0.0 <= rebalance_threshold <= 1.0, \
        "rebalance_threshold must be between 0 and 1"
    assert initial_portfolio_value > 0, \
        "initial_portfolio_value must be positive"
    assert datetime.strptime(backtest_start, "%Y-%m-%d") < \
           datetime.strptime(backtest_end,   "%Y-%m-%d"), \
        f"backtest_start ({backtest_start}) must be before backtest_end ({backtest_end})"
    assert WF_TRAIN_YEARS > 0 and WF_TEST_YEARS > 0 and WF_STEP_YEARS > 0, \
        "Walk-forward year parameters must all be positive"
    assert WF_TRAIN_YEARS + WF_TEST_YEARS <= \
           (datetime.strptime(backtest_end, "%Y-%m-%d") -
            datetime.strptime(backtest_start, "%Y-%m-%d")).days / 365.25, \
        "WF_TRAIN_YEARS + WF_TEST_YEARS exceeds the total backtest period"
    assert OPT_METHOD in ("random", "calmar", "differential_evolution", "sharpe_slsqp"), \
        f"Unknown OPT_METHOD: '{OPT_METHOD}'"
    print("Config validation passed.\n")

benchmark_ticker = "SPY"
holdings_file    = "portfolio_holdings.json"

# ===========================================================================
# OPTIMISER PARAMETERS -- only used when RUN_OPTIMISER = True
# ===========================================================================

RUN_OPTIMISER = True               # Set to True to run the optimiser

# Which optimisation method to use. Options:
#
#   "random"
#       Pure random search. Tries OPT_N_TRIALS random weight combinations and
#       keeps the one with the smallest max drawdown that meets OPT_MIN_CAGR.
#       Simple but inefficient -- most trials are wasted on bad regions.
#
#   "calmar"
#       Same random search but maximises the Calmar ratio instead of minimising
#       drawdown alone. Calmar = CAGR / |max drawdown|. This balances return
#       and risk in one number without needing to choose a weighting parameter.
#       A Calmar of 0.5 means you earn 0.5% of return per 1% of drawdown accepted.
#
#   "differential_evolution"
#       Uses scipy's Differential Evolution algorithm to maximise Calmar ratio.
#       Starts with a population of random allocations and iteratively combines
#       the best ones to produce better candidates. Much more efficient than
#       random search -- finds better results in fewer evaluations because it
#       learns which regions of the weight space are promising.
#       Does not require gradients so works well with max drawdown objectives.
#
#   "sharpe_slsqp"
#       Uses scipy's SLSQP gradient-based optimiser to maximise Sharpe ratio.
#       Sharpe = mean return / volatility (annualised). Unlike max drawdown,
#       Sharpe is a smooth differentiable function so gradient-based methods
#       work correctly here. Fast and deterministic (always finds the same answer).
#       Trades: optimises volatility not drawdown, so the result may have a
#       smaller Sharpe but larger single-event crash than "calmar" or "random".

OPT_METHOD     = "calmar"           # see options above

OPT_MIN_WEIGHT = 0.01               # minimum allowed weight per asset
OPT_MAX_WEIGHT = 0.99               # maximum allowed weight per asset
OPT_MIN_CAGR   = 0.0                # minimum acceptable CAGR (used by random + calmar)
OPT_N_TRIALS   = 2000               # random/calmar only -- higher = better but slower
OPT_RANDOM_SEED = None                # set to None for different results each run

# Pareto frontier: sweep across a range of minimum CAGR constraints and plot
# the resulting drawdown for each. Shows the full risk-return tradeoff curve
# so you can pick the point that matches your personal risk tolerance.
RUN_PARETO     = True                           # Set to True to run the Pareto frontier analysis
PARETO_CAGR_RANGE = np.arange(2.0, 12.0, 1.0)   # CAGR targets to sweep (%)

# Walk-forward validation: splits the backtest period into a training window
# and a test window. Optimises weights on the training window only, then
# evaluates those weights on the unseen test window. If performance holds up
# out-of-sample, the allocation has genuine robustness. If it collapses, the
# optimiser was overfitting to the training period.
RUN_WALK_FORWARD    = True          # Set to True to run walk-forward validation
WF_TRAIN_YEARS      = 8             # years of data used to optimise weights
WF_TEST_YEARS       = 4             # years of unseen data used to evaluate
                                    # The two windows slide forward together:
                                    # window 1: train 2006-2014, test 2014-2018
                                    # window 2: train 2010-2018, test 2018-2022
                                    # etc. until the end of the data is reached.
WF_STEP_YEARS       = 4             # how many years to slide forward each step

# ===========================================================================
# STATS DATACLASS
# ===========================================================================

@dataclass
class StrategyStats:
    """
    Holds performance statistics for a single strategy.

    Using a dataclass instead of a plain dict eliminates the trailing-space
    key hack that was needed before to avoid key collisions between strategies.
    Each strategy now has its own clean StrategyStats object with named fields.
    """
    name:         str
    cagr:         float
    max_drawdown: float
    sharpe:       float
    calmar:       float     # CAGR / |max drawdown| -- the balanced risk/return metric
    final_value:  float
    period_years: float


# ===========================================================================
# RESULTS VERSIONING
# ===========================================================================

def make_results_dir(label: str) -> str:
    """
    Create a timestamped results folder inside 'results/'.
    Format: results/YYYY-MM-DD_HH-MM-SS_<label>/
    Every run gets its own folder so old results are never overwritten.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder    = os.path.join("results", f"{timestamp}_{label}")
    os.makedirs(folder, exist_ok=True)
    return folder


def build_log_row(results_dir: str,
                  stats_list: list[StrategyStats],
                  weights: dict,
                  label: str) -> dict:
    """
    Build a single summary row for the master log from a list of StrategyStats.
    Separated from the file-writing logic so each function does one job.
    """
    row = {
        "Timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Label":          label,
        "Backtest Start": backtest_start,
        "Backtest End":   backtest_end,
        "Tickers":        " | ".join(f"{t}={w:.1%}" for t, w in weights.items()),
        "Results Folder": results_dir,
    }
    for s in stats_list:
        prefix = s.name.replace(" ", "_").replace("&", "and")
        row[f"{prefix}_CAGR (%)"]        = s.cagr
        row[f"{prefix}_Max_DD (%)"]      = s.max_drawdown
        row[f"{prefix}_Sharpe"]          = s.sharpe
        row[f"{prefix}_Calmar"]          = s.calmar
        row[f"{prefix}_Final_Value ($)"] = s.final_value
    return row


def append_to_master_log(results_dir: str,
                         stats_list: list[StrategyStats],
                         weights: dict,
                         label: str):
    """
    Append one summary row to results/master_log.csv.
    Each row represents one run, making it easy to compare allocations.
    """
    log_path = os.path.join("results", "master_log.csv")
    row      = build_log_row(results_dir, stats_list, weights, label)
    log_df   = pd.DataFrame([row])

    if os.path.exists(log_path):
        log_df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        os.makedirs("results", exist_ok=True)
        log_df.to_csv(log_path, mode="w", header=True, index=False)

    print(f"  Master log updated -> {log_path}")


# ===========================================================================
# DATA FETCHING
# ===========================================================================

def fetch_prices(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download adjusted closing prices for the given tickers
    between start_date and end_date.

    yf.download accepts date strings directly so no datetime conversion needed.
    """
    print(f"Fetching price data for: {', '.join(tickers)}")
    print(f"Period: {start_date} -> {end_date}\n")

    raw = yf.download(tickers, start=start_date, end=end_date,
                      progress=False, auto_adjust=True)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.dropna(how="all")
    prices = prices.ffill()
    # Q: does ffill make sense? How is this usually done?
    # A: Yes, ffill (forward-fill) is the standard approach for daily price data.
    #    Markets are closed on weekends and holidays, so those dates simply have
    #    no price. Forward-filling means "use the last known closing price until
    #    a new one is available", which correctly reflects the value of your
    #    holdings on a non-trading day. The alternative (dropping those rows)
    #    would create gaps that break monthly resampling. Backfilling (using the
    #    next available price) would be wrong because it introduces lookahead
    #    bias -- using future prices to fill past dates.

    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        print(f"WARNING: No data found for {missing}. Check ticker symbols.\n")

    return prices


# ===========================================================================
# PORTFOLIO STATE  (load / save JSON)
# ===========================================================================

def load_holdings() -> Optional[dict]:
    """Load current holdings from JSON file, or return None if not found."""
    if os.path.exists(holdings_file):
        with open(holdings_file) as f:
            return json.load(f)
    return None


def save_holdings(holdings: dict):
    """Persist holdings to JSON file."""
    with open(holdings_file, "w") as f:
        json.dump(holdings, f, indent=2)
    print(f"Holdings saved to {holdings_file}")


def initialise_holdings(prices_row: pd.Series,
                        allocation: dict,
                        portfolio_value: float) -> dict:
    """
    Create a fresh set of holdings by splitting portfolio_value
    according to allocation at the current prices.

    Returns a dict like:
        {"VTI": {"shares": 12.34, "last_price": 210.50}, ...}

    Q: Why work with shares instead of dollars?
    A: Because the dollar value of your holdings changes every day as prices
       move, but the number of shares you own stays fixed until you trade.
       Storing shares is the ground truth -- to get the current dollar value
       at any time you just multiply shares * current price. If we stored
       dollar amounts we'd have to update them every time prices changed,
       which would require fetching prices just to read the file.
    """
    holdings = {}
    for ticker, weight in allocation.items():
        dollar_amount    = portfolio_value * weight
        price            = float(prices_row[ticker])
        holdings[ticker] = {
            "shares":     round(dollar_amount / price, 6),
            "last_price": round(price, 4),
        }
    return holdings


# ===========================================================================
# REBALANCING LOGIC
# ===========================================================================

def current_weights(holdings: dict, prices_row: pd.Series):
    """Compute current portfolio weights and total value given live prices."""
    values = {t: h["shares"] * float(prices_row[t]) for t, h in holdings.items()}
    total  = sum(values.values())
    return {t: v / total for t, v in values.items()}, total


def rebalancing_instructions(holdings: dict,
                             prices_row: pd.Series,
                             allocation: dict,
                             threshold: float) -> tuple[pd.DataFrame, float]:
    """
    Compare current weights to targets and produce buy/sell instructions.
    Returns a DataFrame and the total portfolio value.
    """
    weights, total_value = current_weights(holdings, prices_row)

    rows = []
    for ticker, target in allocation.items():
        # Q: Why use weights.get here instead of weights[ticker]?
        # A: weights.get(ticker, 0.0) returns 0.0 if the ticker is not in the
        #    dict instead of raising a KeyError. This is a safety net for the
        #    edge case where a ticker exists in target_allocation but has no
        #    current holding (e.g. if you manually edited the JSON file and
        #    removed an entry). weights[ticker] would crash the program in that
        #    case. After the allocation-change detection we added to main, this
        #    case should never happen, but defensive code is still good practice.
        current = weights.get(ticker, 0.0)
        drift   = current - target
        dollar  = drift * total_value
        action  = "HOLD"
        if abs(drift) > threshold:
            action = "SELL" if drift > 0 else "BUY"

        rows.append({
            "Ticker":         ticker,
            "Current Weight": round(current * 100, 2),
            "Target Weight":  round(target  * 100, 2),
            "Drift (%)":      round(drift   * 100, 2),
            "Action":         action,
            "$ Amount":       round(abs(dollar), 2),
            "Current Price":  round(float(prices_row[ticker]), 2),
            "Current Shares": round(holdings[ticker]["shares"], 4),
        })

    return pd.DataFrame(rows), total_value


def apply_rebalance(holdings: dict,
                    prices_row: pd.Series,
                    allocation: dict,
                    portfolio_value: float) -> dict:
    """
    Reset each holding to its target dollar weight at the current prices.

    Q: Why use shares instead of dollar amounts here?
    A: Same reason as in initialise_holdings -- shares are the stable unit.
       When we rebalance, we calculate the target dollar amount for each asset
       (portfolio_value * weight), then divide by the current price to get
       how many shares that dollar amount buys. Those shares are what we store.
       Next month when prices have changed, we multiply shares * new price to
       get the new dollar value, which will have drifted from target and may
       trigger another rebalance.
    """
    for ticker, target in allocation.items():
        price            = float(prices_row[ticker])
        holdings[ticker] = {
            "shares":     round((portfolio_value * target) / price, 6),
            "last_price": round(price, 4),
        }
    return holdings


# ===========================================================================
# SHARED STAT HELPERS
# ===========================================================================

def compute_cagr(series: pd.Series, years: float) -> float:
    """Compound Annual Growth Rate as a percentage."""
    return ((series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1) * 100


def compute_max_drawdown(series: pd.Series) -> float:
    """Maximum peak-to-trough decline as a percentage (negative number)."""
    peak = series.cummax()
    return ((series - peak) / peak).min() * 100


def compute_sharpe(monthly_ret_series: pd.Series) -> float:
    """Annualised Sharpe ratio from a series of monthly returns (as percentages)."""
    r = monthly_ret_series.dropna() / 100
    return 0.0 if r.std() == 0 else (r.mean() / r.std()) * np.sqrt(12)


def compute_calmar(cagr: float, max_drawdown: float) -> float:
    """
    Calmar ratio = CAGR / |max drawdown|.

    Measures return per unit of drawdown accepted. Higher is better.
    A Calmar of 0.5 means you earn 0.5% of annual return for every 1%
    of maximum drawdown you accept. Maximising Calmar finds the allocation
    with the best balance between return and downside risk without requiring
    you to manually choose how to weight them against each other.

    Returns 0.0 if drawdown is zero (degenerate case).
    """
    if max_drawdown == 0.0:
        return 0.0
    return cagr / abs(max_drawdown)


# ===========================================================================
# BACK-TEST ENGINE
# ===========================================================================

def run_backtest(prices: pd.DataFrame,
                 benchmark_prices: pd.Series,
                 allocation: Optional[dict] = None,
                 portfolio_value: float = None) -> pd.DataFrame:
    """
    Simulate THREE strategies from the first available date:

      1. Rebalanced portfolio  -- uses `allocation` (defaults to target_allocation)
      2. Buy & Hold            -- same starting weights, never rebalanced
      3. S&P 500 buy & hold    -- everything in SPY, never touched

    The `allocation` parameter allows the optimiser to call this function
    with different weight combinations without touching target_allocation.
    """
    if allocation      is None: allocation      = target_allocation
    if portfolio_value is None: portfolio_value = initial_portfolio_value

    tickers = list(allocation.keys())

    # Q: What is ME in the resample method?
    # A: "ME" stands for Month End. It tells pandas to group all the daily
    #    prices within each calendar month and take the last value -- i.e.
    #    the closing price on the final trading day of that month. We use
    #    monthly prices because rebalancing happens monthly and it reduces
    #    noise compared to using daily data for a long-term strategy.
    #    Note: "ME" requires pandas >= 2.2. Older versions use "M".
    monthly = prices[tickers].resample("ME").last().dropna()
    bench   = benchmark_prices.resample("ME").last().dropna()

    # Q: What is happening in this line?
    # A: monthly.index and bench.index are two lists of dates. They may not
    #    perfectly overlap -- for example if one ETF has data starting in
    #    February but SPY starts in January, the January row only exists in
    #    one of them. .intersection() returns only the dates that appear in
    #    BOTH lists. We then filter both DataFrames to those common dates so
    #    they are perfectly aligned row-by-row. Without this, comparing
    #    portfolio value to benchmark value on the same row would be comparing
    #    different dates.
    common  = monthly.index.intersection(bench.index)
    monthly = monthly.loc[common]
    bench   = bench.loc[common]

    if monthly.empty:
        raise ValueError("No overlapping monthly data found. Check date range.")

    # Q: What is iloc?
    # A: iloc stands for "integer location" -- it selects rows by their
    #    position (0, 1, 2...) rather than by their label (a date, a name).
    #    monthly.iloc[0] means "give me the first row", regardless of what
    #    date that row has. The alternative, .loc, selects by label:
    #    monthly.loc["2006-01-31"] would give you the row for that specific
    #    date. iloc is useful when you want "first" or "last" without knowing
    #    the exact date.
    first_row    = monthly.iloc[0]
    bench_shares = portfolio_value / float(bench.iloc[0])

    aw_holdings = {t: (portfolio_value * w) / float(first_row[t])
                   for t, w in allocation.items()}
    bh_holdings = {t: (portfolio_value * w) / float(first_row[t])
                   for t, w in allocation.items()}

    records = []
    for date, row in monthly.iterrows():
        aw_value  = sum(sh * float(row[t]) for t, sh in aw_holdings.items())
        bh_value  = sum(sh * float(row[t]) for t, sh in bh_holdings.items())
        spy_value = bench_shares * float(bench.loc[date])

        bh_weights = {t: (bh_holdings[t] * float(row[t])) / bh_value
                      for t in tickers}

        record = {
            "Date":                   date,
            "All Weather Value":      round(aw_value, 2),
            "Buy & Hold All Weather": round(bh_value, 2),
            "S&P 500 Value":          round(spy_value, 2),
        }
        for t in tickers:
            record[f"B&H {t} Weight (%)"] = round(bh_weights[t] * 100, 1)

        records.append(record)

        for t, w in allocation.items():
            aw_holdings[t] = (aw_value * w) / float(row[t])

    df = pd.DataFrame(records).set_index("Date")
    for col in ["All Weather Value", "Buy & Hold All Weather", "S&P 500 Value"]:
        df[f"{col} Monthly Ret (%)"] = df[col].pct_change() * 100

    return df


# ===========================================================================
# STATISTICS
# ===========================================================================

def compute_stats(backtest: pd.DataFrame) -> list[StrategyStats]:
    """
    Compute key performance statistics for all three strategies.
    Returns a list of StrategyStats dataclasses -- one per strategy.
    """
    years = (backtest.index[-1] - backtest.index[0]).days / 365.25

    def make_stats(name: str, value_col: str, ret_col: str) -> StrategyStats:
        series = backtest[value_col]
        cagr   = round(compute_cagr(series, years), 2)
        mdd    = round(compute_max_drawdown(series), 2)
        return StrategyStats(
            name         = name,
            cagr         = cagr,
            max_drawdown = mdd,
            sharpe       = round(compute_sharpe(backtest[ret_col]), 3),
            calmar       = round(compute_calmar(cagr, mdd), 3),
            final_value  = round(series.iloc[-1], 2),
            period_years = round(years, 1),
        )

    return [
        make_stats("All Weather (Rebalanced)",
                   "All Weather Value",
                   "All Weather Value Monthly Ret (%)"),
        make_stats("Buy & Hold All Weather",
                   "Buy & Hold All Weather",
                   "Buy & Hold All Weather Monthly Ret (%)"),
        make_stats("S&P 500 Buy & Hold",
                   "S&P 500 Value",
                   "S&P 500 Value Monthly Ret (%)"),
    ]


# ===========================================================================
# OPTIMISER
# ===========================================================================

def _score_allocation(weights_array: np.ndarray,
                      tickers: list,
                      prices: pd.DataFrame,
                      benchmark_prices: pd.Series,
                      method: str,
                      min_cagr: float) -> float:
    """
    Score a weight array for use by any optimiser.
    Returns a value where LOWER is better (all optimisers minimise by convention).

    For "random" and "calmar" (random search):
        Returns -Calmar for calmar, -drawdown fraction for random.
    For "differential_evolution":
        Returns -Calmar (DE minimises, so we negate what we want to maximise).
    For "sharpe_slsqp":
        Returns -Sharpe (SLSQP minimises, so we negate).

    Shared by all methods to avoid duplicating backtest + stat logic.
    """
    allocation = dict(zip(tickers, weights_array))
    bt         = run_backtest(prices, benchmark_prices, allocation)
    series     = bt["All Weather Value"]
    years      = (bt.index[-1] - bt.index[0]).days / 365.25

    cagr = compute_cagr(series, years)
    mdd  = compute_max_drawdown(series)     # negative percentage

    # Enforce minimum CAGR as a large penalty rather than a hard constraint
    # (hard constraints are only supported by SLSQP, not DE)
    if cagr < min_cagr:
        return 1e6

    if method == "sharpe_slsqp":
        ret_col = "All Weather Value Monthly Ret (%)"
        sharpe  = compute_sharpe(bt[ret_col])
        return -sharpe                          # negate: minimise(-sharpe) = maximise(sharpe)

    # calmar, random, differential_evolution all use Calmar as the objective
    calmar = compute_calmar(cagr, mdd)
    return -calmar                              # negate: minimise(-calmar) = maximise(calmar)


def optimise_random(prices, benchmark_prices, allocation,
                    min_weight, max_weight, min_cagr, n_trials, method, random_seed) -> dict:
    """
    Random search optimiser shared by "random" and "calmar" methods.
    Tries n_trials random weight combinations and keeps the best score.
    """
    np.random.seed(random_seed)
    tickers       = list(allocation.keys())
    n             = len(tickers)
    best_score    = np.inf
    best_weights  = None

    for i in range(n_trials):
        raw     = np.random.uniform(min_weight, max_weight, n)
        weights = raw / raw.sum()

        if weights.max() > max_weight:
            continue

        score = _score_allocation(weights, tickers, prices,
                                  benchmark_prices, method, min_cagr)

        if score < best_score:
            best_score   = score
            best_weights = weights

        if (i + 1) % 500 == 0:
            display = -best_score if best_score < 1e5 else float("nan")
            metric  = "Calmar" if method == "calmar" else "Calmar"
            print(f"  Trial {i+1:>5}/{n_trials} | Best {metric} so far: {display:.3f}")

    return best_weights, best_score


def optimise_allocation(prices: pd.DataFrame,
                        benchmark_prices: pd.Series,
                        allocation: dict,
                        method: str,
                        min_weight: float,
                        max_weight: float,
                        min_cagr: float,
                        n_trials: int) -> dict:
    """
    Dispatch to the appropriate optimisation method and return optimised weights.

    All methods share _score_allocation() so the objective is consistent.
    The key difference is HOW they search the weight space:
      - random/calmar: blindly sample random combinations
      - differential_evolution: intelligently evolve a population of candidates
      - sharpe_slsqp: follow the gradient of the Sharpe ratio
    """
    tickers = list(allocation.keys())
    n       = len(tickers)
    bounds  = [(min_weight, max_weight)] * n

    print(f"\nRunning optimiser -- method: {method}")
    print(f"  Assets:        {', '.join(tickers)}")
    print(f"  Weight bounds: [{min_weight:.0%}, {max_weight:.0%}] per asset")
    print(f"  Min CAGR:      {min_cagr}%\n")

    # ------------------------------------------------------------------
    if method in ("random", "calmar"):
        objective_label = "Calmar ratio (CAGR / |max drawdown|)"
        print(f"  Objective:  maximise {objective_label}")
        print(f"  Trials:     {n_trials}\n")

        best_weights, best_score = optimise_random(
            prices, benchmark_prices, allocation,
            min_weight, max_weight, min_cagr, n_trials, method, OPT_RANDOM_SEED
        )

        if best_weights is None:
            print("WARNING: No valid allocation found. Try lowering OPT_MIN_CAGR.")
            return allocation

    # ------------------------------------------------------------------
    elif method == "differential_evolution":
        # DE maintains a population of candidate solutions and evolves them
        # over generations by combining the best candidates. It does not need
        # gradients and explores the search space much more efficiently than
        # random search. scipy's implementation handles the weight sum constraint
        # via the 'constraints' parameter.
        print(f"  Objective:  maximise Calmar ratio (CAGR / |max drawdown|)")
        print(f"  Algorithm:  Differential Evolution (scipy)\n")

        def de_objective(w):
            # Normalise weights to sum to 1 before scoring
            w_norm = w / w.sum()
            return _score_allocation(w_norm, tickers, prices,
                                     benchmark_prices, "calmar", min_cagr)

        result = differential_evolution(
            de_objective,
            bounds    = bounds,
            maxiter   = 200,
            popsize   = 10,
            tol       = 1e-6,
            seed      = 42,             # reproducible results
            disp      = False,
        )

        if not result.success and result.fun >= 1e5:
            print(f"WARNING: DE did not converge cleanly: {result.message}")

        raw_weights  = result.x
        best_weights = raw_weights / raw_weights.sum()   # re-normalise
        best_score   = result.fun

    # ------------------------------------------------------------------
    elif method == "sharpe_slsqp":
        # SLSQP (Sequential Least Squares Programming) is a gradient-based
        # method. It computes the gradient of the objective at each step and
        # follows it downhill. This works correctly for Sharpe ratio because
        # Sharpe is a smooth function (mean / std of returns) -- small weight
        # changes produce predictable, non-zero gradients.
        # It does NOT work for max drawdown because drawdown is discontinuous.
        print(f"  Objective:  maximise Sharpe ratio (return / volatility)")
        print(f"  Algorithm:  SLSQP gradient-based (scipy)\n")

        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        w0          = np.array(list(allocation.values()))

        result = minimize(
            lambda w: _score_allocation(w, tickers, prices,
                                        benchmark_prices, "sharpe_slsqp", min_cagr),
            w0,
            method      = "SLSQP",
            bounds      = bounds,
            constraints = constraints,
            options     = {"maxiter": 1000, "ftol": 1e-9},
        )

        if not result.success:
            print(f"WARNING: SLSQP did not converge: {result.message}")
            print("Returning original allocation unchanged.")
            return allocation

        best_weights = result.x
        best_score   = result.fun

    else:
        raise ValueError(f"Unknown OPT_METHOD: '{method}'. "
                         f"Choose from: random, calmar, differential_evolution, sharpe_slsqp")

    # ------------------------------------------------------------------
    # Print results
    optimised = dict(zip(tickers, best_weights))
    print("\nOptimised weights:")
    for t, w in optimised.items():
        original = allocation[t]
        change   = w - original
        sign     = "+" if change >= 0 else ""
        print(f"  {t:5s}  {w:.1%}   (was {original:.1%}, {sign}{change:.1%})")

    metric_label = "Sharpe" if method == "sharpe_slsqp" else "Calmar"
    print(f"\nBest {metric_label}: {-best_score:.3f}")

    return optimised


# ===========================================================================
# PARETO FRONTIER
# ===========================================================================

def run_walk_forward(prices: pd.DataFrame,
                     benchmark_prices: pd.Series,
                     allocation: dict,
                     train_years: int,
                     test_years: int,
                     step_years: int,
                     min_weight: float,
                     max_weight: float,
                     n_trials: int,
                     results_dir: str,
                     random_seed: Optional[int]):
    """
    Walk-forward validation to test whether optimised weights are genuine
    or just overfitted to the training data.

    How it works:
      1. Take the full price history and divide it into sliding windows.
         Each window has a training period and a test period.
      2. For each window, optimise weights using ONLY the training data.
      3. Apply those weights to the test period (which the optimiser never saw).
      4. Compare the out-of-sample test performance to both:
           a. The in-sample training performance (to detect overfitting)
           b. The original target_allocation on the same test period (to check
              whether optimising actually adds value vs just using Dalio's weights)

    What the results mean:
      - If test Calmar is close to train Calmar: low overfitting, robust allocation
      - If test Calmar collapses vs train Calmar: high overfitting, don't trust results
      - If optimised beats original on test: optimisation adds genuine value
      - If optimised matches or loses to original on test: overfitting, no real edge

    The plot shows all three series (optimised train, optimised test, original test)
    so you can visually see whether the out-of-sample performance holds up.
    """
    print(f"\nRunning walk-forward validation...")
    print(f"  Train window: {train_years} years")
    print(f"  Test window:  {test_years} years")
    print(f"  Step size:    {step_years} years\n")

    # Convert the price index to a list of dates for slicing
    all_dates  = prices.index
    start_date = all_dates[0]
    end_date   = all_dates[-1]

    train_delta = pd.DateOffset(years=train_years)
    test_delta  = pd.DateOffset(years=test_years)
    step_delta  = pd.DateOffset(years=step_years)

    windows      = []
    window_start = start_date

    # Build all windows
    while True:
        train_end = window_start + train_delta
        test_end  = train_end   + test_delta

        if test_end > end_date:
            break

        windows.append({
            "train_start": window_start,
            "train_end":   train_end,
            "test_start":  train_end,
            "test_end":    test_end,
        })
        window_start = window_start + step_delta

    if not windows:
        print("Not enough data for walk-forward validation. "
              "Try reducing WF_TRAIN_YEARS or WF_TEST_YEARS.")
        return

    print(f"  Found {len(windows)} windows:\n")
    for i, w in enumerate(windows):
        print(f"  Window {i+1}: "
              f"train {w['train_start'].strftime('%Y-%m')} -> "
              f"{w['train_end'].strftime('%Y-%m')}  |  "
              f"test  {w['test_start'].strftime('%Y-%m')} -> "
              f"{w['test_end'].strftime('%Y-%m')}")
    print()

    records = []

    for i, w in enumerate(windows):
        label = (f"Window {i+1} "
                 f"({w['train_start'].strftime('%Y-%m')} - "
                 f"{w['test_end'].strftime('%Y-%m')})")
        print(f"  --- {label} ---")

        # Slice training data
        train_mask        = (prices.index >= w["train_start"]) & \
                            (prices.index <  w["train_end"])
        bench_train_mask  = (benchmark_prices.index >= w["train_start"]) & \
                            (benchmark_prices.index <  w["train_end"])

        train_prices      = prices[train_mask]
        train_bench       = benchmark_prices[bench_train_mask]

        # Slice test data
        test_mask         = (prices.index >= w["test_start"]) & \
                            (prices.index <  w["test_end"])
        bench_test_mask   = (benchmark_prices.index >= w["test_start"]) & \
                            (benchmark_prices.index <  w["test_end"])

        test_prices       = prices[test_mask]
        test_bench        = benchmark_prices[bench_test_mask]

        if train_prices.empty or test_prices.empty:
            print(f"    Skipping -- insufficient data.")
            continue

        # Step 1: Optimise on training data only
        print(f"    Optimising on training data ({train_years} years)...")
        opt_weights, _ = optimise_random(
            train_prices, train_bench, allocation,
            min_weight, max_weight, 0.0, n_trials, "calmar", random_seed
        )

        if opt_weights is None:
            print(f"    Optimiser failed -- skipping window.")
            continue

        opt_allocation = dict(zip(list(allocation.keys()), opt_weights))

        # Step 2: Evaluate on training data (in-sample)
        bt_train    = run_backtest(train_prices, train_bench, opt_allocation)
        series_tr   = bt_train["All Weather Value"]
        years_tr    = (series_tr.index[-1] - series_tr.index[0]).days / 365.25
        train_cagr  = round(compute_cagr(series_tr, years_tr), 2)
        train_mdd   = round(compute_max_drawdown(series_tr), 2)
        train_calmar = round(compute_calmar(train_cagr, train_mdd), 3)

        # Step 3: Evaluate optimised weights on test data (out-of-sample)
        bt_test     = run_backtest(test_prices, test_bench, opt_allocation)
        series_te   = bt_test["All Weather Value"]
        years_te    = (series_te.index[-1] - series_te.index[0]).days / 365.25
        test_cagr   = round(compute_cagr(series_te, years_te), 2)
        test_mdd    = round(compute_max_drawdown(series_te), 2)
        test_calmar = round(compute_calmar(test_cagr, test_mdd), 3)

        # Step 4: Evaluate ORIGINAL allocation on test data (baseline)
        bt_orig      = run_backtest(test_prices, test_bench, allocation)
        series_or    = bt_orig["All Weather Value"]
        orig_cagr    = round(compute_cagr(series_or, years_te), 2)
        orig_mdd     = round(compute_max_drawdown(series_or), 2)
        orig_calmar  = round(compute_calmar(orig_cagr, orig_mdd), 3)

        # Overfitting score: how much Calmar dropped from train to test
        # Closer to 1.0 = no overfitting. Below 0.5 = significant overfitting.
        overfit_ratio = round(test_calmar / train_calmar, 3) if train_calmar != 0 else 0.0

        print(f"    In-sample  (train): CAGR={train_cagr:>6.2f}%  "
              f"MaxDD={train_mdd:>7.2f}%  Calmar={train_calmar:.3f}")
        print(f"    Out-sample (test):  CAGR={test_cagr:>6.2f}%  "
              f"MaxDD={test_mdd:>7.2f}%  Calmar={test_calmar:.3f}")
        print(f"    Original   (test):  CAGR={orig_cagr:>6.2f}%  "
              f"MaxDD={orig_mdd:>7.2f}%  Calmar={orig_calmar:.3f}")
        print(f"    Overfit ratio (test/train Calmar): {overfit_ratio:.3f}  "
              f"{'ok' if overfit_ratio >= 0.6 else 'WARNING: possible overfit'}\n")

        records.append({
            "Window":               i + 1,
            "Train Start":          w["train_start"].strftime("%Y-%m"),
            "Train End":            w["train_end"].strftime("%Y-%m"),
            "Test Start":           w["test_start"].strftime("%Y-%m"),
            "Test End":             w["test_end"].strftime("%Y-%m"),
            "Opt Weights":          " | ".join(f"{t}={v:.1%}"
                                               for t, v in opt_allocation.items()),
            "Train CAGR (%)":       train_cagr,
            "Train MaxDD (%)":      train_mdd,
            "Train Calmar":         train_calmar,
            "Test CAGR (%)":        test_cagr,
            "Test MaxDD (%)":       test_mdd,
            "Test Calmar":          test_calmar,
            "Original Test CAGR (%)":   orig_cagr,
            "Original Test MaxDD (%)":  orig_mdd,
            "Original Test Calmar":     orig_calmar,
            "Overfit Ratio":        overfit_ratio,
        })

    if not records:
        print("No valid windows completed.")
        return

    df = pd.DataFrame(records)

    # Summary statistics
    mean_overfit   = df["Overfit Ratio"].mean()
    mean_test_cal  = df["Test Calmar"].mean()
    mean_orig_cal  = df["Original Test Calmar"].mean()
    opt_beats_orig = (df["Test Calmar"] > df["Original Test Calmar"]).sum()

    print(f"\n  === WALK-FORWARD SUMMARY ===")
    print(f"  Windows completed:              {len(records)}")
    print(f"  Mean overfit ratio:             {mean_overfit:.3f}  "
          f"(1.0 = no overfit, <0.6 = concerning)")
    print(f"  Mean test Calmar (optimised):   {mean_test_cal:.3f}")
    print(f"  Mean test Calmar (original):    {mean_orig_cal:.3f}")
    print(f"  Windows where opt beat original: "
          f"{opt_beats_orig}/{len(records)}")
    if mean_overfit >= 0.7 and opt_beats_orig > len(records) / 2:
        print(f"\n  VERDICT: Allocation appears robust -- low overfitting, "
              f"optimisation adds value.")
    elif mean_overfit >= 0.5:
        print(f"\n  VERDICT: Moderate overfitting detected. Results should be "
              f"treated with caution.")
    else:
        print(f"\n  VERDICT: High overfitting detected. Do not trust the "
              f"optimised weights for live trading.")

    # Save CSV
    csv_path = os.path.join(results_dir, "walk_forward.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Walk-forward results saved -> {csv_path}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor("#0d1117")

    ax1, ax2 = axes
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#c9d1d9", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.label.set_color("#c9d1d9")
        ax.xaxis.label.set_color("#c9d1d9")
        ax.title.set_color("white")
        ax.grid(axis="y", color="#30363d", alpha=0.4)

    windows_x = df["Window"].astype(str) + "\n" + \
                df["Train Start"] + " - " + df["Test End"]

    # Left panel: Calmar comparison across windows
    x     = np.arange(len(df))
    width = 0.28

    ax1.bar(x - width, df["Train Calmar"],         width,
            color="#58a6ff", alpha=0.85, label="Optimised (in-sample train)")
    ax1.bar(x,          df["Test Calmar"],          width,
            color="#3fb950", alpha=0.85, label="Optimised (out-of-sample test)")
    ax1.bar(x + width,  df["Original Test Calmar"], width,
            color="#f0b429", alpha=0.85, label="Original allocation (test)")

    ax1.set_xticks(x)
    ax1.set_xticklabels(windows_x, fontsize=7)
    ax1.axhline(0, color="#8b949e", lw=0.8)
    ax1.set_ylabel("Calmar Ratio", fontsize=10)
    ax1.set_title("Calmar Ratio: Train vs Test vs Original\n"
                  "Green close to blue = low overfitting",
                  fontsize=11, pad=8)
    ax1.legend(fontsize=8, facecolor="#21262d", edgecolor="#30363d",
               labelcolor="white")

    # Right panel: overfit ratio per window
    colors = ["#3fb950" if r >= 0.6 else "#f85149" for r in df["Overfit Ratio"]]
    ax2.bar(x, df["Overfit Ratio"], color=colors, alpha=0.85)
    ax2.axhline(1.0, color="#8b949e", lw=1.0, linestyle="--",
                label="1.0 = perfect (no overfit)")
    ax2.axhline(0.6, color="#f0b429", lw=1.0, linestyle=":",
                label="0.6 = warning threshold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(windows_x, fontsize=7)
    ax2.set_ylabel("Overfit Ratio (test Calmar / train Calmar)", fontsize=10)
    ax2.set_title("Overfit Ratio per Window\n"
                  "Green = acceptable  |  Red = concerning",
                  fontsize=11, pad=8)
    ax2.legend(fontsize=8, facecolor="#21262d", edgecolor="#30363d",
               labelcolor="white")
    ax2.set_ylim(0, max(df["Overfit Ratio"].max() * 1.2, 1.2))

    plt.suptitle(
        f"Walk-Forward Validation  |  "
        f"Train={train_years}yr  Test={test_years}yr  Step={step_years}yr  |  "
        f"Mean overfit ratio: {mean_overfit:.3f}",
        fontsize=12, color="white", y=1.02
    )
    plt.tight_layout()

    plot_path = os.path.join(results_dir, "walk_forward.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  Walk-forward plot saved -> {plot_path}")
    plt.show()

def run_pareto_frontier(prices: pd.DataFrame,
                        benchmark_prices: pd.Series,
                        allocation: dict,
                        cagr_targets: np.ndarray,
                        min_weight: float,
                        max_weight: float,
                        n_trials: int,
                        results_dir: str,
                        random_seed: Optional[int]):
    """
    Sweep across a range of minimum CAGR constraints and for each one find
    the allocation that minimises max drawdown using random search.

    This maps out the efficient frontier -- the set of portfolios where
    you cannot reduce drawdown further without also reducing CAGR.

    The resulting plot shows you every point on the risk-return tradeoff
    curve so you can choose the allocation that matches your risk tolerance,
    rather than having to commit to a single objective upfront.

    Also marks your current allocation on the chart for reference.
    """
    print(f"\nRunning Pareto frontier analysis...")
    print(f"  CAGR targets: {list(cagr_targets)}")
    print(f"  Trials per target: {n_trials}\n")

    tickers      = list(allocation.keys())
    frontier_pts = []   # list of (cagr, drawdown, weights) tuples

    for min_cagr in cagr_targets:
        print(f"  Optimising for CAGR >= {min_cagr:.1f}%...")
        best_weights, best_score = optimise_random(
            prices, benchmark_prices, allocation,
            min_weight, max_weight, min_cagr, n_trials, "calmar", random_seed
        )

        if best_weights is None:
            print(f"    No valid allocation found for CAGR >= {min_cagr:.1f}% -- skipping.")
            continue

        opt_alloc = dict(zip(tickers, best_weights))
        bt        = run_backtest(prices, benchmark_prices, opt_alloc)
        series    = bt["All Weather Value"]
        years     = (bt.index[-1] - bt.index[0]).days / 365.25
        actual_cagr = compute_cagr(series, years)
        actual_mdd  = compute_max_drawdown(series)

        print(f"    -> CAGR: {actual_cagr:.2f}%  |  Max DD: {actual_mdd:.2f}%  "
              f"|  Calmar: {compute_calmar(actual_cagr, actual_mdd):.3f}")

        frontier_pts.append((actual_cagr, actual_mdd,
                             dict(zip(tickers, best_weights))))

    if not frontier_pts:
        print("No frontier points found. Try a wider CAGR range or more trials.")
        return

    # Current allocation benchmark point
    bt_current   = run_backtest(prices, benchmark_prices, allocation)
    series_cur   = bt_current["All Weather Value"]
    years_cur    = (bt_current.index[-1] - bt_current.index[0]).days / 365.25
    current_cagr = compute_cagr(series_cur, years_cur)
    current_mdd  = compute_max_drawdown(series_cur)

    # Sort frontier by CAGR for a clean curve
    frontier_pts.sort(key=lambda x: x[0])
    cagrs    = [p[0] for p in frontier_pts]
    drawdowns = [abs(p[1]) for p in frontier_pts]   # plot as positive for readability

    # Save frontier data to CSV
    frontier_df = pd.DataFrame([
        {"Min CAGR Target (%)": cagr_targets[i] if i < len(cagr_targets) else "",
         "Actual CAGR (%)":     round(p[0], 2),
         "Max Drawdown (%)":    round(p[1], 2),
         "Calmar":              round(compute_calmar(p[0], p[1]), 3),
         "Weights":             " | ".join(f"{t}={w:.1%}"
                                           for t, w in p[2].items())}
        for i, p in enumerate(frontier_pts)
    ])
    frontier_df.to_csv(os.path.join(results_dir, "pareto_frontier.csv"), index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#c9d1d9")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.label.set_color("#c9d1d9")
    ax.xaxis.label.set_color("#c9d1d9")
    ax.title.set_color("white")
    ax.grid(alpha=0.3, color="#30363d")

    # Frontier curve
    ax.plot(drawdowns, cagrs, "o-", color="#58a6ff", lw=2, markersize=7,
            label="Efficient frontier (Calmar-optimised)")

    # Label each point with its Calmar ratio
    for cagr, mdd_abs in zip(cagrs, drawdowns):
        calmar = compute_calmar(cagr, -mdd_abs)
        ax.annotate(f"Calmar={calmar:.2f}",
                    xy=(mdd_abs, cagr), xytext=(6, 0),
                    textcoords="offset points",
                    color="#8b949e", fontsize=7.5)

    # Current allocation
    ax.scatter([abs(current_mdd)], [current_cagr],
               color="#f0b429", s=120, zorder=5,
               label=f"Current allocation  (Calmar={compute_calmar(current_cagr, current_mdd):.2f})")

    ax.set_xlabel("Max Drawdown (%) -- lower is better", fontsize=11)
    ax.set_ylabel("CAGR (%) -- higher is better", fontsize=11)
    ax.set_title("Pareto Frontier: CAGR vs Max Drawdown\n"
                 "Each point is the best achievable CAGR for a given drawdown tolerance",
                 fontsize=12, pad=10)
    ax.legend(fontsize=9, facecolor="#21262d", edgecolor="#30363d", labelcolor="white")

    save_path = os.path.join(results_dir, "pareto_frontier.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\n  Pareto frontier saved -> {save_path}")
    plt.show()


# ===========================================================================
# PLOTTING
# ===========================================================================

def style_ax(ax):
    """Apply consistent dark theme to an axes object."""
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#c9d1d9", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.label.set_color("#c9d1d9")
    ax.xaxis.label.set_color("#c9d1d9")
    ax.title.set_color("white")
    ax.grid(axis="y", color="#30363d", alpha=0.5, linewidth=0.7)


def plot_backtest(backtest: pd.DataFrame,
                  stats_list: list[StrategyStats],
                  results_dir: str,
                  label: str):
    """
    Two-panel dark-theme chart:
      Panel 1 -- Portfolio value over time for all three strategies
      Panel 2 -- Annual returns side by side
    """
    fig = plt.figure(figsize=(15, 13))
    fig.patch.set_facecolor("#0d1117")

    gs  = fig.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.48)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    COLORS = {"aw": "#58a6ff", "bh": "#f0b429", "spy": "#f78166"}

    for ax in [ax1, ax2]:
        style_ax(ax)

    ax1.plot(backtest.index, backtest["All Weather Value"],
             color=COLORS["aw"], lw=2.2,
             label="All Weather (Rebalanced monthly)")
    ax1.plot(backtest.index, backtest["Buy & Hold All Weather"],
             color=COLORS["bh"], lw=2.0, linestyle=(0, (4, 2)),
             label="Buy & Hold All Weather (never rebalanced)")
    ax1.plot(backtest.index, backtest["S&P 500 Value"],
             color=COLORS["spy"], lw=1.6, linestyle="--", alpha=0.85,
             label="S&P 500 (SPY)")
    ax1.fill_between(backtest.index, backtest["All Weather Value"],
                     alpha=0.08, color=COLORS["aw"])

    ax1.set_title(f"Portfolio Backtest -- {label}  "
                  f"({backtest_start} to {backtest_end})",
                  fontsize=13, pad=10)
    ax1.set_ylabel("Portfolio Value ($)", fontsize=10)
    ax1.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.legend(fontsize=9, facecolor="#21262d", edgecolor="#30363d",
               labelcolor="white", loc="upper left")

    s_aw, s_bh, s_spy = stats_list
    ax1.text(
        0.99, 0.06,
        f"Rebalanced: ${s_aw.final_value:,.0f}   "
        f"Calmar={s_aw.calmar:.2f}     "
        f"Buy & Hold: ${s_bh.final_value:,.0f}     "
        f"S&P 500: ${s_spy.final_value:,.0f}",
        transform=ax1.transAxes, ha="right", va="bottom",
        color="#8b949e", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#21262d",
                  edgecolor="#30363d", alpha=0.9)
    )

    def annual_returns(col):
        return backtest[col].resample("YE").last().pct_change().dropna() * 100

    aw_ann  = annual_returns("All Weather Value")
    bh_ann  = annual_returns("Buy & Hold All Weather")
    spy_ann = annual_returns("S&P 500 Value")

    years_idx = aw_ann.index
    x         = np.arange(len(years_idx))
    width     = 0.26

    ax2.bar(x - width, aw_ann.reindex(years_idx, fill_value=0).values,
            width, label="All Weather (Rebal.)",
            color=[COLORS["aw"] if v >= 0 else "#f85149"
                   for v in aw_ann.reindex(years_idx, fill_value=0).values],
            alpha=0.85)
    ax2.bar(x, bh_ann.reindex(years_idx, fill_value=0).values,
            width, label="Buy & Hold AW",
            color=[COLORS["bh"] if v >= 0 else "#b08800"
                   for v in bh_ann.reindex(years_idx, fill_value=0).values],
            alpha=0.85)
    ax2.bar(x + width, spy_ann.reindex(years_idx, fill_value=0).values,
            width, label="S&P 500",
            color=[COLORS["spy"] if v >= 0 else "#da3633"
                   for v in spy_ann.reindex(years_idx, fill_value=0).values],
            alpha=0.70)

    ax2.set_xticks(x)
    ax2.set_xticklabels([d.year for d in years_idx], rotation=45, fontsize=8)
    ax2.axhline(0, color="#8b949e", lw=0.8)
    ax2.set_ylabel("Annual Return (%)", fontsize=10)
    ax2.set_title("Annual Returns by Year -- All Three Strategies",
                  fontsize=11, pad=8)
    ax2.legend(fontsize=8, facecolor="#21262d", edgecolor="#30363d",
               labelcolor="white", ncol=3)

    save_path = os.path.join(results_dir, "backtest.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  Plot saved -> {save_path}")
    plt.show()


# ===========================================================================
# EXPORT
# ===========================================================================
def save_run_config(allocation: dict, results_dir: str):
    """
    Save all parameters and the target allocation to a JSON file so that
    any run can be reproduced exactly by copying these values back into
    the user parameters section at the top of the script.
    """
    config = {
        "run_label":                run_label,
        "timestamp":                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "backtest_start":           backtest_start,
        "backtest_end":             backtest_end,
        "initial_portfolio_value":  initial_portfolio_value,
        "rebalance_threshold":      rebalance_threshold,
        "benchmark_ticker":         benchmark_ticker,
        "optimiser": {
            "run_optimiser":        RUN_OPTIMISER,
            "method":               OPT_METHOD if RUN_OPTIMISER else "not run",
            "min_weight":           OPT_MIN_WEIGHT,
            "max_weight":           OPT_MAX_WEIGHT,
            "min_cagr":             OPT_MIN_CAGR,
            "n_trials":             OPT_N_TRIALS,
            "random_seed":          OPT_RANDOM_SEED,
        },
        "pareto": {
            "run_pareto":           RUN_PARETO,
            "cagr_range":           list(PARETO_CAGR_RANGE) if RUN_PARETO else [],
        },
        "target_allocation":        allocation,
    }

    path = os.path.join(results_dir, "run_config.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2)

def export_results(backtest: pd.DataFrame,
                   instructions: pd.DataFrame,
                   stats_list: list[StrategyStats],
                   allocation: dict,
                   results_dir: str):
    """Export backtest history, rebalancing instructions, stats, and allocation."""

    stats_rows = []
    for s in stats_list:
        stats_rows.extend([
            {"Strategy": s.name, "Metric": "Period (years)",   "Value": s.period_years},
            {"Strategy": s.name, "Metric": "CAGR (%)",         "Value": s.cagr},
            {"Strategy": s.name, "Metric": "Max Drawdown (%)", "Value": s.max_drawdown},
            {"Strategy": s.name, "Metric": "Sharpe Ratio",     "Value": s.sharpe},
            {"Strategy": s.name, "Metric": "Calmar Ratio",     "Value": s.calmar},
            {"Strategy": s.name, "Metric": "Final Value ($)",  "Value": s.final_value},
        ])
    stats_df = pd.DataFrame(stats_rows)

    backtest.to_csv(os.path.join(results_dir, "backtest_history.csv"))
    instructions.to_csv(os.path.join(results_dir, "rebalancing_instructions.csv"),
                        index=False)
    stats_df.to_csv(os.path.join(results_dir, "stats.csv"), index=False)

    alloc_df = pd.DataFrame([
        {"Ticker": t, "Weight": w, "Weight (%)": f"{w:.1%}"}
        for t, w in allocation.items()
    ])
    alloc_df.to_csv(os.path.join(results_dir, "allocation.csv"), index=False)

    save_run_config(allocation, results_dir)

    print(f"  backtest_history.csv")
    print(f"  rebalancing_instructions.csv")
    print(f"  stats.csv  (now includes Calmar ratio)")
    print(f"  allocation.csv")
    print(f"  run_config.json")

# ===========================================================================
# PRETTY PRINTING
# ===========================================================================

def print_header(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_rebalancing(instructions: pd.DataFrame, total_value: float):
    print_header("MONTHLY REBALANCING INSTRUCTIONS")
    print(f"  Total Portfolio Value: ${total_value:,.2f}\n")

    needs_action = instructions[instructions["Action"] != "HOLD"]
    hold_tickers = instructions[instructions["Action"] == "HOLD"]["Ticker"].tolist()

    if needs_action.empty:
        print("  No rebalancing needed -- all allocations within threshold.\n")
    else:
        for _, row in needs_action.iterrows():
            tag = "SELL" if row["Action"] == "SELL" else "BUY "
            print(f"  {tag}  {row['Ticker']:5s}  ${row['$ Amount']:>8,.2f}"
                  f"   (current {row['Current Weight']:.1f}%  ->"
                  f"  target {row['Target Weight']:.1f}%)")

    if hold_tickers:
        print(f"\n  HOLD: {', '.join(hold_tickers)}")

    print(f"\n  Full breakdown:")
    print(instructions[["Ticker", "Current Weight", "Target Weight",
                         "Drift (%)", "Action", "$ Amount"]].to_string(index=False))


def print_stats(stats_list: list[StrategyStats]):
    """Print performance statistics for all strategies."""
    print_header("PERFORMANCE STATISTICS")
    for s in stats_list:
        print(f"\n  --- {s.name} ---")
        print(f"  {'Period (years)':<25} {s.period_years}")
        print(f"  {'CAGR (%)':<25} {s.cagr}")
        print(f"  {'Max Drawdown (%)':<25} {s.max_drawdown}")
        print(f"  {'Sharpe Ratio':<25} {s.sharpe}")
        print(f"  {'Calmar Ratio':<25} {s.calmar}")
        print(f"  {'Final Value ($)':<25} {s.final_value}")


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    validate_config()

    # ---- Create results folder for this run ----
    results_dir = make_results_dir(run_label)
    print(f"Results will be saved to: {results_dir}\n")

    # ---- Fetch price data ----
    all_tickers = list(dict.fromkeys(
        list(target_allocation.keys()) + [benchmark_ticker]
    ))
    prices = fetch_prices(all_tickers, backtest_start, backtest_end)

    port_prices  = prices[list(target_allocation.keys())]
    bench_prices = prices[benchmark_ticker]

    # ---- Optimiser (optional) ----
    if RUN_OPTIMISER:
        optimised = optimise_allocation(
            prices           = port_prices,
            benchmark_prices = bench_prices,
            allocation       = target_allocation,
            method           = OPT_METHOD,
            min_weight       = OPT_MIN_WEIGHT,
            max_weight       = OPT_MAX_WEIGHT,
            min_cagr         = OPT_MIN_CAGR,
            n_trials         = OPT_N_TRIALS,
        )
        target_allocation.update(optimised)
        print(f"\nTarget allocation updated to optimised weights.")

    # ---- Pareto frontier (optional) ----
    if RUN_PARETO:
        run_pareto_frontier(
            prices           = port_prices,
            benchmark_prices = bench_prices,
            allocation       = target_allocation,
            cagr_targets     = PARETO_CAGR_RANGE,
            min_weight       = OPT_MIN_WEIGHT,
            max_weight       = OPT_MAX_WEIGHT,
            n_trials         = OPT_N_TRIALS,
            results_dir      = results_dir,
        )

    # ---- Walk-forward validation (optional) ----
    if RUN_WALK_FORWARD:
        run_walk_forward(
            prices           = port_prices,
            benchmark_prices = bench_prices,
            allocation       = target_allocation,
            train_years      = WF_TRAIN_YEARS,
            test_years       = WF_TEST_YEARS,
            step_years       = WF_STEP_YEARS,
            min_weight       = OPT_MIN_WEIGHT,
            max_weight       = OPT_MAX_WEIGHT,
            n_trials         = OPT_N_TRIALS,
            results_dir      = results_dir,
        )

    # ---- Current holdings & rebalancing ----
    latest_prices = port_prices.iloc[-1]

    holdings = load_holdings()
    if holdings is None:
        print("No existing holdings found. Initialising with target allocation...\n")
        holdings = initialise_holdings(latest_prices, target_allocation,
                                       initial_portfolio_value)
        save_holdings(holdings)
    elif set(holdings.keys()) != set(target_allocation.keys()):
        print("Target allocation has changed -- resetting holdings...\n")
        print(f"  Old tickers: {sorted(holdings.keys())}")
        print(f"  New tickers: {sorted(target_allocation.keys())}\n")
        holdings = initialise_holdings(latest_prices, target_allocation,
                                       initial_portfolio_value)
        save_holdings(holdings)

    instructions, total_value = rebalancing_instructions(
        holdings, latest_prices, target_allocation, rebalance_threshold
    )
    print_rebalancing(instructions, total_value)

    needs_rebalance = (instructions["Action"] != "HOLD").any()
    if needs_rebalance:
        answer = input("\n  Apply rebalancing now and save? (y/n): ").strip().lower()
        if answer == "y":
            holdings = apply_rebalance(holdings, latest_prices,
                                       target_allocation, total_value)
            save_holdings(holdings)
            print("  Portfolio rebalanced and saved.")

    # ---- Backtest ----
    print_header(f"RUNNING BACKTEST ({backtest_start} to {backtest_end})")
    backtest   = run_backtest(port_prices, bench_prices)
    stats_list = compute_stats(backtest)
    print_stats(stats_list)

    # ---- Export ----
    print_header(f"SAVING RESULTS TO {results_dir}")
    export_results(backtest, instructions, stats_list, target_allocation, results_dir)
    append_to_master_log(results_dir, stats_list, target_allocation, run_label)

    # ---- Plot ----
    plot_backtest(backtest, stats_list, results_dir, run_label)
