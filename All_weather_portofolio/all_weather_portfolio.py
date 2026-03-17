"""
Ray Dalio All Weather Portfolio Tracker
========================================
Tracks your portfolio, gives monthly rebalancing instructions,
backtests performance vs two benchmarks:
  1. S&P 500 (SPY)         -- "what if I just bought the market?"
  2. Buy & Hold All Weather -- "what if I set up Dalio's allocation and did nothing?"

Also includes a portfolio optimiser that searches for the weight combination
that minimises max drawdown while respecting a minimum CAGR constraint.


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
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")   # suppress yfinance / pandas deprecation noise

try:
    import yfinance as yf
except ImportError:
    raise ImportError("Please install yfinance: pip install yfinance")

try:
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Note: scipy not installed. Optimiser will be disabled.")
    print("      Install with: pip install scipy\n")

# ===========================================================================
# USER PARAMETERS -- CHANGE THESE
# ===========================================================================

initial_portfolio_value = 10_000    # Starting portfolio value in USD

backtest_start = "2006-01-01"       # format: YYYY-MM-DD
backtest_end   = "2026-01-01"       # format: YYYY-MM-DD

rebalance_threshold = 0.05          # Rebalance if any asset drifts > this % from target

# Label for this run -- used to name the results folder.
# Change this every time you want to save a distinct set of results.
# Examples: "original_allweather", "no_commodities", "optimised_min_drawdown"
run_label = "original_allweather"

# ===========================================================================
# TARGET ALLOCATION
# ===========================================================================

target_allocation = {
    "SPY":  0.18,   # US Large Cap stocks
    "QQQ":  0.12,   # US Tech stocks
    "TLT":  0.40,   # Long-Term Government Bonds
    "LQD":  0.15,   # Investment Grade Corporate Bonds
    "GLD":  0.075,  # Gold
    "DJP":  0.075,  # Commodities
}

assert abs(sum(target_allocation.values()) - 1.0) < 1e-6, \
    f"Weights must sum to 1.0, got {sum(target_allocation.values()):.4f}"

benchmark_ticker = "SPY"            # S&P 500 benchmark for comparison

holdings_file = "portfolio_holdings.json"

# ===========================================================================
# OPTIMISER PARAMETERS -- only used when RUN_OPTIMISER = True
# ===========================================================================

RUN_OPTIMISER = True               # Set to True to run the optimiser

# Minimum and maximum weight allowed per asset during optimisation.
# Prevents the optimiser from putting 0% or 100% in a single asset.
OPT_MIN_WEIGHT = 0.05               # each asset must be at least 5%
OPT_MAX_WEIGHT = 0.60               # each asset can be at most 60%

# The optimiser minimises max drawdown but also requires the portfolio
# to achieve at least this CAGR. Set to 0.0 to ignore the constraint.
OPT_MIN_CAGR = 4.0                  # minimum acceptable CAGR in percent

# ===========================================================================
# RESULTS VERSIONING
# ===========================================================================

def make_results_dir(label: str) -> str:
    """
    Create a timestamped results folder for this run inside a 'results/' directory.
    Folder name format: results/YYYY-MM-DD_HH-MM_<label>/

    Every run gets its own folder so old results are never overwritten.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    folder    = os.path.join("results", f"{timestamp}_{label}")
    os.makedirs(folder, exist_ok=True)
    return folder


def append_to_master_log(results_dir: str, stats: dict, weights: dict):
    """
    Append a one-row summary of this run to results/master_log.csv.

    This gives a single CSV where every row is one run, making it easy
    to compare many different allocations side by side.

    Columns: Timestamp | Label | Tickers+Weights | CAGR | MaxDD | Sharpe | FinalValue
    """
    log_path = os.path.join("results", "master_log.csv")

    # Extract the All Weather (rebalanced) stats from the stats dict
    def get_stat(key):
        for k, v in stats.items():
            if k.strip() == key:
                return v
        return ""

    # Build one summary row
    row = {
        "Timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Label":           run_label,
        "Backtest Start":  backtest_start,
        "Backtest End":    backtest_end,
        "Tickers":         " | ".join(f"{t}={w:.1%}" for t, w in weights.items()),
        "AW CAGR (%)":     get_stat("CAGR (%)"),
        "AW Max DD (%)":   get_stat("Max Drawdown (%)"),
        "AW Sharpe":       get_stat("Sharpe Ratio"),
        "AW Final ($)":    get_stat("Final Value ($)"),
        "BH CAGR (%)":     get_stat("CAGR (%) "),
        "BH Max DD (%)":   get_stat("Max Drawdown (%) "),
        "SPY CAGR (%)":    get_stat("CAGR (%)  "),
        "SPY Max DD (%)":  get_stat("Max Drawdown (%)  "),
        "Results Folder":  results_dir,
    }

    log_df = pd.DataFrame([row])

    if os.path.exists(log_path):
        # Append without writing header again
        log_df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        # First run -- create the file with header
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
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end   = datetime.strptime(end_date,   "%Y-%m-%d")

    print(f"Fetching price data for: {', '.join(tickers)}")
    print(f"Period: {start_date} -> {end_date}\n")

    raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)

    # yfinance returns a MultiIndex when downloading multiple tickers
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.dropna(how="all")   # drop rows where ALL tickers are NaN
    prices = prices.ffill()             # forward-fill weekends and holidays

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


def initialise_holdings(prices_row: pd.Series, allocation: dict) -> dict:
    """
    Create a fresh set of holdings by splitting initial_portfolio_value
    according to allocation at the current prices.

    Returns a dict like:
        {"VTI": {"shares": 12.34, "last_price": 210.50}, ...}
    """
    holdings = {}
    for ticker, weight in allocation.items():
        dollar_amount = initial_portfolio_value * weight
        price         = float(prices_row[ticker])
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


def rebalancing_instructions(holdings: dict, prices_row: pd.Series) -> pd.DataFrame:
    """
    Compare current weights to targets and produce buy/sell instructions.
    Returns a DataFrame and the total portfolio value.
    """
    weights, total_value = current_weights(holdings, prices_row)

    rows = []
    for ticker, target in target_allocation.items():
        current = weights.get(ticker, 0.0)
        drift   = current - target
        dollar  = drift * total_value       # positive = overweight -> sell
        action  = "HOLD"
        if abs(drift) > rebalance_threshold:
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


def apply_rebalance(holdings: dict, prices_row: pd.Series) -> dict:
    """Reset each holding to its target dollar weight at the current prices."""
    _, total_value = current_weights(holdings, prices_row)
    for ticker, target in target_allocation.items():
        price = float(prices_row[ticker])
        holdings[ticker] = {
            "shares":     round((total_value * target) / price, 6),
            "last_price": round(price, 4),
        }
    return holdings


# ===========================================================================
# BACK-TEST ENGINE
# ===========================================================================

def run_backtest(prices: pd.DataFrame,
                 benchmark_prices: pd.Series,
                 allocation: Optional[dict] = None) -> pd.DataFrame:
    """
    Simulate THREE strategies from the first available date:

      1. Rebalanced portfolio  -- uses `allocation` (defaults to target_allocation)
      2. Buy & Hold            -- same starting weights, never rebalanced
      3. S&P 500 buy & hold    -- everything in SPY, never touched

    The `allocation` parameter allows the optimiser to call this function
    with different weight combinations without changing target_allocation.
    """
    if allocation is None:
        allocation = target_allocation

    tickers = list(allocation.keys())

    # Resample to month-end prices
    monthly = prices[tickers].resample("ME").last().dropna()
    bench   = benchmark_prices.resample("ME").last().dropna()

    # Align to common dates
    common  = monthly.index.intersection(bench.index)
    monthly = monthly.loc[common]
    bench   = bench.loc[common]

    if monthly.empty:
        raise ValueError("No overlapping monthly data found. Check date range.")

    first_row    = monthly.iloc[0]
    bench_shares = initial_portfolio_value / float(bench.iloc[0])

    # Initialise holdings for both strategies
    aw_holdings = {t: (initial_portfolio_value * w) / float(first_row[t])
                   for t, w in allocation.items()}
    bh_holdings = {t: (initial_portfolio_value * w) / float(first_row[t])
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

        # Rebalance the rebalanced portfolio only
        for t, w in allocation.items():
            aw_holdings[t] = (aw_value * w) / float(row[t])
        # Buy & Hold: do nothing

    df = pd.DataFrame(records).set_index("Date")
    for col in ["All Weather Value", "Buy & Hold All Weather", "S&P 500 Value"]:
        df[f"{col} Monthly Ret (%)"] = df[col].pct_change() * 100

    return df


# ===========================================================================
# STATISTICS
# ===========================================================================

def compute_stats(backtest: pd.DataFrame) -> dict:
    """Compute key performance statistics for all three strategies."""
    years = (backtest.index[-1] - backtest.index[0]).days / 365.25

    def cagr(series):
        return ((series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1) * 100

    def max_drawdown(series):
        peak = series.cummax()
        return ((series - peak) / peak).min() * 100

    def sharpe(ret_col):
        r = backtest[ret_col].dropna() / 100
        return 0.0 if r.std() == 0 else (r.mean() / r.std()) * np.sqrt(12)

    aw  = backtest["All Weather Value"]
    bh  = backtest["Buy & Hold All Weather"]
    spy = backtest["S&P 500 Value"]

    return {
        "Period (years)":                    round(years, 1),
        "--- All Weather (Rebalanced) ---":  "",
        "  CAGR (%)":                        round(cagr(aw), 2),
        "  Max Drawdown (%)":                round(max_drawdown(aw), 2),
        "  Sharpe Ratio":                    round(sharpe("All Weather Value Monthly Ret (%)"), 3),
        "  Final Value ($)":                 round(aw.iloc[-1], 2),
        "--- Buy & Hold All Weather ---":    "",
        "  CAGR (%) ":                       round(cagr(bh), 2),
        "  Max Drawdown (%) ":               round(max_drawdown(bh), 2),
        "  Sharpe Ratio ":                   round(sharpe("Buy & Hold All Weather Monthly Ret (%)"), 3),
        "  Final Value ($) ":                round(bh.iloc[-1], 2),
        "--- S&P 500 Buy & Hold ---":        "",
        "  CAGR (%)  ":                      round(cagr(spy), 2),
        "  Max Drawdown (%)  ":              round(max_drawdown(spy), 2),
        "  Sharpe Ratio  ":                  round(sharpe("S&P 500 Value Monthly Ret (%)"), 3),
        "  Final Value ($)  ":               round(spy.iloc[-1], 2),
    }


# ===========================================================================
# OPTIMISER
# ===========================================================================

def optimise_allocation(prices: pd.DataFrame,
                        benchmark_prices: pd.Series) -> dict:
    """
    Find the portfolio weights that minimise max drawdown subject to:
      - weights sum to 1
      - each weight between OPT_MIN_WEIGHT and OPT_MAX_WEIGHT
      - CAGR >= OPT_MIN_CAGR  (so we don't just find "hold cash")

    Uses scipy.optimize.minimize with the SLSQP method, which handles
    equality and inequality constraints efficiently.

    Returns the optimised allocation as a dict {ticker: weight}.
    """
    if not HAS_SCIPY:
        print("scipy not available -- cannot run optimiser.")
        return target_allocation

    tickers = list(target_allocation.keys())
    n       = len(tickers)

    def objective(weights):
        """What we want to minimise: max drawdown (as a positive number)."""
        allocation = dict(zip(tickers, weights))
        bt = run_backtest(prices, benchmark_prices, allocation)
        series = bt["All Weather Value"]
        peak   = series.cummax()
        mdd    = ((series - peak) / peak).min()   # negative number
        return mdd   # minimising a negative number = minimising drawdown magnitude

    def cagr_constraint(weights):
        """
        Inequality constraint: CAGR - OPT_MIN_CAGR >= 0
        scipy requires constraints to be >= 0.
        """
        allocation = dict(zip(tickers, weights))
        bt    = run_backtest(prices, benchmark_prices, allocation)
        series = bt["All Weather Value"]
        years  = (bt.index[-1] - bt.index[0]).days / 365.25
        cagr   = ((series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1) * 100
        return cagr - OPT_MIN_CAGR

    constraints = [
        {"type": "eq",   "fun": lambda w: sum(w) - 1.0},   # weights sum to 1
        {"type": "ineq", "fun": cagr_constraint},           # minimum CAGR
    ]

    bounds = [(OPT_MIN_WEIGHT, OPT_MAX_WEIGHT)] * n

    # Start from the current target allocation as the initial guess
    w0 = np.array(list(target_allocation.values()))

    print(f"\nRunning optimiser ({n} assets)...")
    print(f"  Objective:   minimise max drawdown")
    print(f"  Constraint:  CAGR >= {OPT_MIN_CAGR}%")
    print(f"  Weight bounds: [{OPT_MIN_WEIGHT:.0%}, {OPT_MAX_WEIGHT:.0%}] per asset")
    print(f"  This may take a minute...\n")

    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9},
    )

    if result.success:
        optimised = dict(zip(tickers, result.x))
        print("Optimisation converged successfully.")
        print("\nOptimised weights:")
        for t, w in optimised.items():
            original = target_allocation[t]
            change   = w - original
            arrow    = "+" if change >= 0 else ""
            print(f"  {t:5s}  {w:.1%}   (was {original:.1%}, {arrow}{change:.1%})")
        return optimised
    else:
        print(f"WARNING: Optimiser did not converge: {result.message}")
        print("Returning original target_allocation unchanged.")
        return target_allocation


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


def plot_backtest(backtest: pd.DataFrame, stats: dict, results_dir: str):
    """
    Two-panel dark-theme chart saved into results_dir:
      Panel 1 -- Portfolio value over time for all three strategies
      Panel 2 -- Annual returns side by side
    """
    fig = plt.figure(figsize=(15, 13))
    fig.patch.set_facecolor("#0d1117")

    gs  = fig.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.48)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    COLORS = {
        "aw":  "#58a6ff",
        "bh":  "#f0b429",
        "spy": "#f78166",
    }

    for ax in [ax1, ax2]:
        style_ax(ax)

    # ── Panel 1: Portfolio value over time ──────────────────────────────────
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

    ax1.set_title(f"Portfolio Backtest -- {run_label}  "
                  f"({backtest_start} to {backtest_end})",
                  fontsize=13, pad=10)
    ax1.set_ylabel("Portfolio Value ($)", fontsize=10)
    ax1.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.legend(fontsize=9, facecolor="#21262d", edgecolor="#30363d",
               labelcolor="white", loc="upper left")

    aw_final  = backtest["All Weather Value"].iloc[-1]
    bh_final  = backtest["Buy & Hold All Weather"].iloc[-1]
    spy_final = backtest["S&P 500 Value"].iloc[-1]
    ax1.text(
        0.99, 0.06,
        f"Rebalanced: ${aw_final:,.0f}     "
        f"Buy & Hold: ${bh_final:,.0f}     "
        f"S&P 500: ${spy_final:,.0f}",
        transform=ax1.transAxes, ha="right", va="bottom",
        color="#8b949e", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#21262d",
                  edgecolor="#30363d", alpha=0.9)
    )

    # ── Panel 2: Annual returns bar chart ───────────────────────────────────
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

def export_results(backtest: pd.DataFrame,
                   instructions: pd.DataFrame,
                   stats: dict,
                   results_dir: str):
    """Export backtest history, rebalancing instructions, and stats to results_dir."""

    # Build stats CSV with a Strategy column
    stats_rows       = []
    current_strategy = ""
    for k, v in stats.items():
        if k.startswith("---"):
            current_strategy = k.replace("-", "").strip()
        elif v != "":
            stats_rows.append({
                "Strategy": current_strategy,
                "Metric":   k.strip(),
                "Value":    v,
            })
    stats_df = pd.DataFrame(stats_rows)

    backtest.to_csv(os.path.join(results_dir, "backtest_history.csv"))
    instructions.to_csv(os.path.join(results_dir, "rebalancing_instructions.csv"),
                        index=False)
    stats_df.to_csv(os.path.join(results_dir, "stats.csv"), index=False)

    # Also save the allocation used in this run
    alloc_df = pd.DataFrame([
        {"Ticker": t, "Weight": w, "Weight (%)": f"{w:.1%}"}
        for t, w in target_allocation.items()
    ])
    alloc_df.to_csv(os.path.join(results_dir, "allocation.csv"), index=False)

    print(f"  backtest_history.csv")
    print(f"  rebalancing_instructions.csv")
    print(f"  stats.csv")
    print(f"  allocation.csv")


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


def print_stats(stats: dict):
    print_header("PERFORMANCE STATISTICS")
    for k, v in stats.items():
        if v == "":
            print(f"\n{k}")
        else:
            print(f"  {k:<40} {v}")


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":

    # ---- Create results folder for this run ----
    results_dir = make_results_dir(run_label)
    print(f"Results will be saved to: {results_dir}\n")

    # ---- Fetch price data ----
    # Deduplicate tickers in case benchmark is already in the allocation
    all_tickers = list(dict.fromkeys(
        list(target_allocation.keys()) + [benchmark_ticker]
    ))
    prices = fetch_prices(all_tickers, backtest_start, backtest_end)

    port_prices  = prices[list(target_allocation.keys())]
    bench_prices = prices[benchmark_ticker]

    # ---- Optimiser (optional) ----
    # If RUN_OPTIMISER is True, find the best weights and update target_allocation.
    # The optimised weights are printed and saved, then used for the backtest below.
    if RUN_OPTIMISER:
        optimised = optimise_allocation(port_prices, bench_prices)
        target_allocation.update(optimised)
        print(f"\nTarget allocation updated to optimised weights.")

    # ---- Current holdings & rebalancing ----
    latest_prices = port_prices.iloc[-1]

    holdings = load_holdings()
    if holdings is None:
        print("No existing holdings found. Initialising with target allocation...\n")
        holdings = initialise_holdings(latest_prices, target_allocation)
        save_holdings(holdings)
    elif set(holdings.keys()) != set(target_allocation.keys()):
        print("Target allocation has changed -- resetting holdings...\n")
        print(f"  Old tickers: {sorted(holdings.keys())}")
        print(f"  New tickers: {sorted(target_allocation.keys())}\n")
        holdings = initialise_holdings(latest_prices, target_allocation)
        save_holdings(holdings)

    instructions, total_value = rebalancing_instructions(holdings, latest_prices)
    print_rebalancing(instructions, total_value)

    needs_rebalance = (instructions["Action"] != "HOLD").any()
    if needs_rebalance:
        answer = input("\n  Apply rebalancing now and save? (y/n): ").strip().lower()
        if answer == "y":
            holdings = apply_rebalance(holdings, latest_prices)
            save_holdings(holdings)
            print("  Portfolio rebalanced and saved.")

    # ---- Backtest ----
    print_header(f"RUNNING BACKTEST ({backtest_start} to {backtest_end})")
    backtest = run_backtest(port_prices, bench_prices)
    stats    = compute_stats(backtest)
    print_stats(stats)

    # ---- Export ----
    print_header(f"SAVING RESULTS TO {results_dir}")
    export_results(backtest, instructions, stats, results_dir)
    append_to_master_log(results_dir, stats, target_allocation)

    # ---- Plot ----
    plot_backtest(backtest, stats, results_dir)