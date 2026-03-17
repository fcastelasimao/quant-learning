"""
backtest.py
===========
The simulation engine. Contains:

  - StrategyStats    dataclass for holding per-strategy performance metrics
  - compute_cagr     shared stat helper
  - compute_max_drawdown
  - compute_sharpe
  - compute_calmar
  - run_backtest     simulates three strategies over a price history
  - compute_stats    computes StrategyStats for all three strategies

This module is a pure simulation -- it knows nothing about real holdings,
file I/O, or user parameters beyond what is passed in as arguments.
The only import from config is INITIAL_PORTFOLIO_VALUE as a default,
which can always be overridden by the caller.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

import config


# ===========================================================================
# STATS DATACLASS
# ===========================================================================

@dataclass
class StrategyStats:
    """
    Holds performance statistics for a single strategy.

    Using a dataclass rather than a plain dict means:
      - No string key typos
      - No trailing-space hacks to avoid key collisions between strategies
      - Stats accessed by attribute (s.cagr) not fragile string (s["  CAGR (%) "])
      - Type-checked by static analysis tools
    """
    name:         str
    cagr:         float     # compound annual growth rate (%)
    max_drawdown: float     # peak-to-trough decline (%, negative number)
    sharpe:       float     # annualised Sharpe ratio
    calmar:       float     # CAGR / |max drawdown| -- balanced risk/return
    final_value:  float     # portfolio value at end of period ($)
    period_years: float     # length of the backtest period


# ===========================================================================
# SHARED STAT HELPERS
# ===========================================================================

def compute_cagr(series: pd.Series, years: float) -> float:
    """
    Compound Annual Growth Rate as a percentage.

    Formula: (end / start) ^ (1 / years) - 1
    Example: $10k -> $20k over 10 years = 7.18% CAGR
    """
    return ((series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1) * 100


def compute_max_drawdown(series: pd.Series) -> float:
    """
    Maximum peak-to-trough decline as a percentage.

    Returns a negative number. -20.0 means the portfolio dropped 20% from
    its peak at some point. 0.0 means it never fell below its starting value.

    Formula: min( (value - running_peak) / running_peak ) * 100
    """
    peak = series.cummax()
    return ((series - peak) / peak).min() * 100


def compute_sharpe(monthly_ret_series: pd.Series) -> float:
    """
    Annualised Sharpe ratio from a series of monthly returns (as percentages).

    Sharpe = (mean monthly return / std of monthly returns) * sqrt(12)
    The sqrt(12) annualises the monthly ratio.

    Returns 0.0 if all returns are identical (zero volatility edge case).
    """
    r = monthly_ret_series.dropna() / 100
    if len(r) == 0 or r.std() < 1e-10:
        return 0.0
    return (r.mean() / r.std()) * np.sqrt(12)


def compute_calmar(cagr: float, max_drawdown: float) -> float:
    """
    Calmar ratio = CAGR / |max drawdown|.

    Measures return per unit of drawdown accepted. Higher is better.
    A Calmar of 0.5 means you earn 0.5% of annual return for every 1%
    of maximum drawdown you accept.

    This is the primary optimisation objective because it balances return
    and risk without requiring you to choose how to weight them manually.

    Returns 0.0 if drawdown is zero (degenerate / no-loss case).
    """
    if max_drawdown == 0.0:
        return 0.0
    return cagr / abs(max_drawdown)


# ===========================================================================
# BACKTEST ENGINE
# ===========================================================================

def run_backtest(prices: pd.DataFrame,
                 benchmark_prices: pd.Series,
                 allocation: dict,
                 portfolio_value: Optional[float] = None) -> pd.DataFrame:
    """
    Simulate three strategies over a price history and return monthly values.

    Strategies simulated:
      1. Rebalanced portfolio  -- rebalances to `allocation` every month
      2. Buy & Hold            -- same starting weights, never rebalanced
      3. S&P 500 buy & hold    -- everything in SPY on day one, never touched

    Parameters
    ----------
    prices           : daily price DataFrame, one column per ticker
    benchmark_prices : daily price Series for the benchmark (SPY)
    allocation       : dict of {ticker: weight}, weights must sum to 1.0
    portfolio_value  : starting value in USD (defaults to config value)

    Returns
    -------
    pd.DataFrame indexed by month-end date with columns:
        All Weather Value | Buy & Hold All Weather | S&P 500 Value
        B&H <ticker> Weight (%) for each ticker
        Monthly Ret (%) columns for each strategy

    Notes
    -----
    "ME" resample = Month End. Groups daily prices to one price per month
    (the last trading day of each month). Requires pandas >= 2.2.
    Older pandas versions use "M" instead.

    .intersection() aligns the portfolio and benchmark to common dates so
    we are never comparing portfolio value and benchmark value on different
    dates due to data availability differences.

    .iloc[0] selects the first row by integer position (0-indexed),
    regardless of its date label.
    """
    if portfolio_value is None:
        portfolio_value = config.INITIAL_PORTFOLIO_VALUE

    tickers = list(allocation.keys())

    monthly = prices[tickers].resample("ME").last().dropna()
    bench   = benchmark_prices.resample("ME").last().dropna()

    common  = monthly.index.intersection(bench.index)
    monthly = monthly.loc[common]
    bench   = bench.loc[common]

    if monthly.empty:
        raise ValueError("No overlapping monthly data found. Check date range.")

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

        # Rebalance: restore target weights for the rebalanced strategy only
        for t, w in allocation.items():
            aw_holdings[t] = (aw_value * w) / float(row[t])
        # Buy & Hold: do nothing -- holdings stay fixed

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

    Returns a list of three StrategyStats objects in order:
      [All Weather (Rebalanced), Buy & Hold All Weather, S&P 500 Buy & Hold]
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
