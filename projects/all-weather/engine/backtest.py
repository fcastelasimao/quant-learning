"""
backtest.py
===========
The simulation engine.

  - run_backtest   simulates up to four strategies over a price history
                   (three always; 60/40 only when tlt_prices is provided)

Performance statistics (compute_*, StrategyStats, compute_stats) live in
engine/stats.py and are re-imported here for caller convenience.

Closed investigations (run_backtest_rolling_rp, run_backtest_with_overlay)
have been moved to failed_strategies/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .stats import (
    StrategyStats,
    compute_cagr,
    compute_max_drawdown,
    compute_sharpe,
    compute_calmar,
    compute_max_drawdown_daily,
    compute_avg_drawdown,
    compute_max_drawdown_duration,
    compute_avg_recovery_time,
    compute_ulcer_index,
    compute_sortino,
    compute_stats,
)

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------
SIXTY_FORTY_EQUITY = 0.60
SIXTY_FORTY_BOND   = 0.40


# ===========================================================================
# BACKTEST ENGINE
# ===========================================================================

def run_backtest(prices: pd.DataFrame,
                 benchmark_prices: pd.Series,
                 allocation: dict,
                 portfolio_value: float | None = None,
                 tlt_prices: pd.Series | None = None,
                 transaction_cost_pct: float = 0.0,
                 tax_drag_pct: float = 0.0) -> pd.DataFrame:
    """
    Simulate four strategies over a price history and return monthly values.

    Strategies simulated:
      1. Rebalanced portfolio  -- rebalances to `allocation` every month
      2. Buy & Hold            -- same starting weights, never rebalanced
      3. S&P 500 buy & hold    -- everything in SPY on day one, never touched
      4. 60/40 annually rebal. -- 60% SPY / 40% TLT, rebalanced at the start
                                  of each calendar year
                                  (only included when tlt_prices is provided)

    Parameters
    ----------
    prices           : daily price DataFrame, one column per ticker
    benchmark_prices : daily price Series for the benchmark (SPY)
    allocation       : dict of {ticker: weight}, weights must sum to 1.0
    portfolio_value  : starting value in USD (defaults to config value)
    tlt_prices       : daily price Series for TLT; enables the 60/40 strategy

    Returns
    -------
    pd.DataFrame indexed by month-end date with columns:
        All Weather Value | Buy & Hold All Weather | S&P 500 Value
        60/40 Value (if tlt_prices provided)
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

    monthly     = prices[tickers].resample(config.DATA_FREQUENCY).last().dropna()
    bench       = benchmark_prices.resample(config.DATA_FREQUENCY).last().dropna()
    tlt_monthly = (tlt_prices.resample(config.DATA_FREQUENCY).last().dropna()
                   if tlt_prices is not None else None)

    common  = monthly.index.intersection(bench.index)
    if tlt_monthly is not None:
        common = common.intersection(tlt_monthly.index)
    monthly = monthly.loc[common]
    bench   = bench.loc[common]
    if tlt_monthly is not None:
        tlt_monthly = tlt_monthly.loc[common]

    if monthly.empty:
        raise ValueError("No overlapping monthly data found. Check date range.")

    first_row    = monthly.iloc[0]
    bench_shares = portfolio_value / float(bench.iloc[0])

    aw_holdings = {t: (portfolio_value * w) / float(first_row[t])
                   for t, w in allocation.items()}
    bh_holdings = {t: (portfolio_value * w) / float(first_row[t])
                   for t, w in allocation.items()}

    sixty_forty_spy = None
    sixty_forty_tlt = None
    sixty_forty_prev_year = None
    if tlt_monthly is not None:
        sixty_forty_spy = portfolio_value * SIXTY_FORTY_EQUITY / float(bench.iloc[0])
        sixty_forty_tlt = portfolio_value * SIXTY_FORTY_BOND / float(tlt_monthly.iloc[0])
        sixty_forty_prev_year = monthly.index[0].year

    aw_prev_year = monthly.index[0].year

    # Note: this loop is intentionally iterative due to stateful monthly rebalancing
    # with transaction costs and 60/40 annual rebalance logic.
    records = []
    for date, row in monthly.iterrows():
        # Annual rebalance for 60/40: restore 60/40 split at start of each new year
        if sixty_forty_spy is not None and date.year != sixty_forty_prev_year:
            current_6040 = (sixty_forty_spy * float(bench.loc[date])
                            + sixty_forty_tlt * float(tlt_monthly.loc[date]))
            if transaction_cost_pct > 0:
                spy_val = sixty_forty_spy * float(bench.loc[date])
                tlt_val = sixty_forty_tlt * float(tlt_monthly.loc[date])
                trade_values_6040 = (abs(current_6040 * SIXTY_FORTY_EQUITY - spy_val)
                                     + abs(current_6040 * SIXTY_FORTY_BOND - tlt_val))
                current_6040 -= trade_values_6040 * transaction_cost_pct
            sixty_forty_spy = current_6040 * SIXTY_FORTY_EQUITY / float(bench.loc[date])
            sixty_forty_tlt = current_6040 * SIXTY_FORTY_BOND / float(tlt_monthly.loc[date])
            sixty_forty_prev_year = date.year

        aw_value  = sum(sh * float(row[t]) for t, sh in aw_holdings.items())

        if tax_drag_pct > 0 and date.year != aw_prev_year:
            aw_value *= (1 - tax_drag_pct)
            aw_prev_year = date.year

        if transaction_cost_pct > 0:
            trade_values = sum(
                abs((aw_value * w) - (aw_holdings[t] * float(row[t])))
                for t, w in allocation.items()
            )
            aw_value -= trade_values * transaction_cost_pct
            for t, w in allocation.items():
                aw_holdings[t] = (aw_value * w) / float(row[t])
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

        if sixty_forty_spy is not None:
            sixty_forty_value = (sixty_forty_spy * float(bench.loc[date])
                                 + sixty_forty_tlt * float(tlt_monthly.loc[date]))
            record["60/40 Value"] = round(sixty_forty_value, 2)

        for t in tickers:
            record[f"B&H {t} Weight (%)"] = round(bh_weights[t] * 100, 1)

        records.append(record)

        # Rebalance: restore target weights for the rebalanced strategy only
        # (skipped when transaction_cost_pct > 0 as rebalancing already
        # happened inside the cost block above)
        if transaction_cost_pct == 0.0:
            for t, w in allocation.items():
                aw_holdings[t] = (aw_value * w) / float(row[t])
        # Buy & Hold: do nothing -- holdings stay fixed

    df = pd.DataFrame(records).set_index("Date")
    for col in ["All Weather Value", "Buy & Hold All Weather", "S&P 500 Value"]:
        df[f"{col} Monthly Ret (%)"] = df[col].pct_change() * 100
    if "60/40 Value" in df.columns:
        df["60/40 Value Monthly Ret (%)"] = df["60/40 Value"].pct_change() * 100

    return df
