"""
stats.py
========
Pure performance-statistics functions and the StrategyStats dataclass.

No simulation logic lives here — this module only receives price series
or return series and computes numbers from them.

All compute_* helpers are individually testable with synthetic data.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config


DAYS_PER_YEAR = 365.25


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
    name:              str
    cagr:              float     # compound annual growth rate (%)
    max_drawdown:      float     # peak-to-trough decline (%, negative number)
    sharpe:            float     # annualised Sharpe ratio
    calmar:            float     # CAGR / |max drawdown| -- balanced risk/return
    final_value:       float     # portfolio value at end of period ($)
    period_years:      float     # length of the backtest period
    avg_drawdown:      float     # mean drawdown across all underwater periods (%)
    max_dd_duration:   int       # longest consecutive months below peak
    avg_recovery_time: float     # average months to recover from a drawdown
    ulcer_index:       float     # RMS of all drawdown percentages
    sortino:           float     # downside-only Sharpe ratio
    martin:            float     # CAGR / Ulcer Index — primary optimisation metric (smoother than Calmar)
    max_drawdown_daily: float    # MDD computed on daily price resolution (more accurate than monthly)


# ===========================================================================
# STAT HELPERS
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


def compute_sharpe(monthly_ret_series: pd.Series,
                   rf_annual: float = 0.0) -> float:
    """
    Annualised Sharpe ratio from a series of monthly returns (as percentages).

    Sharpe = ((mean monthly return - rf_monthly) / std of monthly returns) * sqrt(12)
    rf_annual is converted to a monthly equivalent: (1 + rf_annual)^(1/12) - 1.

    Returns 0.0 if all returns are identical (zero volatility edge case).
    Set rf_annual=0.0 to reproduce pre-fix results.
    """
    r = monthly_ret_series.dropna() / 100
    if len(r) == 0 or r.std() < 1e-10:
        return 0.0
    rf_monthly = (1 + rf_annual) ** (1 / 12) - 1
    return ((r.mean() - rf_monthly) / r.std()) * np.sqrt(config.SHARPE_ANNUALISATION)


def compute_calmar(cagr: float, max_drawdown: float) -> float:
    """
    Calmar ratio = CAGR / |max drawdown|.

    Measures return per unit of drawdown accepted. Higher is better.
    A Calmar of 0.5 means you earn 0.5% of annual return for every 1%
    of maximum drawdown you accept.

    Returns 0.0 if drawdown is zero (degenerate / no-loss case).
    """
    if max_drawdown == 0.0:
        return 0.0
    return cagr / abs(max_drawdown)


def compute_max_drawdown_daily(prices: pd.DataFrame,
                               allocation: dict[str, float]) -> float:
    """
    Maximum drawdown computed at daily price resolution.

    Monthly MDD understates true drawdowns because it only sees month-end
    prices. A 20% intramonth drop that recovers by month-end is invisible
    to the monthly engine. Daily MDD captures the true worst-case experience.

    Uses a target-weight approximation: daily portfolio returns are the
    weighted sum of asset returns using target weights. Between monthly
    rebalances weights drift slightly, but the effect on MDD is negligible.

    Returns a negative number (same convention as compute_max_drawdown).
    Returns 0.0 if prices are empty or allocation is empty.
    """
    tickers = list(allocation.keys())
    available = [t for t in tickers if t in prices.columns]
    if not available or prices.empty:
        return 0.0

    daily = prices[available].ffill().dropna()
    if daily.empty or len(daily) < 2:
        return 0.0

    weights = np.array([allocation[t] for t in available])
    weights = weights / weights.sum()

    daily_rets = daily.pct_change().fillna(0.0).values  # (T, N)
    port_rets = daily_rets @ weights                     # (T,)

    port_values = config.INITIAL_PORTFOLIO_VALUE * np.cumprod(1.0 + port_rets)

    running_max = np.maximum.accumulate(port_values)
    drawdowns = (port_values - running_max) / running_max
    return float(np.min(drawdowns)) * 100


def compute_avg_drawdown(series: pd.Series) -> float:
    """
    Average of all drawdown values (not just the maximum).
    Returns a negative number. Closer to 0 is better.
    Returns 0.0 if the portfolio never draws down.
    """
    peak = series.cummax()
    dd_series = ((series - peak) / peak) * 100
    underwater = dd_series[dd_series < 0]
    if underwater.empty:
        return 0.0
    return round(underwater.mean(), 2)


def compute_max_drawdown_duration(series: pd.Series) -> int:
    """
    Maximum number of consecutive periods spent below a previous peak.
    Returns an integer (number of monthly periods). 0 means no drawdown.
    """
    peak = series.cummax()
    underwater = (series < peak)
    if not underwater.any():
        return 0
    groups = (~underwater).cumsum()
    max_duration = int(underwater.groupby(groups).sum().max())
    return max_duration


def compute_avg_recovery_time(series: pd.Series) -> float:
    """
    Average number of months to recover from each distinct drawdown episode.
    Returns 0.0 if no complete recovery episodes exist.
    """
    peak = series.cummax()
    underwater = (series < peak).values
    if not underwater.any():
        return 0.0

    transitions = np.diff(underwater.astype(np.int8), prepend=0)
    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0]

    n_complete = min(len(starts), len(ends))
    if n_complete == 0:
        return 0.0
    durations = ends[:n_complete] - starts[:n_complete]
    return round(float(np.mean(durations)), 1)


def compute_ulcer_index(series: pd.Series) -> float:
    """
    Ulcer Index — measures both depth and duration of drawdowns via RMS.
    Lower is better. Penalises extended periods moderately underwater, not
    just the single worst point.
    """
    peak = series.cummax()
    dd_pct = ((series - peak) / peak) * 100
    return round(float(np.sqrt((dd_pct ** 2).mean())), 4)


def compute_sortino(monthly_ret_series: pd.Series,
                    rf_annual: float = 0.0) -> float:
    """
    Sortino ratio — like Sharpe but only penalises downside volatility.
    Returns 0.0 if there are no below-threshold months or empty series.
    """
    r = monthly_ret_series.dropna() / 100
    if len(r) == 0:
        return 0.0
    rf_monthly = (1 + rf_annual) ** (1 / 12) - 1
    downside = r[r < rf_monthly]
    if len(downside) == 0 or downside.std() < 1e-10:
        return 0.0
    return round(((r.mean() - rf_monthly) / downside.std()) * np.sqrt(config.SHARPE_ANNUALISATION), 3)


# ===========================================================================
# AGGREGATE STATS BUILDER
# ===========================================================================

def compute_stats(backtest: pd.DataFrame,
                  prices: pd.DataFrame | None = None,
                  allocation: dict | None = None) -> list[StrategyStats]:
    """
    Compute key performance statistics for all strategies in the backtest.

    Parameters
    ----------
    backtest   : monthly backtest DataFrame from run_backtest()
    prices     : daily price DataFrame (optional). When provided alongside
                 allocation, daily MDD is computed for the AW_R strategy.
    allocation : allocation dict {ticker: weight} (optional, paired with prices)

    Returns a list of three or four StrategyStats objects:
      [All Weather (Rebalanced), Buy & Hold All Weather, S&P 500 Buy & Hold]
      plus 60/40 as the fourth element if the "60/40 Value" column exists.
    """
    required = {
        "All Weather Value", "Buy & Hold All Weather", "S&P 500 Value",
        "All Weather Value Monthly Ret (%)",
        "Buy & Hold All Weather Monthly Ret (%)",
        "S&P 500 Value Monthly Ret (%)",
    }
    missing = required - set(backtest.columns)
    if missing:
        raise ValueError(f"compute_stats: backtest is missing columns: {sorted(missing)}")

    years = (backtest.index[-1] - backtest.index[0]).days / DAYS_PER_YEAR

    daily_mdd_aw = 0.0
    if prices is not None and allocation is not None:
        mask         = ((prices.index >= backtest.index[0]) &
                        (prices.index <= backtest.index[-1]))
        prices_slice = prices.loc[mask]
        daily_mdd_aw = round(compute_max_drawdown_daily(prices_slice, allocation), 2)

    def make_stats(name: str, value_col: str, ret_col: str,
                   daily_mdd: float = 0.0) -> StrategyStats:
        series      = backtest[value_col]
        cagr        = round(compute_cagr(series, years), 2)
        mdd         = round(compute_max_drawdown(series), 2)
        ulcer       = compute_ulcer_index(series)
        martin      = round(cagr / ulcer if ulcer > 1e-10 else cagr, 3)
        return StrategyStats(
            name               = name,
            cagr               = cagr,
            max_drawdown       = mdd,
            sharpe             = round(compute_sharpe(backtest[ret_col],
                                                      rf_annual=config.RISK_FREE_RATE), 3),
            calmar             = round(compute_calmar(cagr, mdd), 3),
            final_value        = round(series.iloc[-1], 2),
            period_years       = round(years, 1),
            avg_drawdown       = compute_avg_drawdown(series),
            max_dd_duration    = compute_max_drawdown_duration(series),
            avg_recovery_time  = compute_avg_recovery_time(series),
            ulcer_index        = ulcer,
            sortino            = compute_sortino(backtest[ret_col],
                                                rf_annual=config.RISK_FREE_RATE),
            martin             = martin,
            max_drawdown_daily = daily_mdd,
        )

    stats = [
        make_stats("AW_R",
                   "All Weather Value",
                   "All Weather Value Monthly Ret (%)",
                   daily_mdd=daily_mdd_aw),
        make_stats("B&H_AW",
                   "Buy & Hold All Weather",
                   "Buy & Hold All Weather Monthly Ret (%)"),
        make_stats("SPY",
                   "S&P 500 Value",
                   "S&P 500 Value Monthly Ret (%)"),
    ]

    if "60/40 Value" in backtest.columns:
        stats.append(make_stats("60/40",
                                "60/40 Value",
                                "60/40 Value Monthly Ret (%)"))

    return stats


def compute_calendar_year_metrics(backtest: pd.DataFrame) -> pd.DataFrame:
    """
    Compute year-by-year PnL, return, and drawdown for each value series.

    PnL is measured from the previous year-end value to the current year-end
    value. For the first partial year, the start value is the first available
    observation in that year.
    """
    value_cols = [
        "All Weather Value",
        "Buy & Hold All Weather",
        "S&P 500 Value",
        "60/40 Value",
    ]
    value_cols = [c for c in value_cols if c in backtest.columns]

    rows = []
    for col in value_cols:
        series = backtest[col].dropna()
        if series.empty:
            continue

        prev_year_end = None
        for year, year_series in series.groupby(series.index.year):
            if year_series.empty:
                continue
            start_value = float(prev_year_end if prev_year_end is not None else year_series.iloc[0])
            end_value = float(year_series.iloc[-1])
            pnl = end_value - start_value
            ret = pnl / start_value * 100 if start_value else 0.0

            dd_base = pd.concat([
                pd.Series([start_value], index=[year_series.index[0] - pd.Timedelta(days=1)]),
                year_series,
            ])
            rows.append({
                "Year": int(year),
                "Strategy": col,
                "Start Value ($)": round(start_value, 2),
                "End Value ($)": round(end_value, 2),
                "PnL ($)": round(pnl, 2),
                "Return (%)": round(ret, 2),
                "Max Drawdown (%)": round(compute_max_drawdown(dd_base), 2),
            })
            prev_year_end = end_value

    return pd.DataFrame(rows)
