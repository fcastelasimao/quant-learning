"""Constant-notional equity metrics for strategy evaluation.

Self-contained: no dependency on research/ modules.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def constant_notional_metrics(
    df: pd.DataFrame,
    sized_r_col: str = "r",
    exit_time_col: str = "exit_time",
    entry_time_col: str = "entry_time",
) -> dict:
    """Constant-notional Sharpe, MaxDD, Calmar, and CAGR.

    Convention: equity_t = 1 + cumsum(daily_pnl_t). Each trade contributes
    sized_r (already in return units, i.e. pnl_pct / 100 * size) on its
    exit date. Business-day calendar; no-trade days = 0.

    Parameters
    ----------
    df : DataFrame with exit_time_col (datetime), entry_time_col (datetime),
         and sized_r_col (float) columns.
    sized_r_col : name of the per-trade sized return column.

    Returns
    -------
    dict with keys:
        total_return, annualized_return (CAGR), sharpe_daily, max_drawdown,
        calmar, mean_daily_return, std_daily_return, n_trades, years
    """
    g = df.dropna(subset=[sized_r_col]).copy()
    g["_exit_date"] = pd.to_datetime(g[exit_time_col]).dt.normalize()
    g = g.dropna(subset=["_exit_date"])
    if g.empty:
        return {}

    start = pd.to_datetime(g[entry_time_col]).min().normalize()
    end = g["_exit_date"].max()
    bdays = pd.bdate_range(start, end)
    daily = g.groupby("_exit_date")[sized_r_col].sum().reindex(bdays, fill_value=0.0)

    total_return = float(daily.sum())
    n_days = len(bdays)
    years = n_days / 252.0
    mean_d = float(daily.mean())
    std_d = float(daily.std(ddof=1))
    sharpe = mean_d / std_d * np.sqrt(252) if std_d > 0 else float("nan")
    eq = 1.0 + daily.cumsum()
    dd = eq - eq.cummax()
    max_dd = float(dd.min())
    calmar = (
        float((total_return / years) / abs(max_dd))
        if years > 0 and max_dd < 0
        else float("nan")
    )
    return {
        "total_return": total_return,
        "annualized_return": float(total_return / years) if years > 0 else float("nan"),
        "sharpe_daily": sharpe,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "mean_daily_return": mean_d,
        "std_daily_return": std_d,
        "n_trades": len(g),
        "years": years,
    }
