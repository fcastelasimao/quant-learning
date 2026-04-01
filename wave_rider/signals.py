from __future__ import annotations

import numpy as np
import pandas as pd

import config


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.sort_index().pct_change(fill_method=None).dropna(how="all")


def total_return(returns: pd.DataFrame, window: int) -> pd.Series:
    recent = returns.iloc[-window:]
    compounded = (1.0 + recent).prod(min_count=window) - 1.0
    return compounded


def blended_momentum_score(
    returns: pd.DataFrame,
    windows: tuple[int, ...],
    weights: tuple[float, ...],
) -> pd.Series:
    if len(windows) != len(weights):
        raise ValueError("Momentum windows and weights must be the same length.")

    score = pd.Series(0.0, index=returns.columns, dtype=float)
    for window, weight in zip(windows, weights):
        score = score.add(total_return(returns, window) * weight, fill_value=0.0)
    return score


def absolute_trend_filter(returns: pd.DataFrame, window: int) -> pd.Series:
    return total_return(returns, window) > 0


def annualized_volatility(returns: pd.DataFrame, window: int) -> pd.Series:
    recent = returns.iloc[-window:]
    return recent.std() * np.sqrt(config.TRADING_DAYS_PER_YEAR)


def trend_breadth(eligible_assets: pd.Series) -> float:
    clean = eligible_assets.dropna()
    if clean.empty:
        return 0.0
    return float(clean.mean())


def defense_scale(
    breadth: float,
    thresholds: tuple[tuple[float, float], ...] = config.DEFENSE_BREADTH_THRESHOLDS,
) -> float:
    for minimum_breadth, scale in thresholds:
        if breadth >= minimum_breadth:
            return scale
    return 0.25
