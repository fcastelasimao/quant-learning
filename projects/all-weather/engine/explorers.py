"""
explorers.py
============
Data quality diagnostics for use in exploratory notebooks.
No backtest logic, no live IO — pure DataFrame inspection helpers.
"""

from __future__ import annotations

import pandas as pd


def data_quality_report(prices: pd.DataFrame) -> pd.DataFrame:
    """Return per-ticker NaN counts, first valid date, and last valid date."""
    quality = prices.isna().sum().rename("Missing Values").to_frame()
    quality["First Date"] = prices.apply(
        lambda s: s.first_valid_index().date() if s.first_valid_index() is not None else None
    )
    quality["Last Date"] = prices.apply(
        lambda s: s.last_valid_index().date() if s.last_valid_index() is not None else None
    )
    return quality
