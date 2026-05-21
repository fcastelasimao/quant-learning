from __future__ import annotations

import numpy as np
import pandas as pd

from experiments import MethodSpec, evaluate_method


def test_strategy_applies_one_day_signal_lag() -> None:
    index = pd.date_range("2020-01-01", periods=6)
    prices = pd.Series([100, 101, 102, 103, 104, 105], index=index, dtype=float)

    row, equity = evaluate_method(prices, MethodSpec("raw_return", "raw", {}), cost_bps=0.0)

    expected_returns = prices.pct_change(fill_method=None).fillna(0.0)
    expected_position = pd.Series([0, 0, 1, 1, 1, 1], index=index, dtype=float)
    expected_equity = (1.0 + expected_position * expected_returns).cumprod()

    assert row["observations"] == 4
    pd.testing.assert_series_equal(equity, expected_equity.rename("raw_return"))
