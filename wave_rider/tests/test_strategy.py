from __future__ import annotations

import pandas as pd

from allocation import select_distinct_assets
from signals import absolute_trend_filter, defense_scale, trend_breadth
from strategy import (
    DEFAULT_PARAMETERS,
    build_strategy_callback,
    generate_target_weights,
    with_overrides,
)


def make_window_returns(days: int = 260) -> pd.DataFrame:
    index = pd.bdate_range("2023-01-02", periods=days)
    return pd.DataFrame(
        {
            "GLOBAL_EQ": [0.0010] * days,
            "US_LC": [0.0009] * days,
            "US_TECH": [0.0012] * days,
            "EUROPE": [0.0007] * days,
            "EM": [0.0006] * days,
            "LONG_BOND": [0.0001] * days,
            "INT_BOND": [0.0002] * days,
            "INFL_LINKED": [0.0002] * days,
            "GOLD": [0.0005] * days,
            "COMMODITIES": [0.0008] * days,
            "ENERGY": [0.0011] * days,
        },
        index=index,
    )


def test_absolute_trend_filter_and_breadth():
    returns = make_window_returns()
    eligible = absolute_trend_filter(returns, 126)

    assert eligible.all()
    assert trend_breadth(eligible) == 1.0
    assert defense_scale(1.0) == 1.0


def test_select_distinct_assets_respects_bucket_cap():
    scores = pd.Series({"A": 5.0, "B": 4.0, "C": 3.0, "D": 2.0})
    eligible = pd.Series({"A": True, "B": True, "C": True, "D": True})
    bucket_map = {"A": "equity", "B": "equity", "C": "bond", "D": "gold"}

    selected = select_distinct_assets(scores, eligible, bucket_map, max_assets=3, max_per_bucket=1)

    assert selected == ["A", "C", "D"]


def test_generate_target_weights_returns_cash_friendly_weights():
    window_returns = make_window_returns()

    weights, diagnostics = generate_target_weights(window_returns)

    assert weights
    assert sum(weights.values()) <= 1.0
    assert diagnostics["breadth"] > 0.0
    assert diagnostics["gross_scale"] > 0.0


def test_generate_target_weights_can_park_residual_weight():
    window_returns = make_window_returns()
    params = with_overrides(
        DEFAULT_PARAMETERS,
        parking_asset="CASH",
        max_assets=1,
    )

    callback = build_strategy_callback(params)
    weights, diagnostics = callback(window_returns)

    assert "CASH" in weights
    assert round(sum(weights.values()), 8) == 1.0
    assert diagnostics["parking_asset"] == "CASH"
