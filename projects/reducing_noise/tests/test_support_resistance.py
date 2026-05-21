from __future__ import annotations

import numpy as np
import pandas as pd

from support_resistance import TouchConfig, accuracy_over_time, find_touch_events


def test_touch_detection_labels_floor_bounce_with_future_only_outcome() -> None:
    index = pd.date_range("2020-01-01", periods=8)
    residual = pd.Series([0.0, -1.0, 1.0, 0.01, 0.15, 0.35, 0.50, 0.60], index=index)
    fft_component = pd.Series(0.0, index=index)
    prices = np.exp(fft_component + residual)

    events = find_touch_events(
        prices,
        fft_component,
        residual,
        config=TouchConfig(
            touch_threshold=0.25,
            prior_distance_threshold=0.75,
            residual_vol_window=3,
            prior_lookback=3,
            horizons=(2,),
            accuracy_horizon=2,
        ),
    )

    assert not events.empty
    first = events.iloc[0]
    assert first["side"] == "floor"
    assert first["prediction_correct"]
    assert first["outcome"] == "floor_bounce"


def test_touch_detection_labels_ceiling_rejection() -> None:
    index = pd.date_range("2020-01-01", periods=8)
    residual = pd.Series([0.0, 1.0, -1.0, -0.01, -0.15, -0.35, -0.50, -0.60], index=index)
    fft_component = pd.Series(0.0, index=index)
    prices = np.exp(fft_component + residual)

    events = find_touch_events(
        prices,
        fft_component,
        residual,
        config=TouchConfig(
            touch_threshold=0.25,
            prior_distance_threshold=0.75,
            residual_vol_window=3,
            prior_lookback=3,
            horizons=(2,),
            accuracy_horizon=2,
        ),
    )

    assert not events.empty
    first = events.iloc[0]
    assert first["side"] == "ceiling"
    assert first["prediction_correct"]
    assert first["outcome"] == "ceiling_rejection"


def test_accuracy_over_time_computes_cumulative_and_rolling_hit_rates() -> None:
    events = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=5),
            "side": ["floor", "floor", "ceiling", "ceiling", "floor"],
            "future_return_2d": [0.01, -0.01, -0.02, 0.02, 0.03],
        }
    )

    accuracy = accuracy_over_time(events, horizon=2, rolling_event_window=3)

    assert accuracy["prediction_correct"].tolist() == [True, False, True, False, True]
    assert accuracy["cumulative_hit_rate"].iloc[-1] == 0.6
    assert accuracy["rolling_hit_rate"].iloc[-1] == 2 / 3
