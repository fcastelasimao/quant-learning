from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TouchConfig:
    touch_threshold: float = 0.25
    prior_distance_threshold: float = 0.75
    residual_vol_window: int = 128
    prior_lookback: int = 20
    horizons: tuple[int, ...] = (5, 10, 20)
    accuracy_horizon: int = 20
    rolling_event_window: int = 25


def compute_residual_zscore(residual: pd.Series, window: int) -> pd.Series:
    min_periods = min(window, max(3, window // 4))
    volatility = residual.rolling(window, min_periods=min_periods).std()
    return residual / volatility.replace(0.0, np.nan)


def find_touch_events(
    prices: pd.Series,
    fft_component: pd.Series,
    residual: pd.Series,
    *,
    config: TouchConfig = TouchConfig(),
) -> pd.DataFrame:
    """Find possible FFT support/resistance touches.

    A touch is an entry into the near-line zone after recently being farther away.
    Event labels use future returns only after the event date.
    """
    frame = pd.DataFrame(
        {
            "price": prices.astype(float).sort_index(),
            "fft_component": fft_component,
            "residual": residual,
        }
    ).dropna()
    if frame.empty:
        return pd.DataFrame()

    frame["residual_z"] = compute_residual_zscore(frame["residual"], config.residual_vol_window)
    frame["abs_residual_z"] = frame["residual_z"].abs()
    frame["previous_abs_residual_z"] = frame["abs_residual_z"].shift(1)
    frame["previous_residual"] = frame["residual"].shift(1)

    recent_far = (
        frame["abs_residual_z"]
        .shift(1)
        .rolling(config.prior_lookback, min_periods=1)
        .max()
        >= config.prior_distance_threshold
    )
    entered_touch_zone = (
        (frame["abs_residual_z"] <= config.touch_threshold)
        & (frame["previous_abs_residual_z"] > config.touch_threshold)
    )
    touch_mask = recent_far & entered_touch_zone

    for horizon in config.horizons:
        frame[f"future_return_{horizon}d"] = frame["price"].shift(-horizon) / frame["price"] - 1.0

    events = frame.loc[touch_mask].copy()
    if events.empty:
        return events.reset_index(names="date")

    previous_sign = np.sign(events["previous_residual"])
    current_sign = np.sign(events["residual"])
    side_sign = previous_sign.replace(0, np.nan).fillna(current_sign)
    events["side"] = np.where(side_sign >= 0, "floor", "ceiling")
    events["crossed_line"] = (previous_sign != 0) & (current_sign != 0) & (previous_sign != current_sign)

    future = events[f"future_return_{config.accuracy_horizon}d"]
    floor_bounce = (events["side"] == "floor") & (future > 0)
    ceiling_rejection = (events["side"] == "ceiling") & (future < 0)
    upward_break = events["crossed_line"] & (current_sign > 0) & (future > 0)
    downward_break = events["crossed_line"] & (current_sign < 0) & (future < 0)

    events["prediction_correct"] = np.where(events["side"] == "floor", future > 0, future < 0)
    events["outcome"] = np.select(
        [
            upward_break,
            downward_break,
            floor_bounce,
            ceiling_rejection,
            events["side"] == "floor",
            events["side"] == "ceiling",
        ],
        [
            "upward_break",
            "downward_break",
            "floor_bounce",
            "ceiling_rejection",
            "failed_floor_touch",
            "failed_ceiling_touch",
        ],
        default="unknown",
    )

    keep_columns = [
        "price",
        "fft_component",
        "residual",
        "residual_z",
        "previous_residual",
        "previous_abs_residual_z",
        "side",
        "crossed_line",
        "prediction_correct",
        "outcome",
        *[f"future_return_{horizon}d" for horizon in config.horizons],
    ]
    return events[keep_columns].reset_index(names="date")


def summarize_touch_events(
    prices: pd.Series,
    events: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = (5, 10, 20),
) -> pd.DataFrame:
    prices = prices.astype(float).sort_index()
    unconditional = {
        horizon: prices.shift(-horizon) / prices - 1.0
        for horizon in horizons
    }

    rows: list[dict[str, float | str | int]] = []
    for side in ["floor", "ceiling", "all"]:
        side_events = events if side == "all" else events[events["side"] == side]
        row: dict[str, float | str | int] = {"side": side, "events": int(len(side_events))}
        if side_events.empty:
            for horizon in horizons:
                row[f"avg_future_return_{horizon}d"] = np.nan
                row[f"hit_rate_{horizon}d"] = np.nan
                row[f"unconditional_hit_rate_{horizon}d"] = np.nan
            row["bounce_or_rejection_rate"] = np.nan
            row["break_rate"] = np.nan
            rows.append(row)
            continue

        row["bounce_or_rejection_rate"] = float(
            side_events["outcome"].isin(["floor_bounce", "ceiling_rejection"]).mean()
        )
        row["break_rate"] = float(side_events["outcome"].isin(["upward_break", "downward_break"]).mean())

        for horizon in horizons:
            event_returns = side_events[f"future_return_{horizon}d"].dropna()
            if side == "floor":
                hits = event_returns > 0
                unconditional_hits = unconditional[horizon].dropna() > 0
            elif side == "ceiling":
                hits = event_returns < 0
                unconditional_hits = unconditional[horizon].dropna() < 0
            else:
                aligned = side_events.dropna(subset=[f"future_return_{horizon}d"])
                event_returns = aligned[f"future_return_{horizon}d"]
                hits = np.where(aligned["side"] == "floor", event_returns > 0, event_returns < 0)
                unconditional_hits = pd.Series(dtype=bool)

            row[f"avg_future_return_{horizon}d"] = float(event_returns.mean()) if len(event_returns) else np.nan
            row[f"hit_rate_{horizon}d"] = float(pd.Series(hits).mean()) if len(event_returns) else np.nan
            row[f"unconditional_hit_rate_{horizon}d"] = (
                float(unconditional_hits.mean()) if not unconditional_hits.empty else np.nan
            )
        rows.append(row)

    return pd.DataFrame(rows)


def accuracy_over_time(
    events: pd.DataFrame,
    *,
    horizon: int = 20,
    rolling_event_window: int = 25,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["date", "side", "prediction_correct", "cumulative_hit_rate", "rolling_hit_rate"])

    future_col = f"future_return_{horizon}d"
    accuracy = events.dropna(subset=[future_col]).copy()
    accuracy = accuracy.sort_values("date").reset_index(drop=True)
    accuracy["prediction_correct"] = np.where(
        accuracy["side"] == "floor",
        accuracy[future_col] > 0,
        accuracy[future_col] < 0,
    )
    accuracy["cumulative_hit_rate"] = accuracy["prediction_correct"].expanding().mean()
    accuracy["rolling_hit_rate"] = accuracy["prediction_correct"].rolling(
        rolling_event_window,
        min_periods=min(5, rolling_event_window),
    ).mean()
    return accuracy[
        [
            "date",
            "side",
            "prediction_correct",
            "cumulative_hit_rate",
            "rolling_hit_rate",
            future_col,
        ]
    ]
