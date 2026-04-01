from __future__ import annotations

import numpy as np
import pandas as pd


def select_distinct_assets(
    scores: pd.Series,
    eligible_assets: pd.Series,
    bucket_map: dict[str, str],
    max_assets: int,
    max_per_bucket: int,
) -> list[str]:
    ranked = scores[eligible_assets].dropna().sort_values(ascending=False)
    ranked = ranked[ranked > 0]

    selected: list[str] = []
    bucket_counts: dict[str, int] = {}
    for asset in ranked.index:
        bucket = bucket_map.get(asset, asset)
        if bucket_counts.get(bucket, 0) >= max_per_bucket:
            continue
        selected.append(asset)
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        if len(selected) >= max_assets:
            break
    return selected


def allocate(
    scores: pd.Series,
    vol_vals: pd.Series,
    eligible_assets: pd.Series,
    target_vol: float,
    max_assets: int,
    max_weight: float,
    gross_scale: float,
    bucket_map: dict[str, str],
    max_per_bucket: int,
) -> dict[str, float]:
    selected = select_distinct_assets(
        scores=scores,
        eligible_assets=eligible_assets,
        bucket_map=bucket_map,
        max_assets=max_assets,
        max_per_bucket=max_per_bucket,
    )
    if not selected:
        return {}

    vol = vol_vals[selected].replace(0.0, np.nan).dropna()
    if vol.empty:
        return {}

    inverse_vol = 1.0 / (vol + 1e-8)
    base_weights = inverse_vol / inverse_vol.sum()

    estimated_portfolio_vol = float(np.sqrt(np.sum((base_weights * vol) ** 2)))
    vol_scale = min(1.0, target_vol / (estimated_portfolio_vol + 1e-8))
    scaled_weights = (base_weights * vol_scale * gross_scale).clip(upper=max_weight)

    return {
        asset: float(weight)
        for asset, weight in scaled_weights.items()
        if weight > 1e-12
    }
