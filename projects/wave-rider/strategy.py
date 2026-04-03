from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import pandas as pd

import config
from allocation import allocate, select_distinct_assets
from signals import (
    absolute_trend_filter,
    annualized_volatility,
    blended_momentum_score,
    defense_scale,
    hmm_defense_scale,
    trend_breadth,
)


@dataclass(frozen=True)
class StrategyParameters:
    absolute_momentum_window: int
    relative_momentum_windows: tuple[int, ...]
    relative_momentum_weights: tuple[float, ...]
    vol_window: int
    max_assets: int
    max_weight: float
    max_per_bucket: int
    target_vol: float
    defense_breadth_thresholds: tuple[tuple[float, float], ...]
    parking_asset: str | None
    use_hmm_defense: bool = False


DEFAULT_PARAMETERS = StrategyParameters(
    absolute_momentum_window=config.ABSOLUTE_MOMENTUM_WINDOW,
    relative_momentum_windows=config.RELATIVE_MOMENTUM_WINDOWS,
    relative_momentum_weights=config.RELATIVE_MOMENTUM_WEIGHTS,
    vol_window=config.VOL_WINDOW,
    max_assets=config.MAX_ASSETS,
    max_weight=config.MAX_WEIGHT,
    max_per_bucket=config.MAX_PER_BUCKET,
    target_vol=config.TARGET_VOL,
    defense_breadth_thresholds=config.DEFENSE_BREADTH_THRESHOLDS,
    parking_asset=config.PARKING_ASSET,
    use_hmm_defense=False,
)

def with_overrides(params: StrategyParameters, **overrides: object) -> StrategyParameters:
    return replace(params, **overrides)


AGGRESSIVE_PARAMETERS = DEFAULT_PARAMETERS
DEFENSIVE_PARAMETERS = with_overrides(
    DEFAULT_PARAMETERS,
    defense_breadth_thresholds=config.ALTERNATE_DEFENSE_BREADTH_THRESHOLDS,
)


def build_strategy_callback(
    params: StrategyParameters,
    strategy_assets: list[str] | None = None,
    asset_buckets: dict[str, str] | None = None,
    regime_detector: Optional[object] = None,
    equity_prices: Optional[pd.Series] = None,
    vix_prices: Optional[pd.Series] = None,
):
    """
    Build the strategy callback that generates target weights.

    Parameters
    ----------
    regime_detector : fitted RegimeDetector instance (only used if params.use_hmm_defense)
    equity_prices   : full daily close prices for a broad equity index (e.g. SPY)
                      used by the regime detector to compute features
    vix_prices      : VIX close prices (optional, falls back to realised vol proxy)
    """
    active_assets = strategy_assets or config.STRATEGY_ASSETS
    bucket_map = asset_buckets or config.ASSET_BUCKETS

    def generate_target_weights(window_returns: pd.DataFrame) -> tuple[dict[str, float], dict[str, object]]:
        asset_returns = window_returns[active_assets]

        eligible_assets = absolute_trend_filter(
            asset_returns,
            params.absolute_momentum_window,
        )
        momentum_scores = blended_momentum_score(
            asset_returns,
            params.relative_momentum_windows,
            params.relative_momentum_weights,
        )
        vol_values = annualized_volatility(asset_returns, params.vol_window)

        breadth = trend_breadth(eligible_assets)

        # Determine gross_scale from HMM regime or breadth thresholds
        hmm_regime_label = None
        hmm_gross_scale = None
        if params.use_hmm_defense and regime_detector is not None and equity_prices is not None:
            current_date = window_returns.index[-1]
            eq_slice = equity_prices.loc[:current_date]
            vix_slice = vix_prices.loc[:current_date] if vix_prices is not None else None
            regime_probs = regime_detector.predict(eq_slice, vix_slice)
            if not regime_probs.empty:
                latest_probs = regime_probs.iloc[-1]
                gross_scale = hmm_defense_scale(latest_probs)
                hmm_regime_label = str(latest_probs.idxmax())
                hmm_gross_scale = gross_scale
            else:
                gross_scale = defense_scale(breadth, params.defense_breadth_thresholds)
        else:
            gross_scale = defense_scale(breadth, params.defense_breadth_thresholds)

        selected_assets = select_distinct_assets(
            scores=momentum_scores,
            eligible_assets=eligible_assets,
            bucket_map=bucket_map,
            max_assets=params.max_assets,
            max_per_bucket=params.max_per_bucket,
        )

        weights = allocate(
            scores=momentum_scores,
            vol_vals=vol_values,
            eligible_assets=eligible_assets,
            target_vol=params.target_vol,
            max_assets=params.max_assets,
            max_weight=params.max_weight,
            gross_scale=gross_scale,
            bucket_map=bucket_map,
            max_per_bucket=params.max_per_bucket,
        )
        residual_weight = max(0.0, 1.0 - sum(weights.values()))
        parking_weight = 0.0
        if params.parking_asset is not None and residual_weight > 1e-12:
            weights[params.parking_asset] = weights.get(params.parking_asset, 0.0) + residual_weight
            parking_weight = residual_weight

        diagnostics: dict[str, object] = {
            "breadth": breadth,
            "gross_scale": gross_scale,
            "gross_exposure": float(sum(weights.values())),
            "selected_assets": ",".join(selected_assets),
            "parking_asset": params.parking_asset or "cash",
            "parking_weight": parking_weight,
        }

        if hmm_regime_label is not None:
            diagnostics["hmm_regime"] = hmm_regime_label
            diagnostics["hmm_gross_scale"] = hmm_gross_scale

        for asset, value in momentum_scores.sort_values(ascending=False).items():
            diagnostics[f"score_{asset.lower()}"] = float(value) if pd.notna(value) else float("nan")
        for asset, value in eligible_assets.items():
            diagnostics[f"eligible_{asset.lower()}"] = bool(value) if pd.notna(value) else False
        for asset, value in vol_values.items():
            diagnostics[f"vol_{asset.lower()}"] = float(value) if pd.notna(value) else float("nan")

        return weights, diagnostics

    return generate_target_weights


def generate_target_weights(window_returns: pd.DataFrame) -> tuple[dict[str, float], dict[str, object]]:
    return build_strategy_callback(DEFAULT_PARAMETERS)(window_returns)


def warmup_bars(params: StrategyParameters = DEFAULT_PARAMETERS) -> int:
    return max(
        params.absolute_momentum_window,
        params.vol_window,
        max(params.relative_momentum_windows),
    )
