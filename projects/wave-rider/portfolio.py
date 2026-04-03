from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class RebalanceResult:
    turnover_value: float
    transaction_cost: float
    executed: bool
    max_weight_change: float


class Portfolio:
    def __init__(self, initial_capital: float):
        self.positions: dict[str, float] = {}
        self.cash = float(initial_capital)
        self.peak = float(initial_capital)
        self.total_turnover = 0.0
        self.total_costs = 0.0

    @property
    def value(self) -> float:
        return self.cash + sum(self.positions.values())

    def drawdown(self) -> float:
        if self.peak <= 0:
            return 0.0
        return (self.peak - self.value) / self.peak

    def exposure(self) -> float:
        if self.value <= 0:
            return 0.0
        return sum(self.positions.values()) / self.value

    def weights(self) -> dict[str, float]:
        current_value = self.value
        if current_value <= 0:
            return {}
        return {
            asset: position_value / current_value
            for asset, position_value in self.positions.items()
            if position_value > 1e-12
        }

    def apply_returns(self, returns_row: pd.Series) -> None:
        for asset in list(self.positions):
            asset_return = returns_row.get(asset, 0.0)
            if pd.isna(asset_return):
                asset_return = 0.0
            self.positions[asset] *= 1.0 + float(asset_return)
        self.peak = max(self.peak, self.value)

    def rebalance(
        self,
        target_weights: dict[str, float],
        transaction_cost_pct: float = 0.0,
        rebalance_threshold: float = 0.0,
    ) -> RebalanceResult:
        clean_weights = {
            asset: float(weight)
            for asset, weight in target_weights.items()
            if float(weight) > 1e-12
        }
        total_weight = sum(clean_weights.values())
        if total_weight > 1.0 + 1e-8:
            raise ValueError(f"Target weights must sum to <= 1.0, got {total_weight:.4f}")

        starting_value = self.value
        assets = set(self.positions) | set(clean_weights)
        current_values = {asset: self.positions.get(asset, 0.0) for asset in assets}
        target_values = {
            asset: starting_value * clean_weights.get(asset, 0.0) for asset in assets
        }
        current_weights = {
            asset: (current_values[asset] / starting_value) if starting_value > 0 else 0.0
            for asset in assets
        }
        target_cash_weight = 1.0 - total_weight
        current_cash_weight = self.cash / starting_value if starting_value > 0 else 0.0
        max_weight_change = max(
            [
                abs(clean_weights.get(asset, 0.0) - current_weights.get(asset, 0.0))
                for asset in assets
            ]
            + [abs(target_cash_weight - current_cash_weight)]
        )
        if max_weight_change <= rebalance_threshold:
            return RebalanceResult(
                turnover_value=0.0,
                transaction_cost=0.0,
                executed=False,
                max_weight_change=max_weight_change,
            )

        turnover_value = sum(
            abs(target_values[asset] - current_values[asset]) for asset in assets
        )
        transaction_cost = turnover_value * transaction_cost_pct
        investable_value = max(0.0, starting_value - transaction_cost)

        self.positions = {
            asset: investable_value * weight for asset, weight in clean_weights.items()
        }
        self.cash = investable_value - sum(self.positions.values())
        self.total_turnover += turnover_value
        self.total_costs += transaction_cost
        self.peak = max(self.peak, self.value)

        return RebalanceResult(
            turnover_value=turnover_value,
            transaction_cost=transaction_cost,
            executed=True,
            max_weight_change=max_weight_change,
        )

    def reduce_risk(self, exposure_scale: float) -> None:
        if not 0.0 <= exposure_scale <= 1.0:
            raise ValueError("Exposure scale must be between 0 and 1.")

        released_cash = 0.0
        for asset, position_value in list(self.positions.items()):
            new_value = position_value * exposure_scale
            released_cash += position_value - new_value
            self.positions[asset] = new_value
        self.cash += released_cash
