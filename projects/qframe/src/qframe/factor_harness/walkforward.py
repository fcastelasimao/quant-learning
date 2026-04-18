"""
Walk-forward validation harness.

Rules (non-negotiable per CLAUDE.md):
- Expanding window only. No re-fit on OOS data.
- No look-ahead bias. Factor at date t uses only data available at t.
- Net-of-cost IC is the primary result metric, not gross IC.

Usage:
    from qframe.factor_harness.walkforward import WalkForwardValidator

    def my_factor(prices: pd.DataFrame) -> pd.DataFrame:
        # returns (dates x tickers) factor values
        ...

    validator = WalkForwardValidator(
        factor_fn=my_factor,
        oos_start="2018-01-01",
    )
    result = validator.run(prices_df)
    print(result.summary())
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from qframe.factor_harness import DEFAULT_OOS_START
from qframe.factor_harness.ic import (
    compute_ic,
    compute_icir,
    compute_ic_decay,
    compute_slow_icir,
    estimate_ic_halflife,
)
from qframe.factor_harness.costs import (
    CostParams,
    DEFAULT_COST_PARAMS,
    compute_turnover,
    net_ic,
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardResult:
    """
    Container for all walk-forward validation outputs.

    Attributes:
        ic_series:      Daily IC series (gross), full OOS period.
        icir_series:    Rolling ICIR series over OOS period.
        net_ic_series:  Net-of-cost IC series.
        decay_df:       IC decay curve — mean IC at every horizon 1..63 days.
        ic_halflife:    Estimated IC half-life in days.
        slow_icir_21:   ICIR computed on non-overlapping 21-day windows (honest monthly).
        slow_icir_63:   ICIR computed on non-overlapping 63-day windows (honest quarterly).
        weights:        Portfolio weights used (dates x tickers).
        turnover:       One-way turnover series.
        oos_start:      OOS period start date.
        oos_end:        OOS period end date.
        cost_params:    CostParams used.
        horizon:        Primary evaluation horizon in days.
    """
    ic_series: pd.Series
    icir_series: pd.Series
    net_ic_series: pd.Series
    decay_df: pd.DataFrame
    ic_halflife: float
    slow_icir_21: float
    slow_icir_63: float
    weights: pd.DataFrame
    turnover: pd.Series
    oos_start: str
    oos_end: str
    cost_params: CostParams
    horizon: int

    def summary(self) -> dict:
        """Return a flat dict of scalar metrics suitable for logging to SQLite."""
        import json
        oos_ic = self.ic_series.loc[self.oos_start:self.oos_end].dropna()
        oos_net_ic = self.net_ic_series.loc[self.oos_start:self.oos_end].dropna()
        oos_icir = self.icir_series.loc[self.oos_start:self.oos_end].dropna()
        oos_to = self.turnover.loc[self.oos_start:self.oos_end].dropna()

        # Annualised Sharpe of IC series (treat daily IC as daily P&L proxy)
        sharpe = (oos_ic.mean() / oos_ic.std(ddof=1) * np.sqrt(252)) if len(oos_ic) > 1 else np.nan

        decay = self.decay_df

        # Full IC decay curve as JSON for notebook plotting
        decay_json = json.dumps({
            str(h): round(float(v), 6)
            for h, v in decay["mean_ic"].items()
            if np.isfinite(float(v))
        })

        # Sparse key horizons (for backward-compatible DB columns)
        def _h(h: int) -> float:
            return float(decay.loc[h, "mean_ic"]) if h in decay.index else np.nan

        return {
            "ic": float(oos_ic.mean()) if len(oos_ic) > 0 else np.nan,
            "icir": float(oos_icir.iloc[-1]) if len(oos_icir) > 0 else np.nan,
            "net_ic": float(oos_net_ic.mean()) if len(oos_net_ic) > 0 else np.nan,
            "sharpe": float(sharpe),
            "turnover": float(oos_to.mean()) if len(oos_to) > 0 else np.nan,
            "decay_halflife": float(self.ic_halflife),
            "slow_icir_21": float(self.slow_icir_21) if np.isfinite(self.slow_icir_21) else np.nan,
            "slow_icir_63": float(self.slow_icir_63) if np.isfinite(self.slow_icir_63) else np.nan,
            "ic_horizon_1": _h(1),
            "ic_horizon_5": _h(5),
            "ic_horizon_21": _h(21),
            "ic_horizon_63": _h(63),
            "ic_decay_json": decay_json,
            "oos_start": self.oos_start,
            "oos_end": self.oos_end,
            "cost_bps": self.cost_params.spread_bps,
        }


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class WalkForwardValidator:
    """
    Walk-forward validation harness for cross-sectional factors.

    The harness:
    1. Calls factor_fn on the FULL price/return history to compute factor values.
       It is the caller's responsibility to ensure factor_fn uses only
       expanding-window computations (no future data).
    2. Splits into IS [start, oos_start) and OOS [oos_start, end].
    3. Evaluates IC, ICIR, IC decay, turnover, and net-of-cost IC on the OOS period.

    Args:
        factor_fn:   Callable(prices: pd.DataFrame) -> pd.DataFrame
                     Must return (dates x tickers) factor values.
                     NaN = stock not in universe at that date.
        oos_start:   Start of the out-of-sample evaluation period ('YYYY-MM-DD').
        horizon:     Primary holding horizon in days (default 1).
        icir_window: Rolling window for ICIR computation (default 63).
        cost_params: CostParams for net-of-cost IC (default DEFAULT_COST_PARAMS).
        min_stocks:  Minimum stocks per cross-section for valid IC (default 10).
    """

    def __init__(
        self,
        factor_fn: Callable[[pd.DataFrame], pd.DataFrame],
        oos_start: str = DEFAULT_OOS_START,
        horizon: int = 1,
        icir_window: int = 63,
        cost_params: CostParams = DEFAULT_COST_PARAMS,
        min_stocks: int = 10,
        adv_df: pd.DataFrame | None = None,
        portfolio_nav: float = 1e6,
    ):
        """
        Args:
            adv_df:         Optional (dates x tickers) DataFrame of dollar ADV.
                            When provided, per-stock ADV-weighted market impact is used
                            instead of the fixed `cost_params.adv_fraction`. This is the
                            Phase 3 Gate 0 upgrade for more accurate cost estimation.
            portfolio_nav:  Portfolio NAV in dollars, used with adv_df to compute
                            per-stock position sizes for ADV-fraction calculation.
        """
        self.factor_fn = factor_fn
        self.oos_start = oos_start
        self.horizon = horizon
        self.icir_window = icir_window
        self.cost_params = cost_params
        self.min_stocks = min_stocks
        self.adv_df = adv_df
        self.portfolio_nav = portfolio_nav

    def run(self, prices: pd.DataFrame) -> WalkForwardResult:
        """
        Run the walk-forward validation.

        Args:
            prices: pd.DataFrame (dates x tickers) of adjusted close prices,
                    sorted ascending by date.

        Returns:
            WalkForwardResult with all metrics populated.
        """
        prices = prices.sort_index()

        # 1. Compute returns
        returns = prices.pct_change()

        # 2. Compute factor values — caller is responsible for no look-ahead
        factor_df = self.factor_fn(prices)

        # 3. Compute signal weights (cross-sectional z-score → long/short rank weights)
        weights = self._rank_weights(factor_df)

        # 4. IC series (full history, OOS will be sliced in summary)
        ic_series = compute_ic(
            factor_df, returns,
            horizon=self.horizon,
            min_stocks=self.min_stocks,
        )

        # 5. ICIR
        icir_series = compute_icir(ic_series, window=self.icir_window)

        # 6. Turnover
        turnover = compute_turnover(weights)

        # 7. Net-of-cost IC (pass weights so short-borrow cost uses actual short fraction;
        #    pass adv_df for per-stock ADV-weighted impact when available)
        net_ic_series = net_ic(
            ic_series, turnover,
            params=self.cost_params,
            horizon=self.horizon,
            weights=weights,
            adv_df=self.adv_df,
            portfolio_nav=self.portfolio_nav,
        )

        # 8. IC decay curve — full daily 1-63 day curve, OOS dates only.
        # Passing oos_start keeps forward-return computation on the full returns
        # index (avoids boundary NaNs) while restricting IC evaluation to OOS.
        decay_df = compute_ic_decay(
            factor_df, returns,
            horizons=list(range(1, 64)),   # every day 1..63 for smooth curve
            min_stocks=self.min_stocks,
            oos_start=self.oos_start,
        )

        # 9. IC half-life
        ic_halflife = estimate_ic_halflife(decay_df)

        # 10. Slow ICIR at quarterly horizon (non-overlapping windows)
        slow_icir_63 = compute_slow_icir(
            factor_df, returns,
            horizon=63, oos_start=self.oos_start,
        )
        slow_icir_21 = compute_slow_icir(
            factor_df, returns,
            horizon=21, oos_start=self.oos_start,
        )

        oos_end = str(prices.index[-1].date())

        return WalkForwardResult(
            ic_series=ic_series,
            icir_series=icir_series,
            net_ic_series=net_ic_series,
            decay_df=decay_df,
            ic_halflife=ic_halflife,
            slow_icir_21=slow_icir_21,
            slow_icir_63=slow_icir_63,
            weights=weights,
            turnover=turnover,
            oos_start=self.oos_start,
            oos_end=oos_end,
            cost_params=self.cost_params,
            horizon=self.horizon,
        )

    @staticmethod
    def _rank_weights(factor_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert factor values into long/short portfolio weights via cross-sectional
        rank normalisation (vectorized NumPy implementation).

        At each date:
          - Rank stocks by factor value (ascending, average ties)
          - Normalise ranks to [-0.5, +0.5]
          - Subtract cross-sectional mean so weights sum to 0 (dollar-neutral)
          - Rescale so abs(weights).sum() = 1

        Returns:
            pd.DataFrame (dates x tickers) of portfolio weights.
        """
        # Vectorized path — replaces the previous apply(axis=1) Python loop
        # which made ~n_dates individual Python calls.
        ranked = factor_df.rank(axis=1, method="average", na_option="keep").values  # (dates, tickers)
        n_valid = np.sum(~np.isnan(ranked), axis=1, keepdims=True)  # (dates, 1)

        # Normalise to [-0.5, +0.5]: w = (rank - 1) / (n - 1) - 0.5
        # Guard against n_valid == 1 (division by zero) — those rows become NaN below
        with np.errstate(invalid="ignore", divide="ignore"):
            w = (ranked - 1) / np.where(n_valid > 1, n_valid - 1, np.nan) - 0.5

        # Dollar-neutral: subtract cross-sectional mean (NaN-safe)
        row_mean = np.nanmean(w, axis=1, keepdims=True)
        w = w - row_mean

        # Unit gross exposure: divide by abs sum (NaN-safe)
        abs_sum = np.nansum(np.abs(w), axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            w = np.where(abs_sum > 0, w / abs_sum, w)

        # Rows with < 2 valid stocks → all NaN
        w[n_valid[:, 0] < 2] = np.nan

        return pd.DataFrame(w, index=factor_df.index, columns=factor_df.columns)
