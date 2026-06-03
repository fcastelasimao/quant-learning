"""
backtest.py
===========
The simulation engine.

  - run_backtest   simulates up to four strategies over a price history
                   (three always; 60/40 only when tlt_prices is provided)

Performance statistics (compute_*, StrategyStats, compute_stats) live in
engine/stats.py and are re-imported here for caller convenience.

Closed investigations (run_backtest_rolling_rp, run_backtest_with_overlay)
have been moved to failed_strategies/.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from .calendar import pandas_resample_frequency
from .stats import (
    StrategyStats,
    compute_cagr,
    compute_max_drawdown,
    compute_sharpe,
    compute_calmar,
    compute_max_drawdown_daily,
    compute_avg_drawdown,
    compute_max_drawdown_duration,
    compute_avg_recovery_time,
    compute_ulcer_index,
    compute_sortino,
    compute_stats,
)

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------
SIXTY_FORTY_EQUITY = 0.60
SIXTY_FORTY_BOND   = 0.40


# ---------------------------------------------------------------------------
# Rebalance policy (D.15)
# ---------------------------------------------------------------------------
# The "All Weather Value" strategy historically rebalanced to target weights
# every month, unconditionally. RebalancePolicy makes that one option among
# several. The DEFAULT (`monthly_unconditional`) reproduces the old behavior
# byte-for-byte — see tests/test_backtest_golden.py.
#
# Drift is measured at the engine's data cadence (month-end). Because this
# engine resamples to month-end before iterating, "check monthly then act on
# drift" and "act on drift" coincide here; they would differ only in a
# finer-grained (e.g. daily) engine. `monthly_check_then_drift` is kept as a
# distinct, explicitly-named policy so artifacts/manifests can record intent.

MODE_MONTHLY_UNCONDITIONAL = "monthly_unconditional"
MODE_DRIFT_RELATIVE = "drift_relative"
MODE_DRIFT_ABSOLUTE = "drift_absolute"
MODE_MONTHLY_CHECK_THEN_DRIFT = "monthly_check_then_drift"


@dataclass(frozen=True)
class RebalancePolicy:
    """When to rebalance the All Weather sleeve back to target weights.

    Use the factory classmethods rather than the constructor directly:

    * ``RebalancePolicy.monthly_unconditional()`` — rebalance every month
      (the historical default; zero regression).
    * ``RebalancePolicy.drift_relative(pct)`` — rebalance when any asset's
      weight deviates from target by more than ``pct`` *of its target weight*
      (e.g. ``0.20`` = 20% of target).
    * ``RebalancePolicy.drift_absolute(pp)`` — rebalance when any asset's
      weight deviates from target by more than ``pp`` *percentage points*
      expressed as a fraction (e.g. ``0.02`` = 2pp).
    * ``RebalancePolicy.monthly_check_then_drift(pct)`` — same trigger as
      ``drift_relative`` in this month-end engine; named distinctly for intent.

    ``should_rebalance`` is the only behavioral entry point.
    """

    mode: str = MODE_MONTHLY_UNCONDITIONAL
    relative_threshold: float = 0.0
    absolute_threshold: float = 0.0

    # -- factories ----------------------------------------------------------

    @classmethod
    def monthly_unconditional(cls) -> "RebalancePolicy":
        return cls(mode=MODE_MONTHLY_UNCONDITIONAL)

    @classmethod
    def drift_relative(cls, pct: float) -> "RebalancePolicy":
        if pct <= 0.0:
            raise ValueError("drift_relative threshold must be > 0.")
        return cls(mode=MODE_DRIFT_RELATIVE, relative_threshold=float(pct))

    @classmethod
    def drift_absolute(cls, pp: float) -> "RebalancePolicy":
        if pp <= 0.0:
            raise ValueError("drift_absolute threshold must be > 0.")
        return cls(mode=MODE_DRIFT_ABSOLUTE, absolute_threshold=float(pp))

    @classmethod
    def monthly_check_then_drift(cls, pct: float) -> "RebalancePolicy":
        if pct <= 0.0:
            raise ValueError("monthly_check_then_drift threshold must be > 0.")
        return cls(mode=MODE_MONTHLY_CHECK_THEN_DRIFT, relative_threshold=float(pct))

    # -- behavior -----------------------------------------------------------

    def should_rebalance(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
    ) -> bool:
        """Return True if this month's weights breach the policy.

        ``monthly_unconditional`` always rebalances. Drift modes rebalance when
        ANY asset breaches its threshold (drift to target restores all assets,
        consistent with research/rebalance_thresholds.py's full-on-breach rule).
        """
        if self.mode == MODE_MONTHLY_UNCONDITIONAL:
            return True
        if self.mode == MODE_DRIFT_ABSOLUTE:
            return any(
                abs(current_weights.get(t, 0.0) - w) > self.absolute_threshold
                for t, w in target_weights.items()
            )
        if self.mode in (MODE_DRIFT_RELATIVE, MODE_MONTHLY_CHECK_THEN_DRIFT):
            return any(
                w > 0.0
                and abs(current_weights.get(t, 0.0) - w) / w > self.relative_threshold
                for t, w in target_weights.items()
            )
        raise ValueError(f"Unknown rebalance mode: {self.mode!r}")

    @property
    def label(self) -> str:
        """Human/artifact label, e.g. ``drift_relative(0.2)``."""
        if self.mode == MODE_DRIFT_ABSOLUTE:
            return f"{self.mode}({self.absolute_threshold:g})"
        if self.mode in (MODE_DRIFT_RELATIVE, MODE_MONTHLY_CHECK_THEN_DRIFT):
            return f"{self.mode}({self.relative_threshold:g})"
        return self.mode


# ===========================================================================
# BACKTEST ENGINE
# ===========================================================================

def run_backtest(prices: pd.DataFrame,
                 benchmark_prices: pd.Series,
                 allocation: dict,
                 portfolio_value: float | None = None,
                 tlt_prices: pd.Series | None = None,
                 transaction_cost_pct: float = 0.0,
                 tax_drag_pct: float = 0.0,
                 rebalance_policy: "RebalancePolicy | None" = None) -> pd.DataFrame:
    """
    Simulate four strategies over a price history and return monthly values.

    Strategies simulated:
      1. Rebalanced portfolio  -- rebalances to `allocation` per rebalance_policy
                                  (default: every month, unconditionally)
      2. Buy & Hold            -- same starting weights, never rebalanced
      3. S&P 500 buy & hold    -- everything in SPY on day one, never touched
      4. 60/40 annually rebal. -- 60% SPY / 40% TLT, rebalanced at the start
                                  of each calendar year
                                  (only included when tlt_prices is provided)

    Parameters
    ----------
    prices           : daily price DataFrame, one column per ticker
    benchmark_prices : daily price Series for the benchmark (SPY)
    allocation       : dict of {ticker: weight}, weights must sum to 1.0
    portfolio_value  : starting value in USD (defaults to config value)
    tlt_prices       : daily price Series for TLT; enables the 60/40 strategy
    rebalance_policy : when to rebalance strategy 1 to target weights. Defaults
                       to RebalancePolicy.monthly_unconditional() — the historical
                       behavior (rebalance every month). Drift modes only trade
                       when an asset breaches its threshold; on months with no
                       rebalance no transaction cost is charged.

    Returns
    -------
    pd.DataFrame indexed by month-end date with columns:
        All Weather Value | Buy & Hold All Weather | S&P 500 Value
        60/40 Value (if tlt_prices provided)
        B&H <ticker> Weight (%) for each ticker
        Monthly Ret (%) columns for each strategy

    Notes
    -----
    "ME" resample = Month End. Groups daily prices to one price per month
    (the last trading day of each month). Requires pandas >= 2.2.
    Older pandas versions use "M" instead.

    .intersection() aligns the portfolio and benchmark to common dates so
    we are never comparing portfolio value and benchmark value on different
    dates due to data availability differences.

    .iloc[0] selects the first row by integer position (0-indexed),
    regardless of its date label.
    """
    if portfolio_value is None:
        portfolio_value = config.INITIAL_PORTFOLIO_VALUE

    if rebalance_policy is None:
        rebalance_policy = RebalancePolicy.monthly_unconditional()

    tickers = list(allocation.keys())

    resample_freq = pandas_resample_frequency(config.DATA_FREQUENCY)
    monthly     = prices[tickers].resample(resample_freq).last().dropna()
    bench       = benchmark_prices.resample(resample_freq).last().dropna()
    tlt_monthly = (tlt_prices.resample(resample_freq).last().dropna()
                   if tlt_prices is not None else None)

    common  = monthly.index.intersection(bench.index)
    if tlt_monthly is not None:
        common = common.intersection(tlt_monthly.index)
    monthly = monthly.loc[common]
    bench   = bench.loc[common]
    if tlt_monthly is not None:
        tlt_monthly = tlt_monthly.loc[common]

    if monthly.empty:
        raise ValueError("No overlapping monthly data found. Check date range.")

    first_row    = monthly.iloc[0]
    bench_shares = portfolio_value / float(bench.iloc[0])

    aw_holdings = {t: (portfolio_value * w) / float(first_row[t])
                   for t, w in allocation.items()}
    bh_holdings = {t: (portfolio_value * w) / float(first_row[t])
                   for t, w in allocation.items()}

    sixty_forty_spy = None
    sixty_forty_tlt = None
    sixty_forty_prev_year = None
    if tlt_monthly is not None:
        sixty_forty_spy = portfolio_value * SIXTY_FORTY_EQUITY / float(bench.iloc[0])
        sixty_forty_tlt = portfolio_value * SIXTY_FORTY_BOND / float(tlt_monthly.iloc[0])
        sixty_forty_prev_year = monthly.index[0].year

    # Note: this loop is intentionally iterative due to stateful monthly rebalancing
    # with transaction costs and 60/40 annual rebalance logic.
    records = []
    for date, row in monthly.iterrows():
        # Annual rebalance for 60/40: restore 60/40 split at start of each new year
        if sixty_forty_spy is not None and date.year != sixty_forty_prev_year:
            current_6040 = (sixty_forty_spy * float(bench.loc[date])
                            + sixty_forty_tlt * float(tlt_monthly.loc[date]))
            if transaction_cost_pct > 0:
                spy_val = sixty_forty_spy * float(bench.loc[date])
                tlt_val = sixty_forty_tlt * float(tlt_monthly.loc[date])
                trade_values_6040 = (abs(current_6040 * SIXTY_FORTY_EQUITY - spy_val)
                                     + abs(current_6040 * SIXTY_FORTY_BOND - tlt_val))
                current_6040 -= trade_values_6040 * transaction_cost_pct
            sixty_forty_spy = current_6040 * SIXTY_FORTY_EQUITY / float(bench.loc[date])
            sixty_forty_tlt = current_6040 * SIXTY_FORTY_BOND / float(tlt_monthly.loc[date])
            sixty_forty_prev_year = date.year

        raw_value = sum(sh * float(row[t]) for t, sh in aw_holdings.items())

        # Drift check (D.15). In monthly_unconditional mode this is always True
        # and the original code path runs unchanged.
        current_weights = (
            {t: (aw_holdings[t] * float(row[t])) / raw_value for t in tickers}
            if raw_value > 0 else {t: 0.0 for t in tickers}
        )
        do_rebalance = rebalance_policy.should_rebalance(current_weights, allocation)

        aw_value = raw_value

        # Tax drag on December close (annual deduction lands on year-end, not year-start)
        if tax_drag_pct > 0 and date.month == 12:
            aw_value *= (1 - tax_drag_pct)

        # Transaction cost is only incurred when we actually trade.
        if do_rebalance and transaction_cost_pct > 0:
            trade_values = sum(
                abs((aw_value * w) - (aw_holdings[t] * float(row[t])))
                for t, w in allocation.items()
            )
            aw_value -= trade_values * transaction_cost_pct

        bh_value  = sum(sh * float(row[t]) for t, sh in bh_holdings.items())
        spy_value = bench_shares * float(bench.loc[date])

        bh_weights = {t: (bh_holdings[t] * float(row[t])) / bh_value
                      for t in tickers}

        record = {
            "Date":                   date,
            "All Weather Value":      round(aw_value, 2),
            "Buy & Hold All Weather": round(bh_value, 2),
            "S&P 500 Value":          round(spy_value, 2),
        }

        if sixty_forty_spy is not None:
            sixty_forty_value = (sixty_forty_spy * float(bench.loc[date])
                                 + sixty_forty_tlt * float(tlt_monthly.loc[date]))
            record["60/40 Value"] = round(sixty_forty_value, 2)

        for t in tickers:
            record[f"B&H {t} Weight (%)"] = round(bh_weights[t] * 100, 1)

        records.append(record)

        # Rebalance to target weights when the policy fires. In
        # monthly_unconditional mode do_rebalance is always True, so this is the
        # original single path (byte-identical). When a drift policy skips a
        # rebalance, holdings carry forward unchanged — except that any value
        # change that did NOT come from a trade (i.e. December tax drag) must be
        # propagated into the holdings so it carries to next month.
        if do_rebalance:
            for t, w in allocation.items():
                aw_holdings[t] = (aw_value * w) / float(row[t])
        elif raw_value > 0 and aw_value != raw_value:
            scale = aw_value / raw_value
            for t in tickers:
                aw_holdings[t] *= scale

    df = pd.DataFrame(records).set_index("Date")
    for col in ["All Weather Value", "Buy & Hold All Weather", "S&P 500 Value"]:
        df[f"{col} Monthly Ret (%)"] = df[col].pct_change() * 100
    if "60/40 Value" in df.columns:
        df["60/40 Value Monthly Ret (%)"] = df["60/40 Value"].pct_change() * 100

    return df
