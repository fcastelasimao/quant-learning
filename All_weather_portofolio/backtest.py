"""
backtest.py
===========
The simulation engine. Contains:

  - StrategyStats    dataclass for holding per-strategy performance metrics
  - compute_cagr     shared stat helper
  - compute_max_drawdown
  - compute_sharpe
  - compute_calmar
  - run_backtest     simulates up to four strategies over a price history
                     (three always; 60/40 only when tlt_prices is provided)
  - compute_stats    computes StrategyStats for all strategies

This module is a pure simulation -- it knows nothing about real holdings,
file I/O, or user parameters beyond what is passed in as arguments.
The only import from config is INITIAL_PORTFOLIO_VALUE as a default,
which can always be overridden by the caller.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

import config


# ===========================================================================
# STATS DATACLASS
# ===========================================================================

@dataclass
class StrategyStats:
    """
    Holds performance statistics for a single strategy.

    Using a dataclass rather than a plain dict means:
      - No string key typos
      - No trailing-space hacks to avoid key collisions between strategies
      - Stats accessed by attribute (s.cagr) not fragile string (s["  CAGR (%) "])
      - Type-checked by static analysis tools
    """
    name:              str
    cagr:              float     # compound annual growth rate (%)
    max_drawdown:      float     # peak-to-trough decline (%, negative number)
    sharpe:            float     # annualised Sharpe ratio
    calmar:            float     # CAGR / |max drawdown| -- balanced risk/return
    final_value:       float     # portfolio value at end of period ($)
    period_years:      float     # length of the backtest period
    avg_drawdown:      float     # mean drawdown across all underwater periods (%)
    max_dd_duration:   int       # longest consecutive months below peak
    avg_recovery_time: float     # average months to recover from a drawdown
    ulcer_index:       float     # RMS of all drawdown percentages
    sortino:           float     # downside-only Sharpe ratio
    martin:            float     # CAGR / Ulcer Index — primary optimisation metric (smoother than Calmar)
    max_drawdown_daily: float    # MDD computed on daily price resolution (more accurate than monthly)


# ===========================================================================
# SHARED STAT HELPERS
# ===========================================================================

def compute_cagr(series: pd.Series, years: float) -> float:
    """
    Compound Annual Growth Rate as a percentage.

    Formula: (end / start) ^ (1 / years) - 1
    Example: $10k -> $20k over 10 years = 7.18% CAGR
    """
    return ((series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1) * 100


def compute_max_drawdown(series: pd.Series) -> float:
    """
    Maximum peak-to-trough decline as a percentage.

    Returns a negative number. -20.0 means the portfolio dropped 20% from
    its peak at some point. 0.0 means it never fell below its starting value.

    Formula: min( (value - running_peak) / running_peak ) * 100
    """
    peak = series.cummax()
    return ((series - peak) / peak).min() * 100


def compute_sharpe(monthly_ret_series: pd.Series,
                   rf_annual: float = 0.0) -> float:
    """
    Annualised Sharpe ratio from a series of monthly returns (as percentages).

    Sharpe = ((mean monthly return - rf_monthly) / std of monthly returns) * sqrt(12)
    rf_annual is converted to a monthly equivalent: (1 + rf_annual)^(1/12) - 1.

    Returns 0.0 if all returns are identical (zero volatility edge case).
    Set rf_annual=0.0 to reproduce pre-fix results.
    """
    r = monthly_ret_series.dropna() / 100
    if len(r) == 0 or r.std() < 1e-10:
        return 0.0
    rf_monthly = (1 + rf_annual) ** (1 / 12) - 1
    return ((r.mean() - rf_monthly) / r.std()) * np.sqrt(config.SHARPE_ANNUALISATION)



def compute_calmar(cagr: float, max_drawdown: float) -> float:
    """
    Calmar ratio = CAGR / |max drawdown|.

    Measures return per unit of drawdown accepted. Higher is better.
    A Calmar of 0.5 means you earn 0.5% of annual return for every 1%
    of maximum drawdown you accept.

    Used for reporting only. Not used as optimisation objective (Calmar is
    dominated by a single worst data point and creates a discontinuous
    landscape — see Martin ratio for the DE objective).

    Returns 0.0 if drawdown is zero (degenerate / no-loss case).
    """
    if max_drawdown == 0.0:
        return 0.0
    return cagr / abs(max_drawdown)


def compute_max_drawdown_daily(prices: pd.DataFrame,
                               allocation: dict) -> float:
    """
    Maximum drawdown computed at daily price resolution.

    Monthly MDD understates true drawdowns because it only sees month-end
    prices. A 20% intramonth drop that recovers by month-end is invisible
    to the monthly engine. Daily MDD captures the true worst-case experience.

    Simulates the same monthly rebalancing logic as run_backtest() but
    tracks portfolio value at every trading day rather than just month-ends.
    Between rebalancing dates, the portfolio drifts with daily prices.

    Returns a negative number (same convention as compute_max_drawdown).
    Returns 0.0 if prices are empty or allocation is empty.
    """
    tickers = list(allocation.keys())
    available = [t for t in tickers if t in prices.columns]
    if not available or prices.empty:
        return 0.0

    daily = prices[available].ffill().dropna()
    if daily.empty or len(daily) < 2:
        return 0.0

    # Month-end dates used as rebalance triggers
    monthly_dates = set(daily.resample(config.DATA_FREQUENCY).last().dropna().index)

    # Initialise holdings proportionally on first trading day
    first = daily.iloc[0]
    pv    = config.INITIAL_PORTFOLIO_VALUE
    holdings = {t: (pv * allocation[t]) / float(first[t]) for t in available}

    daily_values = []
    for date, row in daily.iterrows():
        pv = sum(holdings[t] * float(row[t]) for t in available)
        daily_values.append(pv)
        if date in monthly_dates:
            for t in available:
                holdings[t] = (pv * allocation[t]) / float(row[t])

    return compute_max_drawdown(pd.Series(daily_values))


def compute_avg_drawdown(series: pd.Series) -> float:
    """
    Average of all drawdown values (not just the maximum).
    Computed as the mean of (value - running_peak) / running_peak * 100
    across all time steps where the portfolio is below its peak.
    Returns 0.0 if the portfolio never draws down.
    Returns a negative number. Closer to 0 is better.
    """
    peak = series.cummax()
    dd_series = ((series - peak) / peak) * 100
    underwater = dd_series[dd_series < 0]
    if underwater.empty:
        return 0.0
    return round(underwater.mean(), 2)


def compute_max_drawdown_duration(series: pd.Series) -> int:
    """
    Maximum number of consecutive periods spent below a previous peak.
    Returns an integer (number of monthly periods).
    A value of 0 means the portfolio never drew down.
    """
    peak = series.cummax()
    underwater = (series < peak)
    max_duration = 0
    current = 0
    for u in underwater:
        if u:
            current += 1
            max_duration = max(max_duration, current)
        else:
            current = 0
    return max_duration


def compute_avg_recovery_time(series: pd.Series) -> float:
    """
    Average number of months to recover from each distinct drawdown episode.
    A drawdown episode starts when the portfolio falls below its peak and
    ends when it returns to or exceeds that peak.
    Returns 0.0 if no complete recovery episodes exist.
    """
    peak = series.cummax()
    underwater = (series < peak).values
    durations = []
    current = 0
    in_drawdown = False
    for u in underwater:
        if u:
            in_drawdown = True
            current += 1
        elif in_drawdown:
            durations.append(current)
            current = 0
            in_drawdown = False
    if not durations:
        return 0.0
    return round(float(np.mean(durations)), 1)


def compute_ulcer_index(series: pd.Series) -> float:
    """
    Ulcer Index — measures both depth and duration of drawdowns via RMS.
    Formula: sqrt(mean(drawdown_pct^2)) where drawdown_pct is the
    percentage decline from the running peak at each time step.
    Lower is better. Unlike max drawdown, this penalises extended
    periods of being moderately underwater, not just the single worst point.
    """
    peak = series.cummax()
    dd_pct = ((series - peak) / peak) * 100
    return round(float(np.sqrt((dd_pct ** 2).mean())), 4)


def compute_sortino(monthly_ret_series: pd.Series,
                    rf_annual: float = 0.0) -> float:
    """
    Sortino ratio — like Sharpe but only penalises downside volatility.
    Formula: ((mean monthly return - rf_monthly) / downside deviation) * sqrt(annualisation)
    Downside deviation uses only months where return < rf_monthly (not zero).
    rf_annual is converted to a monthly equivalent: (1 + rf_annual)^(1/12) - 1.
    Returns 0.0 if there are no below-threshold months or empty series.
    Set rf_annual=0.0 to reproduce pre-fix results.
    """
    r = monthly_ret_series.dropna() / 100
    if len(r) == 0:
        return 0.0
    rf_monthly = (1 + rf_annual) ** (1 / 12) - 1
    downside = r[r < rf_monthly]
    if len(downside) == 0 or downside.std() < 1e-10:
        return 0.0
    return round(((r.mean() - rf_monthly) / downside.std()) * np.sqrt(config.SHARPE_ANNUALISATION), 3)


# ===========================================================================
# BACKTEST ENGINE
# ===========================================================================

def run_backtest(prices: pd.DataFrame,
                 benchmark_prices: pd.Series,
                 allocation: dict,
                 portfolio_value: Optional[float] = None,
                 tlt_prices: Optional[pd.Series] = None,
                 transaction_cost_pct: float = 0.0,
                 tax_drag_pct: float = 0.0) -> pd.DataFrame:
    """
    Simulate four strategies over a price history and return monthly values.

    Strategies simulated:
      1. Rebalanced portfolio  -- rebalances to `allocation` every month
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

    tickers = list(allocation.keys())

    monthly     = prices[tickers].resample(config.DATA_FREQUENCY).last().dropna()
    bench       = benchmark_prices.resample(config.DATA_FREQUENCY).last().dropna()
    tlt_monthly = (tlt_prices.resample(config.DATA_FREQUENCY).last().dropna()
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
        sixty_forty_spy = portfolio_value * 0.60 / float(bench.iloc[0])
        sixty_forty_tlt = portfolio_value * 0.40 / float(tlt_monthly.iloc[0])
        sixty_forty_prev_year = monthly.index[0].year

    aw_prev_year = monthly.index[0].year

    records = []
    for date, row in monthly.iterrows():
        # Annual rebalance for 60/40: restore 60/40 split at start of each new year
        if sixty_forty_spy is not None and date.year != sixty_forty_prev_year:
            current_6040 = (sixty_forty_spy * float(bench.loc[date])
                            + sixty_forty_tlt * float(tlt_monthly.loc[date]))
            if transaction_cost_pct > 0:
                spy_val = sixty_forty_spy * float(bench.loc[date])
                tlt_val = sixty_forty_tlt * float(tlt_monthly.loc[date])
                trade_values_6040 = (abs(current_6040 * 0.60 - spy_val)
                                     + abs(current_6040 * 0.40 - tlt_val))
                current_6040 -= trade_values_6040 * transaction_cost_pct
            sixty_forty_spy = current_6040 * 0.60 / float(bench.loc[date])
            sixty_forty_tlt = current_6040 * 0.40 / float(tlt_monthly.loc[date])
            sixty_forty_prev_year = date.year

        aw_value  = sum(sh * float(row[t]) for t, sh in aw_holdings.items())

        if tax_drag_pct > 0 and date.year != aw_prev_year:
            aw_value *= (1 - tax_drag_pct)
            aw_prev_year = date.year

        if transaction_cost_pct > 0:
            trade_values = sum(
                abs((aw_value * w) - (aw_holdings[t] * float(row[t])))
                for t, w in allocation.items()
            )
            aw_value -= trade_values * transaction_cost_pct
            for t, w in allocation.items():
                aw_holdings[t] = (aw_value * w) / float(row[t])
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

        # Rebalance: restore target weights for the rebalanced strategy only
        # (skipped when transaction_cost_pct > 0 as rebalancing already
        # happened inside the cost block above)
        if transaction_cost_pct == 0.0:
            for t, w in allocation.items():
                aw_holdings[t] = (aw_value * w) / float(row[t])
        # Buy & Hold: do nothing -- holdings stay fixed

    df = pd.DataFrame(records).set_index("Date")
    for col in ["All Weather Value", "Buy & Hold All Weather", "S&P 500 Value"]:
        df[f"{col} Monthly Ret (%)"] = df[col].pct_change() * 100
    if "60/40 Value" in df.columns:
        df["60/40 Value Monthly Ret (%)"] = df["60/40 Value"].pct_change() * 100

    return df


# ===========================================================================
# SPY MOMENTUM OVERLAY
# ===========================================================================

def compute_overlay_signal(asset_prices: pd.Series,
                           threshold: float,
                           d_window: int,
                           reduce_pct: float) -> pd.Series:
    """
    Compute a daily allocation multiplier for any asset based on a
    trend-following drawdown-protection rule.

    Returns a Series aligned to asset_prices.index with values:
      1.0              — full position (overlay inactive)
      1.0 - reduce_pct — reduced position (overlay active)

    Exit conditions (ALL must hold simultaneously):
      1. Asset has fallen > threshold from its running peak  (drawdown filter)
      2. D1 < 0: N-day price return is negative             (falling trend)
      3. D2 < 0: change in D1 over N days is negative       (worsening momentum)

    Re-entry conditions (EITHER triggers re-entry):
      - D1 > 0 AND D2 > 0  (momentum and acceleration both positive)
      - OR price >= exit_price  (full recovery to exit level)

    D1 = d_window-day return: (price_today - price_{today-N}) / price_{today-N}
    D2 = change in D1 over d_window days: D1_today - D1_{today-N}

    During the initial warmup period (first d_window*2 bars where D1/D2 are NaN),
    the signal defaults to 1.0 (full position) to avoid spurious early exits.

    Parameters
    ----------
    asset_prices : daily price series for the asset to be protected
    threshold    : drawdown fraction to trigger exit (e.g. 0.10 = 10%)
    d_window     : lookback days for D1 and D2
    reduce_pct   : fraction of position to exit (1.0 = full exit, 0.5 = half)
    """
    prices = asset_prices.dropna()
    if prices.empty:
        return pd.Series(1.0, index=spy_prices.index)

    running_peak = prices.cummax()
    drawdown     = (prices - running_peak) / running_peak   # negative values

    d1 = prices.pct_change(d_window)
    d2 = d1.diff(d_window)

    in_position = True
    exit_price  = None
    full        = 1.0
    reduced     = 1.0 - reduce_pct
    signals     = []

    for i in range(len(prices)):
        d1_v = d1.iloc[i]
        d2_v = d2.iloc[i]
        p    = prices.iloc[i]
        dd   = drawdown.iloc[i]

        # Warmup: not enough data to compute D1/D2 reliably — stay in
        if np.isnan(d1_v) or np.isnan(d2_v):
            signals.append(full)
            continue

        if in_position:
            if dd < -threshold and d1_v < 0 and d2_v < 0:
                in_position = False
                exit_price  = p
                signals.append(reduced)
            else:
                signals.append(full)
        else:
            if (d1_v > 0 and d2_v > 0) or (p >= exit_price):
                in_position = True
                exit_price  = None
                signals.append(full)
            else:
                signals.append(reduced)

    result = pd.Series(signals, index=prices.index)
    return result.reindex(asset_prices.index, method="ffill").fillna(full)


# Backward-compatible alias used in tests and scripts
compute_spy_overlay_signal = compute_overlay_signal


def run_backtest_with_overlay(prices: pd.DataFrame,
                              benchmark_prices: pd.Series,
                              allocation: dict,
                              portfolio_value: Optional[float] = None,
                              tlt_prices: Optional[pd.Series] = None,
                              transaction_cost_pct: float = 0.0,
                              tax_drag_pct: float = 0.0) -> pd.DataFrame:
    """
    Like run_backtest() but with the SPY momentum overlay active.

    SPY is traded daily whenever the overlay signal fires (using overlay
    parameters from config). Freed capital is held as cash earning
    config.SPY_OVERLAY_CASH_RETURN annually. All other assets are
    rebalanced monthly only.

    The SPY allocation must be present in the allocation dict and in the
    prices DataFrame for the overlay to take effect. If SPY is absent from
    either, the function falls back to standard monthly rebalancing.

    Returns a monthly-indexed DataFrame in the identical format to
    run_backtest(), so all downstream code (compute_stats, export,
    plotting) works unchanged.

    Parameters
    ----------
    prices               : daily price DataFrame, one column per ticker
    benchmark_prices     : daily price Series for the benchmark (SPY)
    allocation           : dict of {ticker: weight}, weights must sum to 1.0
    portfolio_value      : starting value in USD (defaults to config value)
    tlt_prices           : daily TLT prices; enables 60/40 comparison strategy
    transaction_cost_pct : cost as fraction of trade value, applied on each trade
    tax_drag_pct         : annual drag on portfolio value (0.0 for ISA/SIPP)
    """
    if portfolio_value is None:
        portfolio_value = config.INITIAL_PORTFOLIO_VALUE

    tickers = list(allocation.keys())

    # Daily prices aligned across all assets + benchmark
    daily = prices[tickers].ffill().dropna()
    bench = benchmark_prices.ffill().dropna()
    common = daily.index.intersection(bench.index)
    daily = daily.loc[common]
    bench = bench.loc[common]

    # TLT for 60/40 benchmark
    tlt_daily = None
    if tlt_prices is not None:
        tlt_daily = tlt_prices.ffill().dropna().reindex(common, method="ffill")

    # Build per-asset overlay signals from ASSET_OVERLAYS config
    # Only for assets that are (a) in the allocation and (b) have enabled=True
    overlay_signals: dict[str, pd.Series] = {}
    for ticker, ov in config.ASSET_OVERLAYS.items():
        if ov["enabled"] and ticker in tickers and ticker in daily.columns:
            overlay_signals[ticker] = compute_overlay_signal(
                asset_prices = daily[ticker],
                threshold    = ov["threshold"],
                d_window     = ov["d_window"],
                reduce_pct   = ov["reduce_pct"],
            ).reindex(common, method="ffill").fillna(1.0)

    # Month-end dates used as rebalance triggers
    month_ends = set(daily.resample(config.DATA_FREQUENCY).last().dropna().index)

    # --- Initialise holdings ---
    first = daily.iloc[0]
    aw_holdings = {t: (portfolio_value * allocation[t]) / float(first[t])
                   for t in tickers}
    aw_cash = 0.0   # total capital held as cash across all overlay exits

    bh_holdings = {t: (portfolio_value * allocation[t]) / float(first[t])
                   for t in tickers}
    bench_shares = portfolio_value / float(bench.iloc[0])

    sixty_forty_spy_sh = None
    sixty_forty_tlt_sh = None
    sixty_forty_prev_yr = None
    if tlt_daily is not None:
        sixty_forty_spy_sh  = portfolio_value * 0.60 / float(bench.iloc[0])
        sixty_forty_tlt_sh  = portfolio_value * 0.40 / float(tlt_daily.iloc[0])
        sixty_forty_prev_yr = daily.index[0].year

    # Track previous signal for each overlaid asset to detect state changes
    prev_sigs = {t: sig.iloc[0] for t, sig in overlay_signals.items()}
    aw_prev_year = daily.index[0].year

    # Target weights for overlaid assets (used to size cash bucket at month-end)
    overlay_target_ws = {t: allocation.get(t, 0.0) for t in overlay_signals}

    month_records = []

    for date, row in daily.iterrows():

        # ---- Per-asset overlay: detect signal changes, trade ↔ cash ----
        for ticker, sig_series in overlay_signals.items():
            sig  = sig_series.loc[date]
            prev = prev_sigs[ticker]
            if sig != prev:
                price = float(row[ticker])
                ov    = config.ASSET_OVERLAYS[ticker]
                if sig < prev:
                    # EXIT: sell reduce_pct of this asset → cash
                    exit_val   = aw_holdings[ticker] * price * ov["reduce_pct"]
                    cost       = exit_val * transaction_cost_pct
                    aw_cash   += exit_val - cost
                    aw_holdings[ticker] *= (1.0 - ov["reduce_pct"])
                else:
                    # RE-ENTRY: buy this asset back with its share of cash
                    # Apportion cash proportionally by target weight among all
                    # currently-exited assets so re-entries don't steal from each other
                    exited_w = sum(
                        allocation.get(t, 0.0)
                        for t, s in overlay_signals.items()
                        if prev_sigs[t] < 1.0 and t != ticker
                    )
                    this_w     = allocation.get(ticker, 0.0)
                    total_out_w = this_w + exited_w
                    cash_to_use = aw_cash * (this_w / total_out_w) if total_out_w > 0 else aw_cash
                    cost        = cash_to_use * transaction_cost_pct
                    aw_holdings[ticker] += (cash_to_use - cost) / price
                    aw_cash             -= cash_to_use
                prev_sigs[ticker] = sig

        # ---- Cash earns daily rate while assets are out ----
        if aw_cash > 0.0 and config.OVERLAY_CASH_RETURN > 0.0:
            aw_cash *= (1.0 + config.OVERLAY_CASH_RETURN / 252.0)

        # ---- Portfolio values ----
        aw_value  = sum(aw_holdings[t] * float(row[t]) for t in tickers) + aw_cash
        bh_value  = sum(bh_holdings[t] * float(row[t]) for t in tickers)
        spy_value = bench_shares * float(bench.loc[date])

        # ---- Month-end: record + rebalance ----
        if date in month_ends:

            # Tax drag (annual, apply at year boundary)
            if tax_drag_pct > 0.0 and date.year != aw_prev_year:
                aw_value    *= (1.0 - tax_drag_pct)
                aw_prev_year = date.year

            # 60/40 annual rebalance
            sixty_forty_value = None
            if tlt_daily is not None:
                tlt_p = float(tlt_daily.loc[date])
                if date.year != sixty_forty_prev_yr:
                    v6040 = (sixty_forty_spy_sh * float(bench.loc[date])
                             + sixty_forty_tlt_sh * tlt_p)
                    sixty_forty_spy_sh  = v6040 * 0.60 / float(bench.loc[date])
                    sixty_forty_tlt_sh  = v6040 * 0.40 / tlt_p
                    sixty_forty_prev_yr = date.year
                sixty_forty_value = (sixty_forty_spy_sh * float(bench.loc[date])
                                     + sixty_forty_tlt_sh * tlt_p)

            # AW rebalance — respects overlay state
            # Determine which overlay assets are currently active (exited)
            exited_tickers = {
                t for t, s in prev_sigs.items() if s < 1.0
            }

            if not exited_tickers:
                # All overlays inactive: full rebalance to target, absorb cash
                total = aw_value
                if transaction_cost_pct > 0.0:
                    trade = sum(
                        abs((total * w) - (aw_holdings[t] * float(row[t])))
                        for t, w in allocation.items()
                    )
                    total -= trade * transaction_cost_pct
                for t, w in allocation.items():
                    aw_holdings[t] = (total * w) / float(row[t])
                aw_cash = 0.0
            else:
                # Some overlays active: exited assets stay as cash at their
                # target weights. Invested assets rebalanced to their absolute
                # target weights (unchanged — still % of aw_value).
                # e.g. if SPY and TLT are exited: SPY 15% + TLT 30% = 45% cash,
                # and QQQ/TIP/GLD/GSG each rebalance to their target % of aw_value.
                invested = {t: w for t, w in allocation.items()
                            if t not in exited_tickers}
                if transaction_cost_pct > 0.0:
                    trade = sum(
                        abs((aw_value * w) - (aw_holdings[t] * float(row[t])))
                        for t, w in invested.items()
                    )
                    aw_value -= trade * transaction_cost_pct
                # Cash bucket = sum of target weights of all exited assets
                cash_w   = sum(allocation.get(t, 0.0) for t in exited_tickers)
                aw_cash  = aw_value * cash_w
                for t, w in invested.items():
                    aw_holdings[t] = (aw_value * w) / float(row[t])
                for t in exited_tickers:
                    aw_holdings[t] = 0.0

            # B&H weights (never rebalanced)
            bh_weights = {t: (bh_holdings[t] * float(row[t])) / bh_value
                          for t in tickers}

            record = {
                "Date":                   date,
                "All Weather Value":      round(aw_value, 2),
                "Buy & Hold All Weather": round(bh_value, 2),
                "S&P 500 Value":          round(spy_value, 2),
            }
            if sixty_forty_value is not None:
                record["60/40 Value"] = round(sixty_forty_value, 2)
            for t in tickers:
                record[f"B&H {t} Weight (%)"] = round(bh_weights[t] * 100, 1)

            month_records.append(record)

    df = pd.DataFrame(month_records).set_index("Date")
    for col in ["All Weather Value", "Buy & Hold All Weather", "S&P 500 Value"]:
        df[f"{col} Monthly Ret (%)"] = df[col].pct_change() * 100
    if "60/40 Value" in df.columns:
        df["60/40 Value Monthly Ret (%)"] = df["60/40 Value"].pct_change() * 100

    return df


# ===========================================================================
# STATISTICS
# ===========================================================================

def compute_stats(backtest: pd.DataFrame,
                  prices: Optional["pd.DataFrame"] = None,
                  allocation: Optional[dict] = None) -> list[StrategyStats]:
    """
    Compute key performance statistics for all three strategies.

    Parameters
    ----------
    backtest   : monthly backtest DataFrame from run_backtest()
    prices     : daily price DataFrame (optional). When provided alongside
                 allocation, daily MDD is computed for the AW_R strategy.
    allocation : allocation dict {ticker: weight} (optional, paired with prices)

    Returns a list of three or four StrategyStats objects:
      [All Weather (Rebalanced), Buy & Hold All Weather, S&P 500 Buy & Hold]
      plus 60/40 as the fourth element if the "60/40 Value" column exists.
    """
    years = (backtest.index[-1] - backtest.index[0]).days / 365.25

    # Compute daily MDD for AW_R once if daily prices are available
    daily_mdd_aw = 0.0
    if prices is not None and allocation is not None:
        # Slice daily prices to the same date window as the backtest
        mask         = ((prices.index >= backtest.index[0]) &
                        (prices.index <= backtest.index[-1]))
        prices_slice = prices.loc[mask]
        daily_mdd_aw = round(compute_max_drawdown_daily(prices_slice, allocation), 2)

    def make_stats(name: str, value_col: str, ret_col: str,
                   daily_mdd: float = 0.0) -> StrategyStats:
        series      = backtest[value_col]
        cagr        = round(compute_cagr(series, years), 2)
        mdd         = round(compute_max_drawdown(series), 2)
        ulcer       = compute_ulcer_index(series)
        martin      = round(cagr / ulcer if ulcer > 1e-10 else cagr, 3)
        return StrategyStats(
            name               = name,
            cagr               = cagr,
            max_drawdown       = mdd,
            sharpe             = round(compute_sharpe(backtest[ret_col],
                                                      rf_annual=config.RISK_FREE_RATE), 3),
            calmar             = round(compute_calmar(cagr, mdd), 3),
            final_value        = round(series.iloc[-1], 2),
            period_years       = round(years, 1),
            avg_drawdown       = compute_avg_drawdown(series),
            max_dd_duration    = compute_max_drawdown_duration(series),
            avg_recovery_time  = compute_avg_recovery_time(series),
            ulcer_index        = ulcer,
            sortino            = compute_sortino(backtest[ret_col],
                                                rf_annual=config.RISK_FREE_RATE),
            martin             = martin,
            max_drawdown_daily = daily_mdd,
        )

    stats = [
        make_stats("AW_R",
                   "All Weather Value",
                   "All Weather Value Monthly Ret (%)",
                   daily_mdd=daily_mdd_aw),
        make_stats("B&H_AW",
                   "Buy & Hold All Weather",
                   "Buy & Hold All Weather Monthly Ret (%)"),
        make_stats("SPY",
                   "S&P 500 Value",
                   "S&P 500 Value Monthly Ret (%)"),
    ]

    if "60/40 Value" in backtest.columns:
        stats.append(make_stats("60/40",
                                "60/40 Value",
                                "60/40 Value Monthly Ret (%)"))

    return stats
