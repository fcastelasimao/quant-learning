"""
research/momentum_overlay/overlay.py
=============================================
CLOSED INVESTIGATION — SPY Momentum Overlay.

Conclusion: Tested 126 parameter combinations (threshold × d_window × reduce_pct)
on IS data (2006-2020). No combination improved OOS Calmar consistently — the
re-entry timing is not reliably learnable. Overlay closed; production uses
static monthly rebalancing.

See research/momentum_overlay/findings.md for the full investigation log.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine import config
from engine.calendar import pandas_resample_frequency

SIXTY_FORTY_EQUITY = 0.60
SIXTY_FORTY_BOND   = 0.40


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
        return pd.Series(1.0, index=asset_prices.index)

    running_peak = prices.cummax()
    drawdown     = (prices - running_peak) / running_peak   # negative values

    d1 = prices.pct_change(d_window)
    d2 = d1.diff(d_window)

    in_position = True
    exit_price  = None
    full        = 1.0
    reduced     = 1.0 - reduce_pct
    signals     = []

    d1_arr = d1.values
    d2_arr = d2.values
    p_arr  = prices.values
    dd_arr = drawdown.values

    for i in range(len(prices)):
        d1_v = d1_arr[i]
        d2_v = d2_arr[i]
        p    = p_arr[i]
        dd   = dd_arr[i]

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


# Backward-compatible alias
compute_spy_overlay_signal = compute_overlay_signal


def run_backtest_with_overlay(prices: pd.DataFrame,
                              benchmark_prices: pd.Series,
                              allocation: dict,
                              portfolio_value: float | None = None,
                              tlt_prices: pd.Series | None = None,
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
    month_ends = set(daily.resample(pandas_resample_frequency(config.DATA_FREQUENCY)).last().dropna().index)

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
        sixty_forty_spy_sh  = portfolio_value * SIXTY_FORTY_EQUITY / float(bench.iloc[0])
        sixty_forty_tlt_sh  = portfolio_value * SIXTY_FORTY_BOND / float(tlt_daily.iloc[0])
        sixty_forty_prev_yr = daily.index[0].year

    # Track previous signal for each overlaid asset to detect state changes
    prev_sigs = {t: sig.iloc[0] for t, sig in overlay_signals.items()}
    aw_prev_year = daily.index[0].year

    # Target weights for overlaid assets (used to size cash bucket at month-end)
    overlay_target_ws = {t: allocation.get(t, 0.0) for t in overlay_signals}

    # Note: this loop is intentionally iterative due to stateful daily overlay
    # signal tracking, cash management, and monthly rebalancing with transaction costs.
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
        prices_now = {t: float(row[t]) for t in tickers}
        aw_value = sum(sh * prices_now[t] for t, sh in aw_holdings.items()) + aw_cash
        bh_value = sum(sh * prices_now[t] for t, sh in bh_holdings.items())
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
                    sixty_forty_spy_sh  = v6040 * SIXTY_FORTY_EQUITY / float(bench.loc[date])
                    sixty_forty_tlt_sh  = v6040 * SIXTY_FORTY_BOND / tlt_p
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
