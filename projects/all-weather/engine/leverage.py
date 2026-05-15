"""
leverage.py
===========
Reusable ETF overlay helpers for research backtests.

The overlay engine is intentionally separate from the production portfolio
backtest so baseline results remain unchanged. It adds daily overlay returns
on top of an already-built base portfolio value series.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class OverlaySpec:
    """Configuration for one ETF overlay signal."""

    ticker: str
    indicator: str = "rsi"
    lookback: int = 14
    entry_threshold: float = 30.0
    exit_threshold: float = 50.0
    overlay_weight: float = 0.20
    enabled: bool = True


@dataclass(frozen=True)
class OverlayResult:
    """Outputs from applying ETF overlay specs to a base portfolio."""

    value_series: pd.Series
    positions: pd.DataFrame
    raw_signals: pd.DataFrame
    indicators: pd.DataFrame
    signal_history: pd.DataFrame
    daily_diagnostics: pd.DataFrame


def selected_window_metrics(daily: pd.DataFrame,
                            diagnostics: pd.DataFrame | None = None,
                            start: pd.Timestamp | str | None = None,
                            end: pd.Timestamp | str | None = None) -> pd.DataFrame:
    """
    Recompute headline metrics from exported daily_series rows.

    This is used by the marimo notebook for arbitrary user-selected windows.
    It expects the long-form daily export with Date, Strategy, and Value.
    """
    if daily.empty:
        return pd.DataFrame()

    data = daily.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    start_ts = pd.Timestamp(start) if start is not None else data["Date"].min()
    end_ts = pd.Timestamp(end) if end is not None else data["Date"].max()
    data = data[(data["Date"] >= start_ts) & (data["Date"] <= end_ts)]

    diag = None
    if diagnostics is not None and not diagnostics.empty:
        diag = diagnostics.copy()
        diag["Date"] = pd.to_datetime(diag["Date"])
        diag = diag[(diag["Date"] >= start_ts) & (diag["Date"] <= end_ts)]

    rows = []
    for strategy, group in data.groupby("Strategy"):
        s = group.sort_values("Date").set_index("Date")["Value"].dropna().astype(float)
        if len(s) < 2:
            continue
        returns = s.pct_change().dropna()
        drawdown = (s / s.cummax() - 1.0) * 100
        years = max((s.index[-1] - s.index[0]).days / 365.25, 1 / 365.25)
        vol = float(returns.std() * math.sqrt(TRADING_DAYS_PER_YEAR) * 100) if len(returns) else np.nan
        sharpe = float((returns.mean() / returns.std()) * math.sqrt(TRADING_DAYS_PER_YEAR)) if len(returns) and returns.std() > 1e-12 else np.nan
        cagr = float(((s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1) * 100)
        max_dd = float(drawdown.min())

        active_days = 0
        avg_overlay = 0.0
        overlay_contribution = 0.0
        if diag is not None and "Overlay Strategy" in diag.columns:
            d = diag[diag["Overlay Strategy"] == strategy]
            if not d.empty:
                active_days = int((d["Overlay Exposure"] > 0).sum())
                avg_overlay = float(d["Overlay Exposure"].mean() * 100)
                overlay_contribution = float(d["Overlay Return"].sum() * 100)

        rows.append({
            "Strategy": strategy,
            "Start Date": s.index[0].date().isoformat(),
            "End Date": s.index[-1].date().isoformat(),
            "Observations": int(len(s)),
            "Total Return (%)": round(float((s.iloc[-1] / s.iloc[0] - 1) * 100), 4),
            "CAGR (%)": round(cagr, 4),
            "Volatility (%)": round(vol, 4),
            "Sharpe": round(sharpe, 4),
            "Calmar": round(cagr / abs(max_dd), 4) if abs(max_dd) > 1e-12 else np.nan,
            "Max Drawdown (%)": round(max_dd, 4),
            "Worst Day (%)": round(float(returns.min() * 100), 4) if len(returns) else np.nan,
            "Active Days": active_days,
            "Average Overlay Exposure (%)": round(avg_overlay, 4),
            "Overlay Return Contribution (%)": round(overlay_contribution, 4),
        })
    return pd.DataFrame(rows)


def compute_rsi(prices: pd.Series, lookback: int = 14) -> pd.Series:
    """
    Compute Wilder RSI for a price series.

    RSI is 100 when there are gains and no losses, 0 when there are losses
    and no gains, and 50 when both smoothed gains and losses are zero.
    """
    if lookback <= 0:
        raise ValueError("lookback must be positive.")

    s = prices.dropna().astype(float)
    delta = s.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)

    avg_gain = gains.ewm(alpha=1 / lookback, adjust=False, min_periods=lookback).mean()
    avg_loss = losses.ewm(alpha=1 / lookback, adjust=False, min_periods=lookback).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.where(avg_loss != 0.0, 100.0)
    rsi = rsi.where(avg_gain != 0.0, 0.0)
    rsi = rsi.where(~((avg_gain == 0.0) & (avg_loss == 0.0)), 50.0)
    return rsi.reindex(prices.index).rename("RSI")


def generate_hysteresis_signal(indicator: pd.Series,
                               entry_threshold: float,
                               exit_threshold: float) -> pd.Series:
    """
    Return a stateful 0/1 signal with separate entry and exit thresholds.

    For an RSI-style contrarian overlay:
      - enter when RSI < entry_threshold
      - remain active until RSI > exit_threshold
    """
    if exit_threshold <= entry_threshold:
        raise ValueError("exit_threshold must be greater than entry_threshold.")

    active = False
    values: list[int] = []
    for value in indicator:
        if pd.notna(value):
            if not active and float(value) < entry_threshold:
                active = True
            elif active and float(value) > exit_threshold:
                active = False
        values.append(1 if active else 0)
    return pd.Series(values, index=indicator.index, name="signal")


def build_overlay_positions(prices: pd.DataFrame,
                            specs: list[OverlaySpec],
                            global_cap: float = 0.20,
                            execution_lag: int = 1) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build applied overlay weights from ETF specs.

    Returns positions after execution lag, raw 0/1 signals, indicator values,
    and long-form signal history. If requested overlay weights exceed
    global_cap on any date, desired weights are scaled proportionally.
    """
    if global_cap < 0:
        raise ValueError("global_cap must be non-negative.")
    if execution_lag < 0:
        raise ValueError("execution_lag must be non-negative.")

    clean = prices.sort_index().astype(float)
    enabled_specs = [spec for spec in specs if spec.enabled]
    if not enabled_specs:
        empty = pd.DataFrame(index=clean.index)
        return empty, empty, empty, _signal_history(empty, empty, empty, [])

    desired = pd.DataFrame(0.0, index=clean.index, columns=[spec.ticker for spec in enabled_specs])
    raw_signals = pd.DataFrame(0, index=clean.index, columns=desired.columns)
    indicators = pd.DataFrame(np.nan, index=clean.index, columns=desired.columns)

    for spec in enabled_specs:
        if spec.ticker not in clean.columns:
            raise KeyError(f"Overlay ticker '{spec.ticker}' not found in prices.")
        if spec.overlay_weight < 0:
            raise ValueError("overlay_weight must be non-negative.")
        if spec.indicator.lower() != "rsi":
            raise ValueError(f"Unsupported indicator: {spec.indicator}")

        rsi = compute_rsi(clean[spec.ticker], spec.lookback)
        signal = generate_hysteresis_signal(rsi, spec.entry_threshold, spec.exit_threshold)
        indicators[spec.ticker] = rsi
        raw_signals[spec.ticker] = signal
        desired[spec.ticker] = signal * spec.overlay_weight

    totals = desired.sum(axis=1)
    scale = pd.Series(1.0, index=desired.index)
    over_cap = totals > global_cap
    scale.loc[over_cap] = global_cap / totals.loc[over_cap]
    capped = desired.mul(scale, axis=0)
    positions = capped.shift(execution_lag).fillna(0.0)

    history = _signal_history(indicators, raw_signals, desired, enabled_specs, capped, positions)
    return positions, raw_signals, indicators, history


def apply_overlay_to_base(base_values: pd.Series,
                          prices: pd.DataFrame,
                          specs: list[OverlaySpec],
                          global_cap: float = 0.20,
                          financing_cost_annual: float = 0.0,
                          execution_lag: int = 1,
                          name: str = "Leveraged Portfolio") -> OverlayResult:
    """
    Add capped ETF overlay returns to a base portfolio value series.

    The base portfolio is not rebuilt or altered. Its daily returns are used
    as the starting point, then lagged overlay weights add ETF return exposure.
    """
    if financing_cost_annual < 0:
        raise ValueError("financing_cost_annual must be non-negative.")

    base = base_values.dropna().astype(float).sort_index()
    clean_prices = prices.sort_index().astype(float)
    common = base.index.intersection(clean_prices.index)
    if common.empty:
        raise ValueError("No overlapping dates between base_values and prices.")

    base = base.loc[common]
    clean_prices = clean_prices.loc[common]
    positions, raw_signals, indicators, history = build_overlay_positions(
        clean_prices,
        specs,
        global_cap=global_cap,
        execution_lag=execution_lag,
    )

    asset_returns = clean_prices.pct_change().fillna(0.0)
    base_returns = base.pct_change().fillna(0.0)
    overlay_returns = (positions.reindex(asset_returns.index).fillna(0.0) * asset_returns).sum(axis=1)
    gross_exposure = 1.0 + positions.sum(axis=1)
    daily_financing_rate = (1.0 + financing_cost_annual) ** (1 / TRADING_DAYS_PER_YEAR) - 1.0
    financing_cost = positions.sum(axis=1) * daily_financing_rate
    strategy_returns = base_returns + overlay_returns - financing_cost
    values = base.iloc[0] * (1.0 + strategy_returns).cumprod()
    values.iloc[0] = base.iloc[0]
    values = values.rename(name)

    diagnostics = pd.DataFrame({
        "Base Return": base_returns,
        "Overlay Return": overlay_returns,
        "Financing Cost": financing_cost,
        "Strategy Return": strategy_returns,
        "Overlay Exposure": positions.sum(axis=1),
        "Gross Exposure": gross_exposure,
    }, index=base.index)
    diagnostics.index.name = "Date"

    return OverlayResult(
        value_series=values,
        positions=positions,
        raw_signals=raw_signals,
        indicators=indicators,
        signal_history=history,
        daily_diagnostics=diagnostics,
    )


def _signal_history(indicators: pd.DataFrame,
                    raw_signals: pd.DataFrame,
                    desired: pd.DataFrame,
                    specs: list[OverlaySpec],
                    capped: pd.DataFrame | None = None,
                    positions: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    spec_by_ticker = {spec.ticker: spec for spec in specs}
    for ticker in indicators.columns:
        spec = spec_by_ticker[ticker]
        for date_idx in indicators.index:
            rows.append({
                "Date": date_idx,
                "Ticker": ticker,
                "Indicator": spec.indicator.upper(),
                "Lookback": spec.lookback,
                "Entry Threshold": spec.entry_threshold,
                "Exit Threshold": spec.exit_threshold,
                "RSI": round(float(indicators.loc[date_idx, ticker]), 6)
                if pd.notna(indicators.loc[date_idx, ticker]) else pd.NA,
                "Raw Signal": int(raw_signals.loc[date_idx, ticker]),
                "Desired Overlay Weight": round(float(desired.loc[date_idx, ticker]), 8),
                "Capped Overlay Weight": round(float(capped.loc[date_idx, ticker]), 8)
                if capped is not None else pd.NA,
                "Applied Overlay Weight": round(float(positions.loc[date_idx, ticker]), 8)
                if positions is not None else pd.NA,
            })
    return pd.DataFrame(rows)
