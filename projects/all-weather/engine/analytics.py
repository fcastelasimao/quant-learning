"""
analytics.py
============
Pure analytics helpers for bank-facing strategy comparison reports.

These functions receive already-built price or portfolio value series and
return pandas objects ready to export. They do not fetch data or write files.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .calendar import pandas_resample_frequency

TRADING_DAYS_PER_YEAR = 252
MONTH_END = pandas_resample_frequency("ME")


def apply_annual_fee(series: pd.Series, annual_fee: float) -> pd.Series:
    """Apply a compounding annual expense-ratio drag to a value series."""
    s = series.dropna().astype(float)
    discount = (1.0 - annual_fee) ** (np.arange(len(s)) / TRADING_DAYS_PER_YEAR)
    return (s * discount).rename(series.name)


def build_buy_hold_series(prices: pd.DataFrame,
                          allocation: dict[str, float],
                          start_value: float = 100.0) -> pd.Series:
    """Build a daily buy-and-hold portfolio value series."""
    clean = prices[list(allocation)].dropna().astype(float)
    if clean.empty:
        return pd.Series(dtype=float, name="portfolio")
    weights = _normalised_weights(allocation, clean.columns)
    first = clean.iloc[0]
    shares = {ticker: start_value * weights[ticker] / float(first[ticker])
              for ticker in clean.columns}
    return sum(shares[ticker] * clean[ticker] for ticker in clean.columns).rename("portfolio")


def build_monthly_rebalanced_series(prices: pd.DataFrame,
                                    allocation: dict[str, float],
                                    start_value: float = 100.0) -> pd.Series:
    """Build a daily portfolio series rebalanced to target weights at month-end."""
    clean = prices[list(allocation)].dropna().astype(float)
    if clean.empty:
        return pd.Series(dtype=float, name="portfolio")

    weights = _normalised_weights(allocation, clean.columns)
    first = clean.iloc[0]
    shares = {ticker: start_value * weights[ticker] / float(first[ticker])
              for ticker in clean.columns}
    month_ends = set(clean.resample(MONTH_END).last().index)
    values = []

    for date, row in clean.iterrows():
        portfolio_value = sum(shares[ticker] * float(row[ticker]) for ticker in clean.columns)
        values.append(portfolio_value)
        if date in month_ends:
            shares = {
                ticker: portfolio_value * weights[ticker] / float(row[ticker])
                for ticker in clean.columns
            }

    return pd.Series(values, index=clean.index, name="portfolio")


def drawdown_series(series: pd.Series) -> pd.Series:
    """Return percentage drawdown from the running high-water mark."""
    s = series.dropna().astype(float)
    if s.empty:
        return pd.Series(dtype=float, name="Drawdown (%)")
    return (((s - s.cummax()) / s.cummax()) * 100).rename("Drawdown (%)")


def max_drawdown_duration(series: pd.Series) -> int:
    """Longest consecutive observations below the previous high-water mark."""
    s = series.dropna().astype(float)
    if len(s) < 2:
        return 0
    underwater = s < s.cummax()
    if not underwater.any():
        return 0
    groups = (~underwater).cumsum()
    return int(underwater.groupby(groups).sum().max())


def drawdown_events(values: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Return the deepest drawdown events per strategy."""
    rows = []
    for strategy in values.columns:
        s = values[strategy].dropna().astype(float)
        if len(s) < 2:
            continue
        dd = drawdown_series(s)
        in_event = False
        peak_date = s.index[0]
        trough_date = None
        trough_depth = 0.0
        event_start_pos = 0

        for pos, date in enumerate(s.index):
            if not in_event and dd.loc[date] < 0:
                in_event = True
                event_start_pos = max(pos - 1, 0)
                peak_date = s.index[event_start_pos]
                trough_date = date
                trough_depth = float(dd.loc[date])
            elif in_event:
                if dd.loc[date] < trough_depth:
                    trough_depth = float(dd.loc[date])
                    trough_date = date
                if dd.loc[date] >= 0:
                    rows.append(_drawdown_event_row(strategy, peak_date, trough_date, date, trough_depth))
                    in_event = False

        if in_event:
            rows.append(_drawdown_event_row(strategy, peak_date, trough_date, pd.NaT, trough_depth))

    if not rows:
        return pd.DataFrame(columns=[
            "Strategy", "Peak Date", "Trough Date", "Recovery Date",
            "Depth (%)", "Decline Days", "Recovery Days", "Total Days", "Recovered",
        ])
    out = pd.DataFrame(rows)
    return out.sort_values(["Strategy", "Depth (%)"]).groupby("Strategy", as_index=False).head(top_n)


def calendar_year_metrics(values: pd.DataFrame) -> pd.DataFrame:
    """Return yearly return, max drawdown, and max drawdown duration."""
    rows = []
    for strategy in values.columns:
        s = values[strategy].dropna().astype(float)
        previous_year_end = None
        for year, year_series in s.groupby(s.index.year):
            if year_series.empty:
                continue
            start_value = float(previous_year_end if previous_year_end is not None else year_series.iloc[0])
            end_value = float(year_series.iloc[-1])
            dd_base = pd.concat([
                pd.Series([start_value], index=[year_series.index[0] - pd.Timedelta(days=1)]),
                year_series,
            ])
            rows.append({
                "Year": int(year),
                "Strategy": strategy,
                "Start Value": round(start_value, 4),
                "End Value": round(end_value, 4),
                "Return (%)": round(_safe_return(start_value, end_value), 4),
                "Max Drawdown (%)": round(float(drawdown_series(dd_base).min()), 4),
                "Max DD Duration (days)": max_drawdown_duration(dd_base),
                "Calmar": _calendar_calmar(start_value, end_value, dd_base),
            })
            previous_year_end = end_value
    return pd.DataFrame(rows, columns=[
        "Year", "Strategy", "Start Value", "End Value", "Return (%)",
        "Max Drawdown (%)", "Max DD Duration (days)", "Calmar",
    ])


def monthly_returns(values: pd.DataFrame) -> pd.DataFrame:
    """Return long-form month-end return table for heatmaps and distributions."""
    rows = []
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for strategy in values.columns:
        s = values[strategy].dropna().astype(float)
        if len(s) < 2:
            continue
        rets = s.resample(MONTH_END).last().pct_change().dropna() * 100
        for date, value in rets.items():
            rows.append({
                "Date": date,
                "Year": int(date.year),
                "Month": int(date.month),
                "Month Label": month_labels[date.month - 1],
                "Strategy": strategy,
                "Return (%)": round(float(value), 6),
            })
    return pd.DataFrame(rows, columns=[
        "Date", "Year", "Month", "Month Label", "Strategy", "Return (%)",
    ])


def summary_metrics(values: pd.DataFrame,
                    benchmark: str = "SPY",
                    windows: dict[str, tuple[pd.Timestamp | None, pd.Timestamp | None]] | None = None,
                    rf_annual: float = 0.035) -> pd.DataFrame:
    """Return headline performance and risk metrics by strategy and window."""
    if windows is None:
        windows = {"Full History": (None, None)}

    rows = []
    for window_name, (start, end) in windows.items():
        window_values = _slice_frame(values, start, end)
        benchmark_returns = _daily_returns(window_values[benchmark]) if benchmark in window_values else None
        for strategy in window_values.columns:
            s = window_values[strategy].dropna().astype(float)
            if len(s) < 2:
                continue
            returns = _daily_returns(s)
            bench_aligned = None
            if benchmark_returns is not None and strategy != benchmark:
                bench_aligned = benchmark_returns.reindex(returns.index).dropna()
            beta, corr = _beta_corr(returns, bench_aligned)
            downside_beta = _downside_beta(returns, bench_aligned)
            up_capture, down_capture = _capture_ratios(returns, bench_aligned)
            tail = tail_risk_metrics(s)
            cagr = _annualized_return(s)
            max_dd = float(drawdown_series(s).min())
            vol = float(returns.std() * math.sqrt(TRADING_DAYS_PER_YEAR) * 100) if len(returns) else np.nan
            sharpe = _sharpe(returns, rf_annual)
            sortino = _sortino(returns, rf_annual)
            ulcer = float(np.sqrt((drawdown_series(s) ** 2).mean()))

            rows.append({
                "Window": window_name,
                "Strategy": strategy,
                "Start Date": s.index[0].date().isoformat(),
                "End Date": s.index[-1].date().isoformat(),
                "Observations": int(len(s)),
                "Total Return (%)": round(_safe_return(float(s.iloc[0]), float(s.iloc[-1])), 4),
                "CAGR (%)": round(cagr, 4),
                "Volatility (%)": round(vol, 4),
                "Sharpe": round(sharpe, 4),
                "Sortino": round(sortino, 4),
                "Calmar": round(cagr / abs(max_dd), 4) if abs(max_dd) > 1e-12 else np.nan,
                "Ulcer Index": round(ulcer, 4),
                "Max Drawdown (%)": round(max_dd, 4),
                "Max DD Duration (days)": max_drawdown_duration(s),
                "Beta to SPY": round(beta, 4),
                "Correlation to SPY": round(corr, 4),
                "Downside Beta": round(downside_beta, 4),
                "Up Capture (%)": round(up_capture, 4),
                "Down Capture (%)": round(down_capture, 4),
                **tail,
            })
    return pd.DataFrame(rows, columns=[
        "Window", "Strategy", "Start Date", "End Date", "Observations",
        "Total Return (%)", "CAGR (%)", "Volatility (%)", "Sharpe", "Sortino",
        "Calmar", "Ulcer Index", "Max Drawdown (%)", "Max DD Duration (days)",
        "Beta to SPY", "Correlation to SPY", "Downside Beta",
        "Up Capture (%)", "Down Capture (%)", "Worst Day (%)", "Worst Week (%)",
        "Worst Month (%)", "VaR 5% Daily (%)", "CVaR 5% Daily (%)",
        "Skew", "Kurtosis",
    ])


def rolling_metrics(values: pd.DataFrame,
                    benchmark: str = "SPY",
                    windows: tuple[int, ...] = (252, 756),
                    rf_annual: float = 0.035) -> pd.DataFrame:
    """Return rolling performance, risk, correlation, and beta metrics."""
    rows = []
    benchmark_returns = _daily_returns(values[benchmark]) if benchmark in values else None

    for strategy in values.columns:
        s = values[strategy].dropna().astype(float)
        returns = _daily_returns(s)
        for window in windows:
            if len(s) < window:
                continue
            for end_pos in range(window - 1, len(s)):
                chunk = s.iloc[end_pos - window + 1:end_pos + 1]
                chunk_returns = _daily_returns(chunk)
                bench_chunk = None
                if benchmark_returns is not None and strategy != benchmark:
                    bench_chunk = benchmark_returns.reindex(chunk_returns.index).dropna()
                beta, corr = _beta_corr(chunk_returns, bench_chunk)
                rows.append({
                    "Date": chunk.index[-1],
                    "Strategy": strategy,
                    "Window": f"{round(window / TRADING_DAYS_PER_YEAR)}Y",
                    "Window Trading Days": int(window),
                    "Rolling CAGR (%)": round(_annualized_return(chunk), 4),
                    "Rolling Volatility (%)": round(float(chunk_returns.std() * math.sqrt(TRADING_DAYS_PER_YEAR) * 100), 4),
                    "Rolling Sharpe": round(_sharpe(chunk_returns, rf_annual), 4),
                    "Rolling Sortino": round(_sortino(chunk_returns, rf_annual), 4),
                    "Rolling Max Drawdown (%)": round(float(drawdown_series(chunk).min()), 4),
                    "Rolling Corr to SPY": round(corr, 4),
                    "Rolling Beta to SPY": round(beta, 4),
                })
    return pd.DataFrame(rows, columns=[
        "Date", "Strategy", "Window", "Window Trading Days",
        "Rolling CAGR (%)", "Rolling Volatility (%)", "Rolling Sharpe",
        "Rolling Sortino", "Rolling Max Drawdown (%)", "Rolling Corr to SPY",
        "Rolling Beta to SPY",
    ])


def tail_risk_metrics(series: pd.Series) -> dict[str, float]:
    """Return worst-period and distribution tail-risk metrics."""
    s = series.dropna().astype(float)
    daily = _daily_returns(s) * 100
    weekly = s.resample("W-FRI").last().pct_change().dropna() * 100
    monthly = s.resample(MONTH_END).last().pct_change().dropna() * 100
    var_5 = float(daily.quantile(0.05)) if len(daily) else np.nan
    cvar_5 = float(daily[daily <= var_5].mean()) if len(daily) else np.nan
    return {
        "Worst Day (%)": round(float(daily.min()), 4) if len(daily) else np.nan,
        "Worst Week (%)": round(float(weekly.min()), 4) if len(weekly) else np.nan,
        "Worst Month (%)": round(float(monthly.min()), 4) if len(monthly) else np.nan,
        "VaR 5% Daily (%)": round(var_5, 4),
        "CVaR 5% Daily (%)": round(cvar_5, 4),
        "Skew": round(float(daily.skew()), 4) if len(daily) > 2 else np.nan,
        "Kurtosis": round(float(daily.kurtosis()), 4) if len(daily) > 3 else np.nan,
    }


def stress_period_metrics(values: pd.DataFrame,
                          periods: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """Return performance and risk metrics across named stress windows."""
    rows = []
    for name, (start, end) in periods.items():
        sliced = _slice_frame(values, pd.Timestamp(start), pd.Timestamp(end))
        for strategy in sliced.columns:
            s = sliced[strategy].dropna().astype(float)
            if len(s) < 2:
                continue
            rows.append({
                "Period": name,
                "Start Date": s.index[0].date().isoformat(),
                "End Date": s.index[-1].date().isoformat(),
                "Strategy": strategy,
                "Return (%)": round(_safe_return(float(s.iloc[0]), float(s.iloc[-1])), 4),
                "Max Drawdown (%)": round(float(drawdown_series(s).min()), 4),
                "Max DD Duration (days)": max_drawdown_duration(s),
                "Volatility (%)": round(float(_daily_returns(s).std() * math.sqrt(TRADING_DAYS_PER_YEAR) * 100), 4),
            })
    return pd.DataFrame(rows, columns=[
        "Period", "Start Date", "End Date", "Strategy", "Return (%)",
        "Max Drawdown (%)", "Max DD Duration (days)", "Volatility (%)",
    ])


def risk_contribution(asset_prices: pd.DataFrame,
                      allocation: dict[str, float]) -> pd.DataFrame:
    """Return asset volatility contribution from target weights and covariance."""
    clean = asset_prices[list(allocation)].dropna().astype(float)
    if len(clean) < 2:
        return pd.DataFrame(columns=[
            "Asset", "Weight", "Annual Vol Contribution", "Risk Contribution (%)",
        ])
    weights = np.array([allocation[ticker] for ticker in clean.columns], dtype=float)
    weights = weights / weights.sum()
    cov = clean.pct_change().dropna().cov().values * TRADING_DAYS_PER_YEAR
    portfolio_var = float(weights.T @ cov @ weights)
    if portfolio_var <= 0:
        contrib = np.zeros_like(weights)
    else:
        marginal = cov @ weights
        contrib = weights * marginal / portfolio_var
    return pd.DataFrame({
        "Asset": list(clean.columns),
        "Weight": weights,
        "Annual Vol Contribution": contrib * math.sqrt(max(portfolio_var, 0)),
        "Risk Contribution (%)": contrib * 100,
    })


def turnover_costs(asset_prices: pd.DataFrame,
                   allocation: dict[str, float],
                   transaction_cost_pct: float = 0.001,
                   start_value: float = 100.0) -> pd.DataFrame:
    """Estimate monthly rebalance turnover and transaction-cost drag."""
    clean = asset_prices[list(allocation)].dropna().astype(float)
    if clean.empty:
        return pd.DataFrame(columns=[
            "Date", "Portfolio Value Before Cost", "Turnover (%)",
            "Estimated Cost", "Cumulative Cost", "Cumulative Cost Drag (%)",
        ])
    weights = _normalised_weights(allocation, clean.columns)
    first = clean.iloc[0]
    shares = {ticker: start_value * weights[ticker] / float(first[ticker])
              for ticker in clean.columns}
    month_ends = set(clean.resample(MONTH_END).last().index)
    cumulative_cost = 0.0
    rows = []

    for date, row in clean.iterrows():
        portfolio_value = sum(shares[ticker] * float(row[ticker]) for ticker in clean.columns)
        if date not in month_ends:
            continue
        current_values = {ticker: shares[ticker] * float(row[ticker]) for ticker in clean.columns}
        trade_value = sum(abs(portfolio_value * weights[ticker] - current_values[ticker])
                          for ticker in clean.columns)
        cost = trade_value * transaction_cost_pct
        cumulative_cost += cost
        value_after_cost = portfolio_value - cost
        rows.append({
            "Date": date,
            "Portfolio Value Before Cost": round(portfolio_value, 6),
            "Turnover (%)": round(trade_value / portfolio_value * 100 if portfolio_value else 0.0, 6),
            "Estimated Cost": round(cost, 6),
            "Cumulative Cost": round(cumulative_cost, 6),
            "Cumulative Cost Drag (%)": round(cumulative_cost / start_value * 100, 6),
        })
        shares = {
            ticker: value_after_cost * weights[ticker] / float(row[ticker])
            for ticker in clean.columns
        }

    return pd.DataFrame(rows)


def _drawdown_event_row(strategy: str,
                        peak_date: pd.Timestamp,
                        trough_date: pd.Timestamp,
                        recovery_date: pd.Timestamp | pd.NaT,
                        depth: float) -> dict:
    recovered = not pd.isna(recovery_date)
    decline_days = int((trough_date - peak_date).days) if trough_date is not None else 0
    recovery_days = int((recovery_date - trough_date).days) if recovered and trough_date is not None else np.nan
    total_days = int((recovery_date - peak_date).days) if recovered else np.nan
    return {
        "Strategy": strategy,
        "Peak Date": peak_date.date().isoformat(),
        "Trough Date": trough_date.date().isoformat() if trough_date is not None else "",
        "Recovery Date": recovery_date.date().isoformat() if recovered else "",
        "Depth (%)": round(depth, 4),
        "Decline Days": decline_days,
        "Recovery Days": recovery_days,
        "Total Days": total_days,
        "Recovered": bool(recovered),
    }


def _normalised_weights(allocation: dict[str, float], columns: pd.Index) -> dict[str, float]:
    weights = {ticker: float(allocation[ticker]) for ticker in columns}
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Allocation weights must sum to a positive value.")
    return {ticker: weight / total for ticker, weight in weights.items()}


def _daily_returns(series: pd.Series) -> pd.Series:
    return series.dropna().astype(float).pct_change().dropna()


def _annualized_return(series: pd.Series) -> float:
    s = series.dropna().astype(float)
    if len(s) < 2:
        return np.nan
    years = max((s.index[-1] - s.index[0]).days / 365.25, 1 / 365.25)
    return float(((s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1) * 100)


def _safe_return(start_value: float, end_value: float) -> float:
    return ((end_value / start_value) - 1) * 100 if start_value else np.nan


def _calendar_calmar(start_value: float, end_value: float, dd_base: pd.Series) -> float:
    period_return = _safe_return(start_value, end_value)
    max_dd = float(drawdown_series(dd_base).min())
    if abs(max_dd) <= 1e-12:
        return np.nan
    return round(period_return / abs(max_dd), 4)


def _sharpe(returns: pd.Series, rf_annual: float) -> float:
    r = returns.dropna()
    if len(r) == 0 or r.std() < 1e-12:
        return np.nan
    rf_daily = (1 + rf_annual) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    return float(((r.mean() - rf_daily) / r.std()) * math.sqrt(TRADING_DAYS_PER_YEAR))


def _sortino(returns: pd.Series, rf_annual: float) -> float:
    r = returns.dropna()
    if len(r) == 0:
        return np.nan
    rf_daily = (1 + rf_annual) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    downside = r[r < rf_daily]
    if len(downside) == 0 or downside.std() < 1e-12:
        return np.nan
    return float(((r.mean() - rf_daily) / downside.std()) * math.sqrt(TRADING_DAYS_PER_YEAR))


def _beta_corr(returns: pd.Series, benchmark_returns: pd.Series | None) -> tuple[float, float]:
    if benchmark_returns is None:
        return np.nan, np.nan
    aligned = pd.concat([returns.rename("strategy"), benchmark_returns.rename("benchmark")], axis=1).dropna()
    if len(aligned) < 2 or aligned["benchmark"].var() < 1e-12:
        return np.nan, np.nan
    beta = float(aligned["strategy"].cov(aligned["benchmark"]) / aligned["benchmark"].var())
    corr = float(aligned["strategy"].corr(aligned["benchmark"]))
    return beta, corr


def _downside_beta(returns: pd.Series, benchmark_returns: pd.Series | None) -> float:
    if benchmark_returns is None:
        return np.nan
    aligned = pd.concat([returns.rename("strategy"), benchmark_returns.rename("benchmark")], axis=1).dropna()
    aligned = aligned[aligned["benchmark"] < 0]
    if len(aligned) < 2 or aligned["benchmark"].var() < 1e-12:
        return np.nan
    return float(aligned["strategy"].cov(aligned["benchmark"]) / aligned["benchmark"].var())


def _capture_ratios(returns: pd.Series, benchmark_returns: pd.Series | None) -> tuple[float, float]:
    if benchmark_returns is None:
        return np.nan, np.nan
    aligned = pd.concat([returns.rename("strategy"), benchmark_returns.rename("benchmark")], axis=1).dropna()
    up = aligned[aligned["benchmark"] > 0]
    down = aligned[aligned["benchmark"] < 0]
    up_capture = (up["strategy"].mean() / up["benchmark"].mean() * 100
                  if len(up) and abs(up["benchmark"].mean()) > 1e-12 else np.nan)
    down_capture = (down["strategy"].mean() / down["benchmark"].mean() * 100
                    if len(down) and abs(down["benchmark"].mean()) > 1e-12 else np.nan)
    return float(up_capture), float(down_capture)


def _slice_frame(values: pd.DataFrame,
                 start: pd.Timestamp | None,
                 end: pd.Timestamp | None) -> pd.DataFrame:
    out = values.copy()
    if start is not None:
        out = out.loc[out.index >= start]
    if end is not None:
        out = out.loc[out.index <= end]
    return out
