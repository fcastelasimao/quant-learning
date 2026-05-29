"""
research/rebalance_thresholds.py
================================
Compare threshold-based rebalancing rules on the production allocation.

The live rebalancer has two independent concepts:
  1. How a drift threshold is measured.
  2. What happens when a threshold is breached.

For this strategy, once any ETF breaches the selected drift threshold, the
whole portfolio is rebalanced back to target weights. We do not intentionally
hold cash.

This module compares:
  - absolute 5 percentage points of portfolio weight
  - 5% of each ETF target weight
  - max(1 percentage point, 15% of target weight)
  - max(1 percentage point, 20% of target weight)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/allweather-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine import config
from engine.analytics import apply_annual_fee, build_monthly_rebalanced_series
from engine.calendar import pandas_resample_frequency
from engine.data import fetch_prices, get_price_provenance
from engine.plotting import DARK_BG, GRID_COL, PANEL_BG, TEXT_COL, style_ax
from engine.stats import (
    DAYS_PER_YEAR,
    compute_avg_drawdown,
    compute_avg_recovery_time,
    compute_cagr,
    compute_calmar,
    compute_max_drawdown,
    compute_max_drawdown_duration,
    compute_sharpe,
    compute_sortino,
    compute_ulcer_index,
)


@dataclass(frozen=True)
class DriftPolicy:
    """One threshold definition."""

    name: str
    kind: Literal["absolute", "relative", "hybrid"]
    absolute_threshold: float = 0.0
    relative_threshold: float = 0.0
    floor_threshold: float = 0.0

    def threshold_for(self, target_weight: float) -> float:
        if self.kind == "absolute":
            return self.absolute_threshold
        if self.kind == "relative":
            return target_weight * self.relative_threshold
        if self.kind == "hybrid":
            return max(self.floor_threshold, target_weight * self.relative_threshold)
        raise ValueError(f"Unknown drift policy kind: {self.kind}")


DEFAULT_POLICIES = (
    DriftPolicy(
        name="absolute_5pp_portfolio",
        kind="absolute",
        absolute_threshold=0.05,
    ),
    DriftPolicy(
        name="relative_5pct_target",
        kind="relative",
        relative_threshold=0.05,
    ),
    DriftPolicy(
        name="hybrid_1pp_or_15pct_target",
        kind="hybrid",
        relative_threshold=0.15,
        floor_threshold=0.01,
    ),
    DriftPolicy(
        name="hybrid_1pp_or_20pct_target",
        kind="hybrid",
        relative_threshold=0.20,
        floor_threshold=0.01,
    ),
)

SERIES_LABELS = {
    "buy_and_hold | none": "Buy & Hold",
    "monthly_full_rebalance | full_on_breach": "Monthly full",
    "absolute_5pp_portfolio | full_on_breach": "5pp portfolio",
    "relative_5pct_target | full_on_breach": "5% target",
    "hybrid_1pp_or_15pct_target | full_on_breach": "Hybrid 15%",
    "hybrid_1pp_or_20pct_target | full_on_breach": "Hybrid 20%",
}
BENCHMARK_LABELS = {
    "ALLW": "ALLW ETF",
    "SPY": "SPY",
    "60/40": "60/40",
}

SERIES_COLORS = {
    "Buy & Hold": "#f0b429",
    "Monthly full": "#8b949e",
    "5pp portfolio": "#58a6ff",
    "5% target": "#f78166",
    "Hybrid 15%": "#d2a8ff",
    "Hybrid 20%": "#3fb950",
    "ALLW ETF": "#ffa657",
    "SPY": "#ff7b72",
    "60/40": "#7ee787",
}

SERIES_ORDER = [
    "5pp portfolio",
    "Hybrid 20%",
    "Hybrid 15%",
    "5% target",
    "Monthly full",
    "Buy & Hold",
    "ALLW ETF",
    "SPY",
    "60/40",
]
DEFAULT_DAILY_OVERLAP_START = "2025-03-06"
ALLW_ANNUAL_FEE = 0.0085


def monthly_prices(prices: pd.DataFrame, allocation: dict[str, float]) -> pd.DataFrame:
    """Return month-end prices for allocation tickers."""
    tickers = list(allocation)
    freq = pandas_resample_frequency(config.DATA_FREQUENCY)
    out = prices[tickers].resample(freq).last().dropna()
    if out.empty:
        raise ValueError("No overlapping monthly prices for threshold rebalance test.")
    return out


def _portfolio_values(
    holdings: dict[str, float],
    cash: float,
    row: pd.Series,
) -> tuple[dict[str, float], float]:
    values = {ticker: holdings[ticker] * float(row[ticker]) for ticker in holdings}
    return values, cash + sum(values.values())


def simulate_threshold_rebalance(
    prices: pd.DataFrame,
    allocation: dict[str, float],
    policy: DriftPolicy,
    *,
    start_value: float = 10_000.0,
    transaction_cost_pct: float = 0.0,
    tax_drag_pct: float = 0.0,
) -> pd.DataFrame:
    """Simulate one threshold policy: full rebalance when any ETF breaches."""
    monthly = monthly_prices(prices, allocation)
    first = monthly.iloc[0]
    holdings = {
        ticker: (start_value * weight) / float(first[ticker])
        for ticker, weight in allocation.items()
    }
    records: list[dict] = []

    for when, row in monthly.iterrows():
        values, value = _portfolio_values(holdings, 0.0, row)
        if tax_drag_pct > 0 and when.month == 12:
            after_tax = value * (1.0 - tax_drag_pct)
            scale = after_tax / value if value > 0 else 1.0
            holdings = {ticker: shares * scale for ticker, shares in holdings.items()}
            values, value = _portfolio_values(holdings, 0.0, row)

        weights = {ticker: values[ticker] / value if value > 0 else 0.0 for ticker in allocation}
        drifts = {ticker: weights[ticker] - allocation[ticker] for ticker in allocation}
        thresholds = {ticker: policy.threshold_for(allocation[ticker]) for ticker in allocation}
        breached = {
            ticker: abs(drifts[ticker]) > thresholds[ticker]
            for ticker in allocation
        }

        trade_notional = 0.0
        rebalanced = any(breached.values())
        max_abs_drift = max(abs(drift) for drift in drifts.values())
        max_relative_drift = max(
            abs(drifts[ticker]) / allocation[ticker]
            for ticker in allocation
            if allocation[ticker] > 0
        )

        if rebalanced:
            target_values = {ticker: value * weight for ticker, weight in allocation.items()}
            trade_notional = sum(abs(target_values[ticker] - values[ticker]) for ticker in allocation)
            value_after_cost = max(0.0, value - trade_notional * transaction_cost_pct)
            holdings = {
                ticker: (value_after_cost * allocation[ticker]) / float(row[ticker])
                for ticker in allocation
            }

        values_after, value_after = _portfolio_values(holdings, 0.0, row)
        weights_after = {
            ticker: values_after[ticker] / value_after if value_after > 0 else 0.0
            for ticker in allocation
        }

        record = {
            "Date": when,
            "Policy": policy.name,
            "Rebalance Action": "full_on_breach",
            "Value": value_after,
            "Cash": 0.0,
            "Cash Before Trade": 0.0,
            "Rebalanced": bool(rebalanced),
            "Breached Count": sum(1 for is_breached in breached.values() if is_breached),
            "Trade Notional": trade_notional,
            "Max Abs Drift Before (%)": max_abs_drift * 100,
            "Max Relative Drift Before (%)": max_relative_drift * 100,
        }
        for ticker in allocation:
            record[f"{ticker} Weight Before (%)"] = weights[ticker] * 100
            record[f"{ticker} Weight After (%)"] = weights_after[ticker] * 100
            record[f"{ticker} Drift Before (%)"] = drifts[ticker] * 100
            record[f"{ticker} Threshold (%)"] = thresholds[ticker] * 100
            record[f"{ticker} Breached"] = breached[ticker]
        records.append(record)

    df = pd.DataFrame(records).set_index("Date")
    df["Monthly Ret (%)"] = df["Value"].pct_change() * 100
    return df


def simulate_monthly_full_rebalance(
    prices: pd.DataFrame,
    allocation: dict[str, float],
    *,
    start_value: float = 10_000.0,
    transaction_cost_pct: float = 0.0,
    tax_drag_pct: float = 0.0,
) -> pd.DataFrame:
    """Baseline: rebalance fully every month, matching the main engine style."""
    policy = DriftPolicy("monthly_full_rebalance", kind="absolute", absolute_threshold=0.0)
    return simulate_threshold_rebalance(
        prices,
        allocation,
        policy,
        start_value=start_value,
        transaction_cost_pct=transaction_cost_pct,
        tax_drag_pct=tax_drag_pct,
    ).assign(Policy="monthly_full_rebalance")


def simulate_buy_and_hold(
    prices: pd.DataFrame,
    allocation: dict[str, float],
    *,
    start_value: float = 10_000.0,
) -> pd.DataFrame:
    """Baseline: start at target weights and never trade."""
    monthly = monthly_prices(prices, allocation)
    first = monthly.iloc[0]
    holdings = {
        ticker: (start_value * weight) / float(first[ticker])
        for ticker, weight in allocation.items()
    }
    records = []
    for when, row in monthly.iterrows():
        values, value = _portfolio_values(holdings, 0.0, row)
        weights = {ticker: values[ticker] / value if value > 0 else 0.0 for ticker in allocation}
        record = {
            "Date": when,
            "Policy": "buy_and_hold",
            "Rebalance Action": "none",
            "Value": value,
            "Cash": 0.0,
            "Cash Before Trade": 0.0,
            "Rebalanced": False,
            "Breached Count": 0,
            "Trade Notional": 0.0,
            "Max Abs Drift Before (%)": max(abs(weights[t] - allocation[t]) for t in allocation) * 100,
            "Max Relative Drift Before (%)": max(
                abs(weights[t] - allocation[t]) / allocation[t]
                for t in allocation
                if allocation[t] > 0
            ) * 100,
        }
        for ticker in allocation:
            drift = weights[ticker] - allocation[ticker]
            record[f"{ticker} Weight Before (%)"] = weights[ticker] * 100
            record[f"{ticker} Weight After (%)"] = weights[ticker] * 100
            record[f"{ticker} Drift Before (%)"] = drift * 100
            record[f"{ticker} Threshold (%)"] = 0.0
            record[f"{ticker} Breached"] = False
        records.append(record)
    df = pd.DataFrame(records).set_index("Date")
    df["Monthly Ret (%)"] = df["Value"].pct_change() * 100
    return df


def simulate_threshold_rebalance_daily(
    prices: pd.DataFrame,
    allocation: dict[str, float],
    policy: DriftPolicy,
    *,
    start_value: float = 10_000.0,
    transaction_cost_pct: float = 0.0,
) -> pd.Series:
    """Daily-resolution equivalent of simulate_threshold_rebalance.

    Drift is checked every trading day and a full rebalance fires the same day
    the threshold is breached. Returns one equity series indexed by the price
    DataFrame's dates. Used for chart visualisation where benchmark series are
    already at daily resolution.
    """
    tickers = list(allocation)
    daily = prices[tickers].dropna()
    if daily.empty:
        return pd.Series(dtype=float, name=policy.name)

    first = daily.iloc[0]
    holdings = {
        ticker: (start_value * weight) / float(first[ticker])
        for ticker, weight in allocation.items()
    }
    thresholds = {ticker: policy.threshold_for(allocation[ticker]) for ticker in tickers}
    out: dict[pd.Timestamp, float] = {}
    for when, row in daily.iterrows():
        portfolio = {ticker: holdings[ticker] * float(row[ticker]) for ticker in tickers}
        total = sum(portfolio.values())
        if total <= 0:
            out[when] = 0.0
            continue
        weights = {ticker: portfolio[ticker] / total for ticker in tickers}
        breached = any(
            abs(weights[ticker] - allocation[ticker]) > thresholds[ticker]
            for ticker in tickers
        )
        if breached:
            target = {ticker: total * allocation[ticker] for ticker in tickers}
            trade = sum(abs(target[ticker] - portfolio[ticker]) for ticker in tickers)
            total = max(0.0, total - trade * transaction_cost_pct)
            holdings = {
                ticker: (total * allocation[ticker]) / float(row[ticker])
                for ticker in tickers
            }
        out[when] = total
    return pd.Series(out, name=policy.name).sort_index()


def simulate_buy_and_hold_daily(
    prices: pd.DataFrame,
    allocation: dict[str, float],
    *,
    start_value: float = 10_000.0,
) -> pd.Series:
    """Daily-resolution buy-and-hold for the same allocation."""
    tickers = list(allocation)
    daily = prices[tickers].dropna()
    if daily.empty:
        return pd.Series(dtype=float, name="buy_and_hold")
    first = daily.iloc[0]
    holdings = pd.Series({
        ticker: (start_value * weight) / float(first[ticker])
        for ticker, weight in allocation.items()
    })
    series = (daily[tickers] * holdings).sum(axis=1)
    series.name = "buy_and_hold"
    return series


def simulate_monthly_full_rebalance_daily(
    prices: pd.DataFrame,
    allocation: dict[str, float],
    *,
    start_value: float = 10_000.0,
    transaction_cost_pct: float = 0.0,
) -> pd.Series:
    """Daily equity series for a monthly full rebalance baseline.

    Equity is recorded every trading day; a full rebalance to target weights
    fires only on the last trading day of each calendar month.
    """
    tickers = list(allocation)
    daily = prices[tickers].dropna()
    if daily.empty:
        return pd.Series(dtype=float, name="monthly_full_rebalance")
    first = daily.iloc[0]
    holdings = {
        ticker: (start_value * weight) / float(first[ticker])
        for ticker, weight in allocation.items()
    }
    month_ends = set(daily.resample("ME").last().index)
    out: dict[pd.Timestamp, float] = {}
    for when, row in daily.iterrows():
        portfolio = {ticker: holdings[ticker] * float(row[ticker]) for ticker in tickers}
        total = sum(portfolio.values())
        if total > 0 and when in month_ends:
            target = {ticker: total * allocation[ticker] for ticker in tickers}
            trade = sum(abs(target[ticker] - portfolio[ticker]) for ticker in tickers)
            total = max(0.0, total - trade * transaction_cost_pct)
            holdings = {
                ticker: (total * allocation[ticker]) / float(row[ticker])
                for ticker in tickers
            }
        out[when] = total
    return pd.Series(out, name="monthly_full_rebalance").sort_index()


def run_daily_policy_grid(
    prices: pd.DataFrame,
    allocation: dict[str, float],
    *,
    start_value: float = 10_000.0,
    transaction_cost_pct: float = 0.0,
    policies: tuple[DriftPolicy, ...] = DEFAULT_POLICIES,
) -> pd.DataFrame:
    """Run buy & hold, monthly full, and all named policies at daily resolution.

    Returns a wide DataFrame keyed by Date with one column per policy series key
    (matching the keys in SERIES_LABELS).
    """
    series: dict[str, pd.Series] = {}
    series["buy_and_hold | none"] = simulate_buy_and_hold_daily(
        prices, allocation, start_value=start_value
    )
    series["monthly_full_rebalance | full_on_breach"] = simulate_monthly_full_rebalance_daily(
        prices,
        allocation,
        start_value=start_value,
        transaction_cost_pct=transaction_cost_pct,
    )
    for policy in policies:
        series[f"{policy.name} | full_on_breach"] = simulate_threshold_rebalance_daily(
            prices,
            allocation,
            policy,
            start_value=start_value,
            transaction_cost_pct=transaction_cost_pct,
        )
    frame = pd.concat(series, axis=1)
    frame.index.name = "Date"
    return frame


def build_daily_overlap_benchmarks(
    prices: pd.DataFrame,
    *,
    start_value: float = 10_000.0,
    allw_fee: float = ALLW_ANNUAL_FEE,
) -> pd.DataFrame:
    """Build ALLW, SPY, and 60/40 benchmark value series for the overlap chart."""
    series: dict[str, pd.Series] = {}

    if "ALLW" in prices:
        allw = prices["ALLW"].dropna().astype(float)
        if not allw.empty:
            values = allw / allw.iloc[0] * start_value
            series["ALLW"] = apply_annual_fee(values.rename("ALLW"), allw_fee)

    if "SPY" in prices:
        spy = prices["SPY"].dropna().astype(float)
        if not spy.empty:
            series["SPY"] = (spy / spy.iloc[0] * start_value).rename("SPY")

    if {"SPY", "TLT"}.issubset(prices.columns):
        sixty_forty = prices[["SPY", "TLT"]].dropna()
        if not sixty_forty.empty:
            series["60/40"] = build_monthly_rebalanced_series(
                sixty_forty,
                {"SPY": 0.60, "TLT": 0.40},
                start_value=start_value,
            ).rename("60/40")

    if not series:
        return pd.DataFrame()
    frame = pd.concat(series, axis=1)
    frame.index.name = "Date"
    return frame


def summarize_result(df: pd.DataFrame) -> dict[str, float | int | str]:
    """Build one metric row for a simulated policy."""
    years = (df.index[-1] - df.index[0]).days / DAYS_PER_YEAR
    value = df["Value"]
    cagr = round(compute_cagr(value, years), 2)
    mdd = round(compute_max_drawdown(value), 2)
    ulcer = compute_ulcer_index(value)
    turnover = float(df["Trade Notional"].sum())
    return {
        "Policy": str(df["Policy"].iloc[0]),
        "Rebalance Action": str(df["Rebalance Action"].iloc[0]),
        "Period Years": round(years, 1),
        "Final Value": round(float(value.iloc[-1]), 2),
        "CAGR (%)": cagr,
        "Max Drawdown (%)": mdd,
        "Sharpe": round(compute_sharpe(df["Monthly Ret (%)"], rf_annual=config.RISK_FREE_RATE), 3),
        "Sortino": compute_sortino(df["Monthly Ret (%)"], rf_annual=config.RISK_FREE_RATE),
        "Calmar": round(compute_calmar(cagr, mdd), 3),
        "Martin": round(cagr / ulcer if ulcer > 1e-10 else cagr, 3),
        "Ulcer Index": ulcer,
        "Avg Drawdown (%)": compute_avg_drawdown(value),
        "Max DD Duration (months)": compute_max_drawdown_duration(value),
        "Avg Recovery (months)": compute_avg_recovery_time(value),
        "Rebalance Count": int(df["Rebalanced"].sum()),
        "Total Turnover $": round(turnover, 2),
        "Avg Annual Turnover $": round(turnover / years, 2) if years > 0 else 0.0,
        "Avg Trade Notional $": round(
            float(df.loc[df["Trade Notional"] > 0, "Trade Notional"].mean() or 0.0),
            2,
        ),
        "Max Abs Drift Before (%)": round(float(df["Max Abs Drift Before (%)"].max()), 2),
        "Max Relative Drift Before (%)": round(float(df["Max Relative Drift Before (%)"].max()), 2),
        "Avg Cash $": round(float(df["Cash"].mean()), 2),
    }


def rolling_policy_metrics(
    diagnostics: pd.DataFrame,
    *,
    windows_months: tuple[int, ...] = (36, 60),
) -> pd.DataFrame:
    """Compute rolling window metrics for each threshold policy series."""
    rows: list[dict] = []
    data = diagnostics.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    data.sort_values(["Series", "Date"], inplace=True)

    for series_name, group in data.groupby("Series", sort=False):
        group = group.set_index("Date").sort_index()
        label = SERIES_LABELS.get(series_name, series_name)
        for window in windows_months:
            if len(group) < window:
                continue
            for end_idx in range(window - 1, len(group)):
                chunk = group.iloc[end_idx - window + 1:end_idx + 1]
                values = chunk["Value"]
                years = (values.index[-1] - values.index[0]).days / DAYS_PER_YEAR
                if years <= 0 or values.iloc[0] <= 0:
                    continue
                cagr = compute_cagr(values, years)
                mdd = compute_max_drawdown(values)
                turnover = float(chunk["Trade Notional"].sum())
                avg_equity = float(values.mean())
                rows.append({
                    "Date": values.index[-1],
                    "Series": series_name,
                    "Label": label,
                    "Window Months": window,
                    "Window": f"{window // 12}Y" if window % 12 == 0 else f"{window}M",
                    "Rolling CAGR (%)": round(cagr, 2),
                    "Rolling Max Drawdown (%)": round(mdd, 2),
                    "Rolling Calmar": round(compute_calmar(cagr, mdd), 3),
                    "Rolling Rebalance Count": int(chunk["Rebalanced"].sum()),
                    "Rolling Turnover $": round(turnover, 2),
                    "Rolling Turnover / Avg Equity (%)": round(
                        turnover / avg_equity * 100 if avg_equity > 0 else 0.0,
                        2,
                    ),
                    "Rolling Max Abs Drift Before (%)": round(float(chunk["Max Abs Drift Before (%)"].max()), 2),
                    "Rolling Max Relative Drift Before (%)": round(
                        float(chunk["Max Relative Drift Before (%)"].max()),
                        2,
                    ),
                })
    return pd.DataFrame(rows)


def _ordered_labels(labels) -> list[str]:  # noqa: ANN001
    present = set(labels)
    return [label for label in SERIES_ORDER if label in present] + sorted(present - set(SERIES_ORDER))


def _series_color(label: str) -> str:
    return SERIES_COLORS.get(label, "#8b949e")


def _endpoint_offsets(endpoints: list[tuple[str, float]]) -> dict[str, int]:
    """Return small text offsets that keep clustered endpoint labels readable."""
    valid = [(label, value) for label, value in endpoints if pd.notna(value) and value > 0]
    offsets = {label: 0 for label, _ in endpoints}
    if len(valid) < 2:
        return offsets

    clusters: list[list[tuple[str, float]]] = []
    for item in sorted(valid, key=lambda pair: pair[1]):
        if not clusters:
            clusters.append([item])
            continue
        _, last_value = clusters[-1][-1]
        if abs(item[1] / last_value - 1.0) <= 0.08:
            clusters[-1].append(item)
        else:
            clusters.append([item])

    for cluster in clusters:
        if len(cluster) == 1:
            continue
        midpoint = (len(cluster) - 1) / 2
        for idx, (label, _) in enumerate(cluster):
            offsets[label] = int(round((idx - midpoint) * 12))
    return offsets


def plot_threshold_growth(values: pd.DataFrame, out_dir: Path) -> Path:
    """Save an indexed growth chart for all policies."""
    indexed = values / values.iloc[0] * 100
    indexed = indexed.rename(columns=lambda c: SERIES_LABELS.get(c, c))
    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    fig.patch.set_facecolor(DARK_BG)
    style_ax(ax)

    endpoints: list[tuple[str, pd.Timestamp, float]] = []
    for label in _ordered_labels(indexed.columns):
        series = indexed[label].dropna()
        if series.empty:
            continue
        lw = 2.2 if label in {"5pp portfolio", "Hybrid 20%", "Hybrid 15%"} else 1.5
        alpha = 1.0 if label in {"5pp portfolio", "Hybrid 20%", "Hybrid 15%"} else 0.72
        ax.plot(series.index, series.values, label=label, color=_series_color(label), lw=lw, alpha=alpha)
        endpoints.append((label, series.index[-1], float(series.iloc[-1])))

    label_offsets = _endpoint_offsets([(label, value) for label, _, value in endpoints])
    for label, when, value in endpoints:
        ax.annotate(
            f"{value:.0f}",
            xy=(when, value),
            xytext=(6, label_offsets.get(label, 0)),
            textcoords="offset points",
            color=_series_color(label),
            fontsize=8,
            va="center",
        )

    ax.set_title("Threshold Rebalance Policies - Growth of $100", fontsize=12, pad=10)
    ax.set_ylabel("Indexed value")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL, ncol=3)
    fig.tight_layout(pad=1.2)
    out = out_dir / "threshold_growth.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def plot_daily_policy_overlap(
    values: pd.DataFrame,
    out_dir: Path,
    *,
    start_date: str,
) -> Path:
    """Save the daily-resolution policy chart for the ALLW overlap window."""
    if values.empty:
        raise ValueError("No daily overlap values available for plotting.")

    indexed = values / values.iloc[0] * 100
    indexed = indexed.rename(
        columns=lambda c: SERIES_LABELS.get(
            BENCHMARK_LABELS.get(c, c),
            BENCHMARK_LABELS.get(c, c),
        )
    )
    fig, ax = plt.subplots(figsize=(13.5, 6.4))
    fig.patch.set_facecolor(DARK_BG)
    style_ax(ax)

    endpoints: list[tuple[str, pd.Timestamp, float, bool]] = []
    for label in _ordered_labels(indexed.columns):
        series = indexed[label].dropna()
        if series.empty:
            continue
        is_benchmark = label in set(BENCHMARK_LABELS.values())
        ax.plot(
            series.index,
            series.values,
            label=label,
            color=_series_color(label),
            lw=2.3 if is_benchmark else 1.5,
            alpha=0.95 if is_benchmark else 0.72,
            linestyle="--" if is_benchmark else "-",
        )
        endpoints.append((label, series.index[-1], float(series.iloc[-1]), is_benchmark))

    label_offsets = _endpoint_offsets([(label, value) for label, _, value, _ in endpoints])
    for label, when, value, is_benchmark in endpoints:
        ax.annotate(
            f"{value:.1f}",
            xy=(when, value),
            xytext=(6, label_offsets.get(label, 0)),
            textcoords="offset points",
            color=_series_color(label),
            fontsize=8,
            va="center",
            fontweight="bold" if is_benchmark else "normal",
        )

    ax.set_title(
        f"Threshold Rebalance Policies - ALLW Overlap Daily Since {start_date}",
        fontsize=12,
        pad=10,
    )
    ax.set_ylabel("Indexed value")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL, ncol=3)
    fig.tight_layout(pad=1.2)
    out = out_dir / "threshold_allw_overlap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def plot_rolling_metrics(rolling: pd.DataFrame, out_dir: Path, *, window: str = "5Y") -> Path:
    """Save a rolling performance/behavior plot for one window."""
    data = rolling[rolling["Window"] == window].copy()
    if data.empty:
        raise ValueError(f"No rolling metrics for window {window}")

    panels = [
        ("Rolling CAGR (%)", "Rolling CAGR"),
        ("Rolling Max Drawdown (%)", "Rolling max drawdown"),
        ("Rolling Calmar", "Rolling Calmar"),
        ("Rolling Rebalance Count", "Rolling rebalance count"),
        ("Rolling Turnover / Avg Equity (%)", "Rolling turnover / avg equity"),
        ("Rolling Max Relative Drift Before (%)", "Worst relative drift before rebalance"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
    fig.patch.set_facecolor(DARK_BG)

    for ax, (metric, title) in zip(axes.flat, panels):
        style_ax(ax)
        for label in _ordered_labels(data["Label"].unique()):
            s = data[data["Label"] == label].sort_values("Date")
            if s.empty:
                continue
            ax.plot(s["Date"], s[metric], color=_series_color(label), lw=1.7, label=label)
        ax.set_title(f"{title} ({window})", fontsize=10, pad=7)
        if "%" in metric or "Drawdown" in metric:
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        if metric == "Rolling Max Drawdown (%)":
            ax.axhline(0, color="#8b949e", lw=0.7)

    axes[0, 0].legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL, ncol=2)
    for ax in axes[-1, :]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout(pad=1.4)
    out = out_dir / f"threshold_rolling_{window.lower()}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def run_threshold_comparison(
    prices: pd.DataFrame,
    allocation: dict[str, float],
    *,
    start_value: float,
    transaction_cost_pct: float,
    tax_drag_pct: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return summary, wide values, and long diagnostics."""
    diagnostics = [
        simulate_buy_and_hold(prices, allocation, start_value=start_value),
        simulate_monthly_full_rebalance(
            prices,
            allocation,
            start_value=start_value,
            transaction_cost_pct=transaction_cost_pct,
            tax_drag_pct=tax_drag_pct,
        ),
    ]
    for policy in DEFAULT_POLICIES:
        diagnostics.append(
            simulate_threshold_rebalance(
                prices,
                allocation,
                policy,
                start_value=start_value,
                transaction_cost_pct=transaction_cost_pct,
                tax_drag_pct=tax_drag_pct,
            )
        )

    summary = pd.DataFrame([summarize_result(df) for df in diagnostics])
    long = pd.concat(diagnostics).reset_index()
    long["Series"] = long["Policy"] + " | " + long["Rebalance Action"]
    values = long.pivot(index="Date", columns="Series", values="Value").sort_index()
    return summary, values, long


def _load_strategy_allocation(strategy_id: str) -> tuple[str, dict[str, float]]:
    payload = config.load_strategy(strategy_id)
    canonical = config.resolve_strategy_id(strategy_id)
    return canonical, {ticker: float(weight) for ticker, weight in payload["allocation"].items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest threshold rebalance policies.")
    parser.add_argument("--strategy-id", default=config.DEFAULT_STRATEGY)
    parser.add_argument("--start-date", default=config.BACKTEST_START)
    parser.add_argument("--end-date", default=config.BACKTEST_END)
    parser.add_argument("--data-source", choices=("yfinance", "fmp"), default=config.DATA_SOURCE)
    parser.add_argument("--fmp-price-column", default=config.FMP_PRICE_COLUMN,
                        choices=("open", "high", "low", "close", "adj_close"))
    parser.add_argument("--pricing-model", choices=("total_return", "price_return"),
                        default=config.PRICING_MODEL)
    parser.add_argument("--transaction-cost-pct", type=float, default=config.TRANSACTION_COST_PCT)
    parser.add_argument("--tax-drag-pct", type=float, default=config.TAX_DRAG_PCT)
    parser.add_argument("--start-value", type=float, default=config.INITIAL_PORTFOLIO_VALUE)
    parser.add_argument("--rolling-windows-months", type=int, nargs="+", default=[36, 60])
    parser.add_argument(
        "--daily-overlap-start",
        default=DEFAULT_DAILY_OVERLAP_START,
        help="Start date for the precomputed daily ALLW-overlap rebalance chart.",
    )
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "results" / "rebalance_thresholds"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.DATA_SOURCE = args.data_source
    config.FMP_PRICE_COLUMN = args.fmp_price_column
    config.PRICING_MODEL = args.pricing_model

    canonical_strategy, allocation = _load_strategy_allocation(args.strategy_id)
    prices = fetch_prices(list(allocation), args.start_date, args.end_date)
    benchmark_frames: list[pd.DataFrame] = []
    for ticker in ("ALLW", "SPY", "TLT"):
        if ticker in prices.columns:
            continue
        try:
            benchmark_frames.append(fetch_prices([ticker], args.start_date, args.end_date))
        except Exception as exc:  # noqa: BLE001 - optional benchmark availability varies by source
            print(f"Warning: could not load optional benchmark {ticker}: {exc}")
    all_prices = (
        pd.concat([prices, *benchmark_frames], axis=1)
        .loc[:, lambda frame: ~frame.columns.duplicated()]
        .sort_index()
    )
    summary, values, diagnostics = run_threshold_comparison(
        prices,
        allocation,
        start_value=args.start_value,
        transaction_cost_pct=args.transaction_cost_pct,
        tax_drag_pct=args.tax_drag_pct,
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(args.output_root) / f"{timestamp}_{canonical_strategy}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "threshold_summary.csv", index=False)
    values.to_csv(out_dir / "threshold_values.csv")
    diagnostics.to_csv(out_dir / "threshold_diagnostics.csv", index=False)
    rolling = rolling_policy_metrics(
        diagnostics,
        windows_months=tuple(args.rolling_windows_months),
    )
    rolling.to_csv(out_dir / "threshold_rolling_metrics.csv", index=False)
    plot_paths = [plot_threshold_growth(values, out_dir)]
    daily_overlap_start = pd.Timestamp(args.daily_overlap_start)
    daily_overlap_prices = all_prices.loc[all_prices.index >= daily_overlap_start]
    daily_overlap_values = pd.DataFrame()
    if not daily_overlap_prices.empty:
        daily_overlap_values = run_daily_policy_grid(
            daily_overlap_prices,
            allocation,
            start_value=args.start_value,
            transaction_cost_pct=args.transaction_cost_pct,
        )
        daily_overlap_benchmarks = build_daily_overlap_benchmarks(
            daily_overlap_prices,
            start_value=args.start_value,
        )
        daily_overlap_plot_values = pd.concat(
            [daily_overlap_values, daily_overlap_benchmarks],
            axis=1,
        ).dropna(how="all")
        daily_overlap_values.to_csv(out_dir / "threshold_allw_overlap_daily_values.csv")
        daily_overlap_benchmarks.to_csv(out_dir / "threshold_allw_overlap_benchmarks.csv")
        daily_overlap_plot_values.to_csv(out_dir / "threshold_allw_overlap_plot_values.csv")
        plot_paths.append(
            plot_daily_policy_overlap(
                daily_overlap_plot_values,
                out_dir,
                start_date=args.daily_overlap_start,
            )
        )
    for window in sorted(rolling["Window"].unique()):
        plot_paths.append(plot_rolling_metrics(rolling, out_dir, window=window))
    (out_dir / "run_config.json").write_text(json.dumps({
        "strategy_id": canonical_strategy,
        "allocation": allocation,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "data_source": args.data_source,
        "fmp_price_column": args.fmp_price_column,
        "pricing_model": args.pricing_model,
        "rebalance_action": "full_on_breach",
        "cash_policy": "fully_invested",
        "transaction_cost_pct": args.transaction_cost_pct,
        "tax_drag_pct": args.tax_drag_pct,
        "start_value": args.start_value,
        "daily_overlap_start": args.daily_overlap_start,
        "daily_overlap_rows": int(len(daily_overlap_values)),
        "rolling_windows_months": args.rolling_windows_months,
            "price_provenance": get_price_provenance(all_prices),
    }, indent=2))

    print(f"Results written to: {out_dir}")
    display_cols = [
        "Policy",
        "Rebalance Action",
        "Final Value",
        "CAGR (%)",
        "Max Drawdown (%)",
        "Calmar",
        "Rebalance Count",
        "Total Turnover $",
        "Max Abs Drift Before (%)",
        "Max Relative Drift Before (%)",
    ]
    print(summary[display_cols].to_string(index=False))
    print("\nPlots:")
    for path in plot_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
