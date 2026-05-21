"""
strategy_plotting.py
====================
Matplotlib figure builders for the strategy-comparison notebook.

Each function receives plain DataFrames and returns a matplotlib Figure.
No marimo imports — callers handle mo.as_html() and mo.vstack() wrapping.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from engine.plotting import DARK_BG, PANEL_BG, TEXT_COL, GRID_COL, style_ax, order_strategies

COLORS: dict[str, str] = {
    "My Strategy (DIY)": "#58a6ff",
    "SPY 34/42 @ 30% cap": "#d2a8ff",
    "GLD 32/64 @ 30% cap": "#ffb86c",
    "SPY 32/42 + GLD 36/52 @ 30% cap": "#39c5cf",
    "S&P 500 (SPY)": "#f78166",
    "ALLW (Bridgewater)": "#f0b429",
    "60/40 (SPY/TLT)": "#3fb950",
}
STRATEGY_ORDER = list(COLORS)
TARGET_LEVERAGE_STRATEGIES = (
    "SPY 34/42 @ 30% cap",
    "GLD 32/64 @ 30% cap",
    "SPY 32/42 + GLD 36/52 @ 30% cap",
)
LEGACY_STRATEGY_RENAMES = {
    "SPY only 34/42 @ 30% cap": "SPY 34/42 @ 30% cap",
    "GLD only 32/64 @ 30% cap": "GLD 32/64 @ 30% cap",
}
DROPPED_LEGACY_STRATEGIES = {
    "SPY selective + GLD default 25% cap",
    "Full-grid top Calmar 30% cap",
}
UNKNOWN_STRATEGY_COLOR = "#8b949e"


def strategy_color(strategy: str) -> str:
    """Return a valid plotting color for known and unexpected strategy names."""
    return COLORS.get(strategy, UNKNOWN_STRATEGY_COLOR)


def clean_strategy_labels(
    data: pd.DataFrame,
    columns: tuple[str, ...] = ("Strategy", "Candidate", "Overlay Strategy"),
) -> pd.DataFrame:
    """Normalize renamed candidate labels and remove superseded leverage rows."""
    if data.empty:
        return data.copy()

    out = data.copy()
    for column in columns:
        if column not in out.columns:
            continue
        out[column] = out[column].replace(LEGACY_STRATEGY_RENAMES)
        out = out[~out[column].isin(DROPPED_LEGACY_STRATEGIES)]
    return out


def latest_bundle(roots: list[Path]) -> str:
    """Return the path of the most recently modified valid result bundle."""
    dirs: list[Path] = []
    for root in roots:
        if root.exists():
            dirs.extend(p for p in root.iterdir() if p.is_dir() and (p / "manifest.json").exists())
    return str(max(dirs, key=lambda p: p.stat().st_mtime)) if dirs else ""


def plot_growth(
    daily: pd.DataFrame,
    strategies: list[str] | tuple[str, ...] | None = None,
    full_history_scale: str = "log",
    overlap_scale: str = "linear",
    leverage_events: pd.DataFrame | None = None,
    show_leverage_events: bool = True,
) -> plt.Figure:
    """Two-panel growth-of-money chart with configurable strategies and y-scales."""
    if full_history_scale not in {"linear", "log"}:
        raise ValueError("full_history_scale must be 'linear' or 'log'.")
    if overlap_scale not in {"linear", "log"}:
        raise ValueError("overlap_scale must be 'linear' or 'log'.")

    value = daily.pivot(index="Date", columns="Strategy", values="Indexed Value")
    overlap = daily.pivot(index="Date", columns="Strategy", values="Overlap Indexed Value")
    if strategies is not None:
        selected = [strategy for strategy in strategies if strategy in value.columns]
        value = value[selected]
        overlap = overlap[[strategy for strategy in selected if strategy in overlap.columns]]

    fig, axes = plt.subplots(2, 1, figsize=(13, 8.5), sharex=False)
    fig.patch.set_facecolor(DARK_BG)

    for ax, data, title, ylabel, scale in [
        (axes[0], value, "Full available history", "Indexed value", full_history_scale),
        (axes[1], overlap, "ALLW overlap window", "Indexed value", overlap_scale),
    ]:
        style_ax(ax)
        for strategy in order_strategies(data.columns, STRATEGY_ORDER):
            s = data[strategy].dropna()
            if s.empty:
                continue
            ax.plot(s.index, s.values, color=strategy_color(strategy), lw=2.0, label=strategy)
            ax.annotate(
                f"{s.iloc[-1]:.0f}",
                xy=(s.index[-1], s.iloc[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                color=strategy_color(strategy),
                fontsize=8,
                va="center",
            )
        ax.set_title(title, fontsize=11, pad=8)
        ax.set_ylabel(ylabel)
        ax.set_yscale(scale)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
        if len(data.columns):
            ax.legend(fontsize=8, facecolor="#21262d", edgecolor=GRID_COL, labelcolor=TEXT_COL)
        if show_leverage_events and leverage_events is not None:
            _plot_leverage_events(ax, data, leverage_events)

    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=20, ha="right")
    plt.tight_layout(pad=1.4)
    return fig


def _plot_leverage_events(ax: plt.Axes, data: pd.DataFrame, events: pd.DataFrame) -> None:
    """Draw SPY/GLD entry/exit markers on the visible leverage candidate series."""
    if events.empty:
        return

    required = {"Date", "Candidate", "Ticker", "Event"}
    if not required.issubset(events.columns):
        return

    styles = {
        ("SPY", "Entry"): ("^", "#58a6ff"),
        ("SPY", "Exit"): ("v", "#58a6ff"),
        ("GLD", "Entry"): ("^", "#f0b429"),
        ("GLD", "Exit"): ("v", "#f0b429"),
    }
    used_labels: set[str] = set()
    event_data = events[events["Candidate"].isin(data.columns)].copy()
    if event_data.empty:
        return
    event_data["Date"] = pd.to_datetime(event_data["Date"])
    for (candidate, ticker, event), group in event_data.groupby(["Candidate", "Ticker", "Event"], sort=False):
        visible = data[candidate].dropna()
        if visible.empty or (ticker, event) not in styles:
            continue
        marker, color = styles[(ticker, event)]
        points = group[group["Date"].isin(visible.index)]
        if points.empty:
            continue
        y = visible.reindex(points["Date"]).to_numpy(dtype=float)
        label = f"{ticker} {event.lower()}"
        ax.scatter(
            points["Date"],
            y,
            marker=marker,
            s=115,
            color=color,
            edgecolor="#0d1117",
            linewidth=1.05,
            alpha=0.98,
            zorder=5,
            label=label if label not in used_labels else None,
        )
        used_labels.add(label)

    if used_labels:
        ax.legend(fontsize=8, facecolor="#21262d", edgecolor=GRID_COL, labelcolor=TEXT_COL)


def plot_drawdowns(daily: pd.DataFrame) -> plt.Figure:
    """Underwater drawdown curve for all strategies."""
    drawdowns = daily.pivot(index="Date", columns="Strategy", values="Drawdown (%)")
    fig, ax = plt.subplots(figsize=(13, 4.8))
    fig.patch.set_facecolor(DARK_BG)
    style_ax(ax)

    for strategy in order_strategies(drawdowns.columns, STRATEGY_ORDER):
        s = drawdowns[strategy].dropna()
        if not s.empty:
            ax.plot(s.index, s.values, color=strategy_color(strategy), lw=1, label=strategy)

    ax.axhline(0, color="#8b949e", lw=0.8)
    ax.set_title("Underwater drawdown curve", fontsize=11, pad=8)
    ax.set_ylabel("Drawdown (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(fontsize=8, facecolor="#21262d", edgecolor=GRID_COL, labelcolor=TEXT_COL)
    plt.tight_layout(pad=1.2)
    return fig


def plot_calendar_profile(calendar: pd.DataFrame) -> plt.Figure:
    """Multi-panel bar chart of calendar-year metrics."""
    metrics = [
        ("Return (%)", "Calendar return"),
        ("Max Drawdown (%)", "Max drawdown"),
        ("Max DD Duration (days)", "Max DD duration"),
        ("Calmar", "Calmar"),
    ]
    metrics = [(m, t) for m, t in metrics if m in calendar.columns]
    strategies = order_strategies(calendar["Strategy"].unique(), STRATEGY_ORDER)
    years = sorted(calendar["Year"].dropna().unique())
    x = np.arange(len(years))
    width = 0.8 / max(len(strategies), 1)

    fig, axes = plt.subplots(len(metrics), 1, figsize=(13, 3.3 * len(metrics)), sharex=True)
    fig.patch.set_facecolor(DARK_BG)
    axes = np.atleast_1d(axes)

    for ax, (metric, title) in zip(axes, metrics):
        style_ax(ax)
        for idx, strategy in enumerate(strategies):
            data = calendar[calendar["Strategy"] == strategy].set_index("Year")
            values = [data.loc[yr, metric] if yr in data.index else np.nan for yr in years]
            offset = (idx - (len(strategies) - 1) / 2) * width
            ax.bar(x + offset, values, width=width, color=strategy_color(strategy), alpha=0.85, label=strategy)
        ax.axhline(0, color="#8b949e", lw=0.7)
        ax.set_title(title, fontsize=10, pad=6)
        ax.set_ylabel(metric, fontsize=8)

    axes[0].legend(fontsize=8, facecolor="#21262d", edgecolor=GRID_COL, labelcolor=TEXT_COL)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([str(yr) for yr in years], rotation=25, ha="right")
    plt.tight_layout(pad=1.3)
    return fig


def plot_rolling_behaviour(rolling: pd.DataFrame) -> plt.Figure:
    """2×2 grid of rolling metrics (CAGR, max DD, correlation, beta)."""
    panels = [
        ("Rolling CAGR (%)", "Rolling CAGR"),
        ("Rolling Max Drawdown (%)", "Rolling max drawdown"),
        ("Rolling Corr to SPY", "Rolling correlation to SPY"),
        ("Rolling Beta to SPY", "Rolling beta to SPY"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.patch.set_facecolor(DARK_BG)

    for ax, (metric, title) in zip(axes.flat, panels):
        style_ax(ax)
        for strategy in order_strategies(rolling["Strategy"].unique(), STRATEGY_ORDER):
            for window, linestyle in [("1Y", "-"), ("3Y", "--")]:
                data = rolling[(rolling["Strategy"] == strategy) & (rolling["Window"] == window)]
                if data.empty or metric not in data:
                    continue
                ax.plot(
                    data["Date"], data[metric],
                    color=strategy_color(strategy), linestyle=linestyle,
                    lw=1.6, alpha=0.9, label=f"{strategy} {window}",
                )
        ax.axhline(0, color="#8b949e", lw=0.7)
        ax.set_title(title, fontsize=10, pad=6)
    axes[0, 0].legend(fontsize=7, facecolor="#21262d", edgecolor=GRID_COL, labelcolor=TEXT_COL, ncol=2)
    plt.tight_layout(pad=1.2)
    return fig


def plot_monthly_returns(monthly: pd.DataFrame) -> plt.Figure:
    """Monthly return heatmaps and distributions for each strategy."""
    strategies = order_strategies(monthly["Strategy"].unique(), STRATEGY_ORDER)
    fig, axes = plt.subplots(len(strategies), 2, figsize=(13, max(4, len(strategies) * 2.6)))
    fig.patch.set_facecolor(DARK_BG)
    if len(strategies) == 1:
        axes = np.array([axes])

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    for row, strategy in enumerate(strategies):
        data = monthly[monthly["Strategy"] == strategy]
        ax_hm, ax_hist = axes[row]
        pivot = data.pivot(index="Year", columns="Month", values="Return (%)").reindex(columns=range(1, 13))
        ax_hm.set_facecolor(PANEL_BG)
        arr = pivot.values
        valid = arr[~np.isnan(arr)]
        vmax = max(abs(valid).max(), 1) if len(valid) else 1
        ax_hm.imshow(arr, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
        ax_hm.set_xticks(range(12))
        ax_hm.set_xticklabels(month_labels, fontsize=7, color=TEXT_COL)
        ax_hm.set_yticks(range(len(pivot.index)))
        ax_hm.set_yticklabels(pivot.index.astype(str), fontsize=7, color=TEXT_COL)
        ax_hm.set_title(f"{strategy} monthly returns", color="white", fontsize=9)
        for spine in ax_hm.spines.values():
            spine.set_color(GRID_COL)

        ax_hist.set_facecolor(PANEL_BG)
        ax_hist.hist(data["Return (%)"].dropna(), bins=24, color=strategy_color(strategy), alpha=0.85)
        ax_hist.axvline(0, color="#8b949e", lw=0.8)
        ax_hist.tick_params(colors=TEXT_COL, labelsize=7)
        ax_hist.set_title(f"{strategy} monthly distribution", color="white", fontsize=9)
        for spine in ax_hist.spines.values():
            spine.set_color(GRID_COL)

    plt.tight_layout(pad=1.2)
    return fig


def plot_risk_diagnostics(summary: pd.DataFrame) -> plt.Figure:
    """3×4 institutional risk-metric grid for the ALLW overlap window."""
    overlap = summary[summary["Window"] == "ALLW Overlap"]
    if overlap.empty:
        overlap = summary[summary["Window"] == "Full History"]

    metrics = [
        "CAGR (%)", "Volatility (%)", "Sharpe", "Sortino",
        "Calmar", "Ulcer Index", "Max Drawdown (%)", "Max DD Duration (days)",
        "VaR 5% Daily (%)", "CVaR 5% Daily (%)", "Downside Beta", "Up Capture (%)",
    ]
    strategies = order_strategies(overlap["Strategy"].unique(), STRATEGY_ORDER)
    x = np.arange(len(strategies))

    fig, axes = plt.subplots(3, 4, figsize=(14, 9))
    fig.patch.set_facecolor(DARK_BG)

    for ax, metric in zip(axes.flat, metrics):
        style_ax(ax)
        vals = [
            overlap.loc[overlap["Strategy"] == s, metric].iloc[0]
            if metric in overlap and not overlap.loc[overlap["Strategy"] == s, metric].empty
            else np.nan
            for s in strategies
        ]
        ax.bar(x, vals, color=[strategy_color(s) for s in strategies], alpha=0.85)
        ax.set_title(metric, fontsize=8, pad=4)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace(" ", "\n", 1) for s in strategies], fontsize=6, color=TEXT_COL)

    plt.tight_layout(pad=1.0)
    return fig


def plot_implementation_realism(risk_contrib: pd.DataFrame, turnover: pd.DataFrame) -> plt.Figure:
    """Risk-contribution bar chart + cumulative cost-drag line chart."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    fig.patch.set_facecolor(DARK_BG)

    style_ax(axes[0])
    if not risk_contrib.empty:
        axes[0].bar(risk_contrib["Asset"], risk_contrib["Risk Contribution (%)"],
                    color="#58a6ff", alpha=0.85)
    axes[0].set_title("Risk contribution by asset", fontsize=10, pad=6)
    axes[0].set_ylabel("Risk contribution (%)")

    style_ax(axes[1])
    if not turnover.empty:
        axes[1].plot(turnover["Date"], turnover["Cumulative Cost Drag (%)"],
                     color="#f0b429", lw=2.0)
    axes[1].set_title("Estimated cumulative transaction-cost drag", fontsize=10, pad=6)
    axes[1].set_ylabel("Cost drag (%)")

    plt.tight_layout(pad=1.2)
    return fig
