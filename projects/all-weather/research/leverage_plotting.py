"""
leverage_plotting.py
====================
Matplotlib figure builders for the leverage-comparison notebook.

Each function receives plain DataFrames and returns a matplotlib Figure or
the return value of an axes-level helper (plot_grid_heatmap).
No marimo imports — callers handle mo.as_html() and mo.vstack() wrapping.
"""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from engine.plotting import DARK_BG, PANEL_BG, TEXT_COL, GRID_COL, style_ax
from research.leverage_analysis import BASE, BENCHMARK, PALETTE, strategy_label


def colour_map(strategies) -> dict[str, str]:
    """Assign a consistent colour to each strategy string."""
    colors = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(strategies)}
    if BASE in colors:
        colors[BASE] = "#58a6ff"
    if BENCHMARK in colors:
        colors[BENCHMARK] = "#f78166"
    return colors


def plot_grid_heatmap(ax: plt.Axes, data: pd.DataFrame, ticker: str, metric: str,
                      entry=None, exit_=None):
    """Render a leverage-vs-threshold heatmap onto ax and return the image artist."""
    if entry is not None:
        heatmap_data = data[(data["Ticker"] == ticker) & (data["Entry Threshold"] == float(entry))]
        pivot = heatmap_data.pivot_table(
            index="Overlay Weight (%)", columns="Exit Threshold", values=metric, aggfunc="max",
        ).sort_index(ascending=True)
        xlabel, suffix = "Exit RSI", f"entry {entry:g}"
    else:
        heatmap_data = data[(data["Ticker"] == ticker) & (data["Exit Threshold"] == float(exit_))]
        pivot = heatmap_data.pivot_table(
            index="Overlay Weight (%)", columns="Entry Threshold", values=metric, aggfunc="max",
        ).sort_index(ascending=True)
        xlabel, suffix = "Entry RSI", f"exit {exit_:g}"

    ax.set_facecolor(PANEL_BG)
    image = ax.imshow(pivot.values.astype(float), aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{x:g}" for x in pivot.columns], color=TEXT_COL, fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{x:g}" for x in pivot.index], color=TEXT_COL, fontsize=7)
    ax.set_xlabel(xlabel, color=TEXT_COL)
    ax.set_ylabel("Overlay weight (%)", color=TEXT_COL)
    ax.set_title(f"{ticker} {metric} ({suffix})", color="white", fontsize=10, pad=6)
    ax.tick_params(colors=TEXT_COL)
    for spine in ax.spines.values():
        spine.set_color(GRID_COL)
    return image


def plot_growth_and_drawdown_figure(
    filtered_daily: pd.DataFrame,
    selected_strategies: list[str],
) -> plt.Figure:
    """Growth-of-money (log) + drawdown two-panel figure for selected strategies."""
    colors = colour_map(selected_strategies)
    data = filtered_daily[filtered_daily["Strategy"].isin(selected_strategies)]
    values = data.pivot(index="Date", columns="Strategy", values="Value")
    indexed = values / values.ffill().bfill().iloc[0] * 100.0
    drawdowns = (values / values.cummax() - 1.0) * 100.0

    fig, axes = plt.subplots(2, 1, figsize=(13, 8.2), sharex=True)
    fig.patch.set_facecolor(DARK_BG)

    style_ax(axes[0])
    for s in selected_strategies:
        if s not in indexed:
            continue
        series = indexed[s].dropna()
        if series.empty:
            continue
        axes[0].plot(series.index, series.values, color=colors.get(s), lw=1.6, label=strategy_label(s))
        axes[0].annotate(
            f"{series.iloc[-1]:.0f}",
            xy=(series.index[-1], series.iloc[-1]),
            xytext=(6, 0), textcoords="offset points",
            color=colors.get(s, TEXT_COL), fontsize=8, va="center",
        )
    axes[0].set_yscale("log")
    axes[0].set_title("Growth of money, rebased to selected window", fontsize=11, pad=8)
    axes[0].set_ylabel("Indexed value")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    axes[0].legend(fontsize=8, facecolor="#21262d", edgecolor=GRID_COL, labelcolor=TEXT_COL)

    style_ax(axes[1])
    for s in selected_strategies:
        if s not in drawdowns:
            continue
        series = drawdowns[s].dropna()
        if not series.empty:
            axes[1].plot(series.index, series.values, color=colors.get(s), lw=0.8, label=strategy_label(s))
    axes[1].axhline(0, color="#8b949e", lw=0.8)
    axes[1].set_title("Drawdown inside selected window", fontsize=11, pad=8)
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout(pad=1.2)
    return fig


def plot_default_overlay_figure(
    default_rank: pd.DataFrame,
    overlay_summary: pd.DataFrame,
) -> plt.Figure:
    """Bar chart comparison (CAGR, Calmar, MaxDD) of the default overlay rule across ETFs."""
    base = overlay_summary[overlay_summary["Ticker"] == "BASE"].head(1)
    data = default_rank.sort_values("Ticker")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    fig.patch.set_facecolor(DARK_BG)

    for ax, (metric, title, invert) in zip(axes, [
        ("CAGR (%)", "CAGR (%)", False),
        ("Calmar", "Calmar", False),
        ("Max Drawdown (%)", "Max drawdown (%)", True),
    ]):
        style_ax(ax)
        colors = ["#3fb950" if t == "GLD" else "#58a6ff" for t in data["Ticker"]]
        ax.bar(data["Ticker"], data[metric], color=colors, alpha=0.9)
        if not base.empty and metric in base:
            base_val = float(base[metric].iloc[0])
            ax.axhline(base_val, color="#f0b429", lw=1.2, linestyle="--", label="Base")
            ax.legend(fontsize=7, facecolor="#21262d", edgecolor=GRID_COL, labelcolor=TEXT_COL)
        ax.set_title(title, fontsize=10, pad=6)
        if invert:
            ax.set_ylim(
                min(data[metric].min(), base[metric].iloc[0] if not base.empty else data[metric].min()) * 1.12,
                0,
            )
    plt.tight_layout(pad=1.0)
    return fig


def plot_oos_validation_figure(oos_summary: pd.DataFrame) -> plt.Figure:
    """OOS delta bar chart for the default 30/50 rule across all ETFs and splits."""
    default = oos_summary[oos_summary["Selector"] == "default_30_50_20"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    fig.patch.set_facecolor(DARK_BG)

    for ax, (metric, title) in zip(axes, [
        ("OOS Calmar Delta", "Default overlay: OOS Calmar delta"),
        ("OOS MaxDD Delta (%)", "Default overlay: OOS MaxDD delta, pp"),
    ]):
        style_ax(ax)
        pivot = default.pivot_table(index="Ticker", columns="Split", values=metric, aggfunc="mean").sort_index()
        x = np.arange(len(pivot.index))
        width = 0.22
        for i, split in enumerate(pivot.columns):
            ax.bar(x + (i - 1) * width, pivot[split], width=width, label=str(split))
        ax.axhline(0, color="#8b949e", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, color=TEXT_COL)
        ax.set_title(title, fontsize=10, pad=6)
        ax.legend(fontsize=7, facecolor="#21262d", edgecolor=GRID_COL, labelcolor=TEXT_COL)
    plt.tight_layout(pad=1.0)
    return fig


def plot_etf_oos_bars_figure(
    oos_df: pd.DataFrame,
    etf_name: str,
    metrics: list[tuple[str, str]],
) -> plt.Figure:
    """Grouped bar chart of OOS metric deltas per split for a single ETF."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(13, 4.6))
    fig.patch.set_facecolor(DARK_BG)
    axes = np.atleast_1d(axes)

    for ax, (metric, title) in zip(axes, metrics):
        style_ax(ax)
        pivot = oos_df.pivot_table(index="Split", columns="Rule", values=metric, aggfunc="mean").sort_index()
        x = np.arange(len(pivot.index))
        width = 0.32
        for i, rule in enumerate(pivot.columns):
            ax.bar(x + (i - 0.5) * width, pivot[rule], width=width, label=rule)
        ax.axhline(0, color="#8b949e", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, color=TEXT_COL)
        ax.set_title(title, fontsize=10, pad=6)
        ax.legend(fontsize=7, facecolor="#21262d", edgecolor=GRID_COL, labelcolor=TEXT_COL)
    plt.tight_layout(pad=1.0)
    return fig


def plot_grid_heatmaps_figure(
    threshold_grid: pd.DataFrame,
    ticker: str,
    heatmap_specs: list[tuple],
) -> plt.Figure:
    """One subplot per (metric, entry/exit) spec; returns figure with colorbars added."""
    fig, axes = plt.subplots(1, len(heatmap_specs), figsize=(13, 4.8))
    fig.patch.set_facecolor(DARK_BG)
    axes = np.atleast_1d(axes)

    for ax, spec in zip(axes, heatmap_specs):
        metric, entry, exit_ = spec
        image = plot_grid_heatmap(ax, threshold_grid, ticker, metric, entry=entry, exit_=exit_)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(colors=TEXT_COL)
    plt.tight_layout(pad=1.0)
    return fig


def plot_all_etf_rsi_figure(
    signals: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> plt.Figure:
    """3×2 small-multiples grid: RSI + overlay exposure for each ETF."""
    tickers = sorted(signals["Ticker"].dropna().unique())
    fig, axes = plt.subplots(3, 2, figsize=(14, 10.5), sharex=True)
    fig.patch.set_facecolor(DARK_BG)

    for ax, ticker in zip(axes.flat, tickers):
        style_ax(ax)
        sig = signals[signals["Ticker"] == ticker]
        diag = diagnostics[diagnostics["Ticker"] == ticker]
        ax.plot(sig["Date"], sig["RSI"], color="#f0b429", lw=1.1, label="RSI")
        if not sig.empty:
            ax.axhline(float(sig["Entry Threshold"].iloc[0]), color="#ff7b72", lw=0.8, linestyle="--")
            ax.axhline(float(sig["Exit Threshold"].iloc[0]), color="#3fb950", lw=0.8, linestyle="--")
        ax2 = ax.twinx()
        ax2.set_facecolor("none")
        ax2.tick_params(colors=TEXT_COL, labelsize=7)
        for spine in ax2.spines.values():
            spine.set_color(GRID_COL)
        if not diag.empty:
            ax2.fill_between(diag["Date"], diag["Overlay Exposure"] * 100, 0,
                             color="#58a6ff", alpha=0.22, label="Overlay exposure")
        ax2.set_ylim(0, max(25, float(diag["Overlay Exposure"].max() * 120) if not diag.empty else 25))
        ax.set_ylim(0, 100)
        ax.set_title(ticker, fontsize=10, pad=5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    for ax in axes.flat[len(tickers):]:
        ax.set_visible(False)
    plt.tight_layout(pad=1.0)
    return fig


def plot_yearly_figure(
    data: pd.DataFrame,
    strategies: list[str],
) -> plt.Figure:
    """Year-by-year overlay metrics for selected strategies."""
    metrics = [
        ("Annual Return (%)", "Annual return"),
        ("Annual Calmar", "Annual Calmar"),
        ("Max Drawdown (%)", "Annual max drawdown"),
        ("Annual Volatility (%)", "Annual volatility"),
        ("Active Days", "Active overlay days"),
        ("Overlay Return Contribution (%)", "Overlay contribution"),
    ]
    colors = colour_map(strategies)
    fig, axes = plt.subplots(3, 2, figsize=(14, 10.5), sharex=False)
    fig.patch.set_facecolor(DARK_BG)

    for ax, (metric, title) in zip(axes.flat, metrics):
        style_ax(ax)
        for strat in strategies:
            s = data[data["Strategy"] == strat].sort_values("Year")
            if s.empty or metric not in s:
                continue
            ax.plot(s["Year"], s[metric], marker="o", lw=1.6,
                    color=colors.get(strat), label=strategy_label(strat))
        ax.axhline(0, color="#8b949e", lw=0.7)
        ax.set_title(title, fontsize=10, pad=6)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    axes[0, 0].legend(fontsize=7, facecolor="#21262d", edgecolor=GRID_COL, labelcolor=TEXT_COL, ncol=2)
    plt.tight_layout(pad=1.1)
    return fig


def plot_threshold_heatmap_figure(
    pivot: pd.DataFrame,
    inspect_etf: str,
    metric: str,
    xlabel: str,
    ylabel: str,
    title_suffix: str,
) -> plt.Figure:
    """Standalone heatmap for the appendix threshold-grid inspector."""
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)

    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad(color="#21262d")
    image = ax.imshow(np.ma.masked_invalid(pivot.values.astype(float)), aspect="auto", cmap=cmap)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{x:g}" for x in pivot.columns], color=TEXT_COL)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{x:g}" for x in pivot.index], color=TEXT_COL)
    ax.set_xlabel(xlabel, color=TEXT_COL)
    ax.set_ylabel(ylabel, color=TEXT_COL)
    ax.set_title(f"{inspect_etf} leverage grid: {metric} ({title_suffix})", color="white", pad=8)
    ax.tick_params(colors=TEXT_COL)
    for spine in ax.spines.values():
        spine.set_color(GRID_COL)

    cbar = fig.colorbar(image, ax=ax)
    cbar.ax.tick_params(colors=TEXT_COL)
    plt.tight_layout(pad=1.0)
    return fig
