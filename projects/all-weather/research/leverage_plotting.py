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


# ---------------------------------------------------------------------------
# Mixed SPY+GLD OOS validation figures
# ---------------------------------------------------------------------------


def _short_label(name: str, name_col: str, max_len: int = 24) -> str:
    """Shorten a candidate/selector name for chart labels."""
    from research.leverage_analysis import label_selector
    if name_col == "Selector":
        return label_selector(name)
    return name if len(name) <= max_len else name[:max_len - 1] + "…"


def plot_mixed_oos_delta_bars_figure(
    oos_df: pd.DataFrame,
    name_col: str,
    title_prefix: str,
    metrics: list[tuple[str, str]] | None = None,
) -> plt.Figure:
    """Grouped bar chart of OOS deltas per candidate/selector, split by OOS window."""
    if metrics is None:
        metrics = [
            ("OOS Calmar Delta", "Calmar delta"),
            ("OOS MaxDD Delta (%)", "MaxDD delta (pp)"),
            ("OOS CAGR Delta (%)", "CAGR delta (pp)"),
        ]
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(14, 4.6))
    fig.patch.set_facecolor(DARK_BG)
    axes = np.atleast_1d(axes)

    df = oos_df.copy()
    df["_label"] = df[name_col].apply(lambda n: _short_label(n, name_col))

    for ax, (metric, title) in zip(axes, metrics):
        style_ax(ax)
        pivot = df.pivot_table(
            index="_label", columns="Split", values=metric, aggfunc="mean",
        ).sort_index()
        x = np.arange(len(pivot.index))
        n_splits = len(pivot.columns)
        width = 0.7 / max(n_splits, 1)
        for i, split in enumerate(sorted(pivot.columns)):
            offset = (i - (n_splits - 1) / 2) * width
            ax.bar(x + offset, pivot[split], width=width, label=str(split))
        ax.axhline(0, color="#8b949e", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, color=TEXT_COL, rotation=25, ha="right", fontsize=7)
        ax.set_title(f"{title_prefix}: {title}", fontsize=10, pad=6)
        ax.legend(fontsize=7, facecolor="#21262d", edgecolor=GRID_COL, labelcolor=TEXT_COL)
    plt.tight_layout(pad=1.0)
    return fig


def plot_mixed_is_vs_oos_figure(
    oos_df: pd.DataFrame,
    name_col: str,
    title: str = "IS vs OOS Calmar — Overfitting Check",
) -> plt.Figure:
    """Paired bar chart comparing IS and OOS Calmar for each name × split."""
    fig, ax = plt.subplots(figsize=(14, 5.2))
    fig.patch.set_facecolor(DARK_BG)
    style_ax(ax)

    df = oos_df.copy().sort_values([name_col, "Split"])
    df["_label"] = df[name_col].apply(lambda n: _short_label(n, name_col))
    labels = [f"{row['_label']}\n{row['Split']}" for _, row in df.iterrows()]
    x = np.arange(len(df))
    width = 0.35
    is_vals = df["IS Calmar"].astype(float).values
    oos_vals = df["OOS Overlay Calmar"].astype(float).values

    ax.bar(x - width / 2, is_vals, width, label="IS Calmar", color=PALETTE[0], alpha=0.55)
    ax.bar(x + width / 2, oos_vals, width, label="OOS Calmar", color=PALETTE[0])

    if "OOS Base Calmar" in df.columns:
        base_vals = df["OOS Base Calmar"].astype(float).values
        for i, bv in enumerate(base_vals):
            ax.plot(
                [i - width, i + width], [bv, bv],
                color="#f0b429", lw=1.0, linestyle="--",
            )
        ax.plot([], [], color="#f0b429", linestyle="--", label="OOS Base Calmar")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=TEXT_COL, fontsize=7, rotation=0, ha="center")
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_ylabel("Calmar ratio")
    ax.legend(fontsize=8, facecolor="#21262d", edgecolor=GRID_COL, labelcolor=TEXT_COL)
    plt.tight_layout(pad=1.2)
    return fig


def plot_mixed_growth_figure(
    daily: pd.DataFrame,
    split: str,
    strategies: list[str] | None = None,
) -> plt.Figure:
    """Growth + drawdown for mixed overlay strategies in one OOS split."""
    _split_val = int(split) if str(split).isdigit() else split
    data = daily[daily["Split"] == _split_val].copy()
    if data.empty:
        fig, ax = plt.subplots(figsize=(13, 4))
        fig.patch.set_facecolor(DARK_BG)
        style_ax(ax)
        ax.set_title(f"No daily data for split {split}", fontsize=10)
        return fig

    data["Date"] = pd.to_datetime(data["Date"])

    base_rows = data[data["Selector"] == "base"]
    base_label = base_rows["Strategy"].iloc[0] if not base_rows.empty else None

    if strategies is None:
        all_strats = data["Overlay Strategy"].dropna().unique().tolist()
        non_base = [s for s in all_strats if s != base_label]
        finals = {}
        for s in non_base:
            sub = data[data["Overlay Strategy"] == s].sort_values("Date")
            if not sub.empty:
                finals[s] = float(sub["Value"].iloc[-1])
        top = sorted(finals, key=finals.get, reverse=True)[:4]
        strategies = ([base_label] if base_label else []) + top

    strat_col = "Overlay Strategy" if "Overlay Strategy" in data.columns else "Strategy"
    data = data[data[strat_col].isin(strategies)]
    values = data.pivot_table(index="Date", columns=strat_col, values="Value", aggfunc="first")
    indexed = values / values.ffill().bfill().iloc[0] * 100.0
    drawdowns = (values / values.cummax() - 1.0) * 100.0

    short_labels = {s: (s[:35] + "…" if len(s) > 35 else s) for s in strategies}
    colors = colour_map(strategies)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8.2), sharex=True)
    fig.patch.set_facecolor(DARK_BG)

    style_ax(axes[0])
    for s in strategies:
        if s not in indexed:
            continue
        series = indexed[s].dropna()
        if series.empty:
            continue
        axes[0].plot(series.index, series.values, color=colors.get(s), lw=1.6, label=short_labels[s])
        axes[0].annotate(
            f"{series.iloc[-1]:.0f}",
            xy=(series.index[-1], series.iloc[-1]),
            xytext=(6, 0), textcoords="offset points",
            color=colors.get(s, TEXT_COL), fontsize=8, va="center",
        )
    axes[0].set_yscale("log")
    axes[0].set_title(f"Growth of $100 — OOS split {split}", fontsize=11, pad=8)
    axes[0].set_ylabel("Indexed value")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    axes[0].legend(fontsize=7, facecolor="#21262d", edgecolor=GRID_COL, labelcolor=TEXT_COL)

    style_ax(axes[1])
    for s in strategies:
        if s not in drawdowns:
            continue
        series = drawdowns[s].dropna()
        if not series.empty:
            axes[1].plot(series.index, series.values, color=colors.get(s), lw=0.8, label=short_labels[s])
    axes[1].axhline(0, color="#8b949e", lw=0.8)
    axes[1].set_title(f"Drawdown — OOS split {split}", fontsize=11, pad=8)
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout(pad=1.2)
    return fig


def plot_walk_forward_heatmap_figure(walk_forward: pd.DataFrame) -> plt.Figure:
    """Selector x year heatmap showing whether annual walk-forward Calmar improved."""
    fig, ax = plt.subplots(figsize=(13, 4.8))
    fig.patch.set_facecolor(DARK_BG)
    style_ax(ax)
    if walk_forward.empty or "Calmar Improvement" not in walk_forward:
        ax.set_title("No walk-forward data", fontsize=10)
        return fig

    data = walk_forward.copy()
    if "Is Partial Year" in data:
        data = data[~data["Is Partial Year"].astype(bool)]
    data["Calmar Improvement"] = data["Calmar Improvement"].astype(str).str.lower().isin(["true", "1"])
    pivot = data.pivot_table(
        index="Selector",
        columns="Year",
        values="Calmar Improvement",
        aggfunc="max",
    ).sort_index()
    values = pivot.astype(float).values
    cmap = plt.get_cmap("RdYlGn").copy()
    image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(int(x)) for x in pivot.columns], color=TEXT_COL, fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([_short_label(str(x), "Selector", max_len=28) for x in pivot.index], color=TEXT_COL, fontsize=8)
    ax.set_title("Annual walk-forward Calmar improvement", fontsize=11, pad=8)
    ax.set_xlabel("Evaluation year")
    ax.set_ylabel("Selector")
    cbar = fig.colorbar(image, ax=ax, ticks=[0, 1])
    cbar.ax.set_yticklabels(["No", "Yes"], color=TEXT_COL)
    cbar.ax.tick_params(colors=TEXT_COL)
    plt.tight_layout(pad=1.1)
    return fig


def plot_calmar_maxdd_scatter_figure(candidates: pd.DataFrame) -> plt.Figure:
    """Scatter plot of Calmar improvement against drawdown impact."""
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    fig.patch.set_facecolor(DARK_BG)
    style_ax(ax)
    required = {"Worst OOS Calmar Delta", "Worst OOS MaxDD Delta (%)"}
    if candidates.empty or not required <= set(candidates.columns):
        ax.set_title("No Calmar/MaxDD candidate data", fontsize=10)
        return fig

    data = candidates.copy()
    x = pd.to_numeric(data["Worst OOS Calmar Delta"], errors="coerce")
    y = pd.to_numeric(data["Worst OOS MaxDD Delta (%)"], errors="coerce")
    size = pd.to_numeric(data.get("Average OOS Exposure (%)", 1.0), errors="coerce").fillna(1.0)
    passes = (
        data["Overall Pass"].fillna(False).astype(bool)
        if "Overall Pass" in data
        else pd.Series(False, index=data.index)
    )
    colors = np.where(passes, "#3fb950", "#f78166")
    sizes = 60 + np.clip(size, 0, 15) * 18
    ax.scatter(x, y, s=sizes, c=colors, alpha=0.82, edgecolor="#0d1117", linewidth=0.8)
    for _, row in data.iterrows():
        label = _short_label(str(row.get("Candidate", row.get("Selector", ""))), "Candidate Name", max_len=28)
        xv = pd.to_numeric(row.get("Worst OOS Calmar Delta"), errors="coerce")
        yv = pd.to_numeric(row.get("Worst OOS MaxDD Delta (%)"), errors="coerce")
        if pd.notna(xv) and pd.notna(yv):
            ax.annotate(label, xy=(float(xv), float(yv)), xytext=(5, 4),
                        textcoords="offset points", color=TEXT_COL, fontsize=7)
    ax.axvline(0, color="#8b949e", lw=0.8)
    ax.axhline(0, color="#8b949e", lw=0.8)
    ax.set_title("Candidate frontier: worst Calmar delta vs worst MaxDD delta", fontsize=11, pad=8)
    ax.set_xlabel("Worst OOS Calmar delta vs base")
    ax.set_ylabel("Worst OOS MaxDD delta vs base (pp)")
    plt.tight_layout(pad=1.0)
    return fig


def plot_exposure_timeline_figure(
    diagnostics: pd.DataFrame,
    strategy: str | None = None,
) -> plt.Figure:
    """Gross exposure and SPY/GLD sleeve timeline for one mixed overlay strategy."""
    fig, ax = plt.subplots(figsize=(13, 4.8))
    fig.patch.set_facecolor(DARK_BG)
    style_ax(ax)
    if diagnostics.empty:
        ax.set_title("No exposure diagnostics loaded", fontsize=10)
        return fig

    data = diagnostics.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    if strategy and "Overlay Strategy" in data:
        data = data[data["Overlay Strategy"] == strategy]
    elif "Overlay Strategy" in data:
        strategy = data["Overlay Strategy"].dropna().iloc[0]
        data = data[data["Overlay Strategy"] == strategy]
    if data.empty:
        ax.set_title("No exposure diagnostics for selected strategy", fontsize=10)
        return fig

    ax.plot(data["Date"], data["Gross Exposure"] * 100, color="#f0b429", lw=1.4, label="Gross exposure")
    if "SPY Position" in data:
        ax.fill_between(data["Date"], data["SPY Position"] * 100, 0, color="#58a6ff", alpha=0.28, label="SPY sleeve")
    if "GLD Position" in data:
        ax.fill_between(data["Date"], data["GLD Position"] * 100, 0, color="#3fb950", alpha=0.28, label="GLD sleeve")
    ax.axhline(100, color="#8b949e", lw=0.8)
    ax.set_title(f"Daily overlay exposure — {_short_label(str(strategy), 'Candidate Name', max_len=48)}", fontsize=11, pad=8)
    ax.set_ylabel("Exposure (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(fontsize=8, facecolor="#21262d", edgecolor=GRID_COL, labelcolor=TEXT_COL)
    plt.tight_layout(pad=1.1)
    return fig


def plot_trade_episode_distribution_figure(episodes: pd.DataFrame) -> plt.Figure:
    """Trade episode count and contribution distribution for selected candidates."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    fig.patch.set_facecolor(DARK_BG)
    for ax in axes:
        style_ax(ax)
    if episodes.empty:
        axes[0].set_title("No trade episode data", fontsize=10)
        axes[1].set_visible(False)
        return fig

    data = episodes.copy()
    name_col = "Overlay Strategy" if "Overlay Strategy" in data else "Ticker"
    group = data.groupby(name_col, as_index=False).agg(
        Episodes=("Episode", "count"),
        Median_Days=("Trading Days", "median"),
        Total_Contribution=("Overlay Return Contribution (%)", "sum"),
    )
    group = group.sort_values("Total_Contribution", ascending=False).head(10)
    labels = [_short_label(str(x), "Candidate Name", max_len=26) for x in group[name_col]]
    y = np.arange(len(group))
    axes[0].barh(y, group["Episodes"], color="#58a6ff")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, color=TEXT_COL, fontsize=7)
    axes[0].invert_yaxis()
    axes[0].set_title("Trade episode count", fontsize=10, pad=6)
    axes[1].barh(y, group["Total_Contribution"], color="#3fb950")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(labels, color=TEXT_COL, fontsize=7)
    axes[1].invert_yaxis()
    axes[1].axvline(0, color="#8b949e", lw=0.8)
    axes[1].set_title("Total overlay contribution (%)", fontsize=10, pad=6)
    plt.tight_layout(pad=1.0)
    return fig
