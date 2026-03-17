"""
plotting.py
===========
All matplotlib visualisation. One public function: plot_backtest().

The style_ax() helper applies a consistent dark theme to any axes object
and is used here and in validation.py.

This module knows nothing about file I/O, optimisation, or real holdings --
it only receives data and produces charts.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from backtest import StrategyStats
import config


def style_ax(ax):
    """Apply consistent dark theme to a matplotlib axes object."""
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#c9d1d9", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.label.set_color("#c9d1d9")
    ax.xaxis.label.set_color("#c9d1d9")
    ax.title.set_color("white")
    ax.grid(axis="y", color="#30363d", alpha=0.5, linewidth=0.7)


def plot_backtest(backtest: pd.DataFrame,
                  stats_list: list[StrategyStats],
                  results_dir: str,
                  label: str):
    """
    Two-panel dark-theme backtest chart saved to results_dir/backtest.png.

    Panel 1 -- Portfolio value over time for all three strategies.
               Includes a final-value annotation with Calmar ratio.

    Panel 2 -- Annual returns bar chart for all three strategies.
               Positive bars use the strategy colour, negative bars use red/amber.
    """
    fig = plt.figure(figsize=(15, 13))
    fig.patch.set_facecolor("#0d1117")

    gs  = fig.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.48)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    COLORS = {
        "aw":  "#58a6ff",   # blue  -- All Weather rebalanced
        "bh":  "#f0b429",   # amber -- Buy & Hold All Weather
        "spy": "#f78166",   # coral -- S&P 500
    }

    for ax in [ax1, ax2]:
        style_ax(ax)

    # ── Panel 1: Portfolio value over time ──────────────────────────────────
    ax1.plot(backtest.index, backtest["All Weather Value"],
             color=COLORS["aw"], lw=2.2,
             label="All Weather (Rebalanced monthly)")
    ax1.plot(backtest.index, backtest["Buy & Hold All Weather"],
             color=COLORS["bh"], lw=2.0, linestyle=(0, (4, 2)),
             label="Buy & Hold All Weather (never rebalanced)")
    ax1.plot(backtest.index, backtest["S&P 500 Value"],
             color=COLORS["spy"], lw=1.6, linestyle="--", alpha=0.85,
             label="S&P 500 (SPY)")
    ax1.fill_between(backtest.index, backtest["All Weather Value"],
                     alpha=0.08, color=COLORS["aw"])

    ax1.set_title(
        f"Portfolio Backtest -- {label}  "
        f"({config.BACKTEST_START} to {config.BACKTEST_END})",
        fontsize=13, pad=10
    )
    ax1.set_ylabel("Portfolio Value ($)", fontsize=10)
    ax1.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.legend(fontsize=9, facecolor="#21262d", edgecolor="#30363d",
               labelcolor="white", loc="upper left")

    s_aw, s_bh, s_spy = stats_list
    ax1.text(
        0.99, 0.06,
        f"Rebalanced: ${s_aw.final_value:,.0f}   Calmar={s_aw.calmar:.2f}     "
        f"Buy & Hold: ${s_bh.final_value:,.0f}     "
        f"S&P 500: ${s_spy.final_value:,.0f}",
        transform=ax1.transAxes, ha="right", va="bottom",
        color="#8b949e", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#21262d",
                  edgecolor="#30363d", alpha=0.9)
    )

    # ── Panel 2: Annual returns bar chart ───────────────────────────────────
    def annual_returns(col: str) -> pd.Series:
        return backtest[col].resample("YE").last().pct_change().dropna() * 100

    aw_ann  = annual_returns("All Weather Value")
    bh_ann  = annual_returns("Buy & Hold All Weather")
    spy_ann = annual_returns("S&P 500 Value")

    years_idx = aw_ann.index
    x         = np.arange(len(years_idx))
    width     = 0.26

    ax2.bar(x - width, aw_ann.reindex(years_idx, fill_value=0).values,
            width, label="All Weather (Rebal.)",
            color=[COLORS["aw"] if v >= 0 else "#f85149"
                   for v in aw_ann.reindex(years_idx, fill_value=0).values],
            alpha=0.85)
    ax2.bar(x, bh_ann.reindex(years_idx, fill_value=0).values,
            width, label="Buy & Hold AW",
            color=[COLORS["bh"] if v >= 0 else "#b08800"
                   for v in bh_ann.reindex(years_idx, fill_value=0).values],
            alpha=0.85)
    ax2.bar(x + width, spy_ann.reindex(years_idx, fill_value=0).values,
            width, label="S&P 500",
            color=[COLORS["spy"] if v >= 0 else "#da3633"
                   for v in spy_ann.reindex(years_idx, fill_value=0).values],
            alpha=0.70)

    ax2.set_xticks(x)
    ax2.set_xticklabels([d.year for d in years_idx], rotation=45, fontsize=8)
    ax2.axhline(0, color="#8b949e", lw=0.8)
    ax2.set_ylabel("Annual Return (%)", fontsize=10)
    ax2.set_title("Annual Returns by Year -- All Three Strategies",
                  fontsize=11, pad=8)
    ax2.legend(fontsize=8, facecolor="#21262d", edgecolor="#30363d",
               labelcolor="white", ncol=3)

    save_path = os.path.join(results_dir, "backtest.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  Plot saved -> {save_path}")
    plt.show()
