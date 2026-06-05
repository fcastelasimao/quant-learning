"""
research/plot_all_findings.py
=============================
Generate a summary visualisation for every research investigation.
Reads from existing result CSVs where available, falls back to hardcoded
numbers from the findings docs for investigations without on-disk artifacts.

Output: research/<investigation>/findings_plot.png for each investigation,
        plus research/findings_overview.png (combined summary).

Run:
    conda run -n allweather python research/plot_all_findings.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Style constants (match engine/plotting.py dark theme) ────────────────────

DARK_BG = "#0d1117"
PANEL_BG = "#161b22"
TEXT_COL = "#c9d1d9"
GRID_COL = "#30363d"
ACCENT = "#58a6ff"
GREEN = "#3fb950"
RED = "#f85149"
ORANGE = "#d29922"
PURPLE = "#bc8cff"
CYAN = "#39d2c0"

VERDICT_COLORS = {
    "closed": RED,
    "reopened": ORANGE,
    "active-research": PURPLE,
    "production": GREEN,
    "production (gated)": CYAN,
    "todo": GRID_COL,
}


def style_ax(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COL, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID_COL)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.label.set_color(TEXT_COL)
    ax.xaxis.label.set_color(TEXT_COL)
    ax.title.set_color("white")
    ax.grid(axis="y", color=GRID_COL, alpha=0.5, linewidth=0.7)


def save_fig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Universe Selection
# ─────────────────────────────────────────────────────────────────────────────

def plot_universe_selection(out_dir):
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=DARK_BG)
    style_ax(ax)

    universes = ["6-asset\n(production)", "8-asset A\n(no TIP)", "8-asset B\n(with TIP)"]
    windows = ["2018 OOS", "2020 OOS", "2022 OOS"]
    data = {
        "6-asset\n(production)": [0.452, 0.457, 0.376],
        "8-asset A\n(no TIP)":   [0.368, 0.385, 0.345],
        "8-asset B\n(with TIP)": [0.395, 0.432, 0.359],
    }

    x = np.arange(len(windows))
    width = 0.25
    colors = [GREEN, RED, ORANGE]

    for i, (univ, calmars) in enumerate(data.items()):
        bars = ax.bar(x + i * width, calmars, width, label=univ, color=colors[i], alpha=0.85)
        for bar, val in zip(bars, calmars):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", color=TEXT_COL, fontsize=7)

    ax.set_xticks(x + width)
    ax.set_xticklabels(windows)
    ax.set_ylabel("Calmar Ratio")
    ax.set_title("Universe Selection: 6-asset wins all OOS windows", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COL,
              labelcolor=TEXT_COL)
    ax.set_ylim(0, 0.55)
    save_fig(fig, out_dir / "findings_plot.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Optimiser Comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_optimiser_comparison(out_dir):
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=DARK_BG)
    style_ax(ax)

    categories = ["IS Calmar\n(2006-2020)", "OOS Calmar\n(2020-2026)"]
    de_vals = [0.72, 0.15]
    rp_vals = [0.45, 0.45]

    x = np.arange(len(categories))
    width = 0.3

    ax.bar(x - width/2, de_vals, width, label="Differential Evolution", color=RED, alpha=0.85)
    ax.bar(x + width/2, rp_vals, width, label="Risk Parity (SLSQP)", color=GREEN, alpha=0.85)

    for i, (d, r) in enumerate(zip(de_vals, rp_vals)):
        ax.text(i - width/2, d + 0.01, f"{d:.2f}", ha="center", color=TEXT_COL, fontsize=9)
        ax.text(i + width/2, r + 0.01, f"{r:.2f}", ha="center", color=TEXT_COL, fontsize=9)

    ax.annotate("DE overfits:\nIS-heavy TLT weights\ncollapse in 2022 rate shock",
                xy=(1 - width/2, 0.15), xytext=(0.7, 0.55),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.5),
                fontsize=8, color=ORANGE, ha="center")

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Calmar Ratio")
    ax.set_title("Optimiser Comparison: DE overfits, RP generalises", fontweight="bold")
    ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL)
    ax.set_ylim(0, 0.85)
    save_fig(fig, out_dir / "findings_plot.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Rolling RP
# ─────────────────────────────────────────────────────────────────────────────

def plot_rolling_rp(out_dir):
    weight_csv = PROJECT_ROOT / "results" / "2026-03-27_00-12-12_full_backtest_6assets_M_2006_2020_rolling_rp_is" / "weight_history_is.csv"
    if not weight_csv.exists():
        print(f"  skipping rolling_rp: {weight_csv} not found")
        return

    df = pd.read_csv(weight_csv, parse_dates=["date"])
    tickers = [c for c in df.columns if c != "date"]

    fig, ax = plt.subplots(figsize=(10, 4), facecolor=DARK_BG)
    style_ax(ax)

    colors = [ACCENT, GREEN, RED, ORANGE, PURPLE, CYAN]
    for i, ticker in enumerate(tickers):
        ax.plot(df["date"], df[ticker], label=ticker, color=colors[i % len(colors)],
                linewidth=1.2, alpha=0.9)

    ax.set_ylabel("RP Weight")
    ax.set_title("Rolling RP Weights Converge to Static", fontweight="bold")
    ax.legend(ncol=3, fontsize=7, facecolor=PANEL_BG, edgecolor=GRID_COL,
              labelcolor=TEXT_COL, loc="upper right")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    save_fig(fig, out_dir / "findings_plot.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Rebalance Frequency (pre-tax)
# ─────────────────────────────────────────────────────────────────────────────

def plot_rebalance_frequency(out_dir):
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=DARK_BG)
    style_ax(ax)

    windows = ["2018 OOS", "2020 OOS", "2022 OOS"]
    monthly = [0.452, 0.457, 0.376]
    weekly = [0.446, 0.449, 0.369]

    x = np.arange(len(windows))
    width = 0.3

    ax.bar(x - width/2, monthly, width, label="Monthly", color=GREEN, alpha=0.85)
    ax.bar(x + width/2, weekly, width, label="Weekly", color=ORANGE, alpha=0.85)

    for i in range(len(windows)):
        ax.text(i - width/2, monthly[i] + 0.005, f"{monthly[i]:.3f}",
                ha="center", color=TEXT_COL, fontsize=8)
        ax.text(i + width/2, weekly[i] + 0.005, f"{weekly[i]:.3f}",
                ha="center", color=TEXT_COL, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(windows)
    ax.set_ylabel("Calmar Ratio")
    ax.set_title("Rebalance Frequency: No pre-tax improvement\n(reopened under tax modelling — see tax_drift_trigger)",
                 fontweight="bold", fontsize=10)
    ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL)
    ax.set_ylim(0, 0.55)
    save_fig(fig, out_dir / "findings_plot.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Momentum Overlay
# ─────────────────────────────────────────────────────────────────────────────

def plot_momentum_overlay(out_dir):
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=DARK_BG)
    style_ax(ax)

    labels = ["Baseline\n(no overlay)", "Best IS combo\n(d=20, t=0.10)", "Same combo\nOOS (hardest)"]
    calmars = [0.48, 0.72, 0.43]
    colors_bar = [GREEN, ACCENT, RED]

    bars = ax.bar(labels, calmars, color=colors_bar, alpha=0.85, width=0.5)
    for bar, val in zip(bars, calmars):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", color=TEXT_COL, fontsize=10)

    ax.axhline(y=0.48, color=GREEN, linestyle="--", alpha=0.4, linewidth=1)
    ax.annotate("IS winner hurts OOS",
                xy=(2, 0.43), xytext=(2, 0.58),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5),
                fontsize=9, color=RED, ha="center")

    ax.set_ylabel("Calmar Ratio")
    ax.set_title("SPY Momentum Overlay: Re-entry timing not learnable", fontweight="bold")
    ax.set_ylim(0, 0.85)
    save_fig(fig, out_dir / "findings_plot.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Bond Leverage
# ─────────────────────────────────────────────────────────────────────────────

def plot_bond_leverage(out_dir):
    csv_path = PROJECT_ROOT / "results" / "leverage_experiment.csv"
    if not csv_path.exists():
        print(f"  skipping bond_leverage: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), facecolor=DARK_BG)
    for ax in (ax1, ax2):
        style_ax(ax)

    splits = df["split"].unique()
    colors_split = [ACCENT, GREEN, ORANGE]

    for i, split in enumerate(splits):
        sub = df[df["split"] == split]
        ax1.plot(sub["leverage"], sub["calmar"], marker="o", label=split,
                 color=colors_split[i % len(colors_split)], linewidth=1.5, markersize=4)
        ax2.plot(sub["leverage"], sub["max_dd"].abs(), marker="o", label=split,
                 color=colors_split[i % len(colors_split)], linewidth=1.5, markersize=4)

    ax1.set_xlabel("Bond Leverage Factor")
    ax1.set_ylabel("Calmar Ratio")
    ax1.set_title("Calmar collapses with leverage", fontweight="bold")
    ax1.legend(fontsize=7, facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL)

    ax2.set_xlabel("Bond Leverage Factor")
    ax2.set_ylabel("Max Drawdown (%)")
    ax2.set_title("Drawdowns deepen with leverage", fontweight="bold")
    ax2.legend(fontsize=7, facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(100))

    fig.suptitle("Bond Leverage: Destroys risk-adjusted returns in rising-rate regime",
                 color="white", fontweight="bold", fontsize=11, y=1.02)
    save_fig(fig, out_dir / "findings_plot.png")


# ─────────────────────────────────────────────────────────────────────────────
# 7. ALLW Benchmark
# ─────────────────────────────────────────────────────────────────────────────

def plot_allw_benchmark(out_dir):
    from engine.data import fetch_prices

    # Load overlap-indexed data from latest strategy_comparison bundle
    csv_dir = PROJECT_ROOT / "results" / "strategy_comparison"
    latest = sorted(csv_dir.iterdir())[-1] if csv_dir.exists() else None
    daily_csv = latest / "daily_series.csv" if latest else None

    if not daily_csv or not daily_csv.exists():
        print(f"  skipping allw_benchmark: daily_series.csv not found")
        return

    ds = pd.read_csv(daily_csv)
    ds["Date"] = pd.to_datetime(ds["Date"])

    allw_start = pd.Timestamp("2025-03-06")
    overlap = ds[ds["Date"] >= allw_start].copy()

    # Fetch JEPQ separately (added in E.20, may not be in older bundles)
    try:
        jepq_prices = fetch_prices(["JEPQ"], "2025-03-06", overlap["Date"].max().strftime("%Y-%m-%d"))
        jepq_prices = jepq_prices.dropna()
        first_jepq = jepq_prices.iloc[0]["JEPQ"]
        jepq_series = (jepq_prices["JEPQ"] / first_jepq * 100).reset_index()
        jepq_series.columns = ["Date", "Overlap Indexed Value"]
        jepq_series["Strategy"] = "JEPQ"
        has_jepq = True
    except Exception:
        has_jepq = False

    fig, ax = plt.subplots(figsize=(12, 6), facecolor=DARK_BG)
    style_ax(ax)

    strategy_colors = {
        "My Strategy (DIY)": GREEN,
        "S&P 500 (SPY)": ORANGE,
        "ALLW (Bridgewater)": ACCENT,
        "60/40 (SPY/TLT)": PURPLE,
        "JEPQ": "#c678dd",
    }
    strategy_order = ["My Strategy (DIY)", "ALLW (Bridgewater)", "JEPQ",
                      "S&P 500 (SPY)", "60/40 (SPY/TLT)"]

    for strat in strategy_order:
        if strat == "JEPQ":
            if not has_jepq:
                continue
            sub = jepq_series
        else:
            sub = overlap[overlap["Strategy"] == strat]
            if sub.empty:
                continue
        color = strategy_colors.get(strat, TEXT_COL)
        short = strat.replace("My Strategy (DIY)", "DIY RP").replace("S&P 500 (SPY)", "SPY")
        short = short.replace("ALLW (Bridgewater)", "ALLW").replace("60/40 (SPY/TLT)", "60/40")
        ax.plot(sub["Date"], sub["Overlap Indexed Value"], label=short,
                color=color, linewidth=1.5, alpha=0.9)

    ax.axhline(y=100, color=GRID_COL, linestyle="--", alpha=0.4, linewidth=0.8)
    ax.set_ylabel("Indexed Value (100 = ALLW launch)")
    ax.set_xlabel("")
    ax.set_title("Performance Since ALLW Launch (2025-03-06)", fontweight="bold", fontsize=13)
    ax.legend(fontsize=9, facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL,
              loc="upper left")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))

    save_fig(fig, out_dir / "findings_plot.png")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Data Source Validation
# ─────────────────────────────────────────────────────────────────────────────

def plot_data_source_validation(out_dir):
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=DARK_BG)
    style_ax(ax)

    windows = ["2018 OOS", "2020 OOS", "2022 OOS"]
    yf_total = [0.487, 0.503, 0.452]
    fmp_adj = [0.488, 0.504, 0.453]
    yf_price = [0.385, 0.391, 0.341]

    x = np.arange(len(windows))
    width = 0.22

    ax.bar(x - width, yf_total, width, label="yfinance total-return", color=GREEN, alpha=0.85)
    ax.bar(x, fmp_adj, width, label="FMP adj_close", color=ACCENT, alpha=0.85)
    ax.bar(x + width, yf_price, width, label="yfinance price-return", color=RED, alpha=0.85)

    for i in range(len(windows)):
        ax.text(i - width, yf_total[i] + 0.005, f"{yf_total[i]:.3f}",
                ha="center", color=TEXT_COL, fontsize=7)
        ax.text(i, fmp_adj[i] + 0.005, f"{fmp_adj[i]:.3f}",
                ha="center", color=TEXT_COL, fontsize=7)
        ax.text(i + width, yf_price[i] + 0.005, f"{yf_price[i]:.3f}",
                ha="center", color=TEXT_COL, fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(windows)
    ax.set_ylabel("Calmar Ratio")
    ax.set_title("Data Source Validation: total-return = FMP adj_close; price-return understates",
                 fontweight="bold", fontsize=10)
    ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL)
    ax.set_ylim(0, 0.6)
    save_fig(fig, out_dir / "findings_plot.png")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Tax & Drift Trigger
# ─────────────────────────────────────────────────────────────────────────────

def plot_tax_drift_trigger(out_dir):
    sweep_dir = PROJECT_ROOT / "results" / "tax_threshold_sweep"
    latest = sorted(sweep_dir.iterdir())[-1] if sweep_dir.exists() else None
    csv_path = latest / "threshold_sweep_summary.csv" if latest else None

    if not csv_path or not csv_path.exists():
        print(f"  skipping tax_drift_trigger: sweep CSV not found")
        return

    df = pd.read_csv(csv_path)
    fifo_us = df[(df["selector"] == "fifo") & (df["regime"] == "us")]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor=DARK_BG)
    for ax in (ax1, ax2):
        style_ax(ax)

    # Left: Calmar by policy for FIFO/US
    policies = fifo_us["policy"].unique()
    windows = sorted(fifo_us["window"].unique())
    window_colors = {2018: ACCENT, 2020: GREEN, 2022: ORANGE}

    policy_calmars = {}
    for pol in policies:
        sub = fifo_us[fifo_us["policy"] == pol]
        policy_calmars[pol] = {w: sub[sub["window"] == w]["calmar"].values[0] for w in windows}

    sorted_policies = sorted(policies, key=lambda p: np.mean(list(policy_calmars[p].values())),
                             reverse=True)

    x = np.arange(len(sorted_policies))
    w = 0.25
    for i, win in enumerate(windows):
        vals = [policy_calmars[p][win] for p in sorted_policies]
        ax1.bar(x + i * w, vals, w, label=f"{win} OOS", color=window_colors[win], alpha=0.85)

    monthly_calmar = np.mean(list(policy_calmars["monthly_unconditional"].values()))
    ax1.axhline(y=monthly_calmar, color=RED, linestyle="--", alpha=0.7, linewidth=1,
                label="Monthly avg")

    short_labels = [p.replace("drift_", "d_").replace("_unconditional", "")
                    .replace("relative_", "r").replace("absolute_", "a")
                    for p in sorted_policies]
    ax1.set_xticks(x + w)
    ax1.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=7)
    ax1.set_ylabel("Calmar Ratio")
    ax1.set_title("Rebalance Policy Sweep (FIFO / US Tax):\nEvery drift threshold beats monthly", fontweight="bold", fontsize=9)
    ax1.legend(fontsize=7, facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL)

    # Right: Tax cost + rebalance count comparison
    monthly_row = fifo_us[fifo_us["policy"] == "monthly_unconditional"].iloc[0]
    best_drift = fifo_us[fifo_us["policy"] == "drift_absolute_5pp"].iloc[0] if "drift_absolute_5pp" in fifo_us["policy"].values else fifo_us[fifo_us["policy"] != "monthly_unconditional"].sort_values("calmar", ascending=False).iloc[0]

    comparison_data = {
        "Monthly": [monthly_row["cumulative_tax"], monthly_row["rebalances"]],
        "Best drift\n(abs 5pp)": [best_drift["cumulative_tax"], best_drift["rebalances"]],
    }

    labels = list(comparison_data.keys())
    tax_vals = [v[0] for v in comparison_data.values()]
    rebal_vals = [v[1] for v in comparison_data.values()]

    x2 = np.arange(len(labels))
    bars_tax = ax2.bar(x2 - 0.2, tax_vals, 0.35, label="Cumulative Tax ($)", color=RED, alpha=0.8)
    ax2_twin = ax2.twinx()
    bars_rebal = ax2_twin.bar(x2 + 0.2, rebal_vals, 0.35, label="# Rebalances", color=ACCENT, alpha=0.8)
    ax2_twin.tick_params(colors=TEXT_COL, labelsize=8)
    ax2_twin.spines["right"].set_color(GRID_COL)

    for bar, val in zip(bars_tax, tax_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                 f"${val:,.0f}", ha="center", color=TEXT_COL, fontsize=8)
    for bar, val in zip(bars_rebal, rebal_vals):
        ax2_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                      f"{int(val)}", ha="center", color=TEXT_COL, fontsize=8)

    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Cumulative Tax ($)", color=RED)
    ax2_twin.set_ylabel("Rebalances", color=ACCENT)
    ax2.set_title("Tax deferral: fewer trades, lower tax bill", fontweight="bold")

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=7,
               facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL)

    fig.suptitle("Rebalance Policy: Which threshold? Drift beats monthly under US tax (best: abs 5pp)",
                 color="white", fontweight="bold", fontsize=11, y=1.02)
    fig.tight_layout()
    save_fig(fig, out_dir / "findings_plot.png")


# ─────────────────────────────────────────────────────────────────────────────
# 10. RSI Leverage Overlay
# ─────────────────────────────────────────────────────────────────────────────

def plot_rsi_leverage_overlay(out_dir):
    csv_dir = PROJECT_ROOT / "results" / "leverage_comparison"
    latest = sorted(csv_dir.iterdir())[-1] if csv_dir.exists() else None
    grid_csv = latest / "threshold_grid.csv" if latest else None

    if not grid_csv or not grid_csv.exists():
        print(f"  skipping rsi_leverage_overlay: threshold_grid.csv not found")
        return

    grid = pd.read_csv(grid_csv)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=DARK_BG)

    for idx, ticker in enumerate(["SPY", "GLD"]):
        ax = axes[idx]
        style_ax(ax)

        sub = grid[(grid["Ticker"] == ticker) & (grid["Overlay Weight"] == 0.3)]
        if sub.empty:
            ax.set_title(f"{ticker}: no data at 30%", color="white")
            continue

        entries = sorted(sub["Entry Threshold"].unique())
        exits = sorted(sub["Exit Threshold"].unique())

        heatmap = np.full((len(entries), len(exits)), np.nan)
        for _, row in sub.iterrows():
            ei = entries.index(row["Entry Threshold"])
            xi = exits.index(row["Exit Threshold"])
            heatmap[ei, xi] = row["Calmar"]

        im = ax.imshow(heatmap, cmap="RdYlGn", aspect="auto",
                       vmin=np.nanmin(heatmap) * 0.95, vmax=np.nanmax(heatmap) * 1.02)

        ax.set_xticks(range(len(exits)))
        ax.set_xticklabels([f"{int(e)}" for e in exits], fontsize=6, rotation=45)
        ax.set_yticks(range(len(entries)))
        ax.set_yticklabels([f"{int(e)}" for e in entries], fontsize=6)
        ax.set_xlabel("RSI Exit Threshold", fontsize=9)
        ax.set_ylabel("RSI Entry Threshold", fontsize=9)
        ax.set_title(f"{ticker} @ 30% leverage — Calmar heatmap", fontweight="bold")

        # Annotate best cell
        best_idx = np.unravel_index(np.nanargmax(heatmap), heatmap.shape)
        best_val = heatmap[best_idx]
        ax.plot(best_idx[1], best_idx[0], 'w*', markersize=12)
        ax.annotate(f"{best_val:.3f}\n(E={int(entries[best_idx[0]])}/X={int(exits[best_idx[1]])})",
                    xy=(best_idx[1], best_idx[0]),
                    xytext=(best_idx[1] + 1.5, best_idx[0] - 1.5),
                    fontsize=7, color="white", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="white", lw=1))

        fig.colorbar(im, ax=ax, shrink=0.8, label="Calmar")

    fig.suptitle("RSI Leverage Overlay: Entry x Exit heatmap at 30% overlay weight",
                 color="white", fontweight="bold", fontsize=12, y=1.02)
    fig.tight_layout()
    save_fig(fig, out_dir / "findings_plot.png")


# ─────────────────────────────────────────────────────────────────────────────
# 11. Production Validation (strategy comparison summary)
# ─────────────────────────────────────────────────────────────────────────────

def plot_production_validation(out_dir):
    csv_dir = PROJECT_ROOT / "results" / "strategy_comparison"
    latest = sorted(csv_dir.iterdir())[-1] if csv_dir.exists() else None
    csv_path = latest / "summary_metrics.csv" if latest else None

    if not csv_path or not csv_path.exists():
        print(f"  skipping production_validation: summary CSV not found")
        return

    df = pd.read_csv(csv_path)

    # Prefer ALLW overlap window for apples-to-apples comparison
    allw_overlap = df[df["Window"] == "ALLW overlap"]
    if allw_overlap.empty:
        allw_overlap = df[df["Window"] == "Full History"]
        window_label = "Full History"
    else:
        window_label = "ALLW Overlap (since 2025-03-06)"

    full = allw_overlap.copy()

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=DARK_BG)
    style_ax(ax)

    strategies = full["Strategy"].values
    calmars = full["Calmar"].values
    mdds = full["Max Drawdown (%)"].abs().values
    cagrs = full["CAGR (%)"].values

    scatter = ax.scatter(mdds, cagrs, c=calmars, cmap="RdYlGn", s=120,
                         edgecolors="white", linewidths=0.5,
                         vmin=min(calmars) * 0.8, vmax=max(calmars) * 1.1)
    for i, name in enumerate(strategies):
        short = name.replace("My Strategy", "DIY").replace(" (DIY)", "")
        short = short.replace("S&P 500 (SPY)", "SPY").replace("60/40 (SPY/TLT)", "60/40")
        short = short.replace("ALLW (Bridgewater)", "ALLW").replace("JEPQ (JPM Nasdaq Income)", "JEPQ")
        short = short[:25]
        ax.annotate(short, (mdds[i], cagrs[i]), fontsize=7, color=TEXT_COL,
                    xytext=(8, 0), textcoords="offset points")

    cb = fig.colorbar(scatter, ax=ax, label="Calmar Ratio", shrink=0.8)
    cb.ax.yaxis.label.set_color(TEXT_COL)
    cb.ax.tick_params(colors=TEXT_COL)

    ax.set_xlabel("Max Drawdown (%)")
    ax.set_ylabel("CAGR (%)")
    ax.set_title(f"Strategy Comparison: Risk-Return Frontier ({window_label})", fontweight="bold")
    save_fig(fig, out_dir / "findings_plot.png")


# ─────────────────────────────────────────────────────────────────────────────
# Overview — combined verdict panel
# ─────────────────────────────────────────────────────────────────────────────

def plot_overview(out_dir):
    investigations = [
        ("universe_selection", "closed", "6-asset confirmed optimal"),
        ("optimiser_comparison", "closed", "DE fails OOS; RP dominates"),
        ("rolling_rp", "closed", "Converges to static weights"),
        ("rebalance_frequency", "reopened", "Failed pre-tax; drift wins post-tax"),
        ("momentum_overlay", "closed", "Re-entry timing unsolvable"),
        ("bond_leverage", "closed", "Calmar collapse in rising rates"),
        ("allw_benchmark", "production", "DIY beats ALLW on Calmar"),
        ("data_source_validation", "production", "yfinance = FMP adj_close"),
        ("tax_drift_trigger", "production (gated)", "Drift beats monthly under US tax"),
        ("rsi_leverage_overlay", "active-research", "SPY+GLD strongest; gate pending"),
        ("production_validation", "production", "Bundle builder for marimo"),
        ("shadow_comparison", "todo", "Live vs simulated (not started)"),
    ]

    fig, ax = plt.subplots(figsize=(12, 6), facecolor=DARK_BG)
    style_ax(ax)
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, len(investigations) - 0.5)
    ax.invert_yaxis()
    ax.set_axis_off()

    for i, (name, verdict, summary) in enumerate(investigations):
        color = VERDICT_COLORS.get(verdict, TEXT_COL)
        ax.barh(i, 9.5, left=0.25, height=0.7, color=color, alpha=0.15, edgecolor=color, linewidth=0.8)

        ax.text(0.4, i, name.replace("_", " "), fontsize=9, color="white",
                fontweight="bold", va="center", fontfamily="monospace")
        ax.text(5.5, i, summary, fontsize=8, color=TEXT_COL, va="center")

        badge_x = 4.2
        ax.text(badge_x, i, verdict.upper(), fontsize=7, color=color,
                fontweight="bold", va="center", ha="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=DARK_BG, edgecolor=color, linewidth=0.8))

    ax.set_title("All Weather Research Investigations — Verdict Overview",
                 color="white", fontweight="bold", fontsize=13, pad=15)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, alpha=0.4, edgecolor=c, label=v.upper())
                       for v, c in VERDICT_COLORS.items() if v != "todo"]
    ax.legend(handles=legend_elements, loc="lower center", ncol=5, fontsize=7,
              facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL,
              bbox_to_anchor=(0.5, -0.08))

    save_fig(fig, out_dir / "findings_overview.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    research_dir = PROJECT_ROOT / "research"

    print("Generating research findings visualisations...\n")

    plots = [
        ("universe_selection", plot_universe_selection),
        ("optimiser_comparison", plot_optimiser_comparison),
        ("rolling_rp", plot_rolling_rp),
        ("rebalance_frequency", plot_rebalance_frequency),
        ("momentum_overlay", plot_momentum_overlay),
        ("bond_leverage", plot_bond_leverage),
        ("allw_benchmark", plot_allw_benchmark),
        ("data_source_validation", plot_data_source_validation),
        ("tax_drift_trigger", plot_tax_drift_trigger),
        ("rsi_leverage_overlay", plot_rsi_leverage_overlay),
        ("production_validation", plot_production_validation),
    ]

    for name, plot_fn in plots:
        out_dir = research_dir / name
        print(f"[{name}]")
        try:
            plot_fn(out_dir)
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\n[overview]")
    plot_overview(research_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
