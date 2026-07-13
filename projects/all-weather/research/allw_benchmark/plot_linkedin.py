"""
plot_linkedin.py
================
Generate a two-panel LinkedIn figure comparing DIY Risk Parity vs ALLW.

Top panel:  Full equity curves (prototype, ALLW, SPY) with metrics table inset.
Bottom panel: Zoomed into the current max-drawdown window.

Usage:
    conda run -n allweather python -m research.plot_linkedin

Output:
    research/results/linkedin_comparison.png
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from datetime import date

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import yfinance as yf

from engine.calendar import pandas_resample_frequency

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*auto_adjust.*")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
_RESULTS_DIR = os.path.join(_SCRIPT_DIR, "results")
os.makedirs(_RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

DATE_START = "2025-03-06"  # ALLW launch
DATE_END = date.today().strftime("%Y-%m-%d")

# Load live-ticker DIY allocation from strategies.json
def _load_live_allocation() -> dict[str, float]:
    path = os.path.join(_PROJECT_ROOT, "strategies.json")
    with open(path, "r") as f:
        data = json.load(f)
    strategy = data["strategies"]["6asset_tip_gsg_rpavg"]
    allocation = strategy["allocation"]
    live_tickers = strategy.get("live_tickers", {})
    live_allocation: dict[str, float] = {}
    for backtest_ticker, weight in allocation.items():
        live_ticker = live_tickers.get(backtest_ticker, backtest_ticker)
        live_allocation[live_ticker] = live_allocation.get(live_ticker, 0.0) + weight
    return live_allocation

ALLOCATION = _load_live_allocation()
DIY_FEE = 0.0012
ALLW_FEE = 0.0085

# Colours
DARK_BG = "#0d1117"
PANEL_BG = "#161b22"
GRID_COL = "#30363d"
TEXT_COL = "#c9d1d9"
BORDER_COL = "#30363d"

COL_DIY  = "#58a6ff"  # blue
COL_ALLW = "#f0b429"  # amber
COL_SPY  = "#f78166"  # coral

PLOT_SPECS = {
    "diy":  ("DIY unlevered ETF prototype", COL_DIY, 2.8, "-"),
    "allw": ("ALLW", COL_ALLW, 2.3, "-"),
    "spy":  ("S&P 500 B&H", COL_SPY, 1.6, ":"),
}

MONTH_END = pandas_resample_frequency("ME")

TRADING_DAYS_PER_YEAR = 252

# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------

def _strip_tz(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if idx.tz is not None:
        return idx.tz_localize(None)
    return idx


def fetch_prices() -> pd.DataFrame:
    tickers = sorted(set(list(ALLOCATION.keys()) + ["ALLW", "SPY", "TLT"]))
    print(f"Fetching: {' '.join(tickers)}")
    frames = {}
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            hist = t.history(start=DATE_START, end=DATE_END, auto_adjust=True)
            if hist.empty:
                print(f"  WARNING: no data for {ticker}")
                continue
            hist.index = _strip_tz(hist.index)
            frames[ticker] = hist["Close"].rename(ticker)
        except Exception as e:
            print(f"  ERROR fetching {ticker}: {e}")

    if "ALLW" not in frames:
        print("FATAL: cannot fetch ALLW data")
        sys.exit(1)

    prices = pd.DataFrame(frames).dropna(how="all").ffill()
    print(f"  {len(prices)} trading days: {prices.index[0].date()} → {prices.index[-1].date()}")
    return prices


# ---------------------------------------------------------------------------
# PORTFOLIO CONSTRUCTION
# ---------------------------------------------------------------------------

def build_monthly_rebalanced(prices: pd.DataFrame,
                              allocation: dict[str, float],
                              start_val: float = 10_000.0) -> pd.Series:
    """Daily portfolio series with monthly rebalancing."""
    tickers = [t for t in allocation if t in prices.columns]
    alloc = {t: allocation[t] for t in tickers}
    total_w = sum(alloc.values())
    alloc = {t: w / total_w for t, w in alloc.items()}

    value = start_val
    values = [value]
    shares = {t: value * alloc[t] / float(prices[t].iloc[0]) for t in tickers}
    last_month = prices.index[0].month

    for i in range(1, len(prices)):
        current_month = prices.index[i].month
        # Rebalance on month change
        if current_month != last_month:
            port_val = sum(shares[t] * float(prices[t].iloc[i - 1]) for t in tickers)
            shares = {t: port_val * alloc[t] / float(prices[t].iloc[i - 1]) for t in tickers}
            last_month = current_month

        value = sum(shares[t] * float(prices[t].iloc[i]) for t in tickers)
        values.append(value)

    return pd.Series(values, index=prices.index)


def apply_fee(series: pd.Series, annual_fee: float) -> pd.Series:
    """Apply daily fee drag."""
    daily_drag = (1 - annual_fee) ** (1 / TRADING_DAYS_PER_YEAR)
    n = len(series)
    drag = pd.Series([daily_drag ** i for i in range(n)], index=series.index)
    return series * drag


# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------

def compute_metrics(series: pd.Series) -> dict:
    """Compute key performance metrics from a daily value series."""
    total_days = (series.index[-1] - series.index[0]).days
    years = total_days / 365.25
    total_ret = series.iloc[-1] / series.iloc[0] - 1.0
    cagr = (series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1.0

    running_max = series.cummax()
    drawdowns = (series - running_max) / running_max
    max_dd = drawdowns.min()

    calmar = cagr / abs(max_dd) if max_dd != 0 else float("inf")

    monthly = series.resample(MONTH_END).last().pct_change().dropna()
    vol = monthly.std() * np.sqrt(12)

    return {
        "total_ret": total_ret,
        "cagr": cagr,
        "max_dd": max_dd,
        "calmar": calmar,
        "vol": vol
    }


# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------

def _style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COL, labelsize=9)
    for sp in ax.spines.values():
        sp.set_color(BORDER_COL)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.label.set_color(TEXT_COL)
    ax.xaxis.label.set_color(TEXT_COL)
    ax.title.set_color("white")
    ax.grid(axis="y", color=GRID_COL, alpha=0.45, linewidth=0.7)
    ax.grid(axis="x", color=GRID_COL, alpha=0.25, linewidth=0.4)


def find_drawdown_window(series: pd.Series, pad_days: int = 14) -> tuple:
    """Find the peak-to-trough window of the max drawdown, with padding."""
    running_max = series.cummax()
    drawdowns = (series - running_max) / running_max
    trough_idx = drawdowns.idxmin()

    # Find the peak before the trough
    peak_idx = series.loc[:trough_idx].idxmax()

    # Find recovery (or end of series)
    post_trough = series.loc[trough_idx:]
    recovered = post_trough[post_trough >= series[peak_idx]]
    if len(recovered) > 0:
        recovery_idx = recovered.index[0]
    else:
        recovery_idx = series.index[-1]

    # Add padding
    pad = pd.Timedelta(days=pad_days)
    zoom_start = max(series.index[0], peak_idx - pad)
    zoom_end = min(series.index[-1], recovery_idx + pad)

    return zoom_start, zoom_end, peak_idx, trough_idx


def plot_linkedin(all_series: dict[str, pd.Series],
                  metrics: dict[str, dict]) -> None:
    """Create the two-panel LinkedIn figure."""

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(12, 10),
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.28},
    )
    fig.patch.set_facecolor(DARK_BG)
    _style_ax(ax_top)
    _style_ax(ax_bot)

    # ── Top panel: full equity curves ──────────────────────────────────────
    plot_order = [
        (key, *PLOT_SPECS[key])
        for key in ["diy", "allw", "spy"]
        if key in all_series
    ]

    for key, label, color, lw, ls in plot_order:
        s = all_series[key]
        ax_top.plot(s.index, s.values, color=color, lw=lw, linestyle=ls,
                    label=label, alpha=0.92)

        final_val = s.iloc[-1]
        ax_top.annotate(
            f"${final_val:,.0f}",
            xy=(s.index[-1], final_val),
            xytext=(8, 0),
            textcoords="offset points",
            color=color, fontsize=9, fontweight="bold", va="center",
        )

    ax_top.set_title(
        "$10,000 invested at DIY portfolio vs ALLW launch vs S&P 500 B&H",
        fontsize=13, pad=12, color="white", fontweight="bold",
    )
    ax_top.set_ylabel("Portfolio Value ($)", fontsize=10)
    ax_top.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax_top.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax_top.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax_top.xaxis.get_majorticklabels(), rotation=25, ha="right")

    ax_top.legend(
        fontsize=8.5, facecolor="#21262d", edgecolor=BORDER_COL,
        labelcolor=TEXT_COL, loc="upper left", framealpha=0.92,
    )

    # ── Metrics table inset (lower right of top panel) ─────────────────────
    m = metrics
    table_keys = [key for key in ["diy", "allw", "spy"] if key in all_series]
    short_labels = {
        "diy": "DIY prototype",
        "allw": "ALLW",
        "spy": "SPY B&H",
    }
    header = [""] + [short_labels[key] for key in table_keys]
    table_data = [header]
    for label, metric, fmt in [
        ("Total return", "total_ret", ".1%"),
        ("CAGR", "cagr", ".1%"),
        ("Max DD", "max_dd", ".1%"),
        ("Calmar", "calmar", ".2f"),
        ("Vol", "vol", ".1%"),
    ]:
        table_data.append([label] + [format(m[key][metric], fmt) for key in table_keys])

    col_colors = {i + 1: PLOT_SPECS[key][1] for i, key in enumerate(table_keys)}

    table = ax_top.table(
        cellText=table_data,
        cellLoc="center",
        loc="lower right",
        bbox=[0.43, 0.03, 0.55, 0.35],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID_COL)
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor("#21262d")
            c = col_colors.get(col, "white")
            cell.set_text_props(color=c, fontweight="bold", fontsize=8)
        else:
            cell.set_facecolor(PANEL_BG)
            if col in col_colors:
                cell.set_text_props(color=col_colors[col], fontweight="bold", fontsize=8)
            else:
                cell.set_text_props(color=TEXT_COL, fontsize=8)
        if col == 0:
            cell.set_text_props(color="#8b949e", fontweight="bold", fontsize=8)
            cell.set_facecolor("#21262d" if row == 0 else "#1c2128")

    # ── Bottom panel: drawdown zoom ───────────────────────────────────────
    diy_s = all_series["diy"]
    allw_s = all_series["allw"]

    diy_start, diy_end, _, _ = find_drawdown_window(diy_s, pad_days=15)
    allw_start, allw_end, _, _ = find_drawdown_window(allw_s, pad_days=15)
    zoom_start = min(diy_start, allw_start)
    zoom_end = max(diy_end, allw_end)

    zoom_plot_order = [row for row in plot_order if row[0] in {"diy", "allw"}]
    for key, label, color, lw, ls in zoom_plot_order:
        s = all_series[key]
        zoomed = s.loc[zoom_start:zoom_end]
        if len(zoomed) == 0:
            continue
        drawdown = (zoomed / zoomed.cummax() - 1.0) * 100
        ax_bot.plot(drawdown.index, drawdown.values, color=color, lw=lw,
                    linestyle=ls, label=label, alpha=0.92)

    ax_bot.axhline(0, color=GRID_COL, lw=0.8, alpha=0.5)

    annot_offsets = {
        "diy": (-130, 30),
        "allw": (-130, 30),
    }
    for key, label_short, color in [
        ("diy", "Prototype", COL_DIY),
        ("allw", "ALLW", COL_ALLW),
    ]:
        s = all_series[key]
        zoomed = s.loc[zoom_start:zoom_end]
        if len(zoomed) == 0:
            continue
        drawdown = (zoomed / zoomed.cummax() - 1.0) * 100
        trough_val = drawdown.min()
        trough_date = drawdown.idxmin()

        ox, oy = annot_offsets[key]
        ax_bot.annotate(
            f"{label_short}: {trough_val:.1f}%",
            xy=(trough_date, trough_val),
            xytext=(ox, oy),
            textcoords="offset points",
            color=color, fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor=PANEL_BG, edgecolor="none", alpha=0.85),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
        )

    ax_bot.set_title(
        "Largest drawdown episode in the public overlap",
        fontsize=12, pad=10, color="white", fontweight="bold",
    )
    ax_bot.set_ylabel("Drawdown from prior high (%)", fontsize=10)
    ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%d %b '%y"))
    ax_bot.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax_bot.xaxis.get_majorticklabels(), rotation=25, ha="right")

    ax_bot.legend().remove()

    fig.text(
        0.5, 0.012,
        "Mar 2025-May 2026 | total-return data | fee-adjusted where applicable | research only, not investment advice",
        ha="center", va="bottom", fontsize=8, color="#8b949e",
    )

    # Save
    out_path = os.path.join(_RESULTS_DIR, "linkedin_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nSaved → {out_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    prices = fetch_prices()

    # Align everything to ALLW start
    allw_start = prices["ALLW"].first_valid_index()
    prices = prices.loc[allw_start:]

    # Build series (all starting at $10k)
    diy_gross = build_monthly_rebalanced(prices, ALLOCATION, 10_000)
    diy = apply_fee(diy_gross, DIY_FEE)

    allw_raw = prices["ALLW"] / prices["ALLW"].iloc[0] * 10_000
    allw = apply_fee(allw_raw, ALLW_FEE)

    spy = prices["SPY"] / prices["SPY"].iloc[0] * 10_000

    all_series = {
        "diy": diy,
        "allw": allw,
        "spy": spy,
    }

    # Compute metrics
    metrics = {key: compute_metrics(s) for key, s in all_series.items()}

    # Print summary
    print(f"\n{'='*68}")
    print(f"{'':>20} {'CAGR':>8} {'MaxDD':>8} {'Calmar':>8}")
    print(f"{'-'*20} {'-'*8} {'-'*8} {'-'*8}")
    labels = {
        "diy": "Prototype",
        "allw": "ALLW", "spy": "SPY",
    }
    for key, m in metrics.items():
        print(f"{labels[key]:>20} {m['cagr']:>7.1%} {m['max_dd']:>7.1%} {m['calmar']:>8.2f}")
    print(f"{'='*68}\n")

    plot_linkedin(all_series, metrics)
    print("Done.")


if __name__ == "__main__":
    main()
