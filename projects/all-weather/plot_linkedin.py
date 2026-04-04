"""
plot_linkedin.py
================
Generate a two-panel LinkedIn figure comparing DIY Risk Parity vs ALLW.

Top panel:  Full equity curves (RP, ALLW, SPY, 60/40) with metrics table inset.
Bottom panel: Zoomed into the worst drawdown period to emphasise the DD difference.

Usage:
    conda run -n allweather python3 plot_linkedin.py

Output:
    results/linkedin_comparison.png
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

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*auto_adjust.*")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_RESULTS_DIR = os.path.join(_SCRIPT_DIR, "results")
os.makedirs(_RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

DATE_START = "2025-03-06"  # ALLW launch
DATE_END = date.today().strftime("%Y-%m-%d")

# Load DIY allocation from strategies.json
def _load_allocation() -> dict[str, float]:
    path = os.path.join(_SCRIPT_DIR, "strategies.json")
    with open(path, "r") as f:
        data = json.load(f)
    return dict(data["strategies"]["6asset_tip_gsg_rpavg"]["allocation"])

ALLOCATION = _load_allocation()
DIY_FEE = 0.0012
ALLW_FEE = 0.0085

# Colours
DARK_BG = "#0d1117"
PANEL_BG = "#161b22"
GRID_COL = "#30363d"
TEXT_COL = "#c9d1d9"
BORDER_COL = "#30363d"

COL_DIY = "#58a6ff"   # blue
COL_ALLW = "#f0b429"  # amber
COL_SPY = "#f78166"   # coral
COL_6040 = "#3fb950"  # green

WATERMARK = "github.com/fcastelasimao/quant-learning"

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

    daily_rets = prices[tickers].pct_change().fillna(0.0)

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

    monthly = series.resample("ME").last().pct_change().dropna()
    vol = monthly.std() * np.sqrt(12)

    return {
        "cagr": cagr,
        "max_dd": max_dd,
        "calmar": calmar,
        "vol": vol,
        "total_ret": total_ret,
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


def _add_watermark(ax: plt.Axes) -> None:
    ax.text(
        0.995, 0.012, WATERMARK,
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=7, color="#444c56", alpha=0.8,
        style="italic",
    )


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
        ("diy", "DIY Risk Parity", COL_DIY, 2.5, "-"),
        ("allw", "ALLW (Bridgewater)", COL_ALLW, 2.2, "-"),
        ("spy", "S&P 500", COL_SPY, 1.5, ":"),
        ("6040", "60/40 (SPY/TLT)", COL_6040, 1.5, "-."),
    ]

    for key, label, color, lw, ls in plot_order:
        s = all_series[key]
        ax_top.plot(s.index, s.values, color=color, lw=lw, linestyle=ls,
                    label=label, alpha=0.92)

        # Final value annotation
        final_val = s.iloc[-1]
        ax_top.annotate(
            f"${final_val:,.0f}",
            xy=(s.index[-1], final_val),
            xytext=(8, 0),
            textcoords="offset points",
            color=color, fontsize=9, fontweight="bold", va="center",
        )

    ax_top.set_title(
        "$10,000 invested at ALLW launch — where are you today?",
        fontsize=13, pad=12, color="white", fontweight="bold",
    )
    ax_top.set_ylabel("Portfolio Value ($)", fontsize=10)
    ax_top.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax_top.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax_top.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax_top.xaxis.get_majorticklabels(), rotation=25, ha="right")

    ax_top.legend(
        fontsize=9.5, facecolor="#21262d", edgecolor=BORDER_COL,
        labelcolor=TEXT_COL, loc="upper left", framealpha=0.92,
    )

    # ── Metrics table inset (lower right of top panel) ─────────────────────
    table_data = [
        ["", "DIY RP", "ALLW", "SPY", "60/40"],
        ["CAGR",
         f"{metrics['diy']['cagr']:.1%}",
         f"{metrics['allw']['cagr']:.1%}",
         f"{metrics['spy']['cagr']:.1%}",
         f"{metrics['6040']['cagr']:.1%}"],
        ["Max DD",
         f"{metrics['diy']['max_dd']:.1%}",
         f"{metrics['allw']['max_dd']:.1%}",
         f"{metrics['spy']['max_dd']:.1%}",
         f"{metrics['6040']['max_dd']:.1%}"],
        ["Calmar",
         f"{metrics['diy']['calmar']:.2f}",
         f"{metrics['allw']['calmar']:.2f}",
         f"{metrics['spy']['calmar']:.2f}",
         f"{metrics['6040']['calmar']:.2f}"],
        ["Cost/yr",
         "$120", "$850", "$9", "$12"],
    ]

    table = ax_top.table(
        cellText=table_data,
        cellLoc="center",
        loc="lower right",
        bbox=[0.58, 0.02, 0.41, 0.38],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID_COL)
        cell.set_linewidth(0.5)
        if row == 0:
            # Header row
            cell.set_facecolor("#21262d")
            cell.set_text_props(color="white", fontweight="bold", fontsize=8.5)
        else:
            cell.set_facecolor(PANEL_BG)
            cell.set_text_props(color=TEXT_COL, fontsize=8)
            # Highlight DIY column
            if col == 1:
                cell.set_text_props(color=COL_DIY, fontweight="bold", fontsize=8.5)
            elif col == 2:
                cell.set_text_props(color=COL_ALLW, fontweight="bold", fontsize=8.5)
        # First column (metric names)
        if col == 0:
            cell.set_text_props(color="#8b949e", fontweight="bold", fontsize=8)
            cell.set_facecolor("#21262d" if row == 0 else "#1c2128")

    # ── Bottom panel: drawdown zoom ───────────────────────────────────────
    # Find the worst drawdown window across DIY and ALLW
    diy_s = all_series["diy"]
    allw_s = all_series["allw"]

    # Use ALLW's drawdown window (likely deeper)
    zoom_start, zoom_end, _, _ = find_drawdown_window(allw_s, pad_days=15)

    # Also check DIY's drawdown window and take the wider range for start
    diy_start, diy_end, _, _ = find_drawdown_window(diy_s, pad_days=15)
    zoom_start = min(zoom_start, diy_start)
    zoom_end = pd.Timestamp("2025-05-31")  # hard cutoff for cleaner zoom

    for key, label, color, lw, ls in plot_order:
        s = all_series[key]
        zoomed = s.loc[zoom_start:zoom_end]
        if len(zoomed) == 0:
            continue
        # Normalise to 100 at zoom start for percentage comparison
        normed = zoomed / zoomed.iloc[0] * 100
        ax_bot.plot(normed.index, normed.values, color=color, lw=lw,
                    linestyle=ls, label=label, alpha=0.92)

    ax_bot.axhline(100, color=GRID_COL, lw=0.8, alpha=0.5)

    # Annotate max drawdowns in the zoom window
    # Place labels well clear of graph lines: DIY above, ALLW below
    annot_offsets = {
        "diy":  (-35,65),   # right and well above trough
        "allw": (-150, 35),  # right and well below trough
    }
    for key, label_short, color in [("diy", "DIY", COL_DIY), ("allw", "ALLW", COL_ALLW)]:
        s = all_series[key]
        zoomed = s.loc[zoom_start:zoom_end]
        if len(zoomed) == 0:
            continue
        normed = zoomed / zoomed.iloc[0] * 100
        trough_val = normed.min()
        trough_date = normed.idxmin()
        dd_pct = (trough_val - 100)

        ox, oy = annot_offsets[key]
        ax_bot.annotate(
            f"{label_short}: {dd_pct:+.1f}%",
            xy=(trough_date, trough_val),
            xytext=(ox, oy),
            textcoords="offset points",
            color=color, fontsize=10, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
        )

    ax_bot.set_title(
        "Zoomed: Worst Drawdown Period",
        fontsize=12, pad=10, color="white", fontweight="bold",
    )
    ax_bot.set_ylabel("Indexed Value (start = 100)", fontsize=10)
    ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%d %b '%y"))
    ax_bot.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax_bot.xaxis.get_majorticklabels(), rotation=25, ha="right")

    ax_bot.legend(
        fontsize=9, facecolor="#21262d", edgecolor=BORDER_COL,
        labelcolor=TEXT_COL, loc="lower left", framealpha=0.92,
    )

    #_add_watermark(ax_bot)

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

    # 60/40
    spy_sh = 0.60 / float(prices["SPY"].iloc[0])
    tlt_sh = 0.40 / float(prices["TLT"].iloc[0])
    s_6040 = (spy_sh * prices["SPY"] + tlt_sh * prices["TLT"]) * 10_000
    s_6040 = s_6040 / s_6040.iloc[0] * 10_000

    all_series = {
        "diy": diy,
        "allw": allw,
        "spy": spy,
        "6040": s_6040,
    }

    # Compute metrics
    metrics = {key: compute_metrics(s) for key, s in all_series.items()}

    # Print summary
    print(f"\n{'='*60}")
    print(f"{'':>12} {'CAGR':>8} {'MaxDD':>8} {'Calmar':>8}")
    print(f"{'-'*12} {'-'*8} {'-'*8} {'-'*8}")
    labels = {"diy": "DIY RP", "allw": "ALLW", "spy": "SPY", "6040": "60/40"}
    for key, m in metrics.items():
        print(f"{labels[key]:>12} {m['cagr']:>7.1%} {m['max_dd']:>7.1%} {m['calmar']:>8.2f}")
    print(f"{'='*60}\n")

    plot_linkedin(all_series, metrics)
    print("Done.")


if __name__ == "__main__":
    main()
