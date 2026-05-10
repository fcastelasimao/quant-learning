"""
compare_live_etfs.py
====================
Analyses whether candidate live-trading ETF substitutions
(SPY→IVV, QQQ→QQQM, GLD→GLDM, GSG→PDBC) deliver their claimed cost savings
without meaningfully altering the portfolio's risk profile.

Two sections
------------
1. Pairwise  — per-pair cumulative return, correlation, tracking error, and
               ER-implied drag over each pair's full common history.
2. Portfolio — side-by-side backtest of the production weights using the
               backtest ETF set vs the live ETF set over the overlap window
               starting October 2020 (constrained by QQQM's launch date).

Usage
-----
  conda run -n allweather python3 -m research.compare_live_etfs
"""

from __future__ import annotations

import os
import sys
import warnings
from datetime import date

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*auto_adjust.*")

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_RESULTS_DIR = os.path.join(_SCRIPT_DIR, "results")
os.makedirs(_RESULTS_DIR, exist_ok=True)

from engine.data import fetch_prices
from engine.backtest import run_backtest
from engine.stats import compute_stats

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

# QQQM launched 2020-10-13 — binding constraint for full portfolio comparison.
OVERLAP_START = "2020-10-01"
DATE_END      = date.today().strftime("%Y-%m-%d")

# Each pair's longest available common history.
PAIR_WINDOWS: dict[tuple[str, str], str] = {
    ("SPY",  "IVV"):  "2000-05-01",   # IVV launched May 2000
    ("QQQ",  "QQQM"): "2020-10-01",   # QQQM launched Oct 2020
    ("GLD",  "GLDM"): "2018-06-01",   # GLDM launched Jun 2018
    ("GSG",  "PDBC"): "2014-11-01",   # PDBC launched Nov 2014
}

# Expense ratios in basis points — sourced from ETF provider pages.
EXPENSE_RATIOS_BP: dict[str, float] = {
    "SPY": 9.45,  "IVV":  3.00,
    "QQQ": 20.00, "QQQM": 15.00,
    "GLD": 40.00, "GLDM": 10.00,
    "GSG": 75.00, "PDBC": 59.00,
}

# Production RP-averaged weights from strategies.json.
BACKTEST_ALLOCATION: dict[str, float] = {
    "SPY": 0.134, "QQQ": 0.103, "TLT": 0.175,
    "TIP": 0.348, "GLD": 0.142, "GSG": 0.098,
}
LIVE_ALLOCATION: dict[str, float] = {
    "IVV": 0.134, "QQQM": 0.103, "TLT": 0.175,
    "TIP": 0.348, "GLDM": 0.142, "PDBC": 0.098,
}

# GSG→PDBC is the only pair that is NOT a same-index substitution.
SAME_INDEX_PAIRS = {("SPY", "IVV"), ("QQQ", "QQQM"), ("GLD", "GLDM")}

# Dark-theme palette
DARK_BG  = "#0d1117"
PANEL_BG = "#161b22"
GRID_COL = "#30363d"
TEXT_COL = "#c9d1d9"
PAIR_COLOURS: dict[tuple[str, str], tuple[str, str]] = {
    ("SPY",  "IVV"):  ("#58a6ff", "#f78166"),
    ("QQQ",  "QQQM"): ("#3fb950", "#d2a8ff"),
    ("GLD",  "GLDM"): ("#e3b341", "#58a6ff"),
    ("GSG",  "PDBC"): ("#f78166", "#79c0ff"),
}
WATERMARK = "github.com/fcastelasimao/quant-learning"


# ---------------------------------------------------------------------------
# PAIRWISE ANALYSIS
# ---------------------------------------------------------------------------

def _fetch_pair(t1: str, t2: str, start: str) -> pd.DataFrame:
    df = fetch_prices([t1, t2], start, DATE_END)
    return df[[t1, t2]].dropna()


def _pairwise_stats(df: pd.DataFrame, t1: str, t2: str) -> dict:
    r1 = np.log(df[t1]).diff().dropna()
    r2 = np.log(df[t2]).diff().dropna()
    common = r1.index.intersection(r2.index)
    r1, r2 = r1.loc[common], r2.loc[common]

    corr    = r1.corr(r2)
    te_ann  = (r1 - r2).std() * np.sqrt(252)

    years  = (df.index[-1] - df.index[0]).days / 365.25
    ann1   = (df[t1].iloc[-1] / df[t1].iloc[0]) ** (1 / years) - 1
    ann2   = (df[t2].iloc[-1] / df[t2].iloc[0]) ** (1 / years) - 1
    return {
        "corr":            corr,
        "tracking_error":  te_ann * 100,
        "ann_return_1":    ann1   * 100,
        "ann_return_2":    ann2   * 100,
        "return_spread":   (ann2 - ann1) * 100,
        "years":           years,
    }


# ---------------------------------------------------------------------------
# PAIRWISE CHART
# ---------------------------------------------------------------------------

def _plot_pairwise(pairs_data: dict) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("Live ETF Substitutions — Pairwise Cumulative Return",
                 color=TEXT_COL, fontsize=13, y=0.98)

    for idx, ax in enumerate(axes.flat):
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT_COL)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COL)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
        ax.grid(True, color=GRID_COL, linewidth=0.5)

        pair = list(PAIR_WINDOWS.keys())[idx]
        t1, t2 = pair
        df, stats = pairs_data[pair]

        cum1 = (df[t1] / df[t1].iloc[0] - 1) * 100
        cum2 = (df[t2] / df[t2].iloc[0] - 1) * 100
        c1, c2 = PAIR_COLOURS[pair]
        ax.plot(cum1.index, cum1.values, color=c1, linewidth=1.8, label=t1)
        ax.plot(cum2.index, cum2.values, color=c2, linewidth=1.8,
                linestyle="--", label=t2)

        er_save = EXPENSE_RATIOS_BP[t1] - EXPENSE_RATIOS_BP[t2]
        valid   = "same index" if pair in SAME_INDEX_PAIRS else "DIFFERENT index"
        info    = (f"Corr: {stats['corr']:.4f}   TE: {stats['tracking_error']:.2f}%/yr\n"
                   f"ER saving: {er_save:.2f}bp   {valid}")
        ax.set_title(f"{t1} → {t2}   ({stats['years']:.1f} yrs)",
                     color=TEXT_COL, fontsize=10)
        ax.text(0.02, 0.02, info, transform=ax.transAxes, color=TEXT_COL,
                fontsize=7.5, verticalalignment="bottom",
                bbox=dict(facecolor=DARK_BG, edgecolor=GRID_COL, alpha=0.7))
        ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COL,
                  labelcolor=TEXT_COL, fontsize=8)

    fig.text(0.5, 0.01, WATERMARK, ha="center", color=GRID_COL, fontsize=7)
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    out = os.path.join(_RESULTS_DIR, "live_etf_pairwise.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# PORTFOLIO COMPARISON CHART
# ---------------------------------------------------------------------------

def _dd_series(s: pd.Series) -> pd.Series:
    peak = s.cummax()
    return ((s - peak) / peak) * 100


def _plot_portfolio(bt_bt: pd.DataFrame, bt_live: pd.DataFrame) -> str:
    fig, (ax_eq, ax_dd) = plt.subplots(
        2, 1, figsize=(13, 8), gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor(DARK_BG)

    for ax in (ax_eq, ax_dd):
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT_COL)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COL)
        ax.grid(True, color=GRID_COL, linewidth=0.5)

    col         = "All Weather Value"
    s_bt        = bt_bt[col]
    s_live      = bt_live[col]

    ax_eq.plot(s_bt.index,   s_bt.values,   color="#58a6ff", linewidth=2.2,
               label="Backtest ETFs  (SPY / QQQ / GLD / GSG)")
    ax_eq.plot(s_live.index, s_live.values, color="#3fb950", linewidth=2.2,
               linestyle="--", label="Live ETFs  (IVV / QQQM / GLDM / PDBC)")
    ax_eq.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda y, _: f"${y:,.0f}"))
    ax_eq.set_title(
        f"Portfolio: Backtest ETFs vs Live ETFs  |  "
        f"{OVERLAP_START[:7]} – {DATE_END[:7]}",
        color=TEXT_COL, fontsize=11)
    ax_eq.legend(facecolor=PANEL_BG, edgecolor=GRID_COL,
                 labelcolor=TEXT_COL, fontsize=9)

    ax_dd.fill_between(s_bt.index,   _dd_series(s_bt),   0,
                       color="#58a6ff", alpha=0.4, label="Backtest ETFs")
    ax_dd.fill_between(s_live.index, _dd_series(s_live), 0,
                       color="#3fb950", alpha=0.4, label="Live ETFs")
    ax_dd.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax_dd.set_ylabel("Drawdown", color=TEXT_COL, fontsize=9)

    fig.text(0.5, 0.01, WATERMARK, ha="center", color=GRID_COL, fontsize=7)
    plt.tight_layout(rect=[0, 0.02, 1, 1])
    out = os.path.join(_RESULTS_DIR, "live_etf_portfolio.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# STDOUT TABLES
# ---------------------------------------------------------------------------

def _print_pairwise_verdict(pairs_data: dict) -> None:
    W = 74
    print("\n" + "=" * W)
    print(f"{'PAIRWISE ANALYSIS':^{W}}")
    print("=" * W)
    print(f"{'Pair':<13} {'History':>8} {'Corr':>7} {'TE/yr':>7} "
          f"{'ER save':>8} {'Ret Δ':>7} {'Same index?':>12}")
    print("-" * W)

    for pair, (df, stats) in pairs_data.items():
        t1, t2    = pair
        er_save   = EXPENSE_RATIOS_BP[t1] - EXPENSE_RATIOS_BP[t2]
        same      = "yes" if pair in SAME_INDEX_PAIRS else "NO — different"
        print(f"{t1}→{t2:<8} "
              f"{stats['years']:>7.1f}y "
              f"{stats['corr']:>7.4f} "
              f"{stats['tracking_error']:>6.2f}% "
              f"{er_save:>6.2f}bp "
              f"{stats['return_spread']:>+6.2f}% "
              f"{same:>14}")

    print()
    print("  Ret Δ = annualised return of live ETF minus backtest ETF over pair history.")
    print("  TE    = annualised tracking error of daily log-return differences.")


def _print_portfolio_verdict(m_bt: dict, m_live: dict) -> None:
    # Portfolio-weighted ER saving (only substituted tickers)
    pairs_in_portfolio = [(t1, t2) for (t1, t2) in PAIR_WINDOWS
                          if t1 in BACKTEST_ALLOCATION]
    total_bp = sum(
        (EXPENSE_RATIOS_BP[t1] - EXPENSE_RATIOS_BP[t2]) * BACKTEST_ALLOCATION[t1]
        for t1, t2 in pairs_in_portfolio
    )

    W = 74
    print("\n" + "=" * W)
    print(f"{'PORTFOLIO COMPARISON  (Oct 2020 → today)':^{W}}")
    print("=" * W)
    print(f"{'Metric':<22} {'Backtest ETFs':>15} {'Live ETFs':>15} {'Diff':>10}")
    print("-" * W)
    rows = [
        ("CAGR (%)",     "cagr",   ".2f"),
        ("Max Drawdown", "mdd",    ".2f"),
        ("Calmar",       "calmar", ".3f"),
        ("Sharpe",       "sharpe", ".3f"),
    ]
    for label, key, fmt in rows:
        v1   = m_bt[key]
        v2   = m_live[key]
        diff = v2 - v1
        sign = "+" if diff >= 0 else ""
        print(f"{label:<22} {v1:>15{fmt}} {v2:>15{fmt}} {sign}{diff:>{10}{fmt}}")

    print()
    print(f"  Portfolio-weighted ER saving: {total_bp:.2f} bp/yr")
    print(f"  (~${total_bp * 10:.0f}/yr on $100k,  ~${total_bp * 100:.0f}/yr on $1M)")
    print()
    print("VERDICT PER PAIR:")
    print("  SPY → IVV    same S&P 500 index.         6.45bp × 13.4% wt = 0.86bp/yr")
    print("  QQQ → QQQM   same Nasdaq-100.              5bp × 10.3% wt = 0.52bp/yr")
    print("  GLD → GLDM   same LBMA gold (physical).   30bp × 14.2% wt = 4.26bp/yr")
    print("  GSG → PDBC   DIFFERENT products:")
    print("               GSG = S&P GSCI (passive, ~60% energy)")
    print("               PDBC = active roll, 35% sector cap, less energy tilt")
    print("               16bp × 9.8% wt = 1.57bp/yr — thin justification for")
    print("               changing the stagflation-hedge exposure profile.")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    print("─" * 60)
    print("Live ETF Substitution Analysis")
    print("─" * 60)

    print("\n[1/3] Fetching pairwise data...")
    pairs_data: dict[tuple[str, str], tuple[pd.DataFrame, dict]] = {}
    for pair, start in PAIR_WINDOWS.items():
        t1, t2 = pair
        df     = _fetch_pair(t1, t2, start)
        stats  = _pairwise_stats(df, t1, t2)
        pairs_data[pair] = (df, stats)

    print("\n[2/3] Fetching portfolio data (Oct 2020 → today)...")
    all_tickers = list(dict.fromkeys(
        list(BACKTEST_ALLOCATION.keys()) + list(LIVE_ALLOCATION.keys())
    ))
    prices_all  = fetch_prices(all_tickers, OVERLAP_START, DATE_END)

    bench_prices = prices_all["SPY"]
    tlt_prices   = prices_all["TLT"]
    prices_bt    = prices_all[list(BACKTEST_ALLOCATION.keys())]
    prices_live  = prices_all[list(LIVE_ALLOCATION.keys())]

    bt_bt   = run_backtest(prices_bt,   bench_prices, BACKTEST_ALLOCATION,
                           tlt_prices=tlt_prices, transaction_cost_pct=0.001)
    bt_live = run_backtest(prices_live, bench_prices, LIVE_ALLOCATION,
                           tlt_prices=tlt_prices, transaction_cost_pct=0.001)

    stats_bt   = compute_stats(bt_bt)
    stats_live = compute_stats(bt_live)
    m_bt   = {"cagr": stats_bt[0].cagr,   "mdd": stats_bt[0].max_drawdown,
              "calmar": stats_bt[0].calmar, "sharpe": stats_bt[0].sharpe}
    m_live = {"cagr": stats_live[0].cagr,  "mdd": stats_live[0].max_drawdown,
              "calmar": stats_live[0].calmar, "sharpe": stats_live[0].sharpe}

    _print_pairwise_verdict(pairs_data)
    _print_portfolio_verdict(m_bt, m_live)

    print("\n[3/3] Generating charts...")
    p1 = _plot_pairwise(pairs_data)
    p2 = _plot_portfolio(bt_bt, bt_live)
    print(f"  {p1}")
    print(f"  {p2}")
    print("\nDone.")


if __name__ == "__main__":
    main()
