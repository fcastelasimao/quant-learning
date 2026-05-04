"""
run_threshold_sensitivity.py
============================
Sweep drift thresholds (0% → 20%) across both rebalancing modes on the
6-asset universe.  Runs on three windows separately:

  Full period  2006 → today
  IS           2006 → 2018  (in-sample, used for optimisation)
  OOS          2018 → today (out-of-sample, the honest test)

Metrics reported per threshold × mode:
  CAGR, Max DD, Calmar, Sharpe, rebalance months, avg cash %, turnover $k

Outputs a summary table to the terminal and saves a chart to results/.

Run with:
    conda run -n allweather python3 run_threshold_sensitivity.py

Optional:
    --cost FLOAT   transaction cost fraction (default: 0.001)
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from engine import config
from engine.data import fetch_prices
from run_rebalance_mode_comparison import _simulate, _stats_row

THRESHOLDS = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07,
              0.08, 0.09, 0.10, 0.12, 0.15, 0.20]

MODES = ["per_asset", "full_on_breach"]

WINDOWS = {
    "Full (2006–today)": (config.BACKTEST_START, config.BACKTEST_END),
    "IS   (2006–2018)":  (config.BACKTEST_START, config.OOS_START),
    "OOS  (2018–today)": (config.OOS_START,       config.BACKTEST_END),
}

DARK_BG   = "#0d1117"
PANEL_BG  = "#161b22"
GRID_COL  = "#30363d"
TEXT_COL  = "#c9d1d9"
BORDER_COL = "#30363d"

COLORS = {
    "per_asset":      {"Full (2006–today)": "#58a6ff", "IS   (2006–2018)": "#79c0ff", "OOS  (2018–today)": "#1f6feb"},
    "full_on_breach": {"Full (2006–today)": "#f0b429", "IS   (2006–2018)": "#ffd166", "OOS  (2018–today)": "#d29922"},
}
LINESTYLES = {
    "Full (2006–today)": "-",
    "IS   (2006–2018)":  "--",
    "OOS  (2018–today)": "-.",
}


def _build_monthly(prices: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    tickers = list(config.TARGET_ALLOCATION.keys())
    port = prices[tickers]
    port = port[(port.index >= start) & (port.index < end)]
    return port.resample(config.DATA_FREQUENCY).last().dropna()


def _run_sweep(
    monthly: pd.DataFrame,
    cost: float,
) -> dict[tuple[float, str], dict]:
    """Return {(threshold, mode): stats_dict} for all combinations."""
    results = {}
    allocation = dict(config.TARGET_ALLOCATION)
    for threshold in THRESHOLDS:
        for mode in MODES:
            # threshold=0 + per_asset is identical to threshold=0 + full_on_breach
            # (both rebalance everything every month), so skip the duplicate
            if threshold == 0.0 and mode == "full_on_breach":
                continue
            result = _simulate(
                monthly=monthly,
                allocation=allocation,
                start_value=config.INITIAL_PORTFOLIO_VALUE,
                drift_threshold=threshold,
                rebalance_mode=mode,
                transaction_cost_pct=cost,
            )
            results[(threshold, mode)] = _stats_row(result, rf=config.RISK_FREE_RATE)
    return results


def _print_table(sweep: dict, window_label: str) -> None:
    rows = []
    for (threshold, mode), stats in sweep.items():
        rows.append({
            "Threshold": f"{threshold:.0%}",
            "Mode":      mode,
            **{k: v for k, v in stats.items() if k != "Mode"},
        })
    df = pd.DataFrame(rows)

    col_w   = 14
    idx_w   = 8
    mode_w  = 17

    header = (f"{'Thresh':>{idx_w}}  {'Mode':<{mode_w}}"
              + "".join(f"{c:>{col_w}}" for c in ["CAGR %", "Max DD %", "Calmar",
                                                    "Sharpe", "Rebal mths",
                                                    "Turnover $k", "Avg cash %"]))
    bar = "─" * len(header)
    print(f"\n{'═' * len(header)}")
    print(f"  {window_label}")
    print(f"{'═' * len(header)}")
    print(header)
    print(bar)

    best_calmar = max(s["Calmar"] for s in sweep.values())

    for (threshold, mode), stats in sweep.items():
        marker = " ◄" if stats["Calmar"] == best_calmar else "  "
        line = (f"{threshold:>{idx_w}.0%}  {mode:<{mode_w}}"
                + f"{stats['CAGR %']:>{col_w}}"
                + f"{stats['Max DD %']:>{col_w}}"
                + f"{stats['Calmar']:>{col_w}}"
                + f"{stats['Sharpe']:>{col_w}}"
                + f"{stats['Rebal months']:>{col_w}}"
                + f"{stats['Turnover $k']:>{col_w}}"
                + f"{stats['Avg cash %']:>{col_w}}"
                + marker)
        print(line)
    print(bar)


def _plot_sensitivity(
    sweeps: dict[str, dict],
    cost: float,
    out_path: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle(
        f"Threshold sensitivity — 6-asset risk parity  (tx cost: {cost:.2%})",
        color="white", fontsize=13, fontweight="bold", y=0.98,
    )

    metric_panels = [
        (axes[0, 0], "Calmar",      "Calmar ratio  (higher = better)"),
        (axes[0, 1], "Sharpe",      "Sharpe ratio  (higher = better)"),
        (axes[1, 0], "CAGR %",      "CAGR  (%)"),
        (axes[1, 1], "Max DD %",    "Max drawdown  (%)  (less negative = better)"),
    ]

    for ax, metric_key, ylabel in metric_panels:
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT_COL, labelsize=9)
        for sp in ax.spines.values():
            sp.set_color(BORDER_COL)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Drift threshold", color=TEXT_COL, fontsize=9)
        ax.set_ylabel(ylabel, color=TEXT_COL, fontsize=9)
        ax.grid(axis="y", color=GRID_COL, alpha=0.45, lw=0.7)
        ax.grid(axis="x", color=GRID_COL, alpha=0.25, lw=0.4)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))

        for window_label, sweep in sweeps.items():
            for mode in MODES:
                xs, ys = [], []
                for threshold in THRESHOLDS:
                    key = (threshold, mode)
                    if threshold == 0.0 and mode == "full_on_breach":
                        key = (0.0, "per_asset")
                    if key not in sweep:
                        continue
                    xs.append(threshold)
                    ys.append(sweep[key][metric_key])

                color = COLORS[mode][window_label]
                ls    = LINESTYLES[window_label]
                lw    = 2.0 if "OOS" in window_label else 1.3
                alpha = 0.95 if "OOS" in window_label else 0.65
                label = f"{mode.replace('_', ' ')} — {window_label.split('(')[0].strip()}"

                ax.plot(xs, ys, color=color, ls=ls, lw=lw, alpha=alpha,
                        label=label, marker="o", markersize=3)

        # Mark current default threshold (0.05)
        ax.axvline(0.05, color="#ff7b72", lw=1.0, ls=":", alpha=0.7)
        ax.text(0.051, ax.get_ylim()[0], "current\n0.05",
                color="#ff7b72", fontsize=7, va="bottom", alpha=0.8)

    # Shared legend on first panel only
    handles, labels = axes[0, 0].get_legend_handles_labels()
    # Deduplicate
    seen = set()
    unique = [(h, l) for h, l in zip(handles, labels) if l not in seen and not seen.add(l)]
    axes[0, 0].legend(
        *zip(*unique), fontsize=7.5, facecolor="#21262d",
        edgecolor=BORDER_COL, labelcolor=TEXT_COL,
        loc="best", framealpha=0.92,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nChart saved → {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Threshold sensitivity analysis on the 6-asset universe."
    )
    parser.add_argument("--cost", type=float, default=config.TRANSACTION_COST_PCT,
                        help=f"Transaction cost fraction (default: {config.TRANSACTION_COST_PCT})")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    allocation = dict(config.TARGET_ALLOCATION)
    tickers = list(allocation.keys())

    print(f"Fetching prices for {tickers}...")
    all_tickers = list(dict.fromkeys(tickers + [config.BENCHMARK_TICKER]))
    prices = fetch_prices(all_tickers, config.BACKTEST_START, config.BACKTEST_END)

    sweeps: dict[str, dict] = {}
    for window_label, (start, end) in WINDOWS.items():
        monthly = _build_monthly(prices, start, end)
        n_months = len(monthly)
        print(f"\n{window_label}: {monthly.index[0].date()} → {monthly.index[-1].date()} "
              f"({n_months} months)  — running {len(THRESHOLDS) * len(MODES) - 1} scenarios...")
        sweeps[window_label] = _run_sweep(monthly, args.cost)
        _print_table(sweeps[window_label], window_label)

    out_path = os.path.join(os.path.dirname(__file__), "results",
                            "threshold_sensitivity.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    _plot_sensitivity(sweeps, args.cost, out_path)


if __name__ == "__main__":
    main()
