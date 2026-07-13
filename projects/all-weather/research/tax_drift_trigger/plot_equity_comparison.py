"""
research/tax_drift_trigger/plot_equity_comparison.py
=====================================================
Produce equity-curve visualisations for the drift-trigger research.

This script runs run_tax_aware_backtest for a curated set of rebalance
policies under FIFO / US-tax, then writes:

  results/equity_comparison/<ts>_<strategy>/
    equity_comparison.csv   — monthly after-tax portfolio value per policy
    rebalance_dates.csv     — every rebalance date, trade notional, policy
    tax_cumulative.csv      — cumulative tax paid over time per policy
    plots/
      equity_curves.png     — overlaid equity curves, all policies
      rebalance_markers.png — same curves + axvline per trigger per policy
      tax_cumulative.png    — cumulative tax paid per policy
      panel.png             — combined 3-panel figure

Designed to be called standalone or imported by notebooks (plot_all_findings.py,
research_overview.py, tax_drift_analysis.py).

Usage
-----
    conda run -n allweather python research/tax_drift_trigger/plot_equity_comparison.py

Options
-------
    --strategy-id     Strategy id (default: 6asset_tip_gsg_rpavg)
    --start-date      ISO date (default: 2018-01-01, widest OOS window)
    --output-root     Root dir for results (default: results/equity_comparison)
    --no-plots        Skip PNG output (CSV artifacts only)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine import config
from engine.backtest import RebalancePolicy
from engine.data import fetch_dividends, fetch_prices
from engine.lot_ledger import LotSelector
from engine.stats import compute_cagr, compute_calmar, compute_max_drawdown
from engine.tax import TaxRegime
from engine.tax_backtest import run_tax_aware_backtest

# ── Policies to compare ──────────────────────────────────────────────────────
# Chosen to span the full findings table: baseline + representative thresholds
# including the recommended production candidate.

POLICIES: dict[str, RebalancePolicy] = {
    "Monthly (baseline)": RebalancePolicy.monthly_unconditional(),
    "Drift abs 2pp": RebalancePolicy.drift_absolute(0.02),
    "Drift abs 5pp": RebalancePolicy.drift_absolute(0.05),  # recommended
    "Drift rel 25%": RebalancePolicy.drift_relative(0.25),
    "Drift rel 40%": RebalancePolicy.drift_relative(0.40),
}

# Colour palette (dark-theme consistent with engine/plotting.py)
DARK_BG = "#0d1117"
PANEL_BG = "#161b22"
TEXT_COL = "#c9d1d9"
GRID_COL = "#30363d"

POLICY_COLORS = {
    "Monthly (baseline)": "#f85149",   # red
    "Drift abs 2pp": "#d29922",        # amber
    "Drift abs 5pp": "#3fb950",        # green  ← recommended
    "Drift rel 25%": "#58a6ff",        # blue
    "Drift rel 40%": "#bc8cff",        # purple
}
POLICY_LS = {
    "Monthly (baseline)": "--",
    "Drift abs 2pp": "-",
    "Drift abs 5pp": "-",
    "Drift rel 25%": "-",
    "Drift rel 40%": "-",
}
POLICY_LW = {
    "Monthly (baseline)": 1.5,
    "Drift abs 2pp": 1.2,
    "Drift abs 5pp": 2.2,   # highlighted
    "Drift rel 25%": 1.2,
    "Drift rel 40%": 1.2,
}

MARKER_ALPHA = {
    "Monthly (baseline)": 0.06,
    "Drift abs 2pp": 0.12,
    "Drift abs 5pp": 0.18,
    "Drift rel 25%": 0.12,
    "Drift rel 40%": 0.12,
}

TRANSACTION_COST = 0.001  # 0.1% per trade, consistent with D.18 sweep


# ── Style helpers ─────────────────────────────────────────────────────────────

def _style_ax(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COL, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID_COL)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.label.set_color(TEXT_COL)
    ax.xaxis.label.set_color(TEXT_COL)
    ax.title.set_color("white")
    ax.grid(axis="y", color=GRID_COL, alpha=0.4, linewidth=0.6)
    ax.grid(axis="x", color=GRID_COL, alpha=0.2, linewidth=0.4)


def _save(fig, path: Path, dpi: int = 150):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved: {path}")


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_data(strategy_id: str, start_date: str, end_date: str
               ) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    canonical = config.resolve_strategy_id(strategy_id)
    allocation = {
        t: float(w)
        for t, w in config.load_strategy(strategy_id)["allocation"].items()
    }
    print(f"  Fetching prices {start_date} → {end_date} for {list(allocation)}")
    prices = fetch_prices(list(allocation), start_date, end_date)
    print(f"  Fetching dividends …")
    dividends = fetch_dividends(list(allocation), start_date, end_date)
    return allocation, prices, dividends


# ── Run all policies ──────────────────────────────────────────────────────────

def run_all_policies(
    allocation: dict,
    prices: pd.DataFrame,
    dividends: pd.DataFrame,
) -> dict[str, object]:
    """Return {policy_label: TaxBacktestResult} for all POLICIES."""
    results = {}
    for label, policy in POLICIES.items():
        print(f"  Running: {label} …", end=" ", flush=True)
        res = run_tax_aware_backtest(
            prices, allocation,
            regime=TaxRegime.us(),
            rebalance_policy=policy,
            lot_selector=LotSelector.FIFO,
            dividends=dividends,
            transaction_cost_pct=TRANSACTION_COST,
        )
        print(f"  {len(res.monthly)} months, "
              f"{int(res.monthly['Rebalanced'].sum())} rebalances, "
              f"Calmar(2018-end)={_calmar_from_2018(res.monthly):.3f}")
        results[label] = res
    return results


def _calmar_from_2018(monthly: pd.DataFrame) -> float:
    sub = monthly.loc[monthly.index >= "2018-01-01", "Value"].dropna()
    if len(sub) < 2:
        return float("nan")
    years = (sub.index[-1] - sub.index[0]).days / 365.25
    cagr = compute_cagr(sub, years)
    mdd = compute_max_drawdown(sub)
    return compute_calmar(round(cagr, 4), round(mdd, 4))


# ── Build CSVs ───────────────────────────────────────────────────────────────

def build_equity_csv(results: dict) -> pd.DataFrame:
    """Wide frame: Date index, one Value column per policy label."""
    frames = {}
    for label, res in results.items():
        frames[label] = res.monthly["Value"].rename(label)
    return pd.concat(frames.values(), axis=1)


def build_rebalance_csv(results: dict) -> pd.DataFrame:
    """Long frame: Date, Policy, Trade Notional, Value at rebalance."""
    rows = []
    for label, res in results.items():
        reb = res.monthly[res.monthly["Rebalanced"]][
            ["Value", "Trade Notional"]
        ].copy()
        reb["Policy"] = label
        reb = reb.reset_index().rename(columns={"Date": "Date"})
        rows.append(reb)
    if not rows:
        return pd.DataFrame(columns=["Date", "Policy", "Value", "Trade Notional"])
    return pd.concat(rows, ignore_index=True)


def build_tax_csv(results: dict) -> pd.DataFrame:
    """Wide frame: Date index, one CumTax column per policy label."""
    frames = {}
    for label, res in results.items():
        frames[label] = res.monthly["Cumulative Tax Paid"].rename(label)
    return pd.concat(frames.values(), axis=1)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_equity_curves(equity_df: pd.DataFrame, title_suffix: str = "") -> plt.Figure:
    """Clean overlaid equity curves (no rebalance markers)."""
    fig, ax = plt.subplots(figsize=(13, 6), facecolor=DARK_BG)
    _style_ax(ax)

    for label in equity_df.columns:
        ax.plot(
            equity_df.index, equity_df[label],
            label=label,
            color=POLICY_COLORS[label],
            linestyle=POLICY_LS[label],
            linewidth=POLICY_LW[label],
            alpha=0.9,
        )

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.set_ylabel("Portfolio value (after-tax, USD)")
    ax.set_title(
        f"After-tax equity curves by rebalance policy (FIFO · US tax){title_suffix}",
        fontweight="bold",
    )
    ax.legend(
        fontsize=9, facecolor=PANEL_BG, edgecolor=GRID_COL,
        labelcolor=TEXT_COL, loc="upper left",
    )
    fig.tight_layout()
    return fig


def plot_equity_with_markers(
    equity_df: pd.DataFrame,
    rebalance_df: pd.DataFrame,
    title_suffix: str = "",
) -> plt.Figure:
    """Equity curves + vertical lines at each rebalance trigger per policy."""
    fig, ax = plt.subplots(figsize=(14, 6), facecolor=DARK_BG)
    _style_ax(ax)

    # Draw rebalance markers first (so they sit behind the lines)
    for label in equity_df.columns:
        alpha = MARKER_ALPHA.get(label, 0.10)
        sub = rebalance_df[rebalance_df["Policy"] == label]
        for _, row in sub.iterrows():
            ax.axvline(
                row["Date"],
                color=POLICY_COLORS[label],
                alpha=alpha,
                linewidth=0.7,
            )

    # Then draw the equity curves on top
    for label in equity_df.columns:
        ax.plot(
            equity_df.index, equity_df[label],
            label=label,
            color=POLICY_COLORS[label],
            linestyle=POLICY_LS[label],
            linewidth=POLICY_LW[label],
            alpha=0.92,
        )

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.set_ylabel("Portfolio value (after-tax, USD)")
    ax.set_title(
        f"Equity curves with rebalance triggers (FIFO · US tax){title_suffix}",
        fontweight="bold",
    )
    ax.legend(
        fontsize=9, facecolor=PANEL_BG, edgecolor=GRID_COL,
        labelcolor=TEXT_COL, loc="upper left",
    )
    # Rebalance count annotation
    for i, label in enumerate(equity_df.columns):
        n = len(rebalance_df[rebalance_df["Policy"] == label])
        ax.annotate(
            f"{n} trades",
            xy=(1.0, 0.97 - i * 0.055),
            xycoords="axes fraction",
            ha="right", va="top",
            fontsize=7.5,
            color=POLICY_COLORS[label],
        )

    fig.tight_layout()
    return fig


def plot_tax_cumulative(tax_df: pd.DataFrame, title_suffix: str = "") -> plt.Figure:
    """Cumulative tax paid over time per rebalance policy."""
    fig, ax = plt.subplots(figsize=(13, 5), facecolor=DARK_BG)
    _style_ax(ax)

    for label in tax_df.columns:
        ax.plot(
            tax_df.index, tax_df[label],
            label=label,
            color=POLICY_COLORS[label],
            linestyle=POLICY_LS[label],
            linewidth=POLICY_LW[label],
            alpha=0.9,
        )

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.set_ylabel("Cumulative tax paid (USD)")
    ax.set_title(
        f"Cumulative tax cost by rebalance policy (FIFO · US tax){title_suffix}",
        fontweight="bold",
    )
    ax.legend(
        fontsize=9, facecolor=PANEL_BG, edgecolor=GRID_COL,
        labelcolor=TEXT_COL, loc="upper left",
    )
    fig.tight_layout()
    return fig


def plot_panel(
    equity_df: pd.DataFrame,
    rebalance_df: pd.DataFrame,
    tax_df: pd.DataFrame,
    title_suffix: str = "",
) -> plt.Figure:
    """Combined 3-panel figure for use in the notebook and findings doc."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 16), facecolor=DARK_BG)
    fig.subplots_adjust(hspace=0.38)

    ax1, ax2, ax3 = axes

    # Panel 1: equity curves
    _style_ax(ax1)
    for label in equity_df.columns:
        ax1.plot(
            equity_df.index, equity_df[label],
            label=label,
            color=POLICY_COLORS[label],
            linestyle=POLICY_LS[label],
            linewidth=POLICY_LW[label],
            alpha=0.9,
        )
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax1.set_title("After-tax equity curve by rebalance policy", fontweight="bold")
    ax1.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COL,
               labelcolor=TEXT_COL, loc="upper left")

    # Panel 2: equity curves + markers
    _style_ax(ax2)
    for label in equity_df.columns:
        alpha = MARKER_ALPHA.get(label, 0.10)
        sub = rebalance_df[rebalance_df["Policy"] == label]
        for _, row in sub.iterrows():
            ax2.axvline(row["Date"], color=POLICY_COLORS[label],
                        alpha=alpha, linewidth=0.7)
    for label in equity_df.columns:
        ax2.plot(
            equity_df.index, equity_df[label],
            label=label,
            color=POLICY_COLORS[label],
            linestyle=POLICY_LS[label],
            linewidth=POLICY_LW[label],
            alpha=0.92,
        )
        n = len(rebalance_df[rebalance_df["Policy"] == label])
        ax2.annotate(
            f"{n} trades",
            xy=(1.0, 0.97 - list(equity_df.columns).index(label) * 0.055),
            xycoords="axes fraction",
            ha="right", va="top", fontsize=7.5,
            color=POLICY_COLORS[label],
        )
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax2.set_title("Equity curves with rebalance-trigger markers", fontweight="bold")
    ax2.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COL,
               labelcolor=TEXT_COL, loc="upper left")

    # Panel 3: cumulative tax
    _style_ax(ax3)
    for label in tax_df.columns:
        ax3.plot(
            tax_df.index, tax_df[label],
            label=label,
            color=POLICY_COLORS[label],
            linestyle=POLICY_LS[label],
            linewidth=POLICY_LW[label],
            alpha=0.9,
        )
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax3.set_title("Cumulative tax paid over time", fontweight="bold")
    ax3.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COL,
               labelcolor=TEXT_COL, loc="upper left")

    fig.suptitle(
        f"Drift-trigger rebalancing: equity, markers, and tax cost{title_suffix}",
        color="white", fontweight="bold", fontsize=13, y=0.995,
    )
    return fig


# ── Summary stats table ───────────────────────────────────────────────────────

def _summary_table(results: dict) -> pd.DataFrame:
    rows = []
    for label, res in results.items():
        monthly = res.monthly
        sub = monthly.loc[monthly.index >= "2018-01-01", "Value"].dropna()
        if len(sub) < 2:
            continue
        years = (sub.index[-1] - sub.index[0]).days / 365.25
        cagr = compute_cagr(sub, years)
        mdd = compute_max_drawdown(sub)
        calmar = compute_calmar(round(cagr, 4), round(mdd, 4))
        n_rebalances = int(monthly["Rebalanced"].sum())
        cum_tax = float(monthly["Cumulative Tax Paid"].iloc[-1])
        rows.append({
            "Policy": label,
            "Calmar (2018–)": round(calmar, 3),
            "CAGR % (2018–)": round(cagr, 2),
            "MDD % (2018–)": round(mdd, 2),
            "Rebalances": n_rebalances,
            "Cum. Tax ($)": round(cum_tax, 0),
            "Final Value ($)": round(float(monthly["Value"].iloc[-1]), 0),
        })
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate equity-curve comparison plots for the drift-trigger research."
    )
    parser.add_argument("--strategy-id", default="6asset_tip_gsg_rpavg")
    parser.add_argument("--start-date", default="2018-01-01",
                        help="Start of OOS window (default: 2018-01-01)")
    parser.add_argument("--end-date", default=None,
                        help="End date (default: latest available)")
    parser.add_argument("--output-root",
                        default=str(PROJECT_ROOT / "results" / "equity_comparison"))
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip PNG generation; write CSVs only")
    args = parser.parse_args()

    strategy_id = args.strategy_id
    start_date = args.start_date
    # Give 6 months of history before the OOS window for initial lot seeding
    data_start = "2006-01-01"
    end_date = args.end_date or datetime.today().strftime("%Y-%m-%d")

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    canonical = config.resolve_strategy_id(strategy_id)
    out_dir = Path(args.output_root) / f"{ts}_{canonical}"
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nEquity comparison: {canonical}")
    print(f"OOS window:  {start_date} → {end_date}")
    print(f"Output dir:  {out_dir}\n")

    # 1. Load data
    allocation, prices, dividends = _load_data(canonical, data_start, end_date)

    # 2. Run all policies
    print("\nRunning tax-aware backtests …")
    results = run_all_policies(allocation, prices, dividends)

    # 3. Build CSVs
    equity_df = build_equity_csv(results)
    rebalance_df = build_rebalance_csv(results)
    tax_df = build_tax_csv(results)

    # Trim to OOS window for visualisation (full history for computation)
    equity_plot = equity_df.loc[equity_df.index >= start_date]
    rebalance_plot = rebalance_df[rebalance_df["Date"] >= pd.Timestamp(start_date)]
    tax_plot = tax_df.loc[tax_df.index >= start_date]

    # 4. Summary table
    summary = _summary_table(results)
    print(f"\nSummary (OOS from {start_date}):\n")
    print(summary.to_string(index=False))

    # 5. Save CSVs
    equity_df.reset_index().to_csv(out_dir / "equity_comparison.csv", index=False)
    rebalance_df.to_csv(out_dir / "rebalance_dates.csv", index=False)
    tax_df.reset_index().to_csv(out_dir / "tax_cumulative.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)

    run_cfg = {
        "strategy_id": canonical,
        "allocation": allocation,
        "start_date": start_date,
        "data_start": data_start,
        "end_date": end_date,
        "regime": "us",
        "selector": "fifo",
        "transaction_cost_pct": TRANSACTION_COST,
        "policies": list(POLICIES.keys()),
        "generated_at": ts,
    }
    (out_dir / "run_config.json").write_text(
        json.dumps(run_cfg, indent=2), encoding="utf-8"
    )

    print(f"\nCSV artifacts written to: {out_dir}")

    # 6. Plots
    if not args.no_plots:
        print("\nGenerating plots …")
        title_suffix = f"\n{start_date} → {end_date}  |  {canonical}"

        fig1 = plot_equity_curves(equity_plot, title_suffix)
        _save(fig1, plots_dir / "equity_curves.png")

        fig2 = plot_equity_with_markers(equity_plot, rebalance_plot, title_suffix)
        _save(fig2, plots_dir / "rebalance_markers.png")

        fig3 = plot_tax_cumulative(tax_plot, title_suffix)
        _save(fig3, plots_dir / "tax_cumulative.png")

        fig4 = plot_panel(equity_plot, rebalance_plot, tax_plot, title_suffix)
        _save(fig4, plots_dir / "panel.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
