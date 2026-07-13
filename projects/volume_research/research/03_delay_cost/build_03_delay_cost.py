"""Block 3 / Stage 3 — Delay / timing cost for TQQQ / SQQQ / QQQ.

Measures the signed price drift from each 15-min decision point to a fill `k` minutes
later, from 1-min closes. The mean is ≈0 (drift is unpredictable) so this is a *risk*
(the std), not a systematic drag — but for 3× ETFs the per-minute vol is large, so the
timing risk is expected to dwarf the ~1 bp spread floor from Block 1.

Outputs: a per-horizon summary (mean/std/tails), a by-volatility-regime breakdown, a
dark-theme PNG (std vs horizon with √-time reference), and a console summary comparing
the timing risk against the spread floor and the backtest's flat 15 bps exit.

Run:
    /Users/franciscosimao/opt/anaconda3/envs/quant/bin/python \
        research/03_delay_cost/build_03_delay_cost.py
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from slippage.delay import delay_costs_bps, timing_risk_bps  # noqa: E402

from quantcore import config  # noqa: E402

TICKERS = ["TQQQ", "SQQQ", "QQQ"]
HORIZONS = [1, 2, 3, 5, 10, 15]   # minutes between decision and fill
SPREAD_FLOOR_BPS = 1.0            # Block 1 typical half-spread, for context
FLAT_EXIT_BPS = 15.0             # the size-blind exit cost baked into current backtests
DIST_HORIZONS = [1, 5, 15]       # horizons shown in the distribution plot
DIST_CLIP_BPS = 150              # x-range for the density view (tails extend past this)
OUT = Path(__file__).resolve().parent / "results"

DARK_BG, PANEL_BG, TEXT_COL, GRID_COL = "#0d1117", "#161b22", "#c9d1d9", "#30363d"
TICKER_COLORS = {"TQQQ": "#58a6ff", "SQQQ": "#f0b429", "QQQ": "#3fb950"}
REGIME_COLORS = {"calm": "#58a6ff", "normal": "#f0b429", "stress": "#f78166"}


def _style_ax(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COL, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID_COL)
    ax.grid(axis="y", color=GRID_COL, alpha=0.5, linewidth=0.7)
    for lab in (ax.yaxis.label, ax.xaxis.label, ax.title):
        lab.set_color(TEXT_COL)


def load_close_1min(sym: str) -> pd.Series:
    db = config.data_dir() / f"DB_{sym}_historical_data.db"
    with sqlite3.connect(db) as c:
        df = pd.read_sql("SELECT et_datetime, close FROM candles_1min ORDER BY ts", c)
    df["dt"] = pd.to_datetime(df["et_datetime"])
    return df.set_index("dt")["close"]


def vol_regime(sym: str) -> pd.Series:
    """Daily calm/normal/stress label by terciles of trailing 20-day realized vol."""
    db = config.data_dir() / f"DB_{sym}_historical_data.db"
    with sqlite3.connect(db) as c:
        d = pd.read_sql("SELECT et_datetime, close FROM candles_1d ORDER BY ts", c)
    d["dt"] = pd.to_datetime(d["et_datetime"])
    rv = d.set_index("dt")["close"].pct_change().rolling(20).std()
    q1, q2 = rv.quantile([1 / 3, 2 / 3])
    reg = pd.cut(rv, [-np.inf, q1, q2, np.inf], labels=["calm", "normal", "stress"])
    reg.index = reg.index.normalize()
    return reg


def by_regime_std(costs: pd.DataFrame, reg: pd.Series) -> pd.DataFrame:
    lab = reg.reindex(costs.index.normalize()).astype(object).values
    rows = []
    for k in costs.columns:
        tmp = pd.DataFrame({"v": costs[k].values, "regime": lab}).dropna()
        for rg, std in tmp.groupby("regime")["v"].std().items():
            rows.append({"regime": rg, "horizon_min": k, "std_bps": std})
    return pd.DataFrame(rows)


def make_plot(summaries: dict[str, pd.DataFrame], tq_regime: pd.DataFrame, path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(DARK_BG)

    for t, summ in summaries.items():
        ax1.plot(summ.index, summ["std_bps"], marker="o", ms=4, color=TICKER_COLORS[t], label=t)
    # √-time reference anchored at horizon 1 for TQQQ.
    s1 = summaries["TQQQ"].loc[1, "std_bps"]
    hz = np.array(HORIZONS)
    ax1.plot(hz, s1 * np.sqrt(hz), color=TEXT_COL, ls=":", lw=1.0, label="√t reference (TQQQ)")
    ax1.axhline(SPREAD_FLOOR_BPS, color="#3fb950", ls="--", lw=0.9, label=f"spread floor (~{SPREAD_FLOOR_BPS:.0f} bp)")
    ax1.axhline(FLAT_EXIT_BPS, color="#bc8cff", ls="--", lw=0.9, label=f"flat exit ({FLAT_EXIT_BPS:.0f} bps)")
    ax1.set_title("Timing risk (std of delay cost) vs fill horizon")
    ax1.set_xlabel("fill delay (min)")
    ax1.set_ylabel("timing risk, 1σ (bps)")
    _style_ax(ax1)
    ax1.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, fontsize=7, labelcolor=TEXT_COL)

    for rg, g in tq_regime.groupby("regime"):
        ax2.plot(g["horizon_min"], g["std_bps"], marker="o", ms=4,
                 color=REGIME_COLORS.get(rg, TEXT_COL), label=rg)
    ax2.set_title("TQQQ timing risk by volatility regime")
    ax2.set_xlabel("fill delay (min)")
    ax2.set_ylabel("timing risk, 1σ (bps)")
    _style_ax(ax2)
    ax2.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, fontsize=8, labelcolor=TEXT_COL)

    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=DARK_BG)
    plt.close(fig)


def make_distribution_plot(costs: pd.DataFrame, summ: pd.DataFrame, path: Path, ticker="TQQQ"):
    """Show the delay-cost *distribution*: it stays centred at ≈0 but widens with horizon.

    Left  — overlaid densities for a few horizons (same centre, growing spread).
    Right — the per-horizon mean (the ±0.4 bps wobble) inside the ±1σ risk band, so it is
            visibly negligible: the horizon changes the distribution's *width*, not its *centre*.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(DARK_BG)

    bins = np.linspace(-DIST_CLIP_BPS, DIST_CLIP_BPS, 161)
    centers = 0.5 * (bins[:-1] + bins[1:])
    dist_colors = {1: "#58a6ff", 5: "#f0b429", 15: "#f78166"}
    for k in DIST_HORIZONS:
        v = costs[k].dropna().values
        dens, _ = np.histogram(v, bins=bins, density=True)
        ax1.plot(centers, dens, color=dist_colors.get(k, TEXT_COL), lw=1.4,
                 label=f"{k}-min (σ={summ.loc[k, 'std_bps']:.0f}, mean={summ.loc[k, 'mean_bps']:+.2f})")
    ax1.axvline(0, color=TEXT_COL, ls=":", lw=1.0)
    ax1.set_title(f"{ticker} delay-cost distribution by fill horizon")
    ax1.set_xlabel("delay cost (bps)")
    ax1.set_ylabel("density")
    _style_ax(ax1)
    ax1.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, fontsize=8, labelcolor=TEXT_COL)

    h = summ.index.values.astype(float)
    mean, std = summ["mean_bps"].values, summ["std_bps"].values
    n = costs.count().reindex(summ.index).values
    se = std / np.sqrt(n)
    ax2.fill_between(h, -std, std, color="#58a6ff", alpha=0.15, label="±1σ (the risk)")
    ax2.errorbar(h, mean, yerr=se, marker="o", ms=5, color="#f78166", lw=1.2,
                 capsize=3, label="mean ± standard error")
    ax2.axhline(0, color=TEXT_COL, ls=":", lw=1.0)
    ax2.set_title(f"{ticker} mean delay cost vs the ±1σ risk band")
    ax2.set_xlabel("fill delay (min)")
    ax2.set_ylabel("bps")
    _style_ax(ax2)
    ax2.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, fontsize=8, labelcolor=TEXT_COL)

    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=DARK_BG)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    summaries, regime_rows = {}, []
    tq_costs = None
    for t in TICKERS:
        costs = delay_costs_bps(load_close_1min(t), HORIZONS)
        summ = timing_risk_bps(costs)
        summ.index.name = "horizon_min"
        summaries[t] = summ
        if t == "TQQQ":
            tq_costs = costs
        rg = by_regime_std(costs, vol_regime(t))
        rg["ticker"] = t
        regime_rows.append(rg)

    summary_df = pd.concat([s.assign(ticker=t) for t, s in summaries.items()]).reset_index()
    summary_df.to_csv(OUT / "delay_summary.csv", index=False)
    by_regime = pd.concat(regime_rows, ignore_index=True)
    by_regime.to_csv(OUT / "delay_by_regime.csv", index=False)
    make_plot(summaries, by_regime[by_regime.ticker == "TQQQ"], OUT / "delay_cost.png")
    make_distribution_plot(tq_costs, summaries["TQQQ"], OUT / "delay_distribution.png")

    pd.set_option("display.float_format", lambda x: f"{x:.2f}")
    print("=== Timing risk (std of delay cost, bps) by horizon ===")
    pivot = summary_df.pivot(index="horizon_min", columns="ticker", values="std_bps")
    print(pivot[TICKERS].to_string())
    print(f"\nContext: spread floor ~{SPREAD_FLOOR_BPS:.0f} bp, flat backtest exit {FLAT_EXIT_BPS:.0f} bps.")
    for t in TICKERS:
        s = summaries[t]["std_bps"]
        print(f"  {t}: 1-min {s[1]:.1f} bps · 5-min {s[5]:.1f} bps · 15-min {s[15]:.1f} bps (1σ)")
    print(f"\nOutputs written to {OUT}")


if __name__ == "__main__":
    main()
