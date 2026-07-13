"""Diagnostic: how the Corwin–Schultz spread estimate depends on bar resolution,
time period, and volatility regime — for TQQQ / SQQQ / QQQ.

Motivation: CS has no plateau (the estimate ramps with bar size) and breaks at very
fine resolution. This script maps that ramp across 1→30 min, checks whether it
stabilises by 30-min, and slices it by 5-year window and by volatility regime.

All intraday bars are resampled from the 1-min source, **open-aligned per session**
(bins start at 09:30), so 20/25-min — which don't divide the 390-min day evenly —
are handled correctly and no bin ever spans the overnight gap.

Signed (un-clamped) means are reported throughout: a negative signed mean is the
signature of the estimator breaking down, so we must not hide it by clamping.

Run:
    /Users/franciscosimao/opt/anaconda3/envs/quant/bin/python \
        research/01_spread_estimate/diagnose_resolution.py
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
from slippage.spread import corwin_schultz_intraday, half_spread_bps  # noqa: E402

from quantcore import config  # noqa: E402

TICKERS = ["TQQQ", "SQQQ", "QQQ"]
RESOLUTIONS = [1, 5, 10, 15, 20, 25, 30]  # minutes
OPEN_MIN = 570   # 09:30 in minutes since midnight
CLOSE_MIN = 390  # length of the RTH session in minutes
OUT = Path(__file__).resolve().parent / "results"

DARK_BG, PANEL_BG, TEXT_COL, GRID_COL = "#0d1117", "#161b22", "#c9d1d9", "#30363d"
TICKER_COLORS = {"TQQQ": "#58a6ff", "SQQQ": "#f0b429", "QQQ": "#3fb950"}
SEQ = ["#58a6ff", "#f0b429", "#f78166", "#3fb950", "#bc8cff"]


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


def load_1min(sym: str) -> pd.DataFrame:
    db = config.data_dir() / f"DB_{sym}_historical_data.db"
    with sqlite3.connect(db) as c:
        df = pd.read_sql("SELECT et_datetime,high,low FROM candles_1min ORDER BY ts", c)
    df["dt"] = pd.to_datetime(df["et_datetime"])
    return df.set_index("dt")[["high", "low"]]


def resample_open_aligned(df1: pd.DataFrame, freq_min: int) -> pd.DataFrame:
    """Aggregate 1-min high/low into freq-min bars aligned to the 09:30 open,
    one set of bins per trading day. Returns a DatetimeIndex frame (label = bin open)."""
    if freq_min == 1:
        return df1
    idx = df1.index
    mins = idx.hour * 60 + idx.minute - OPEN_MIN
    keep = (mins >= 0) & (mins < CLOSE_MIN)
    df1, mins = df1[keep], mins[keep]
    day = df1.index.normalize()
    binid = (mins // freq_min).astype(int)
    out = df1.groupby([day, binid]).agg(high=("high", "max"), low=("low", "min"))
    out.index.names = ["day", "binid"]
    out = out.reset_index()
    out["dt"] = out["day"] + pd.to_timedelta(OPEN_MIN + out["binid"] * freq_min, unit="m")
    return out.set_index("dt")[["high", "low"]].sort_index()


def halfspread_series(df1: pd.DataFrame, freq_min: int) -> pd.Series:
    """Signed per-pair half-spread (bps), within-session, at the given resolution."""
    bars = resample_open_aligned(df1, freq_min)
    s = corwin_schultz_intraday(bars[["high", "low"]], clamp_negative=False)
    return half_spread_bps(s).dropna()


def vol_regime(sym: str) -> pd.Series:
    """Daily volatility-regime label (calm/normal/stress) by terciles of trailing
    20-day realized vol of daily close-to-close returns. Indexed by date."""
    db = config.data_dir() / f"DB_{sym}_historical_data.db"
    with sqlite3.connect(db) as c:
        d = pd.read_sql("SELECT et_datetime,close FROM candles_1d ORDER BY ts", c)
    d["dt"] = pd.to_datetime(d["et_datetime"])
    rv = d.set_index("dt")["close"].pct_change().rolling(20).std()
    q1, q2 = rv.quantile([1 / 3, 2 / 3])
    reg = pd.cut(rv, [-np.inf, q1, q2, np.inf], labels=["calm", "normal", "stress"])
    reg.index = reg.index.normalize()
    return reg


def window_label(years: pd.Index) -> pd.Series:
    edges = [(2010, 2014), (2015, 2019), (2020, 2024), (2025, 2026)]
    out = pd.Series(index=years, dtype=object)
    for lo, hi in edges:
        out[(years >= lo) & (years <= hi)] = f"{lo}–{hi}"
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # Precompute the signed half-spread series once per (ticker, resolution).
    series = {t: {} for t in TICKERS}
    for t in TICKERS:
        d1 = load_1min(t)
        for r in RESOLUTIONS:
            series[t][r] = halfspread_series(d1, r)

    # (1) Resolution sweep by ticker.
    sweep = pd.DataFrame(
        {t: {r: series[t][r].mean() for r in RESOLUTIONS} for t in TICKERS}
    )
    sweep.index.name = "resolution_min"
    sweep.to_csv(OUT / "diag_sweep_by_ticker.csv")

    # (2) Resolution sweep by 5-year window (per ticker).
    rows = []
    for t in TICKERS:
        for r in RESOLUTIONS:
            hs = series[t][r]
            win = window_label(hs.index.year)
            for w, m in hs.groupby(win.values).mean().items():
                rows.append({"ticker": t, "window": w, "resolution_min": r, "signed_mean_bps": m})
    by_window = pd.DataFrame(rows)
    by_window.to_csv(OUT / "diag_sweep_by_window.csv", index=False)

    # (3) Resolution sweep by volatility regime (per ticker).
    rows = []
    regimes = {t: vol_regime(t) for t in TICKERS}
    for t in TICKERS:
        reg = regimes[t]
        for r in RESOLUTIONS:
            hs = series[t][r]
            lab = reg.reindex(hs.index.normalize()).astype(object).values
            tmp = pd.DataFrame({"hs": hs.values, "regime": lab}).dropna()
            g = tmp.groupby("regime")["hs"].mean()
            for rg, m in g.items():
                rows.append({"ticker": t, "regime": rg, "resolution_min": r, "signed_mean_bps": m})
    by_regime = pd.DataFrame(rows)
    by_regime.to_csv(OUT / "diag_sweep_by_regime.csv", index=False)

    _plot(sweep, by_window, by_regime)
    _print_summary(sweep, by_window, by_regime)
    print(f"\nOutputs written to {OUT}")


def _plot(sweep, by_window, by_regime):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor(DARK_BG)

    ax = axes[0]
    for t in TICKERS:
        ax.plot(sweep.index, sweep[t], marker="o", ms=4, color=TICKER_COLORS[t], label=t)
    ax.axhline(0, color="#f78166", ls="--", lw=0.9)
    ax.set_title("By ticker")
    ax.set_xlabel("bar resolution (min)")
    ax.set_ylabel("signed mean half-spread (bps)")
    _style_ax(ax)
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, fontsize=8, labelcolor=TEXT_COL)

    ax = axes[1]
    tq = by_window[by_window.ticker == "TQQQ"]
    for i, (w, g) in enumerate(tq.groupby("window")):
        ax.plot(g.resolution_min, g.signed_mean_bps, marker="o", ms=4, color=SEQ[i % len(SEQ)], label=w)
    ax.axhline(0, color="#f78166", ls="--", lw=0.9)
    ax.set_title("TQQQ by 5-year window")
    ax.set_xlabel("bar resolution (min)")
    _style_ax(ax)
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, fontsize=8, labelcolor=TEXT_COL)

    ax = axes[2]
    tq = by_regime[by_regime.ticker == "TQQQ"]
    order = {"calm": 0, "normal": 1, "stress": 2}
    for rg, g in sorted(tq.groupby("regime"), key=lambda kv: order.get(kv[0], 9)):
        ax.plot(g.resolution_min, g.signed_mean_bps, marker="o", ms=4,
                color=SEQ[order.get(rg, 0)], label=rg)
    ax.axhline(0, color="#f78166", ls="--", lw=0.9)
    ax.set_title("TQQQ by volatility regime")
    ax.set_xlabel("bar resolution (min)")
    _style_ax(ax)
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, fontsize=8, labelcolor=TEXT_COL)

    fig.tight_layout()
    fig.savefig(OUT / "diag_resolution.png", dpi=120, facecolor=DARK_BG)
    plt.close(fig)


def _print_summary(sweep, by_window, by_regime):
    pd.set_option("display.float_format", lambda x: f"{x:.2f}")
    print("=== Signed mean half-spread (bps) vs resolution, by ticker ===")
    print(sweep.to_string())
    print("\n=== TQQQ by 5-year window (signed mean bps) ===")
    piv = by_window[by_window.ticker == "TQQQ"].pivot(
        index="resolution_min", columns="window", values="signed_mean_bps")
    print(piv.to_string())
    print("\n=== TQQQ by volatility regime (signed mean bps) ===")
    piv = by_regime[by_regime.ticker == "TQQQ"].pivot(
        index="resolution_min", columns="regime", values="signed_mean_bps")
    print(piv[["calm", "normal", "stress"]].to_string())


if __name__ == "__main__":
    main()
