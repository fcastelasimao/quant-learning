"""Stage 12 — EDGE vs Corwin–Schultz half-spread on the real 15-min bars.

The gate for adopting EDGE as the default spread estimator. Both estimators back a proportional
spread `S` out of OHLC; we compute the aggregate half-spread each way (within-session, unbiased
mean, aggregate clamped ≥0 — the calibrate convention) for TQQQ/SQQQ/QQQ and compare. EDGE uses
all four OHLC prices + the previous close and is unbiased under sparse/discrete trading; CS uses
only high/low. If they agree at the ~0.7–1 bp scale, EDGE is a safe drop-in; a large divergence
is a red flag to resolve before re-baselining the capacity chain.

Run:
    /Users/franciscosimao/opt/anaconda3/envs/quant/bin/python \
        research/10_edge_vs_cs/build_10_edge_vs_cs.py
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
from slippage.spread import (  # noqa: E402
    corwin_schultz_intraday, edge_intraday, half_spread_bps,
)

from quantcore import config  # noqa: E402

TICKERS = ["TQQQ", "SQQQ", "QQQ"]
OUT = Path(__file__).resolve().parent / "results"


def load_15min(sym: str) -> pd.DataFrame:
    db = config.data_dir() / f"DB_{sym}_historical_data.db"
    with sqlite3.connect(db) as c:
        d = pd.read_sql("SELECT et_datetime, open, high, low, close FROM candles_15min ORDER BY ts", c)
    d["dt"] = pd.to_datetime(d["et_datetime"])
    return d.set_index("dt")[["open", "high", "low", "close"]]


def _agg_half(s: pd.Series) -> tuple[float, float]:
    """Aggregate half-spread (bps): unbiased mean clamped ≥0, and p90 (stress context)."""
    hs = half_spread_bps(s)
    return float(max(np.nanmean(hs), 0.0)), float(np.nanpercentile(hs.dropna(), 90))


# --------------------------------------------------------------------------- plot
DARK_BG, PANEL_BG, TEXT_COL, GRID_COL = "#0d1117", "#161b22", "#c9d1d9", "#30363d"
CS_COL, EDGE_COL = "#58a6ff", "#f0b429"


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


def make_plot(res_dir: Path):
    """Regenerate edge_vs_cs.png from edge_vs_cs.csv (no DB access)."""
    df = pd.read_csv(res_dir / "edge_vs_cs.csv")
    x = np.arange(len(df))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.patch.set_facecolor(DARK_BG)
    for ax, (cs_col, edge_col, title) in zip((ax1, ax2), (
            ("cs_half_bps", "edge_half_bps", "aggregate mean half-spread"),
            ("cs_p90_bps", "edge_p90_bps", "p90 half-spread (stress context)"))):
        ax.bar(x - 0.18, df[cs_col], width=0.36, color=CS_COL, label="Corwin–Schultz")
        ax.bar(x + 0.18, df[edge_col], width=0.36, color=EDGE_COL, label="EDGE")
        for xi, (a, b) in zip(x, zip(df[cs_col], df[edge_col])):
            ax.text(xi - 0.18, a, f"{a:.2f}", ha="center", va="bottom", fontsize=7, color=TEXT_COL)
            ax.text(xi + 0.18, b, f"{b:.2f}", ha="center", va="bottom", fontsize=7, color=TEXT_COL)
        ax.set_xticks(x)
        ax.set_xticklabels(df["ticker"])
        ax.set_ylabel("half-spread (bps, one-way)")
        ax.set_title(f"S10 — {title}")
        _style_ax(ax)
        ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, fontsize=8, labelcolor=TEXT_COL)
    fig.tight_layout()
    fig.savefig(res_dir / "edge_vs_cs.png", dpi=120, facecolor=DARK_BG)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for sym in TICKERS:
        df = load_15min(sym)
        cs_s = corwin_schultz_intraday(df[["high", "low"]], clamp_negative=False)
        edge_s = edge_intraday(df, clamp_negative=False)
        cs_mean, cs_p90 = _agg_half(cs_s)
        edge_mean, edge_p90 = _agg_half(edge_s)
        rows.append({
            "ticker": sym,
            "cs_half_bps": round(cs_mean, 3), "edge_half_bps": round(edge_mean, 3),
            "diff_bps": round(edge_mean - cs_mean, 3),
            "cs_p90_bps": round(cs_p90, 2), "edge_p90_bps": round(edge_p90, 2),
            "cs_n": int(cs_s.notna().sum()), "edge_n_sessions": int(edge_s.notna().sum()),
        })
    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT / "edge_vs_cs.csv", index=False)
    make_plot(OUT)

    print("=== Half-spread (bps, one-way): EDGE vs Corwin–Schultz ===\n")
    hdr = f"{'ticker':6} {'CS':>8} {'EDGE':>8} {'Δ(E−CS)':>9} {'CS p90':>8} {'EDGE p90':>9}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['ticker']:6} {r['cs_half_bps']:>8.3f} {r['edge_half_bps']:>8.3f} "
              f"{r['diff_bps']:>+9.3f} {r['cs_p90_bps']:>8.2f} {r['edge_p90_bps']:>9.2f}")
    print(f"\n(CS = per-pair within-session; EDGE = per-session, ≥3 bars/day. Both unbiased mean, "
          f"aggregate clamped ≥0.)")
    print(f"Outputs written to {OUT}")


if __name__ == "__main__":
    main()
