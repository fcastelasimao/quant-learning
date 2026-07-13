"""E02 — Interruption risk: the hazard of a mid-fill trailing-stop exit / signal flip.

The two mid-fill hazards the scheduler must plan for: the trailing stop fires, or the signal
flips, while an order is still being worked. Both are measurable from the existing TRADES CSVs
with no live experiment needed — a trade's `hold_days` already tells us exactly when the position
was closed, and `exit_reason` tells us why.

  1. Hazard curve P(trade exits within h of entry) — overall, by volatility regime (S03's exact
     tercile split), and by `exit_reason`.
  2. A simple interruption-cost model: if interrupted at filled-fraction phi, either the residue
     is cancelled (forfeit its edge entirely) or execution completes immediately (uniform g(h)
     forfeiture across the whole order) — both parametrized, both Modeled, not Measured.

Note on scope: the TRADES CSVs' `exit_reason` is overwhelmingly TRAIL_STOP (>99.9% of trades) —
there is no separately-labeled "signal flip" exit in this dataset, so the hazard curve measures
the *observed* exit-timing hazard (dominated by the trailing stop) as a proxy for "any
interruption", not a cause-decomposed one. Flagged in findings, not silently assumed away.

Run:
    /Users/franciscosimao/opt/anaconda3/envs/quant/bin/python \
        research/scheduler/02_interruption_risk/build_02_interruption_risk.py
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

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from quantcore import config  # noqa: E402

TICKERS = ["TQQQ", "SQQQ"]
TRADES_ROOT = Path(__file__).resolve().parents[4] / "TQQQ_SQQQ_analysis"
OUT = Path(__file__).resolve().parent / "results"
HAZARD_GRID_MIN = [15, 30, 60, 120, 240, 390, 780, 1560, 3120]   # 15m .. ~1 week

DARK_BG, PANEL_BG, TEXT_COL, GRID_COL = "#0d1117", "#161b22", "#c9d1d9", "#30363d"
TICKER_COLORS = {"TQQQ": "#58a6ff", "SQQQ": "#f0b429"}
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


def load_trades(sym: str) -> pd.DataFrame:
    path = TRADES_ROOT / "full_history_canonical" / f"TRADES_{sym}_full_history.csv"
    df = pd.read_csv(path, parse_dates=["entry_time"])[["entry_time", "hold_days", "exit_reason"]]
    df["hold_min"] = df["hold_days"] * 1440.0
    return df


def vol_regime(sym: str) -> pd.Series:
    """Reproduces S03's exact 20-day trailing realized-vol tercile split."""
    db = config.data_dir() / f"DB_{sym}_historical_data.db"
    with sqlite3.connect(db) as c:
        d = pd.read_sql("SELECT et_datetime, close FROM candles_1d ORDER BY ts", c)
    d["dt"] = pd.to_datetime(d["et_datetime"])
    rv = d.set_index("dt")["close"].pct_change().rolling(20).std()
    q1, q2 = rv.quantile([1 / 3, 2 / 3])
    reg = pd.cut(rv, [-np.inf, q1, q2, np.inf], labels=["calm", "normal", "stress"])
    reg.index = reg.index.normalize()
    return reg


def hazard_curve(hold_min: pd.Series) -> pd.Series:
    """Empirical P(exit within h) at each grid point."""
    return pd.Series({h: float((hold_min <= h).mean()) for h in HAZARD_GRID_MIN})


def make_plot(curves: dict[str, pd.Series], by_regime: dict[str, dict[str, pd.Series]], path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor(DARK_BG)
    for sym, curve in curves.items():
        ax1.plot(curve.index, curve.values, marker="o", ms=4, color=TICKER_COLORS[sym], label=sym)
    ax1.axvline(168, color="#bc8cff", ls=":", lw=1.0, label="p25 hold ~2.8h (168min)")
    ax1.set_xscale("log")
    ax1.set_xlabel("h (min, log scale)")
    ax1.set_ylabel("P(exit within h)")
    ax1.set_title("Interruption hazard curve, overall")
    _style_ax(ax1)
    ax1.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, fontsize=8, labelcolor=TEXT_COL)

    for rg, curve in by_regime["TQQQ"].items():
        ax2.plot(curve.index, curve.values, marker="o", ms=4, color=REGIME_COLORS[rg], label=rg)
    ax2.set_xscale("log")
    ax2.set_xlabel("h (min, log scale)")
    ax2.set_ylabel("P(exit within h)")
    ax2.set_title("TQQQ hazard by volatility regime")
    _style_ax(ax2)
    ax2.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, fontsize=8, labelcolor=TEXT_COL)

    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=DARK_BG)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    curves, by_regime_all = {}, {}
    for sym in TICKERS:
        trades = load_trades(sym)
        print(f"=== {sym}: n={len(trades)} trades ===")
        print(f"  exit_reason counts: {trades['exit_reason'].value_counts().to_dict()}")
        p25 = trades["hold_min"].quantile(0.25)
        print(f"  p25 hold = {p25:.0f} min ({p25/60:.1f}h), p10 = "
              f"{trades['hold_min'].quantile(0.10):.0f} min")

        curve = hazard_curve(trades["hold_min"])
        curves[sym] = curve
        curve.to_csv(OUT / f"hazard_overall_{sym}.csv", header=["p_exit_by_h"])
        print(f"  overall hazard:\n{(curve*100).round(1).to_string()}\n  (values in %)")

        reg = vol_regime(sym)
        trades["regime"] = reg.reindex(trades["entry_time"].dt.normalize()).astype(object).values
        by_regime = {}
        for rg, g in trades.groupby("regime", observed=True):
            by_regime[rg] = hazard_curve(g["hold_min"])
        by_regime_all[sym] = by_regime
        pd.DataFrame(by_regime).to_csv(OUT / f"hazard_by_regime_{sym}.csv")
        print(f"  by regime (h=240min / 4h): "
              f"{ {rg: round(float(c.loc[240])*100, 1) for rg, c in by_regime.items()} } (%)")

        by_reason = {}
        for reason, g in trades.groupby("exit_reason"):
            if len(g) >= 5:
                by_reason[reason] = hazard_curve(g["hold_min"])
        if by_reason:
            pd.DataFrame(by_reason).to_csv(OUT / f"hazard_by_exit_reason_{sym}.csv")
        print()

    make_plot(curves, by_regime_all, OUT / "interruption_hazard.png")
    print(f"Outputs written to {OUT}")


if __name__ == "__main__":
    main()
