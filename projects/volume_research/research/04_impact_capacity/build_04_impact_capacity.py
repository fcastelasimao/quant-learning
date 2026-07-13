"""Block 4 / Stage 4 — Market impact (√-law) and the capacity curve.

The headline deliverable. We measure what we *can* from data — daily volatility σ and
dollar volume V — for TQQQ/SQQQ/QQQ, then apply the square-root impact law with the
constant Y **adopted from the literature** (Almgren 2005 / Bouchaud, Y ≈ 0.3–1.0; we
report a band, never a single line, because Y is NOT identifiable from our OHLC).

Outputs: an impact-vs-order-size curve (with Y band), a capacity table ($ tradable before
impact crosses a budget) for normal vs stress conditions, a dark-theme PNG, and a console
summary. Stress = thin volume (10th-pct day) + high vol (90th-pct regime) together.

Run:
    /Users/franciscosimao/opt/anaconda3/envs/quant/bin/python \
        research/04_impact_capacity/build_04_impact_capacity.py
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
from slippage.impact import capacity, impact_bps  # noqa: E402

from quantcore import config  # noqa: E402

TICKERS = ["TQQQ", "SQQQ", "QQQ"]
WINDOW = 504               # ~2 trading years for "current" conditions
Y_BAND = (0.3, 0.5, 1.0)   # (optimistic, central, conservative) — literature prior
BUDGETS_BPS = [5, 10, 25, 50]
CEILING_USD = {"TQQQ": 50e6, "SQQQ": 50e6, "QQQ": 500e6}  # single-name √-law validity (soft)
OUT = Path(__file__).resolve().parent / "results"

DARK_BG, PANEL_BG, TEXT_COL, GRID_COL = "#0d1117", "#161b22", "#c9d1d9", "#30363d"
TICKER_COLORS = {"TQQQ": "#58a6ff", "SQQQ": "#f0b429", "QQQ": "#3fb950"}


def _style_ax(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COL, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID_COL)
    ax.grid(True, color=GRID_COL, alpha=0.4, linewidth=0.7)
    for lab in (ax.yaxis.label, ax.xaxis.label, ax.title):
        lab.set_color(TEXT_COL)


def market_params(sym: str) -> dict:
    """Measured inputs: daily σ (bps) and dollar volume V ($), normal and stress."""
    db = config.data_dir() / f"DB_{sym}_historical_data.db"
    with sqlite3.connect(db) as c:
        d = pd.read_sql("SELECT et_datetime, close, volume FROM candles_1d ORDER BY ts", c)
    d["dt"] = pd.to_datetime(d["et_datetime"])
    d = d.set_index("dt").tail(WINDOW)
    ret = d["close"].pct_change()
    rv_bps = ret.rolling(20).std() * 1e4          # rolling 20-day vol, bps
    dollar_vol = d["volume"] * d["close"]
    return {
        # σ normal is a right-skewed series → report both (findings_09). Headline = mean
        # (expected-cost / time-average moment); median = typical-day sensitivity.
        "sigma_normal": rv_bps.mean(),
        "sigma_normal_median": rv_bps.median(),
        "sigma_stress": rv_bps.quantile(0.90),
        "adv_normal": dollar_vol.mean(),
        "adv_stress": dollar_vol.quantile(0.10),
        "price": float(d["close"].iloc[-1]),
    }


def capacity_table(params: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for t, p in params.items():
        for cond, sig, V in [("normal", p["sigma_normal"], p["adv_normal"]),
                             ("stress", p["sigma_stress"], p["adv_stress"])]:
            for b in BUDGETS_BPS:
                lo = capacity(b, V, sig, Y=Y_BAND[2])   # conservative Y -> small capacity
                mid = capacity(b, V, sig, Y=Y_BAND[1])
                hi = capacity(b, V, sig, Y=Y_BAND[0])   # optimistic Y -> large capacity
                rows.append({"ticker": t, "condition": cond, "budget_bps": b,
                             "cap_usd_low": lo, "cap_usd_mid": mid, "cap_usd_high": hi})
    return pd.DataFrame(rows)


def make_plot(params, captab, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.patch.set_facecolor(DARK_BG)

    # Panel 1: impact vs order size for TQQQ, Y band + stress.
    p = params["TQQQ"]
    sizes = np.logspace(5, 9, 60)  # $100k -> $1B
    lo = impact_bps(sizes, p["adv_normal"], p["sigma_normal"], Y=Y_BAND[0])
    hi = impact_bps(sizes, p["adv_normal"], p["sigma_normal"], Y=Y_BAND[2])
    mid = impact_bps(sizes, p["adv_normal"], p["sigma_normal"], Y=Y_BAND[1])
    mid_stress = impact_bps(sizes, p["adv_stress"], p["sigma_stress"], Y=Y_BAND[1])
    ax1.fill_between(sizes, lo, hi, color="#58a6ff", alpha=0.22, label="Y∈[0.3,1.0] (normal)")
    ax1.plot(sizes, mid, color="#58a6ff", lw=1.5, label="Y=0.5 central (normal)")
    ax1.plot(sizes, mid_stress, color="#f78166", lw=1.5, ls="--", label="Y=0.5 (stress)")
    ax1.axhline(10, color="#3fb950", ls=":", lw=1.0, label="10 bps budget")
    ax1.axvline(CEILING_USD["TQQQ"], color="#bc8cff", ls=":", lw=1.0,
                label=f"single-name ceiling ~${CEILING_USD['TQQQ']/1e6:.0f}M")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_title("TQQQ market impact vs order size (√-law)")
    ax1.set_xlabel("order size ($)"); ax1.set_ylabel("impact (bps)")
    _style_ax(ax1)
    ax1.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, fontsize=7, labelcolor=TEXT_COL)

    # Panel 2: capacity at 10 bps per ticker, normal vs stress, band as error bars.
    sub = captab[captab.budget_bps == 10]
    x = np.arange(len(TICKERS))
    for i, cond in enumerate(["normal", "stress"]):
        s = sub[sub.condition == cond].set_index("ticker").loc[TICKERS]
        mid = s["cap_usd_mid"].values
        err = np.vstack([mid - s["cap_usd_low"].values, s["cap_usd_high"].values - mid])
        ax2.errorbar(x + (i - 0.5) * 0.18, mid, yerr=err, fmt="o", ms=6, capsize=4,
                     color=("#58a6ff" if cond == "normal" else "#f78166"), label=cond)
    ax2.set_yscale("log")
    ax2.set_xticks(x); ax2.set_xticklabels(TICKERS)
    ax2.set_title("Capacity at 10 bps impact budget ($, band over Y)")
    ax2.set_ylabel("max order size ($)")
    _style_ax(ax2)
    ax2.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, fontsize=8, labelcolor=TEXT_COL)

    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=DARK_BG)
    plt.close(fig)


def _usd(x):
    if x >= 1e9:
        return f"${x/1e9:.2f}B"
    if x >= 1e6:
        return f"${x/1e6:.0f}M"
    return f"${x/1e3:.0f}k"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    params = {t: market_params(t) for t in TICKERS}
    captab = capacity_table(params)

    pd.DataFrame(params).T.to_csv(OUT / "market_params.csv")
    captab.to_csv(OUT / "capacity_table.csv", index=False)
    make_plot(params, captab, OUT / "impact_capacity.png")

    print("=== Measured inputs (recent ~2y) ===")
    for t, p in params.items():
        print(f"  {t}: σ≈{p['sigma_normal']:.0f} bps/day mean (median {p['sigma_normal_median']:.0f}, "
              f"stress {p['sigma_stress']:.0f}), "
              f"$ADV≈{_usd(p['adv_normal'])} (thin {_usd(p['adv_stress'])}), px ${p['price']:.0f}")
    print("\n=== Capacity at 10 bps impact budget (Y∈[0.3,1.0], central 0.5) ===")
    print("  headline σ = mean (expected-cost); 'typical' = median-σ typical-day sensitivity")
    sub = captab[captab.budget_bps == 10]
    for t in TICKERS:
        for cond in ["normal", "stress"]:
            r = sub[(sub.ticker == t) & (sub.condition == cond)].iloc[0]
            print(f"  {t} {cond:6}: {_usd(r['cap_usd_mid'])}  "
                  f"[{_usd(r['cap_usd_low'])} – {_usd(r['cap_usd_high'])}]")
        cap_typ = capacity(10, params[t]["adv_normal"], params[t]["sigma_normal_median"], Y=Y_BAND[1])
        print(f"  {t} typical: {_usd(cap_typ)}  (median σ, central Y — typical-day sensitivity)")
    print("\nNOTE: Y is adopted from literature, not fitted — capacity is a band. Above the "
          "single-name ceiling the binding liquidity is the underlying (QQQ/futures), and the "
          "single-name √-law underestimates cost.")
    print(f"\nOutputs written to {OUT}")


if __name__ == "__main__":
    main()
