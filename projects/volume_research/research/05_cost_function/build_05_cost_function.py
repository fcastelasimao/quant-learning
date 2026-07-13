"""Block 5 / Stage 5 — the size-aware cost function, assembled.

Composes Blocks 1+3+4 into cost(Q, urgency) and shows the Almgren–Chriss trade-off:
faster fills cost more impact, slower fills cost more timing risk, linked by
t = (Q/ADV)/participation·day. The headline correction: Block 4's capacity assumed
leisurely day-long execution; our strategy must fill within ~15 min, which forces fast
trading and **cuts the real per-trade capacity well below the day-execution number**.

Outputs: the impact/timing trade-off curve, a capacity-vs-execution-horizon table, a
dark-theme PNG, and a console summary contrasting day vs 15-min execution.

Run:
    /Users/franciscosimao/opt/anaconda3/envs/quant/bin/python \
        research/05_cost_function/build_05_cost_function.py
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
from slippage.cost import (  # noqa: E402
    MarketParams, expected_slippage_bps, optimal_participation, capacity_at_horizon,
)

from quantcore import config  # noqa: E402

# Half-spreads (one-way, bps) from Block 1 / findings_01 (15-min Corwin–Schultz means).
HALF_SPREAD_BPS = {"TQQQ": 0.74, "SQQQ": 1.00, "QQQ": 0.72}
TICKERS = ["TQQQ", "SQQQ", "QQQ"]
HORIZONS_MIN = [1, 5, 15, 60, 390]      # execution windows (390 = full day)
BUDGET_BPS = 25.0                        # expected-cost (spread+impact) budget
WINDOW = 504
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


def market_params(sym: str) -> MarketParams:
    db = config.data_dir() / f"DB_{sym}_historical_data.db"
    with sqlite3.connect(db) as c:
        d = pd.read_sql("SELECT et_datetime, close, volume FROM candles_1d ORDER BY ts", c)
    d["dt"] = pd.to_datetime(d["et_datetime"])
    d = d.set_index("dt").tail(WINDOW)
    # σ normal = window MEAN of rolling vol — the expected-cost headline moment (findings_09).
    # (The median/typical-day sensitivity is carried explicitly in the Stage-4/6 capacity tables.)
    sigma = (d["close"].pct_change().rolling(20).std() * 1e4).mean()
    adv = (d["volume"] * d["close"]).mean()
    return MarketParams(sigma_daily_bps=sigma, adv_usd=adv, half_spread_bps=HALF_SPREAD_BPS[sym])


def make_plot(params, captab, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.patch.set_facecolor(DARK_BG)

    # Panel 1: the trade-off for a representative TQQQ $10M order.
    p = params["TQQQ"]
    Q = 10e6
    part = np.linspace(0.01, 0.5, 200)
    c = expected_slippage_bps(Q, part, p)
    ax1.plot(part * 100, c["impact_bps"], color="#f78166", lw=1.5, label="impact (drag)")
    ax1.plot(part * 100, c["timing_risk_bps"], color="#bc8cff", lw=1.5, label="timing (1σ risk)")
    ax1.plot(part * 100, c["risk_adjusted_bps"], color="#58a6ff", lw=2.0, label="spread+impact+timing")
    pstar = optimal_participation(Q, p) * 100
    ax1.axvline(pstar, color="#3fb950", ls=":", lw=1.0, label=f"optimum ≈ {pstar:.0f}% POV")
    ax1.set_title("TQQQ $10M: impact vs timing trade-off")
    ax1.set_xlabel("participation rate (% of volume)")
    ax1.set_ylabel("cost (bps)")
    _style_ax(ax1)
    ax1.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, fontsize=7, labelcolor=TEXT_COL)

    # Panel 2: capacity ($) vs execution horizon, per ticker.
    for t in TICKERS:
        sub = captab[captab.ticker == t]
        ax2.plot(sub.exec_min, sub.capacity_usd, marker="o", ms=4, color=TICKER_COLORS[t], label=t)
    ax2.axvline(15, color="#bc8cff", ls=":", lw=1.0, label="15-min cadence")
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_title(f"Capacity vs fill horizon (≤{BUDGET_BPS:.0f} bps expected cost)")
    ax2.set_xlabel("execution window (min)")
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
        return f"${x/1e6:.1f}M"
    return f"${x/1e3:.0f}k"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    params = {t: market_params(t) for t in TICKERS}

    rows = []
    for t in TICKERS:
        for T in HORIZONS_MIN:
            rows.append({"ticker": t, "exec_min": T,
                         "capacity_usd": capacity_at_horizon(BUDGET_BPS, params[t], T)})
    captab = pd.DataFrame(rows)
    captab.to_csv(OUT / "capacity_by_horizon.csv", index=False)
    make_plot(params, captab, OUT / "cost_function.png")

    pd.set_option("display.float_format", lambda x: f"{x:.2f}")
    print(f"=== Capacity ($) at ≤{BUDGET_BPS:.0f} bps expected cost, by fill horizon ===")
    piv = captab.pivot(index="ticker", columns="exec_min", values="capacity_usd").loc[TICKERS]
    print(piv.map(_usd).to_string())

    print("\n=== The correction: same $14M TQQQ order, day vs 15-min execution ===")
    p = params["TQQQ"]
    for T in (390, 15):
        part = (14e6 / p.adv_usd) / (T / p.minutes_per_day)
        c = expected_slippage_bps(14e6, part, p)
        print(f"  fill in {T:4} min (POV {part*100:4.1f}%): impact {c['impact_bps']:5.1f} bps, "
              f"timing {c['timing_risk_bps']:5.1f} bps (1σ)")

    print("\n=== Optimal execution for representative sizes (TQQQ) ===")
    for Q in (1e6, 5e6, 10e6):
        ps = optimal_participation(Q, p)
        c = expected_slippage_bps(Q, ps, p)
        print(f"  {_usd(Q):>7}: POV {ps*100:4.1f}%, fill {c['fill_minutes']:4.1f} min, "
              f"impact {c['impact_bps']:4.1f} + timing {c['timing_risk_bps']:4.1f} bps")
    print(f"\nOutputs written to {OUT}")


if __name__ == "__main__":
    main()
