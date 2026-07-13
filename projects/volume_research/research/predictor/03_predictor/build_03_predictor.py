"""P03 — The predictor: worked examples of predict_slippage() across size x regime x order-type.

Answers the client's question directly: "I set an order at X and filled at X + diff — predict
diff." This stage doesn't measure anything new — it packages P01 (market state) and P02 (chase
cost) behind `slippage/predict.py::predict_slippage()` and demonstrates it on a representative
grid: $100k / $1M / $10M x calm/normal/stress x cross/limit_chase, for TQQQ.

Run:
    /Users/franciscosimao/opt/anaconda3/envs/quant/bin/python \
        research/predictor/03_predictor/build_03_predictor.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from slippage.predict import predict_slippage  # noqa: E402
from slippage.state import MarketState  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"

SIZES_USD = [100_000, 1_000_000, 10_000_000]
REGIME_SIGMA_BPS = {"calm": 200.0, "normal": 300.0, "stress": 450.0}   # ~S03/P01 tercile-ish levels
SYMBOL, PRICE = "TQQQ", 77.0
SPREAD_BPS, INTERVAL_VOLUME_SHARES = 0.74, 3_000_000.0   # representative 10:00-bin values (P01)


def make_state(regime: str) -> MarketState:
    return MarketState(
        ts=pd.Timestamp("2026-01-05 10:00:00"), symbol=SYMBOL, bin_label="10:00",
        expected_interval_volume=INTERVAL_VOLUME_SHARES,
        thin_volume_p10=INTERVAL_VOLUME_SHARES * 0.4, thin_volume_p20=INTERVAL_VOLUME_SHARES * 0.6,
        sigma_now_bps=REGIME_SIGMA_BPS[regime], regime=regime, spread_bps=SPREAD_BPS,
    )


# --------------------------------------------------------------------------- plot
DARK_BG, PANEL_BG, TEXT_COL, GRID_COL = "#0d1117", "#161b22", "#c9d1d9", "#30363d"
STYLE_COLORS = {"cross": "#3fb950", "limit_chase": "#f0b429"}


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


def _usd(x):
    return f"${x/1e6:.0f}M" if x >= 1e6 else f"${x/1e3:.0f}k"


def make_plot(res_dir: Path):
    """Regenerate predictor_worked_examples.png from predictor_worked_examples.csv."""
    df = pd.read_csv(res_dir / "predictor_worked_examples.csv")
    regimes = ("calm", "normal", "stress")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    fig.patch.set_facecolor(DARK_BG)
    for ax, regime in zip(axes, regimes):
        sub = df[df.regime == regime]
        sizes = sorted(sub["size_usd"].unique())
        x = np.arange(len(sizes))
        for j, ot in enumerate(("cross", "limit_chase")):
            s = sub[sub.order_type == ot].set_index("size_usd").loc[sizes]
            xs = x + (j - 0.5) * 0.36
            ax.bar(xs, s["mean_bps"], width=0.34, color=STYLE_COLORS[ot],
                   label=ot if regime == "calm" else None)
            # whisker: the p90–p95 timing tail above the mean (the risk the mean hides)
            ax.vlines(xs, s["mean_bps"], s["p95_bps"], color=STYLE_COLORS[ot], lw=1.2, alpha=0.8)
            ax.scatter(xs, s["p95_bps"], color=STYLE_COLORS[ot], marker="_", s=90)
            ax.scatter(xs, s["p90_bps"], color=STYLE_COLORS[ot], marker="_", s=50, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([_usd(v) for v in sizes])
        ax.set_title(f"{regime} (σ_now={REGIME_SIGMA_BPS[regime]:.0f} bps)")
        _style_ax(ax)
    axes[0].set_ylabel("slippage (bps): bar = mean, ticks = p90 / p95")
    fig.suptitle("P03 — predict_slippage(): mean vs timing tail, by size / regime / order type "
                 "(TQQQ, 15-min latency)", color=TEXT_COL, fontsize=11)
    fig.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, fontsize=8, labelcolor=TEXT_COL,
               loc="upper right")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(res_dir / "predictor_worked_examples.png", dpi=120, facecolor=DARK_BG)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for regime in ("calm", "normal", "stress"):
        state = make_state(regime)
        for size in SIZES_USD:
            for order_type in ("cross", "limit_chase"):
                r = predict_slippage(size, "buy", order_type, state, price=PRICE, latency_min=15.0)
                lo, mid, hi = r["components"]["impact_band_bps"]
                rows.append({
                    "size_usd": size, "regime": regime, "order_type": order_type,
                    "participation": r["participation"],
                    "spread_bps": r["components"]["spread_bps"],
                    "impact_lo_bps": lo, "impact_mid_bps": mid, "impact_hi_bps": hi,
                    "drag_bps": r["components"]["drag_bps"],
                    "timing_sigma_bps": r["components"]["timing_sigma_bps"],
                    "mean_bps": r["mean_bps"], "p50_bps": r["p50_bps"],
                    "p90_bps": r["p90_bps"], "p95_bps": r["p95_bps"],
                })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "predictor_worked_examples.csv", index=False)
    make_plot(OUT)

    pd.set_option("display.float_format", lambda x: f"{x:.2f}")
    for regime in ("calm", "normal", "stress"):
        print(f"\n=== {SYMBOL}, regime={regime} (sigma_now={REGIME_SIGMA_BPS[regime]:.0f} bps) ===")
        sub = df[df.regime == regime][["size_usd", "order_type", "mean_bps", "p50_bps",
                                       "p90_bps", "p95_bps"]]
        print(sub.to_string(index=False))
    print(f"\nOutputs written to {OUT}")


if __name__ == "__main__":
    main()
