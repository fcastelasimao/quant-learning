"""C01 — Almgren temporary + two-model envelope: before/after capacity, sigma held identical.

The useful core of the paused Almgren plan (`plans/2026-07_almgren_adoption.md`), post-S12
(permanent term dropped — wrong mechanism for an arbitrage-pinned, elastic-supply ETF; see
`slippage/impact.py`'s module note). Compares the library's two temporary-impact models —
the adopted sqrt-law (Y-band) and Almgren et al. (2005)'s fitted temporary term — with **sigma
held identical** across both, so any movement in the capacity number is the coefficient/exponent
choice alone, nothing else (not a sigma re-baseline, not a data change).

**The library default stays "sqrt". This stage does not flip it — that is an owner/GATE
decision**, per plans/2026-07_execution_track.md's explicit instruction.

Run:
    /Users/franciscosimao/opt/anaconda3/envs/quant/bin/python \
        research/capacity/01_almgren_envelope/build_01_almgren_envelope.py
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
from slippage import CostModel, MarketParams  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"

# Identical to research/06_capacity_curve/build_06_capacity_curve.py's MARKET dict — the whole
# point of this stage is that sigma/ADV/spread do NOT change; only impact_model does.
MARKET = {
    "TQQQ": dict(sigma=370.0, adv=4.9e9, half_spread=0.74),
    "SQQQ": dict(sigma=372.0, adv=2.8e9, half_spread=1.00),
}
DECISION_CADENCE_MIN = 15
NOTIONAL_GRID = [1e5, 1e6, 5e6, 1e7, 3e7, 1e8]
BUDGET_BPS = 25.0   # round-trip cost budget, matching S05/S06's convention


def params_for(symbol) -> MarketParams:
    m = MARKET[symbol]
    return MarketParams(sigma_daily_bps=m["sigma"], adv_usd=m["adv"], half_spread_bps=m["half_spread"])


def cost_table(symbol) -> pd.DataFrame:
    p = params_for(symbol)
    rows = []
    for notional in NOTIONAL_GRID:
        row = {"notional": notional}
        rt_lo = CostModel(p).roundtrip(notional, horizon_min=DECISION_CADENCE_MIN, Y=0.3)
        rt_mid = CostModel(p).roundtrip(notional, horizon_min=DECISION_CADENCE_MIN, Y=0.5)
        rt_hi = CostModel(p).roundtrip(notional, horizon_min=DECISION_CADENCE_MIN, Y=1.0)
        rt_alm = CostModel(p).roundtrip(notional, horizon_min=DECISION_CADENCE_MIN,
                                        impact_model="almgren")
        row.update({
            "sqrt_Y03_bps": rt_lo.expected_slippage_bps, "sqrt_Y05_bps": rt_mid.expected_slippage_bps,
            "sqrt_Y10_bps": rt_hi.expected_slippage_bps, "almgren_bps": rt_alm.expected_slippage_bps,
            "envelope_lo_bps": min(rt_lo.expected_slippage_bps, rt_hi.expected_slippage_bps,
                                   rt_alm.expected_slippage_bps),
            "envelope_hi_bps": max(rt_lo.expected_slippage_bps, rt_hi.expected_slippage_bps,
                                   rt_alm.expected_slippage_bps),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def capacity_numeric(p: MarketParams, budget_bps: float, *, impact_model: str,
                     horizon_min: float = DECISION_CADENCE_MIN, Y: float = 0.5) -> float:
    """Binary search for the largest notional whose round-trip expected cost stays <= budget_bps.
    General (works for any impact_model) — mirrors cost.py's closed-form capacity_at_horizon,
    which is sqrt-only."""
    lo, hi = 1e3, 1e11
    for _ in range(60):
        mid = (lo + hi) / 2
        rt = CostModel(p).roundtrip(mid, horizon_min=horizon_min, Y=Y, impact_model=impact_model)
        if rt.expected_slippage_bps <= budget_bps:
            lo = mid
        else:
            hi = mid
    return lo


def capacity_table(symbol) -> dict:
    p = params_for(symbol)
    return {
        "sqrt_Y03": capacity_numeric(p, BUDGET_BPS, impact_model="sqrt", Y=0.3),
        "sqrt_Y05": capacity_numeric(p, BUDGET_BPS, impact_model="sqrt", Y=0.5),
        "sqrt_Y10": capacity_numeric(p, BUDGET_BPS, impact_model="sqrt", Y=1.0),
        "almgren": capacity_numeric(p, BUDGET_BPS, impact_model="almgren"),
    }


def _usd(x):
    if x >= 1e9:
        return f"${x/1e9:.2f}B"
    if x >= 1e6:
        return f"${x/1e6:.1f}M"
    return f"${x/1e3:.0f}k"


# --------------------------------------------------------------------------- plot
DARK_BG, PANEL_BG, TEXT_COL, GRID_COL = "#0d1117", "#161b22", "#c9d1d9", "#30363d"
MODEL_COLORS = {"sqrt_Y03": "#79c0ff", "sqrt_Y05": "#58a6ff", "sqrt_Y10": "#1f6feb",
                "almgren": "#f0b429"}


def _style_ax(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COL, labelsize=8, which="both")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID_COL)
    ax.grid(axis="y", color=GRID_COL, alpha=0.5, linewidth=0.7)
    for lab in (ax.yaxis.label, ax.xaxis.label, ax.title):
        lab.set_color(TEXT_COL)


def make_plot(res_dir: Path):
    """Regenerate almgren_envelope.png from the cost_table/capacity_table CSVs."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    fig.patch.set_facecolor(DARK_BG)

    for ax, sym in zip(axes[:2], MARKET):
        ct = pd.read_csv(res_dir / f"cost_table_{sym}.csv")
        x = ct["notional"]
        ax.fill_between(x, ct["sqrt_Y03_bps"], ct["sqrt_Y10_bps"], color=MODEL_COLORS["sqrt_Y05"],
                        alpha=0.18, label="sqrt Y∈[0.3, 1.0] band")
        ax.plot(x, ct["sqrt_Y05_bps"], color=MODEL_COLORS["sqrt_Y05"], lw=1.6,
                label="sqrt Y=0.5 (library default)")
        ax.plot(x, ct["almgren_bps"], color=MODEL_COLORS["almgren"], lw=1.6, marker="o", ms=3.5,
                label="Almgren temporary (2005 fit)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("order notional ($, 15-min fill)")
        ax.set_ylabel("round-trip expected slippage (bps)")
        ax.set_title(f"{sym} — Almgren sits below the whole sqrt band")
        _style_ax(ax)
        if sym == "TQQQ":
            ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, fontsize=7.5, labelcolor=TEXT_COL)

    ax3 = axes[2]
    models = ["sqrt_Y10", "sqrt_Y05", "sqrt_Y03", "almgren"]
    x3 = np.arange(len(models))
    for k, sym in enumerate(MARKET):
        cap = pd.read_csv(res_dir / f"capacity_table_{sym}.csv").set_index("model")["capacity_usd"]
        col = "#58a6ff" if sym == "TQQQ" else "#f0b429"
        ax3.bar(x3 + (k - 0.5) * 0.36, cap.loc[models], width=0.34, color=col, label=sym)
        for xi, v in zip(x3 + (k - 0.5) * 0.36, cap.loc[models]):
            ax3.text(xi, v, _usd(v), ha="center", va="bottom", fontsize=6.5, color=TEXT_COL)
    ax3.set_yscale("log")
    ax3.set_xticks(x3)
    ax3.set_xticklabels(["sqrt\nY=1.0", "sqrt\nY=0.5", "sqrt\nY=0.3", "Almgren"], fontsize=8)
    ax3.set_ylabel(f"capacity at {BUDGET_BPS:.0f} bps round-trip ($)")
    ax3.set_title("capacity by model (~7.4× sqrt-vs-Almgren gap)")
    _style_ax(ax3)
    ax3.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, fontsize=8, labelcolor=TEXT_COL)

    fig.suptitle("C01 — sqrt-law vs Almgren temporary, σ/ADV/spread held identical (default stays "
                 "sqrt; flip = owner GATE)", color=TEXT_COL, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(res_dir / "almgren_envelope.png", dpi=120, facecolor=DARK_BG)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for sym in MARKET:
        print(f"\n=== {sym} — round-trip cost (bps) by impact model, sigma/ADV/spread FIXED ===")
        ct = cost_table(sym)
        ct.to_csv(OUT / f"cost_table_{sym}.csv", index=False)
        disp = ct.copy()
        disp["notional"] = disp["notional"].map(_usd)
        pd.set_option("display.float_format", lambda x: f"{x:.1f}")
        print(disp.to_string(index=False))

        print(f"\n=== {sym} — capacity (round-trip <= {BUDGET_BPS:.0f} bps) by impact model ===")
        cap = capacity_table(sym)
        for k, v in cap.items():
            print(f"  {k:12}: {_usd(v)}")
        pd.DataFrame([{"model": k, "capacity_usd": v} for k, v in cap.items()]).to_csv(
            OUT / f"capacity_table_{sym}.csv", index=False)

        # Reconciliation: where does almgren sit relative to the sqrt Y=0.3 (optimistic) edge?
        ratio = cap["almgren"] / cap["sqrt_Y03"]
        print(f"  reconciliation: almgren capacity / sqrt(Y=0.3) capacity = {ratio:.2f}")

    make_plot(OUT)
    print(f"\nOutputs written to {OUT}")


if __name__ == "__main__":
    main()
