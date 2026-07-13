"""E03 — The scheduler: worked schedules vs a naive 15-min single fill.

Demonstrates `slippage/schedule.py::schedule_order()` on a representative grid: TQQQ,
$1M / $5M / $20M, normal vs stress regime. For each cell: the chosen horizon h*, the slice plan,
the expected-cost band, and — the headline comparison — what a naive single fill pinned to the
strategy's 15-min decision cadence would have cost instead.

`edge_bps` (the per-trade edge that anchors the alpha-forfeiture term) is a strategy input this
library does not measure — an illustrative 50 bps is used throughout, documented as such; the
owner should supply the strategy's actual measured edge.

Run:
    /Users/franciscosimao/opt/anaconda3/envs/quant/bin/python \
        research/scheduler/03_scheduler/build_03_scheduler.py
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
from slippage.schedule import schedule_order, alpha_interruption_bps  # noqa: E402
from slippage.state import MarketState  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"

PRICE = 77.0
EDGE_BPS = 50.0   # illustrative — the owner should supply the strategy's actual measured edge
NOTIONALS = [1e6, 5e6, 20e6]
# Representative TQQQ params (matches S06/C01's MARKET dict): sigma normal=370bps, stress=510bps;
# ADV $4.9B normal, $2.9B thin/stress. Interval volume = ADV_shares / 26 bins/day (a "typical
# bin" proxy -- the real per-bin curve is P01's job; this stage doesn't load a VolumeProfile).
REGIMES = {
    "normal": dict(sigma=370.0, adv_usd=4.9e9),
    "stress": dict(sigma=510.0, adv_usd=2.9e9),
}
DECISION_CADENCE_MIN = 15.0


def make_state(regime: str) -> MarketState:
    r = REGIMES[regime]
    interval_volume_shares = (r["adv_usd"] / PRICE) / 26.0
    return MarketState(
        ts=pd.Timestamp("2026-01-05 10:00:00"), symbol="TQQQ", bin_label="10:00",
        expected_interval_volume=interval_volume_shares,
        thin_volume_p10=interval_volume_shares * 0.4, thin_volume_p20=interval_volume_shares * 0.6,
        sigma_now_bps=r["sigma"], regime=regime, spread_bps=0.74,
    )


def _usd(x):
    if x >= 1e9:
        return f"${x/1e9:.2f}B"
    if x >= 1e6:
        return f"${x/1e6:.1f}M"
    return f"${x/1e3:.0f}k"


# --------------------------------------------------------------------------- plot
DARK_BG, PANEL_BG, TEXT_COL, GRID_COL = "#0d1117", "#161b22", "#c9d1d9", "#30363d"
COMP_COLORS = {"exec": "#58a6ff", "alpha": "#f0b429", "interrupt": "#f85149"}


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
    """Regenerate scheduler_worked_examples.png from the summary_{regime}_{notional}.csv files."""
    files = sorted(res_dir.glob("summary_*.csv"))
    if not files:
        return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.sort_values(["regime", "notional"]).reset_index(drop=True)
    regimes = list(df["regime"].unique())
    fig, axes = plt.subplots(1, len(regimes), figsize=(7.5 * len(regimes), 5), sharey=True)
    fig.patch.set_facecolor(DARK_BG)
    axes = np.atleast_1d(axes)
    for ax, regime in zip(axes, regimes):
        sub = df[df.regime == regime].sort_values("notional").reset_index(drop=True)
        x = np.arange(len(sub))
        for j, kind in enumerate(("scheduled", "naive")):
            xs = x + (j - 0.5) * 0.38
            bottom = np.zeros(len(sub))
            for comp in ("exec", "alpha", "interrupt"):
                vals = sub[f"{kind}_{comp}_bps"].values
                ax.bar(xs, vals, bottom=bottom, width=0.36, color=COMP_COLORS[comp],
                       alpha=1.0 if kind == "scheduled" else 0.45,
                       label=comp if (regime == regimes[0] and kind == "scheduled") else None)
                bottom += vals
            for xi, tot in zip(xs, bottom):
                ax.text(xi, tot, f"{tot:.0f}", ha="center", va="bottom", fontsize=7, color=TEXT_COL)
        for xi, r in zip(x, sub.itertuples()):
            ax.text(xi, -0.06, f"h*={r.horizon_min:.0f}m\n{'+' if r.total_improvement_bps >= 0 else ''}"
                    f"{r.total_improvement_bps:.1f} bps", ha="center", va="top", fontsize=7,
                    color=TEXT_COL, transform=ax.get_xaxis_transform())
        ax.set_xticks(x)
        ax.set_xticklabels([_usd(v) for v in sub["notional"]])
        ax.tick_params(axis="x", pad=32)
        ax.set_title(f"{regime} — solid = scheduled (h*), faded = naive 15-min fill")
        _style_ax(ax)
    axes[0].set_ylabel("total objective (bps): exec + alpha forfeit + interruption")
    fig.suptitle("E03 — schedule_order() vs the naive 15-min single fill (TQQQ, edge=50 bps "
                 "illustrative)", color=TEXT_COL, fontsize=11)
    fig.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, fontsize=8, labelcolor=TEXT_COL,
               loc="upper right")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(res_dir / "scheduler_worked_examples.png", dpi=120, facecolor=DARK_BG)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for regime in ("normal", "stress"):
        state = make_state(regime)
        print(f"\n{'='*78}\nTQQQ, regime={regime} (sigma_now={REGIMES[regime]['sigma']:.0f} bps, "
              f"interval volume={state.expected_interval_volume/1e6:.2f}M sh)")
        for notional in NOTIONALS:
            sched = schedule_order(notional, "buy", state, price=PRICE, edge_bps=EDGE_BPS)
            naive = predict_slippage(notional, "buy", "cross", state, price=PRICE,
                                     latency_min=DECISION_CADENCE_MIN)
            # Same objective decomposition as the scheduler (shared helper) at the 15-min pin, so
            # the comparison is apples-to-apples and scheduled_total <= naive_total by construction.
            naive_alpha, naive_interrupt = alpha_interruption_bps(
                DECISION_CADENCE_MIN, state, EDGE_BPS, "cancel")
            naive_total = naive["mean_bps"] + naive_alpha + naive_interrupt
            sched_total = (sched.expected_slippage_bps + sched.alpha_forfeit_bps +
                          sched.interruption_summary["expected_cost_bps"])

            print(f"\n  {_usd(notional)}: h*={sched.horizon_min:.0f} min, "
                  f"{len(sched.slices)} slice(s), feasible={sched.feasible}")
            print(f"    scheduled : exec {sched.expected_slippage_bps:6.2f} + alpha "
                  f"{sched.alpha_forfeit_bps:5.2f} + interrupt "
                  f"{sched.interruption_summary['expected_cost_bps']:4.2f} = "
                  f"{sched_total:6.2f} bps total (exec band "
                  f"{sched.expected_slippage_band_bps[0]:.1f}-{sched.expected_slippage_band_bps[1]:.1f})")
            print(f"    naive 15m : exec {naive['mean_bps']:6.2f} + alpha {naive_alpha:5.2f} "
                  f"+ interrupt {naive_interrupt:4.2f} = {naive_total:6.2f} bps total")
            print(f"    TOTAL improvement from scheduling: {naive_total - sched_total:+.2f} bps "
                  f"(the objective h* minimizes over — must be >= 0 by construction, h=15 is "
                  f"itself in the search grid)")

            for sl in sched.slices:
                rows.append({"regime": regime, "notional": notional, "time_offset_min": sl.time_offset_min,
                            "child_notional_usd": sl.child_notional_usd, "order_style": sl.order_style})
            rows_summary = {"regime": regime, "notional": notional, "horizon_min": sched.horizon_min,
                           "n_slices": len(sched.slices), "feasible": sched.feasible,
                           "scheduled_exec_bps": sched.expected_slippage_bps,
                           "scheduled_alpha_bps": sched.alpha_forfeit_bps,
                           "scheduled_interrupt_bps": sched.interruption_summary["expected_cost_bps"],
                           "scheduled_total_bps": sched_total,
                           "naive_exec_bps": naive["mean_bps"], "naive_alpha_bps": naive_alpha,
                           "naive_interrupt_bps": naive_interrupt, "naive_total_bps": naive_total,
                           "total_improvement_bps": naive_total - sched_total}
            pd.DataFrame([rows_summary]).to_csv(
                OUT / f"summary_{regime}_{int(notional)}.csv", index=False)

    pd.DataFrame(rows).to_csv(OUT / "all_slices.csv", index=False)
    make_plot(OUT)
    print(f"\nOutputs written to {OUT}")


if __name__ == "__main__":
    main()
