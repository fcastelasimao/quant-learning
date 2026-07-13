"""19_sqqq_target_exploration: grid search over severity targets × sizing functions for SQQQ.

Depends on:
  - research/06_context_enrichment/enriched_trades_SQQQ.csv

Runs a 3 × 5 grid (targets × sizing functions) using the enriched feature set
from item 06 with walk-forward L1-logit (same setup as item 17). Produces
the full results table and a scatter plot comparing Sharpe vs MaxDD.

Targets:  pnl_pct <= {-1.0, -1.5, -2.0}
Sizing:   {linear_skip, sqrt_skip, aggressive_2x, moderate_1p5x, step_skip_at_50}
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-quant")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _walkforward import equity_metrics, extended_sizing_functions, fit_predict_walkforward_logit  # noqa: E402

ROOT = Path(__file__).resolve().parent
ITEM_06 = ROOT.parent / "06_context_enrichment"
OUT = ROOT

CURATED_NUMERIC = [
    "atr_pct", "RSI_entry", "BBP_entry",
    "dist_to_MA20", "dist_to_MA50", "dist_to_MA100",
    "MA20_D5", "MA50_D5", "MA100_D1",
    "log_volume_ratio", "bars_since_last_stop", "hour_of_entry",
]
DAILY_CONTEXT = [
    "QQQ_RSI_14", "QQQ_dist_MA20", "QQQ_dist_MA50", "QQQ_dist_MA200",
    "QQQ_realized_vol_20d", "QQQ_dist_high_20d", "QQQ_50d_return",
    "QQQ_50d_return_pctile_252", "QQQ_drawdown_5d", "QQQ_drawdown_60d",
    "QQQ_gap_overnight",
    "SPY_RSI_14", "SPY_dist_MA50",
    "VIX_level", "VIX_5d_change", "VIX_pctile_252d", "VIX_term_structure",
    "HYG_LQD_ratio", "HYG_5d_change",
    "yield_curve_slope", "TNX_5d_change",
]
REGIME_DUMMIES = ["regime_chop_highvol", "regime_sideways_lowvol"]
FEATURE_COLS = CURATED_NUMERIC + REGIME_DUMMIES + DAILY_CONTEXT

TARGETS = [
    ("is_severe_loss", "1pct"),
    ("is_severe_loss_1p5pct", "1p5pct"),
    ("is_severe_loss_2pct", "2pct"),
]

SIZING_LABELS = [
    "linear_skip",
    "sqrt_skip",
    "aggressive_2x",
    "moderate_1p5x",
    "step_skip_at_50",
]


def load_sqqq() -> pd.DataFrame:
    df = pd.read_csv(ITEM_06 / "enriched_trades_SQQQ.csv",
                     parse_dates=["entry_time", "exit_time"])
    for c in CURATED_NUMERIC + DAILY_CONTEXT:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["year"] = df["entry_time"].dt.year
    for r in ("chop_highvol", "sideways_lowvol"):
        df[f"regime_{r}"] = (df["regime_entry"] == r).astype(int)
    df["is_severe_loss"]        = (df["pnl_pct"] <= -1.0).astype(int)
    df["is_severe_loss_1p5pct"] = (df["pnl_pct"] <= -1.5).astype(int)
    df["is_severe_loss_2pct"]   = (df["pnl_pct"] <= -2.0).astype(int)
    return df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_sqqq()

    # Compute walk-forward p_severe for each target
    for target_col, label in TARGETS:
        df[f"p_{label}"] = fit_predict_walkforward_logit(df, FEATURE_COLS, target_col)

    # Restrict to rows with predictions on all targets
    p_cols = [f"p_{label}" for _, label in TARGETS]
    df_eval = df.dropna(subset=p_cols).copy()
    df_eval["r"] = df_eval["pnl_pct"] / 100.0

    rows = []
    # Baseline (always full size)
    m = equity_metrics(df_eval, "r")
    m.update({"target": "baseline", "sizing": "baseline_full", "mean_size": 1.0})
    rows.append(m)

    for _, label in TARGETS:
        p = df_eval[f"p_{label}"].clip(0.0, 1.0)
        sf = extended_sizing_functions(p)
        for sz_name in SIZING_LABELS:
            if sz_name not in sf:
                continue
            size = sf[sz_name]
            r_col = f"r_{label}_{sz_name}"
            df_eval[r_col] = df_eval["r"] * size
            m = equity_metrics(df_eval, r_col)
            m.update({"target": label, "sizing": sz_name, "mean_size": float(size.mean())})
            rows.append(m)

    grid = pd.DataFrame(rows)
    # Rename annualized_return → cagr for consistency
    if "annualized_return" in grid.columns:
        grid = grid.rename(columns={"annualized_return": "cagr"})
    col_order = ["target", "sizing", "mean_size", "total_return", "cagr",
                 "sharpe_daily", "max_drawdown", "calmar",
                 "mean_daily_return", "std_daily_return"]
    col_order = [c for c in col_order if c in grid.columns]
    grid = grid[col_order]
    grid.to_csv(OUT / "sqqq_target_sizing_grid.csv", index=False)
    print(grid.to_string(index=False))

    # Scatter: Sharpe vs MaxDD coloured by target
    target_colors = {"1pct": "#2563eb", "1p5pct": "#f59e0b", "2pct": "#dc2626", "baseline": "#374151"}
    sizing_markers = {
        "baseline_full": "D",
        "linear_skip": "o",
        "sqrt_skip": "s",
        "aggressive_2x": "^",
        "moderate_1p5x": "v",
        "step_skip_at_50": "x",
    }

    fig, ax = plt.subplots(figsize=(10, 7))
    for _, row in grid.iterrows():
        color = target_colors.get(row["target"], "black")
        marker = sizing_markers.get(row["sizing"], "o")
        ax.scatter(-row["max_drawdown"] * 100, row["sharpe_daily"],
                   color=color, marker=marker, s=100, edgecolor="black", linewidth=0.5,
                   zorder=3)
        lbl = f"{row['target']} {row['sizing'].split('_')[0]}"
        ax.annotate(lbl, (-row["max_drawdown"] * 100, row["sharpe_daily"]),
                    xytext=(5, 2), textcoords="offset points", fontsize=7)
    # Legend for targets only
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=t) for t, c in target_colors.items()]
    ax.legend(handles=handles, fontsize=8, title="target", loc="lower right")
    ax.axvline(18.1, color="grey", linewidth=0.6, linestyle="--", label="baseline MaxDD")
    ax.set_xlabel("Max drawdown (%)  ← smaller is better")
    ax.set_ylabel("Sharpe (daily, annualized)")
    ax.set_title("Item 19: SQQQ sizing grid — Sharpe vs Max DD\n(top-left = ideal)")
    fig.tight_layout()
    fig.savefig(OUT / "sqqq_target_exploration.png", dpi=140)
    plt.close(fig)
    print(f"\nOutputs written to {OUT}")


if __name__ == "__main__":
    main()
