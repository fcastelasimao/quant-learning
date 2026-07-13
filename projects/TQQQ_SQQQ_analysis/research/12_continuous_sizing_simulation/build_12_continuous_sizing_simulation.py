"""12_continuous_sizing_simulation: scale position by (1 - p_severe).

Instead of a binary skip rule, use the L1-logistic OOS-predicted P(severe loss)
as a continuous *down-scale* of position size. A trade in a high-risk region
is taken smaller; a trade in a low-risk region is taken at full size.

Slippage caveat: the canonical pnl_pct already reflects realized slippage at
the actual fill size. Linear scaling under reduced size is mildly conservative
(smaller orders -> less market impact in reality).

Sizing functions compared (all bounded in [0, 1]):
  baseline_full     = 1                                  (current behavior)
  linear_skip       = 1 - p_severe                        (graceful skip)
  sqrt_skip         = (1 - p_severe) ** 0.5               (mild de-risking)
  step_skip_at_50   = 1 if p_severe < 0.5 else 0          (binary, for compare)

Targets compared:
  is_severe_loss          pnl_pct <= -1.0 %  (baseline candidate)
  is_severe_loss_1p5pct   pnl_pct <= -1.5 %
  is_severe_loss_2pct     pnl_pct <= -2.0 %

We re-fit the L1-logistic per WF window so the simulation is causal (no
look-ahead from later data).
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
from _walkforward import equity_metrics, fit_predict_walkforward_logit, sizing_functions


ROOT = Path(__file__).resolve().parent
CANON = ROOT.parent.parent / "full_history_canonical"
OUT = ROOT

CURATED_NUMERIC = [
    "atr_pct", "RSI_entry", "BBP_entry",
    "dist_to_MA20", "dist_to_MA50", "dist_to_MA100",
    "MA20_D5", "MA50_D5", "MA100_D1",
    "log_volume_ratio", "bars_since_last_stop", "hour_of_entry",
]

TARGETS = [
    ("is_severe_loss",       "1pct"),
    ("is_severe_loss_1p5pct","1p5pct"),
    ("is_severe_loss_2pct",  "2pct"),
]


def load(sym: str) -> pd.DataFrame:
    df = pd.read_csv(CANON / f"TRADES_{sym}_full_history.csv",
                     parse_dates=["entry_time", "exit_time"])
    df = df[df["regime_entry"].notna()].copy()
    df["year"] = df["entry_time"].dt.year
    for c in CURATED_NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for r in ("chop_highvol", "sideways_lowvol"):
        df[f"regime_{r}"] = (df["regime_entry"] == r).astype(int)
    df["is_severe_loss_1p5pct"] = (df["pnl_pct"] <= -1.5).astype(int)
    df["is_severe_loss_2pct"]   = (df["pnl_pct"] <= -2.0).astype(int)
    return df.dropna(subset=CURATED_NUMERIC).copy()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    feature_cols = CURATED_NUMERIC + ["regime_chop_highvol", "regime_sideways_lowvol"]
    summary_rows = []
    equity_panels: dict = {}

    for sym in ("TQQQ", "SQQQ"):
        df = load(sym)

        for target_col, target_label in TARGETS:
            p = fit_predict_walkforward_logit(df, feature_cols, target_col)
            df_eval = df.copy()
            df_eval[f"p_{target_label}"] = p
            df_eval = df_eval.dropna(subset=[f"p_{target_label}"]).copy()
            df_eval["r"] = df_eval["pnl_pct"] / 100.0

            sizings = sizing_functions(df_eval[f"p_{target_label}"].values)
            for name, size in sizings.items():
                sized_col = f"r_{name}"
                df_eval[sized_col] = df_eval["r"] * size
                m = equity_metrics(df_eval, sized_col)
                m["symbol"] = sym
                m["target"] = target_label
                m["sizing"] = name
                m["n_trades"] = len(df_eval)
                m["mean_size"] = float(size.mean())
                summary_rows.append(m)

            # Cache for plotting (first target for simplicity of equity panel)
            if target_label == "1pct":
                equity_panels[sym] = df_eval[["exit_time", "r"] + [f"r_{n}" for n in sizings]].copy()

        # Per-year breakdown (1pct target only, for backward compat)
        df_eval_1pct = df.copy()
        df_eval_1pct["p_1pct"] = fit_predict_walkforward_logit(df, feature_cols, "is_severe_loss")
        df_eval_1pct = df_eval_1pct.dropna(subset=["p_1pct"]).copy()
        df_eval_1pct["r"] = df_eval_1pct["pnl_pct"] / 100.0
        sizings_1pct = sizing_functions(df_eval_1pct["p_1pct"].values)
        for name in sizings_1pct:
            df_eval_1pct[f"r_{name}"] = df_eval_1pct["r"] * sizings_1pct[name]
        for name in sizings_1pct:
            for y, ysub in df_eval_1pct.groupby("year"):
                m = equity_metrics(ysub, f"r_{name}")
                if not m:
                    continue
                m["symbol"] = sym
                m["target"] = "1pct"
                m["sizing"] = name
                m["year"] = int(y)
                m["scope"] = "yearly"
                summary_rows.append(m)

    pd.DataFrame(summary_rows).to_csv(OUT / "sizing_simulation_summary.csv", index=False)

    # Plot equity curves per symbol (1pct target)
    fig, axes = plt.subplots(2, 1, figsize=(11, 9))
    for ax, sym in zip(axes, ("TQQQ", "SQQQ")):
        d = equity_panels[sym].sort_values("exit_time")
        d["exit_date"] = d["exit_time"].dt.normalize()
        bdays = pd.bdate_range(d["exit_date"].min(), d["exit_date"].max())
        for name, color in [("baseline_full", "#374151"),
                             ("linear_skip", "#2563eb"),
                             ("sqrt_skip", "#16a34a"),
                             ("step_skip_at_50", "#dc2626")]:
            col = f"r_{name}"
            daily = d.groupby("exit_date")[col].sum().reindex(bdays, fill_value=0.0)
            eq = 1.0 + daily.cumsum()
            ax.plot(bdays, eq.values, color=color, linewidth=1.2, label=name)
        ax.axhline(1.0, color="black", linewidth=0.5)
        ax.set_title(f"{sym}: constant-notional equity under sizing (target: -1%)")
        ax.set_ylabel("equity (1 + Σ sized_r)")
        ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "equity_under_sizing.png", dpi=140)
    plt.close(fig)

    # Print headline (totals only)
    headline = pd.DataFrame([r for r in summary_rows if r.get("scope") != "yearly"])
    print(headline[["symbol", "target", "sizing", "n_trades", "mean_size",
                    "total_return", "annualized_return", "sharpe_daily",
                    "max_drawdown", "calmar"]].to_string(index=False))


if __name__ == "__main__":
    main()
