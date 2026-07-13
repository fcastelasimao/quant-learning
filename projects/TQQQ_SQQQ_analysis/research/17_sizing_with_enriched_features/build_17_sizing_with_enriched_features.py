"""17_sizing_with_enriched_features: re-run item 12 with the item-06 feature set.

Depends on:
  - research/06_context_enrichment/enriched_trades_<sym>.csv

This is the deployment-target script. Its `linear_skip_enriched_1pct` scenario
is the recommendation in SYNTHESIS.md.


Item 12 used the curated_12 + regime-dummy features (L1-logit OOS AUC ~0.59 on
is_severe_loss). Item 06 lifted that AUC to ~0.72 by adding 21 daily cross-
asset context features. The synthesis flagged that re-running the sizing sim
with the enriched model is the highest-value open follow-up.

This script:
  - Loads the per-trade enriched canonical from item 06.
  - Fits L1-logistic per WF window (train on [start..Y-1], predict on Y) on
    is_severe_loss @ -1 % and is_severe_loss @ -2 % targets.
  - Applies four sizing functions:
        baseline_full           = 1
        linear_skip             = 1 - p_severe
        sqrt_skip               = sqrt(1 - p_severe)
        step_skip_at_50         = 1 if p_severe < 0.5 else 0
  - Reports per-symbol equity / Sharpe / MaxDD / Calmar across scenarios.
  - Compares directly to item 12's numbers in the findings doc.

All prior research directions left untouched.
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
from sklearn.metrics import brier_score_loss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _walkforward import equity_metrics as _equity_metrics, fit_predict_walkforward_logit  # noqa: E402


ROOT = Path(__file__).resolve().parent
PROJ = ROOT.parent.parent
ITEM_06 = ROOT.parent / "06_context_enrichment"
OUT = ROOT
SEED = 42

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


def load_enriched(sym: str) -> pd.DataFrame:
    df = pd.read_csv(ITEM_06 / f"enriched_trades_{sym}.csv",
                     parse_dates=["entry_time", "exit_time"])
    for c in CURATED_NUMERIC + DAILY_CONTEXT:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["year"] = df["entry_time"].dt.year
    for r in ("chop_highvol", "sideways_lowvol"):
        df[f"regime_{r}"] = (df["regime_entry"] == r).astype(int)
    df["is_severe_loss_at_1p5pct"] = (df["pnl_pct"] <= -1.5).astype(int)
    df["is_severe_loss_at_2pct"]   = (df["pnl_pct"] <= -2.0).astype(int)
    return df


def fit_predict_walkforward(df: pd.DataFrame, feature_cols: list[str],
                            target_col: str) -> pd.Series:
    """Thin wrapper around shared _walkforward helper for backward compatibility."""
    return fit_predict_walkforward_logit(df, feature_cols, target_col)


def sizing_functions(p: pd.Series) -> dict[str, pd.Series]:
    p_clip = p.clip(0.0, 1.0)
    return {
        "baseline_full": pd.Series(1.0, index=p.index),
        "linear_skip": 1.0 - p_clip,
        "sqrt_skip": np.sqrt(np.clip(1.0 - p_clip, 0.0, 1.0)),
        "step_skip_at_50": (p_clip < 0.5).astype(float),
    }


def equity_metrics(df: pd.DataFrame, sized_r_col: str) -> dict:
    """Thin wrapper around shared _walkforward helper for backward compatibility."""
    return _equity_metrics(df, sized_r_col)


def calibration_tables(df: pd.DataFrame, sym: str, p_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return decile and yearly calibration summaries for p_severe."""
    work = df.dropna(subset=[p_col, "is_severe_loss"]).copy()
    work["p_decile"] = pd.qcut(work[p_col], 10, labels=False, duplicates="drop")
    decile = work.groupby("p_decile", observed=True).agg(
        symbol=("symbol", lambda s: sym),
        n=("is_severe_loss", "size"),
        p_mean=(p_col, "mean"),
        severe_rate=("is_severe_loss", "mean"),
    ).reset_index()
    decile["abs_calibration_error"] = (decile["p_mean"] - decile["severe_rate"]).abs()

    yearly = work.groupby("year").agg(
        symbol=("symbol", lambda s: sym),
        n=("is_severe_loss", "size"),
        p_mean=(p_col, "mean"),
        severe_rate=("is_severe_loss", "mean"),
    ).reset_index()
    yearly["brier_score"] = [
        brier_score_loss(work.loc[work["year"] == y, "is_severe_loss"].astype(int),
                         work.loc[work["year"] == y, p_col])
        for y in yearly["year"]
    ]
    yearly["abs_calibration_error"] = (yearly["p_mean"] - yearly["severe_rate"]).abs()
    return decile, yearly


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    calibration_deciles = []
    calibration_yearly = []
    equity_panels: dict = {}

    for sym in ("TQQQ", "SQQQ"):
        df = load_enriched(sym)
        feature_cols_enriched = CURATED_NUMERIC + REGIME_DUMMIES + DAILY_CONTEXT

        # Predict p_severe at -1%, -1.5%, and -2% thresholds using ENRICHED features.
        df["p_severe_1pct_enriched"] = fit_predict_walkforward(
            df, feature_cols_enriched, "is_severe_loss"
        )
        df["p_severe_1p5pct_enriched"] = fit_predict_walkforward(
            df, feature_cols_enriched, "is_severe_loss_at_1p5pct"
        )
        df["p_severe_2pct_enriched"] = fit_predict_walkforward(
            df, feature_cols_enriched, "is_severe_loss_at_2pct"
        )
        # For comparability, also predict with the BASELINE (curated_12 only)
        # — replicates item 12's setup.
        feature_cols_baseline = CURATED_NUMERIC + REGIME_DUMMIES
        df["p_severe_1pct_baseline"] = fit_predict_walkforward(
            df, feature_cols_baseline, "is_severe_loss"
        )

        # Restrict to rows with non-null predictions across all probability vars
        df_eval = df.dropna(subset=["p_severe_1pct_enriched",
                                    "p_severe_1p5pct_enriched",
                                    "p_severe_2pct_enriched",
                                    "p_severe_1pct_baseline"]).copy()
        df_eval["r"] = df_eval["pnl_pct"] / 100.0
        decile, yearly = calibration_tables(df_eval, sym, "p_severe_1pct_enriched")
        calibration_deciles.append(decile)
        calibration_yearly.append(yearly)

        scenarios = {
            "baseline_full": pd.Series(1.0, index=df_eval.index),
            "linear_skip_baseline_1pct (item 12 replica)":
                1.0 - df_eval["p_severe_1pct_baseline"].clip(0.0, 1.0),
            "linear_skip_enriched_1pct (NEW)":
                1.0 - df_eval["p_severe_1pct_enriched"].clip(0.0, 1.0),
            "sqrt_skip_enriched_1pct (NEW)":
                np.sqrt(np.clip(1.0 - df_eval["p_severe_1pct_enriched"].clip(0.0, 1.0), 0.0, 1.0)),
            "linear_skip_enriched_1p5pct (NEW)":
                1.0 - df_eval["p_severe_1p5pct_enriched"].clip(0.0, 1.0),
            "sqrt_skip_enriched_1p5pct (NEW)":
                np.sqrt(np.clip(1.0 - df_eval["p_severe_1p5pct_enriched"].clip(0.0, 1.0), 0.0, 1.0)),
            "linear_skip_enriched_2pct (NEW)":
                1.0 - df_eval["p_severe_2pct_enriched"].clip(0.0, 1.0),
            "sqrt_skip_enriched_2pct (NEW)":
                np.sqrt(np.clip(1.0 - df_eval["p_severe_2pct_enriched"].clip(0.0, 1.0), 0.0, 1.0)),
        }

        for name, size in scenarios.items():
            col = f"r_{name}"
            df_eval[col] = df_eval["r"] * size
            m = equity_metrics(df_eval, col)
            m["symbol"] = sym
            m["sizing"] = name
            m["n_trades"] = len(df_eval)
            m["mean_size"] = float(size.mean())
            summary_rows.append(m)

        equity_panels[sym] = df_eval.copy()

    summ = pd.DataFrame(summary_rows)
    summ = summ[["symbol", "sizing", "n_trades", "mean_size",
                 "total_return", "annualized_return", "sharpe_daily",
                 "max_drawdown", "calmar", "mean_daily_return", "std_daily_return"]]
    summ.to_csv(OUT / "sizing_enriched_summary.csv", index=False)
    pd.concat(calibration_deciles, ignore_index=True).to_csv(
        OUT / "calibration_deciles_enriched_1pct.csv", index=False
    )
    pd.concat(calibration_yearly, ignore_index=True).to_csv(
        OUT / "calibration_yearly_enriched_1pct.csv", index=False
    )
    print(summ.to_string(index=False))

    # Plot equity curves per symbol
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    colors = {
        "baseline_full": "#374151",
        "linear_skip_baseline_1pct (item 12 replica)": "#9ca3af",
        "linear_skip_enriched_1pct (NEW)": "#2563eb",
        "sqrt_skip_enriched_1pct (NEW)": "#16a34a",
        "linear_skip_enriched_1p5pct (NEW)": "#f59e0b",
        "sqrt_skip_enriched_1p5pct (NEW)": "#84cc16",
        "linear_skip_enriched_2pct (NEW)": "#dc2626",
        "sqrt_skip_enriched_2pct (NEW)": "#9333ea",
    }
    for ax, sym in zip(axes, ("TQQQ", "SQQQ")):
        d = equity_panels[sym].sort_values("exit_time")
        d["exit_date"] = d["exit_time"].dt.normalize()
        d = d.dropna(subset=["exit_date"])
        bdays = pd.bdate_range(d["exit_date"].min(), d["exit_date"].max())
        for name, color in colors.items():
            col = f"r_{name}"
            daily = d.groupby("exit_date")[col].sum().reindex(bdays, fill_value=0.0)
            eq = 1.0 + daily.cumsum()
            ax.plot(bdays, eq.values, color=color, linewidth=1.0 if "NEW" in name else 1.3,
                     label=name, alpha=0.95 if "NEW" in name else 0.7,
                     linestyle="-" if "NEW" in name or "baseline" in name else "--")
        ax.axhline(1.0, color="black", linewidth=0.5)
        ax.set_title(f"{sym}: equity under sizing functions (baseline vs item-06 enriched)")
        ax.set_ylabel("equity (1 + Σ sized_r)")
        ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "equity_enriched_sizing.png", dpi=140)
    plt.close(fig)

    # Also plot Sharpe vs Max-DD scatter for the headline comparison
    fig, ax = plt.subplots(figsize=(9, 6))
    for sym, marker in (("TQQQ", "o"), ("SQQQ", "s")):
        d = summ[summ["symbol"] == sym]
        for _, r in d.iterrows():
            color = colors.get(r["sizing"], "black")
            ax.scatter(-r["max_drawdown"] * 100, r["sharpe_daily"],
                       s=120, color=color, marker=marker, edgecolor="black", linewidth=0.5)
            ax.annotate(f"{sym} {r['sizing'].split(' ')[0]}",
                        (-r["max_drawdown"] * 100, r["sharpe_daily"]),
                        xytext=(6, 0), textcoords="offset points", fontsize=7)
    ax.set_xlabel("Max drawdown (%)  ← smaller is better")
    ax.set_ylabel("Sharpe (daily, annualized)")
    ax.set_title("Item 17: Sharpe vs Max-DD across sizing scenarios\n"
                 "(top-left = ideal: high Sharpe, small drawdown)")
    fig.tight_layout()
    fig.savefig(OUT / "sharpe_vs_drawdown.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
