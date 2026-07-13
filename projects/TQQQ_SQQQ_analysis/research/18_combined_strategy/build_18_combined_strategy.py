"""18_combined_strategy: combine p_severe continuous sizing + crisp skip rules.

Depends on:
  - research/06_context_enrichment/enriched_trades_{sym}.csv

Combines three independently validated components:
  (A) p_severe continuous sizing — walk-forward L1-logit, is_severe_loss @ -1%
  (B) SQQQ focus rule — skip RSI ∈ [56.4, 59.85] AND atr ∈ [0.39, 0.47]
  (C) TQQQ regime rule — skip sideways_lowvol AND atr ∈ (0.424, 0.476] AND MA100_D1 ≤ 0.000204

Scenarios per symbol:
  baseline_full   — always trade at full size
  p_severe_only   — size = 1 - p_severe (continuous)
  rules_only      — size = 0 if rule fires, else 1 (binary skip)
  combined        — size = 0 if rule fires, else (1 - p_severe)

Best target per item 17:
  TQQQ: is_severe_loss @ -1%  (linear_skip)
  SQQQ: is_severe_loss @ -2%  (sqrt_skip — but item 18 uses linear for consistency)

We report both TQQQ (linear_skip @ -1%) and SQQQ (linear_skip @ -2%) in the
sizing scenarios, plus the combined rule override.
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
from _walkforward import equity_metrics, fit_predict_walkforward_logit  # noqa: E402

# Import rule masks from deployable_strategies
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from deployable_strategies.focus_rules.sqqq_rsi_atr_skip import sqqq_focus_rule_mask  # noqa: E402
from deployable_strategies.regime_rules.tqqq_sideways_skip import tqqq_regime_rule_mask  # noqa: E402

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


def load_enriched(sym: str) -> pd.DataFrame:
    df = pd.read_csv(ITEM_06 / f"enriched_trades_{sym}.csv",
                     parse_dates=["entry_time", "exit_time"])
    for c in CURATED_NUMERIC + DAILY_CONTEXT:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["year"] = df["entry_time"].dt.year
    for r in ("chop_highvol", "sideways_lowvol"):
        df[f"regime_{r}"] = (df["regime_entry"] == r).astype(int)
    df["is_severe_loss"]      = (df["pnl_pct"] <= -1.0).astype(int)
    df["is_severe_loss_2pct"] = (df["pnl_pct"] <= -2.0).astype(int)
    return df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # Per-symbol best target per item 17
    symbol_targets = {
        "TQQQ": ("is_severe_loss", "1pct"),
        "SQQQ": ("is_severe_loss_2pct", "2pct"),
    }

    summary_rows = []
    attr_rows = []
    equity_panels: dict = {}

    for sym in ("TQQQ", "SQQQ"):
        df = load_enriched(sym)
        target_col, target_label = symbol_targets[sym]

        # Walk-forward p_severe
        df["p_severe"] = fit_predict_walkforward_logit(df, FEATURE_COLS, target_col)
        df_eval = df.dropna(subset=["p_severe"]).copy()
        df_eval["r"] = df_eval["pnl_pct"] / 100.0

        # Rule mask
        if sym == "SQQQ":
            df_eval["rule_fires"] = sqqq_focus_rule_mask(df_eval).astype(float)
        else:
            df_eval["rule_fires"] = tqqq_regime_rule_mask(df_eval).astype(float)

        n_rule = int(df_eval["rule_fires"].sum())
        n_psevere_high = int((df_eval["p_severe"] > 0.5).sum())
        n_both = int(((df_eval["rule_fires"] == 1) & (df_eval["p_severe"] > 0.5)).sum())

        attr_rows.append({
            "symbol": sym, "target": target_label,
            "n_trades_eval": len(df_eval),
            "n_rule_fires": n_rule,
            "n_p_severe_gt50": n_psevere_high,
            "n_both": n_both,
            "pct_rule": n_rule / len(df_eval) * 100,
        })

        # Scenarios
        p = df_eval["p_severe"].clip(0.0, 1.0)
        scenarios: dict[str, pd.Series] = {
            "baseline_full": pd.Series(1.0, index=df_eval.index),
            "p_severe_only": 1.0 - p,
            "rules_only":    (1.0 - df_eval["rule_fires"]),
            "combined":      (1.0 - df_eval["rule_fires"]) * (1.0 - p),
        }

        for name, size in scenarios.items():
            r_col = f"r_{name}"
            df_eval[r_col] = df_eval["r"] * size
            m = equity_metrics(df_eval, r_col)
            m["symbol"] = sym
            m["scenario"] = name
            m["target"] = target_label
            m["mean_size"] = float(size.mean())
            m["n_trades"] = len(df_eval)
            summary_rows.append(m)

        equity_panels[sym] = df_eval.copy()

    summary = pd.DataFrame(summary_rows)
    if "annualized_return" in summary.columns:
        summary = summary.rename(columns={"annualized_return": "cagr"})
    col_order = ["symbol", "scenario", "target", "n_trades", "mean_size",
                 "total_return", "cagr", "sharpe_daily", "max_drawdown", "calmar"]
    summary = summary[[c for c in col_order if c in summary.columns]]
    summary.to_csv(OUT / "combined_strategy_summary.csv", index=False)

    attr = pd.DataFrame(attr_rows)
    attr.to_csv(OUT / "component_attribution.csv", index=False)

    print(summary.to_string(index=False))
    print("\nComponent attribution:")
    print(attr.to_string(index=False))

    # Equity curves — one panel per symbol, all 4 scenarios
    colors = {
        "baseline_full": "#374151",
        "p_severe_only": "#2563eb",
        "rules_only":    "#16a34a",
        "combined":      "#dc2626",
    }
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    for ax, sym in zip(axes, ("TQQQ", "SQQQ")):
        d = equity_panels[sym].sort_values("exit_time").copy()
        d["exit_date"] = d["exit_time"].dt.normalize()
        d = d.dropna(subset=["exit_date"])
        bdays = pd.bdate_range(d["exit_date"].min(), d["exit_date"].max())
        for name, color in colors.items():
            r_col = f"r_{name}"
            daily = d.groupby("exit_date")[r_col].sum().reindex(bdays, fill_value=0.0)
            eq = 1.0 + daily.cumsum()
            ax.plot(bdays, eq.values, color=color, linewidth=1.2, label=name, alpha=0.9)
        ax.axhline(1.0, color="black", linewidth=0.5)
        ax.set_title(f"{sym}: combined strategy equity curves")
        ax.set_ylabel("equity (1 + Σ sized_r)")
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "combined_equity_curves.png", dpi=140)
    plt.close(fig)
    print(f"\nOutputs written to {OUT}")


if __name__ == "__main__":
    main()
