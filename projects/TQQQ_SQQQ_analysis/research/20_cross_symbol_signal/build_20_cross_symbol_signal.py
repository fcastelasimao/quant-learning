"""20_cross_symbol_signal: does training on TQQQ predict SQQQ severe losses, or vice versa?

Depends on:
  - research/06_context_enrichment/enriched_trades_TQQQ.csv
  - research/06_context_enrichment/enriched_trades_SQQQ.csv

Four walk-forward experiments:
  (1) train TQQQ → score TQQQ   (own-symbol baseline)
  (2) train SQQQ → score SQQQ   (own-symbol baseline)
  (3) train TQQQ → score SQQQ   (cross-symbol)
  (4) train SQQQ → score TQQQ   (cross-symbol)

For each experiment, reports:
  - Per-year OOS AUC (WF fold stability)
  - Aggregate OOS AUC over the full OOS window

If cross AUC ≈ own AUC, the features capture shared market-regime effects.
If cross << own, the symbols have distinct loss mechanisms.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent
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
FEATURE_COLS = CURATED_NUMERIC + REGIME_DUMMIES + DAILY_CONTEXT
TARGET_COL = "is_severe_loss"


def load_sym(sym: str) -> pd.DataFrame:
    df = pd.read_csv(ITEM_06 / f"enriched_trades_{sym}.csv",
                     parse_dates=["entry_time", "exit_time"])
    for c in CURATED_NUMERIC + DAILY_CONTEXT:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["year"] = df["entry_time"].dt.year
    for r in ("chop_highvol", "sideways_lowvol"):
        df[f"regime_{r}"] = (df["regime_entry"] == r).astype(int)
    df[TARGET_COL] = (df["pnl_pct"] <= -1.0).astype(int)
    return df


def wf_cross_auc(
    df_train: pd.DataFrame,
    df_score: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    min_train: int = 100,
    min_positives: int = 10,
) -> tuple[list[dict], float]:
    """Walk-forward: for each year Y in df_score, train on df_train[year < Y], score on df_score[year == Y].

    Returns: list of per-year dicts, and aggregate OOS AUC over all scored rows.
    """
    years = sorted(df_score["year"].dropna().astype(int).unique())
    yearly_rows = []
    all_labels = []
    all_preds = []

    for y in years:
        train = df_train[df_train["year"] < y].dropna(subset=feature_cols + [target_col])
        test  = df_score[df_score["year"] == y].dropna(subset=feature_cols + [target_col])

        if len(train) < min_train or int(train[target_col].sum()) < min_positives:
            continue
        if len(test) < 5 or int(test[target_col].sum()) < 1:
            continue

        sc = StandardScaler().fit(train[feature_cols].values)
        mdl = LogisticRegression(penalty="l1", solver="liblinear", C=0.1,
                                  random_state=SEED, max_iter=2000)
        mdl.fit(sc.transform(train[feature_cols].values),
                train[target_col].astype(int).values)
        preds = mdl.predict_proba(sc.transform(test[feature_cols].values))[:, 1]
        labels = test[target_col].astype(int).values

        try:
            auc = roc_auc_score(labels, preds)
        except Exception:
            continue

        yearly_rows.append({"year": y, "n_test": len(test), "n_pos": int(labels.sum()), "oos_auc": float(auc)})
        all_labels.extend(labels.tolist())
        all_preds.extend(preds.tolist())

    agg_auc = float(roc_auc_score(all_labels, all_preds)) if len(set(all_labels)) == 2 else float("nan")
    return yearly_rows, agg_auc


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tqqq = load_sym("TQQQ")
    sqqq = load_sym("SQQQ")

    experiments = [
        ("TQQQ", "TQQQ", tqqq, tqqq, "own_symbol"),
        ("SQQQ", "SQQQ", sqqq, sqqq, "own_symbol"),
        ("TQQQ", "SQQQ", tqqq, sqqq, "cross_symbol"),
        ("SQQQ", "TQQQ", sqqq, tqqq, "cross_symbol"),
    ]

    summary_rows = []
    yearly_rows_all = []

    for train_sym, score_sym, df_tr, df_sc, kind in experiments:
        label = f"train_{train_sym}_score_{score_sym}"
        print(f"\n{label}...")
        yr_rows, agg_auc = wf_cross_auc(df_tr, df_sc, FEATURE_COLS, TARGET_COL)
        if not yr_rows:
            print(f"  No valid WF folds — skipping")
            continue
        yr_df = pd.DataFrame(yr_rows)
        yr_df["train_symbol"] = train_sym
        yr_df["score_symbol"] = score_sym
        yr_df["kind"] = kind
        yearly_rows_all.append(yr_df)

        wf_median = float(yr_df["oos_auc"].median())
        wf_min    = float(yr_df["oos_auc"].min())
        wf_max    = float(yr_df["oos_auc"].max())
        summary_rows.append({
            "train_symbol": train_sym,
            "score_symbol": score_sym,
            "kind": kind,
            "agg_oos_auc": agg_auc,
            "wf_median_auc": wf_median,
            "wf_min_auc": wf_min,
            "wf_max_auc": wf_max,
            "n_folds": len(yr_rows),
        })
        print(f"  agg_oos_auc={agg_auc:.3f}  wf_median={wf_median:.3f}  [{wf_min:.3f}, {wf_max:.3f}]")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "cross_symbol_auc.csv", index=False)
    yearly = pd.concat(yearly_rows_all, ignore_index=True) if yearly_rows_all else pd.DataFrame()
    yearly.to_csv(OUT / "cross_symbol_yearly_auc.csv", index=False)
    print("\nSummary:")
    print(summary[["train_symbol", "score_symbol", "kind", "agg_oos_auc", "wf_median_auc"]].to_string(index=False))

    # Bar chart: own vs cross AUC per scored symbol
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    for ax, score_sym in zip(axes, ["TQQQ", "SQQQ"]):
        sub = summary[summary["score_symbol"] == score_sym]
        if sub.empty:
            ax.set_title(f"Score {score_sym}: no data")
            continue
        labels_bar = [f"Train {r['train_symbol']}\n({r['kind']})" for _, r in sub.iterrows()]
        vals = sub["agg_oos_auc"].tolist()
        colors = ["#2563eb" if k == "own_symbol" else "#dc2626" for k in sub["kind"]]
        bars = ax.bar(labels_bar, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.axhline(0.5, color="black", linewidth=0.6, linestyle="--")
        ax.set_title(f"Score {score_sym}: own vs cross AUC")
        ax.set_ylabel("Aggregate OOS AUC")
        ax.set_ylim(0.4, 0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color="#2563eb", label="own-symbol"),
                         Patch(color="#dc2626", label="cross-symbol")],
               loc="lower center", ncol=2, fontsize=9)
    fig.suptitle("Item 20: Cross-symbol AUC — does TQQQ signal predict SQQQ?")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(OUT / "cross_symbol_auc.png", dpi=140)
    plt.close(fig)
    print(f"\nOutputs written to {OUT}")


if __name__ == "__main__":
    main()
