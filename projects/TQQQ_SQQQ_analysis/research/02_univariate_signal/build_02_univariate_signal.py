"""02_univariate_signal: per-feature signal vs is_loser, is_severe_loss, pnl_pct.

Regime-labeled subset only. Full 28-feature list (redundancy kept visible).
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-quant")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parent
CANON = ROOT.parent.parent / "full_history_canonical"
OUT = ROOT

NUMERIC_FEATURES = [
    "BBP_entry", "BBP_entry_roll_pctile_252",
    "MA100", "MA100_D1", "MA20", "MA20_D1", "MA20_D3", "MA20_D5",
    "MA50", "MA50_D1", "MA50_D3", "MA50_D5",
    "RSI_entry", "RSI_entry_roll_pctile_252",
    "atr", "atr_pct", "atr_pct_roll_pctile_252",
    "bars_since_last_stop",
    "dist_to_MA100", "dist_to_MA20", "dist_to_MA50",
    "high_water_mark_entry", "hour_of_entry", "is_bullish_c1c2",
    "log_volume_ratio", "volume_cur",
    "volume_ratio", "volume_ratio_roll_pctile_252",
]

CLUSTER_REPRESENTATIVES = {
    "atr_pct", "RSI_entry", "BBP_entry",
    "dist_to_MA20", "dist_to_MA50", "dist_to_MA100",
    "MA20_D5", "MA50_D5", "MA100_D1",
    "log_volume_ratio", "bars_since_last_stop",
    "hour_of_entry", "is_bullish_c1c2",
}

TARGETS = [
    ("is_loser", "binary"),
    ("is_severe_loss", "binary"),
    ("pnl_pct", "continuous"),
]


def load(sym: str) -> pd.DataFrame:
    df = pd.read_csv(
        CANON / f"TRADES_{sym}_full_history.csv",
        parse_dates=["entry_time", "exit_time"],
    )
    return df[df["regime_entry"].notna()].copy()


def univariate(df: pd.DataFrame, feat: str, target: str, kind: str) -> dict | None:
    valid = df[[feat, target]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(valid) < 30 or valid[feat].nunique() < 2:
        return None
    x = valid[feat].values
    y = valid[target].values
    out = {"feature": feat, "target": target, "n": int(len(valid)),
           "is_cluster_representative": feat in CLUSTER_REPRESENTATIVES}

    rho = stats.spearmanr(x, y)
    out["spearman"] = float(rho.statistic)
    out["spearman_p"] = float(rho.pvalue)

    pear = stats.pearsonr(x, y)
    out["pearson"] = float(pear.statistic)
    out["pearson_p"] = float(pear.pvalue)

    if kind == "binary":
        try:
            auc = roc_auc_score(y, x)
        except ValueError:
            auc = np.nan
        out["auc"] = float(auc)
        out["auc_directional"] = float(max(auc, 1 - auc))

        x0 = valid.loc[y == 0, feat].values
        x1 = valid.loc[y == 1, feat].values
        out["mean_y0"] = float(np.mean(x0))
        out["mean_y1"] = float(np.mean(x1))
        if len(x0) > 5 and len(x1) > 5:
            ks = stats.ks_2samp(x0, x1)
            mw = stats.mannwhitneyu(x0, x1, alternative="two-sided")
            out["ks_stat"] = float(ks.statistic)
            out["ks_p"] = float(ks.pvalue)
            out["mannwhitney_p"] = float(mw.pvalue)
        try:
            mi = mutual_info_classif(x.reshape(-1, 1), y, random_state=42, discrete_features=False)[0]
            out["mutual_info"] = float(mi)
        except Exception:
            out["mutual_info"] = np.nan
    else:
        try:
            mi = mutual_info_regression(x.reshape(-1, 1), y, random_state=42)[0]
            out["mutual_info"] = float(mi)
        except Exception:
            out["mutual_info"] = np.nan

    return out


def rank_by(kind: str, df: pd.DataFrame) -> pd.DataFrame:
    if kind == "binary":
        df["score"] = (df["auc_directional"].fillna(0.5) - 0.5) + df["mutual_info"].fillna(0)
        return df.sort_values(["score", "auc_directional", "mutual_info"], ascending=[False, False, False])
    df["score"] = df["spearman"].abs().fillna(0) + df["mutual_info"].fillna(0)
    return df.sort_values(["score", "spearman", "mutual_info"],
                          ascending=[False, False, False],
                          key=lambda s: s.abs() if s.name in ("spearman",) else s)


def plot_auc_bars(table: pd.DataFrame, sym: str, target: str, kind: str, out_dir: Path) -> None:
    """Horizontal bar chart of top-12 features by AUC (binary) or |Spearman| (continuous)."""
    t = table.head(12).copy()
    if kind == "binary":
        x_col, x_label = "auc_directional", "directional AUC"
        baseline = 0.5
    else:
        t["abs_spearman"] = t["spearman"].abs()
        x_col, x_label = "abs_spearman", "|Spearman| with pnl_pct"
        baseline = 0.0
    t = t.iloc[::-1]  # reverse so biggest is on top in barh
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = ["#dc2626" if r else "#2563eb" for r in t["is_cluster_representative"]]
    ax.barh(range(len(t)), t[x_col], color=colors, edgecolor="black", linewidth=0.4)
    ax.axvline(baseline, color="black", linewidth=0.6, linestyle="--",
               label=f"baseline ({baseline:.2f})")
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels(t["feature"], fontsize=8)
    ax.set_xlabel(x_label)
    ax.set_title(f"{sym}: top features for {target} (red = curated-set representative)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / f"top_features_{sym}_{target}.png", dpi=140)
    plt.close(fig)


def main() -> None:
    for sym in ("TQQQ", "SQQQ"):
        df = load(sym)
        for target, kind in TARGETS:
            rows = []
            for feat in NUMERIC_FEATURES:
                if feat not in df.columns:
                    continue
                r = univariate(df, feat, target, kind)
                if r:
                    rows.append(r)
            t = pd.DataFrame(rows)
            t = rank_by(kind, t)
            t.insert(0, "symbol", sym)
            path = OUT / f"univariate_{sym}_{target}.csv"
            t.to_csv(path, index=False)
            plot_auc_bars(t, sym, target, kind, OUT)
            print(f"wrote {path.name}: {len(t)} features + plot")


if __name__ == "__main__":
    main()
