"""07_severity_threshold_sweep: where does predictability of severity peak?

For each (symbol, side, threshold) combination, fit a depth-4 tree AND an
L1-logistic regression on the binary target and report base rate, OOS AUC,
OOS lift on top-decile-predicted, and per-WF-year AUC stability.
Helps locate the threshold that maximises the *predictability × usefulness*
trade-off and whether tree or L1-logit is more informative.

Sides:
  loss thresholds: pnl_pct <= {-0.25, -0.5, -1.0, -1.5, -2.0, -3.0}
  win thresholds:  pnl_pct >= {+0.5, +1.0, +1.5, +2.0, +3.0}

Uses item 04's curated 12 features + regime dummies. Class imbalance is
handled by class_weight='balanced' (tree) and C=0.1 (L1-logit).
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parent
CANON = ROOT.parent.parent / "full_history_canonical"
OUT = ROOT
SEED = 42
IS_END_YEAR = 2020

CURATED_NUMERIC = [
    "atr_pct", "RSI_entry", "BBP_entry",
    "dist_to_MA20", "dist_to_MA50", "dist_to_MA100",
    "MA20_D5", "MA50_D5", "MA100_D1",
    "log_volume_ratio", "bars_since_last_stop", "hour_of_entry",
]

LOSS_THRESHOLDS = [-0.25, -0.5, -1.0, -1.5, -2.0, -3.0]
WIN_THRESHOLDS = [0.5, 1.0, 1.5, 2.0, 3.0]


def load(sym: str) -> pd.DataFrame:
    df = pd.read_csv(CANON / f"TRADES_{sym}_full_history.csv",
                     parse_dates=["entry_time", "exit_time"])
    df = df[df["regime_entry"].notna()].copy()
    df["year"] = df["entry_time"].dt.year
    for c in CURATED_NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for r in ("chop_highvol", "sideways_lowvol"):
        df[f"regime_{r}"] = (df["regime_entry"] == r).astype(int)
    return df.dropna(subset=CURATED_NUMERIC).copy()


def fit_one(df_is: pd.DataFrame, df_oos: pd.DataFrame, feature_cols: list[str],
            y_is: np.ndarray, y_oos: np.ndarray) -> dict | None:
    if y_is.sum() < 10 or len(y_is) - y_is.sum() < 10:
        return None
    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=30,
                                   class_weight="balanced", random_state=SEED)
    tree.fit(df_is[feature_cols].values, y_is)
    if y_oos.sum() == 0 or len(y_oos) - y_oos.sum() == 0:
        return None
    p_oos = tree.predict_proba(df_oos[feature_cols].values)[:, 1]
    auc = roc_auc_score(y_oos, p_oos)
    cutoff = np.quantile(p_oos, 0.9)
    bot = y_oos[p_oos >= cutoff]
    top_dec_rate = float(bot.mean()) if len(bot) else np.nan
    base_rate = float(y_oos.mean())
    return {
        "oos_auc": float(auc),
        "n_oos_positive": int(y_oos.sum()),
        "n_oos_total": int(len(y_oos)),
        "base_rate_oos": base_rate,
        "top_decile_rate": top_dec_rate,
        "top_decile_lift": top_dec_rate - base_rate if not np.isnan(top_dec_rate) else np.nan,
    }


def fit_one_logit(df_is: pd.DataFrame, df_oos: pd.DataFrame, feature_cols: list[str],
                  y_is: np.ndarray, y_oos: np.ndarray) -> dict | None:
    if y_is.sum() < 10 or len(y_is) - y_is.sum() < 10:
        return None
    if y_oos.sum() == 0 or len(y_oos) - y_oos.sum() == 0:
        return None
    sc = StandardScaler().fit(df_is[feature_cols].values)
    logit = LogisticRegression(penalty="l1", solver="liblinear", C=0.1,
                                random_state=SEED, max_iter=2000)
    logit.fit(sc.transform(df_is[feature_cols].values), y_is)
    p_oos = logit.predict_proba(sc.transform(df_oos[feature_cols].values))[:, 1]
    auc = roc_auc_score(y_oos, p_oos)
    cutoff = np.quantile(p_oos, 0.9)
    bot = y_oos[p_oos >= cutoff]
    top_dec_rate = float(bot.mean()) if len(bot) else np.nan
    base_rate = float(y_oos.mean())
    return {
        "oos_auc": float(auc),
        "n_oos_positive": int(y_oos.sum()),
        "n_oos_total": int(len(y_oos)),
        "base_rate_oos": base_rate,
        "top_decile_rate": top_dec_rate,
        "top_decile_lift": top_dec_rate - base_rate if not np.isnan(top_dec_rate) else np.nan,
    }


def wf_auc_stability(df: pd.DataFrame, feature_cols: list[str], y_col_fn,
                     model: str = "tree") -> list[float]:
    aucs = []
    for y_end in range(2017, 2026):
        train = df[df["year"] <= y_end]
        test = df[df["year"] == y_end + 1]
        if len(train) < 100 or len(test) < 20:
            continue
        ytr = y_col_fn(train).values
        yte = y_col_fn(test).values
        if ytr.sum() < 10 or yte.sum() == 0 or len(yte) - yte.sum() == 0:
            continue
        if model == "tree":
            clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=30,
                                          class_weight="balanced", random_state=SEED)
            clf.fit(train[feature_cols].values, ytr)
            p = clf.predict_proba(test[feature_cols].values)[:, 1]
        else:
            sc = StandardScaler().fit(train[feature_cols].values)
            clf = LogisticRegression(penalty="l1", solver="liblinear", C=0.1,
                                      random_state=SEED, max_iter=2000)
            clf.fit(sc.transform(train[feature_cols].values), ytr)
            p = clf.predict_proba(sc.transform(test[feature_cols].values))[:, 1]
        try:
            aucs.append(float(roc_auc_score(yte, p)))
        except ValueError:
            continue
    return aucs


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    feature_cols = CURATED_NUMERIC + ["regime_chop_highvol", "regime_sideways_lowvol"]

    for sym in ("TQQQ", "SQQQ"):
        df = load(sym)
        df_is = df[df["year"] <= IS_END_YEAR]
        df_oos = df[df["year"] > IS_END_YEAR]

        for thr in LOSS_THRESHOLDS:
            yfn = lambda d, t=thr: (d["pnl_pct"] <= t).astype(int)
            y_is = yfn(df_is).values
            y_oos = yfn(df_oos).values

            # Tree
            r = fit_one(df_is, df_oos, feature_cols, y_is, y_oos)
            if r is not None:
                wf = wf_auc_stability(df, feature_cols, yfn, model="tree")
                r.update({
                    "symbol": sym, "side": "loss", "threshold": thr, "model": "tree",
                    "wf_auc_median": float(np.median(wf)) if wf else np.nan,
                    "wf_auc_min": float(np.min(wf)) if wf else np.nan,
                    "wf_auc_max": float(np.max(wf)) if wf else np.nan,
                    "wf_n_windows": len(wf),
                    "is_positive_rate": float(y_is.mean()),
                })
                rows.append(r)

            # L1-logit
            r2 = fit_one_logit(df_is, df_oos, feature_cols, y_is, y_oos)
            if r2 is not None:
                wf2 = wf_auc_stability(df, feature_cols, yfn, model="logit")
                r2.update({
                    "symbol": sym, "side": "loss", "threshold": thr, "model": "l1_logit",
                    "wf_auc_median": float(np.median(wf2)) if wf2 else np.nan,
                    "wf_auc_min": float(np.min(wf2)) if wf2 else np.nan,
                    "wf_auc_max": float(np.max(wf2)) if wf2 else np.nan,
                    "wf_n_windows": len(wf2),
                    "is_positive_rate": float(y_is.mean()),
                })
                rows.append(r2)

        for thr in WIN_THRESHOLDS:
            yfn = lambda d, t=thr: (d["pnl_pct"] >= t).astype(int)
            y_is = yfn(df_is).values
            y_oos = yfn(df_oos).values

            # Tree
            r = fit_one(df_is, df_oos, feature_cols, y_is, y_oos)
            if r is not None:
                wf = wf_auc_stability(df, feature_cols, yfn, model="tree")
                r.update({
                    "symbol": sym, "side": "win", "threshold": thr, "model": "tree",
                    "wf_auc_median": float(np.median(wf)) if wf else np.nan,
                    "wf_auc_min": float(np.min(wf)) if wf else np.nan,
                    "wf_auc_max": float(np.max(wf)) if wf else np.nan,
                    "wf_n_windows": len(wf),
                    "is_positive_rate": float(y_is.mean()),
                })
                rows.append(r)

            # L1-logit
            r2 = fit_one_logit(df_is, df_oos, feature_cols, y_is, y_oos)
            if r2 is not None:
                wf2 = wf_auc_stability(df, feature_cols, yfn, model="logit")
                r2.update({
                    "symbol": sym, "side": "win", "threshold": thr, "model": "l1_logit",
                    "wf_auc_median": float(np.median(wf2)) if wf2 else np.nan,
                    "wf_auc_min": float(np.min(wf2)) if wf2 else np.nan,
                    "wf_auc_max": float(np.max(wf2)) if wf2 else np.nan,
                    "wf_n_windows": len(wf2),
                    "is_positive_rate": float(y_is.mean()),
                })
                rows.append(r2)

    out = pd.DataFrame(rows)
    cols = ["symbol", "side", "threshold", "model", "is_positive_rate", "base_rate_oos",
            "oos_auc", "wf_auc_median", "wf_auc_min", "wf_auc_max", "wf_n_windows",
            "top_decile_rate", "top_decile_lift", "n_oos_positive", "n_oos_total"]
    out = out[cols].sort_values(["symbol", "side", "threshold", "model"])
    out.to_csv(OUT / "severity_sweep.csv", index=False)
    print(out.to_string(index=False))

    # Plot: WF AUC median vs threshold per (symbol, side) for both models
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax, side in zip(axes, ("loss", "win")):
        for sym, base_color in (("TQQQ", "#2563eb"), ("SQQQ", "#dc2626")):
            d_tree = out[(out["symbol"] == sym) & (out["side"] == side) & (out["model"] == "tree")]
            d_logit = out[(out["symbol"] == sym) & (out["side"] == side) & (out["model"] == "l1_logit")]
            ax.plot(d_tree["threshold"], d_tree["wf_auc_median"], marker="o",
                    label=f"{sym} tree (WF median)", color=base_color, linewidth=1.5)
            ax.fill_between(d_tree["threshold"], d_tree["wf_auc_min"], d_tree["wf_auc_max"],
                            alpha=0.12, color=base_color)
            ax.plot(d_logit["threshold"], d_logit["wf_auc_median"], marker="s",
                    label=f"{sym} L1-logit (WF median)", color=base_color,
                    linewidth=1.5, linestyle="--")
        ax.axhline(0.5, color="black", linewidth=0.5, linestyle="--")
        ax.set_title(f"{side.upper()} side: WF AUC vs threshold\n(pnl_pct {'≤' if side=='loss' else '≥'} thr)")
        ax.set_xlabel("threshold (pnl_pct)")
        ax.set_ylabel("OOS AUC")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "severity_sweep_auc.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
