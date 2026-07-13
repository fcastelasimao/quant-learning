"""04_loss_region_models: trees + L1-logistic + GBM with walk-forward.

Curated 12 numeric features + 2 regime dummies. IS = entry_time year <= 2020.
OOS = 2021-2026.
- Depth-4 DecisionTreeClassifier on is_loser and on is_severe_loss
- L1-LogisticRegression on is_loser and on is_severe_loss
- Depth-3 GradientBoostingRegressor on pnl_pct
- Walk-forward (expanding window, yearly step) for all three model families
- Tree-leaf rule extraction with OOS-yearly evaluation + random-baseline comparison
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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _rule_naming import rule_description, rule_hash, rule_name  # noqa: E402


ROOT = Path(__file__).resolve().parent
CANON = ROOT.parent.parent / "full_history_canonical"
OUT = ROOT
SEED = 42

CURATED_NUMERIC = [
    "atr_pct", "RSI_entry", "BBP_entry",
    "dist_to_MA20", "dist_to_MA50", "dist_to_MA100",
    "MA20_D5", "MA50_D5", "MA100_D1",
    "log_volume_ratio", "bars_since_last_stop", "hour_of_entry",
]
REGIMES = ("bull", "chop_highvol", "sideways_lowvol")  # bull = reference

IS_END_YEAR = 2020
OOS_YEARS = [2021, 2022, 2023, 2024, 2025, 2026]
WF_TRAIN_END_YEARS = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]


def load(sym: str) -> pd.DataFrame:
    df = pd.read_csv(
        CANON / f"TRADES_{sym}_full_history.csv",
        parse_dates=["entry_time", "exit_time"],
    )
    df = df[df["regime_entry"].notna()].copy()
    df["year"] = df["entry_time"].dt.year
    return df


def build_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    work = df[CURATED_NUMERIC + ["regime_entry", "is_loser", "is_severe_loss",
                                  "pnl_pct", "year", "entry_time"]].copy()
    for col in CURATED_NUMERIC:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=CURATED_NUMERIC + ["is_loser", "pnl_pct"]).copy()
    for r in REGIMES[1:]:
        work[f"regime_{r}"] = (work["regime_entry"] == r).astype(int)
    feature_cols = CURATED_NUMERIC + [f"regime_{r}" for r in REGIMES[1:]]
    return work, feature_cols


# --------------------------------------------------------------------------- #
# Rule extraction from a fitted DecisionTreeClassifier
# --------------------------------------------------------------------------- #

def tree_leaves(tree: DecisionTreeClassifier, feature_names: list[str]) -> list[dict]:
    """Walk the fitted tree and emit one record per leaf.

    REQUIRES sklearn >= 1.5. In 1.5+ `tree_.value` stores class FRACTIONS;
    before 1.5 it stored COUNTS. This function reads fracs[1] as the positive-
    class fraction, which would be silently wrong on older sklearn. If you
    downgrade sklearn, also recompute the leaf precisions from the actual IS
    data (which the caller does anyway via apply_path).
    """
    t = tree.tree_
    leaves: list[dict] = []

    def recurse(node: int, conditions: list[tuple[str, str, float]]) -> None:
        if t.feature[node] == -2:
            fracs = t.value[node].ravel()
            pos_frac = float(fracs[1]) if len(fracs) > 1 else float("nan")
            leaves.append({
                "leaf_id": node,
                "conditions": list(conditions),
                "n_samples": int(t.n_node_samples[node]),
                "pos_class_fraction": pos_frac,
            })
            return
        feat = feature_names[t.feature[node]]
        thr = float(t.threshold[node])
        recurse(t.children_left[node], conditions + [(feat, "<=", thr)])
        recurse(t.children_right[node], conditions + [(feat, ">", thr)])

    recurse(0, [])
    return leaves


def pretty_path(conditions: list[tuple[str, str, float]]) -> str:
    return " AND ".join(f"{f} {op} {thr:.4g}" for f, op, thr in conditions)


def apply_path(df: pd.DataFrame, conditions: list[tuple[str, str, float]]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for feat, op, thr in conditions:
        col = df[feat]
        m = (col <= thr) if op == "<=" else (col > thr)
        mask &= m.fillna(False)
    return mask


# --------------------------------------------------------------------------- #
# Per-year rule evaluation
# --------------------------------------------------------------------------- #

def evaluate_rule_year(df: pd.DataFrame, mask: pd.Series, base_rate: float) -> dict:
    flagged = df.loc[mask]
    n = int(mask.sum())
    if n == 0:
        return {"n_flagged": 0, "trigger_rate": 0.0, "precision_loser_rate": np.nan,
                "lift_vs_baseline": np.nan, "skipped_loser_pnl_avoided": 0.0,
                "skipped_winner_pnl_sacrificed": 0.0, "net_pnl_pct_impact": 0.0}
    loser_rate = float(flagged["is_loser"].mean())
    skipped_loser_pnl = float(flagged.loc[flagged["is_loser"], "pnl_pct"].sum())
    skipped_winner_pnl = float(flagged.loc[~flagged["is_loser"], "pnl_pct"].sum())
    return {
        "n_flagged": n,
        "trigger_rate": float(mask.mean()),
        "precision_loser_rate": loser_rate,
        "lift_vs_baseline": loser_rate - base_rate,
        "skipped_loser_pnl_avoided": -skipped_loser_pnl,
        "skipped_winner_pnl_sacrificed": skipped_winner_pnl,
        "net_pnl_pct_impact": float(-flagged["pnl_pct"].sum()),
    }


def random_baseline_precision(df: pd.DataFrame, trigger_rate: float, rng: np.random.Generator, n_iters: int = 200) -> float:
    """Median loser-rate of random masks at the same trigger rate."""
    if trigger_rate <= 0 or len(df) == 0:
        return np.nan
    losers = df["is_loser"].values
    rates = []
    for _ in range(n_iters):
        mask = rng.random(len(df)) < trigger_rate
        if mask.sum() == 0:
            continue
        rates.append(float(losers[mask].mean()))
    return float(np.median(rates)) if rates else np.nan


# --------------------------------------------------------------------------- #
# Main analysis
# --------------------------------------------------------------------------- #

def run_symbol(sym: str) -> None:
    rng = np.random.default_rng(SEED)
    df, feature_cols = build_xy(load(sym))
    is_mask = df["year"] <= IS_END_YEAR
    df_is = df[is_mask].copy()
    df_oos = df[~is_mask].copy()
    print(f"{sym}: IS={len(df_is)}, OOS={len(df_oos)}, features={len(feature_cols)}")

    X_is = df_is[feature_cols].values
    X_oos = df_oos[feature_cols].values

    # -------------------- Tree on is_loser --------------------
    tree_loser = DecisionTreeClassifier(max_depth=4, min_samples_leaf=30, random_state=SEED)
    tree_loser.fit(X_is, df_is["is_loser"].astype(int))
    tree_loser_oos_auc = roc_auc_score(df_oos["is_loser"], tree_loser.predict_proba(X_oos)[:, 1])

    tree_severe = DecisionTreeClassifier(max_depth=4, min_samples_leaf=30, random_state=SEED)
    tree_severe.fit(X_is, df_is["is_severe_loss"].astype(int))
    tree_severe_oos_auc = roc_auc_score(df_oos["is_severe_loss"], tree_severe.predict_proba(X_oos)[:, 1])

    # -------------------- L1-logistic --------------------
    scaler = StandardScaler().fit(X_is)
    Xs_is = scaler.transform(X_is)
    Xs_oos = scaler.transform(X_oos)
    logit_loser = LogisticRegression(penalty="l1", solver="liblinear", C=0.1, random_state=SEED, max_iter=2000)
    logit_loser.fit(Xs_is, df_is["is_loser"].astype(int))
    logit_loser_oos_auc = roc_auc_score(df_oos["is_loser"], logit_loser.predict_proba(Xs_oos)[:, 1])

    logit_severe = LogisticRegression(penalty="l1", solver="liblinear", C=0.1, random_state=SEED, max_iter=2000)
    logit_severe.fit(Xs_is, df_is["is_severe_loss"].astype(int))
    logit_severe_oos_auc = roc_auc_score(df_oos["is_severe_loss"], logit_severe.predict_proba(Xs_oos)[:, 1])

    # -------------------- GBM on pnl_pct --------------------
    gbm = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=SEED, subsample=0.9,
    )
    gbm.fit(X_is, df_is["pnl_pct"].values)
    pnl_pred_oos = gbm.predict(X_oos)
    gbm_oos_r2 = r2_score(df_oos["pnl_pct"], pnl_pred_oos)
    gbm_oos_auc_vs_loser = roc_auc_score(df_oos["is_loser"], -pnl_pred_oos)
    gbm_oos_auc_vs_severe = roc_auc_score(df_oos["is_severe_loss"], -pnl_pred_oos)

    # -------------------- Outputs: model coefs / importances --------------------
    pd.DataFrame({
        "symbol": sym,
        "feature": feature_cols,
        "logit_loser_coef": logit_loser.coef_.ravel(),
        "logit_loser_abs": np.abs(logit_loser.coef_.ravel()),
        "logit_severe_coef": logit_severe.coef_.ravel(),
        "logit_severe_abs": np.abs(logit_severe.coef_.ravel()),
    }).sort_values("logit_loser_abs", ascending=False).to_csv(OUT / f"logistic_coefs_{sym}.csv", index=False)

    perm_loser = permutation_importance(tree_loser, X_oos, df_oos["is_loser"].astype(int),
                                        n_repeats=20, random_state=SEED, scoring="roc_auc")
    perm_severe = permutation_importance(tree_severe, X_oos, df_oos["is_severe_loss"].astype(int),
                                         n_repeats=20, random_state=SEED, scoring="roc_auc")
    perm_gbm = permutation_importance(gbm, X_oos, df_oos["pnl_pct"].values,
                                      n_repeats=20, random_state=SEED, scoring="neg_mean_squared_error")
    pd.DataFrame({
        "symbol": sym,
        "feature": feature_cols,
        "gbm_impurity_importance": gbm.feature_importances_,
        "gbm_oos_perm_importance": perm_gbm.importances_mean,
        "tree_loser_oos_perm_importance": perm_loser.importances_mean,
        "tree_severe_oos_perm_importance": perm_severe.importances_mean,
    }).sort_values("gbm_oos_perm_importance", ascending=False).to_csv(OUT / f"feature_importances_{sym}.csv", index=False)

    # -------------------- Tree leaves (rule candidates) --------------------
    leaf_rows = []
    rule_eval_rows = []
    for tree_name, tree, target_col in [
        ("tree_loser", tree_loser, "is_loser"),
        ("tree_severe", tree_severe, "is_severe_loss"),
    ]:
        leaves = tree_leaves(tree, feature_cols)
        for L in leaves:
            n_is_tree = L["n_samples"]
            path_str = pretty_path(L["conditions"])
            mask_is = apply_path(df_is, L["conditions"])
            mask_oos = apply_path(df_oos, L["conditions"])
            # Realized precision on IS, the discovery-side ground truth
            prec_is = float(df_is.loc[mask_is, target_col].mean()) if mask_is.sum() else 0.0
            n_is_realized = int(mask_is.sum())
            oos_eval = evaluate_rule_year(df_oos, mask_oos, base_rate=float(df_oos[target_col].mean()))

            is_candidate = (prec_is >= 0.65 and n_is_realized >= 30)
            r_name = rule_name(sym, target_col, L["conditions"])
            r_hash = rule_hash(L["conditions"])
            r_desc = rule_description(L["conditions"])
            leaf_rows.append({
                "symbol": sym, "tree": tree_name, "target": target_col,
                "leaf_id": L["leaf_id"],
                "rule_name": r_name,
                "rule_hash": r_hash,
                "n_is": n_is_realized,
                "precision_is": prec_is,
                "tree_pos_frac": L["pos_class_fraction"],
                "n_oos": int(mask_oos.sum()),
                "precision_oos": oos_eval["precision_loser_rate"],
                "lift_oos_vs_baseline": oos_eval["lift_vs_baseline"],
                "net_pnl_oos_pct": oos_eval["net_pnl_pct_impact"],
                "is_candidate_rule": is_candidate,
                "description": r_desc,
                "path": path_str,
            })

            if not is_candidate:
                continue

            # per-OOS-year evaluation of this rule
            for year in OOS_YEARS:
                year_df = df_oos[df_oos["year"] == year]
                if len(year_df) == 0:
                    continue
                mask_y = apply_path(year_df, L["conditions"])
                base = float(year_df[target_col].mean())
                eval_y = evaluate_rule_year(year_df, mask_y, base_rate=base)
                rand_p = random_baseline_precision(year_df, eval_y["trigger_rate"], rng) if eval_y["trigger_rate"] > 0 else np.nan
                rule_eval_rows.append({
                    "symbol": sym, "tree": tree_name, "target": target_col,
                    "leaf_id": L["leaf_id"],
                    "rule_name": r_name,
                    "rule_hash": r_hash,
                    "year": year,
                    "n_total": len(year_df),
                    **eval_y,
                    "random_baseline_precision": rand_p,
                    "precision_minus_random": (eval_y["precision_loser_rate"] - rand_p) if rand_p is not None and not np.isnan(rand_p) else np.nan,
                    "path": path_str,
                })

    pd.DataFrame(leaf_rows).to_csv(OUT / f"tree_leaves_{sym}.csv", index=False)
    pd.DataFrame(rule_eval_rows).to_csv(OUT / f"wf_rule_eval_{sym}.csv", index=False)

    # -------------------- Walk-forward model evaluation --------------------
    wf_rows = []
    for y_end in WF_TRAIN_END_YEARS:
        y_test = y_end + 1
        if y_test not in df["year"].unique():
            continue
        train = df[df["year"] <= y_end]
        test = df[df["year"] == y_test]
        if len(train) < 100 or len(test) < 20:
            continue
        Xtr = train[feature_cols].values
        Xte = test[feature_cols].values
        ytr_loser = train["is_loser"].astype(int).values
        yte_loser = test["is_loser"].astype(int).values
        ytr_severe = train["is_severe_loss"].astype(int).values
        yte_severe = test["is_severe_loss"].astype(int).values
        ytr_pnl = train["pnl_pct"].values
        yte_pnl = test["pnl_pct"].values

        # tree on is_loser
        t = DecisionTreeClassifier(max_depth=4, min_samples_leaf=30, random_state=SEED)
        t.fit(Xtr, ytr_loser)
        try:
            auc_tree_loser = roc_auc_score(yte_loser, t.predict_proba(Xte)[:, 1])
        except ValueError:
            auc_tree_loser = np.nan
        # tree on is_severe
        ts = DecisionTreeClassifier(max_depth=4, min_samples_leaf=30, random_state=SEED)
        ts.fit(Xtr, ytr_severe)
        try:
            auc_tree_severe = roc_auc_score(yte_severe, ts.predict_proba(Xte)[:, 1])
        except ValueError:
            auc_tree_severe = np.nan
        # L1 logistic
        sc = StandardScaler().fit(Xtr)
        l = LogisticRegression(penalty="l1", solver="liblinear", C=0.1, random_state=SEED, max_iter=2000)
        l.fit(sc.transform(Xtr), ytr_loser)
        try:
            auc_l1_loser = roc_auc_score(yte_loser, l.predict_proba(sc.transform(Xte))[:, 1])
        except ValueError:
            auc_l1_loser = np.nan
        ls = LogisticRegression(penalty="l1", solver="liblinear", C=0.1, random_state=SEED, max_iter=2000)
        ls.fit(sc.transform(Xtr), ytr_severe)
        try:
            auc_l1_severe = roc_auc_score(yte_severe, ls.predict_proba(sc.transform(Xte))[:, 1])
        except ValueError:
            auc_l1_severe = np.nan
        # GBM
        g = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=SEED, subsample=0.9)
        g.fit(Xtr, ytr_pnl)
        pnl_pred = g.predict(Xte)
        r2 = r2_score(yte_pnl, pnl_pred)
        try:
            auc_gbm_vs_loser = roc_auc_score(yte_loser, -pnl_pred)
        except ValueError:
            auc_gbm_vs_loser = np.nan

        # top-decile loser rate at -pnl_pred  (i.e. the 10% most pessimistic predictions)
        if len(test) >= 20:
            cutoff = np.quantile(pnl_pred, 0.1)
            bottom = test.loc[pnl_pred <= cutoff]
            top_dec_loser_rate = float(bottom["is_loser"].mean()) if len(bottom) else np.nan
            top_dec_mean_pnl = float(bottom["pnl_pct"].mean()) if len(bottom) else np.nan
        else:
            top_dec_loser_rate = np.nan
            top_dec_mean_pnl = np.nan

        wf_rows.append({
            "symbol": sym, "train_through_year": y_end, "test_year": y_test,
            "n_train": len(train), "n_test": len(test),
            "auc_tree_loser": auc_tree_loser,
            "auc_tree_severe": auc_tree_severe,
            "auc_l1_loser": auc_l1_loser,
            "auc_l1_severe": auc_l1_severe,
            "gbm_r2": r2,
            "auc_gbm_vs_loser": auc_gbm_vs_loser,
            "gbm_bottom_decile_loser_rate": top_dec_loser_rate,
            "gbm_bottom_decile_mean_pnl_pct": top_dec_mean_pnl,
            "baseline_loser_rate": float(test["is_loser"].mean()),
            "baseline_mean_pnl_pct": float(test["pnl_pct"].mean()),
        })

    wf_df = pd.DataFrame(wf_rows)
    wf_df.to_csv(OUT / f"wf_model_eval_{sym}.csv", index=False)

    # -------------------- Plots --------------------
    # 1. Tree visualizations
    for tname, tmod, tlabel in [
        ("tree_loser", tree_loser, "is_loser"),
        ("tree_severe", tree_severe, "is_severe_loss"),
    ]:
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(tmod, feature_names=feature_cols, class_names=["winner", tlabel],
                  filled=True, rounded=True, fontsize=8, ax=ax,
                  proportion=True, impurity=False)
        ax.set_title(f"{sym}: {tname} (target={tlabel}, max_depth=4, min_leaf=30)\n"
                     "Leaves filled by predicted-class proportion")
        fig.tight_layout()
        fig.savefig(OUT / f"tree_{sym}_{tname}.png", dpi=120)
        plt.close(fig)

    # 2. WF AUC line plot
    if not wf_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for col, label, color in [
            ("auc_tree_loser", "tree_loser", "#2563eb"),
            ("auc_tree_severe", "tree_severe", "#dc2626"),
            ("auc_l1_loser", "L1-logit_loser", "#7c3aed"),
            ("auc_l1_severe", "L1-logit_severe", "#16a34a"),
        ]:
            ax.plot(wf_df["test_year"], wf_df[col], marker="o", label=label, color=color)
        ax.axhline(0.5, color="black", linewidth=0.6, linestyle="--", label="random")
        ax.set_xlabel("Walk-forward test year (trained on [2015..year-1])")
        ax.set_ylabel("OOS AUC")
        ax.set_title(f"{sym}: walk-forward AUC across years by model family")
        ax.legend(loc="lower left")
        ax.set_ylim(0.35, 0.85)
        fig.tight_layout()
        fig.savefig(OUT / f"wf_auc_line_{sym}.png", dpi=140)
        plt.close(fig)

    # 3. Per-rule per-year net-pnl heatmap (candidate rules only)
    rule_df = pd.DataFrame(rule_eval_rows)
    if not rule_df.empty:
        pivot = rule_df.pivot_table(index="rule_name", columns="year",
                                    values="net_pnl_pct_impact", aggfunc="sum")
        if not pivot.empty:
            v = max(float(np.nanmax(np.abs(pivot.values))), 1.0)
            fig, ax = plt.subplots(figsize=(11, max(3.0, 0.5 * len(pivot))))
            im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=-v, vmax=v)
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=7)
            ax.set_xlabel("OOS year")
            ax.set_title(f"{sym}: candidate-rule net pnl_pct impact per year (green = skipping helped)")
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    val = pivot.values[i, j]
                    if pd.notna(val):
                        ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7)
            fig.colorbar(im, ax=ax, label="net pnl_pct (centered)")
            fig.tight_layout()
            fig.savefig(OUT / f"rule_per_year_heatmap_{sym}.png", dpi=140)
            plt.close(fig)

    # 4. L1-logistic OOS calibration curve (on is_severe_loss — the stronger target)
    p_oos = logit_severe.predict_proba(Xs_oos)[:, 1]
    df_oos_cal = pd.DataFrame({"p": p_oos, "y": df_oos["is_severe_loss"].astype(int).values})
    df_oos_cal["bin"] = pd.qcut(df_oos_cal["p"].rank(method="first"), 10, labels=False, duplicates="drop")
    cal = df_oos_cal.groupby("bin").agg(mean_pred=("p", "mean"), mean_actual=("y", "mean"), n=("y", "size")).reset_index()
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0, 1], [0, 1], color="black", linewidth=0.6, linestyle="--", label="perfect")
    ax.plot(cal["mean_pred"], cal["mean_actual"], marker="o", color="#dc2626", label="L1-logit severe (OOS)")
    for _, r in cal.iterrows():
        ax.annotate(f"n={int(r['n'])}", (r["mean_pred"], r["mean_actual"]), fontsize=7,
                    xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("predicted P(severe loss)")
    ax.set_ylabel("actual fraction severe")
    ax.set_title(f"{sym}: L1-logistic OOS calibration (is_severe_loss)")
    ax.legend(loc="upper left")
    ax.set_xlim(0, max(0.6, cal["mean_pred"].max() * 1.1))
    ax.set_ylim(0, max(0.6, cal["mean_actual"].max() * 1.1))
    fig.tight_layout()
    fig.savefig(OUT / f"calibration_logit_severe_{sym}.png", dpi=140)
    plt.close(fig)

    # -------------------- IS-fit headline AUCs --------------------
    headline = pd.DataFrame([{
        "symbol": sym,
        "tree_loser_oos_auc": tree_loser_oos_auc,
        "tree_severe_oos_auc": tree_severe_oos_auc,
        "logit_loser_oos_auc": logit_loser_oos_auc,
        "logit_severe_oos_auc": logit_severe_oos_auc,
        "gbm_oos_r2": gbm_oos_r2,
        "gbm_oos_auc_vs_loser": gbm_oos_auc_vs_loser,
        "gbm_oos_auc_vs_severe": gbm_oos_auc_vs_severe,
        "oos_baseline_loser_rate": float(df_oos["is_loser"].mean()),
    }])
    headline.to_csv(OUT / f"headline_model_perf_{sym}.csv", index=False)
    print(headline.to_string(index=False))


def main() -> None:
    for sym in ("TQQQ", "SQQQ"):
        run_symbol(sym)


if __name__ == "__main__":
    main()
