"""Item 06 tree overfitting diagnostic.

Walk-forward IS vs OOS AUC comparison for depth-4 tree vs L1-logit on
is_severe_loss. Each fold trains on years < Y, evaluates on year Y, and
records both train and test AUC so the generalisation gap is visible.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from build_06_context_enrichment import (
    CONTEXT_FEATURES,
    CURATED_NUMERIC,
    SEED,
    build_xy,
)

OUT = HERE
SYMBOLS = ["TQQQ", "SQQQ"]
TARGET = "is_severe_loss"
MIN_TRAIN = 100
MIN_POSITIVES = 10


def load_enriched(sym: str) -> pd.DataFrame:
    path = HERE / f"enriched_trades_{sym}.csv"
    df = pd.read_csv(path, parse_dates=["entry_time"])
    return df


def walkforward_is_oos(df: pd.DataFrame, feature_cols: list[str]) -> list[dict]:
    sub = df.dropna(subset=feature_cols + [TARGET]).copy()
    sub["year"] = sub["entry_time"].dt.year
    years = sorted(sub["year"].unique())

    records = []
    for y in years:
        train = sub[sub["year"] < y]
        test = sub[sub["year"] == y]
        if len(train) < MIN_TRAIN or len(test) == 0:
            continue
        if train[TARGET].sum() < MIN_POSITIVES:
            continue

        X_train = train[feature_cols].values
        y_train = train[TARGET].astype(int).values
        X_test = test[feature_cols].values
        y_test = test[TARGET].astype(int).values

        # Tree
        tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=30, random_state=SEED)
        tree.fit(X_train, y_train)
        auc_tree_train = roc_auc_score(y_train, tree.predict_proba(X_train)[:, 1])
        auc_tree_test = roc_auc_score(y_test, tree.predict_proba(X_test)[:, 1])

        # L1-logit
        sc = StandardScaler().fit(X_train)
        logit = LogisticRegression(penalty="l1", solver="liblinear", C=0.1,
                                    random_state=SEED, max_iter=2000)
        logit.fit(sc.transform(X_train), y_train)
        auc_l1_train = roc_auc_score(y_train, logit.predict_proba(sc.transform(X_train))[:, 1])
        auc_l1_test = roc_auc_score(y_test, logit.predict_proba(sc.transform(X_test))[:, 1])

        records.append({"year": y, "model": "tree",
                        "auc_train": auc_tree_train, "auc_test": auc_tree_test,
                        "gap": auc_tree_train - auc_tree_test})
        records.append({"year": y, "model": "l1_logit",
                        "auc_train": auc_l1_train, "auc_test": auc_l1_test,
                        "gap": auc_l1_train - auc_l1_test})
    return records


def main() -> None:
    all_records = []

    for sym in SYMBOLS:
        print(f"{sym}: loading enriched trades...")
        df = load_enriched(sym)
        df, feature_cols = build_xy(df, extra=CONTEXT_FEATURES)
        print(f"  {sym}: {len(df)} rows, {len(feature_cols)} features")

        records = walkforward_is_oos(df, feature_cols)
        for r in records:
            r["symbol"] = sym
        all_records.extend(records)

    detail = pd.DataFrame(all_records)[["symbol", "year", "model", "auc_train", "auc_test", "gap"]]
    detail.to_csv(OUT / "overfit_diagnostic.csv", index=False)
    print("\nPer-fold detail saved.")

    summary_rows = []
    for sym in SYMBOLS:
        for model in ["tree", "l1_logit"]:
            sub = detail[(detail["symbol"] == sym) & (detail["model"] == model)]
            summary_rows.append({
                "symbol": sym,
                "model": model,
                "mean_auc_train": round(sub["auc_train"].mean(), 3),
                "mean_auc_test": round(sub["auc_test"].mean(), 3),
                "mean_gap": round(sub["gap"].mean(), 3),
                "std_gap": round(sub["gap"].std(), 3),
                "max_gap": round(sub["gap"].max(), 3),
            })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "overfit_summary.csv", index=False)
    print("\nSummary:")
    print(summary.to_string(index=False))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    colors = {"tree": "tab:orange", "l1_logit": "tab:blue"}
    for ax, sym in zip(axes, SYMBOLS):
        sub = detail[detail["symbol"] == sym]
        for model, grp in sub.groupby("model"):
            grp = grp.sort_values("year")
            c = colors[model]
            ax.plot(grp["year"], grp["auc_train"], linestyle="--", marker="o",
                    color=c, label=f"{model} IS")
            ax.plot(grp["year"], grp["auc_test"], linestyle="-", marker="s",
                    color=c, alpha=0.7, label=f"{model} OOS")
        ax.set_title(sym)
        ax.set_xlabel("Year (test fold)")
        ax.set_ylabel("ROC-AUC")
        ax.legend(fontsize=8)
        ax.axhline(0.5, color="gray", linewidth=0.8, linestyle=":")
    fig.suptitle("Item 06: IS vs OOS AUC per fold — tree vs L1-logit (is_severe_loss)")
    fig.tight_layout()
    fig.savefig(OUT / "overfit_diagnostic.png", dpi=120)
    print("\nPlot saved to overfit_diagnostic.png")

    # Write findings markdown
    md_lines = [
        "# Item 06 — Tree overfitting diagnostic\n",
        "Walk-forward IS vs OOS AUC for depth-4 tree and L1-logit on `is_severe_loss`.",
        "Each fold trains on years < Y, evaluates on year Y.",
        "A large mean IS–OOS gap (mean_gap) relative to L1-logit indicates overfitting.\n",
        "## Summary table\n",
        summary.to_string(index=False),
        "\n## Interpretation\n",
    ]
    tree_tqqq = summary[(summary["symbol"] == "TQQQ") & (summary["model"] == "tree")].iloc[0]
    l1_tqqq   = summary[(summary["symbol"] == "TQQQ") & (summary["model"] == "l1_logit")].iloc[0]
    tree_sqqq = summary[(summary["symbol"] == "SQQQ") & (summary["model"] == "tree")].iloc[0]
    l1_sqqq   = summary[(summary["symbol"] == "SQQQ") & (summary["model"] == "l1_logit")].iloc[0]

    for sym, t, l in [("TQQQ", tree_tqqq, l1_tqqq), ("SQQQ", tree_sqqq, l1_sqqq)]:
        verdict = "**overfitting confirmed**" if t["mean_gap"] > 2 * l["mean_gap"] else "gap is modest"
        md_lines.append(
            f"- **{sym}**: tree mean gap = {t['mean_gap']:.3f} vs L1 mean gap = {l['mean_gap']:.3f} — {verdict}."
        )

    md_lines += [
        "\n## Artifacts\n",
        "| file | content |",
        "|---|---|",
        "| `overfit_diagnostic.csv` | Per-fold IS and OOS AUC for both models |",
        "| `overfit_summary.csv` | Mean/std/max gap per symbol × model |",
        "| `overfit_diagnostic.png` | IS vs OOS AUC line plot per fold |",
        "| `findings_06_tree_overfit.md` | This note |",
    ]

    (OUT / "findings_06_tree_overfit.md").write_text("\n".join(md_lines))
    print("Findings written to findings_06_tree_overfit.md")


if __name__ == "__main__":
    main()
