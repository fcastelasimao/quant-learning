"""03_multivariate_structure: PCA, PLS-DA, LDA on the curated decorrelated set.

Standardize the 12-feature curated set + regime dummies and look for loser
clustering in PC / PLS / LD space. Per symbol.
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
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
CANON = ROOT.parent.parent / "full_history_canonical"
OUT = ROOT

CURATED_NUMERIC = [
    "atr_pct", "RSI_entry", "BBP_entry",
    "dist_to_MA20", "dist_to_MA50", "dist_to_MA100",
    "MA20_D5", "MA50_D5", "MA100_D1",
    "log_volume_ratio", "bars_since_last_stop", "hour_of_entry",
]
REGIMES = ("bull", "chop_highvol", "sideways_lowvol")  # `bull` is reference


def load(sym: str) -> pd.DataFrame:
    df = pd.read_csv(
        CANON / f"TRADES_{sym}_full_history.csv",
        parse_dates=["entry_time", "exit_time"],
    )
    return df[df["regime_entry"].notna()].copy()


def build_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, list[str]]:
    feats = list(CURATED_NUMERIC)
    X = df[feats + ["regime_entry", "is_loser", "pnl_pct"]].apply(
        lambda c: pd.to_numeric(c, errors="coerce") if c.name != "regime_entry" else c
    )
    X = X.dropna(subset=feats + ["is_loser", "pnl_pct"]).copy()
    for r in REGIMES[1:]:
        X[f"regime_{r}"] = (X["regime_entry"] == r).astype(int)
    feature_cols = feats + [f"regime_{r}" for r in REGIMES[1:]]
    y_loser = X["is_loser"].astype(int)
    y_pnl = X["pnl_pct"]
    return X[feature_cols], y_loser, y_pnl, X["regime_entry"], feature_cols


def run_pca(Xs: np.ndarray, feature_cols: list[str], sym: str) -> tuple[PCA, pd.DataFrame, pd.DataFrame]:
    pca = PCA(n_components=min(len(feature_cols), Xs.shape[1]))
    pca.fit(Xs)
    var = pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
        "var_explained": pca.explained_variance_ratio_,
        "var_explained_cum": np.cumsum(pca.explained_variance_ratio_),
    })
    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_cols,
        columns=[f"PC{i+1}" for i in range(pca.components_.shape[0])],
    ).reset_index().rename(columns={"index": "feature"})
    var.insert(0, "symbol", sym)
    loadings.insert(0, "symbol", sym)
    return pca, var, loadings


def run_pls(Xs: np.ndarray, y: np.ndarray, feature_cols: list[str], sym: str, n_components: int = 3):
    n_components = min(n_components, Xs.shape[1])
    pls = PLSRegression(n_components=n_components, scale=False)
    pls.fit(Xs, y.astype(float))
    loadings = pd.DataFrame(
        pls.x_loadings_,
        index=feature_cols,
        columns=[f"PLS{i+1}" for i in range(n_components)],
    ).reset_index().rename(columns={"index": "feature"})
    loadings.insert(0, "symbol", sym)
    return pls, loadings


def run_lda(Xs: np.ndarray, y: np.ndarray, feature_cols: list[str], sym: str):
    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(Xs, y)
    coefs = pd.DataFrame({
        "symbol": sym,
        "feature": feature_cols,
        "lda_coef": lda.coef_.ravel(),
        "lda_coef_abs": np.abs(lda.coef_.ravel()),
    }).sort_values("lda_coef_abs", ascending=False)
    return lda, coefs


def plot_2panel(pca, pls, Xs, y_loser, sym: str) -> Path:
    pc_scores = pca.transform(Xs)
    pls_scores = pls.transform(Xs)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, scores, name, comps in (
        (axes[0], pc_scores, "PCA", ("PC1", "PC2")),
        (axes[1], pls_scores, "PLS-DA", ("PLS1", "PLS2")),
    ):
        win = y_loser.values == 0
        lose = y_loser.values == 1
        ax.scatter(scores[win, 0], scores[win, 1], s=10, alpha=0.35, color="#2563eb", label=f"winner (n={int(win.sum())})")
        ax.scatter(scores[lose, 0], scores[lose, 1], s=10, alpha=0.35, color="#dc2626", label=f"loser (n={int(lose.sum())})")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel(comps[0])
        ax.set_ylabel(comps[1])
        ax.set_title(f"{sym} {name}: winners vs losers")
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    path = OUT / f"projection_{sym}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def summarize_class_separation(pls_scores: np.ndarray, lda_scores: np.ndarray, y: np.ndarray) -> dict:
    pls1_w = pls_scores[y == 0, 0]
    pls1_l = pls_scores[y == 1, 0]
    pls2_w = pls_scores[y == 0, 1]
    pls2_l = pls_scores[y == 1, 1]
    lda_w = lda_scores[y == 0]
    lda_l = lda_scores[y == 1]

    def std_diff(a, b):
        pool = np.std(np.concatenate([a, b]))
        return float((np.mean(a) - np.mean(b)) / pool) if pool > 0 else np.nan

    return {
        "pls1_std_diff": std_diff(pls1_l, pls1_w),
        "pls2_std_diff": std_diff(pls2_l, pls2_w),
        "lda_std_diff": std_diff(lda_l, lda_w),
        "n_winner": int((y == 0).sum()),
        "n_loser": int((y == 1).sum()),
    }


def main() -> None:
    summary_rows = []
    for sym in ("TQQQ", "SQQQ"):
        df = load(sym)
        X, y_loser, y_pnl, regime_label, feature_cols = build_xy(df)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X.values)

        pca, var, pca_loadings = run_pca(Xs, feature_cols, sym)
        pca_loadings.to_csv(OUT / f"pca_loadings_{sym}.csv", index=False)
        var.to_csv(OUT / f"pca_variance_explained_{sym}.csv", index=False)

        pls, pls_loadings = run_pls(Xs, y_loser.values, feature_cols, sym, n_components=3)
        pls_loadings.to_csv(OUT / f"pls_loadings_{sym}.csv", index=False)

        lda, lda_coefs = run_lda(Xs, y_loser.values, feature_cols, sym)
        lda_coefs.to_csv(OUT / f"lda_coefs_{sym}.csv", index=False)

        plot_2panel(pca, pls, Xs, y_loser, sym)

        sep = summarize_class_separation(
            pls.transform(Xs), lda.transform(Xs).ravel(), y_loser.values
        )
        sep["symbol"] = sym
        summary_rows.append(sep)
        print(f"{sym}: PC1 var = {pca.explained_variance_ratio_[0]:.3f}, "
              f"PC1+2 cum = {pca.explained_variance_ratio_[:2].sum():.3f}, "
              f"PLS1 sep = {sep['pls1_std_diff']:.3f}, LDA sep = {sep['lda_std_diff']:.3f}")

    pd.DataFrame(summary_rows)[
        ["symbol", "n_winner", "n_loser", "pls1_std_diff", "pls2_std_diff", "lda_std_diff"]
    ].to_csv(OUT / "class_separation_summary.csv", index=False)


if __name__ == "__main__":
    main()
