"""01_data_diagnostics: sample sizes, missingness, multicollinearity audit.

Restricts to rows where regime_entry is non-null.
"""
from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-quant")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform


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

FAMILY_MAP = {
    "MA20": "ma_raw_level", "MA50": "ma_raw_level", "MA100": "ma_raw_level",
    "high_water_mark_entry": "price_level_ref",
    "MA20_D1": "ma_slope", "MA20_D3": "ma_slope", "MA20_D5": "ma_slope",
    "MA50_D1": "ma_slope", "MA50_D3": "ma_slope", "MA50_D5": "ma_slope",
    "MA100_D1": "ma_slope",
    "dist_to_MA20": "ma_dist", "dist_to_MA50": "ma_dist", "dist_to_MA100": "ma_dist",
    "atr": "atr", "atr_pct": "atr", "atr_pct_roll_pctile_252": "atr",
    "RSI_entry": "rsi", "RSI_entry_roll_pctile_252": "rsi",
    "BBP_entry": "bbp", "BBP_entry_roll_pctile_252": "bbp",
    "volume_cur": "volume", "volume_ratio": "volume",
    "log_volume_ratio": "volume", "volume_ratio_roll_pctile_252": "volume",
    "bars_since_last_stop": "stop_history", "hour_of_entry": "session_clock",
    "is_bullish_c1c2": "candle_state",
}

CLUSTER_THRESHOLD = 0.85  # |Spearman| >= this -> same multicollinearity cluster


def load(sym: str) -> pd.DataFrame:
    df = pd.read_csv(
        CANON / f"TRADES_{sym}_full_history.csv",
        parse_dates=["entry_time", "exit_time"],
    )
    return df[df["regime_entry"].notna()].copy()


def sample_sizes(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for sym, df in dfs.items():
        for regime, g in df.groupby("regime_entry"):
            rows.append({"symbol": sym, "regime_entry": regime, "n": len(g),
                         "loser_rate": float(g["is_loser"].mean()),
                         "severe_loss_rate": float(g["is_severe_loss"].mean()) if "is_severe_loss" in g else np.nan,
                         "mean_pnl_pct": float(g["pnl_pct"].mean()),
                         "median_pnl_pct": float(g["pnl_pct"].median())})
        rows.append({"symbol": sym, "regime_entry": "_TOTAL_regime_labeled", "n": len(df),
                     "loser_rate": float(df["is_loser"].mean()),
                     "severe_loss_rate": float(df["is_severe_loss"].mean()) if "is_severe_loss" in df else np.nan,
                     "mean_pnl_pct": float(df["pnl_pct"].mean()),
                     "median_pnl_pct": float(df["pnl_pct"].median())})
    return pd.DataFrame(rows)


def year_sizes(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for sym, df in dfs.items():
        years = df["entry_time"].dt.year
        for year, g in df.groupby(years):
            rows.append({"symbol": sym, "year": int(year), "n": len(g),
                         "loser_rate": float(g["is_loser"].mean()),
                         "mean_pnl_pct": float(g["pnl_pct"].mean())})
    return pd.DataFrame(rows).sort_values(["symbol", "year"])


def regime_x_year(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for sym, df in dfs.items():
        years = df["entry_time"].dt.year
        for (year, regime), g in df.groupby([years, "regime_entry"]):
            rows.append({"symbol": sym, "year": int(year), "regime_entry": regime, "n": len(g)})
    return pd.DataFrame(rows).sort_values(["symbol", "year", "regime_entry"])


def missingness(dfs: dict[str, pd.DataFrame], features: list[str]) -> pd.DataFrame:
    rows = []
    for sym, df in dfs.items():
        for feat in features:
            if feat in df.columns:
                rows.append({
                    "symbol": sym, "feature": feat,
                    "missing_count": int(df[feat].isna().sum()),
                    "missing_pct": float(df[feat].isna().mean()),
                    "n_total": len(df),
                })
            else:
                rows.append({
                    "symbol": sym, "feature": feat,
                    "missing_count": -1, "missing_pct": np.nan,
                    "n_total": len(df),
                })
    return pd.DataFrame(rows)


def corr_matrix(df: pd.DataFrame, features: list[str], method: str) -> pd.DataFrame:
    sub = df[features].apply(pd.to_numeric, errors="coerce")
    return sub.corr(method=method)


def cluster_features(corr_abs: pd.DataFrame, threshold: float) -> dict[str, int]:
    dist = 1.0 - corr_abs.values
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    dist = np.clip(dist, 0.0, None)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    clusters = fcluster(Z, t=1.0 - threshold, criterion="distance")
    return dict(zip(corr_abs.index, clusters))


def spearman_with_pnl(df: pd.DataFrame, features: list[str]) -> dict[str, float]:
    out = {}
    for feat in features:
        valid = df[[feat, "pnl_pct"]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(valid) >= 30 and valid[feat].nunique() >= 2:
            rho = stats.spearmanr(valid[feat], valid["pnl_pct"]).statistic
            out[feat] = float(rho) if not np.isnan(rho) else 0.0
        else:
            out[feat] = 0.0
    return out


def pick_representative(members: list[str], spearman_pnl: dict[str, float]) -> str:
    return max(members, key=lambda f: abs(spearman_pnl.get(f, 0.0)))


def high_corr_pairs(corr: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    feats = list(corr.index)
    for i, a in enumerate(feats):
        for b in feats[i + 1:]:
            v = corr.loc[a, b]
            if not np.isnan(v) and abs(v) >= threshold:
                rows.append({"feature_a": a, "feature_b": b, "abs_corr": float(abs(v)), "corr": float(v)})
    return pd.DataFrame(rows).sort_values("abs_corr", ascending=False)


def plot_corr_heatmap(corr: pd.DataFrame, sym: str, out_dir: Path) -> None:
    """Spearman heatmap, features reordered by hierarchical clustering for block structure."""
    abs_c = corr.abs().fillna(0.0).values
    dist = 1.0 - abs_c
    np.fill_diagonal(dist, 0.0)
    dist = np.clip((dist + dist.T) / 2.0, 0.0, None)
    Z = linkage(squareform(dist, checks=False), method="average")
    order = leaves_list(Z)
    feats = list(corr.index)
    reord = [feats[i] for i in order]
    M = corr.loc[reord, reord].values

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(reord)))
    ax.set_yticks(range(len(reord)))
    ax.set_xticklabels(reord, rotation=90, fontsize=7)
    ax.set_yticklabels(reord, fontsize=7)
    ax.set_title(f"{sym}: Spearman correlation matrix (hierarchical-cluster ordered)\n"
                 "red = +ρ, blue = −ρ. Diagonal blocks of dark red = multicollinearity clusters")
    fig.colorbar(im, ax=ax, label="Spearman ρ", shrink=0.7)
    fig.tight_layout()
    fig.savefig(out_dir / f"corr_heatmap_{sym}.png", dpi=140)
    plt.close(fig)


def sanity_checks(dfs: dict[str, pd.DataFrame]) -> list[str]:
    msgs = []
    for sym, df in dfs.items():
        # pnl_pct is in percentage points -> single-digit means typical
        med = float(df["pnl_pct"].median())
        if abs(med) > 50:
            msgs.append(f"[WARN] {sym} median pnl_pct={med:.2f} -- looks scaled.")
        # all rows have regime
        if df["regime_entry"].isna().any():
            msgs.append(f"[ERROR] {sym} still has NaN regime rows after filter.")
        # regime values
        regimes = set(df["regime_entry"].unique())
        expected = {"bull", "chop_highvol", "sideways_lowvol"}
        if not regimes.issubset(expected):
            msgs.append(f"[WARN] {sym} unexpected regime values: {regimes - expected}")
    return msgs


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dfs = {sym: load(sym) for sym in ("TQQQ", "SQQQ")}

    msgs = sanity_checks(dfs)
    for m in msgs:
        print(m)

    sample_sizes(dfs).to_csv(OUT / "sample_sizes.csv", index=False)
    year_sizes(dfs).to_csv(OUT / "year_sizes.csv", index=False)
    regime_x_year(dfs).to_csv(OUT / "regime_x_year.csv", index=False)
    missingness(dfs, NUMERIC_FEATURES).to_csv(OUT / "missingness.csv", index=False)

    cluster_rows = []
    decorr_rows = []
    pair_rows = []

    for sym, df in dfs.items():
        present = [f for f in NUMERIC_FEATURES if f in df.columns]
        cp = corr_matrix(df, present, "pearson")
        cs = corr_matrix(df, present, "spearman")
        cp.to_csv(OUT / f"corr_pearson_{sym}.csv")
        cs.to_csv(OUT / f"corr_spearman_{sym}.csv")

        pairs = high_corr_pairs(cs, CLUSTER_THRESHOLD)
        pairs.insert(0, "symbol", sym)
        pair_rows.append(pairs)

        plot_corr_heatmap(cs, sym, OUT)

        clusters = cluster_features(cs.abs().fillna(0.0), CLUSTER_THRESHOLD)
        sp_pnl = spearman_with_pnl(df, present)

        clust_to_feats: dict[int, list[str]] = defaultdict(list)
        for feat, cid in clusters.items():
            clust_to_feats[cid].append(feat)

        for cid, feats in sorted(clust_to_feats.items()):
            rep = pick_representative(feats, sp_pnl)
            for feat in feats:
                cluster_rows.append({
                    "symbol": sym,
                    "cluster_id": int(cid),
                    "feature": feat,
                    "family": FAMILY_MAP.get(feat, "unknown"),
                    "spearman_with_pnl_pct": sp_pnl.get(feat, np.nan),
                    "is_cluster_representative": (feat == rep),
                    "cluster_size": len(feats),
                })
            decorr_rows.append({
                "symbol": sym,
                "cluster_id": int(cid),
                "representative": rep,
                "family": FAMILY_MAP.get(rep, "unknown"),
                "cluster_size": len(feats),
                "members": "|".join(sorted(feats)),
                "spearman_with_pnl_pct": sp_pnl.get(rep, np.nan),
            })

    pd.DataFrame(cluster_rows).sort_values(
        ["symbol", "cluster_id", "feature"]
    ).to_csv(OUT / "feature_clusters.csv", index=False)
    pd.DataFrame(decorr_rows).sort_values(
        ["symbol", "cluster_id"]
    ).to_csv(OUT / "decorrelated_feature_set.csv", index=False)
    pd.concat(pair_rows, ignore_index=True).to_csv(OUT / "high_corr_pairs.csv", index=False)

    print("Wrote diagnostics to", OUT)


if __name__ == "__main__":
    main()
