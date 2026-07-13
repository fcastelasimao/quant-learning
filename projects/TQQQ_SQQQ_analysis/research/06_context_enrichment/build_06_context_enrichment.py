"""06_context_enrichment: cross-asset daily context features at trade entry.

Depends on:
  - full_history_canonical/TRADES_<SYM>_full_history.csv  (the canonical trades)
  - Shared QuantFinance data store (DB_<TKR>_historical_data.db)
    for TKR in {QQQ, SPY, ^VIX, ^VIX3M, ^TNX, ^IRX, HYG, LQD}. These FMP DBs
    live OUTSIDE this repo; refresh with `quantcore-ingest --intervals 1d`.

Produces enriched_trades_<sym>.csv which several downstream items consume
(11, 14, 16, 17).


For each trade, looks up the *prior business day's* values from the FMP daily
DBs (resolved via quantcore.config.data_dir()) and computes:

  QQQ:  RSI_14, dist_to_MA20/50/200, realized_vol_20d, dist_to_20d_high,
        50d_return, 50d_return_pctile_252 (bubble proxy), drawdown_5d,
        drawdown_60d, gap_overnight
  SPY:  RSI_14, dist_to_MA50
  VIX:  level, 5d_change, pctile_252d, term_structure (VIX/VIX3M)
  Credit:  HYG_LQD_ratio, HYG_5d_change
  Rates:   yield_curve_slope (TNX - IRX), TNX_5d_change

Joins to the regime-labeled canonical trades, saves an enriched CSV per symbol,
then re-fits the depth-4 tree + L1-logistic on the curated 12 + ~17 new context
features to see whether OOS AUC moves.
"""
from __future__ import annotations

import os
import sqlite3
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
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _rule_naming import rule_description, rule_hash, rule_name  # noqa: E402

from quantcore import config as _qc_config

ROOT = Path(__file__).resolve().parent
PROJ = ROOT.parent.parent
CANON = PROJ / "full_history_canonical"
DATA_DIR = _qc_config.data_dir()
OUT = ROOT
SEED = 42
IS_END_YEAR = 2020

# Curated base set (item 01)
CURATED_NUMERIC = [
    "atr_pct", "RSI_entry", "BBP_entry",
    "dist_to_MA20", "dist_to_MA50", "dist_to_MA100",
    "MA20_D5", "MA50_D5", "MA100_D1",
    "log_volume_ratio", "bars_since_last_stop", "hour_of_entry",
]

CONTEXT_FEATURES = [
    "QQQ_RSI_14", "QQQ_dist_MA20", "QQQ_dist_MA50", "QQQ_dist_MA200",
    "QQQ_realized_vol_20d", "QQQ_dist_high_20d", "QQQ_50d_return",
    "QQQ_50d_return_pctile_252", "QQQ_drawdown_5d", "QQQ_drawdown_60d",
    "QQQ_gap_overnight",
    "SPY_RSI_14", "SPY_dist_MA50",
    "VIX_level", "VIX_5d_change", "VIX_pctile_252d", "VIX_term_structure",
    "HYG_LQD_ratio", "HYG_5d_change",
    "yield_curve_slope", "TNX_5d_change",
]


# --------------------------------------------------------------------------- #
# Daily-bar loaders and feature builders
# --------------------------------------------------------------------------- #

def load_daily(symbol: str) -> pd.DataFrame:
    db_path = DATA_DIR / f"DB_{symbol}_historical_data.db"
    if not db_path.exists():
        raise FileNotFoundError(f"FMP DB missing: {db_path}")
    conn = sqlite3.connect(f"{db_path.as_uri()}?mode=ro&immutable=1", uri=True)
    df = pd.read_sql(
        "SELECT et_datetime, open, high, low, close, "
        "adj_close, volume FROM candles_1d ORDER BY ts",
        conn,
    )
    conn.close()
    df["date"] = pd.to_datetime(df["et_datetime"]).dt.normalize()
    # Use adj_close where available (split/div-adjusted) for return-based features.
    df["px"] = df["adj_close"].fillna(df["close"])
    return df[["date", "open", "high", "low", "close", "px", "volume"]].sort_values("date").reset_index(drop=True)


def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(n, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).rolling(n, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def realized_vol(series: pd.Series, n: int = 20) -> pd.Series:
    return series.pct_change().rolling(n, min_periods=n).std() * np.sqrt(252)


def rolling_pctile(series: pd.Series, n: int = 252) -> pd.Series:
    return series.rolling(n, min_periods=max(60, n // 4)).rank(pct=True)


def build_context_daily() -> pd.DataFrame:
    """Returns a per-date dataframe of all context features, indexed by date."""
    qqq = load_daily("QQQ")
    spy = load_daily("SPY")
    vix = load_daily("^VIX")
    vix3m = load_daily("^VIX3M")
    tnx = load_daily("^TNX")
    irx = load_daily("^IRX")
    hyg = load_daily("HYG")
    lqd = load_daily("LQD")

    ctx = pd.DataFrame({"date": qqq["date"]})
    # QQQ features
    ctx["QQQ_RSI_14"] = rsi(qqq["px"], 14).values
    ctx["QQQ_dist_MA20"] = (qqq["px"] / qqq["px"].rolling(20).mean() - 1).values
    ctx["QQQ_dist_MA50"] = (qqq["px"] / qqq["px"].rolling(50).mean() - 1).values
    ctx["QQQ_dist_MA200"] = (qqq["px"] / qqq["px"].rolling(200).mean() - 1).values
    ctx["QQQ_realized_vol_20d"] = realized_vol(qqq["px"], 20).values
    ctx["QQQ_dist_high_20d"] = (qqq["px"] / qqq["px"].rolling(20).max() - 1).values
    ctx["QQQ_50d_return"] = (qqq["px"].pct_change(50)).values
    ctx["QQQ_50d_return_pctile_252"] = rolling_pctile(ctx["QQQ_50d_return"], 252).values
    # Drawdowns: distance from rolling-max
    ctx["QQQ_drawdown_5d"] = (qqq["px"] / qqq["px"].rolling(5).max() - 1).values
    ctx["QQQ_drawdown_60d"] = (qqq["px"] / qqq["px"].rolling(60).max() - 1).values
    ctx["QQQ_gap_overnight"] = (qqq["open"] / qqq["close"].shift(1) - 1).values

    # SPY
    ctx = ctx.merge(spy[["date", "px"]].rename(columns={"px": "spy_px"}), on="date", how="left")
    ctx["SPY_RSI_14"] = rsi(ctx["spy_px"], 14).values
    ctx["SPY_dist_MA50"] = (ctx["spy_px"] / ctx["spy_px"].rolling(50).mean() - 1).values

    # VIX + term structure
    vix_df = vix[["date", "close"]].rename(columns={"close": "vix_lvl"})
    vix3m_df = vix3m[["date", "close"]].rename(columns={"close": "vix3m_lvl"})
    ctx = ctx.merge(vix_df, on="date", how="left").merge(vix3m_df, on="date", how="left")
    ctx["VIX_level"] = ctx["vix_lvl"]
    ctx["VIX_5d_change"] = ctx["vix_lvl"].pct_change(5)
    ctx["VIX_pctile_252d"] = rolling_pctile(ctx["vix_lvl"], 252)
    ctx["VIX_term_structure"] = ctx["vix_lvl"] / ctx["vix3m_lvl"]

    # Credit
    hyg_df = hyg[["date", "px"]].rename(columns={"px": "hyg_px"})
    lqd_df = lqd[["date", "px"]].rename(columns={"px": "lqd_px"})
    ctx = ctx.merge(hyg_df, on="date", how="left").merge(lqd_df, on="date", how="left")
    ctx["HYG_LQD_ratio"] = ctx["hyg_px"] / ctx["lqd_px"]
    ctx["HYG_5d_change"] = ctx["hyg_px"].pct_change(5)

    # Yield curve
    tnx_df = tnx[["date", "close"]].rename(columns={"close": "tnx"})
    irx_df = irx[["date", "close"]].rename(columns={"close": "irx"})
    ctx = ctx.merge(tnx_df, on="date", how="left").merge(irx_df, on="date", how="left")
    ctx["yield_curve_slope"] = ctx["tnx"] - ctx["irx"]
    ctx["TNX_5d_change"] = ctx["tnx"].diff(5)

    return ctx[["date"] + CONTEXT_FEATURES].copy()


def join_to_trades(symbol: str, ctx: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(CANON / f"TRADES_{symbol}_full_history.csv",
                     parse_dates=["entry_time", "exit_time"])
    df = df[df["regime_entry"].notna()].copy()
    df["entry_date"] = df["entry_time"].dt.normalize()
    # Join to the strictly PRIOR business day's context. Daily bars are complete
    # only after the session closes, so same-date context would leak information
    # for intraday entries.
    sorted_ctx = ctx.sort_values("date").reset_index(drop=True)
    df = df.sort_values("entry_date").reset_index(drop=True)
    merged = pd.merge_asof(
        df,
        sorted_ctx,
        left_on="entry_date",
        right_on="date",
        direction="backward",
        allow_exact_matches=False,
    )
    return merged


# --------------------------------------------------------------------------- #
# Modeling
# --------------------------------------------------------------------------- #

def build_xy(df: pd.DataFrame, extra: list[str]) -> tuple[pd.DataFrame, list[str]]:
    work = df.copy()
    for c in CURATED_NUMERIC + extra:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work["year"] = work["entry_time"].dt.year
    for r in ("chop_highvol", "sideways_lowvol"):
        work[f"regime_{r}"] = (work["regime_entry"] == r).astype(int)
    feature_cols = CURATED_NUMERIC + [f"regime_{r}" for r in ("chop_highvol", "sideways_lowvol")] + extra
    work = work.dropna(subset=feature_cols + ["is_loser", "is_severe_loss", "pnl_pct"])
    return work, feature_cols


def fit_compare(df_is: pd.DataFrame, df_oos: pd.DataFrame, feature_cols: list[str], label: str) -> dict:
    X_is = df_is[feature_cols].values
    X_oos = df_oos[feature_cols].values

    tree_severe = DecisionTreeClassifier(max_depth=4, min_samples_leaf=30, random_state=SEED)
    tree_severe.fit(X_is, df_is["is_severe_loss"].astype(int))
    auc_t_severe = roc_auc_score(df_oos["is_severe_loss"], tree_severe.predict_proba(X_oos)[:, 1])

    tree_loser = DecisionTreeClassifier(max_depth=4, min_samples_leaf=30, random_state=SEED)
    tree_loser.fit(X_is, df_is["is_loser"].astype(int))
    auc_t_loser = roc_auc_score(df_oos["is_loser"], tree_loser.predict_proba(X_oos)[:, 1])

    sc = StandardScaler().fit(X_is)
    logit_severe = LogisticRegression(penalty="l1", solver="liblinear", C=0.1, random_state=SEED, max_iter=2000)
    logit_severe.fit(sc.transform(X_is), df_is["is_severe_loss"].astype(int))
    auc_l_severe = roc_auc_score(df_oos["is_severe_loss"], logit_severe.predict_proba(sc.transform(X_oos))[:, 1])

    return {
        "feature_set": label,
        "n_features": len(feature_cols),
        "auc_tree_severe": float(auc_t_severe),
        "auc_tree_loser": float(auc_t_loser),
        "auc_l1_severe": float(auc_l_severe),
        "tree_severe": tree_severe,
        "tree_loser": tree_loser,
        "logit_severe": logit_severe,
    }


def permutation_importance_oos(df: pd.DataFrame, feature_cols: list[str],
                                target_col: str = "is_severe_loss",
                                n_repeats: int = 10, seed: int = SEED) -> pd.DataFrame:
    """Walk-forward permutation importance via L1-logit.

    For each WF year Y (where sufficient training data exists), fits L1-logit
    on [start..Y-1], evaluates baseline OOS AUC on year Y, then shuffles each
    feature n_repeats times and records the AUC drop. Averages drops across folds.

    Positive mean_auc_delta = shuffling this feature hurts AUC = feature is useful.
    Returns DataFrame sorted by mean_auc_delta descending (rank 1 = most important).
    """
    rng = np.random.default_rng(seed)
    years = sorted(df["year"].dropna().astype(int).unique())
    fold_deltas: dict[str, list[float]] = {f: [] for f in feature_cols}

    for y in years:
        train = df[df["year"] < y].dropna(subset=feature_cols + [target_col])
        test  = df[df["year"] == y].dropna(subset=feature_cols + [target_col])
        if len(train) < 100 or int(train[target_col].sum()) < 10:
            continue
        if len(test) < 10 or int(test[target_col].sum()) < 2:
            continue

        sc = StandardScaler().fit(train[feature_cols].values)
        X_tr = sc.transform(train[feature_cols].values)
        X_te = sc.transform(test[feature_cols].values)
        y_tr = train[target_col].astype(int).values
        y_te = test[target_col].astype(int).values

        mdl = LogisticRegression(penalty="l1", solver="liblinear", C=0.1,
                                  random_state=seed, max_iter=2000)
        mdl.fit(X_tr, y_tr)
        try:
            baseline_auc = roc_auc_score(y_te, mdl.predict_proba(X_te)[:, 1])
        except Exception:
            continue

        for fi, feat in enumerate(feature_cols):
            deltas: list[float] = []
            for _ in range(n_repeats):
                X_perm = X_te.copy()
                X_perm[:, fi] = rng.permutation(X_perm[:, fi])
                try:
                    perm_auc = roc_auc_score(y_te, mdl.predict_proba(X_perm)[:, 1])
                    deltas.append(baseline_auc - perm_auc)
                except Exception:
                    pass
            if deltas:
                fold_deltas[feat].append(float(np.mean(deltas)))

    rows = []
    for feat in feature_cols:
        vals = fold_deltas[feat]
        rows.append({
            "feature": feat,
            "mean_auc_delta": float(np.mean(vals)) if vals else 0.0,
            "std_auc_delta": float(np.std(vals)) if len(vals) > 1 else 0.0,
            "n_folds": len(vals),
        })
    result = (pd.DataFrame(rows)
              .sort_values("mean_auc_delta", ascending=False)
              .reset_index(drop=True))
    result["rank"] = result.index + 1
    return result


def prune_and_refit(df_is: pd.DataFrame, df_oos: pd.DataFrame,
                     feature_cols: list[str], importance: pd.DataFrame,
                     target_col: str = "is_severe_loss") -> dict:
    """Refit L1-logit keeping only features with mean_auc_delta > 0. Return comparison."""
    kept = importance[importance["mean_auc_delta"] > 0]["feature"].tolist()
    if not kept:
        kept = feature_cols[:5]  # fallback: at least top-5

    sc_full = StandardScaler().fit(df_is[feature_cols].values)
    mdl_full = LogisticRegression(penalty="l1", solver="liblinear", C=0.1,
                                   random_state=SEED, max_iter=2000)
    mdl_full.fit(sc_full.transform(df_is[feature_cols].values),
                  df_is[target_col].astype(int))
    full_auc = roc_auc_score(
        df_oos[target_col].astype(int),
        mdl_full.predict_proba(sc_full.transform(df_oos[feature_cols].values))[:, 1])

    sc_pruned = StandardScaler().fit(df_is[kept].values)
    mdl_pruned = LogisticRegression(penalty="l1", solver="liblinear", C=0.1,
                                     random_state=SEED, max_iter=2000)
    mdl_pruned.fit(sc_pruned.transform(df_is[kept].values),
                   df_is[target_col].astype(int))
    pruned_auc = roc_auc_score(
        df_oos[target_col].astype(int),
        mdl_pruned.predict_proba(sc_pruned.transform(df_oos[kept].values))[:, 1])

    return {
        "pruned_features": kept,
        "n_kept": len(kept),
        "n_total": len(feature_cols),
        "full_auc": float(full_auc),
        "pruned_auc": float(pruned_auc),
        "delta_auc": float(pruned_auc - full_auc),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Loading FMP daily DBs and computing context features...")
    ctx = build_context_daily()
    ctx_n_complete = int(ctx[CONTEXT_FEATURES].notna().all(axis=1).sum())
    print(f"  context daily rows = {len(ctx)}, fully complete = {ctx_n_complete}")
    ctx.to_csv(OUT / "context_daily_features.csv", index=False)

    headline = []
    pruned_rows: list[dict] = []
    for sym in ("TQQQ", "SQQQ"):
        df = join_to_trades(sym, ctx)
        coverage = df[CONTEXT_FEATURES].notna().all(axis=1).mean()
        print(f"{sym}: trades with full context coverage = {coverage:.1%}")
        df.to_csv(OUT / f"enriched_trades_{sym}.csv", index=False)

        # Baseline model on curated set only
        base_df, base_cols = build_xy(df, extra=[])
        base_is = base_df[base_df["year"] <= IS_END_YEAR]
        base_oos = base_df[base_df["year"] > IS_END_YEAR]
        base_res = fit_compare(base_is, base_oos, base_cols, "curated_12_only")

        # Enriched: curated + context
        enr_df, enr_cols = build_xy(df, extra=CONTEXT_FEATURES)
        enr_is = enr_df[enr_df["year"] <= IS_END_YEAR]
        enr_oos = enr_df[enr_df["year"] > IS_END_YEAR]
        enr_res = fit_compare(enr_is, enr_oos, enr_cols, "curated_plus_context")

        # Walk-forward permutation importance on enriched feature set
        print(f"  {sym}: computing walk-forward permutation importance ({len(enr_cols)} features)…")
        importance = permutation_importance_oos(enr_df, enr_cols, target_col="is_severe_loss")
        importance["symbol"] = sym
        importance.to_csv(OUT / f"permutation_importance_{sym}.csv", index=False)
        print(f"  {sym} top-5 by mean AUC delta:")
        print(importance.head(5)[["rank", "feature", "mean_auc_delta", "std_auc_delta"]].to_string(index=False))

        # Prune and refit
        pruned = prune_and_refit(enr_is, enr_oos, enr_cols, importance)
        pruned["symbol"] = sym
        pd.DataFrame({"feature": pruned["pruned_features"], "symbol": sym}).to_csv(
            OUT / f"pruned_feature_set_{sym}.csv", index=False)
        pruned_rows.append({k: v for k, v in pruned.items() if k != "pruned_features"})
        print(f"  {sym}: pruned {pruned['n_total']} → {pruned['n_kept']} features, "
              f"full_auc={pruned['full_auc']:.3f}, pruned_auc={pruned['pruned_auc']:.3f}")

        for r in (base_res, enr_res):
            r2 = {k: v for k, v in r.items() if not k.startswith(("tree_", "logit_"))}
            r2["symbol"] = sym
            r2["n_is"] = int(len(base_is if r["feature_set"] == "curated_12_only" else enr_is))
            r2["n_oos"] = int(len(base_oos if r["feature_set"] == "curated_12_only" else enr_oos))
            headline.append(r2)

        # Save enriched-tree leaves with rule names
        leaves = []
        feature_cols = enr_cols
        for tree_name, tmod, target_col in [
            ("tree_severe_enriched", enr_res["tree_severe"], "is_severe_loss"),
            ("tree_loser_enriched", enr_res["tree_loser"], "is_loser"),
        ]:
            t = tmod.tree_
            def recurse(node, conditions):
                if t.feature[node] == -2:
                    mask_is = pd.Series(True, index=enr_is.index)
                    mask_oos = pd.Series(True, index=enr_oos.index)
                    for f, op, thr in conditions:
                        if op == "<=":
                            mask_is &= enr_is[f] <= thr
                            mask_oos &= enr_oos[f] <= thr
                        else:
                            mask_is &= enr_is[f] > thr
                            mask_oos &= enr_oos[f] > thr
                    n_is, n_oos = int(mask_is.sum()), int(mask_oos.sum())
                    if n_is == 0:
                        return
                    prec_is = float(enr_is.loc[mask_is, target_col].mean())
                    prec_oos = float(enr_oos.loc[mask_oos, target_col].mean()) if n_oos else np.nan
                    net_pnl = float(-enr_oos.loc[mask_oos, "pnl_pct"].sum()) if n_oos else 0.0
                    base_oos_rate = float(enr_oos[target_col].mean())
                    leaves.append({
                        "symbol": sym, "tree": tree_name, "target": target_col,
                        "rule_name": rule_name(sym, target_col, conditions),
                        "rule_hash": rule_hash(conditions),
                        "n_is": n_is, "precision_is": prec_is,
                        "n_oos": n_oos, "precision_oos": prec_oos,
                        "lift_oos_vs_baseline": prec_oos - base_oos_rate if not np.isnan(prec_oos) else np.nan,
                        "net_pnl_oos_pct": net_pnl,
                        "is_candidate_rule": (prec_is >= 0.65 and n_is >= 30),
                        "uses_context": any(f in CONTEXT_FEATURES for f, _, _ in conditions),
                        "description": rule_description(conditions),
                    })
                    return
                feat = feature_cols[t.feature[node]]
                thr = float(t.threshold[node])
                recurse(t.children_left[node], conditions + [(feat, "<=", thr)])
                recurse(t.children_right[node], conditions + [(feat, ">", thr)])
            recurse(0, [])
        pd.DataFrame(leaves).to_csv(OUT / f"tree_leaves_enriched_{sym}.csv", index=False)

    pd.DataFrame(headline).to_csv(OUT / "headline_auc_compare.csv", index=False)
    pd.DataFrame(pruned_rows).to_csv(OUT / "pruned_vs_full_auc.csv", index=False)
    print("\nHeadline:")
    print(pd.DataFrame(headline)[["symbol", "feature_set", "n_features",
                                  "auc_tree_severe", "auc_tree_loser", "auc_l1_severe"]].to_string(index=False))

    # Plot: baseline vs enriched AUC
    h = pd.DataFrame(headline)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    width = 0.35
    syms = ["TQQQ", "SQQQ"]
    x = np.arange(len(syms))
    for i, metric in enumerate(["auc_tree_severe", "auc_tree_loser", "auc_l1_severe"]):
        base_vals = [h[(h["symbol"] == s) & (h["feature_set"] == "curated_12_only")][metric].iloc[0] for s in syms]
        enr_vals = [h[(h["symbol"] == s) & (h["feature_set"] == "curated_plus_context")][metric].iloc[0] for s in syms]
        ax.bar(x + i * 0.15 - 0.15, base_vals, 0.10, label=f"{metric} (base)", color=f"C{i}")
        ax.bar(x + i * 0.15 - 0.05, enr_vals, 0.10, label=f"{metric} (+context)", color=f"C{i}", edgecolor="black", hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels(syms)
    ax.axhline(0.5, color="black", linewidth=0.6, linestyle="--")
    ax.set_ylabel("OOS AUC")
    ax.set_title("Item 06: OOS AUC, curated_12 vs curated_12 + context")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "headline_auc_compare.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
