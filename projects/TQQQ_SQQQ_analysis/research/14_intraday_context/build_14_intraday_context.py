"""14_intraday_context: add intraday QQQ + TQQQ/SQQQ features at each trade's entry.

Depends on:
  - research/06_context_enrichment/enriched_trades_<sym>.csv (baseline + daily context)
  - Shared QuantFinance data store (DB_<TKR>_historical_data.db)
    for TKR in {QQQ, ^VIX, TQQQ, SQQQ}, candles_15min table. Refresh with
    `quantcore-ingest --intervals 15min --symbols QQQ ^VIX TQQQ SQQQ`.

NOTE: ^VIX intraday coverage in FMP starts 2023-09-25. VIX intraday features
are computed descriptively but EXCLUDED from modeling (zero IS rows).


For each trade with entry_time T on session day D, compute (strictly pre-entry):
  - QQQ_intraday_return_since_open   = QQQ last bar before T / QQQ open of D - 1
  - QQQ_intraday_range_position      = (QQQ last close - day's low so far) / (high - low)
  - QQQ_intraday_realized_vol_13bar  = stdev of last 13 15-min bar returns (annualized)
  - QQQ_intraday_volume_vs_5d_avg    = today's bar volume / 5-session avg at same time
  - SELF_intraday_return_since_open  = TQQQ/SQQQ same metric (own ticker)
  - SELF_intraday_dist_to_prior_close = (last bar close / yesterday close) - 1
  - VIX_intraday_change_since_open   = VIX last bar / VIX open - 1   (only post 2023-09-25; NaN before)
  - VIX_intraday_5bar_change         = VIX last bar / VIX 5 bars ago - 1 (same coverage limit)

Then re-fit the L1-logistic and depth-4 tree on `is_severe_loss` using:
    curated_12 + daily_context_21 + intraday_N
and compare OOS AUC vs item 06 (which had only daily context).
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

from quantcore import config as _qc_config

ROOT = Path(__file__).resolve().parent
PROJ = ROOT.parent.parent
CANON = PROJ / "full_history_canonical"
DATA_DIR = _qc_config.data_dir()
ITEM_06 = ROOT.parent / "06_context_enrichment"
OUT = ROOT
SEED = 42
IS_END_YEAR = 2020

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
# Modeling features: QQQ + SELF intraday only (full coverage 2015+).
# VIX intraday is computed and stored but EXCLUDED from modeling because FMP
# only has VIX 15-min back to 2023-09-25, which leaves 0 IS rows with VIX
# intraday data. Saved descriptively for future use.
INTRADAY_FEATURES = [
    "QQQ_intraday_return_since_open",
    "QQQ_intraday_range_position",
    "QQQ_intraday_realized_vol_13bar",
    "QQQ_intraday_volume_vs_5d_avg",
    "SELF_intraday_return_since_open",
    "SELF_intraday_dist_to_prior_close",
]
INTRADAY_DESCRIPTIVE_ONLY = [
    "VIX_intraday_change_since_open",
    "VIX_intraday_5bar_change",
]


def load_15min(symbol: str) -> pd.DataFrame:
    db = DATA_DIR / f"DB_{symbol}_historical_data.db"
    conn = sqlite3.connect(db)
    df = pd.read_sql(
        "SELECT et_datetime, open, high, low, close, volume FROM candles_15min ORDER BY ts",
        conn,
    )
    conn.close()
    df["dt"] = pd.to_datetime(df["et_datetime"])
    df["date"] = df["dt"].dt.normalize()
    df = df.sort_values("dt").reset_index(drop=True)
    return df


def load_1d(symbol: str) -> pd.DataFrame:
    db = DATA_DIR / f"DB_{symbol}_historical_data.db"
    conn = sqlite3.connect(db)
    df = pd.read_sql(
        "SELECT et_datetime, close FROM candles_1d ORDER BY ts",
        conn,
    )
    conn.close()
    df["date"] = pd.to_datetime(df["et_datetime"]).dt.normalize()
    return df[["date", "close"]].sort_values("date").reset_index(drop=True)


def build_intraday_at_entry(trades: pd.DataFrame, sym: str,
                            qqq_15: pd.DataFrame, vix_15: pd.DataFrame,
                            self_15: pd.DataFrame, self_1d: pd.DataFrame) -> pd.DataFrame:
    """For each trade entry_time, compute the intraday features strictly pre-entry.

    Implementation: for each entry_dt T, look at bars with dt <= T - 1 second
    (i.e., strictly before the bar containing T). We approximate by dt < T.
    """
    qqq_15 = qqq_15.set_index("dt", drop=False)
    vix_15 = vix_15.set_index("dt", drop=False) if not vix_15.empty else None
    self_15 = self_15.set_index("dt", drop=False)
    self_1d = self_1d.set_index("date")

    rows = []
    qqq_dt = qqq_15["dt"].values
    vix_dt = vix_15["dt"].values if vix_15 is not None else None
    self_dt = self_15["dt"].values

    for _, tr in trades.iterrows():
        T = tr["entry_time"]
        D = pd.Timestamp(T).normalize()

        # ---- QQQ intraday ----
        q_day = qqq_15[qqq_15["date"] == D]
        q_pre = q_day[q_day["dt"] < T]
        if len(q_pre) >= 1:
            day_open = float(q_day["open"].iloc[0])
            day_low_so_far = float(q_pre["low"].min())
            day_high_so_far = float(q_pre["high"].max())
            last_close = float(q_pre["close"].iloc[-1])
            qqq_return_since_open = (last_close / day_open - 1) if day_open else np.nan
            rng = (day_high_so_far - day_low_so_far)
            qqq_range_pos = ((last_close - day_low_so_far) / rng) if rng > 0 else np.nan
            # 13-bar realized vol (annualized)
            qlast13 = q_pre.tail(14)
            if len(qlast13) >= 5:
                rets = qlast13["close"].pct_change().dropna()
                qqq_vol_13 = float(rets.std() * np.sqrt(252 * 26)) if len(rets) >= 2 else np.nan
            else:
                qqq_vol_13 = np.nan
            # volume vs 5-day avg same-time-of-day
            tod = T.time()
            same_tod = qqq_15[qqq_15["dt"].dt.time == tod]
            recent_same_tod = same_tod[(same_tod["date"] < D) & (same_tod["date"] >= D - pd.Timedelta(days=10))]
            if len(recent_same_tod) >= 3 and len(q_pre) >= 1:
                cur_bar_vol = float(q_pre["volume"].iloc[-1])
                avg_vol = float(recent_same_tod["volume"].mean())
                qqq_vol_vs_avg = cur_bar_vol / avg_vol if avg_vol > 0 else np.nan
            else:
                qqq_vol_vs_avg = np.nan
        else:
            qqq_return_since_open = qqq_range_pos = qqq_vol_13 = qqq_vol_vs_avg = np.nan

        # ---- SELF (TQQQ or SQQQ) intraday ----
        s_day = self_15[self_15["date"] == D]
        s_pre = s_day[s_day["dt"] < T]
        if len(s_pre) >= 1:
            day_open = float(s_day["open"].iloc[0])
            last_close = float(s_pre["close"].iloc[-1])
            self_return_since_open = (last_close / day_open - 1) if day_open else np.nan
            # Prior close from daily series
            prior_dates = self_1d.index[self_1d.index < D]
            if len(prior_dates) > 0:
                prior_close = float(self_1d.loc[prior_dates[-1], "close"])
                self_dist_to_prior = (last_close / prior_close - 1) if prior_close else np.nan
            else:
                self_dist_to_prior = np.nan
        else:
            self_return_since_open = self_dist_to_prior = np.nan

        # ---- VIX intraday (limited coverage) ----
        if vix_15 is not None and not vix_15.empty:
            v_day = vix_15[vix_15["date"] == D]
            v_pre = v_day[v_day["dt"] < T]
            if len(v_pre) >= 1:
                v_open = float(v_day["open"].iloc[0])
                v_last = float(v_pre["close"].iloc[-1])
                vix_change = (v_last / v_open - 1) if v_open else np.nan
                v_recent = v_pre.tail(6)
                if len(v_recent) >= 5:
                    vix_5bar = float(v_recent["close"].iloc[-1] / v_recent["close"].iloc[0] - 1)
                else:
                    vix_5bar = np.nan
            else:
                vix_change = vix_5bar = np.nan
        else:
            vix_change = vix_5bar = np.nan

        rows.append({
            "entry_time": T,
            "QQQ_intraday_return_since_open": qqq_return_since_open,
            "QQQ_intraday_range_position": qqq_range_pos,
            "QQQ_intraday_realized_vol_13bar": qqq_vol_13,
            "QQQ_intraday_volume_vs_5d_avg": qqq_vol_vs_avg,
            "SELF_intraday_return_since_open": self_return_since_open,
            "SELF_intraday_dist_to_prior_close": self_dist_to_prior,
            "VIX_intraday_change_since_open": vix_change,
            "VIX_intraday_5bar_change": vix_5bar,
        })
    return pd.DataFrame(rows)


def fit_compare(df_is, df_oos, feature_cols, label):
    Xis = df_is[feature_cols].values
    Xoos = df_oos[feature_cols].values
    tree_severe = DecisionTreeClassifier(max_depth=4, min_samples_leaf=30, random_state=SEED)
    tree_severe.fit(Xis, df_is["is_severe_loss"].astype(int))
    auc_t = roc_auc_score(df_oos["is_severe_loss"], tree_severe.predict_proba(Xoos)[:, 1])

    sc = StandardScaler().fit(Xis)
    logit = LogisticRegression(penalty="l1", solver="liblinear", C=0.1, random_state=SEED, max_iter=2000)
    logit.fit(sc.transform(Xis), df_is["is_severe_loss"].astype(int))
    auc_l = roc_auc_score(df_oos["is_severe_loss"], logit.predict_proba(sc.transform(Xoos))[:, 1])

    return {"feature_set": label, "n_features": len(feature_cols),
            "auc_tree_severe": float(auc_t), "auc_l1_severe": float(auc_l)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("Loading 15-min DBs (this may take a moment)...")
    qqq_15 = load_15min("QQQ")
    vix_15 = load_15min("^VIX")
    print(f"  QQQ 15min n={len(qqq_15)}, VIX 15min n={len(vix_15)} (start {vix_15['dt'].min()})")

    headline = []
    for sym in ("TQQQ", "SQQQ"):
        print(f"\n{sym}: loading 15min + 1d + enriched trades...")
        self_15 = load_15min(sym)
        self_1d = load_1d(sym)
        enr = pd.read_csv(ITEM_06 / f"enriched_trades_{sym}.csv",
                          parse_dates=["entry_time", "exit_time"])
        # Compute intraday features
        intra = build_intraday_at_entry(enr[["entry_time"]], sym, qqq_15, vix_15, self_15, self_1d)
        merged = enr.merge(intra, on="entry_time", how="left")
        merged.to_csv(OUT / f"enriched_intraday_{sym}.csv", index=False)
        cov = merged[INTRADAY_FEATURES + INTRADAY_DESCRIPTIVE_ONLY].notna().mean()
        print(f"  intraday feature coverage:\n{cov.to_string()}")

        # Build modeling X
        for c in CURATED_NUMERIC + DAILY_CONTEXT + INTRADAY_FEATURES:
            if c not in merged.columns:
                continue
            merged[c] = pd.to_numeric(merged[c], errors="coerce")
        merged["year"] = merged["entry_time"].dt.year
        for r in ("chop_highvol", "sideways_lowvol"):
            merged[f"regime_{r}"] = (merged["regime_entry"] == r).astype(int)

        regime_dummies = ["regime_chop_highvol", "regime_sideways_lowvol"]
        # 3 feature-set scenarios
        scenarios = {
            "curated_12 only": CURATED_NUMERIC + regime_dummies,
            "+ 21 daily ctx (item 06)": CURATED_NUMERIC + regime_dummies + DAILY_CONTEXT,
            "+ daily + intraday (this item)": CURATED_NUMERIC + regime_dummies + DAILY_CONTEXT + INTRADAY_FEATURES,
        }
        for label, feats in scenarios.items():
            sub = merged.dropna(subset=feats + ["is_severe_loss"])
            is_data = sub[sub["year"] <= IS_END_YEAR]
            oos_data = sub[sub["year"] > IS_END_YEAR]
            r = fit_compare(is_data, oos_data, feats, label)
            r["symbol"] = sym
            r["n_is"] = int(len(is_data))
            r["n_oos"] = int(len(oos_data))
            headline.append(r)

    h = pd.DataFrame(headline)
    h.to_csv(OUT / "headline_auc_intraday.csv", index=False)
    print("\nHeadline AUC comparison:")
    print(h[["symbol", "feature_set", "n_features", "n_is", "n_oos",
             "auc_tree_severe", "auc_l1_severe"]].to_string(index=False))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    syms = ["TQQQ", "SQQQ"]
    x = np.arange(len(syms))
    width = 0.28
    for i, sc in enumerate(["curated_12 only", "+ 21 daily ctx (item 06)", "+ daily + intraday (this item)"]):
        v_tree = [h[(h["symbol"] == s) & (h["feature_set"] == sc)]["auc_tree_severe"].iloc[0] for s in syms]
        v_l1 = [h[(h["symbol"] == s) & (h["feature_set"] == sc)]["auc_l1_severe"].iloc[0] for s in syms]
        ax.bar(x + (i - 1) * width - width / 3, v_tree, width / 2, label=f"tree | {sc}", color=f"C{i}")
        ax.bar(x + (i - 1) * width + width / 6, v_l1, width / 2, label=f"L1 | {sc}", color=f"C{i}", edgecolor="black", hatch="//")
    ax.axhline(0.5, color="black", linewidth=0.6, linestyle="--")
    ax.set_xticks(x); ax.set_xticklabels(syms)
    ax.set_ylabel("OOS AUC on is_severe_loss")
    ax.set_title("Item 14: AUC under curated / +daily ctx / +daily+intraday ctx")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "intraday_auc_compare.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
