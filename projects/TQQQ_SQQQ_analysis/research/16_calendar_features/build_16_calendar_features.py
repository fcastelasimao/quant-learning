"""16_calendar_features: day-of-week, FOMC distance, holiday proximity, seasonality.

Depends on:
  - full_history_canonical/TRADES_<SYM>_full_history.csv (the canonical trades)
  - research/06_context_enrichment/enriched_trades_<sym>.csv (for the AUC-comparison block)

The FOMC_DATES list is hardcoded through 2026-12-16. After that, fetch fresh
dates from https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
and append before rerunning.


For each trade, compute calendar features and:
  1. Run univariate tests (same methodology as item 02) vs is_loser, is_severe_loss, pnl_pct.
  2. Add to L1-logit + depth-4 tree on is_severe_loss and compare OOS AUC against:
       - curated_12 only
       - curated_12 + 21 daily context (item 06 baseline)
       - curated_12 + 21 daily context + calendar features (this item)

All prior research directions are left untouched. This is a self-contained pass.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parent
PROJ = ROOT.parent.parent
CANON = PROJ / "full_history_canonical"
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

# FOMC announcement dates (the 2nd day of each meeting; Fed announcement at 2pm ET).
# Source: Federal Reserve published calendar 2013-2026.
# 114 dates: 8 regular meetings per year × 13 normal years (2013-2019, 2021-2026)
# plus 10 in 2020 (8 regular + 2 emergency cuts at Mar 3 and Mar 23).
# Verify against https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
# before relying past Dec 2026.
FOMC_DATES = pd.to_datetime([
    # 2013
    "2013-01-30", "2013-03-20", "2013-05-01", "2013-06-19",
    "2013-07-31", "2013-09-18", "2013-10-30", "2013-12-18",
    # 2014
    "2014-01-29", "2014-03-19", "2014-04-30", "2014-06-18",
    "2014-07-30", "2014-09-17", "2014-10-29", "2014-12-17",
    # 2015
    "2015-01-28", "2015-03-18", "2015-04-29", "2015-06-17",
    "2015-07-29", "2015-09-17", "2015-10-28", "2015-12-16",
    # 2016
    "2016-01-27", "2016-03-16", "2016-04-27", "2016-06-15",
    "2016-07-27", "2016-09-21", "2016-11-02", "2016-12-14",
    # 2017
    "2017-02-01", "2017-03-15", "2017-05-03", "2017-06-14",
    "2017-07-26", "2017-09-20", "2017-11-01", "2017-12-13",
    # 2018
    "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13",
    "2018-08-01", "2018-09-26", "2018-11-08", "2018-12-19",
    # 2019
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19",
    "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11",
    # 2020 (incl. emergency cuts in March)
    "2020-01-29", "2020-03-03", "2020-03-15", "2020-03-23",
    "2020-04-29", "2020-06-10", "2020-07-29", "2020-09-16",
    "2020-11-05", "2020-12-16",
    # 2021
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16",
    "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    # 2022
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
    "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    # 2026
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-16",
]).normalize()

# US market holidays (full close) and half-day sessions (early close at 1pm ET).
# Static through 2026; pandas_market_calendars could give these more authoritatively,
# but a small hard-coded list keeps this script dependency-free.
HALF_DAY_RULES = [
    # Day before Independence Day (when Jul 4 is a weekday)
    ("07-03", "is_half_day_jul3"),
    # Day before Thanksgiving NOT a half-day; the DAY AFTER is.
    # Black Friday is always a half-day:
    # we compute it dynamically from Thanksgiving below.
    # Christmas Eve when Dec 25 is a weekday
    ("12-24", "is_half_day_xmas_eve"),
]


def fourth_thursday(year: int) -> pd.Timestamp:
    """Thanksgiving = 4th Thursday of November."""
    d = pd.Timestamp(f"{year}-11-01")
    # weekday(): Mon=0 ... Sun=6. Thursday = 3.
    offset = (3 - d.weekday()) % 7
    first_thu = d + pd.Timedelta(days=offset)
    return first_thu + pd.Timedelta(weeks=3)


def black_friday(year: int) -> pd.Timestamp:
    return fourth_thursday(year) + pd.Timedelta(days=1)


def is_third_friday(d: pd.Timestamp) -> bool:
    if d.weekday() != 4:  # Friday
        return False
    return 15 <= d.day <= 21


def is_quad_witching(d: pd.Timestamp) -> bool:
    return is_third_friday(d) and d.month in (3, 6, 9, 12)


def signed_days_to_nearest_fomc(d: pd.Timestamp) -> tuple[float, float]:
    """Returns (signed_days_to_nearest, abs_days_to_nearest)."""
    diffs = (FOMC_DATES - d).days  # numpy int array
    abs_diffs = np.abs(diffs)
    idx = int(np.argmin(abs_diffs))
    signed = float(diffs[idx])
    return signed, float(abs_diffs[idx])


def build_calendar_features(trades: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=trades.index)
    et = trades["entry_time"].dt
    out["entry_time"] = trades["entry_time"]
    out["dow_of_entry"] = et.dayofweek.astype(float)  # 0=Mon ... 4=Fri
    out["month_of_entry"] = et.month.astype(float)
    out["is_monday"] = (et.dayofweek == 0).astype(int)
    out["is_friday"] = (et.dayofweek == 4).astype(int)
    out["is_summer"] = et.month.isin([6, 7, 8]).astype(int)
    out["is_santa_rally_window"] = ((et.month == 12) & (et.day >= 20)).astype(int) | \
                                    ((et.month == 1) & (et.day <= 5)).astype(int)
    out["is_first_session_of_month"] = (et.day <= 3).astype(int)
    out["is_last_session_of_month"] = (et.day >= 27).astype(int)

    # OPEX / quad-witching
    out["is_monthly_opex"] = trades["entry_time"].apply(
        lambda t: int(is_third_friday(pd.Timestamp(t).normalize()))
    )
    out["is_quad_witching"] = trades["entry_time"].apply(
        lambda t: int(is_quad_witching(pd.Timestamp(t).normalize()))
    )

    # Half-day sessions (approximation — see HALF_DAY_RULES)
    def is_hd(t: pd.Timestamp) -> int:
        d = pd.Timestamp(t).normalize()
        # Jul 3 if it's a weekday
        if d.month == 7 and d.day == 3 and d.weekday() < 5:
            return 1
        if d.month == 12 and d.day == 24 and d.weekday() < 5:
            return 1
        # Black Friday
        if d == black_friday(d.year):
            return 1
        return 0

    out["is_half_day"] = trades["entry_time"].apply(is_hd)

    # FOMC distance
    fomc = trades["entry_time"].apply(lambda t: signed_days_to_nearest_fomc(pd.Timestamp(t).normalize()))
    out["fomc_signed_days_to_nearest"] = fomc.apply(lambda x: x[0])
    out["fomc_abs_days_to_nearest"] = fomc.apply(lambda x: x[1])
    out["is_fomc_day"] = (out["fomc_abs_days_to_nearest"] == 0).astype(int)
    out["is_within_3d_of_fomc"] = (out["fomc_abs_days_to_nearest"] <= 3).astype(int)
    out["is_week_after_fomc"] = ((out["fomc_signed_days_to_nearest"] >= -7) &
                                  (out["fomc_signed_days_to_nearest"] <= -1)).astype(int)
    return out


CALENDAR_FEATURES = [
    "dow_of_entry", "month_of_entry",
    "is_monday", "is_friday", "is_summer", "is_santa_rally_window",
    "is_first_session_of_month", "is_last_session_of_month",
    "is_monthly_opex", "is_quad_witching", "is_half_day",
    "fomc_signed_days_to_nearest", "fomc_abs_days_to_nearest",
    "is_fomc_day", "is_within_3d_of_fomc", "is_week_after_fomc",
]
TARGETS = [("is_loser", "binary"), ("is_severe_loss", "binary"), ("pnl_pct", "continuous")]


def univariate(df: pd.DataFrame, feat: str, target: str, kind: str) -> dict | None:
    valid = df[[feat, target]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(valid) < 30 or valid[feat].nunique() < 2:
        return None
    x = valid[feat].values
    y = valid[target].values
    out = {"feature": feat, "target": target, "n": int(len(valid))}
    rho = stats.spearmanr(x, y)
    out["spearman"] = float(rho.statistic)
    out["spearman_p"] = float(rho.pvalue)
    if kind == "binary":
        try:
            auc = roc_auc_score(y, x)
        except ValueError:
            auc = np.nan
        out["auc"] = float(auc)
        out["auc_directional"] = float(max(auc, 1 - auc))
        try:
            mi = mutual_info_classif(x.reshape(-1, 1), y, random_state=SEED, discrete_features=False)[0]
            out["mutual_info"] = float(mi)
        except Exception:
            out["mutual_info"] = np.nan
    else:
        try:
            mi = mutual_info_regression(x.reshape(-1, 1), y, random_state=SEED)[0]
            out["mutual_info"] = float(mi)
        except Exception:
            out["mutual_info"] = np.nan
    return out


def fit_compare(df_is, df_oos, feature_cols, label):
    Xis = df_is[feature_cols].values
    Xoos = df_oos[feature_cols].values
    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=30, random_state=SEED)
    tree.fit(Xis, df_is["is_severe_loss"].astype(int))
    auc_t = roc_auc_score(df_oos["is_severe_loss"], tree.predict_proba(Xoos)[:, 1])
    sc = StandardScaler().fit(Xis)
    logit = LogisticRegression(penalty="l1", solver="liblinear", C=0.1, random_state=SEED, max_iter=2000)
    logit.fit(sc.transform(Xis), df_is["is_severe_loss"].astype(int))
    auc_l = roc_auc_score(df_oos["is_severe_loss"], logit.predict_proba(sc.transform(Xoos))[:, 1])
    return {"feature_set": label, "n_features": len(feature_cols),
            "auc_tree_severe": float(auc_t), "auc_l1_severe": float(auc_l)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # --------------- Univariate tests (per-symbol per-target) ---------------
    univ_rows = []
    for sym in ("TQQQ", "SQQQ"):
        df = pd.read_csv(CANON / f"TRADES_{sym}_full_history.csv",
                         parse_dates=["entry_time", "exit_time"])
        df = df[df["regime_entry"].notna()].copy()
        cal = build_calendar_features(df)
        df = df.merge(cal, on="entry_time", how="left")
        df.to_csv(OUT / f"enriched_calendar_{sym}.csv", index=False)
        for feat in CALENDAR_FEATURES:
            for target, kind in TARGETS:
                r = univariate(df, feat, target, kind)
                if r:
                    r["symbol"] = sym
                    univ_rows.append(r)
    univ = pd.DataFrame(univ_rows)
    univ["score"] = univ.apply(
        lambda r: (r.get("auc_directional", 0.5) - 0.5 + r.get("mutual_info", 0))
                  if r["target"] != "pnl_pct"
                  else (abs(r.get("spearman", 0)) + r.get("mutual_info", 0)),
        axis=1,
    )
    univ.sort_values(["symbol", "target", "score"], ascending=[True, True, False], inplace=True)
    univ.to_csv(OUT / "univariate_calendar.csv", index=False)
    print("Univariate (top 5 per (sym, target)):")
    for (sym, target), g in univ.groupby(["symbol", "target"]):
        print(f"\n  {sym} {target}:")
        cols = ["feature", "score", "spearman"]
        if target != "pnl_pct":
            cols += ["auc_directional", "mutual_info"]
        else:
            cols += ["mutual_info"]
        print(g[cols].head(5).to_string(index=False))

    # --------------- Modeling: AUC comparison on is_severe_loss ---------------
    headline = []
    for sym in ("TQQQ", "SQQQ"):
        enr = pd.read_csv(ITEM_06 / f"enriched_trades_{sym}.csv",
                          parse_dates=["entry_time", "exit_time"])
        cal = build_calendar_features(enr)
        enr = enr.merge(cal, on="entry_time", how="left")
        for c in CURATED_NUMERIC + DAILY_CONTEXT + CALENDAR_FEATURES:
            if c in enr.columns:
                enr[c] = pd.to_numeric(enr[c], errors="coerce")
        enr["year"] = enr["entry_time"].dt.year
        for r in ("chop_highvol", "sideways_lowvol"):
            enr[f"regime_{r}"] = (enr["regime_entry"] == r).astype(int)
        regime_dummies = ["regime_chop_highvol", "regime_sideways_lowvol"]
        scenarios = {
            "curated_12 only": CURATED_NUMERIC + regime_dummies,
            "+ 21 daily ctx (item 06)": CURATED_NUMERIC + regime_dummies + DAILY_CONTEXT,
            "+ daily + calendar (item 16)": CURATED_NUMERIC + regime_dummies + DAILY_CONTEXT + CALENDAR_FEATURES,
        }
        for label, feats in scenarios.items():
            sub = enr.dropna(subset=feats + ["is_severe_loss"])
            is_data = sub[sub["year"] <= IS_END_YEAR]
            oos_data = sub[sub["year"] > IS_END_YEAR]
            r = fit_compare(is_data, oos_data, feats, label)
            r["symbol"] = sym
            r["n_is"] = int(len(is_data))
            r["n_oos"] = int(len(oos_data))
            headline.append(r)
    h = pd.DataFrame(headline)
    h.to_csv(OUT / "headline_auc_calendar.csv", index=False)
    print("\nHeadline AUC comparison:")
    print(h[["symbol", "feature_set", "n_features", "n_is", "n_oos",
             "auc_tree_severe", "auc_l1_severe"]].to_string(index=False))

    # --------------- Plot: per-day-of-week mean pnl and loser rate ---------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    for col, sym in enumerate(("TQQQ", "SQQQ")):
        df = pd.read_csv(CANON / f"TRADES_{sym}_full_history.csv",
                         parse_dates=["entry_time"])
        df = df[df["regime_entry"].notna()].copy()
        df["dow"] = df["entry_time"].dt.dayofweek
        g = df.groupby("dow").agg(
            n=("pnl_pct", "size"),
            mean_pnl=("pnl_pct", "mean"),
            loser_rate=("is_loser", "mean"),
            severe_rate=("is_severe_loss", "mean"),
        )
        axes[0, col].bar(g.index, g["mean_pnl"], color="#2563eb", edgecolor="black")
        axes[0, col].axhline(0, color="black", linewidth=0.6)
        axes[0, col].set_xticks(range(5))
        axes[0, col].set_xticklabels(dow_labels)
        axes[0, col].set_ylabel("mean pnl_pct")
        axes[0, col].set_title(f"{sym}: mean pnl_pct by day of week")
        for i, n in enumerate(g["n"]):
            axes[0, col].annotate(f"n={int(n)}", (i, g["mean_pnl"].iloc[i]),
                                  xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7)
        axes[1, col].bar(g.index, g["loser_rate"], color="#dc2626", edgecolor="black",
                          label="loser_rate")
        axes[1, col].bar(g.index, g["severe_rate"], color="#7c3aed", edgecolor="black",
                          label="severe_rate", alpha=0.7, width=0.5)
        axes[1, col].axhline(float(df["is_loser"].mean()), color="black",
                             linewidth=0.6, linestyle="--", label="baseline loser_rate")
        axes[1, col].set_xticks(range(5))
        axes[1, col].set_xticklabels(dow_labels)
        axes[1, col].set_ylabel("rate")
        axes[1, col].set_title(f"{sym}: loser / severe-loss rate by day of week")
        axes[1, col].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "dow_outcomes.png", dpi=140)
    plt.close(fig)

    # --------------- Plot: FOMC proximity vs mean pnl ---------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, sym in zip(axes, ("TQQQ", "SQQQ")):
        df = pd.read_csv(CANON / f"TRADES_{sym}_full_history.csv",
                         parse_dates=["entry_time"])
        df = df[df["regime_entry"].notna()].copy()
        cal = build_calendar_features(df)
        df = df.merge(cal, on="entry_time", how="left")
        # Bucket by signed days to FOMC: bins around 0
        df["fomc_bucket"] = pd.cut(df["fomc_signed_days_to_nearest"],
                                    bins=[-9999, -10, -3, -1, 0, 1, 3, 10, 9999],
                                    labels=["≤-10", "-9:-4", "-3:-2", "-1", "+1:0", "+2:+3", "+4:+10", ">+10"])
        g = df.groupby("fomc_bucket", observed=True).agg(
            n=("pnl_pct", "size"),
            mean_pnl=("pnl_pct", "mean"),
            severe_rate=("is_severe_loss", "mean"),
        )
        x = range(len(g))
        ax.bar(x, g["mean_pnl"], color="#2563eb", edgecolor="black")
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(g.index.astype(str), rotation=30, ha="right")
        ax.set_ylabel("mean pnl_pct")
        ax.set_title(f"{sym}: mean pnl_pct by FOMC-day distance")
        for i, n in enumerate(g["n"]):
            ax.annotate(f"n={int(n)}", (i, g["mean_pnl"].iloc[i]),
                        xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fomc_proximity_pnl.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
