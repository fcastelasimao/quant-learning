"""08_focus_rule_recheck: stricter look at the SQQQ rsi_x_atr_cell_3_1 rule.

The rule from the existing pipeline:
    SQQQ, RSI_entry in [56.4, 59.85] AND atr_pct in [0.39, 0.47]

Previously reported as the ONE OOS survivor (precision 87.5% on 8 trades,
bootstrap CI [+2.43, +8.45] on net pnl impact).

We re-check it under harder tests:
  1. Re-evaluate on the regime-labeled subset (item 01 filter).
  2. Per-OOS-year breakdown — are the 8 wins concentrated in 2024-2026?
  3. Naive bootstrap (the original) vs **block bootstrap by year** — the
     naive CI is misleading because 6 of 8 trades came from a single regime
     window. Block bootstrap by year correctly reflects "would this rule
     work in a different year-sample?"
  4. Random-baseline comparison at the same trigger rate.
  5. Per-regime breakdown.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _rule_naming import rule_hash, rule_name  # noqa: E402


ROOT = Path(__file__).resolve().parent
CANON = ROOT.parent.parent / "full_history_canonical"
OUT = ROOT
SEED = 42
IS_END_YEAR = 2020

SYMBOL = "SQQQ"
RSI_LO, RSI_HI = 56.4, 59.85
ATR_LO, ATR_HI = 0.39, 0.47
N_BOOT = 2000


def load(sym: str, regime_only: bool) -> pd.DataFrame:
    df = pd.read_csv(CANON / f"TRADES_{sym}_full_history.csv",
                     parse_dates=["entry_time", "exit_time"])
    if regime_only:
        df = df[df["regime_entry"].notna()].copy()
    df["year"] = df["entry_time"].dt.year
    df["period"] = np.where(df["year"] <= IS_END_YEAR, "IS_2015_2020", "OOS_2021_2026")
    return df


def apply_focus_rule(df: pd.DataFrame) -> pd.Series:
    return ((df["RSI_entry"] >= RSI_LO) & (df["RSI_entry"] <= RSI_HI) &
            (df["atr_pct"] >= ATR_LO) & (df["atr_pct"] <= ATR_HI)).fillna(False)


def summarize(df: pd.DataFrame, mask: pd.Series, label: str) -> dict:
    n = int(mask.sum())
    if n == 0:
        return {"slice": label, "n_total": len(df), "n_flagged": 0,
                "trigger_rate": 0.0, "precision_loser": np.nan,
                "mean_flagged_pnl_pct": np.nan,
                "net_pnl_impact_avoided_if_skipped": 0.0}
    flagged = df[mask]
    return {
        "slice": label,
        "n_total": len(df),
        "n_flagged": n,
        "trigger_rate": float(mask.mean()),
        "precision_loser": float(flagged["is_loser"].mean()),
        "precision_severe": float(flagged["is_severe_loss"].mean()),
        "mean_flagged_pnl_pct": float(flagged["pnl_pct"].mean()),
        "median_flagged_pnl_pct": float(flagged["pnl_pct"].median()),
        "min_flagged_pnl_pct": float(flagged["pnl_pct"].min()),
        "max_flagged_pnl_pct": float(flagged["pnl_pct"].max()),
        "net_pnl_impact_avoided_if_skipped": float(-flagged["pnl_pct"].sum()),
    }


def naive_bootstrap(pnls: np.ndarray, n_boot: int, rng: np.random.Generator) -> dict:
    if len(pnls) == 0:
        return {"observed": 0.0, "ci_lo": np.nan, "ci_hi": np.nan, "mean_boot": np.nan}
    impacts = -pnls[rng.integers(0, len(pnls), size=(n_boot, len(pnls)))].sum(axis=1)
    return {
        "observed": float(-pnls.sum()),
        "ci_lo": float(np.percentile(impacts, 2.5)),
        "ci_hi": float(np.percentile(impacts, 97.5)),
        "mean_boot": float(impacts.mean()),
    }


def block_bootstrap_by_year(flagged_df: pd.DataFrame, n_boot: int, rng: np.random.Generator) -> dict:
    """Resample WHICH YEARS' flagged trades to include, with replacement.

    Captures the right uncertainty when some years contribute many trades and
    others contribute none. Each bootstrap iteration picks `n_unique_years`
    years with replacement and aggregates THEIR flagged-trade pnls.
    """
    if flagged_df.empty:
        return {"observed": 0.0, "ci_lo": np.nan, "ci_hi": np.nan,
                "mean_boot": np.nan, "n_unique_years": 0}
    years = list(flagged_df["year"].unique())
    by_year = {y: flagged_df.loc[flagged_df["year"] == y, "pnl_pct"].values for y in years}
    n_y = len(years)
    impacts = np.empty(n_boot)
    for b in range(n_boot):
        sample_years = rng.choice(years, size=n_y, replace=True)
        sample_pnls = np.concatenate([by_year[y] for y in sample_years]) if n_y > 0 else np.array([])
        impacts[b] = -sample_pnls.sum()
    return {
        "observed": float(-flagged_df["pnl_pct"].sum()),
        "ci_lo": float(np.percentile(impacts, 2.5)),
        "ci_hi": float(np.percentile(impacts, 97.5)),
        "mean_boot": float(impacts.mean()),
        "n_unique_years": int(n_y),
    }


def random_baseline(df: pd.DataFrame, trigger_rate: float, n_iters: int, rng: np.random.Generator) -> dict:
    if trigger_rate <= 0 or df.empty:
        return {"median_precision": np.nan, "median_net_impact": np.nan}
    losers = df["is_loser"].values
    pnls = df["pnl_pct"].values
    precs = []
    nets = []
    for _ in range(n_iters):
        m = rng.random(len(df)) < trigger_rate
        if m.sum() == 0:
            continue
        precs.append(float(losers[m].mean()))
        nets.append(float(-pnls[m].sum()))
    return {
        "median_precision": float(np.median(precs)),
        "median_net_impact": float(np.median(nets)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    rows = []
    detail_rows = []

    for regime_only in (False, True):
        df = load(SYMBOL, regime_only=regime_only)
        scope = "regime_labeled" if regime_only else "full_data"
        mask_all = apply_focus_rule(df)

        for period in ("IS_2015_2020", "OOS_2021_2026"):
            sub = df[df["period"] == period]
            if sub.empty:
                continue
            m = apply_focus_rule(sub)
            s = summarize(sub, m, f"{scope}/{period}")
            s["scope"] = scope
            s["period"] = period
            rows.append(s)

            # per-year inside OOS
            if period == "OOS_2021_2026":
                for y, ysub in sub.groupby("year"):
                    my = apply_focus_rule(ysub)
                    sy = summarize(ysub, my, f"{scope}/year_{int(y)}")
                    sy["scope"] = scope
                    sy["period"] = str(int(y))
                    rows.append(sy)

        # Per-regime breakdown (regime_only data)
        if regime_only and "regime_entry" in df.columns:
            for regime, rsub in df[df["period"] == "OOS_2021_2026"].groupby("regime_entry"):
                mr = apply_focus_rule(rsub)
                sr = summarize(rsub, mr, f"{scope}/OOS_regime_{regime}")
                sr["scope"] = scope
                sr["period"] = f"OOS_regime_{regime}"
                rows.append(sr)

        # OOS bootstrap (regime_only is what we care about going forward)
        if regime_only:
            oos = df[df["period"] == "OOS_2021_2026"]
            flagged = oos[apply_focus_rule(oos)]
            naive = naive_bootstrap(flagged["pnl_pct"].values, N_BOOT, rng)
            block = block_bootstrap_by_year(flagged, N_BOOT, rng)
            trig = float(apply_focus_rule(oos).mean())
            rand = random_baseline(oos, trig, n_iters=500, rng=rng)
            detail_rows.append({
                "scope": scope, "n_oos_flagged": len(flagged),
                "observed_net_impact": naive["observed"],
                "naive_boot_ci_lo": naive["ci_lo"],
                "naive_boot_ci_hi": naive["ci_hi"],
                "naive_boot_mean": naive["mean_boot"],
                "block_boot_ci_lo": block["ci_lo"],
                "block_boot_ci_hi": block["ci_hi"],
                "block_boot_mean": block["mean_boot"],
                "block_n_unique_years": block["n_unique_years"],
                "random_same_rate_precision_median": rand["median_precision"],
                "random_same_rate_net_impact_median": rand["median_net_impact"],
                "trigger_rate_oos": trig,
            })

            # Dump the actual flagged trades for inspection
            flagged_cols = [c for c in ["entry_time", "exit_time", "year", "regime_entry",
                                         "RSI_entry", "atr_pct", "pnl_pct", "is_loser", "is_severe_loss"]
                            if c in flagged.columns]
            flagged[flagged_cols].to_csv(OUT / "oos_flagged_trades.csv", index=False)

    pd.DataFrame(rows).to_csv(OUT / "focus_rule_summary.csv", index=False)
    pd.DataFrame(detail_rows).to_csv(OUT / "focus_rule_bootstrap.csv", index=False)
    print("Summary:")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nBootstrap detail:")
    print(pd.DataFrame(detail_rows).to_string(index=False))

    # Plot: yearly observed pnl of flagged trades
    summ = pd.DataFrame(rows)
    yr_rows = summ[(summ["scope"] == "regime_labeled") & (summ["period"].str.fullmatch(r"\d{4}"))]
    if not yr_rows.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        yr_rows = yr_rows.assign(period_int=yr_rows["period"].astype(int)).sort_values("period_int")
        ax.bar(yr_rows["period_int"], yr_rows["n_flagged"], color="#7c3aed", edgecolor="black", linewidth=0.5)
        for _, r in yr_rows.iterrows():
            prec = r["precision_loser"]
            label = f"prec={prec:.0%}" if r["n_flagged"] > 0 else "n=0"
            ax.annotate(label, (r["period_int"], r["n_flagged"]), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=8)
        ax.set_xlabel("OOS year")
        ax.set_ylabel("n_flagged by focus rule")
        ax.set_title(f"SQQQ focus rule (regime-labeled subset): yearly trigger count + precision")
        fig.tight_layout()
        fig.savefig(OUT / "yearly_triggers.png", dpi=140)
        plt.close(fig)


if __name__ == "__main__":
    main()
