"""13_within_csv_compounding: realistic per-CSV compounded return as an
auxiliary view alongside the constant-notional convention.

Background: the six source CSVs each start capital at $10k, so chaining
(1+r) across them is meaningless. But chaining WITHIN one CSV is valid
because capital_before is continuous there. We report:

  - constant_notional_total_return  (item 05 convention, additive)
  - per_csv_compounded_finals       (one final equity per source_file)
  - geo_mean_compounded             (geometric mean of per-CSV finals,
                                     re-annualized to the union span)
  - within_csv_sharpe               (per-CSV Sharpe, then averaged)

These three give complementary views of the same data. The constant-notional
number is what we use for cross-rule comparison; the per-CSV compounded set
is the "realistic upper bound" for absolute equity figures.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
CANON = ROOT.parent.parent / "full_history_canonical"
OUT = ROOT


def load(sym: str) -> pd.DataFrame:
    return pd.read_csv(
        CANON / f"TRADES_{sym}_full_history.csv",
        parse_dates=["entry_time", "exit_time"],
    )


def per_csv_compound(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per source_file with that CSV's compounded final equity
    and per-trade return statistics."""
    rows = []
    if "source_file" not in df.columns:
        df = df.copy()
        df["source_file"] = "single"
    for src, g in df.groupby("source_file"):
        g = g.sort_values("entry_time").copy()
        g["r"] = (g["pnl"] / g["capital_before"].replace(0, np.nan)).fillna(g["pnl_pct"] / 100.0)
        # Compound trade-by-trade
        final_eq = float((1.0 + g["r"]).prod())
        n = len(g)
        start = g["entry_time"].min()
        end = g["exit_time"].max()
        years = (end - start).total_seconds() / (365.25 * 86400) if pd.notna(end) and pd.notna(start) else np.nan
        # Daily Sharpe within this CSV
        g["exit_date"] = g["exit_time"].dt.normalize()
        daily = (1.0 + g.groupby("exit_date")["r"].apply(lambda s: float((1 + s).prod() - 1)))
        daily_r = daily - 1.0
        if len(daily_r) > 2 and daily_r.std() > 0:
            sharpe = float(daily_r.mean() / daily_r.std() * np.sqrt(252))
        else:
            sharpe = np.nan
        rows.append({
            "source_file": src,
            "n_trades": n,
            "entry_start": start,
            "exit_end": end,
            "years_span": float(years) if not pd.isna(years) else np.nan,
            "compounded_final_equity": final_eq,
            "compounded_total_return": final_eq - 1.0,
            "annualized_compounded_cagr": (final_eq ** (1.0 / years) - 1.0) if years and years > 0 and final_eq > 0 else np.nan,
            "within_csv_daily_sharpe": sharpe,
            "mean_per_trade_r": float(g["r"].mean()),
            "std_per_trade_r": float(g["r"].std(ddof=1)),
        })
    return pd.DataFrame(rows).sort_values("source_file").reset_index(drop=True)


def aggregate(per_csv: pd.DataFrame, df: pd.DataFrame) -> dict:
    """Roll up the per-CSV rows to a single set of headline numbers."""
    if per_csv.empty:
        return {}
    # Geometric mean of per-CSV finals (each CSV represents a separate $10k run)
    finals = per_csv["compounded_final_equity"].values
    geo_mean = float(np.exp(np.mean(np.log(np.maximum(finals, 1e-12)))))
    # Constant-notional totals (recomputed for reference)
    g = df.sort_values("exit_time").copy()
    g["r"] = g["pnl_pct"] / 100.0
    constant_total = float(g["r"].sum())
    # Union time span
    start = df["entry_time"].min()
    end = df["exit_time"].max()
    union_years = (end - start).total_seconds() / (365.25 * 86400)
    # Re-annualize geo-mean to union span
    geo_cagr = float(geo_mean ** (1.0 / union_years) - 1.0) if union_years > 0 and geo_mean > 0 else np.nan
    return {
        "n_csvs": int(len(per_csv)),
        "union_years_span": float(union_years),
        "geo_mean_final_equity": geo_mean,
        "geo_mean_cagr": geo_cagr,
        "median_within_csv_sharpe": float(per_csv["within_csv_daily_sharpe"].median()),
        "constant_notional_total_return": constant_total,
        "constant_notional_annualized": constant_total / union_years if union_years > 0 else np.nan,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    per_csv_all = []
    agg_rows = []
    for sym in ("TQQQ", "SQQQ"):
        df = load(sym)
        per_csv = per_csv_compound(df)
        per_csv.insert(0, "symbol", sym)
        per_csv_all.append(per_csv)
        agg = aggregate(per_csv, df)
        agg["symbol"] = sym
        agg_rows.append(agg)

    pd.concat(per_csv_all, ignore_index=True).to_csv(OUT / "per_csv_compounded.csv", index=False)
    pd.DataFrame(agg_rows).to_csv(OUT / "compounding_compare.csv", index=False)
    print(pd.DataFrame(agg_rows).to_string(index=False))


if __name__ == "__main__":
    main()
