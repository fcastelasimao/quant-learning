"""05_capital_normalization: constant-notional equity metrics.

Each trade contributes (pnl_pct / 100) to a fixed-base daily P&L stream.
No compounding, no capital_before chain. Computed on:
  - full data (matches the existing pipeline's row universe for comparison)
  - regime-labeled subset (the universe for this exploratory pass)

Compared side-by-side with the existing inflated metrics from
`full_history_research/20260528_1402_feature_scan/05_validation/
traditional_metrics_baseline.csv`.

Depends on:
  - full_history_canonical/TRADES_<SYM>_full_history.csv  (the canonical trades)
  - full_history_research/20260528_1402_feature_scan/05_validation/traditional_metrics_baseline.csv
    (a frozen snapshot of the OLD inflated metrics, for side-by-side comparison)
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


ROOT = Path(__file__).resolve().parent
PROJ = ROOT.parent.parent
CANON = PROJ / "full_history_canonical"
OLD_BASELINE = PROJ / "full_history_research" / "20260528_1402_feature_scan" / "05_validation" / "traditional_metrics_baseline.csv"
OUT = ROOT


def load(sym: str) -> pd.DataFrame:
    return pd.read_csv(
        CANON / f"TRADES_{sym}_full_history.csv",
        parse_dates=["entry_time", "exit_time"],
    )


def constant_notional_metrics(df: pd.DataFrame, label: str) -> dict:
    g = df.sort_values("exit_time").copy()
    g["exit_date"] = g["exit_time"].dt.normalize()
    g = g.dropna(subset=["exit_date"])
    if g.empty:
        return {"label": label, "n_trades": 0}

    # Per-trade return on a fixed notional: pnl_pct is in percentage points.
    g["r"] = g["pnl_pct"] / 100.0

    # Daily return = sum of trade returns exiting that day.
    start = g["entry_time"].min().normalize()
    end = g["exit_date"].max()
    bdays = pd.bdate_range(start, end)
    daily = g.groupby("exit_date")["r"].sum().reindex(bdays, fill_value=0.0)

    equity = 1.0 + daily.cumsum()
    n_days = len(daily)
    years = n_days / 252.0
    total_return = float(daily.sum())
    annualized_arith = float(total_return / years) if years > 0 else np.nan

    mean_d = float(daily.mean())
    std_d = float(daily.std(ddof=1))
    sharpe = mean_d / std_d * np.sqrt(252) if std_d > 0 else np.nan
    neg = daily[daily < 0]
    down_std = float(neg.std(ddof=1)) if len(neg) > 1 else np.nan
    sortino = mean_d / down_std * np.sqrt(252) if down_std and down_std > 0 else np.nan

    running_max = equity.cummax()
    dd = equity - running_max
    max_dd = float(dd.min())
    calmar = annualized_arith / abs(max_dd) if max_dd < 0 else np.nan
    ulcer = float(np.sqrt(np.mean(np.square(np.minimum(dd.values, 0.0) * 100.0))))

    return {
        "label": label,
        "n_trades": int(len(g)),
        "n_days": int(n_days),
        "years": float(years),
        "total_return_constant_notional": total_return,
        "annualized_arith_return": annualized_arith,
        "sharpe_daily": sharpe,
        "sortino_daily": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "ulcer_index": ulcer,
        "mean_daily_return": mean_d,
        "std_daily_return": std_d,
        "n_neg_days": int((daily < 0).sum()),
        "n_pos_days": int((daily > 0).sum()),
        "max_pos_day": float(daily.max()),
        "max_neg_day": float(daily.min()),
    }


def main() -> None:
    rows = []
    for sym in ("TQQQ", "SQQQ"):
        full = load(sym)
        regime = full[full["regime_entry"].notna()].copy()

        m_full = constant_notional_metrics(full, label=f"{sym}_full")
        m_full["symbol"] = sym
        m_full["scope"] = "full_history"
        rows.append(m_full)

        m_reg = constant_notional_metrics(regime, label=f"{sym}_regime_labeled")
        m_reg["symbol"] = sym
        m_reg["scope"] = "regime_labeled"
        rows.append(m_reg)

    new_metrics = pd.DataFrame(rows)
    new_metrics.to_csv(OUT / "metrics_constant_notional.csv", index=False)

    # Side-by-side compare vs the existing inflated baseline
    old = pd.read_csv(OLD_BASELINE)
    compare_rows = []
    for sym in ("TQQQ", "SQQQ"):
        old_row = old[old["symbol"] == sym].iloc[0]
        new_full = new_metrics[(new_metrics["symbol"] == sym) & (new_metrics["scope"] == "full_history")].iloc[0]
        new_reg = new_metrics[(new_metrics["symbol"] == sym) & (new_metrics["scope"] == "regime_labeled")].iloc[0]
        compare_rows.append({
            "symbol": sym,
            "metric": "n_trades",
            "old_pipeline_inflated": int(old_row["n"]),
            "new_constant_notional_full": int(new_full["n_trades"]),
            "new_constant_notional_regime_labeled": int(new_reg["n_trades"]),
        })
        for key_old, key_new in [
            ("cagr", "annualized_arith_return"),
            ("sharpe", "sharpe_daily"),
            ("sortino", "sortino_daily"),
            ("calmar", "calmar"),
            ("ulcer_index", "ulcer_index"),
            ("max_drawdown", "max_drawdown"),
        ]:
            compare_rows.append({
                "symbol": sym,
                "metric": key_old,
                "old_pipeline_inflated": float(old_row[key_old]),
                "new_constant_notional_full": float(new_full[key_new]),
                "new_constant_notional_regime_labeled": float(new_reg[key_new]),
            })
    pd.DataFrame(compare_rows).to_csv(OUT / "metrics_compare.csv", index=False)

    # ---------- Equity curve comparison: compounded vs constant-notional ----------
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=False)
    for i, sym in enumerate(("TQQQ", "SQQQ")):
        df = load(sym).sort_values("exit_time").copy()
        df["exit_date"] = df["exit_time"].dt.normalize()
        df = df.dropna(subset=["exit_date"])
        df["r"] = df["pnl_pct"] / 100.0
        start = df["entry_time"].min().normalize()
        end = df["exit_date"].max()
        bdays = pd.bdate_range(start, end)
        # constant notional
        daily_add = df.groupby("exit_date")["r"].sum().reindex(bdays, fill_value=0.0)
        eq_const = 1.0 + daily_add.cumsum()
        # compounded with per-CSV capital_before reset baked in (as in the old pipeline)
        df["src"] = df["source_file"] if "source_file" in df.columns else "single"
        df["compound_r"] = (df["pnl"] / df["capital_before"].replace(0, np.nan)).fillna(df["r"])
        daily_compound_r = df.groupby("exit_date")["compound_r"].apply(lambda s: float((1 + s).prod() - 1)).reindex(bdays, fill_value=0.0)
        eq_compound = (1.0 + daily_compound_r).cumprod()

        ax = axes[i]
        ax.plot(bdays, eq_const.values, color="#2563eb", linewidth=1.2, label="constant notional (additive)")
        ax2 = ax.twinx()
        ax2.plot(bdays, eq_compound.values, color="#dc2626", linewidth=1.2, label="compounded across CSV resets")
        ax.set_title(f"{sym}: equity curves — constant-notional (left axis, blue) vs compounded (right axis, red, log)")
        ax.set_ylabel("constant-notional equity (1 + Σr)")
        ax2.set_yscale("log")
        ax2.set_ylabel("compounded equity (log scale)")
        ax.legend(loc="upper left")
        ax2.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "equity_curve_compare.png", dpi=140)
    plt.close(fig)
    print(pd.DataFrame(compare_rows).to_string(index=False))


if __name__ == "__main__":
    main()
