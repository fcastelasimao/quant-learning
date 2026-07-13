"""Full-history feature discovery research pack.

Reads the 2013-2026 TQQQ/SQQQ CSV trade logs in
``full_history_canonical/trades_backtest/`` and
creates an organized research folder with data-quality audits, single-variable
diagnostics, nonlinear plots, interaction/regime tables, candidate loser rules,
validation, and a short memo.
"""
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-quant")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from quantcore import stats as qcs


BASE_DIR = Path(__file__).resolve().parent
PROJ = BASE_DIR.parent
DEFAULT_INPUT_DIR = PROJ / "full_history_canonical" / "trades_backtest"
DEFAULT_OUTPUT_ROOT = PROJ / "full_history_research"
CANONICAL_DIR = PROJ / "full_history_canonical"
FEATURE_DICTIONARY_PATH = PROJ / "FEATURE_DICTIONARY.md"

SYMBOLS = ("TQQQ", "SQQQ")
IS_END = pd.Timestamp("2020-12-31 23:59:59")
SEVERE_LOSS_THRESHOLD = -1.0  # pnl_pct is in percentage points.
RANDOM_SEED = 42
N_QUANTILES = 10
BOOTSTRAP_N = 1000
FOCUS_RULES = [("SQQQ", "rsi_x_atr_cell_3_1_high_loser_rate")]
PNL_HEATMAP_ABS_FLOOR = 5.0  # pct-points; keeps diverging scale visually comparable across runs.

POST_TRADE_OR_LEAKAGE_COLUMNS = {
    "exit_time",
    "exit_decision_price",
    "exit_avg_order_price",
    "exit_reason",
    "pnl",
    "pnl_pct",
    "cumulative_profit",
    "capital_after",
    "capital_end",
}

ADMIN_COLUMNS = {"trade_id", "global_trade_id", "run_started_at", "mode", "source_file", "period"}
IDENTITY_COLUMNS = {"symbol", "entry_time"}
PRETRADE_ACCOUNTING_COLUMNS = {"capital_before", "decision_price", "avg_order_price", "qty"}

BASE_FEATURES_TO_PLOT = [
    "RSI_entry",
    "atr_pct",
    "BBP_entry",
    "volume_ratio",
    "bars_since_last_stop",
    "hour_of_entry",
]

INTERACTION_PAIRS = [
    ("RSI_entry", "atr_pct", "rsi_x_atr"),
    ("RSI_entry", "BBP_entry", "rsi_x_bbp"),
    ("atr_pct", "volume_ratio", "atr_x_volume"),
    ("hour_of_entry", "atr_pct", "hour_x_atr"),
    ("bars_since_last_stop", "atr_pct", "bars_since_stop_x_atr"),
]


@dataclass
class Manifest:
    rows: list[dict]

    def add(self, path: Path, section: str, description: str, symbol: str = "", feature: str = "", plot_type: str = "") -> None:
        self.rows.append(
            {
                "section": section,
                "symbol": symbol,
                "feature": feature,
                "plot_type": plot_type,
                "path": str(path),
                "description": description,
            }
        )

    def write(self, path: Path) -> None:
        pd.DataFrame(self.rows).to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full-history feature discovery.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--run-id",
        default=f"{datetime.now():%Y%m%d_%H%M}_feature_scan",
        help="Output folder name under full_history_research/.",
    )
    return parser.parse_args()


def progress(message: str) -> None:
    print(f"[full-history] {message}", flush=True)


def make_dirs(run_dir: Path) -> dict[str, Path]:
    dirs = {
        "quality": run_dir / "00_data_quality",
        "single": run_dir / "01_single_variable",
        "interactions": run_dir / "02_interactions",
        "regime": run_dir / "03_regime_analysis",
        "rules": run_dir / "04_candidate_rules",
        "validation": run_dir / "05_validation",
        "reports": run_dir / "06_reports",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def load_trades(input_dir: Path) -> pd.DataFrame:
    files = sorted(input_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV trade logs found in {input_dir}")
    frames = []
    for path in files:
        df = pd.read_csv(path)
        df["source_file"] = path.name
        frames.append(df)
    trades = pd.concat(frames, ignore_index=True)
    for col in ("entry_time", "exit_time"):
        if col in trades.columns:
            # Current source CSVs use ISO-style yyyy-mm-dd timestamps. Do not
            # parse day-first here; that can flip ambiguous May/December dates.
            trades[col] = pd.to_datetime(trades[col], errors="coerce")
    if "qty" in trades.columns and "shares" not in trades.columns:
        trades["shares"] = trades["qty"]
    trades = trades.sort_values(["symbol", "entry_time", "exit_time"]).reset_index(drop=True)
    trades["global_trade_id"] = trades.groupby("symbol").cumcount() + 1
    trades["is_loser"] = trades["pnl_pct"] < 0
    trades["is_severe_loss"] = trades["pnl_pct"] <= SEVERE_LOSS_THRESHOLD
    trades["abs_pnl_pct"] = trades["pnl_pct"].abs()
    trades["hold_days"] = (trades["exit_time"] - trades["entry_time"]).dt.total_seconds() / 86400.0
    trades["period"] = np.where(trades["entry_time"] <= IS_END, "IS_2013_2020", "OOS_2021_2026")
    return trades


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["atr_pct"] = safe_div(out.get("atr"), out.get("avg_order_price")) * 100.0
    if "volume_ratio" in out:
        out["log_volume_ratio"] = np.log(out["volume_ratio"].replace(0, np.nan))
    if "high_water_mark_entry" in out:
        out["dist_to_high_water_mark"] = safe_div(out["avg_order_price"], out["high_water_mark_entry"]) - 1.0

    ma_level_cols = [
        c
        for c in out.columns
        if c.startswith(("MA", "EMA"))
        and "_D" not in c
        and not c.endswith("_HT")
        and out[c].dtype.kind in "if"
    ]
    for col in ma_level_cols:
        out[f"dist_to_{col}"] = safe_div(out["avg_order_price"], out[col]) - 1.0

    for col in ["RSI_entry", "atr_pct", "BBP_entry", "volume_ratio"]:
        if col in out:
            out[f"{col}_roll_pctile_252"] = (
                out.groupby("symbol")[col]
                .transform(lambda s: s.rolling(252, min_periods=60).apply(last_percentile_rank, raw=False))
            )
    return out.replace([np.inf, -np.inf], np.nan)


def safe_div(a: pd.Series | None, b: pd.Series | None) -> pd.Series:
    if a is None or b is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(a, errors="coerce") / pd.to_numeric(b, errors="coerce").replace(0, np.nan)


def last_percentile_rank(window: pd.Series) -> float:
    values = window.dropna()
    if len(values) < 2:
        return np.nan
    return float((values <= values.iloc[-1]).mean())


def audit_schema(df: pd.DataFrame, out_dir: Path, manifest: Manifest) -> None:
    rows = []
    for col in df.columns:
        rows.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "missing_count": int(df[col].isna().sum()),
                "missing_pct": float(df[col].isna().mean()),
                "nunique": int(df[col].nunique(dropna=True)),
                "sample_values": "; ".join(map(str, df[col].dropna().head(3).tolist())),
            }
        )
    schema = pd.DataFrame(rows)
    path = out_dir / "schema_audit.csv"
    schema.to_csv(path, index=False)
    manifest.add(path, "00_data_quality", "Column-level schema, missingness, uniqueness, and samples.")

    summary = (
        df.groupby("symbol")
        .agg(
            n_trades=("pnl_pct", "size"),
            entry_start=("entry_time", "min"),
            exit_end=("exit_time", "max"),
            mean_pnl_pct=("pnl_pct", "mean"),
            median_pnl_pct=("pnl_pct", "median"),
            loser_rate=("is_loser", "mean"),
            severe_loss_rate=("is_severe_loss", "mean"),
        )
        .reset_index()
    )
    path = out_dir / "dataset_summary.csv"
    summary.to_csv(path, index=False)
    manifest.add(path, "00_data_quality", "Per-symbol row counts, spans, and basic outcome rates.")

    for sym, g in df.groupby("symbol"):
        path = out_dir / f"canonical_{sym}.csv"
        g.to_csv(path, index=False)
        manifest.add(path, "00_data_quality", f"Full-history canonical research table for {sym}.", symbol=sym)

        CANONICAL_DIR.mkdir(parents=True, exist_ok=True)
        stable_path = CANONICAL_DIR / f"TRADES_{sym}_full_history.csv"
        g.to_csv(stable_path, index=False)
        manifest.add(stable_path, "00_data_quality", f"Stable joined full-history CSV for {sym}.", symbol=sym)


def build_feature_audit(df: pd.DataFrame, out_dir: Path, manifest: Manifest) -> tuple[list[str], list[str], list[str]]:
    rows = []
    included_numeric = []
    included_categorical = []
    excluded = []

    for col in df.columns:
        reason = ""
        include = False
        role = "exclude"
        if col in POST_TRADE_OR_LEAKAGE_COLUMNS or col in {"abs_pnl_pct", "is_loser", "is_severe_loss", "hold_days"}:
            reason = "post-trade/leakage/outcome"
        elif col in ADMIN_COLUMNS:
            reason = "administrative"
        elif col in IDENTITY_COLUMNS:
            reason = "identity/time key"
        elif col in PRETRADE_ACCOUNTING_COLUMNS or col == "shares":
            reason = "pre-trade accounting/sizing context; excluded from signal scan"
        elif df[col].isna().mean() >= 0.98:
            reason = "mostly missing"
        elif df[col].nunique(dropna=True) <= 1:
            reason = "constant"
        elif col == "regime_entry":
            include = True
            role = "categorical"
            reason = "included as pre-trade regime label; causality must be confirmed from strategy code"
        elif df[col].dtype.kind in "ifb":
            include = True
            role = "numeric"
            reason = "included numeric pre-trade feature"
        elif df[col].dtype == object:
            include = True
            role = "categorical"
            reason = "included categorical pre-trade feature"
        else:
            reason = "unsupported dtype"

        if include and role == "numeric":
            included_numeric.append(col)
        elif include and role == "categorical":
            included_categorical.append(col)
        else:
            excluded.append(col)

        rows.append(
            {
                "column": col,
                "role": role,
                "included": include,
                "reason": reason,
                "missing_pct": float(df[col].isna().mean()),
                "nunique": int(df[col].nunique(dropna=True)),
            }
        )

    audit = pd.DataFrame(rows).sort_values(["included", "role", "column"], ascending=[False, True, True])
    path = out_dir / "feature_inclusion_exclusion.csv"
    audit.to_csv(path, index=False)
    manifest.add(path, "00_data_quality", "Leakage audit and feature inclusion/exclusion decisions.")

    leak_path = out_dir / "leakage_audit.csv"
    audit[audit["reason"].str.contains("leakage|post-trade|outcome", case=False, na=False)].to_csv(leak_path, index=False)
    manifest.add(leak_path, "00_data_quality", "Columns explicitly excluded as leakage or outcomes.")
    return included_numeric, included_categorical, excluded


def pearson_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float, float, float]:
    z = pd.concat([x, y], axis=1).dropna()
    if len(z) < 5 or z.iloc[:, 0].nunique() < 2:
        return np.nan, np.nan, np.nan, np.nan
    pear = stats.pearsonr(z.iloc[:, 0], z.iloc[:, 1])
    spear = stats.spearmanr(z.iloc[:, 0], z.iloc[:, 1], nan_policy="omit")
    spear_stat = getattr(spear, "statistic", getattr(spear, "correlation", np.nan))
    return float(pear.statistic), float(pear.pvalue), float(spear_stat), float(spear.pvalue)


def single_variable_tables(
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    out_dir: Path,
    manifest: Manifest,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    corr_rows = []
    bucket_rows = []
    for sym, g in df.groupby("symbol"):
        for feature in numeric_features:
            if feature not in g:
                continue
            x = pd.to_numeric(g[feature], errors="coerce")
            valid = g[x.notna()].copy()
            if len(valid) < 30 or x.nunique(dropna=True) < 2:
                continue
            pear, pear_p, spear, spear_p = pearson_spearman(valid[feature], valid["pnl_pct"])
            pear_abs, pear_abs_p, spear_abs, spear_abs_p = pearson_spearman(valid[feature], valid["abs_pnl_pct"])
            winners = valid.loc[~valid["is_loser"], feature].dropna()
            losers = valid.loc[valid["is_loser"], feature].dropna()
            if len(winners) > 5 and len(losers) > 5:
                mw = stats.mannwhitneyu(winners, losers, alternative="two-sided")
                loser_mean = losers.mean()
                winner_mean = winners.mean()
            else:
                mw = None
                loser_mean = np.nan
                winner_mean = np.nan
            binned = quantile_bucket(valid[feature], 5)
            tmp = valid.assign(bucket=binned).dropna(subset=["bucket"])
            bucket_summary = summarize_buckets(tmp, "bucket", feature)
            loser_spread = float(bucket_summary["loser_rate"].max() - bucket_summary["loser_rate"].min()) if len(bucket_summary) else np.nan
            severe_spread = float(bucket_summary["severe_loss_rate"].max() - bucket_summary["severe_loss_rate"].min()) if len(bucket_summary) else np.nan
            corr_rows.append(
                {
                    "symbol": sym,
                    "feature": feature,
                    "n": len(valid),
                    "pearson_pnl": pear,
                    "pearson_pnl_pvalue": pear_p,
                    "spearman_pnl": spear,
                    "spearman_pnl_pvalue": spear_p,
                    "pearson_abs_pnl": pear_abs,
                    "spearman_abs_pnl": spear_abs,
                    "winner_mean_feature": winner_mean,
                    "loser_mean_feature": loser_mean,
                    "winner_loser_mannwhitney_pvalue": float(mw.pvalue) if mw is not None else np.nan,
                    "loser_rate_spread_q5": loser_spread,
                    "severe_loss_rate_spread_q5": severe_spread,
                    "score": np.nanmean([abs(spear), abs(pear), loser_spread, severe_spread]),
                }
            )
            for _, row in bucket_summary.iterrows():
                row_dict = row.to_dict()
                row_dict.update({"symbol": sym, "feature": feature, "bucket_type": "quantile"})
                bucket_rows.append(row_dict)

        for feature in categorical_features:
            if feature not in g:
                continue
            tmp = g.dropna(subset=[feature]).copy()
            if len(tmp) < 30 or tmp[feature].nunique(dropna=True) < 2:
                continue
            bucket_summary = summarize_buckets(tmp, feature)
            for _, row in bucket_summary.iterrows():
                row_dict = row.to_dict()
                row_dict.update({"symbol": sym, "feature": feature, "bucket_type": "category"})
                bucket_rows.append(row_dict)

    correlations = pd.DataFrame(corr_rows).sort_values(["symbol", "score"], ascending=[True, False])
    path = out_dir / "single_variable_correlations.csv"
    correlations.to_csv(path, index=False)
    manifest.add(path, "01_single_variable", "Pearson/Spearman correlations and winner-vs-loser tests.")

    buckets = pd.DataFrame(bucket_rows)
    path = out_dir / "single_variable_bucket_summary.csv"
    buckets.to_csv(path, index=False)
    manifest.add(path, "01_single_variable", "Bucket-level PnL, loser rate, severe-loss rate, and PnL contribution.")
    return correlations, buckets


def quantile_bucket(s: pd.Series, q: int = 5) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if x.nunique(dropna=True) < 2:
        return pd.Series(np.nan, index=s.index)
    try:
        buckets = pd.qcut(x.rank(method="first"), q, labels=False, duplicates="drop")
    except ValueError:
        return pd.Series(np.nan, index=s.index)
    return buckets.astype(float)


def summarize_buckets(df: pd.DataFrame, bucket_col: str, value_col: str | None = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    total_pnl = df["pnl_pct"].sum()
    for bucket, g in df.groupby(bucket_col, dropna=False):
        rows.append(
            {
                "bucket": bucket,
                "n": len(g),
                "feature_min": numeric_min(g[value_col]) if value_col else numeric_min(g[bucket_col]),
                "feature_max": numeric_max(g[value_col]) if value_col else numeric_max(g[bucket_col]),
                "mean_pnl_pct": g["pnl_pct"].mean(),
                "median_pnl_pct": g["pnl_pct"].median(),
                "total_pnl_pct": g["pnl_pct"].sum(),
                "pct_total_pnl": g["pnl_pct"].sum() / total_pnl if total_pnl else np.nan,
                "loser_rate": g["is_loser"].mean(),
                "severe_loss_rate": g["is_severe_loss"].mean(),
                "win_rate": 1.0 - g["is_loser"].mean(),
                "min_pnl_pct": g["pnl_pct"].min(),
                "max_pnl_pct": g["pnl_pct"].max(),
            }
        )
    return pd.DataFrame(rows)


def numeric_min(s: pd.Series) -> float | str:
    return float(s.min()) if pd.api.types.is_numeric_dtype(s) else str(s.min())


def numeric_max(s: pd.Series) -> float | str:
    return float(s.max()) if pd.api.types.is_numeric_dtype(s) else str(s.max())


def selected_plot_features(correlations: pd.DataFrame, numeric_features: list[str], symbol: str) -> list[str]:
    base = [f for f in BASE_FEATURES_TO_PLOT if f in numeric_features]
    sym_corr = correlations[correlations["symbol"] == symbol].copy()
    top_ma = [
        f
        for f in sym_corr["feature"].tolist()
        if f.startswith("dist_to_") or "_D" in f or f.endswith("_roll_pctile_252")
    ][:4]
    top_other = [f for f in sym_corr["feature"].tolist() if f not in base + top_ma][:4]
    out = []
    for f in base + top_ma + top_other:
        if f in numeric_features and f not in out:
            out.append(f)
    return out[:14]


def make_single_variable_plots(
    df: pd.DataFrame,
    correlations: pd.DataFrame,
    numeric_features: list[str],
    out_dir: Path,
    manifest: Manifest,
) -> list[str]:
    plotted = []
    for sym, g in df.groupby("symbol"):
        features = selected_plot_features(correlations, numeric_features, sym)
        for feature in features:
            if feature not in g:
                continue
            plot_feature_vs_pnl(g, sym, feature, out_dir, manifest)
            plot_nonlinear(g, sym, feature, out_dir, manifest)
            plot_contribution_by_bucket(g, sym, feature, out_dir, manifest)
            plot_distribution_by_bucket(g, sym, feature, out_dir, manifest)
            plotted.append(feature)
    return plotted


def _bucket_axis_labels(grouped: pd.DataFrameGroupBy, sorted_buckets: list, feature: str, include_n: bool) -> list[str]:
    labels = []
    for bk in sorted_buckets:
        cell = grouped.get_group(bk)
        lo, hi = cell[feature].min(), cell[feature].max()
        base = f"B{int(bk)}\n[{lo:.2g}, {hi:.2g}]"
        if include_n:
            base += f"\nn={len(cell)}"
        labels.append(base)
    return labels


def plot_contribution_by_bucket(g: pd.DataFrame, symbol: str, feature: str, out_dir: Path, manifest: Manifest) -> None:
    data = g[[feature, "pnl_pct"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 30 or data[feature].nunique() < 2:
        return
    b = quantile_bucket(data[feature], N_QUANTILES)
    tmp = data.assign(bucket=b).dropna(subset=["bucket"]).copy()
    grouped = tmp.groupby("bucket")
    sorted_buckets = sorted(grouped.groups.keys())
    total_pnl = float(tmp["pnl_pct"].sum())
    total_n = len(tmp)
    pct_pnl = [(float(grouped.get_group(bk)["pnl_pct"].sum()) / total_pnl * 100.0) if total_pnl else np.nan for bk in sorted_buckets]
    pct_trades = [len(grouped.get_group(bk)) / total_n * 100.0 for bk in sorted_buckets]
    labels = _bucket_axis_labels(grouped, sorted_buckets, feature, include_n=False)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(sorted_buckets))
    width = 0.4
    ax.bar(x - width / 2, pct_pnl, width, color="#2563eb", label="% of total pnl_pct sum")
    ax.bar(x + width / 2, pct_trades, width, color="#fb923c", label="% of total trades")
    equal_share = 100.0 / N_QUANTILES
    ax.axhline(equal_share, color="black", linewidth=0.8, linestyle="--", label=f"equal share ({equal_share:.0f}%)")
    ax.axhline(0, color="#374151", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("% of total")
    ax.set_title(
        f"{symbol} {feature}: per-bucket P/L contribution vs trade share | total_pnl_sum={total_pnl:.1f} pp | n={total_n}"
    )
    ax.legend(loc="best")
    fig.tight_layout()
    path = feature_dir(out_dir, symbol, feature) / "contribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    manifest.add(path, "01_single_variable", f"Per-bucket pnl_pct contribution vs trade share for {feature}.", symbol, feature, "contribution_bar")


def plot_distribution_by_bucket(g: pd.DataFrame, symbol: str, feature: str, out_dir: Path, manifest: Manifest) -> None:
    data = g[[feature, "pnl_pct"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 30 or data[feature].nunique() < 2:
        return
    b = quantile_bucket(data[feature], N_QUANTILES)
    tmp = data.assign(bucket=b).dropna(subset=["bucket"]).copy()
    grouped = tmp.groupby("bucket")
    sorted_buckets = sorted(grouped.groups.keys())
    labels = _bucket_axis_labels(grouped, sorted_buckets, feature, include_n=True)

    p_low, p_high = np.nanquantile(data["pnl_pct"], [0.01, 0.99])
    p_low = min(p_low, -2.0)
    p_high = max(p_high, 2.0)
    all_data = [np.clip(grouped.get_group(bk)["pnl_pct"].values, p_low, p_high) for bk in sorted_buckets]
    loss_data = [grouped.get_group(bk).loc[grouped.get_group(bk)["pnl_pct"] < 0, "pnl_pct"].values for bk in sorted_buckets]

    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    axes[0].boxplot(all_data, tick_labels=labels, showmeans=True, meanline=True,
                    meanprops={"color": "#dc2626"}, medianprops={"color": "#16a34a"})
    axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].axhline(-1.0, color="#dc2626", linewidth=0.8, linestyle=":", label="severe-loss (-1%)")
    axes[0].set_ylabel("pnl_pct (clipped to ±p99)")
    axes[0].set_title(f"{symbol} {feature}: trade-return distribution by bucket (all trades; red mean / green median)")
    axes[0].legend(loc="best")

    has_losses = [len(d) > 0 for d in loss_data]
    if any(has_losses):
        valid_loss = [d for d, h in zip(loss_data, has_losses) if h]
        valid_labels = [lab for lab, h in zip(labels, has_losses) if h]
        axes[1].boxplot(valid_loss, tick_labels=valid_labels)
    axes[1].axhline(-1.0, color="#dc2626", linewidth=0.8, linestyle=":", label="severe-loss (-1%)")
    axes[1].set_ylabel("pnl_pct (losses only)")
    axes[1].set_xlabel(feature)
    axes[1].set_title(f"{symbol} {feature}: loss distribution by bucket (pnl_pct < 0 only)")
    axes[1].legend(loc="best")

    fig.tight_layout()
    path = feature_dir(out_dir, symbol, feature) / "distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    manifest.add(path, "01_single_variable", f"Per-bucket pnl_pct distribution for {feature} (top: all trades clipped to ±p99; bottom: losses only).", symbol, feature, "boxplot")


def plot_feature_vs_pnl(g: pd.DataFrame, symbol: str, feature: str, out_dir: Path, manifest: Manifest) -> None:
    data = g[[feature, "pnl_pct", "is_loser"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 30 or data[feature].nunique() < 2:
        return
    pear, pear_p, spear, spear_p = pearson_spearman(data[feature], data["pnl_pct"])
    y_low, y_high = np.nanquantile(data["pnl_pct"], [0.01, 0.99])
    y_low = min(y_low, -2.0)
    y_high = max(y_high, 2.0)
    clipped = int(((data["pnl_pct"] < y_low) | (data["pnl_pct"] > y_high)).sum())

    # Use N_QUANTILES (= 5) so the plot's buckets match the CSV bucket tables.
    b = quantile_bucket(data[feature], N_QUANTILES)
    tmp = data.assign(bucket=b).dropna(subset=["bucket"])
    grouped = tmp.groupby("bucket")
    bucket_min = grouped[feature].min()
    bucket_max = grouped[feature].max()
    bucket_n = grouped.size()
    centers = grouped[feature].median()
    means = grouped["pnl_pct"].mean()
    medians = grouped["pnl_pct"].median()
    loser = grouped["is_loser"].mean()
    sorted_buckets = sorted(grouped.groups.keys())

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    # Alternating light-grey bands so bucket boundaries are visible on both panels.
    for idx, bk in enumerate(sorted_buckets):
        if idx % 2 == 1:
            for ax in axes:
                ax.axvspan(bucket_min[bk], bucket_max[bk], color="#e5e7eb", alpha=0.5, zorder=0)

    axes[0].scatter(data[feature], data["pnl_pct"].clip(y_low, y_high), s=12, alpha=0.25, color="#2563eb", zorder=2)
    axes[0].axhline(0, color="black", linewidth=0.8, zorder=1)
    axes[0].plot(centers, means, color="#dc2626", marker="o", label="bucket mean", zorder=3)
    axes[0].plot(centers, medians, color="#16a34a", marker="s", label="bucket median", zorder=3)
    axes[0].set_ylabel("pnl_pct")
    axes[0].set_ylim(y_low, y_high)
    axes[0].legend(loc="best")
    axes[0].set_title(
        f"{symbol} {feature} vs pnl_pct | n={len(data)} | {N_QUANTILES} quantile buckets | "
        f"pear={pear:.3f} p={pear_p:.3g} | spear={spear:.3f} p={spear_p:.3g} | clipped={clipped}"
    )
    # n-per-bucket labels just under the top edge of the upper panel.
    for bk in sorted_buckets:
        axes[0].annotate(
            f"n={int(bucket_n[bk])}",
            xy=(centers[bk], y_high),
            xytext=(0, -4),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8,
            color="#374151",
        )

    axes[1].plot(centers, loser, color="#7c3aed", marker="o", zorder=3)
    axes[1].axhline(data["is_loser"].mean(), color="black", linewidth=0.8, linestyle="--", zorder=1)
    axes[1].set_ylabel("loser rate")
    axes[1].set_xlabel(feature)
    axes[1].set_ylim(0, 1)
    fig.tight_layout()
    path = feature_dir(out_dir, symbol, feature) / "scatter.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    manifest.add(path, "01_single_variable", f"{feature} vs PnL with {N_QUANTILES}-bucket bands, mean/median markers, n-per-bucket labels, and loser rate.", symbol, feature, "scatter_bucket")


def plot_nonlinear(g: pd.DataFrame, symbol: str, feature: str, out_dir: Path, manifest: Manifest) -> None:
    data = g[[feature, "pnl_pct"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 50 or data[feature].nunique() < 5:
        return
    x = data[feature].astype(float).values
    y = data["pnl_pct"].astype(float).values
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    lin = np.polyfit(x, y, 1)
    quad = np.polyfit(x, y, 2)
    cubic = np.polyfit(x, y, 3)
    x_grid = np.linspace(np.nanpercentile(x, 1), np.nanpercentile(x, 99), 200)

    def r2(coefs: np.ndarray) -> float:
        pred = np.polyval(coefs, x)
        ss_res = np.square(y - pred).sum()
        ss_tot = np.square(y - y.mean()).sum()
        return float(1 - ss_res / ss_tot) if ss_tot else np.nan

    # Robust-ish visual smooth: rolling median over sorted observations.
    window = max(25, min(175, len(data) // 12))
    smooth_y = pd.Series(y_sorted).rolling(window, min_periods=max(10, window // 4), center=True).median()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y_low, y_high = np.nanquantile(y, [0.01, 0.99])
    ax.scatter(x, np.clip(y, y_low, y_high), s=10, alpha=0.18, color="#64748b", label="trades")
    ax.plot(x_grid, np.polyval(lin, x_grid), color="#2563eb", label=f"linear R2={r2(lin):.4f}")
    ax.plot(x_grid, np.polyval(quad, x_grid), color="#dc2626", label=f"quadratic R2={r2(quad):.4f}")
    ax.plot(x_grid, np.polyval(cubic, x_grid), color="#9333ea", linestyle="--", label=f"cubic R2={r2(cubic):.4f}")
    ax.plot(x_sorted, smooth_y, color="#16a34a", linewidth=2, label="rolling median smooth")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"{symbol} nonlinear fit: {feature} vs pnl_pct")
    ax.set_xlabel(feature)
    ax.set_ylabel("pnl_pct")
    ax.legend(loc="best")
    fig.tight_layout()
    path = feature_dir(out_dir, symbol, feature) / "nonlinear.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    manifest.add(path, "01_single_variable", f"Linear/quadratic/cubic and rolling-median fit for {feature}.", symbol, feature, "nonlinear_fit")


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)


def symbol_dir(out_dir: Path, symbol: str) -> Path:
    p = out_dir / symbol
    p.mkdir(parents=True, exist_ok=True)
    return p


def feature_dir(out_dir: Path, symbol: str, feature: str) -> Path:
    p = out_dir / symbol / safe_name(feature)
    p.mkdir(parents=True, exist_ok=True)
    return p


def pair_dir(out_dir: Path, symbol: str, pair_name: str) -> Path:
    p = out_dir / symbol / safe_name(pair_name)
    p.mkdir(parents=True, exist_ok=True)
    return p


def interaction_tables_and_plots(df: pd.DataFrame, out_dir: Path, manifest: Manifest) -> None:
    for sym, g in df.groupby("symbol"):
        for x_col, y_col, name in INTERACTION_PAIRS:
            if x_col not in g or y_col not in g:
                continue
            table = interaction_table(g, x_col, y_col)
            if table.empty:
                continue
            pdir = pair_dir(out_dir, sym, name)
            path = pdir / "table.csv"
            table.to_csv(path, index=False)
            manifest.add(path, "02_interactions", f"{x_col} x {y_col} interaction table.", sym, f"{x_col}|{y_col}", "table")
            plot_heatmap(table, "loser_rate", sym, x_col, y_col, name, out_dir, manifest)
            plot_heatmap(table, "mean_pnl_pct", sym, x_col, y_col, name, out_dir, manifest)

        if "regime_entry" in g:
            for feature in ("RSI_entry", "atr_pct"):
                table = regime_feature_table(g, feature)
                if table.empty:
                    continue
                name = f"regime_x_{safe_name(feature)}"
                pdir = pair_dir(out_dir, sym, name)
                path = pdir / "table.csv"
                table.to_csv(path, index=False)
                manifest.add(path, "02_interactions", f"Regime x {feature} bucket diagnostics.", sym, feature, "table")
                plot_heatmap(table, "loser_rate", sym, "regime_entry", feature, name, out_dir, manifest)
                plot_heatmap(table, "mean_pnl_pct", sym, "regime_entry", feature, name, out_dir, manifest)


def interaction_table(g: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    data = g[[x_col, y_col, "pnl_pct", "is_loser", "is_severe_loss"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 60 or data[x_col].nunique() < 2 or data[y_col].nunique() < 2:
        return pd.DataFrame()
    x_bucket = quantile_bucket(data[x_col], 5)
    y_bucket = quantile_bucket(data[y_col], 5)
    data = data.assign(x_bucket=x_bucket, y_bucket=y_bucket).dropna(subset=["x_bucket", "y_bucket"])
    rows = []
    for (xb, yb), cell in data.groupby(["x_bucket", "y_bucket"]):
        rows.append(
            {
                "x_bucket": int(xb),
                "y_bucket": int(yb),
                "x_median": cell[x_col].median(),
                "y_median": cell[y_col].median(),
                "n": len(cell),
                "loser_rate": cell["is_loser"].mean(),
                "severe_loss_rate": cell["is_severe_loss"].mean(),
                "mean_pnl_pct": cell["pnl_pct"].mean(),
                "median_pnl_pct": cell["pnl_pct"].median(),
                "total_pnl_pct": cell["pnl_pct"].sum(),
            }
        )
    return pd.DataFrame(rows)


def regime_feature_table(g: pd.DataFrame, feature: str) -> pd.DataFrame:
    data = g[["regime_entry", feature, "pnl_pct", "is_loser", "is_severe_loss"]].copy()
    data["regime_entry"] = data["regime_entry"].fillna("MISSING")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=[feature])
    data["feature_bucket"] = data.groupby("regime_entry")[feature].transform(lambda s: quantile_bucket(s, 5))
    rows = []
    for (regime, bucket), cell in data.dropna(subset=["feature_bucket"]).groupby(["regime_entry", "feature_bucket"]):
        rows.append(
            {
                "regime_entry": regime,
                "feature_bucket": int(bucket),
                "feature_median": cell[feature].median(),
                "n": len(cell),
                "loser_rate": cell["is_loser"].mean(),
                "severe_loss_rate": cell["is_severe_loss"].mean(),
                "mean_pnl_pct": cell["pnl_pct"].mean(),
                "median_pnl_pct": cell["pnl_pct"].median(),
                "total_pnl_pct": cell["pnl_pct"].sum(),
            }
        )
    return pd.DataFrame(rows)


def plot_heatmap(table: pd.DataFrame, value_col: str, symbol: str, x_col: str, y_col: str, pair_name: str, out_dir: Path, manifest: Manifest) -> None:
    if {"x_bucket", "y_bucket", "x_median", "y_median"}.issubset(table.columns):
        pivot = table.pivot(index="y_bucket", columns="x_bucket", values=value_col)
        n_pivot = table.pivot(index="y_bucket", columns="x_bucket", values="n")
        # One representative feature value per bucket (median across row/column cell medians).
        x_label_median = table.groupby("x_bucket")["x_median"].median()
        y_label_median = table.groupby("y_bucket")["y_median"].median()
        y_tick_labels = [f"{y_label_median[r]:.2g}" for r in pivot.index]
        y_axis_label = f"{y_col} (bucket median)"
    elif {"feature_bucket", "regime_entry", "feature_median"}.issubset(table.columns):
        # Regime diagnostics are categorical on y and quantile buckets on x.
        pivot = table.pivot(index="regime_entry", columns="feature_bucket", values=value_col)
        n_pivot = table.pivot(index="regime_entry", columns="feature_bucket", values="n")
        x_label_median = table.groupby("feature_bucket")["feature_median"].median()
        y_tick_labels = [str(v) for v in pivot.index]
        y_axis_label = f"{y_col} (category)"
    else:
        # Defensive fallback for unexpected table shape.
        return
    fig, ax = plt.subplots(figsize=(7, 5.5))

    cmap = "RdYlGn_r" if value_col == "loser_rate" else "RdYlGn"
    heatmap_kwargs = {"origin": "lower", "aspect": "auto", "cmap": cmap}
    if value_col == "mean_pnl_pct":
        finite_values = pivot.values[np.isfinite(pivot.values)]
        observed_abs_max = float(np.max(np.abs(finite_values))) if finite_values.size else 0.0
        v = max(observed_abs_max, PNL_HEATMAP_ABS_FLOOR)
        heatmap_kwargs.update({"vmin": -v, "vmax": v})
    im = ax.imshow(pivot.values, **heatmap_kwargs)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{x_label_median[c]:.2g}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(y_tick_labels)
    ax.set_xlabel(f"{x_col} (bucket median)")
    ax.set_ylabel(y_axis_label)
    total_n = int(np.nansum(n_pivot.values))
    ax.set_title(f"{symbol}: {value_col} by {x_col} x {y_col} (N total={total_n})")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if pd.notna(val):
                ax.text(j, i + 0.15, f"{val:.2f}", ha="center", va="center", fontsize=9)
                n_val = n_pivot.values[i, j]
                if pd.notna(n_val):
                    pct = (float(n_val) / total_n * 100.0) if total_n > 0 else 0.0
                    ax.text(j, i - 0.22, f"n={int(n_val)} ({pct:.1f}%)", ha="center", va="center", fontsize=7, color="#1f2937")
    fig.colorbar(im, ax=ax, label=value_col)
    fig.tight_layout()
    path = pair_dir(out_dir, symbol, pair_name) / f"{value_col}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    manifest.add(path, "02_interactions", f"Heatmap of {value_col} by {x_col} and {y_col} (axes show bucket medians; cells show value + n + share%).", symbol, f"{x_col}|{y_col}", "heatmap")


def regime_analysis(df: pd.DataFrame, out_dir: Path, manifest: Manifest) -> None:
    rows = []
    for sym, g in df.groupby("symbol"):
        tmp = g.copy()
        tmp["regime_entry"] = tmp["regime_entry"].fillna("MISSING") if "regime_entry" in tmp else "MISSING"
        for regime, cell in tmp.groupby("regime_entry"):
            rows.append(
                {
                    "symbol": sym,
                    "regime_entry": regime,
                    "n": len(cell),
                    "mean_pnl_pct": cell["pnl_pct"].mean(),
                    "median_pnl_pct": cell["pnl_pct"].median(),
                    "loser_rate": cell["is_loser"].mean(),
                    "severe_loss_rate": cell["is_severe_loss"].mean(),
                    "profit_factor": profit_factor(cell["pnl_pct"]),
                    "max_loss_pct": cell["pnl_pct"].min(),
                    "max_gain_pct": cell["pnl_pct"].max(),
                    "total_pnl_pct": cell["pnl_pct"].sum(),
                }
            )
    summary = pd.DataFrame(rows).sort_values(["symbol", "mean_pnl_pct"], ascending=[True, False])
    path = out_dir / "regime_summary.csv"
    summary.to_csv(path, index=False)
    manifest.add(path, "03_regime_analysis", "Per-symbol performance by source regime_entry label.")

    for sym in SYMBOLS:
        plot_regime_summary(summary[summary["symbol"] == sym], sym, out_dir, manifest)


def profit_factor(pnl_pct: pd.Series) -> float:
    gains = pnl_pct[pnl_pct > 0].sum()
    losses = -pnl_pct[pnl_pct < 0].sum()
    return float(gains / losses) if losses else np.inf


def plot_regime_summary(summary: pd.DataFrame, symbol: str, out_dir: Path, manifest: Manifest) -> None:
    if summary.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    order = summary.sort_values("mean_pnl_pct", ascending=False)
    axes[0].bar(order["regime_entry"].astype(str), order["mean_pnl_pct"], color="#2563eb")
    axes[0].set_title(f"{symbol}: mean pnl by regime")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[1].bar(order["regime_entry"].astype(str), order["loser_rate"], color="#dc2626")
    axes[1].set_title(f"{symbol}: loser rate by regime")
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    path = symbol_dir(out_dir, symbol) / "regime_summary.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    manifest.add(path, "03_regime_analysis", "Mean PnL and loser rate by regime.", symbol, "regime_entry", "bar")


def discover_candidate_rules(df: pd.DataFrame, correlations: pd.DataFrame, out_dir: Path, manifest: Manifest) -> tuple[pd.DataFrame, pd.DataFrame]:
    rule_defs = []
    for sym, g in df.groupby("symbol"):
        is_data = g[g["period"] == "IS_2013_2020"].copy()
        if is_data.empty:
            continue
        top_features = correlations[correlations["symbol"] == sym]["feature"].head(8).tolist()
        for feature in [f for f in top_features if f in is_data and is_data[f].dtype.kind in "ifb"]:
            rules = best_univariate_rules(is_data, feature)
            for rule in rules:
                rule["symbol"] = sym
                rule_defs.append(rule)
        for x_col, y_col, base_name in INTERACTION_PAIRS[:3]:
            if x_col in is_data and y_col in is_data:
                rule = best_interaction_rule(is_data, x_col, y_col, base_name)
                if rule:
                    rule["symbol"] = sym
                    rule_defs.append(rule)
        if "regime_entry" in is_data:
            rule = best_regime_rule(is_data)
            if rule:
                rule["symbol"] = sym
                rule_defs.append(rule)

    definitions = pd.DataFrame(rule_defs)
    if definitions.empty:
        return definitions, pd.DataFrame()
    definitions = definitions.sort_values(["symbol", "is_loser_precision", "net_pnl_pct_impact"], ascending=[True, False, False])
    path = out_dir / "candidate_rule_definitions_from_is.csv"
    definitions.to_csv(path, index=False)
    manifest.add(path, "04_candidate_rules", "Candidate loser rules frozen from in-sample data.")

    rng = np.random.default_rng(RANDOM_SEED)
    eval_rows = []
    for _, rule in definitions.iterrows():
        sym_data = df[df["symbol"] == rule["symbol"]].copy()
        for period_name, period_data in [("FULL", sym_data), ("IS_2013_2020", sym_data[sym_data["period"] == "IS_2013_2020"]), ("OOS_2021_2026", sym_data[sym_data["period"] == "OOS_2021_2026"])]:
            if period_data.empty:
                continue
            flags = apply_rule(period_data, rule)
            eval_rows.append(evaluate_rule(period_data, flags, rule["rule_name"], rule["symbol"], period_name, "candidate"))
            eval_rows.append(evaluate_random_filter(period_data, flags.mean(), rule["rule_name"], rule["symbol"], period_name, rng))
    evals = pd.DataFrame(eval_rows)
    path = out_dir / "candidate_rule_skip_impact.csv"
    evals.to_csv(path, index=False)
    manifest.add(path, "04_candidate_rules", "Skip-trade impact of candidate rules and same-rate random filters.")
    plot_rule_comparison(evals, out_dir, manifest)
    return definitions, evals


def best_univariate_rules(g: pd.DataFrame, feature: str) -> list[dict]:
    data = g[[feature, "is_loser", "pnl_pct"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 60 or data[feature].nunique() < 3:
        return []
    bucket = quantile_bucket(data[feature], 5)
    data = data.assign(bucket=bucket).dropna(subset=["bucket"])
    summaries = []
    for b, cell in data.groupby("bucket"):
        if len(cell) < 20:
            continue
        summaries.append(
            {
                "bucket": int(b),
                "precision": cell["is_loser"].mean(),
                "net_impact": -cell["pnl_pct"].sum(),
                "n": len(cell),
                "min": cell[feature].min(),
                "max": cell[feature].max(),
            }
        )
    if not summaries:
        return []
    ranked = sorted(summaries, key=lambda r: (r["precision"], r["net_impact"]), reverse=True)[:2]
    out = []
    for r in ranked:
        out.append(
            {
                "rule_name": f"{feature}_bucket_{r['bucket']}_high_loser_rate",
                "rule_type": "univariate_bucket",
                "feature_a": feature,
                "feature_b": "",
                "lower_a": r["min"],
                "upper_a": r["max"],
                "bucket_a_index": r["bucket"],
                "bucket_b_index": -1,
                "category_value": "",
                "is_loser_precision": r["precision"],
                "net_pnl_pct_impact": r["net_impact"],
                "n_flagged_is": r["n"],
            }
        )
    return out


def best_interaction_rule(g: pd.DataFrame, x_col: str, y_col: str, base_name: str) -> dict | None:
    data = g[[x_col, y_col, "is_loser", "pnl_pct"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 80:
        return None
    data = data.assign(x_bucket=quantile_bucket(data[x_col], 5), y_bucket=quantile_bucket(data[y_col], 5)).dropna()
    best = None
    for (xb, yb), cell in data.groupby(["x_bucket", "y_bucket"]):
        if len(cell) < 15:
            continue
        candidate = {
            "precision": cell["is_loser"].mean(),
            "net_impact": -cell["pnl_pct"].sum(),
            "n": len(cell),
            "x_min": cell[x_col].min(),
            "x_max": cell[x_col].max(),
            "y_min": cell[y_col].min(),
            "y_max": cell[y_col].max(),
            "xb": int(xb),
            "yb": int(yb),
        }
        if best is None or (candidate["precision"], candidate["net_impact"]) > (best["precision"], best["net_impact"]):
            best = candidate
    if best is None:
        return None
    return {
        "rule_name": f"{base_name}_cell_{best['xb']}_{best['yb']}_high_loser_rate",
        "rule_type": "interaction_bucket",
        "feature_a": x_col,
        "feature_b": y_col,
        "lower_a": best["x_min"],
        "upper_a": best["x_max"],
        "bucket_a_index": best["xb"],
        "lower_b": best["y_min"],
        "upper_b": best["y_max"],
        "bucket_b_index": best["yb"],
        "category_value": "",
        "is_loser_precision": best["precision"],
        "net_pnl_pct_impact": best["net_impact"],
        "n_flagged_is": best["n"],
    }


def best_regime_rule(g: pd.DataFrame) -> dict | None:
    data = g[["regime_entry", "is_loser", "pnl_pct"]].copy()
    data["regime_entry"] = data["regime_entry"].fillna("MISSING")
    best = None
    for regime, cell in data.groupby("regime_entry"):
        if len(cell) < 20:
            continue
        candidate = {"regime": regime, "precision": cell["is_loser"].mean(), "net_impact": -cell["pnl_pct"].sum(), "n": len(cell)}
        if best is None or (candidate["precision"], candidate["net_impact"]) > (best["precision"], best["net_impact"]):
            best = candidate
    if best is None:
        return None
    return {
        "rule_name": f"regime_{safe_name(str(best['regime']))}_high_loser_rate",
        "rule_type": "categorical",
        "feature_a": "regime_entry",
        "feature_b": "",
        "lower_a": np.nan,
        "upper_a": np.nan,
        "bucket_a_index": -1,
        "bucket_b_index": -1,
        "category_value": best["regime"],
        "is_loser_precision": best["precision"],
        "net_pnl_pct_impact": best["net_impact"],
        "n_flagged_is": best["n"],
    }


def _bucket_mask(s: pd.Series, lo: float, hi: float, idx: int) -> pd.Series:
    """Apply IS bucket boundary, extending edges to ±inf for OOS coverage."""
    if idx == 0:
        return s.le(hi)
    if idx == N_QUANTILES - 1:
        return s.ge(lo)
    return s.between(lo, hi, inclusive="both")


def apply_rule(df: pd.DataFrame, rule: pd.Series) -> pd.Series:
    typ = rule["rule_type"]
    idx_a = int(rule.get("bucket_a_index", -1) or -1)
    if typ == "univariate_bucket":
        a = rule["feature_a"]
        return _bucket_mask(df[a], rule["lower_a"], rule["upper_a"], idx_a).fillna(False)
    if typ == "interaction_bucket":
        a = rule["feature_a"]
        b = rule["feature_b"]
        idx_b = int(rule.get("bucket_b_index", -1) or -1)
        return (
            _bucket_mask(df[a], rule["lower_a"], rule["upper_a"], idx_a)
            & _bucket_mask(df[b], rule["lower_b"], rule["upper_b"], idx_b)
        ).fillna(False)
    if typ == "categorical":
        return (df[rule["feature_a"]].fillna("MISSING") == rule["category_value"]).fillna(False)
    return pd.Series(False, index=df.index)


def evaluate_rule(df: pd.DataFrame, flags: pd.Series, rule_name: str, symbol: str, period: str, benchmark: str) -> dict:
    flags = flags.reindex(df.index).fillna(False).astype(bool)
    flagged = df[flags]
    unflagged = df[~flags]
    total_losers = int(df["is_loser"].sum())
    skipped_loser_pnl = flagged.loc[flagged["is_loser"], "pnl_pct"].sum()
    skipped_winner_pnl = flagged.loc[~flagged["is_loser"], "pnl_pct"].sum()
    baseline_metrics = performance_metrics(df, "baseline")
    filtered_metrics = performance_metrics(unflagged, "filtered")
    out = {
        "symbol": symbol,
        "period": period,
        "rule_name": rule_name,
        "benchmark": benchmark,
        "n": len(df),
        "n_flagged": len(flagged),
        "trigger_rate": len(flagged) / len(df) if len(df) else np.nan,
        "precision_loser_rate_flagged": flagged["is_loser"].mean() if len(flagged) else np.nan,
        "loser_rate_unflagged": unflagged["is_loser"].mean() if len(unflagged) else np.nan,
        "recall_losers_caught": flagged["is_loser"].sum() / total_losers if total_losers else np.nan,
        "skipped_loser_pnl_avoided": -skipped_loser_pnl,
        "skipped_winner_pnl_sacrificed": skipped_winner_pnl,
        "net_pnl_pct_impact": -flagged["pnl_pct"].sum(),
    }
    for key in ["cagr", "sharpe", "sortino", "calmar", "ulcer_index", "max_drawdown"]:
        out[f"baseline_{key}"] = baseline_metrics.get(key)
        out[f"filtered_{key}"] = filtered_metrics.get(key)
        out[f"delta_{key}"] = filtered_metrics.get(key) - baseline_metrics.get(key)
    return out


def evaluate_random_filter(df: pd.DataFrame, trigger_rate: float, rule_name: str, symbol: str, period: str, rng: np.random.Generator) -> dict:
    flags = pd.Series(rng.random(len(df)) < trigger_rate, index=df.index)
    return evaluate_rule(df, flags, rule_name, symbol, period, "random_same_rate")


def performance_metrics(df: pd.DataFrame, label: str) -> dict:
    _nan = {k: np.nan for k in ["cagr", "sharpe", "sortino", "calmar", "ulcer_index", "max_drawdown"]}
    if df.empty:
        return _nan
    g = df.sort_values("entry_time").copy()
    # Per-trade return; prefer pnl/capital_before, fallback to pnl_pct/100
    trade_r = (
        (g["pnl"] / g["capital_before"].replace(0, np.nan))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(g["pnl_pct"] / 100.0)
    )
    # Compound trades that exit on the same calendar day
    exit_dates = g["exit_time"].dt.normalize()
    if exit_dates.isna().all():
        return _nan
    daily_r = (1.0 + trade_r).groupby(exit_dates).prod() - 1.0
    # Reindex to business-day calendar; days with no exits get 0 return
    start = g["entry_time"].min().normalize()
    end = g["exit_time"].max().normalize()
    bdays = pd.bdate_range(start, end)
    if len(bdays) < 2:
        return _nan
    daily_r = daily_r.reindex(bdays, fill_value=0.0)
    equity = (1.0 + daily_r).cumprod()
    n_days = len(daily_r)
    years = n_days / 252
    if years <= 0 or equity.iloc[-1] <= 0:
        return _nan
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0)   # CAGR stays in-project (n/252 basis)
    mean_d = float(daily_r.mean())
    std_d = float(daily_r.std(ddof=1))
    sharpe = qcs.sharpe(daily_r, periods_per_year=252, rf_annual=0.0) if std_d > 0 else np.nan
    neg = daily_r[daily_r < 0]
    down_std = float(neg.std(ddof=1)) if len(neg) > 1 else np.nan
    sortino = qcs.sortino(daily_r, periods_per_year=252, rf_annual=0.0, downside="std", threshold=0.0) if down_std and down_std > 0 else np.nan
    max_dd = qcs.max_drawdown(equity)
    calmar = qcs.calmar(cagr, max_dd) if max_dd < 0 else np.nan
    ulcer = qcs.ulcer_index(equity, pct=True)
    return {"cagr": cagr, "sharpe": sharpe, "sortino": sortino, "calmar": calmar, "ulcer_index": ulcer, "max_drawdown": max_dd}


def plot_rule_comparison(evals: pd.DataFrame, out_dir: Path, manifest: Manifest) -> None:
    if evals.empty:
        return
    for sym in SYMBOLS:
        data = evals[(evals["symbol"] == sym) & (evals["period"] == "OOS_2021_2026") & (evals["benchmark"] == "candidate")].copy()
        if data.empty:
            continue
        data = data.sort_values("net_pnl_pct_impact", ascending=False).head(12)
        fig, ax = plt.subplots(figsize=(11, 5.5))
        ax.barh(data["rule_name"], data["net_pnl_pct_impact"], color="#2563eb")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"{sym}: OOS net pnl_pct impact from skipping flagged trades")
        ax.set_xlabel("net pnl_pct impact")
        fig.tight_layout()
        path = out_dir / f"{sym}_oos_rule_net_impact.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        manifest.add(path, "04_candidate_rules", "OOS candidate rule net impact comparison.", sym, "candidate_rules", "bar")


def validation_outputs(evals: pd.DataFrame, df: pd.DataFrame, definitions: pd.DataFrame, out_dir: Path, manifest: Manifest) -> None:
    if evals.empty:
        return
    pivot_cols = [
        "precision_loser_rate_flagged",
        "recall_losers_caught",
        "net_pnl_pct_impact",
        "delta_calmar",
        "delta_ulcer_index",
        "delta_max_drawdown",
    ]
    rows = []
    cand = evals[evals["benchmark"] == "candidate"]
    for (sym, rule), g in cand.groupby(["symbol", "rule_name"]):
        row = {"symbol": sym, "rule_name": rule}
        for period in ["IS_2013_2020", "OOS_2021_2026"]:
            p = g[g["period"] == period]
            if p.empty:
                continue
            for col in pivot_cols:
                row[f"{period}_{col}"] = p.iloc[0][col]
        rows.append(row)
    table = pd.DataFrame(rows)
    path = out_dir / "is_oos_rule_validation.csv"
    table.to_csv(path, index=False)
    manifest.add(path, "05_validation", "Side-by-side in-sample/out-of-sample rule validation.")

    walk = walk_forward_yearly_rule_checks(df, definitions)
    path = out_dir / "walk_forward_yearly_rule_checks.csv"
    walk.to_csv(path, index=False)
    manifest.add(path, "05_validation", "Year-by-year rule stability: trigger rate, precision, and net pnl impact per calendar year.")

    plot_is_vs_oos_scatter(table, out_dir, manifest)
    plot_walk_forward_heatmap(walk, out_dir, manifest)


def walk_forward_yearly_rule_checks(df: pd.DataFrame, definitions: pd.DataFrame) -> pd.DataFrame:
    if definitions.empty or df.empty:
        return pd.DataFrame()
    data = df.copy()
    data["year"] = data["entry_time"].dt.year
    rows = []
    for _, rule in definitions.iterrows():
        sym_data = data[data["symbol"] == rule["symbol"]]
        for year, year_data in sym_data.groupby("year"):
            if year_data.empty:
                continue
            flags = apply_rule(year_data, rule)
            flagged = year_data[flags]
            total_losers = int(year_data["is_loser"].sum())
            rows.append(
                {
                    "symbol": rule["symbol"],
                    "rule_name": rule["rule_name"],
                    "year": int(year),
                    "period": year_data["period"].iloc[0],
                    "n": len(year_data),
                    "n_flagged": int(flags.sum()),
                    "trigger_rate": float(flags.mean()),
                    "precision_loser_rate_flagged": float(flagged["is_loser"].mean()) if len(flagged) else np.nan,
                    "recall_losers_caught": float(flagged["is_loser"].sum() / total_losers) if total_losers else np.nan,
                    "net_pnl_pct_impact": float(-flagged["pnl_pct"].sum()),
                }
            )
    return pd.DataFrame(rows)


def plot_is_vs_oos_scatter(table: pd.DataFrame, out_dir: Path, manifest: Manifest) -> None:
    if table.empty:
        return
    for sym, g in table.groupby("symbol"):
        if g.empty:
            continue
        fig, axes = plt.subplots(2, 1, figsize=(9, 10))

        # Precision panel
        xp = g["IS_2013_2020_precision_loser_rate_flagged"]
        yp = g["OOS_2021_2026_precision_loser_rate_flagged"]
        axes[0].scatter(xp, yp, s=40, alpha=0.7, color="#2563eb")
        axes[0].plot([0, 1], [0, 1], color="black", linewidth=0.8, linestyle="--", label="y = x (stable)")
        axes[0].axhline(0.5, color="#9ca3af", linewidth=0.5)
        axes[0].axvline(0.5, color="#9ca3af", linewidth=0.5)
        axes[0].set_xlim(0, 1)
        axes[0].set_ylim(0, 1)
        axes[0].set_xlabel("IS precision (loser rate of flagged trades)")
        axes[0].set_ylabel("OOS precision")
        axes[0].set_title(f"{sym}: IS vs OOS rule precision (above-diagonal = OOS stronger)")
        axes[0].legend(loc="best")

        # Net pnl impact panel
        xn = g["IS_2013_2020_net_pnl_pct_impact"]
        yn = g["OOS_2021_2026_net_pnl_pct_impact"]
        mn = float(min(xn.min(), yn.min(), 0))
        mx = float(max(xn.max(), yn.max(), 0))
        pad = max((mx - mn) * 0.05, 1.0)
        axes[1].scatter(xn, yn, s=40, alpha=0.7, color="#2563eb")
        axes[1].plot([mn, mx], [mn, mx], color="black", linewidth=0.8, linestyle="--", label="y = x (stable)")
        axes[1].axhline(0, color="#9ca3af", linewidth=0.6)
        axes[1].axvline(0, color="#9ca3af", linewidth=0.6)
        axes[1].set_xlim(mn - pad, mx + pad)
        axes[1].set_ylim(mn - pad, mx + pad)
        axes[1].set_xlabel("IS net pnl_pct impact (positive = rule helped)")
        axes[1].set_ylabel("OOS net pnl_pct impact")
        axes[1].set_title(f"{sym}: IS vs OOS net pnl impact (top-right quadrant = survivors)")
        axes[1].legend(loc="best")

        # Label top-3 and bottom-3 by OOS impact
        g_sorted = g.sort_values("OOS_2021_2026_net_pnl_pct_impact")
        for _, row in pd.concat([g_sorted.head(3), g_sorted.tail(3)]).drop_duplicates("rule_name").iterrows():
            axes[1].annotate(
                row["rule_name"][:32],
                xy=(row["IS_2013_2020_net_pnl_pct_impact"], row["OOS_2021_2026_net_pnl_pct_impact"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7,
                color="#374151",
            )

        fig.tight_layout()
        path = symbol_dir(out_dir, sym) / "is_vs_oos_scatter.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        manifest.add(path, "05_validation", "IS vs OOS scatter of precision and net pnl impact across all candidate rules.", sym, "all_rules", "scatter")


def plot_walk_forward_heatmap(walk: pd.DataFrame, out_dir: Path, manifest: Manifest) -> None:
    if walk.empty:
        return
    for sym, g in walk.groupby("symbol"):
        pivot = g.pivot(index="rule_name", columns="year", values="net_pnl_pct_impact")
        if pivot.empty:
            continue
        v = float(np.nanmax(np.abs(pivot.values)))
        v = max(v, 1.0)
        fig, ax = plt.subplots(figsize=(11, max(4.0, 0.45 * len(pivot))))
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=-v, vmax=v)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_xlabel("year")
        ax.set_title(f"{sym}: walk-forward net pnl_pct impact per rule per year (positive = rule helped)")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if pd.notna(val):
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, label="net pnl_pct impact (diverging, centered at 0)")
        fig.tight_layout()
        path = symbol_dir(out_dir, sym) / "walk_forward_yearly_heatmap.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        manifest.add(path, "05_validation", "Year-by-year net pnl impact heatmap for all candidate rules.", sym, "all_rules", "heatmap")


def plot_focus_dashboard(breakdown: pd.DataFrame, boot_impacts: np.ndarray, observed: float, ci_lo: float, ci_hi: float, sym: str, rule_name: str, focus_dir: Path, manifest: Manifest) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: per-year trigger bar, colored by precision
    year_rows = breakdown[breakdown["period"].astype(str).str.fullmatch(r"\d{4}")].copy()
    if not year_rows.empty:
        years = year_rows["period"].astype(str).tolist()
        n_flagged = year_rows["n_flagged"].astype(float).tolist()
        precision = year_rows["precision_loser_rate_flagged"].fillna(0.0).astype(float).tolist()
        cmap = plt.get_cmap("RdYlGn")
        colors = [cmap(p) for p in precision]
        bars = axes[0].bar(years, n_flagged, color=colors, edgecolor="black", linewidth=0.5)
        for bar, p, n in zip(bars, precision, n_flagged):
            label = f"prec={p:.0%}" if n > 0 else "no trades"
            axes[0].annotate(
                label,
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        axes[0].set_xlabel("year")
        axes[0].set_ylabel("n_flagged")
        axes[0].set_title(f"{sym} {rule_name}\nyearly trigger count (color = precision)")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        sm.set_array([])
        fig.colorbar(sm, ax=axes[0], label="precision (loser rate of flagged)")
    else:
        axes[0].text(0.5, 0.5, "no yearly breakdown rows", ha="center", va="center", transform=axes[0].transAxes)
        axes[0].set_title(f"{sym} {rule_name}: yearly trigger count")

    # Right: bootstrap histogram
    if len(boot_impacts) > 1 and not np.all(np.isnan(boot_impacts)):
        axes[1].hist(boot_impacts, bins=40, color="#93c5fd", edgecolor="#1e40af", linewidth=0.4)
        axes[1].axvline(observed, color="#dc2626", linewidth=2.0, label=f"observed = {observed:.2f}")
        axes[1].axvline(ci_lo, color="#7c3aed", linewidth=1.4, linestyle="--", label=f"CI 2.5 % = {ci_lo:.2f}")
        axes[1].axvline(ci_hi, color="#7c3aed", linewidth=1.4, linestyle="--", label=f"CI 97.5 % = {ci_hi:.2f}")
        axes[1].axvline(0, color="black", linewidth=0.8, linestyle=":")
        axes[1].set_xlabel("net pnl_pct impact (per bootstrap resample)")
        axes[1].set_ylabel("count")
        axes[1].set_title(f"OOS bootstrap distribution ({len(boot_impacts)} iters)")
        axes[1].legend(loc="best", fontsize=8)
    else:
        axes[1].text(0.5, 0.5, "no bootstrap data", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_title("OOS bootstrap distribution")

    fig.tight_layout()
    path = focus_dir / "dashboard.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    manifest.add(path, "05_validation", "Focus rule dashboard: yearly trigger bar (color = precision) + OOS bootstrap distribution.", sym, rule_name, "dashboard")


def focus_rule_validation(df: pd.DataFrame, definitions: pd.DataFrame, out_dir: Path, manifest: Manifest) -> None:
    if definitions.empty:
        return
    rng = np.random.default_rng(RANDOM_SEED)
    for sym, rule_name in FOCUS_RULES:
        match = definitions[(definitions["symbol"] == sym) & (definitions["rule_name"] == rule_name)]
        if match.empty:
            progress(f"Focus rule {sym}/{rule_name} not found in definitions — skipping.")
            continue
        rule = match.iloc[0]
        sym_data = df[df["symbol"] == sym].copy()
        sym_data["year"] = sym_data["entry_time"].dt.year

        # --- period breakdown ---
        periods: list[tuple[str, pd.DataFrame]] = [
            ("FULL", sym_data),
            ("IS_2013_2020", sym_data[sym_data["period"] == "IS_2013_2020"]),
            ("OOS_2021_2026", sym_data[sym_data["period"] == "OOS_2021_2026"]),
        ]
        for yr in sorted(sym_data[sym_data["period"] == "OOS_2021_2026"]["year"].unique()):
            periods.append((str(int(yr)), sym_data[sym_data["year"] == yr]))

        breakdown_rows = []
        for period_label, pdata in periods:
            if pdata.empty:
                continue
            flags = apply_rule(pdata, rule)
            flagged = pdata[flags]
            total_losers = int(pdata["is_loser"].sum())
            breakdown_rows.append(
                {
                    "period": period_label,
                    "n": len(pdata),
                    "n_flagged": int(flags.sum()),
                    "trigger_rate": float(flags.mean()),
                    "precision_loser_rate_flagged": float(flagged["is_loser"].mean()) if len(flagged) else np.nan,
                    "recall_losers_caught": float(flagged["is_loser"].sum() / total_losers) if total_losers else np.nan,
                    "net_pnl_pct_impact": float(-flagged["pnl_pct"].sum()),
                    "mean_flagged_pnl_pct": float(flagged["pnl_pct"].mean()) if len(flagged) else np.nan,
                    "median_flagged_pnl_pct": float(flagged["pnl_pct"].median()) if len(flagged) else np.nan,
                }
            )
        breakdown = pd.DataFrame(breakdown_rows)
        slug = safe_name(rule_name)
        focus_dir = symbol_dir(out_dir, sym) / f"focus_{slug}"
        focus_dir.mkdir(parents=True, exist_ok=True)
        path = focus_dir / "period_breakdown.csv"
        breakdown.to_csv(path, index=False)
        manifest.add(path, "05_validation", f"Period breakdown for focus rule {rule_name}.", sym, rule_name)

        # --- OOS flagged trades ---
        oos_data = sym_data[sym_data["period"] == "OOS_2021_2026"]
        oos_flags = apply_rule(oos_data, rule)
        oos_flagged = oos_data[oos_flags].copy()
        cols = [c for c in ["entry_time", "exit_time", "RSI_entry", "atr_pct", "pnl_pct", "is_loser", "year"] if c in oos_flagged.columns]
        path = focus_dir / "oos_flagged_trades.csv"
        oos_flagged[cols].to_csv(path, index=False)
        manifest.add(path, "05_validation", f"OOS flagged trades for focus rule {rule_name}.", sym, rule_name)

        # --- bootstrap on OOS net pnl impact ---
        oos_pnl = oos_flagged["pnl_pct"].values
        boot_impacts = np.array([-oos_pnl[rng.integers(0, len(oos_pnl), size=len(oos_pnl))].sum() for _ in range(BOOTSTRAP_N)]) if len(oos_pnl) > 0 else np.array([np.nan])
        observed_impact = float(-oos_pnl.sum()) if len(oos_pnl) else np.nan
        ci_lo = float(np.nanpercentile(boot_impacts, 2.5))
        ci_hi = float(np.nanpercentile(boot_impacts, 97.5))
        boot_row = {
            "n_oos_flagged": len(oos_pnl),
            "observed_net_pnl_pct_impact": observed_impact,
            "bootstrap_mean": float(np.nanmean(boot_impacts)),
            "bootstrap_ci_lo_2p5": ci_lo,
            "bootstrap_ci_hi_97p5": ci_hi,
            "n_bootstrap_iters": BOOTSTRAP_N,
        }
        path = focus_dir / "bootstrap.csv"
        pd.DataFrame([boot_row]).to_csv(path, index=False)
        manifest.add(path, "05_validation", f"Bootstrap CI on OOS net pnl impact for focus rule {rule_name}.", sym, rule_name)

        # --- dashboard plot ---
        plot_focus_dashboard(breakdown, boot_impacts, observed_impact, ci_lo, ci_hi, sym, rule_name, focus_dir, manifest)

        # --- 1-page MD summary ---
        lines = [
            f"# Focus Rule Validation: `{rule_name}`",
            f"Symbol: **{sym}** | Rule type: {rule['rule_type']}",
            "",
            "## Period breakdown",
            "",
            simple_markdown_table(breakdown),
            "",
            "## Bootstrap (OOS, {n} iters, seed={seed})".format(n=BOOTSTRAP_N, seed=RANDOM_SEED),
            "",
            simple_markdown_table(pd.DataFrame([boot_row])),
            "",
            f"## OOS flagged trades ({len(oos_flagged)} total)",
            "",
            simple_markdown_table(oos_flagged[cols].head(30)) if not oos_flagged.empty else "_None flagged._",
            "",
            "## Dashboard",
            "",
            "See `dashboard.png` in this folder.",
        ]
        path = focus_dir / "summary.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        manifest.add(path, "05_validation", f"1-page focus validation summary for {rule_name}.", sym, rule_name)


def traditional_metrics(df: pd.DataFrame, evals: pd.DataFrame, out_dir: Path, manifest: Manifest) -> None:
    rows = []
    for sym, g in df.groupby("symbol"):
        base = performance_metrics(g, "baseline")
        base.update(
            {
                "symbol": sym,
                "scenario": "baseline",
                "n": len(g),
                "total_return_chain": (1 + (g["pnl"] / g["capital_before"].replace(0, np.nan)).fillna(g["pnl_pct"] / 100.0)).prod() - 1,
                "win_rate": (~g["is_loser"]).mean(),
                "profit_factor": profit_factor(g["pnl_pct"]),
                "expectancy_pnl_pct": g["pnl_pct"].mean(),
                "var_95": g["pnl_pct"].quantile(0.05),
                "cvar_95": g.loc[g["pnl_pct"] <= g["pnl_pct"].quantile(0.05), "pnl_pct"].mean(),
                "skew": g["pnl_pct"].skew(),
                "excess_kurtosis": g["pnl_pct"].kurt(),
            }
        )
        rows.append(base)
    metrics = pd.DataFrame(rows)
    path = out_dir / "traditional_metrics_baseline.csv"
    metrics.to_csv(path, index=False)
    manifest.add(path, "05_validation", "Traditional baseline performance metrics per symbol.")


def write_memo(
    df: pd.DataFrame,
    correlations: pd.DataFrame,
    rule_evals: pd.DataFrame,
    run_dir: Path,
    reports_dir: Path,
    manifest: Manifest,
) -> None:
    lines = [
        "# Full-History Feature Scan",
        "",
        "## Scope",
        "",
        f"- Source files: `{DEFAULT_INPUT_DIR}`",
        f"- Rows: {len(df):,}",
        f"- Symbols: {', '.join(SYMBOLS)}",
        "- TQQQ and SQQQ are analyzed separately.",
        "- Primary target: `pnl_pct < 0`.",
        "- Severe-loss target: `pnl_pct <= -1%`.",
        "",
        "## Key Review Order",
        "",
        "1. `00_data_quality/feature_inclusion_exclusion.csv`",
        "2. `01_single_variable/single_variable_correlations.csv`",
        "3. `01_single_variable/single_variable_bucket_summary.csv`",
        "4. `02_interactions/` heatmaps and tables",
        "5. `03_regime_analysis/regime_summary.csv`",
        "6. `04_candidate_rules/candidate_rule_skip_impact.csv`",
        "7. `05_validation/is_oos_rule_validation.csv`",
        "",
        "## Top Feature Scores",
        "",
    ]
    for sym in SYMBOLS:
        top = correlations[correlations["symbol"] == sym].head(8)
        lines.append(f"### {sym}")
        if top.empty:
            lines.append("- No ranked features available.")
        else:
            lines.append(simple_markdown_table(top[["feature", "spearman_pnl", "pearson_pnl", "loser_rate_spread_q5", "score"]]))
        lines.append("")

    lines += ["## Candidate Rule Validation", ""]
    if rule_evals.empty:
        lines.append("- No candidate rules were generated.")
    else:
        view = rule_evals[(rule_evals["benchmark"] == "candidate") & (rule_evals["period"] == "OOS_2021_2026")]
        cols = ["symbol", "rule_name", "trigger_rate", "precision_loser_rate_flagged", "recall_losers_caught", "net_pnl_pct_impact", "delta_calmar", "delta_ulcer_index"]
        lines.append(simple_markdown_table(view[cols].sort_values(["symbol", "net_pnl_pct_impact"], ascending=[True, False]).head(20)))
    lines += [
        "",
        "## Notes",
        "",
        "- Polynomial fits are exploratory only. Treat a pattern as credible only if bucket tables and OOS validation agree.",
        "- `regime_entry` is included as a source pre-trade label, but its exact construction should be confirmed before live predictive use.",
        "- Random same-trigger-rate comparisons are included for candidate rules to expose rules that only look good because they skip many trades.",
    ]
    path = reports_dir / "full_history_research_memo.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    manifest.add(path, "06_reports", "Main review memo and suggested review order.")

    if FEATURE_DICTIONARY_PATH.exists():
        dictionary_path = reports_dir / "FEATURE_DICTIONARY.md"
        dictionary_path.write_text(FEATURE_DICTIONARY_PATH.read_text(encoding="utf-8"), encoding="utf-8")
        manifest.add(dictionary_path, "06_reports", "Plain-English dictionary for generated feature names and rule names.")

    excerpt = reports_dir / "findings_excerpt.md"
    excerpt.write_text(
        "\n".join(
            [
                "## Full-history feature scan",
                "",
                f"- Research pack: `{run_dir}`",
                "- Data source: six CSV files in `full_history_canonical/trades_backtest/`.",
                "- TQQQ and SQQQ analyzed separately.",
                "- Added RSI/ATR/BBP/volume/MA feature scans, nonlinear fits, interactions, regime tables, candidate loser rules, and IS/OOS validation.",
            ]
        ),
        encoding="utf-8",
    )
    manifest.add(excerpt, "06_reports", "Short findings excerpt for appending to FINDINGS.md.")


def simple_markdown_table(df: pd.DataFrame) -> str:
    """Render a compact markdown table without pandas' optional tabulate dependency."""
    if df.empty:
        return ""
    view = df.copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        else:
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else str(x))
    headers = [str(c) for c in view.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in view.columns) + " |")
    return "\n".join(lines)


def update_docs(run_dir: Path) -> None:
    findings = PROJ / "archive" / "FINDINGS.md"
    text = findings.read_text(encoding="utf-8")
    marker = "## Full-history feature scan"
    addition = (
        "\n## Full-history feature scan (2026-05-28)\n\n"
        f"- Implemented organized research pack at `{run_dir}`.\n"
        "- Uses the six 2013-2026 CSV files in `full_history_canonical/trades_backtest/` as source of truth.\n"
        "- Analyzes TQQQ and SQQQ separately across pre-trade features, correlations, bucket tables, nonlinear fits, interactions, regimes, candidate loser rules, and IS/OOS validation.\n"
        "- `RSI_DATA_READINESS.md` was removed because the CSV files contain `RSI_entry` and `atr`.\n"
    )
    if marker not in text:
        findings.write_text(text.rstrip() + "\n" + addition, encoding="utf-8")

    # PROJECT_PLAN.md is now authoritative and human-maintained.
    # Do not auto-mutate it from the pipeline.

    stale = PROJ / "RSI_DATA_READINESS.md"
    if stale.exists():
        stale.unlink()


def main() -> None:
    args = parse_args()
    run_dir = args.output_root / args.run_id
    dirs = make_dirs(run_dir)
    manifest = Manifest(rows=[])

    progress("Loading and transforming full-history CSV trades")
    trades = add_features(load_trades(args.input_dir))

    progress("Writing data-quality and leakage audits")
    audit_schema(trades, dirs["quality"], manifest)
    numeric_features, categorical_features, _ = build_feature_audit(trades, dirs["quality"], manifest)

    progress("Running single-variable discovery")
    correlations, _ = single_variable_tables(trades, numeric_features, categorical_features, dirs["single"], manifest)

    progress("Creating single-variable and nonlinear plots")
    make_single_variable_plots(trades, correlations, numeric_features, dirs["single"], manifest)

    progress("Building interaction heatmaps and tables")
    interaction_tables_and_plots(trades, dirs["interactions"], manifest)

    progress("Running regime analysis")
    regime_analysis(trades, dirs["regime"], manifest)

    progress("Discovering and evaluating candidate loser rules")
    definitions, rule_evals = discover_candidate_rules(trades, correlations, dirs["rules"], manifest)

    progress("Writing validation and traditional metrics")
    validation_outputs(rule_evals, trades, definitions, dirs["validation"], manifest)
    traditional_metrics(trades, rule_evals, dirs["validation"], manifest)

    progress("Running focused rule validation")
    focus_rule_validation(trades, definitions, dirs["validation"], manifest)

    progress("Writing memo, manifest, and updating docs")
    write_memo(trades, correlations, rule_evals, run_dir, dirs["reports"], manifest)
    manifest_path = run_dir / "manifest.csv"
    manifest.write(manifest_path)
    update_docs(run_dir)

    print(f"Research pack written to: {run_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
