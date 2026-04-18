"""
Visualization helpers for qframe research results.

All functions accept a knowledge_base path (or pre-loaded DataFrame) and return
a matplotlib Figure — they do NOT call plt.show() so they work cleanly in
notebooks and in scripts.

Usage:
    from qframe.viz.charts import (
        plot_ic_decay_curves, plot_ic_vs_icir, plot_ic1_vs_ic63,
        plot_cumulative_ic, plot_slow_icir_comparison,
        plot_correlation_heatmap, plot_turnover_scatter,
        plot_sharpe_histogram, plot_domain_breakdown, plot_error_rate,
        plot_leaderboard,
    )
    fig = plot_ic_decay_curves(kb_path='knowledge_base/qframe.db')
    fig.savefig('decay_curves.png', dpi=150, bbox_inches='tight')
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from qframe.factor_harness import DEFAULT_OOS_START

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_DEFAULT_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]

_FACTOR_TYPE_COLORS = {
    "momentum":      "#4C72B0",
    "mean_reversion": "#DD8452",
    "volatility":    "#55A868",
    "quality":       "#C44E52",
    "value":         "#8172B2",
    "macro":         "#937860",
    "microstructure": "#DA8BC3",
}


def _load_results(kb_path: str | Path) -> pd.DataFrame:
    """Load full results table joined with hypothesis and implementation info."""
    conn = sqlite3.connect(str(kb_path))
    df = pd.read_sql("""
        SELECT
            r.*,
            h.factor_name,
            h.description,
            h.mechanism_score,
            h.status      AS hyp_status,
            i.notes       AS impl_notes,
            i.code,
            i.created_at  AS impl_created_at
        FROM backtest_results r
        JOIN implementations i ON i.id = r.implementation_id
        JOIN hypotheses h ON h.id = i.hypothesis_id
        ORDER BY r.ic DESC NULLS LAST
    """, conn)
    conn.close()
    # Parse created_at as datetime for time-series plots
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    return df


def _load_correlations(kb_path: str | Path) -> pd.DataFrame:
    conn = sqlite3.connect(str(kb_path))
    df = pd.read_sql(
        "SELECT * FROM factor_correlations ORDER BY ABS(correlation) DESC",
        conn,
    )
    conn.close()
    return df


def _short_name(row: pd.Series, max_len: int = 30) -> str:
    """Return factor_name if available, else truncated description."""
    name = row.get("factor_name") or ""
    if name and str(name) != "nan":
        return str(name)
    desc = str(row.get("description", ""))
    return desc[:max_len] + ("…" if len(desc) > max_len else "")


def _factor_type_from_notes(notes: str | None) -> str:
    if not notes:
        return "unknown"
    parts = str(notes).split("|")
    for p in parts:
        if p.startswith("factor_type="):
            return p.split("=", 1)[1].strip()
    return "unknown"


# ---------------------------------------------------------------------------
# Chart 1 — Leaderboard: IC and Sharpe horizontal bar charts
# ---------------------------------------------------------------------------

def plot_leaderboard(
    kb_path: str | Path,
    top_n: int = 20,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """
    Horizontal bar chart: top_n factors ranked by mean OOS IC.
    Left panel = IC, right panel = IC Sharpe.

    Each factor is shown once — the best result (highest IC) per factor_name
    is kept. Without this deduplication, factors that were run multiple times
    (e.g. after a code fix) show up with both their positive and negative results
    as separate bars in the same chart.
    """
    df = _load_results(kb_path)

    # Deduplicate: keep only the best IC result per factor_name.
    # Fall back to implementation_id when factor_name is missing/null.
    dedup_key = df["factor_name"].where(df["factor_name"].notna(), "impl_" + df["implementation_id"].astype(str))
    df = (
        df.assign(_dedup_key=dedup_key)
        .sort_values("ic", ascending=False)
        .drop_duplicates(subset="_dedup_key", keep="first")
        .drop(columns="_dedup_key")
        .reset_index(drop=True)
    )

    df = df.head(top_n)
    labels = [_short_name(r) for _, r in df.iterrows()]

    fig_h = max(4.0, len(df) * 0.45 + 1.0)
    fig, axes = plt.subplots(1, 2, figsize=figsize or (16, fig_h))

    def _bar(ax, values, title, threshold_lines=None):
        colors = ["#4C72B0" if v >= 0 else "#C44E52" for v in values]
        bars = ax.barh(range(len(labels)), values, color=colors, height=0.7)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.axvline(0, color="k", lw=0.8)
        if threshold_lines:
            for xv, lbl, col in threshold_lines:
                ax.axvline(xv, color=col, lw=1.2, ls="--", label=lbl)
            ax.legend(fontsize=7, loc="lower right")
        ax.set_title(title, fontsize=11)
        ax.invert_yaxis()
        # Annotate values on bars
        for i, (bar, val) in enumerate(zip(bars, values)):
            if np.isfinite(val):
                ax.text(
                    val + 0.0002 * np.sign(val) if val >= 0 else val - 0.0002,
                    i, f"{val:+.4f}", va="center", ha="left" if val >= 0 else "right",
                    fontsize=7,
                )

    _bar(
        axes[0], df["ic"].fillna(0).tolist(),
        "Mean OOS IC — top factors",
        threshold_lines=[
            (0.015, "weak gate (0.015)", "#2ca02c"),
            (0.030, "pass gate (0.030)", "#006400"),
        ],
    )
    _bar(axes[1], df["sharpe"].fillna(0).tolist(), "IC Sharpe")

    fig.suptitle("Factor Leaderboard", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Chart 2 — IC Decay Curves (individual lines per factor)
# ---------------------------------------------------------------------------

def plot_ic_decay_curves(
    kb_path: str | Path,
    top_n: int = 10,
    figsize: tuple[float, float] = (12, 6),
) -> plt.Figure:
    """
    Line chart: IC decay curve for each of the top_n factors.
    x-axis = horizon (days 1..63), y-axis = mean IC.
    Factors where IC increases with horizon are highlighted (mean-reversion type).

    Falls back to sparse horizons [1, 5, 21, 63] when ic_decay_json is not stored
    (results produced before the harness upgrade of 2026-04-13).
    """
    df = _load_results(kb_path)
    has_json = "ic_decay_json" in df.columns
    if has_json:
        df = df[df["ic_decay_json"].notna()].head(top_n)
    else:
        df = df.dropna(subset=["ic_horizon_1"]).head(top_n)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axhline(0, color="k", lw=0.8, ls="--")

    palette = _DEFAULT_PALETTE

    for idx, (_, row) in enumerate(df.iterrows()):
        name = _short_name(row)
        # Build decay dict from either dense JSON or sparse horizon columns
        decay_dict: dict[str, float] = {}
        if has_json and pd.notna(row.get("ic_decay_json")):
            try:
                decay_dict = {str(k): v for k, v in json.loads(row["ic_decay_json"]).items()}
            except Exception:
                pass
        if not decay_dict:
            # Fallback: use sparse horizon columns
            for h, col in [(1, "ic_horizon_1"), (5, "ic_horizon_5"),
                           (21, "ic_horizon_21"), (63, "ic_horizon_63")]:
                v = row.get(col)
                if pd.notna(v):
                    decay_dict[str(h)] = float(v)
        if not decay_dict:
            continue

        horizons = sorted(int(k) for k in decay_dict.keys())
        ic_vals = [decay_dict[str(h)] for h in horizons]

        # Detect slow-signal type: |IC| increases with horizon.
        # Option A — same-sign slow signal: |IC@63| > |IC@1| AND same sign.
        # A sign flip between horizons (e.g. IC@1d = -0.01, IC@63d = +0.04) means
        # two different economic effects are present — not a clean slow signal.
        ic1 = decay_dict.get("1", decay_dict.get(str(min(int(k) for k in decay_dict)), np.nan))
        ic63 = decay_dict.get("63", decay_dict.get(str(max(int(k) for k in decay_dict)), np.nan))
        try:
            _ic1, _ic63 = float(ic1), float(ic63)
            is_slow = (
                abs(_ic63) > abs(_ic1)
                and _ic1 != 0 and _ic63 != 0
                and (_ic1 * _ic63 > 0)   # same sign
            )
        except (TypeError, ValueError):
            is_slow = False

        color = palette[idx % len(palette)]
        lw = 2.0 if is_slow else 1.2
        ls = "-" if is_slow else "--"
        label = f"[SLOW] {name}" if is_slow else name

        ax.plot(horizons, ic_vals, color=color, lw=lw, ls=ls, label=label, alpha=0.85)

    ax.set_xlabel("Horizon (trading days)", fontsize=10)
    ax.set_ylabel("Mean IC", fontsize=10)
    ax.set_title(
        f"IC Decay Curves — top {min(top_n, len(df))} factors\n"
        "Solid = slow (mean-reversion) type; dashed = normal decay",
        fontsize=11,
    )
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(7))
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Chart 3 — IC Decay Heatmap with slow-signal highlighting
# ---------------------------------------------------------------------------

def plot_ic_decay_heatmap(
    kb_path: str | Path,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """
    Heatmap of IC at horizons 1, 5, 21, 63 for all factors.
    Cells where IC increases with horizon are outlined in orange.
    """
    try:
        import seaborn as sns
    except ImportError:
        raise ImportError("seaborn is required: pip install seaborn")

    df = _load_results(kb_path)
    decay_cols = ["ic_horizon_1", "ic_horizon_5", "ic_horizon_21", "ic_horizon_63"]
    available = [c for c in decay_cols if c in df.columns]
    if not available:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No decay data available", ha="center", va="center")
        return fig

    decay_data = df[available].copy()
    decay_data.index = [_short_name(r) for _, r in df.iterrows()]
    decay_data.columns = [c.replace("ic_horizon_", "") + "d" for c in available]

    fig_h = max(4.0, len(decay_data) * 0.45 + 1.5)
    fig, ax = plt.subplots(figsize=figsize or (9, fig_h))
    sns.heatmap(
        decay_data, annot=True, fmt=".3f", center=0,
        cmap="RdYlGn", ax=ax, linewidths=0.5, cbar_kws={"shrink": 0.6},
    )

    # Outline cells where |IC@63d| > |IC@1d| (slow-signal pattern).
    # Using absolute value correctly identifies both long and short slow signals:
    # Option A — same-sign slow signal: |IC@63| > |IC@1| AND same sign.
    # A sign flip between horizons indicates two different effects are at play,
    # not a clean slow/mean-reversion signal.
    if "63d" in decay_data.columns and "1d" in decay_data.columns:
        ic1  = decay_data["1d"].fillna(0)
        ic63 = decay_data["63d"].fillna(0)
        slow_rows = decay_data.index[
            (ic63.abs() > ic1.abs()) & (np.sign(ic63) == np.sign(ic1)) & (ic1 != 0)
        ]
        col_idx = list(decay_data.columns).index("63d")
        for row_i, label in enumerate(decay_data.index):
            if label in slow_rows:
                ax.add_patch(plt.Rectangle(
                    (col_idx, row_i), 1, 1,
                    fill=False, edgecolor="darkorange", lw=2.5,
                ))

    ax.set_title(
        "IC Decay Heatmap — all factors\n"
        "Orange outline = |IC| grows with horizon, same sign (clean slow/mean-reversion signal)",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Chart 4 — IC vs ICIR scatter (efficiency frontier)
# ---------------------------------------------------------------------------

def plot_ic_vs_icir(
    kb_path: str | Path,
    figsize: tuple[float, float] = (9, 7),
) -> plt.Figure:
    """
    Scatter: x = mean IC, y = ICIR.
    Dot size ∝ turnover. Colour = factor_type.
    Gate thresholds drawn as reference lines.
    """
    df = _load_results(kb_path)
    df = df.dropna(subset=["ic", "icir"])

    fig, ax = plt.subplots(figsize=figsize)

    for ft, group in df.groupby("impl_notes"):
        ft_parsed = _factor_type_from_notes(ft)
        color = _FACTOR_TYPE_COLORS.get(ft_parsed, "#8C8C8C")
        sizes = (group["turnover"].fillna(0.05) * 1500).clip(20, 600)
        ax.scatter(
            group["ic"], group["icir"],
            c=color, s=sizes, alpha=0.75, edgecolors="white", lw=0.5,
            label=ft_parsed,
        )

    # Gate lines
    ax.axvline(0.015, color="#2ca02c", ls="--", lw=1.2, label="weak IC=0.015")
    ax.axvline(0.030, color="#006400", ls="--", lw=1.2, label="pass IC=0.030")
    ax.axhline(0.15,  color="#2ca02c", ls=":",  lw=1.2, label="weak ICIR=0.15")
    ax.axhline(0.40,  color="#006400", ls=":",  lw=1.2, label="pass ICIR=0.40")
    ax.axvline(0, color="k", lw=0.5)
    ax.axhline(0, color="k", lw=0.5)

    # Annotate top factors
    top = df.nlargest(5, "ic")
    for _, r in top.iterrows():
        ax.annotate(
            _short_name(r, max_len=18),
            (r["ic"], r["icir"]),
            fontsize=7, xytext=(4, 4), textcoords="offset points",
        )

    # Deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    seen: dict[str, int] = {}
    uniq_h, uniq_l = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            uniq_h.append(h)
            uniq_l.append(l)
            seen[l] = 1
    ax.legend(uniq_h, uniq_l, fontsize=8, loc="upper left")
    ax.set_xlabel("Mean OOS IC", fontsize=10)
    ax.set_ylabel("ICIR (rolling 63d)", fontsize=10)
    ax.set_title(
        "IC vs ICIR — efficiency frontier\n"
        "dot size ∝ daily turnover  |  colour = factor type",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Chart 5 — IC at 1-day vs IC at 63-day scatter
# ---------------------------------------------------------------------------

def plot_ic1_vs_ic63(
    kb_path: str | Path,
    figsize: tuple[float, float] = (8, 7),
) -> plt.Figure:
    """
    Scatter: x = IC@1d, y = IC@63d.
    Factors above the diagonal are slow (mean-reversion); below are fast (momentum).
    """
    df = _load_results(kb_path)
    df = df.dropna(subset=["ic_horizon_1", "ic_horizon_63"])

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(df["ic_horizon_1"], df["ic_horizon_63"], alpha=0.75,
               color="#4C72B0", s=60, edgecolors="white", lw=0.5)

    # Diagonal y = x
    lim = max(
        abs(df[["ic_horizon_1", "ic_horizon_63"]].values).max() * 1.1,
        0.01,
    )
    diag = np.linspace(-lim, lim, 100)
    ax.plot(diag, diag, "k--", lw=1.0, label="y = x (no decay)")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)

    # Shade slow-signal region (Option A — same-sign only):
    # A point is a slow signal when |IC@63d| > |IC@1d| AND the signs match.
    # This corresponds to:
    #   Q1 (both positive): above the y=x line → shade above y=x for x>0
    #   Q3 (both negative): below the y=x line → shade below y=x for x<0
    # Cross-sign quadrants (Q2, Q4) are NOT shaded — sign flips are a
    # different phenomenon, not a clean slow signal.
    pos = diag[diag >= 0]
    neg = diag[diag <= 0]
    ax.fill_between(pos, pos, lim, alpha=0.07, color="orange")   # Q1 slow region
    ax.fill_between(neg, neg, -lim, alpha=0.07, color="orange",
                    label="slow signal region (|IC@63d| > |IC@1d|, same sign)")
    ax.fill_between(diag, diag, np.where(diag >= 0, -lim, lim), alpha=0.03,
                    color="steelblue",
                    label="fast/decaying region")

    # Annotate outliers (far from diagonal)
    df["diff"] = (df["ic_horizon_63"] - df["ic_horizon_1"]).abs()
    for _, r in df.nlargest(5, "diff").iterrows():
        ax.annotate(
            _short_name(r, max_len=18),
            (r["ic_horizon_1"], r["ic_horizon_63"]),
            fontsize=7, xytext=(4, 4), textcoords="offset points",
        )

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("IC at 1-day horizon", fontsize=10)
    ax.set_ylabel("IC at 63-day horizon", fontsize=10)
    ax.set_title(
        "IC@1d vs IC@63d\n"
        "Orange = slow signal (|IC@63d| > |IC@1d|, same sign); blue = fast/decaying; cross-sign = mixed effects",
        fontsize=11,
    )
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Chart 6 — Cumulative IC time series for top N factors
# ---------------------------------------------------------------------------

def plot_cumulative_ic(
    kb_path: str | Path,
    top_n: int = 5,
    oos_start: str = DEFAULT_OOS_START,
    figsize: tuple[float, float] = (14, 6),
) -> plt.Figure:
    """
    Cumulative sum of daily OOS IC for the top_n factors ranked by mean IC.
    A straight upward line = consistent signal. Lumpy = episodic alpha.

    NOTE: Requires the ic_series to be re-computed from stored code (not stored in DB).
    This chart instead uses the stored scalar IC decay data and approximates
    cumulative IC as a conceptual illustration using the 4 key horizons.
    For the true time-series version, use result.wf_result.ic_series directly.
    """
    df = _load_results(kb_path)
    df = df[df["ic"].notna()].head(top_n)

    # Each factor's decay curve is stored as JSON; plot horizon IC as a mini-bar
    fig, ax = plt.subplots(figsize=figsize)
    palette = _DEFAULT_PALETTE
    horizons = [1, 5, 21, 63]

    for idx, (_, row) in enumerate(df.iterrows()):
        name = _short_name(row, max_len=25)
        vals = []
        for h in horizons:
            col = f"ic_horizon_{h}"
            v = row.get(col, np.nan)
            vals.append(float(v) if pd.notna(v) else 0.0)

        # Cumulative IC across horizons (illustrative)
        cum_vals = np.cumsum(vals)
        ax.plot(horizons, cum_vals, "o-", color=palette[idx % len(palette)],
                lw=1.8, ms=5, label=name)

    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Horizon (days)", fontsize=10)
    ax.set_ylabel("Cumulative IC (sum across horizons)", fontsize=10)
    ax.set_title(
        f"Cumulative IC across horizons — top {min(top_n, len(df))} factors\n"
        "Upward slope = IC builds with horizon (slow signal). Flat = already captured at 1d.",
        fontsize=11,
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xticks(horizons)
    ax.set_xticklabels(["1d", "5d", "21d", "63d"])
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Chart 7 — Slow ICIR comparison (21d vs 63d vs standard)
# ---------------------------------------------------------------------------

def plot_slow_icir_comparison(
    kb_path: str | Path,
    top_n: int = 15,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """
    Grouped bar chart: standard ICIR vs slow_icir_21 vs slow_icir_63 for top factors.
    Makes it immediately visible which factors look better at longer horizons.
    """
    df = _load_results(kb_path)
    df = df[df["ic"].notna()].head(top_n)

    has_slow21 = "slow_icir_21" in df.columns
    has_slow63 = "slow_icir_63" in df.columns

    labels = [_short_name(r) for _, r in df.iterrows()]
    n = len(df)
    x = np.arange(n)
    w = 0.27

    fig_h = max(5.0, n * 0.5 + 1.5)
    fig, ax = plt.subplots(figsize=figsize or (max(12, n * 0.9), fig_h))

    ax.bar(x - w, df["icir"].fillna(0), w, label="ICIR (1d rolling)", color="#4C72B0", alpha=0.85)
    if has_slow21:
        ax.bar(x,     df["slow_icir_21"].fillna(0), w, label="Slow ICIR 21d (monthly)", color="#DD8452", alpha=0.85)
    if has_slow63:
        ax.bar(x + w, df["slow_icir_63"].fillna(0), w, label="Slow ICIR 63d (quarterly)", color="#55A868", alpha=0.85)

    ax.axhline(0,    color="k", lw=0.8)
    ax.axhline(0.15, color="#2ca02c", ls="--", lw=1.0, label="weak ICIR=0.15")
    ax.axhline(0.40, color="#006400", ls="--", lw=1.0, label="pass ICIR=0.40")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("ICIR", fontsize=10)
    ax.set_title(
        "Standard ICIR vs Slow ICIR (21d / 63d non-overlapping windows)\n"
        "Slow ICIR is the honest metric for mean-reversion and slow-decay factors",
        fontsize=11,
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Chart 8 — Turnover vs IC scatter
# ---------------------------------------------------------------------------

def plot_turnover_scatter(
    kb_path: str | Path,
    figsize: tuple[float, float] = (8, 6),
) -> plt.Figure:
    """
    Scatter: x = daily turnover, y = mean IC.
    Shows the trade-off between signal quality and trading costs.
    """
    df = _load_results(kb_path)
    df = df.dropna(subset=["ic", "turnover"])

    fig, ax = plt.subplots(figsize=figsize)

    colors = ["#4C72B0" if v >= 0 else "#C44E52" for v in df["ic"]]
    ax.scatter(df["turnover"], df["ic"], c=colors, s=50, alpha=0.75, edgecolors="white")

    # Cost break-even line: net_ic ≈ 0 when IC ≈ RT_cost × turnover
    rt_cost_bps = 20.0  # 10 bps one-way round-trip → 20 bps round-trip
    to_range = np.linspace(0, df["turnover"].max() * 1.1, 200)
    break_even_ic = rt_cost_bps / 10_000 * to_range
    ax.plot(to_range, break_even_ic, "r--", lw=1.2,
            label=f"Break-even line (RT cost = {rt_cost_bps} bps)")

    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Daily one-way turnover", fontsize=10)
    ax.set_ylabel("Mean OOS IC", fontsize=10)
    ax.set_title(
        "Turnover vs IC — signal quality vs cost trade-off\n"
        "Factors below the red line are net-negative after costs",
        fontsize=11,
    )
    ax.legend(fontsize=8)

    # Annotate top IC factors
    for _, r in df.nlargest(5, "ic").iterrows():
        ax.annotate(
            _short_name(r, max_len=15),
            (r["turnover"], r["ic"]),
            fontsize=7, xytext=(4, 4), textcoords="offset points",
        )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Chart 9 — Factor correlation heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(
    kb_path: str | Path,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """
    Heatmap of pairwise factor rank correlations from the factor_correlations table.
    Returns a note figure if the table is empty (run run_correlation_analysis() first).
    """
    try:
        import seaborn as sns
    except ImportError:
        raise ImportError("seaborn is required: pip install seaborn")

    df = _load_correlations(kb_path)

    fig, ax = plt.subplots(figsize=figsize or (8, 7))

    if df.empty:
        ax.text(
            0.5, 0.5,
            "Factor correlations table is empty.\n"
            "Run loop.run_correlation_analysis() first.",
            ha="center", va="center", fontsize=10,
            transform=ax.transAxes,
        )
        ax.set_title("Factor Correlation Heatmap (no data yet)", fontsize=11)
        fig.tight_layout()
        return fig

    # Pivot to square matrix
    all_factors = sorted(set(df["factor_a"].tolist() + df["factor_b"].tolist()))
    mat = pd.DataFrame(np.eye(len(all_factors)), index=all_factors, columns=all_factors)
    for _, row in df.iterrows():
        mat.loc[row["factor_a"], row["factor_b"]] = row["correlation"]
        mat.loc[row["factor_b"], row["factor_a"]] = row["correlation"]

    n = len(mat)
    fig_size = figsize or (max(7, n * 0.7), max(6, n * 0.6))
    fig.set_size_inches(fig_size)

    # Shorten labels
    short_labels = [
        (lbl[:18] + "…" if len(lbl) > 18 else lbl) for lbl in mat.index
    ]
    mat.index = short_labels
    mat.columns = short_labels

    mask = np.triu(np.ones_like(mat.values, dtype=bool), k=1)
    sns.heatmap(
        mat, mask=mask, annot=True, fmt=".2f", center=0,
        cmap="RdBu_r", ax=ax, linewidths=0.5, vmin=-1, vmax=1,
        cbar_kws={"shrink": 0.6},
    )
    ax.set_title(
        "Pairwise Factor Rank Correlations (Spearman)\n"
        "Low values = factors bring independent information",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Chart 10 — Sharpe histogram
# ---------------------------------------------------------------------------

def plot_sharpe_histogram(
    kb_path: str | Path,
    figsize: tuple[float, float] = (8, 5),
) -> plt.Figure:
    """
    Histogram of IC-Sharpe across all factors.
    Shows how the distribution of factor quality looks.
    """
    df = _load_results(kb_path)
    df = df.dropna(subset=["sharpe"])

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(df["sharpe"], bins=20, color="#4C72B0", alpha=0.8, edgecolor="white")
    ax.axvline(0, color="k", lw=1.0)
    ax.axvline(df["sharpe"].mean(), color="tomato", lw=1.5, ls="--",
               label=f"Mean = {df['sharpe'].mean():.2f}")
    ax.set_xlabel("IC Sharpe", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Distribution of IC Sharpe across all tested factors", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Chart 11 — Domain breakdown (pass / fail counts by domain)
# ---------------------------------------------------------------------------

def plot_domain_breakdown(
    kb_path: str | Path,
    figsize: tuple[float, float] = (10, 5),
) -> plt.Figure:
    """
    Stacked bar: for each domain, how many factors passed WEAK gate vs failed.
    Also shows mean IC per domain.
    """
    df = _load_results(kb_path)

    # Extract factor_type from impl_notes
    df["factor_type"] = df["impl_notes"].apply(_factor_type_from_notes)

    grouped = df.groupby("factor_type").agg(
        count=("ic", "count"),
        mean_ic=("ic", "mean"),
        positive_ic=("ic", lambda x: (x > 0).sum()),
        weak_gate=("ic", lambda x: (x >= 0.015).sum()),
    ).reset_index()
    grouped = grouped.sort_values("mean_ic", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: count by domain
    x = np.arange(len(grouped))
    axes[0].bar(x, grouped["count"], color="#4C72B0", alpha=0.85, label="Total tested", width=0.5)
    axes[0].bar(x, grouped["positive_ic"], color="#55A868", alpha=0.85, label="Positive IC", width=0.4)
    axes[0].bar(x, grouped["weak_gate"], color="#DD8452", alpha=0.9, label="IC ≥ 0.015 (weak gate)", width=0.3)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(grouped["factor_type"], rotation=30, ha="right", fontsize=9)
    axes[0].set_ylabel("Count")
    axes[0].set_title("Factors tested per domain")
    axes[0].legend(fontsize=8)

    # Right: mean IC per domain
    colors = [_FACTOR_TYPE_COLORS.get(ft, "#8C8C8C") for ft in grouped["factor_type"]]
    axes[1].bar(x, grouped["mean_ic"], color=colors, alpha=0.85, width=0.6)
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].axhline(0.015, color="#2ca02c", ls="--", lw=1.0, label="weak gate IC=0.015")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(grouped["factor_type"], rotation=30, ha="right", fontsize=9)
    axes[1].set_ylabel("Mean OOS IC")
    axes[1].set_title("Mean IC by domain")
    axes[1].legend(fontsize=8)

    fig.suptitle("Domain Breakdown — factor quality by type", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Chart 12 — Error rate over time
# ---------------------------------------------------------------------------

def plot_error_rate(
    kb_path: str | Path,
    figsize: tuple[float, float] = (11, 5),
) -> plt.Figure:
    """
    Bar chart showing pass/fail/error counts over time (grouped by calendar date).
    Shows whether implementation quality improved as prompt engineering evolved.
    """
    conn = sqlite3.connect(str(kb_path))
    hyp = pd.read_sql("""
        SELECT h.status, h.created_at,
               CASE WHEN i.id IS NULL THEN 'no_code' ELSE h.status END AS outcome
        FROM hypotheses h
        LEFT JOIN implementations i ON i.hypothesis_id = h.id
        ORDER BY h.created_at
    """, conn)
    conn.close()

    hyp["created_at"] = pd.to_datetime(hyp["created_at"], errors="coerce")
    hyp["date"] = hyp["created_at"].dt.date

    pivot = hyp.groupby(["date", "status"]).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=figsize)

    dates = [str(d) for d in pivot.index]
    x = np.arange(len(dates))
    w = 0.25

    status_config = [
        ("passed", "#55A868", "Passed"),
        ("failed", "#C44E52", "Failed / Error"),
        ("active", "#4C72B0", "Active (pending)"),
    ]

    for i, (col, color, label) in enumerate(status_config):
        if col in pivot.columns:
            ax.bar(x + (i - 1) * w, pivot[col], w, color=color, alpha=0.85, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(dates, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Count")
    ax.set_title(
        "Hypothesis outcomes over time\n"
        "Expect fail rate to decrease as prompt engineering matures",
        fontsize=11,
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Chart 13 — Net IC vs Gross IC (cost drag visualisation)
# ---------------------------------------------------------------------------

def plot_net_vs_gross_ic(
    kb_path: str | Path,
    top_n: int = 15,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """
    Grouped bar chart: gross IC vs net IC for top factors.
    The gap shows how much IC is consumed by transaction costs.
    """
    df = _load_results(kb_path)
    df = df.dropna(subset=["ic"]).head(top_n)

    labels = [_short_name(r) for _, r in df.iterrows()]
    n = len(df)
    x = np.arange(n)
    w = 0.35

    fig_h = max(5.0, n * 0.5)
    fig, ax = plt.subplots(figsize=figsize or (max(12, n * 0.85), fig_h))

    ax.bar(x - w / 2, df["ic"].fillna(0), w, label="Gross IC", color="#4C72B0", alpha=0.85)
    ax.bar(x + w / 2, df["net_ic"].fillna(0), w, label="Net IC (after costs)", color="#55A868", alpha=0.85)

    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(0.015, color="#2ca02c", ls="--", lw=1.0, label="weak gate IC=0.015")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("IC")
    ax.set_title(
        "Gross IC vs Net IC — cost drag per factor\n"
        "Gap = IC consumed by transaction costs (spread + market impact)",
        fontsize=11,
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Chart 14 — IC by calendar period (temporal stability)
# ---------------------------------------------------------------------------

def plot_ic_by_period(
    kb_path: str | Path,
    impl_id: int | None = None,
    factor_name: str | None = None,
    prices_path: str | Path | None = None,
    oos_start: str = DEFAULT_OOS_START,
    period_years: float = 2.0,
    horizon: int = 1,
    figsize: tuple[float, float] = (11, 5),
) -> plt.Figure:
    """
    Bar chart: mean IC in consecutive OOS sub-periods (temporal stability diagnostic).

    The chart shows whether a factor's IC was consistent across different
    market environments (e.g. 2018–2020, 2020–2022, 2022–2024) or only
    worked in one particular era.

    Requires the factor code to be re-executed on price data.
    Either supply impl_id (from the KB) or factor_name to look it up.

    Args:
        kb_path:      Path to knowledge_base/qframe.db.
        impl_id:      Implementation ID to look up code (preferred).
        factor_name:  Hypothesis factor_name (fallback lookup).
        prices_path:  Path to sp500_close.parquet. Defaults to
                      'data/processed/sp500_close.parquet' relative to repo root.
        oos_start:    OOS evaluation start date.
        period_years: Length of each sub-period in years.
        horizon:      Forward return horizon in days.
        figsize:      Figure size.

    Returns:
        matplotlib.Figure with period bars (mean IC ± 95% CI envelope).
    """
    from pathlib import Path as _Path
    import pandas as _pd
    import numpy as _np

    # --- Resolve prices path ---
    if prices_path is None:
        here = _Path(__file__).resolve()
        candidate = here
        for _ in range(8):
            candidate = candidate.parent
            p = candidate / "data" / "processed" / "sp500_close.parquet"
            if p.exists():
                prices_path = p
                break
    if prices_path is None or not _Path(str(prices_path)).exists():
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Price data not found.\nSupply prices_path= argument.",
                ha="center", va="center", fontsize=11, transform=ax.transAxes)
        ax.set_title("IC by Period — price data missing", fontsize=11)
        return fig

    # --- Load prices & returns ---
    prices = _pd.read_parquet(str(prices_path))
    returns = prices.pct_change()

    # --- Look up factor code ---
    conn = sqlite3.connect(str(kb_path))
    if impl_id is not None:
        row = conn.execute(
            "SELECT i.code, h.description, h.factor_name "
            "FROM implementations i JOIN hypotheses h ON h.id = i.hypothesis_id "
            "WHERE i.id = ?", (impl_id,)
        ).fetchone()
    elif factor_name is not None:
        row = conn.execute(
            "SELECT i.code, h.description, h.factor_name "
            "FROM implementations i JOIN hypotheses h ON h.id = i.hypothesis_id "
            "WHERE h.factor_name = ? ORDER BY i.id DESC LIMIT 1", (factor_name,)
        ).fetchone()
    else:
        conn.close()
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Provide impl_id= or factor_name= to identify factor.",
                ha="center", va="center", fontsize=11, transform=ax.transAxes)
        return fig
    conn.close()

    if row is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"No factor found for impl_id={impl_id} / factor_name={factor_name}.",
                ha="center", va="center", fontsize=11, transform=ax.transAxes)
        return fig

    code, description, fname = row

    # Strip markdown fences if present
    lines = code.strip().splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    code = "\n".join(lines)

    # --- Execute factor code ---
    exec_ns: dict = {"pd": _pd, "np": _np}
    try:
        exec(f"def _factor(prices):\n" + "\n".join("    " + ln for ln in code.splitlines()), exec_ns)
        factor_df = exec_ns["_factor"](prices)
    except Exception as e:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"Factor execution error:\n{e}",
                ha="center", va="center", fontsize=9, transform=ax.transAxes)
        ax.set_title("IC by Period — factor execution failed", fontsize=11)
        return fig

    # --- Compute IC by period ---
    try:
        from qframe.factor_harness.ic import compute_ic_by_period
        period_df = compute_ic_by_period(
            factor_df, returns,
            oos_start=oos_start,
            period_years=period_years,
            horizon=horizon,
        )
    except Exception as e:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"compute_ic_by_period error:\n{e}",
                ha="center", va="center", fontsize=9, transform=ax.transAxes)
        return fig

    if period_df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"No OOS data after {oos_start}.",
                ha="center", va="center", fontsize=11, transform=ax.transAxes)
        return fig

    # --- Plot ---
    fig, ax = plt.subplots(figsize=figsize)

    labels = period_df["period_label"].tolist()
    x = np.arange(len(labels))
    mean_ic = period_df["mean_ic"].values
    std_ic = period_df["std_ic"].values
    n_days = period_df["n_days"].values

    colors = ["#4C72B0" if v >= 0 else "#C44E52" for v in mean_ic]
    ax.bar(x, mean_ic, color=colors, alpha=0.85, width=0.6, zorder=3)

    # 95% CI error bars: ± 1.96 × std / √n
    se = std_ic / np.sqrt(np.maximum(n_days, 1))
    ax.errorbar(x, mean_ic, yerr=1.96 * se, fmt="none", color="k",
                capsize=4, lw=1.2, zorder=4)

    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(0.015, color="#2ca02c", ls="--", lw=1.0, label="weak gate (0.015)")
    ax.axhline(-0.015, color="#2ca02c", ls="--", lw=1.0, alpha=0.4)

    # Annotate ICIR per period above each bar
    for i, (_, prow) in enumerate(period_df.iterrows()):
        icir_val = prow.get("icir", float("nan"))
        if np.isfinite(icir_val):
            y_pos = mean_ic[i] + 1.96 * se[i] + 0.001
            ax.text(i, y_pos, f"ICIR={icir_val:.2f}", ha="center", va="bottom",
                    fontsize=7, color="dimgray")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean IC", fontsize=10)
    ax.set_title(
        f"IC by Calendar Period — {fname or description[:40]}\n"
        f"Horizon: {horizon}d  |  Each bar ≈ {period_years:.0f} years  |  "
        "Error bars = 95% CI",
        fontsize=11,
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Chart 15 — Multiple testing significance chart
# ---------------------------------------------------------------------------

def plot_multiple_testing(
    kb_path: str | Path,
    alpha: float = 0.05,
    n_oos_days: int = 1762,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """
    Horizontal bar chart of t-statistics with BHY / HLZ / Bonferroni threshold lines.

    Shows which factors survive multiple testing correction.
    Uses:
    - Slow t-stat  (t = slow_icir_63 × √(N/63)) when slow_icir_63 is available
    - Fast t-stat  (t = sharpe/√252 × √N) otherwise

    Args:
        kb_path:     Path to knowledge_base/qframe.db.
        alpha:       FDR level for BHY (default 0.05).
        n_oos_days:  OOS trading days (default 1762 ≈ 7 years).
        figsize:     Figure size.

    Returns:
        matplotlib.Figure with one bar per positive-IC factor.
    """
    import math as _math
    from scipy import stats as _stats

    df = _load_results(kb_path)
    df = df[df["ic"].fillna(0) > 0].copy()

    if df.empty:
        fig, ax = plt.subplots(figsize=figsize or (10, 4))
        ax.text(0.5, 0.5, "No factors with positive IC found.",
                ha="center", va="center", fontsize=11, transform=ax.transAxes)
        return fig

    # Compute t-stats (prefer slow formula when available)
    def _t_stat(row) -> float:
        slow63 = row.get("slow_icir_63")
        if pd.notna(slow63) and float(slow63) != 0:
            n_windows = _math.floor(n_oos_days / 63)
            return float(slow63) * _math.sqrt(n_windows)
        sharpe = row.get("sharpe")
        if pd.notna(sharpe):
            return float(sharpe) / _math.sqrt(252) * _math.sqrt(n_oos_days)
        return 0.0

    df["t_stat"] = df.apply(_t_stat, axis=1)
    df = df.sort_values("t_stat", ascending=True)

    m = len(df)
    labels = [_short_name(r) for _, r in df.iterrows()]

    # Multiple testing thresholds
    c_m = sum(1.0 / i for i in range(1, m + 1))
    bhy_threshold_p = alpha / (m * c_m)
    bhy_t = float(_stats.t.ppf(1 - bhy_threshold_p, df=n_oos_days - 1))
    hlz_t = max(3.0, _math.sqrt(2 * _math.log(m))) if m >= 2 else 3.0
    bonf_t = float(_stats.t.ppf(1 - alpha / m, df=n_oos_days - 1))

    fig_h = max(4.0, m * 0.45 + 1.5)
    fig, ax = plt.subplots(figsize=figsize or (12, fig_h))

    t_vals = df["t_stat"].tolist()
    colors = []
    for t in t_vals:
        if t >= bhy_t:
            colors.append("#006400")   # BHY significant
        elif t >= hlz_t:
            colors.append("#2ca02c")   # HLZ significant
        elif t >= bonf_t:
            colors.append("#8fbc8f")   # Bonferroni significant
        elif t >= 1.96:
            colors.append("#9ecae1")   # nominally p < 0.05
        else:
            colors.append("#C44E52")   # not significant

    bars = ax.barh(range(m), t_vals, color=colors, height=0.7)
    ax.set_yticks(range(m))
    ax.set_yticklabels(labels, fontsize=8)

    ax.axvline(bhy_t,  color="#006400", lw=2.0, ls="-",  label=f"BHY  t≥{bhy_t:.2f}  (m={m})")
    ax.axvline(hlz_t,  color="#2ca02c", lw=1.5, ls="--", label=f"HLZ  t≥{hlz_t:.2f}")
    ax.axvline(bonf_t, color="#8fbc8f", lw=1.2, ls=":",  label=f"Bonferroni  t≥{bonf_t:.2f}")
    ax.axvline(1.96,   color="#9ecae1", lw=1.0, ls=":",  label="Nominal p<0.05  t=1.96")
    ax.axvline(0, color="k", lw=0.8)

    for i, (bar, t) in enumerate(zip(bars, t_vals)):
        ax.text(max(t + 0.05, 0.05), i, f"t={t:.2f}", va="center", fontsize=7)

    ax.set_xlabel("t-statistic", fontsize=10)
    ax.set_title(
        f"Multiple Testing Significance — {m} positive-IC factors\n"
        f"BHY FDR={alpha:.0%}  |  n_oos={n_oos_days}d  |  "
        f"c(m)={c_m:.2f}  |  BHY threshold t≥{bhy_t:.2f}",
        fontsize=11,
    )
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


# ===========================================================================
# Chart 16 — Regime timeline
# ===========================================================================

def plot_regime_timeline(
    proba_df: pd.DataFrame,
    market_returns: pd.Series,
    oos_start: Optional[str] = None,
    state_labels: Optional[dict] = None,
    title: str = "Market Regime Timeline",
) -> plt.Figure:
    """
    Stacked area chart of regime posterior probabilities with cumulative
    return overlay.

    Args:
        proba_df:        (dates × n_states) posterior DataFrame from
                         RegimeICAnalyzer.proba_df.  Columns 0..n-1 ordered
                         ascending by mean return (0 = most bearish).
        market_returns:  Daily market return series (aligned to proba_df).
        oos_start:       If provided, draw a vertical line at this date.
        state_labels:    Optional dict {state_int: "label_str"} for legend.
                         Defaults to {0: "Bear", 1: "Neutral", 2: "Bull"} for
                         3-state models.
        title:           Figure title.

    Returns:
        matplotlib.Figure (never calls plt.show()).
    """
    valid = proba_df.dropna()
    n_states = proba_df.shape[1]

    if state_labels is None:
        # Default labels: bearish end = 0, bullish end = n_states-1
        default_names = ["Bear", "Bear-Vol", "Neutral", "Bull-Vol", "Bull"]
        state_labels = {s: default_names[s] if s < len(default_names) else f"State {s}"
                        for s in range(n_states)}

    # Colour palette: red → amber → green
    palette_3 = ["#d62728", "#ff7f0e", "#2ca02c"]
    palette_5 = ["#d62728", "#e87e04", "#bcbd22", "#17becf", "#2ca02c"]
    palette = palette_5 if n_states == 5 else palette_3

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    # Stacked area
    proba_vals = valid.values
    x = valid.index
    bottoms = np.zeros(len(x))
    for s in range(n_states):
        col = palette[s % len(palette)]
        ax1.fill_between(x, bottoms, bottoms + proba_vals[:, s],
                         color=col, alpha=0.75, label=state_labels[s])
        bottoms += proba_vals[:, s]

    if oos_start:
        ax1.axvline(pd.Timestamp(oos_start), color="k", lw=1.5, ls="--",
                    label=f"OOS start ({oos_start})")

    ax1.set_ylabel("Regime probability", fontsize=10)
    ax1.set_ylim(0, 1)
    ax1.legend(loc="upper left", fontsize=8, ncol=n_states)
    ax1.set_title(title, fontsize=12)

    # Cumulative return overlay
    common = market_returns.reindex(x).dropna()
    cum_ret = (1 + common).cumprod() - 1
    ax2.plot(cum_ret.index, cum_ret.values * 100, color="#333333", lw=1.2)
    ax2.fill_between(cum_ret.index, 0, cum_ret.values * 100,
                     where=cum_ret.values >= 0, color="#2ca02c", alpha=0.3)
    ax2.fill_between(cum_ret.index, 0, cum_ret.values * 100,
                     where=cum_ret.values < 0, color="#d62728", alpha=0.3)
    ax2.axhline(0, color="k", lw=0.8)
    if oos_start:
        ax2.axvline(pd.Timestamp(oos_start), color="k", lw=1.5, ls="--")
    ax2.set_ylabel("Cum. return (%)", fontsize=10)
    ax2.set_xlabel("Date", fontsize=10)

    fig.tight_layout()
    return fig


# ===========================================================================
# Chart 17 — Regime-conditional IC bar chart
# ===========================================================================

def plot_regime_ic(
    decomp_df: pd.DataFrame,
    unconditional_ic: float,
    factor_name: str = "",
    horizon: int = 1,
) -> plt.Figure:
    """
    Bar chart of IC per regime state with error bars and unconditional IC line.

    Args:
        decomp_df:        by_state DataFrame from RegimeDecomposition.by_state.
                          Must have columns: ic, icir, t_stat, n_days, pct_time.
        unconditional_ic: Overall unconditional IC (drawn as horizontal line).
        factor_name:      Factor identifier for the title.
        horizon:          Forward return horizon (for title).

    Returns:
        matplotlib.Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    states = decomp_df.index.tolist()
    ics = decomp_df["ic"].values
    t_stats = decomp_df["t_stat"].values
    n_days = decomp_df["n_days"].values.astype(int)
    pct_time = decomp_df["pct_time"].values

    # Standard error of IC estimate: IC / t_stat  (or IC / sqrt(n) as fallback)
    se = np.where(
        np.isfinite(t_stats) & (t_stats != 0),
        np.abs(ics / np.where(t_stats != 0, t_stats, np.nan)),
        np.abs(ics) / np.sqrt(np.maximum(n_days, 1)),
    )

    # Colours: green if IC > unconditional, red if below, grey if NaN
    colours = []
    for ic in ics:
        if not np.isfinite(ic):
            colours.append("#aaaaaa")
        elif ic > unconditional_ic:
            colours.append("#2ca02c")
        else:
            colours.append("#d62728")

    x = np.arange(len(states))
    bars = ax.bar(x, ics, yerr=se, color=colours, alpha=0.8, capsize=5, width=0.6,
                  error_kw={"elinewidth": 1.5})

    ax.axhline(unconditional_ic, color="#1f77b4", lw=2.0, ls="--",
               label=f"Unconditional IC = {unconditional_ic:.4f}")
    ax.axhline(0, color="k", lw=0.8)

    # Annotations: t-stat and n_days below each bar
    for i, (ic, t, n, pct) in enumerate(zip(ics, t_stats, n_days, pct_time)):
        if np.isfinite(ic):
            ax.text(i, ic + (0.001 if ic >= 0 else -0.002),
                    f"t={t:.2f}", ha="center", va="bottom" if ic >= 0 else "top",
                    fontsize=8, fontweight="bold")
        ax.text(i, ax.get_ylim()[0] * 0.95,
                f"n={n}\n({pct:.0%})", ha="center", va="bottom", fontsize=7, color="#555555")

    ax.set_xticks(x)
    ax.set_xticklabels([f"State {s}" for s in states], fontsize=10)
    ax.set_ylabel("Information Coefficient (IC)", fontsize=10)
    ax.set_title(
        f"Regime-Conditional IC — {factor_name}  (horizon={horizon}d)\n"
        f"Green = above unconditional, Red = below",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


# ===========================================================================
# Chart 18 — Regime transition velocity
# ===========================================================================

def plot_velocity(
    velocity_raw: pd.Series,
    velocity_smooth: pd.Series,
    hard_labels: Optional[pd.Series] = None,
    oos_start: Optional[str] = None,
    title: str = "Regime Transition Velocity (KL Divergence)",
) -> plt.Figure:
    """
    Velocity time series (raw + EWM-smoothed) with optional regime colouring.

    Args:
        velocity_raw:    Raw KL-divergence velocity series.
        velocity_smooth: EWM-smoothed velocity series.
        hard_labels:     Optional regime hard-label series for background shading.
        oos_start:       Optional OOS start date line.
        title:           Figure title.

    Returns:
        matplotlib.Figure.
    """
    fig, ax = plt.subplots(figsize=(14, 4))

    # Background regime shading
    if hard_labels is not None:
        palette = ["#ffcccc", "#fff0cc", "#ccffcc", "#ccf0ff", "#ccccff"]
        for s in range(int(hard_labels.dropna().max()) + 1):
            mask = hard_labels == s
            if not mask.any():
                continue
            col = palette[s % len(palette)]
            # Shade contiguous blocks
            in_block = False
            start_idx = None
            for dt, val in mask.items():
                if val and not in_block:
                    in_block = True
                    start_idx = dt
                elif not val and in_block:
                    ax.axvspan(start_idx, dt, alpha=0.25, color=col, lw=0)
                    in_block = False
            if in_block and start_idx is not None:
                ax.axvspan(start_idx, mask.index[-1], alpha=0.25, color=col, lw=0)

    valid = velocity_raw.dropna()
    ax.plot(valid.index, valid.values, color="#aaaaaa", lw=0.8, alpha=0.7, label="Raw KL velocity")

    valid_s = velocity_smooth.dropna()
    ax.plot(valid_s.index, valid_s.values, color="#d62728", lw=1.8, label="Smoothed (EWM)")

    # 90th percentile threshold line
    p90 = valid_s.quantile(0.90)
    ax.axhline(p90, color="#ff7f0e", lw=1.2, ls="--", label=f"90th pct = {p90:.3f}")

    if oos_start:
        ax.axvline(pd.Timestamp(oos_start), color="k", lw=1.5, ls="--",
                   label=f"OOS start ({oos_start})")

    ax.set_ylabel("KL Divergence", fontsize=10)
    ax.set_xlabel("Date", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


# ===========================================================================
# Chart 19 — Rolling Hurst exponent with regime overlay
# ===========================================================================

def plot_hurst_rolling(
    hurst_series: pd.Series,
    hard_labels: Optional[pd.Series] = None,
    oos_start: Optional[str] = None,
    title: str = "Rolling Hurst Exponent (DFA)",
) -> plt.Figure:
    """
    Rolling Hurst exponent with 0.5 reference line and optional regime overlay.

    H > 0.5 = trending/momentum regime.
    H < 0.5 = mean-reverting regime.
    H ≈ 0.5 = random walk (no exploitable autocorrelation).

    Args:
        hurst_series: Rolling Hurst series from HurstEstimator.fit_rolling().
        hard_labels:  Optional regime hard-label series for background shading.
        oos_start:    Optional OOS start date line.
        title:        Figure title.

    Returns:
        matplotlib.Figure.
    """
    fig, ax = plt.subplots(figsize=(14, 4))

    # Background regime shading (same palette as plot_velocity)
    if hard_labels is not None:
        palette = ["#ffcccc", "#fff0cc", "#ccffcc", "#ccf0ff", "#ccccff"]
        for s in range(int(hard_labels.dropna().max()) + 1):
            mask = hard_labels == s
            if not mask.any():
                continue
            col = palette[s % len(palette)]
            in_block = False
            start_idx = None
            for dt, val in mask.items():
                if val and not in_block:
                    in_block = True
                    start_idx = dt
                elif not val and in_block:
                    ax.axvspan(start_idx, dt, alpha=0.25, color=col, lw=0)
                    in_block = False
            if in_block and start_idx is not None:
                ax.axvspan(start_idx, mask.index[-1], alpha=0.25, color=col, lw=0)

    valid = hurst_series.dropna()
    ax.plot(valid.index, valid.values, color="#1f77b4", lw=1.5, label="Hurst H (DFA)")
    ax.axhline(0.5, color="k", lw=1.5, ls="--", label="H = 0.5 (random walk)")
    ax.fill_between(valid.index, 0.5, valid.values,
                    where=valid.values > 0.5, color="#2ca02c", alpha=0.2,
                    label="H > 0.5 (momentum)")
    ax.fill_between(valid.index, 0.5, valid.values,
                    where=valid.values < 0.5, color="#d62728", alpha=0.2,
                    label="H < 0.5 (mean-reversion)")

    if oos_start:
        ax.axvline(pd.Timestamp(oos_start), color="k", lw=1.5, ls="--",
                   label=f"OOS start ({oos_start})")

    ax.set_ylim(0.2, 0.8)
    ax.set_ylabel("Hurst Exponent H", fontsize=10)
    ax.set_xlabel("Date", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Chart 20 — Combined Equity Curve + Drawdown (Phase 2.5)
# ---------------------------------------------------------------------------

def plot_combined_equity(
    equity_curves: "dict[str, pd.Series]",
    benchmark: "Optional[pd.Series]" = None,
    initial_investment: float = 10_000,
    title: str = "Combined Strategy — Equity Curve & Drawdown",
) -> plt.Figure:
    """
    2-panel chart: top = multiple equity curves + optional benchmark,
    bottom = drawdown from peak for the *first* strategy in ``equity_curves``.

    Args:
        equity_curves:      Ordered dict ``{label: equity_Series}``.
                            The first entry is treated as the *primary* strategy
                            for the drawdown panel.
        benchmark:          Optional equal-weight benchmark equity Series.
        initial_investment: Starting NAV (used for annotation and the dotted baseline).
        title:              Figure suptitle.

    Returns:
        matplotlib.Figure.
    """
    import math as _math

    palette = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
    names = list(equity_curves.keys())
    curves = list(equity_curves.values())

    fig, (ax_eq, ax_dd) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # ── Top panel: equity curves ─────────────────────────────────────────────
    if benchmark is not None:
        bm_r = benchmark.reindex(curves[0].index)
        ax_eq.plot(bm_r.index, bm_r.values, color="grey", lw=1.0, ls="--",
                   label=f"EW S&P 500 → ${bm_r.iloc[-1]:,.0f}")

    for i, (name, eq) in enumerate(zip(names, curves)):
        col = palette[i % len(palette)]
        ax_eq.plot(eq.index, eq.values, color=col, lw=1.8,
                   label=f"{name} → ${eq.iloc[-1]:,.0f}")

    ax_eq.axhline(initial_investment, color="k", lw=0.6, ls=":", alpha=0.6)
    ax_eq.set_ylabel("Portfolio value ($)", fontsize=10)
    ax_eq.legend(fontsize=8, loc="upper left")
    ax_eq.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )
    ax_eq.grid(axis="y", alpha=0.3)

    # ── Bottom panel: drawdown of primary strategy ────────────────────────────
    primary = curves[0]
    peak = primary.cummax()
    dd = (primary - peak) / peak
    ax_dd.fill_between(dd.index, 0, dd.values * 100,
                       where=dd.values < 0,
                       color="#d62728", alpha=0.6, label="Drawdown")
    ax_dd.axhline(0, color="k", lw=0.6)
    max_dd = dd.min()
    ax_dd.annotate(
        f"Max DD: {max_dd:.1%}",
        xy=(dd.idxmin(), max_dd * 100),
        xytext=(0, -14), textcoords="offset points",
        fontsize=8, color="#d62728", ha="center",
    )
    ax_dd.set_ylabel("Drawdown (%)", fontsize=9)
    ax_dd.set_xlabel("Date", fontsize=10)
    ax_dd.legend(fontsize=8, loc="lower left")
    ax_dd.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ---------------------------------------------------------------------------
# Chart 21 — Rolling 12-Month Sharpe Ratio (Phase 2.5)
# ---------------------------------------------------------------------------

def plot_rolling_sharpe(
    ic_series: pd.Series,
    window: int = 252,
    title: str = "Rolling 12-Month Sharpe Ratio",
    threshold: float = 0.5,
) -> plt.Figure:
    """
    Rolling annualised Sharpe ratio of a daily IC series.

    Highlights periods below ``threshold`` in red and above in green to make
    it easy to see whether the strategy's edge is persistent or episodic.

    Args:
        ic_series:  Daily IC (or net IC) series from ``WalkForwardResult``.
        window:     Rolling window in trading days (default 252 = 1 year).
        title:      Figure title.
        threshold:  Gate 3 Sharpe threshold — drawn as a horizontal dashed line.

    Returns:
        matplotlib.Figure.
    """
    import numpy as _np

    rolling_mean = ic_series.rolling(window, min_periods=window // 2).mean()
    rolling_std  = ic_series.rolling(window, min_periods=window // 2).std()
    rolling_sharpe = (rolling_mean / rolling_std.replace(0, _np.nan)) * _np.sqrt(252)
    rolling_sharpe = rolling_sharpe.dropna()

    fig, ax = plt.subplots(figsize=(14, 4))

    ax.plot(rolling_sharpe.index, rolling_sharpe.values,
            color="#1f77b4", lw=1.5, label="Rolling Sharpe")
    ax.axhline(threshold, color="#d62728", lw=1.2, ls="--",
               label=f"Gate 3 threshold ({threshold})")
    ax.axhline(0, color="k", lw=0.6, ls=":")

    ax.fill_between(
        rolling_sharpe.index, threshold, rolling_sharpe.values,
        where=rolling_sharpe.values >= threshold,
        interpolate=True, color="#2ca02c", alpha=0.20,
        label="Above threshold",
    )
    ax.fill_between(
        rolling_sharpe.index, threshold, rolling_sharpe.values,
        where=rolling_sharpe.values < threshold,
        interpolate=True, color="#d62728", alpha=0.18,
        label="Below threshold",
    )

    # Annotate mean Sharpe
    mean_sh = float(rolling_sharpe.mean())
    ax.annotate(
        f"Mean: {mean_sh:.2f}",
        xy=(rolling_sharpe.index[len(rolling_sharpe) // 2], mean_sh),
        xytext=(0, 8), textcoords="offset points",
        fontsize=8, color="#1f77b4",
    )

    ax.set_ylabel(f"Sharpe (rolling {window}d)", fontsize=10)
    ax.set_xlabel("Date", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Chart 22 — Dynamic Blend Weights Over Time (Phase 2.5b)
# ---------------------------------------------------------------------------

def plot_blend_weights(
    blend_weights: pd.DataFrame,
    hard_labels: Optional[pd.Series] = None,
    title: str = "Dynamic Factor Blend Weights (Regime-Posterior Weighted)",
) -> plt.Figure:
    """
    Stacked area chart of time-varying factor blend weights from
    ``RegimeICAnalyzer.regime_blend_weights()``.

    Each factor is a coloured band; the bands sum to 1 at every date.
    Optional regime hard labels add a background shading strip at the bottom.

    Args:
        blend_weights: DataFrame (dates × factors) of weights summing to 1 per row.
                       Produced by ``RegimeICAnalyzer.regime_blend_weights()``.
        hard_labels:   Optional regime hard-label Series for background strip.
        title:         Figure title.

    Returns:
        matplotlib.Figure.
    """
    factor_names = list(blend_weights.columns)
    n_factors = len(factor_names)

    palette = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b",
               "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    colours = [palette[i % len(palette)] for i in range(n_factors)]

    # Layout: 2 rows if regime labels given (tiny strip at bottom), else 1 row
    if hard_labels is not None:
        fig, (ax_w, ax_r) = plt.subplots(
            2, 1, figsize=(14, 6), sharex=True,
            gridspec_kw={"height_ratios": [5, 1]},
        )
    else:
        fig, ax_w = plt.subplots(figsize=(14, 5))
        ax_r = None

    # ── Stacked area chart of blend weights ─────────────────────────────────
    import numpy as _np2
    dates = blend_weights.index
    baseline = _np2.zeros(len(dates))
    for i, (name, col) in enumerate(zip(factor_names, colours)):
        w = blend_weights[name].values
        ax_w.fill_between(dates, baseline, baseline + w,
                          alpha=0.75, color=col, label=name)
        baseline = baseline + w

    ax_w.axhline(1.0 / n_factors, color="k", lw=0.8, ls="--", alpha=0.5,
                 label=f"Equal weight (1/{n_factors} = {1/n_factors:.2f})")
    ax_w.set_ylim(0, 1)
    ax_w.set_ylabel("Blend weight", fontsize=10)
    ax_w.set_title(title, fontsize=11)
    ax_w.legend(fontsize=8, loc="upper right", framealpha=0.9)

    # Annotate mean weight for each factor at right edge
    cumsum_end = 0.0
    for name, col in zip(factor_names, colours):
        mean_w = float(blend_weights[name].mean())
        last_w = float(blend_weights[name].iloc[-1])
        mid_y = cumsum_end + last_w / 2
        ax_w.annotate(
            f"{name}  (μ={mean_w:.2f})",
            xy=(dates[-1], mid_y),
            xytext=(6, 0), textcoords="offset points",
            fontsize=7, color=col, va="center",
            clip_on=False,
        )
        cumsum_end += last_w

    # ── Regime strip ─────────────────────────────────────────────────────────
    if ax_r is not None and hard_labels is not None:
        strip_palette = ["#ffcccc", "#fff0cc", "#ccffcc", "#ccf0ff", "#ccccff"]
        aligned = hard_labels.reindex(dates)
        for s in range(5):
            mask = aligned == s
            if not mask.any():
                continue
            col = strip_palette[s % len(strip_palette)]
            in_block = False
            start_idx = None
            for dt, val in mask.items():
                if val and not in_block:
                    in_block = True
                    start_idx = dt
                elif not val and in_block:
                    ax_r.axvspan(start_idx, dt, alpha=0.8, color=col, lw=0)
                    in_block = False
            if in_block and start_idx is not None:
                ax_r.axvspan(start_idx, mask.index[-1], alpha=0.8, color=col, lw=0)
        ax_r.set_yticks([])
        ax_r.set_ylabel("Regime", fontsize=8, rotation=0, ha="right", va="center",
                        labelpad=30)
        ax_r.set_xlabel("Date", fontsize=10)
    else:
        ax_w.set_xlabel("Date", fontsize=10)

    fig.tight_layout()
    return fig
