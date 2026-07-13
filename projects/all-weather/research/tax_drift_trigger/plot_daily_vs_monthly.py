"""
research/tax_drift_trigger/plot_daily_vs_monthly.py
====================================================
Visualise the daily-vs-monthly engine comparison (L.53).

Reads from the latest results/daily_vs_monthly/ bundle.

Plots generated
---------------
a. calmar_heatmap.png    — Calmar by policy × engine (monthly | daily), faceted by OOS window
b. mdd_comparison.png    — MDD per policy, monthly vs daily engine, 2018 OOS window
c. rebalance_timing.png  — Rebalance date scatter by policy × engine (full period)
d. rank_delta.png        — Per-window rank change: monthly → daily (which policies moved)

Run
---
    conda run -n allweather python research/tax_drift_trigger/plot_daily_vs_monthly.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Style constants (dark theme, matching plot_equity_comparison.py)
# ---------------------------------------------------------------------------

DARK_BG  = "#0d1117"
PANEL_BG = "#161b22"
TEXT_COL = "#c9d1d9"
GRID_COL = "#30363d"
ACCENT   = "#58a6ff"
GREEN    = "#3fb950"
RED      = "#f85149"
ORANGE   = "#d29922"
PURPLE   = "#bc8cff"
CYAN     = "#39d2c0"

WINDOW_COLORS = {2018: ACCENT, 2020: GREEN, 2022: ORANGE}

POLICY_ORDER = [
    "drift_absolute(0.04)",
    "drift_absolute(0.05)",
    "drift_absolute(0.055)",
    "drift_absolute(0.06)",
    "drift_absolute(0.065)",
    "drift_absolute(0.07)",
    "drift_relative(0.4)",
    "monthly_unconditional",
]

SHORT = {
    "drift_absolute(0.04)":    "abs 4pp",
    "drift_absolute(0.05)":    "abs 5pp",
    "drift_absolute(0.055)":   "abs 5.5pp",
    "drift_absolute(0.06)":    "abs 6pp",
    "drift_absolute(0.065)":   "abs 6.5pp ★",
    "drift_absolute(0.07)":    "abs 7pp",
    "drift_relative(0.4)":     "rel 40%",
    "monthly_unconditional":   "monthly",
}


def _style_ax(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COL, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID_COL)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.label.set_color(TEXT_COL)
    ax.xaxis.label.set_color(TEXT_COL)
    ax.title.set_color("white")
    ax.grid(axis="y", color=GRID_COL, alpha=0.45, linewidth=0.6)


def _save(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved → {path.relative_to(PROJECT_ROOT)}")


def _latest_bundle() -> Path | None:
    root = PROJECT_ROOT / "results" / "daily_vs_monthly"
    if not root.exists():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir() and (p / "calmar_comparison.csv").exists()]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


# ---------------------------------------------------------------------------
# Plot a: Calmar heatmap — daily vs monthly, each OOS window as a panel
# ---------------------------------------------------------------------------

def plot_calmar_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    windows = sorted(df["window"].unique())
    fig, axes = plt.subplots(1, len(windows), figsize=(15, 5), facecolor=DARK_BG,
                             sharey=False)
    fig.suptitle("Calmar ratio: monthly engine vs daily engine (FIFO · US tax)",
                 color="white", fontweight="bold", fontsize=12, y=1.02)

    pol_labels = [p for p in POLICY_ORDER if p in df["policy_label"].values]

    for ax, win in zip(axes, windows):
        _style_ax(ax)
        win_df = df[df["window"] == win]

        monthly_vals = []
        daily_vals = []
        for pl in pol_labels:
            m_row = win_df[(win_df["policy_label"] == pl) & (win_df["engine"] == "monthly")]
            d_row = win_df[(win_df["policy_label"] == pl) & (win_df["engine"] == "daily")]
            monthly_vals.append(float(m_row["calmar"].iloc[0]) if not m_row.empty else 0)
            daily_vals.append(float(d_row["calmar"].iloc[0]) if not d_row.empty else 0)

        x = np.arange(len(pol_labels))
        w = 0.34
        bars_m = ax.bar(x - w / 2, monthly_vals, w, label="monthly engine",
                        color=PURPLE, alpha=0.82)
        bars_d = ax.bar(x + w / 2, daily_vals,   w, label="daily engine",
                        color=WINDOW_COLORS[win], alpha=0.82)

        # Annotate daily bars only (to avoid clutter)
        for bar, v in zip(bars_d, daily_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.003,
                    f"{v:.3f}", ha="center", va="bottom", color=TEXT_COL, fontsize=6)

        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[p] for p in pol_labels], rotation=40, ha="right",
                           fontsize=7.5)
        ax.set_ylabel("Calmar Ratio")
        ax.set_ylim(0, max(max(monthly_vals), max(daily_vals)) * 1.18)
        ax.set_title(f"OOS {win}", fontweight="bold", color=WINDOW_COLORS[win])

        # Highlight 6.5pp (winner) bar
        idx_winner = pol_labels.index("drift_absolute(0.065)")
        for bar in [bars_m[idx_winner], bars_d[idx_winner]]:
            bar.set_edgecolor("white")
            bar.set_linewidth(1.5)

        if win == windows[0]:
            ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COL,
                      labelcolor=TEXT_COL, loc="upper left")

    fig.tight_layout()
    _save(fig, out_dir / "calmar_heatmap.png")


# ---------------------------------------------------------------------------
# Plot b: MDD comparison — monthly vs daily, all three OOS windows
# ---------------------------------------------------------------------------

def plot_mdd_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    windows = sorted(df["window"].unique())
    fig, axes = plt.subplots(1, len(windows), figsize=(15, 5), facecolor=DARK_BG,
                             sharey=True)
    fig.suptitle("Maximum Drawdown: monthly engine vs daily engine (FIFO · US tax)",
                 color="white", fontweight="bold", fontsize=12, y=1.02)

    pol_labels = [p for p in POLICY_ORDER if p in df["policy_label"].values]

    for ax, win in zip(axes, windows):
        _style_ax(ax)
        ax.grid(axis="x", color=GRID_COL, alpha=0.45, linewidth=0.6)
        ax.grid(axis="y", visible=False)
        win_df = df[df["window"] == win]

        monthly_mdds = []
        daily_mdds = []
        for pl in pol_labels:
            m_row = win_df[(win_df["policy_label"] == pl) & (win_df["engine"] == "monthly")]
            d_row = win_df[(win_df["policy_label"] == pl) & (win_df["engine"] == "daily")]
            monthly_mdds.append(float(m_row["mdd"].iloc[0]) if not m_row.empty else 0)
            daily_mdds.append(float(d_row["mdd"].iloc[0]) if not d_row.empty else 0)

        y = np.arange(len(pol_labels))
        h = 0.34
        ax.barh(y - h / 2, monthly_mdds, h, label="monthly engine",
                color=PURPLE, alpha=0.82)
        ax.barh(y + h / 2, daily_mdds,   h, label="daily engine",
                color=WINDOW_COLORS[win], alpha=0.82)

        ax.set_yticks(y)
        ax.set_yticklabels([SHORT[p] for p in pol_labels], fontsize=8)
        ax.set_xlabel("Max Drawdown (%)")
        ax.set_title(f"OOS {win}", fontweight="bold", color=WINDOW_COLORS[win])
        ax.invert_xaxis()

        # Annotate 7pp blow-out
        idx_7 = pol_labels.index("drift_absolute(0.07)")
        daily_7 = daily_mdds[idx_7]
        ax.annotate(f"blow-out\n{daily_7:.1f}%",
                    xy=(daily_7, idx_7 + h / 2),
                    xytext=(daily_7 - 0.5, idx_7 + h / 2 + 0.8),
                    arrowprops=dict(arrowstyle="->", color=RED, lw=1.2),
                    fontsize=7, color=RED, ha="center")

        if win == windows[0]:
            ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COL,
                      labelcolor=TEXT_COL)

    fig.tight_layout()
    _save(fig, out_dir / "mdd_comparison.png")


# ---------------------------------------------------------------------------
# Plot c: Rebalance timing scatter — monthly vs daily engine, full period
# ---------------------------------------------------------------------------

def plot_rebalance_timing(timing_df: pd.DataFrame, out_dir: Path) -> None:
    drift_policies = [p for p in POLICY_ORDER if p != "monthly_unconditional"]
    # Map short names to y-axis ticks (exclude monthly baseline for clarity)
    pol_map = {
        "drift_absolute_4pp":   "drift_absolute(0.04)",
        "drift_absolute_5pp":   "drift_absolute(0.05)",
        "drift_absolute_5.5pp": "drift_absolute(0.055)",
        "drift_absolute_6pp":   "drift_absolute(0.06)",
        "drift_absolute_6.5pp": "drift_absolute(0.065)",
        "drift_absolute_7pp":   "drift_absolute(0.07)",
        "drift_relative_40pct": "drift_relative(0.4)",
    }
    timing = timing_df[timing_df["policy"].isin(pol_map)].copy()
    timing["policy_label"] = timing["policy"].map(pol_map)
    timing["date"] = pd.to_datetime(timing["rebalance_date"])

    fig, ax = plt.subplots(figsize=(14, 5), facecolor=DARK_BG)
    _style_ax(ax)
    ax.grid(axis="x", color=GRID_COL, alpha=0.30, linewidth=0.6)
    ax.grid(axis="y", visible=False)

    pol_labels = [p for p in drift_policies if p in timing["policy_label"].values]
    pol_idx = {p: i for i, p in enumerate(pol_labels)}

    for engine, (marker, alpha) in (("monthly", ("o", 0.55)), ("daily", ("^", 0.80))):
        sub = timing[timing["engine"] == engine]
        for _, row in sub.iterrows():
            pl = row["policy_label"]
            if pl not in pol_idx:
                continue
            y = pol_idx[pl] + (0.18 if engine == "daily" else -0.18)
            color = GREEN if engine == "daily" else PURPLE
            ax.scatter(row["date"], y, marker=marker, s=28, color=color,
                       alpha=alpha, zorder=3, linewidths=0)

    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PURPLE,
               markersize=7, label="monthly engine"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=GREEN,
               markersize=7, label="daily engine"),
    ]
    ax.legend(handles=legend_els, fontsize=8, facecolor=PANEL_BG,
              edgecolor=GRID_COL, labelcolor=TEXT_COL)

    ax.set_yticks(range(len(pol_labels)))
    ax.set_yticklabels([SHORT[p] for p in pol_labels], fontsize=9)
    ax.set_xlabel("Date")
    ax.set_title("Rebalance timing: monthly engine vs daily engine (full period)",
                 fontweight="bold")
    ax.axvline(pd.Timestamp("2018-01-01"), color=ACCENT, alpha=0.35,
               linewidth=0.9, linestyle="--")
    ax.axvline(pd.Timestamp("2020-01-01"), color=GREEN, alpha=0.35,
               linewidth=0.9, linestyle="--")
    ax.axvline(pd.Timestamp("2022-01-01"), color=ORANGE, alpha=0.35,
               linewidth=0.9, linestyle="--")
    for date, label, col in [
        ("2018-01-01", "OOS 2018", ACCENT),
        ("2020-01-01", "OOS 2020", GREEN),
        ("2022-01-01", "OOS 2022", ORANGE),
    ]:
        ax.text(pd.Timestamp(date), len(pol_labels) - 0.3, label,
                color=col, fontsize=7, ha="left")

    fig.tight_layout()
    _save(fig, out_dir / "rebalance_timing.png")


# ---------------------------------------------------------------------------
# Plot d: Rank delta — policy rank in monthly engine vs daily engine
# ---------------------------------------------------------------------------

def plot_rank_delta(df: pd.DataFrame, out_dir: Path) -> None:
    windows = sorted(df["window"].unique())
    fig, axes = plt.subplots(1, len(windows), figsize=(15, 5), facecolor=DARK_BG,
                             sharey=True)
    fig.suptitle("Policy rank: monthly engine → daily engine (1 = best Calmar, drift policies only)",
                 color="white", fontweight="bold", fontsize=12, y=1.02)

    drift_labels = [p for p in POLICY_ORDER
                    if p != "monthly_unconditional" and p in df["policy_label"].values]

    for ax, win in zip(axes, windows):
        _style_ax(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID_COL, alpha=0.40, linewidth=0.6)
        win_df = df[df["window"] == win]

        def _rank(engine: str) -> dict[str, int]:
            sub = win_df[
                (win_df["engine"] == engine) &
                (win_df["policy_label"].isin(drift_labels))
            ].copy()
            sub = sub.dropna(subset=["calmar"]).sort_values("calmar", ascending=False)
            return {row["policy_label"]: i + 1 for i, (_, row) in enumerate(sub.iterrows())}

        m_ranks = _rank("monthly")
        d_ranks = _rank("daily")

        y = np.arange(len(drift_labels))
        for i, pl in enumerate(drift_labels):
            mr = m_ranks.get(pl, len(drift_labels))
            dr = d_ranks.get(pl, len(drift_labels))
            # horizontal bars: monthly rank (left of centre) and daily rank (right of centre)
            ax.plot([mr, dr], [i, i], color=GRID_COL, linewidth=1.2, zorder=1)
            ax.scatter(mr, i, s=60, color=PURPLE, zorder=3, linewidths=0)
            ax.scatter(dr, i, s=60, color=WINDOW_COLORS[win], zorder=3, linewidths=0)
            delta = dr - mr
            arrow_col = GREEN if delta < 0 else (RED if delta > 0 else TEXT_COL)
            delta_str = (f"↑{abs(delta)}" if delta < 0 else
                         (f"↓{delta}" if delta > 0 else "—"))
            ax.text(max(mr, dr) + 0.12, i, delta_str,
                    color=arrow_col, fontsize=7.5, va="center")

        ax.set_yticks(y)
        ax.set_yticklabels([SHORT[p] for p in drift_labels], fontsize=8)
        ax.set_xlabel("Rank (1 = best Calmar)")
        ax.set_xlim(0.5, len(drift_labels) + 1.2)
        ax.invert_xaxis()
        ax.set_title(f"OOS {win}", fontweight="bold", color=WINDOW_COLORS[win])
        ax.set_xticks(range(1, len(drift_labels) + 1))

        from matplotlib.lines import Line2D
        if win == windows[0]:
            ax.legend(handles=[
                Line2D([0], [0], marker="o", color="w", markerfacecolor=PURPLE,
                       markersize=7, label="monthly engine"),
                Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=WINDOW_COLORS[win],
                       markersize=7, label="daily engine"),
            ], fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL)

    fig.tight_layout()
    _save(fig, out_dir / "rank_delta.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    bundle = _latest_bundle()
    if bundle is None:
        print("[ERROR] No daily_vs_monthly result bundle found.")
        print("  Run: python research/tax_drift_trigger/daily_vs_monthly_comparison.py")
        return

    print(f"Reading bundle: {bundle.name}")
    df = pd.read_csv(bundle / "calmar_comparison.csv")
    timing_df = pd.read_csv(bundle / "rebalance_timing.csv")

    out_dir = bundle / "plots"
    out_dir.mkdir(exist_ok=True)
    print(f"Writing plots → {out_dir.relative_to(PROJECT_ROOT)}/")

    plot_calmar_heatmap(df, out_dir)
    plot_mdd_comparison(df, out_dir)
    plot_rebalance_timing(timing_df, out_dir)
    plot_rank_delta(df, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
