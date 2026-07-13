import marimo

__generated_with = "0.23.5"
app = marimo.App(width="full")

with app.setup:
    import json
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np
    import pandas as pd

    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

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

    POLICY_COLORS = {
        "Monthly (baseline)": RED,
        "Drift abs 2pp":      ORANGE,
        "Drift abs 5pp":      GREEN,
        "Drift rel 25%":      ACCENT,
        "Drift rel 40%":      PURPLE,
    }
    POLICY_LS = {k: ("--" if k == "Monthly (baseline)" else "-") for k in POLICY_COLORS}
    POLICY_LW = {k: (2.2 if k == "Drift abs 5pp" else 1.3) for k in POLICY_COLORS}
    MARKER_ALPHA = {
        "Monthly (baseline)": 0.05,
        "Drift abs 2pp":      0.12,
        "Drift abs 5pp":      0.18,
        "Drift rel 25%":      0.10,
        "Drift rel 40%":      0.10,
    }

    RESULTS = ROOT / "results"
    EQUITY_CMP_ROOT        = RESULTS / "equity_comparison"
    TAX_SWEEP_ROOT         = RESULTS / "tax_threshold_sweep"
    DAILY_VS_MONTHLY_ROOT  = RESULTS / "daily_vs_monthly"

    def _latest(root: Path, required: str = "run_config.json") -> Path | None:
        if not root.exists():
            return None
        candidates = [p for p in root.iterdir()
                      if p.is_dir() and (p / required).exists()]
        return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None

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
        ax.grid(axis="y", color=GRID_COL, alpha=0.40, linewidth=0.6)
        ax.grid(axis="x", color=GRID_COL, alpha=0.18, linewidth=0.4)


# ── Bundle selectors ──────────────────────────────────────────────────────────

@app.cell
def choose_bundles():
    def _all_dirs(root: Path, required: str) -> list[str]:
        if not root.exists():
            return []
        return sorted(
            [str(p) for p in root.iterdir() if p.is_dir() and (p / required).exists()],
            reverse=True,
        )

    equity_options = _all_dirs(EQUITY_CMP_ROOT, "run_config.json")
    sweep_options  = _all_dirs(TAX_SWEEP_ROOT, "run_config.json")
    dvm_options    = _all_dirs(DAILY_VS_MONTHLY_ROOT, "calmar_comparison.csv")

    equity_bundle_pick = mo.ui.dropdown(
        options=equity_options or ["(none — run plot_equity_comparison.py first)"],
        value=equity_options[0] if equity_options else "(none — run plot_equity_comparison.py first)",
        label="Equity comparison bundle",
        full_width=True,
    )
    sweep_bundle_pick = mo.ui.dropdown(
        options=sweep_options or ["(none — run tax_threshold_sweep.py first)"],
        value=sweep_options[0] if sweep_options else "(none — run tax_threshold_sweep.py first)",
        label="Tax threshold sweep bundle",
        full_width=True,
    )
    dvm_bundle_pick = mo.ui.dropdown(
        options=dvm_options or ["(none — run daily_vs_monthly_comparison.py first)"],
        value=dvm_options[0] if dvm_options else "(none — run daily_vs_monthly_comparison.py first)",
        label="Daily vs monthly engine bundle",
        full_width=True,
    )
    mo.vstack([
        mo.md("# Tax × Drift Trigger — Deep-Dive Analysis"),
        mo.md(
            "All panels read from pre-computed CSV artifacts. "
            "To refresh data, run:\n"
            "```bash\n"
            "python research/tax_drift_trigger/plot_equity_comparison.py\n"
            "python research/tax_drift_trigger/tax_threshold_sweep.py\n"
            "python research/tax_drift_trigger/daily_vs_monthly_comparison.py\n"
            "```"
        ),
        equity_bundle_pick,
        sweep_bundle_pick,
        dvm_bundle_pick,
    ])
    return dvm_bundle_pick, equity_bundle_pick, sweep_bundle_pick


# ── Load equity comparison artifacts ─────────────────────────────────────────

@app.cell
def load_equity(equity_bundle_pick):
    _path = Path(equity_bundle_pick.value)
    _ok = _path.exists() and (_path / "equity_comparison.csv").exists()
    if not _ok:
        mo.stop(True, mo.callout(
            mo.md(f"Bundle not found: `{_path}`. Run `plot_equity_comparison.py` first."),
            kind="warn",
        ))
    equity_df  = pd.read_csv(_path / "equity_comparison.csv",
                              parse_dates=["Date"]).set_index("Date")
    rebalance_df = pd.read_csv(_path / "rebalance_dates.csv",
                                parse_dates=["Date"])
    tax_df     = pd.read_csv(_path / "tax_cumulative.csv",
                              parse_dates=["Date"]).set_index("Date")
    summary_df = pd.read_csv(_path / "summary.csv")
    with open(_path / "run_config.json") as fh:
        run_cfg = json.load(fh)

    mo.callout(
        mo.md(
            f"**Bundle:** `{_path.name}`  \n"
            f"Strategy: `{run_cfg['strategy_id']}` · "
            f"OOS window: {run_cfg['start_date']} → {run_cfg['end_date']}  \n"
            f"Regime: {run_cfg['regime'].upper()} · Selector: {run_cfg['selector'].upper()} · "
            f"Transaction cost: {run_cfg['transaction_cost_pct']*100:.1f}%"
        ),
        kind="success",
    )
    return equity_df, rebalance_df, run_cfg, summary_df, tax_df


# ── OOS window selector ───────────────────────────────────────────────────────

@app.cell
def oos_control(run_cfg):
    oos_start = mo.ui.dropdown(
        options=["2006-01-01", "2018-01-01", "2020-01-01", "2022-01-01"],
        value=run_cfg.get("start_date", "2018-01-01"),
        label="OOS window start (display only — does not recompute)",
    )
    oos_start
    return oos_start,


# ── Summary metrics table ─────────────────────────────────────────────────────

@app.cell
def summary_table(summary_df):
    mo.vstack([
        mo.md("## Summary metrics (OOS window)"),
        mo.md(
            "Key: **Calmar** = CAGR / |MDD|.  "
            "**Drift abs 5pp** (green) is the recommended production candidate (F.26 gate)."
        ),
        mo.ui.table(summary_df, pagination=False),
    ])


# ── Panel 1: Equity curves (clean) ───────────────────────────────────────────

@app.cell
def panel_equity_clean(equity_df, oos_start):
    oos = equity_df.loc[equity_df.index >= oos_start.value]

    fig, ax = plt.subplots(figsize=(13, 5.5), facecolor=DARK_BG)
    _style_ax(ax)
    for label in oos.columns:
        ax.plot(oos.index, oos[label],
                label=label,
                color=POLICY_COLORS.get(label, TEXT_COL),
                linestyle=POLICY_LS.get(label, "-"),
                linewidth=POLICY_LW.get(label, 1.2),
                alpha=0.9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.set_ylabel("Portfolio value (after-tax, USD)")
    ax.set_title("After-tax equity curves by rebalance policy (FIFO · US tax)",
                 fontweight="bold")
    ax.legend(fontsize=9.5, facecolor=PANEL_BG, edgecolor=GRID_COL,
              labelcolor=TEXT_COL, loc="upper left")
    fig.tight_layout()
    mo.vstack([mo.md("## Equity Curves"), mo.as_html(fig)])


# ── Panel 2: Equity curves + rebalance trigger markers ───────────────────────

@app.cell
def panel_equity_markers(equity_df, rebalance_df, oos_start):
    oos = equity_df.loc[equity_df.index >= oos_start.value]
    reb = rebalance_df[rebalance_df["Date"] >= pd.Timestamp(oos_start.value)]

    fig, ax = plt.subplots(figsize=(14, 5.5), facecolor=DARK_BG)
    _style_ax(ax)

    # markers first (behind curves)
    for label in oos.columns:
        sub = reb[reb["Policy"] == label]
        for _, row in sub.iterrows():
            ax.axvline(row["Date"],
                       color=POLICY_COLORS.get(label, TEXT_COL),
                       alpha=MARKER_ALPHA.get(label, 0.08),
                       linewidth=0.6)

    # curves on top
    for label in oos.columns:
        ax.plot(oos.index, oos[label],
                label=label,
                color=POLICY_COLORS.get(label, TEXT_COL),
                linestyle=POLICY_LS.get(label, "-"),
                linewidth=POLICY_LW.get(label, 1.2),
                alpha=0.92)
        n = len(reb[reb["Policy"] == label])
        ax.annotate(f"{n} trades",
                    xy=(1.0, 0.97 - list(oos.columns).index(label) * 0.055),
                    xycoords="axes fraction",
                    ha="right", va="top", fontsize=7.5,
                    color=POLICY_COLORS.get(label, TEXT_COL))

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.set_ylabel("Portfolio value (after-tax, USD)")
    ax.set_title("Equity curves with rebalance-trigger markers",
                 fontweight="bold")
    ax.legend(fontsize=9, facecolor=PANEL_BG, edgecolor=GRID_COL,
              labelcolor=TEXT_COL, loc="upper left")
    fig.tight_layout()
    mo.vstack([
        mo.md("## Equity Curves + Rebalance Triggers"),
        mo.md("Each vertical line = one rebalance event for that policy. "
              "Monthly fires every month (dense); drift fires only when weights diverge enough (sparse)."),
        mo.as_html(fig),
    ])


# ── Panel 3: Rebalance count bar chart ───────────────────────────────────────

@app.cell
def panel_rebalance_counts(rebalance_df, oos_start):
    reb = rebalance_df[rebalance_df["Date"] >= pd.Timestamp(oos_start.value)]
    counts = reb.groupby("Policy").size().reset_index(name="Count")
    # Sort by count descending
    counts = counts.sort_values("Count", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 4), facecolor=DARK_BG)
    _style_ax(ax)
    bars = ax.bar(counts["Policy"], counts["Count"],
                  color=[POLICY_COLORS.get(p, TEXT_COL) for p in counts["Policy"]],
                  alpha=0.85, width=0.55)
    for bar, v in zip(bars, counts["Count"]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1,
                str(int(v)), ha="center", va="bottom", color=TEXT_COL, fontsize=10)
    ax.set_ylabel("Number of rebalances")
    ax.set_title("Rebalance frequency — drift fires far less often",
                 fontweight="bold")
    ax.tick_params(axis="x", labelsize=8.5, rotation=15)
    fig.tight_layout()
    mo.vstack([mo.md("## Rebalance Count"), mo.as_html(fig)])


# ── Panel 4: Cumulative tax paid ──────────────────────────────────────────────

@app.cell
def panel_tax_cumulative(tax_df, oos_start):
    tax_oos = tax_df.loc[tax_df.index >= oos_start.value]

    fig, ax = plt.subplots(figsize=(13, 5), facecolor=DARK_BG)
    _style_ax(ax)
    for label in tax_oos.columns:
        ax.plot(tax_oos.index, tax_oos[label],
                label=label,
                color=POLICY_COLORS.get(label, TEXT_COL),
                linestyle=POLICY_LS.get(label, "-"),
                linewidth=POLICY_LW.get(label, 1.2),
                alpha=0.9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.set_ylabel("Cumulative tax paid (USD)")
    ax.set_title("Cumulative tax cost — monthly rebalancing triggers the highest bill",
                 fontweight="bold")
    ax.legend(fontsize=9, facecolor=PANEL_BG, edgecolor=GRID_COL,
              labelcolor=TEXT_COL, loc="upper left")
    fig.tight_layout()
    mo.vstack([
        mo.md("## Cumulative Tax Paid"),
        mo.md("Tax cost accrues faster under monthly rebalancing because gains are realised every month. "
              "Drift policies defer realisation until weights actually diverge — "
              "the gap compounds over time."),
        mo.as_html(fig),
    ])


# ── Panel 5: Calmar sweep heatmap ─────────────────────────────────────────────

@app.cell
def load_sweep(sweep_bundle_pick):
    _path = Path(sweep_bundle_pick.value)
    _ok = _path.exists() and (_path / "threshold_sweep_summary.csv").exists()
    if not _ok:
        mo.stop(True, mo.callout(
            mo.md(f"Sweep bundle not found: `{_path}`. Run `tax_threshold_sweep.py` first."),
            kind="warn",
        ))
    sweep_df = pd.read_csv(_path / "threshold_sweep_summary.csv")
    verdict = json.loads((_path / "verdict.json").read_text())
    return sweep_df, verdict


@app.cell
def panel_sweep_heatmap(sweep_df, verdict):
    regime_pick = mo.ui.radio(
        options={"US (taxable)": "us", "None (ISA / zero-tax)": "none"},
        value="US (taxable)",
        label="Tax regime",
    )
    selector_pick = mo.ui.radio(
        options={"FIFO (Alpaca reality)": "fifo", "Tax-optimal (research only)": "tax_optimal"},
        value="FIFO (Alpaca reality)",
        label="Lot selector",
    )
    mo.vstack([
        mo.md("## Calmar Sweep Heatmap (policy × OOS window)"),
        mo.hstack([regime_pick, selector_pick]),
    ])
    return regime_pick, selector_pick


@app.cell
def render_heatmap(sweep_df, verdict, regime_pick, selector_pick):
    sel = selector_pick.value
    reg = regime_pick.value

    sub = sweep_df[(sweep_df["selector"] == sel) & (sweep_df["regime"] == reg)]
    if sub.empty:
        mo.stop(True, mo.callout(mo.md(f"No data for {sel}/{reg}."), kind="warn"))

    policies = list(sub["policy"].unique())
    windows  = sorted(sub["window"].unique())

    # Build matrix: rows = policies sorted by avg Calmar, cols = windows
    pol_avg = {p: sub[sub["policy"] == p]["calmar"].mean() for p in policies}
    sorted_pol = sorted(policies, key=lambda p: pol_avg[p], reverse=True)

    matrix = np.full((len(sorted_pol), len(windows)), np.nan)
    for i, pol in enumerate(sorted_pol):
        for j, win in enumerate(windows):
            match = sub[(sub["policy"] == pol) & (sub["window"] == win)]["calmar"]
            if not match.empty:
                matrix[i, j] = match.iloc[0]

    fig, ax = plt.subplots(figsize=(8, max(5, len(sorted_pol) * 0.45 + 1.5)),
                           facecolor=DARK_BG)
    ax.set_facecolor(PANEL_BG)
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto",
                   vmin=np.nanmin(matrix) * 0.92, vmax=np.nanmax(matrix) * 1.03)

    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels([f"{w} OOS" for w in windows], color=TEXT_COL, fontsize=9)
    short_pol = [p.replace("drift_", "").replace("_unconditional", " (baseline)")
                  .replace("relative_", "rel ").replace("absolute_", "abs ")
                 for p in sorted_pol]
    ax.set_yticks(range(len(sorted_pol)))
    ax.set_yticklabels(short_pol, color=TEXT_COL, fontsize=8)

    # Annotate cells
    for i in range(len(sorted_pol)):
        for j in range(len(windows)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        color="white", fontsize=8, fontweight="bold")

    # Highlight baseline row
    baseline_idx = next((i for i, p in enumerate(sorted_pol)
                         if "monthly_unconditional" in p), None)
    if baseline_idx is not None:
        for j in range(len(windows)):
            ax.add_patch(plt.Rectangle((j - 0.5, baseline_idx - 0.5), 1, 1,
                                       fill=False, edgecolor=RED, lw=1.5))

    cb = fig.colorbar(im, ax=ax, shrink=0.7, label="Calmar Ratio")
    cb.ax.yaxis.label.set_color(TEXT_COL)
    cb.ax.tick_params(colors=TEXT_COL)

    title_regime = "US tax" if reg == "us" else "Zero tax (ISA)"
    ax.set_title(f"Calmar by policy ({sel.upper()} · {title_regime})\nRed border = monthly baseline",
                 fontweight="bold", color="white", fontsize=11)
    fig.tight_layout()

    # Verdict callout
    decision = verdict.get("decision", "unknown")
    verdict_md = (
        f"**Kill-criterion verdict:** `{decision.upper()}` · "
        + (f"{len(verdict.get('passing_policies', []))} policy/selector combos pass the ≥5% Calmar improvement gate."
           if decision != "research_only" else
           "No drift policy cleared the 5% threshold on 2+ windows.")
    )

    mo.vstack([
        mo.callout(mo.md(verdict_md), kind="success" if decision != "research_only" else "warn"),
        mo.as_html(fig),
    ])


# ── Panel 6: Tax-deferred value gain ─────────────────────────────────────────

@app.cell
def panel_value_gain(equity_df, tax_df, oos_start):
    oos_eq  = equity_df.loc[equity_df.index >= oos_start.value]
    oos_tax = tax_df.loc[tax_df.index >= oos_start.value]

    if "Monthly (baseline)" not in oos_eq.columns:
        return mo.md("")

    baseline_eq  = oos_eq["Monthly (baseline)"]
    baseline_tax = oos_tax["Monthly (baseline)"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor=DARK_BG)

    # Left: value gap relative to baseline
    _style_ax(ax1)
    for label in oos_eq.columns:
        if label == "Monthly (baseline)":
            continue
        gap = oos_eq[label] - baseline_eq
        ax1.plot(oos_eq.index, gap,
                 label=label,
                 color=POLICY_COLORS.get(label, TEXT_COL),
                 linewidth=POLICY_LW.get(label, 1.2), alpha=0.9)
    ax1.axhline(0, color=RED, linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:+,.0f}"))
    ax1.set_ylabel("Excess value vs monthly baseline (USD)")
    ax1.set_title("Value advantage over monthly rebalancing", fontweight="bold")
    ax1.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL)

    # Right: tax savings relative to baseline
    _style_ax(ax2)
    for label in oos_tax.columns:
        if label == "Monthly (baseline)":
            continue
        saving = baseline_tax - oos_tax[label]
        ax2.plot(oos_tax.index, saving,
                 label=label,
                 color=POLICY_COLORS.get(label, TEXT_COL),
                 linewidth=POLICY_LW.get(label, 1.2), alpha=0.9)
    ax2.axhline(0, color=RED, linestyle="--", linewidth=0.8, alpha=0.6)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:+,.0f}"))
    ax2.set_ylabel("Tax saved vs monthly (USD, cumulative)")
    ax2.set_title("Cumulative tax saved vs monthly rebalancing", fontweight="bold")
    ax2.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL)

    fig.suptitle("The drift advantage: more value, lower tax bill",
                 color="white", fontweight="bold", fontsize=12, y=1.01)
    fig.tight_layout()
    mo.vstack([
        mo.md("## Value & Tax Advantage vs Monthly Baseline"),
        mo.md("Both panels show how much each drift policy has gained (left) "
              "or saved in taxes (right) compared to the monthly baseline over the same window."),
        mo.as_html(fig),
    ])


# ── Panel 7: Daily vs monthly engine validation (L.53) ───────────────────────

@app.cell
def load_dvm(dvm_bundle_pick):
    _path = Path(dvm_bundle_pick.value)
    _ok = _path.exists() and (_path / "calmar_comparison.csv").exists()
    if not _ok:
        mo.stop(True, mo.callout(
            mo.md(f"Bundle not found: `{_path}`. Run `daily_vs_monthly_comparison.py` first."),
            kind="warn",
        ))
    dvm_df = pd.read_csv(_path / "calmar_comparison.csv")
    import json as _json
    dvm_summary = _json.loads((_path / "summary.json").read_text())
    return dvm_df, dvm_summary


@app.cell
def panel_daily_vs_monthly(dvm_df, dvm_summary):
    POLICY_ORDER_DVM = [
        "drift_absolute(0.04)",
        "drift_absolute(0.05)",
        "drift_absolute(0.055)",
        "drift_absolute(0.06)",
        "drift_absolute(0.065)",
        "drift_absolute(0.07)",
        "drift_relative(0.4)",
    ]
    SHORT_DVM = {
        "drift_absolute(0.04)":  "abs 4pp",
        "drift_absolute(0.05)":  "abs 5pp",
        "drift_absolute(0.055)": "abs 5.5pp",
        "drift_absolute(0.06)":  "abs 6pp",
        "drift_absolute(0.065)": "abs 6.5pp ★",
        "drift_absolute(0.07)":  "abs 7pp",
        "drift_relative(0.4)":   "rel 40%",
    }
    WINDOW_COL = {2018: ACCENT, 2020: GREEN, 2022: ORANGE}

    pol_labels = [p for p in POLICY_ORDER_DVM if p in dvm_df["policy_label"].values]
    windows = sorted(dvm_df["window"].unique())

    fig, axes = plt.subplots(1, len(windows), figsize=(15, 5), facecolor=DARK_BG, sharey=False)
    fig.suptitle("Daily engine vs monthly engine — Calmar by policy (FIFO · US tax)",
                 color="white", fontweight="bold", fontsize=12, y=1.02)

    for ax, win in zip(axes, windows):
        _style_ax(ax)
        win_df = dvm_df[dvm_df["window"] == win]
        m_vals, d_vals = [], []
        for pl in pol_labels:
            m_row = win_df[(win_df["policy_label"] == pl) & (win_df["engine"] == "monthly")]
            d_row = win_df[(win_df["policy_label"] == pl) & (win_df["engine"] == "daily")]
            m_vals.append(float(m_row["calmar"].iloc[0]) if not m_row.empty else 0)
            d_vals.append(float(d_row["calmar"].iloc[0]) if not d_row.empty else 0)

        x = np.arange(len(pol_labels))
        w = 0.34
        bars_m = ax.bar(x - w / 2, m_vals, w, label="monthly", color=PURPLE, alpha=0.78)
        bars_d = ax.bar(x + w / 2, d_vals, w, label="daily",   color=WINDOW_COL[win], alpha=0.82)

        for bar, v in zip(bars_d, d_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.003,
                    f"{v:.3f}", ha="center", va="bottom", color=TEXT_COL, fontsize=6)

        # Highlight winner (6.5pp)
        idx_w = pol_labels.index("drift_absolute(0.065)") if "drift_absolute(0.065)" in pol_labels else None
        if idx_w is not None:
            for bar in [bars_m[idx_w], bars_d[idx_w]]:
                bar.set_edgecolor("white")
                bar.set_linewidth(1.4)

        ax.set_xticks(x)
        ax.set_xticklabels([SHORT_DVM[p] for p in pol_labels],
                           rotation=40, ha="right", fontsize=7.5)
        ax.set_ylabel("Calmar Ratio")
        ax.set_ylim(0, max(max(m_vals), max(d_vals)) * 1.18)
        ax.set_title(f"OOS {win}", fontweight="bold", color=WINDOW_COL[win])
        if win == windows[0]:
            ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COL,
                      labelcolor=TEXT_COL, loc="upper left")

    fig.tight_layout()

    kill = dvm_summary["kill_criterion"]
    per_win = kill["per_window"]
    rank_rows = []
    for win_key, wr in per_win.items():
        for r in wr["ranking"][:3]:
            rank_rows.append({
                "OOS": win_key,
                "Rank": r["rank"],
                "Policy (daily engine)": r["policy_label"],
                "Calmar": r["calmar"],
            })

    mo.vstack([
        mo.md("## Daily vs Monthly Engine Validation (L.53)"),
        mo.callout(
            mo.md(
                "**Kill criterion: FAILED (0 / 3 windows).**  \n"
                "Original candidates (5.5pp, 7pp) do not rank top-3 in the daily engine on any window.  \n"
                "**New F.26 candidate confirmed: `drift_absolute(0.065)` (6.5pp)** — #1 in all 3 OOS windows.  \n"
                "7pp disqualified: MDD blows out to −19.10% in the daily engine (vs −17.12% monthly).  \n"
                "See `research/tax_drift_trigger/findings_daily_vs_monthly.md` for full analysis."
            ),
            kind="info",
        ),
        mo.as_html(fig),
        mo.md("**Top-3 policies by Calmar in the daily engine**"),
        mo.ui.table(pd.DataFrame(rank_rows), pagination=False),
    ])


# ── F.26 decision gate summary ────────────────────────────────────────────────

@app.cell
def f26_gate():
    mo.vstack([
        mo.md("## F.26 Decision Gate — What to decide"),
        mo.callout(
            mo.md("""
**Before flipping live, the following must be confirmed by a human:**

1. **Accept the candidate:** `drift_absolute(0.065)` — 6.5pp absolute threshold, FIFO lot selector.
2. **Update `strategies.json`** with a `rebalance_policy` block (currently missing).
3. **Update `live/rebalance.py`** to check drift alongside the 31-day cadence gate.
4. **One paper-month of agreement** between the live paper account and the backtest expectation.
5. **Optional:** confirm yfinance total-return sweep matches FMP (expected near-identical per 2026-05-09 validation).
6. **Daily-engine validation complete (L.53):** 6.5pp ranks #1 in all 3 OOS windows; 7pp disqualified (MDD blow-out).

Until F.26 is actioned, production remains on `monthly_unconditional`.
The estimated Calmar improvement on the table: **+28–33% across all three OOS windows (monthly engine);
+35–47% in the daily engine for the 6.5pp candidate.**
            """),
            kind="info",
        ),
    ])


if __name__ == "__main__":
    app.run()
