import marimo

__generated_with = "0.23.5"
app = marimo.App(width="full")

with app.setup:
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.ticker as mticker
    import numpy as np
    import pandas as pd

    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    # ── Style constants (dark theme) ──────────────────────────────────────────
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

    VERDICT_COLORS = {
        "closed":             RED,
        "reopened":           ORANGE,
        "active-research":    PURPLE,
        "production":         GREEN,
        "production (gated)": CYAN,
        "todo":               GRID_COL,
    }
    VERDICT_LABELS = {
        "closed":             "CLOSED",
        "reopened":           "REOPENED",
        "active-research":    "ACTIVE RESEARCH",
        "production":         "PRODUCTION",
        "production (gated)": "PRODUCTION (GATED)",
        "todo":               "TODO",
    }

    # ── Result bundle locators ────────────────────────────────────────────────
    RESULTS = ROOT / "results"
    EQUITY_CMP_ROOT         = RESULTS / "equity_comparison"
    TAX_SWEEP_ROOT          = RESULTS / "tax_threshold_sweep"
    STRATEGY_CMP_ROOT       = RESULTS / "strategy_comparison"
    PRODUCTION_VAL_ROOT     = RESULTS / "production_validation"
    LEVERAGE_CMP_ROOT       = RESULTS / "leverage_comparison"
    MIXED_OOS_ROOT          = RESULTS / "mixed_leverage_oos_validation"

    def _latest(root: Path, required: str = "run_config.json") -> Path | None:
        if not root.exists():
            return None
        candidates = [p for p in root.iterdir() if p.is_dir() and (p / required).exists()]
        return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None

    def _latest_any(root: Path) -> Path | None:
        if not root.exists():
            return None
        candidates = [p for p in root.iterdir() if p.is_dir()]
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
        ax.grid(axis="y", color=GRID_COL, alpha=0.45, linewidth=0.6)


# ── Research investigations metadata ─────────────────────────────────────────

@app.cell
def investigations():
    INVESTIGATIONS = [
        {
            "id": "universe_selection",
            "name": "Universe Selection",
            "verdict": "closed",
            "summary": "6-asset confirmed optimal over 50k subsets and 8-asset variants",
            "metric": "Calmar",
            "best": 0.452,
            "data": {
                "labels": ["6-asset\n(production)", "8-asset A\n(no TIP)", "8-asset B\n(with TIP)"],
                "windows": ["2018 OOS", "2020 OOS", "2022 OOS"],
                "values": {
                    "6-asset\n(production)": [0.452, 0.457, 0.376],
                    "8-asset A\n(no TIP)":   [0.368, 0.385, 0.345],
                    "8-asset B\n(with TIP)": [0.395, 0.432, 0.359],
                },
            },
        },
        {
            "id": "optimiser_comparison",
            "name": "Optimiser Comparison",
            "verdict": "closed",
            "summary": "Differential Evolution overfits IS; Risk Parity (SLSQP) generalises",
            "metric": "Calmar",
            "best": 0.45,
            "data": {
                "labels": ["IS (2006–2020)", "OOS (2020–2026)"],
                "de":  [0.72, 0.15],
                "rp":  [0.45, 0.45],
            },
        },
        {
            "id": "rolling_rp",
            "name": "Rolling Risk Parity",
            "verdict": "closed",
            "summary": "Rolling RP weights converge to the static solution on every split",
            "metric": "Weight convergence",
            "best": None,
            "data": None,  # needs weight_history.csv from results/
        },
        {
            "id": "rebalance_frequency",
            "name": "Rebalance Frequency (pre-tax)",
            "verdict": "reopened",
            "summary": "No pre-tax improvement — reopened under US tax (see tax_drift_trigger)",
            "metric": "Calmar",
            "best": 0.376,
            "data": {
                "windows": ["2018 OOS", "2020 OOS", "2022 OOS"],
                "monthly": [0.452, 0.457, 0.376],
                "weekly":  [0.446, 0.449, 0.369],
            },
        },
        {
            "id": "momentum_overlay",
            "name": "Momentum Overlay",
            "verdict": "closed",
            "summary": "Re-entry timing not learnable: IS winner collapses OOS",
            "metric": "Calmar",
            "best": 0.48,
            "data": {
                "labels": ["Baseline", "Best IS combo", "Same combo OOS"],
                "values": [0.48, 0.72, 0.43],
                "colors": [GREEN, ACCENT, RED],
            },
        },
        {
            "id": "bond_leverage",
            "name": "Bond Leverage",
            "verdict": "closed",
            "summary": "Calmar collapses in rising-rate regime; leverage destroys risk-adjusted returns",
            "metric": "Calmar vs leverage",
            "best": None,
            "data": None,
        },
        {
            "id": "allw_benchmark",
            "name": "ALLW Benchmark",
            "verdict": "production",
            "summary": "DIY risk-parity beats ALLW (Bridgewater) on Calmar since fund launch",
            "metric": "Calmar",
            "best": 2.797,
            "data": None,  # loads from daily_series.csv
        },
        {
            "id": "data_source_validation",
            "name": "Data Source Validation",
            "verdict": "production",
            "summary": "yfinance total-return ≈ FMP adj_close; price-return materially understates",
            "metric": "Calmar diff",
            "best": 0.001,
            "data": {
                "windows": ["2018 OOS", "2020 OOS", "2022 OOS"],
                "yf_total": [0.487, 0.503, 0.452],
                "fmp_adj":  [0.488, 0.504, 0.453],
                "yf_price": [0.385, 0.391, 0.341],
            },
        },
        {
            "id": "tax_drift_trigger",
            "name": "Tax × Drift Trigger",
            "verdict": "production (gated)",
            "summary": "Every drift policy beats monthly under US tax (+28–33% Calmar); gate: F.26",
            "metric": "Calmar improvement",
            "best": 0.362,
            "data": None,  # loads from sweep CSV
        },
        {
            "id": "daily_vs_monthly_validation",
            "name": "Daily Engine Validation",
            "verdict": "production (gated)",
            "summary": "Daily engine confirms drift wins; 6.5pp #1 all windows; 7pp disqualified (MDD blow-out)",
            "metric": "Calmar (daily)",
            "best": 0.4094,
            "data": None,  # loads from daily_vs_monthly CSV
        },
        {
            "id": "rsi_leverage_overlay",
            "name": "RSI Leverage Overlay",
            "verdict": "active-research",
            "summary": "SPY+GLD strongest; walk-forward validation gate pending",
            "metric": "Calmar delta",
            "best": None,
            "data": None,
        },
        {
            "id": "production_validation",
            "name": "Production Validation",
            "verdict": "production",
            "summary": "Bundle builder confirmed; DIY beats all benchmarks on Calmar",
            "metric": "Calmar",
            "best": None,
            "data": None,
        },
        {
            "id": "shadow_comparison",
            "name": "Shadow Comparison",
            "verdict": "todo",
            "summary": "Live vs simulated reconciliation — not yet started",
            "metric": "—",
            "best": None,
            "data": None,
        },
    ]
    return INVESTIGATIONS,


# ── Verdict filter control ────────────────────────────────────────────────────

@app.cell
def controls(INVESTIGATIONS):
    all_verdicts = list(dict.fromkeys(i["verdict"] for i in INVESTIGATIONS))

    verdict_filter = mo.ui.multiselect(
        options=all_verdicts,
        value=all_verdicts,
        label="Show verdicts",
    )
    mo.vstack([
        mo.md("# All-Weather Research — Overview"),
        mo.md(
            "One panel per investigation. Plots read from CSV artifacts where available; "
            "hardcoded from findings docs for closed investigations without regenerated data."
        ),
        verdict_filter,
    ])
    return verdict_filter,


# ── Overview verdict table ────────────────────────────────────────────────────

@app.cell
def overview_table(INVESTIGATIONS, verdict_filter):
    filtered = [i for i in INVESTIGATIONS if i["verdict"] in verdict_filter.value]
    rows = []
    for inv in filtered:
        rows.append({
            "Investigation":  inv["name"],
            "Verdict":        VERDICT_LABELS[inv["verdict"]],
            "Summary":        inv["summary"],
            "Best metric":    f"{inv['metric']} = {inv['best']}" if inv["best"] else inv["metric"],
        })
    table_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(14, max(2.5, len(filtered) * 0.58 + 0.6)),
                           facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, len(filtered) - 0.5)
    ax.invert_yaxis()
    ax.set_axis_off()

    for i, inv in enumerate(filtered):
        c = VERDICT_COLORS[inv["verdict"]]
        ax.barh(i, 9.6, left=0.2, height=0.72, color=c, alpha=0.10,
                edgecolor=c, linewidth=0.7)
        ax.text(0.35, i, inv["name"], fontsize=9, color="white",
                fontweight="bold", va="center")
        ax.text(3.5, i, VERDICT_LABELS[inv["verdict"]], fontsize=7.5, color=c,
                fontweight="bold", va="center",
                bbox=dict(boxstyle="round,pad=0.25", facecolor=DARK_BG,
                          edgecolor=c, linewidth=0.7))
        ax.text(5.6, i, inv["summary"], fontsize=7.5, color=TEXT_COL, va="center")

    legend_elements = [
        mpatches.Patch(facecolor=c, alpha=0.45, edgecolor=c, label=VERDICT_LABELS[v])
        for v, c in VERDICT_COLORS.items()
    ]
    ax.legend(handles=legend_elements, loc="lower center", ncol=len(legend_elements),
              fontsize=7, facecolor=PANEL_BG, edgecolor=GRID_COL,
              labelcolor=TEXT_COL, bbox_to_anchor=(0.5, -0.06))
    ax.set_title("Research Verdict Map", color="white", fontweight="bold",
                 fontsize=12, pad=10)
    fig.tight_layout()
    mo.vstack([
        mo.md(f"**{len(filtered)} investigation(s) shown**"),
        mo.as_html(fig),
    ])
    return filtered, table_df


# ── Universe selection panel ──────────────────────────────────────────────────

@app.cell
def panel_universe(filtered, INVESTIGATIONS):
    inv = next((i for i in filtered if i["id"] == "universe_selection"), None)
    if inv is None:
        return mo.md("")

    d = inv["data"]
    windows = d["windows"]
    x = np.arange(len(windows))
    width = 0.25
    colors = [GREEN, RED, ORANGE]

    fig, ax = plt.subplots(figsize=(9, 4), facecolor=DARK_BG)
    _style_ax(ax)
    for j, (label, vals) in enumerate(d["values"].items()):
        bars = ax.bar(x + j * width, vals, width, label=label,
                      color=colors[j], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.004,
                    f"{v:.3f}", ha="center", va="bottom", color=TEXT_COL, fontsize=7)
    ax.set_xticks(x + width)
    ax.set_xticklabels(windows)
    ax.set_ylabel("Calmar Ratio")
    ax.set_ylim(0, 0.55)
    ax.set_title("Universe Selection — 6-asset wins all OOS windows  ✓ CLOSED",
                 fontweight="bold", color="white")
    ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL)
    fig.tight_layout()
    mo.vstack([mo.md("## Universe Selection"), mo.as_html(fig)])


# ── Optimiser comparison panel ────────────────────────────────────────────────

@app.cell
def panel_optimiser(filtered, INVESTIGATIONS):
    inv = next((i for i in filtered if i["id"] == "optimiser_comparison"), None)
    if inv is None:
        return mo.md("")

    d = inv["data"]
    x = np.arange(len(d["labels"]))
    width = 0.32

    fig, ax = plt.subplots(figsize=(8, 4), facecolor=DARK_BG)
    _style_ax(ax)
    ax.bar(x - width / 2, d["de"], width, label="Differential Evolution",
           color=RED, alpha=0.85)
    ax.bar(x + width / 2, d["rp"], width, label="Risk Parity (SLSQP)",
           color=GREEN, alpha=0.85)
    for i_, (dv, rv) in enumerate(zip(d["de"], d["rp"])):
        ax.text(i_ - width / 2, dv + 0.01, f"{dv:.2f}", ha="center",
                color=TEXT_COL, fontsize=9)
        ax.text(i_ + width / 2, rv + 0.01, f"{rv:.2f}", ha="center",
                color=TEXT_COL, fontsize=9)
    ax.annotate("DE overfits:\nIS-heavy weights\ncollapse OOS",
                xy=(1 - width / 2, 0.15), xytext=(0.65, 0.60),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.5),
                fontsize=8, color=ORANGE, ha="center")
    ax.set_xticks(x)
    ax.set_xticklabels(d["labels"])
    ax.set_ylabel("Calmar Ratio")
    ax.set_ylim(0, 0.90)
    ax.set_title("Optimiser Comparison — DE overfits, RP generalises  ✓ CLOSED",
                 fontweight="bold")
    ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL)
    fig.tight_layout()
    mo.vstack([mo.md("## Optimiser Comparison"), mo.as_html(fig)])


# ── Rebalance frequency panel ─────────────────────────────────────────────────

@app.cell
def panel_rebalance_freq(filtered, INVESTIGATIONS):
    inv = next((i for i in filtered if i["id"] == "rebalance_frequency"), None)
    if inv is None:
        return mo.md("")

    d = inv["data"]
    x = np.arange(len(d["windows"]))
    width = 0.32

    fig, ax = plt.subplots(figsize=(8, 4), facecolor=DARK_BG)
    _style_ax(ax)
    ax.bar(x - width / 2, d["monthly"], width, label="Monthly", color=GREEN, alpha=0.85)
    ax.bar(x + width / 2, d["weekly"],  width, label="Weekly",  color=ORANGE, alpha=0.85)
    for i_ in range(len(d["windows"])):
        ax.text(i_ - width / 2, d["monthly"][i_] + 0.004, f"{d['monthly'][i_]:.3f}",
                ha="center", color=TEXT_COL, fontsize=8)
        ax.text(i_ + width / 2, d["weekly"][i_]  + 0.004, f"{d['weekly'][i_]:.3f}",
                ha="center", color=TEXT_COL, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(d["windows"])
    ax.set_ylabel("Calmar Ratio")
    ax.set_ylim(0, 0.55)
    ax.set_title(
        "Rebalance Frequency (pre-tax) — No improvement  ⚠ REOPENED under US tax",
        fontweight="bold", color=ORANGE)
    ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL)
    fig.tight_layout()
    mo.vstack([
        mo.md("## Rebalance Frequency (pre-tax)"),
        mo.callout(
            mo.md("**Reopened.** Pre-tax: no improvement. "
                  "Post-tax: drift beats monthly by +28–33% Calmar. "
                  "See **Tax × Drift Trigger** below and `notebooks/tax_drift_analysis.py`."),
            kind="warn",
        ),
        mo.as_html(fig),
    ])


# ── Momentum overlay panel ────────────────────────────────────────────────────

@app.cell
def panel_momentum(filtered, INVESTIGATIONS):
    inv = next((i for i in filtered if i["id"] == "momentum_overlay"), None)
    if inv is None:
        return mo.md("")

    d = inv["data"]
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=DARK_BG)
    _style_ax(ax)
    bars = ax.bar(d["labels"], d["values"], color=d["colors"], alpha=0.85, width=0.5)
    for bar, v in zip(bars, d["values"]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                f"{v:.2f}", ha="center", color=TEXT_COL, fontsize=10)
    ax.axhline(y=0.48, color=GREEN, linestyle="--", alpha=0.4, linewidth=1)
    ax.annotate("IS winner hurts OOS",
                xy=(2, 0.43), xytext=(2, 0.60),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5),
                fontsize=9, color=RED, ha="center")
    ax.set_ylabel("Calmar Ratio")
    ax.set_ylim(0, 0.85)
    ax.set_title("Momentum Overlay — Re-entry timing not learnable  ✓ CLOSED",
                 fontweight="bold")
    fig.tight_layout()
    mo.vstack([mo.md("## Momentum Overlay"), mo.as_html(fig)])


# ── Data source validation panel ──────────────────────────────────────────────

@app.cell
def panel_data_source(filtered, INVESTIGATIONS):
    inv = next((i for i in filtered if i["id"] == "data_source_validation"), None)
    if inv is None:
        return mo.md("")

    d = inv["data"]
    x = np.arange(len(d["windows"]))
    width = 0.24

    fig, ax = plt.subplots(figsize=(9, 4), facecolor=DARK_BG)
    _style_ax(ax)
    ax.bar(x - width, d["yf_total"], width, label="yfinance total-return",
           color=GREEN,  alpha=0.85)
    ax.bar(x,          d["fmp_adj"],  width, label="FMP adj_close",
           color=ACCENT, alpha=0.85)
    ax.bar(x + width,  d["yf_price"], width, label="yfinance price-return",
           color=RED,   alpha=0.85)
    for i_ in range(len(d["windows"])):
        for offset, vals in [(-width, d["yf_total"]), (0, d["fmp_adj"]),
                              (width, d["yf_price"])]:
            ax.text(i_ + offset, vals[i_] + 0.004, f"{vals[i_]:.3f}",
                    ha="center", color=TEXT_COL, fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(d["windows"])
    ax.set_ylabel("Calmar Ratio")
    ax.set_ylim(0, 0.60)
    ax.set_title("Data Source Validation — total-return ≈ adj_close; price-return understates  ✓ PRODUCTION",
                 fontweight="bold", color=GREEN)
    ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL)
    fig.tight_layout()
    mo.vstack([mo.md("## Data Source Validation"), mo.as_html(fig)])


# ── ALLW benchmark panel (loads from daily_series.csv) ───────────────────────

@app.cell
def panel_allw_benchmark(filtered, INVESTIGATIONS):
    inv = next((i for i in filtered if i["id"] == "allw_benchmark"), None)
    if inv is None:
        return mo.md("")

    # Try loading from strategy_comparison bundle
    bundle = _latest_any(STRATEGY_CMP_ROOT) or _latest_any(PRODUCTION_VAL_ROOT)
    if bundle is None:
        return mo.vstack([
            mo.md("## ALLW Benchmark"),
            mo.callout(mo.md("No strategy comparison bundle found. Run `python -m research.production_validation.production_validation` first."), kind="warn"),
        ])

    daily_csv = bundle / "daily_series.csv"
    if not daily_csv.exists():
        return mo.vstack([mo.md("## ALLW Benchmark"), mo.callout(mo.md(f"`daily_series.csv` not in `{bundle.name}`"), kind="warn")])

    ds = pd.read_csv(daily_csv, parse_dates=["Date"])
    allw_start = pd.Timestamp("2025-03-06")
    overlap = ds[ds["Date"] >= allw_start].copy()

    STRAT_COLORS = {
        "My Strategy (DIY)":    GREEN,
        "S&P 500 (SPY)":        ORANGE,
        "ALLW (Bridgewater)":   ACCENT,
        "60/40 (SPY/TLT)":      PURPLE,
        "JEPQ (JPM Nasdaq Income)": "#c678dd",
    }
    ORDER = ["My Strategy (DIY)", "ALLW (Bridgewater)", "JEPQ (JPM Nasdaq Income)",
             "S&P 500 (SPY)", "60/40 (SPY/TLT)"]

    fig, ax = plt.subplots(figsize=(12, 5), facecolor=DARK_BG)
    _style_ax(ax)
    for strat in ORDER:
        sub = overlap[overlap["Strategy"] == strat] if "Strategy" in overlap.columns else pd.DataFrame()
        if sub.empty:
            continue
        short = (strat.replace("My Strategy (DIY)", "DIY RP")
                      .replace("S&P 500 (SPY)", "SPY")
                      .replace("ALLW (Bridgewater)", "ALLW")
                      .replace("60/40 (SPY/TLT)", "60/40")
                      .replace("JEPQ (JPM Nasdaq Income)", "JEPQ"))
        ax.plot(sub["Date"], sub["Overlap Indexed Value"],
                label=short, color=STRAT_COLORS.get(strat, TEXT_COL), linewidth=1.5)
    ax.axhline(100, color=GRID_COL, linestyle="--", alpha=0.4, linewidth=0.8)
    ax.set_ylabel("Indexed value (100 = ALLW launch)")
    ax.set_title("Performance since ALLW launch (2025-03-06)  ✓ PRODUCTION",
                 fontweight="bold", color=GREEN)
    ax.legend(fontsize=9, facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL)
    fig.tight_layout()
    mo.vstack([mo.md("## ALLW Benchmark"), mo.as_html(fig)])


# ── Tax × Drift Trigger panel ─────────────────────────────────────────────────

@app.cell
def panel_tax_drift(filtered, INVESTIGATIONS):
    inv = next((i for i in filtered if i["id"] == "tax_drift_trigger"), None)
    if inv is None:
        return mo.md("")

    # 1. Equity curves (from plot_equity_comparison results)
    eq_bundle = _latest(EQUITY_CMP_ROOT, "run_config.json")
    equity_fig = None
    if eq_bundle:
        eq_csv = eq_bundle / "equity_comparison.csv"
        reb_csv = eq_bundle / "rebalance_dates.csv"
        if eq_csv.exists() and reb_csv.exists():
            equity_df = pd.read_csv(eq_csv, parse_dates=["Date"]).set_index("Date")
            reb_df = pd.read_csv(reb_csv, parse_dates=["Date"])
            oos = equity_df.loc[equity_df.index >= "2018-01-01"]
            reb_oos = reb_df[reb_df["Date"] >= pd.Timestamp("2018-01-01")]

            PCOLS = {
                "Monthly (baseline)": "#f85149",
                "Drift abs 2pp": "#d29922",
                "Drift abs 5pp": "#3fb950",
                "Drift rel 25%": "#58a6ff",
                "Drift rel 40%": "#bc8cff",
            }
            PALPHA = {
                "Monthly (baseline)": 0.05,
                "Drift abs 2pp": 0.12,
                "Drift abs 5pp": 0.18,
                "Drift rel 25%": 0.10,
                "Drift rel 40%": 0.10,
            }
            PLW = {col: (2.2 if col == "Drift abs 5pp" else 1.2) for col in PCOLS}
            PLS = {col: ("--" if col == "Monthly (baseline)" else "-") for col in PCOLS}

            equity_fig, ax = plt.subplots(figsize=(13, 5), facecolor=DARK_BG)
            _style_ax(ax)
            for label in oos.columns:
                sub = reb_oos[reb_oos["Policy"] == label]
                for _, row in sub.iterrows():
                    ax.axvline(row["Date"],
                               color=PCOLS.get(label, TEXT_COL),
                               alpha=PALPHA.get(label, 0.08), linewidth=0.6)
            for label in oos.columns:
                ax.plot(oos.index, oos[label], label=label,
                        color=PCOLS.get(label, TEXT_COL),
                        linestyle=PLS.get(label, "-"),
                        linewidth=PLW.get(label, 1.2), alpha=0.92)
                n = len(reb_oos[reb_oos["Policy"] == label])
                ax.annotate(f"{n} trades",
                            xy=(1.0, 0.97 - list(oos.columns).index(label) * 0.055),
                            xycoords="axes fraction", ha="right", va="top",
                            fontsize=7.5, color=PCOLS.get(label, TEXT_COL))
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
            ax.set_ylabel("Portfolio value (after-tax)")
            ax.set_title("After-tax equity curves by rebalance policy (FIFO · US tax)",
                         fontweight="bold")
            ax.legend(fontsize=8.5, facecolor=PANEL_BG, edgecolor=GRID_COL,
                      labelcolor=TEXT_COL, loc="upper left")
            equity_fig.tight_layout()

    # 2. Calmar bar chart from sweep CSV (fallback if no equity comparison)
    sweep_bundle = _latest(TAX_SWEEP_ROOT, "run_config.json")
    sweep_fig = None
    if sweep_bundle:
        sweep_csv = sweep_bundle / "threshold_sweep_summary.csv"
        if sweep_csv.exists():
            df = pd.read_csv(sweep_csv)
            fifo_us = df[(df["selector"] == "fifo") & (df["regime"] == "us")]
            windows = sorted(fifo_us["window"].unique())
            wc = {2018: ACCENT, 2020: GREEN, 2022: ORANGE}
            policies = fifo_us["policy"].unique()
            pol_avg = {p: fifo_us[fifo_us["policy"] == p]["calmar"].mean() for p in policies}
            sorted_pol = sorted(policies, key=lambda p: pol_avg[p], reverse=True)
            x = np.arange(len(sorted_pol))
            w = 0.26

            sweep_fig, ax = plt.subplots(figsize=(13, 4), facecolor=DARK_BG)
            _style_ax(ax)
            for i, win in enumerate(windows):
                vals = [fifo_us[(fifo_us["policy"] == p) & (fifo_us["window"] == win)]["calmar"].iloc[0]
                        if len(fifo_us[(fifo_us["policy"] == p) & (fifo_us["window"] == win)]) > 0 else 0
                        for p in sorted_pol]
                ax.bar(x + i * w, vals, w, label=f"{win} OOS",
                       color=wc.get(win, ACCENT), alpha=0.82)
            monthly_avg = pol_avg.get("monthly_unconditional", 0)
            ax.axhline(monthly_avg, color=RED, linestyle="--", alpha=0.7, linewidth=1.2,
                       label="Monthly baseline avg")
            short = [p.replace("drift_", "d_").replace("_unconditional", "")
                       .replace("relative_", "r").replace("absolute_", "a")
                     for p in sorted_pol]
            ax.set_xticks(x + w)
            ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
            ax.set_ylabel("Calmar Ratio")
            ax.set_title("Calmar by policy (FIFO/US tax) — every drift policy beats monthly",
                         fontweight="bold")
            ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL)
            sweep_fig.tight_layout()

    parts = [mo.md("## Tax × Drift Trigger")]
    parts.append(mo.callout(
        mo.md("**PRODUCTION (GATED)** — F.26 human gate required before flipping live. "
              "Recommended candidate: `drift_absolute(0.065)` (6.5pp, FIFO) — "
              "confirmed #1 in all 3 OOS windows by the daily engine (L.53). "
              "See `notebooks/tax_drift_analysis.py` for the full deep-dive."),
        kind="info",
    ))
    if equity_fig:
        parts.append(mo.as_html(equity_fig))
    if sweep_fig:
        parts.append(mo.as_html(sweep_fig))
    if not equity_fig and not sweep_fig:
        parts.append(mo.callout(mo.md("No result bundles found. Run `plot_equity_comparison.py` and `tax_threshold_sweep.py` first."), kind="warn"))
    mo.vstack(parts)


# ── RSI Leverage overlay panel ────────────────────────────────────────────────

@app.cell
def panel_leverage(filtered, INVESTIGATIONS):
    inv = next((i for i in filtered if i["id"] == "rsi_leverage_overlay"), None)
    if inv is None:
        return mo.md("")

    bundle = _latest_any(LEVERAGE_CMP_ROOT)
    if bundle is None:
        return mo.vstack([
            mo.md("## RSI Leverage Overlay"),
            mo.callout(mo.md("No leverage comparison bundle found. Run `research/rsi_leverage_overlay/build_leverage_comparison_report.py` first."), kind="warn"),
        ])

    grid_csv = bundle / "threshold_grid.csv"
    if not grid_csv.exists():
        return mo.vstack([mo.md("## RSI Leverage Overlay"), mo.callout(mo.md(f"`threshold_grid.csv` not in `{bundle.name}`"), kind="warn")])

    grid = pd.read_csv(grid_csv)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=DARK_BG)
    for idx, ticker in enumerate(["SPY", "GLD"]):
        ax = axes[idx]
        _style_ax(ax)
        sub = grid[(grid["Ticker"] == ticker) & (grid["Overlay Weight"] == 0.3)]
        if sub.empty:
            ax.set_title(f"{ticker}: no data at 30%", color="white")
            continue
        entries = sorted(sub["Entry Threshold"].unique())
        exits   = sorted(sub["Exit Threshold"].unique())
        hmap = np.full((len(entries), len(exits)), np.nan)
        for _, row in sub.iterrows():
            ei = entries.index(row["Entry Threshold"])
            xi = exits.index(row["Exit Threshold"])
            hmap[ei, xi] = row["Calmar"]
        im = ax.imshow(hmap, cmap="RdYlGn", aspect="auto",
                       vmin=np.nanmin(hmap) * 0.95, vmax=np.nanmax(hmap) * 1.02)
        ax.set_xticks(range(len(exits)));  ax.set_xticklabels([f"{int(e)}" for e in exits], fontsize=6, rotation=45)
        ax.set_yticks(range(len(entries))); ax.set_yticklabels([f"{int(e)}" for e in entries], fontsize=6)
        ax.set_xlabel("RSI Exit", fontsize=9); ax.set_ylabel("RSI Entry", fontsize=9)
        ax.set_title(f"{ticker} @ 30% leverage — Calmar heatmap", fontweight="bold")
        best = np.unravel_index(np.nanargmax(hmap), hmap.shape)
        ax.plot(best[1], best[0], "w*", markersize=12)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("RSI Leverage Overlay: Entry × Exit heatmap  ⚡ ACTIVE RESEARCH",
                 color=PURPLE, fontweight="bold", fontsize=12, y=1.01)
    fig.tight_layout()
    mo.vstack([
        mo.md("## RSI Leverage Overlay"),
        mo.callout(mo.md("Gate: walk-forward / train-test validation before any promotion. "
                         "See `notebooks/leverage_comparison.py` for full analysis."), kind="info"),
        mo.as_html(fig),
    ])


# ── Shadow comparison panel ───────────────────────────────────────────────────

@app.cell
def panel_shadow(filtered, INVESTIGATIONS):
    inv = next((i for i in filtered if i["id"] == "shadow_comparison"), None)
    if inv is None:
        return mo.md("")

    mo.vstack([
        mo.md("## Shadow Comparison (Live vs Simulated)"),
        mo.callout(
            mo.md("**TODO** — Section G in the active work plan. "
                  "Script stub at `research/shadow_comparison/backtest_shadow.py`. "
                  "Needs several months of executed rebalances in `live/logs/` before results are meaningful."),
            kind="warn",
        ),
    ])


if __name__ == "__main__":
    app.run()
