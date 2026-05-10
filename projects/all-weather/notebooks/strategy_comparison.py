import marimo

__generated_with = "0.23.5"
app = marimo.App(width="full")

with app.setup:
    import json
    from pathlib import Path

    import marimo as mo
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np
    import pandas as pd

    ROOT = Path(__file__).resolve().parents[1]
    BUNDLE_ROOTS = [
        ROOT / "results" / "production_validation",
        ROOT / "results" / "strategy_comparison",
    ]

    DARK_BG = "#0d1117"
    PANEL_BG = "#161b22"
    TEXT_COL = "#c9d1d9"
    GRID_COL = "#30363d"
    COLORS = {
        "My Strategy (DIY)": "#58a6ff",
        "S&P 500 (SPY)": "#f78166",
        "ALLW (Bridgewater)": "#f0b429",
        "60/40 (SPY/TLT)": "#3fb950",
    }
    STRATEGY_ORDER = list(COLORS)

    def style_ax(ax):
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT_COL, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(GRID_COL)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.label.set_color(TEXT_COL)
        ax.xaxis.label.set_color(TEXT_COL)
        ax.title.set_color("white")
        ax.grid(axis="y", color=GRID_COL, alpha=0.45, linewidth=0.7)

    def ordered(items):
        return [s for s in STRATEGY_ORDER if s in items] + sorted(set(items) - set(STRATEGY_ORDER))

    def latest_bundle(roots: list[Path]) -> str:
        dirs = []
        for root in roots:
            if root.exists():
                dirs.extend(p for p in root.iterdir() if p.is_dir() and (p / "manifest.json").exists())
        return str(max(dirs, key=lambda p: p.stat().st_mtime)) if dirs else ""


@app.cell
def choose_bundle():
    bundle_path = mo.ui.text(
        value=latest_bundle(BUNDLE_ROOTS),
        label="Result bundle path",
        full_width=True,
    )
    mo.vstack([
        mo.md("# Bank-Facing Strategy Comparison"),
        bundle_path,
    ])
    return (bundle_path,)


@app.cell
def load_bundle(bundle_path):
    from pathlib import Path as _Path

    _raw_bundle_path = bundle_path.value.strip().strip("'\"")
    if not _raw_bundle_path:
        mo.stop(
            True,
            mo.md(
                "Generate a strategy comparison bundle first, then paste its path above. "
                "Example: `results/production_validation/<timestamp>_<strategy_id>`"
            ),
        )

    bundle = _Path(_raw_bundle_path).expanduser()
    if not bundle.is_absolute():
        bundle = ROOT / bundle
    _manifest_path = bundle / "manifest.json"
    if not bundle.exists() or not _manifest_path.exists():
        mo.stop(
            True,
            mo.md(
                f"Could not find a valid result bundle at `{bundle}`. "
                "The selected folder must contain `manifest.json` and the exported CSV files. "
                "Generate one with `python -m research.production_validation`."
            ),
        )

    with open(_manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    _provenance_path = bundle / "price_provenance.json"
    price_provenance = {}
    if _provenance_path.exists():
        with open(_provenance_path, "r", encoding="utf-8") as handle:
            price_provenance = json.load(handle)

    daily = pd.read_csv(bundle / "daily_series.csv", parse_dates=["Date"])
    monthly = pd.read_csv(bundle / "monthly_returns.csv", parse_dates=["Date"])
    summary = pd.read_csv(bundle / "summary_metrics.csv")
    calendar = pd.read_csv(bundle / "calendar_year_metrics.csv")
    rolling = pd.read_csv(bundle / "rolling_metrics.csv", parse_dates=["Date"])
    dd_events = pd.read_csv(bundle / "drawdown_events.csv")
    stress = pd.read_csv(bundle / "stress_period_metrics.csv")
    risk_contrib = pd.read_csv(bundle / "risk_contribution.csv")
    turnover = pd.read_csv(bundle / "turnover_costs.csv", parse_dates=["Date"])
    return (
        calendar,
        daily,
        dd_events,
        manifest,
        monthly,
        price_provenance,
        risk_contrib,
        rolling,
        stress,
        summary,
        turnover,
    )


@app.cell
def show_manifest(manifest, price_provenance):
    allocation = pd.DataFrame([
        {"Asset": asset, "Weight": weight, "Weight (%)": f"{weight:.1%}"}
        for asset, weight in manifest["allocation"].items()
    ])
    fees = pd.DataFrame([
        {"Strategy": key, "Fee (%)": f"{value:.2%}"}
        for key, value in manifest["fees"].items()
    ])
    provenance = price_provenance or {}
    provenance_rows = [
        ("Source", provenance.get("source", manifest.get("data_source", "Unavailable"))),
        ("Price column", provenance.get("price_column", "Unavailable")),
        ("Pricing model", provenance.get("pricing_model") or "Unavailable"),
        ("Retrieved on", provenance.get("retrieved_on", "Unavailable")),
        ("Actual start", provenance.get("actual_start", manifest["date_range"].get("actual_start"))),
        ("Actual end", provenance.get("actual_end", manifest["date_range"].get("actual_end"))),
        ("Returned columns", ", ".join(provenance.get("returned_columns", [])) or "Unavailable"),
    ]
    provenance_df = pd.DataFrame(provenance_rows, columns=["Field", "Value"])
    missingness = provenance.get("missing_fraction_by_column", {})
    missingness_df = pd.DataFrame([
        {"Ticker": ticker, "Missing fraction": value}
        for ticker, value in missingness.items()
    ])

    mo.vstack([
        mo.md(
            f"**Strategy:** `{manifest['strategy_id']}`  |  "
            f"**Generated:** {manifest['generated_at']}  |  "
            f"**Data:** {manifest['data_source']}  |  "
            f"**Actual range:** {manifest['date_range']['actual_start']} to "
            f"{manifest['date_range']['actual_end']}"
        ),
        mo.hstack([
            mo.ui.table(allocation, label="Target Allocation"),
            mo.ui.table(fees, label="Fee Assumptions"),
        ], gap=2),
        mo.hstack([
            mo.ui.table(provenance_df, label="Price Provenance"),
            mo.ui.table(missingness_df, label="Missingness by Column"),
        ], gap=2),
    ])
    return


@app.cell
def plot_growth(daily):
    mo.md("## Overview: Growth of Money")

    _value = daily.pivot(index="Date", columns="Strategy", values="Indexed Value")
    _overlap = daily.pivot(index="Date", columns="Strategy", values="Overlap Indexed Value")

    _fig, _axes = plt.subplots(2, 1, figsize=(13, 8.5), sharex=False)
    _fig.patch.set_facecolor(DARK_BG)

    for _ax, _data, _title, _ylabel in [
        (_axes[0], _value, "Full available history", "Indexed value"),
        (_axes[1], _overlap, "ALLW overlap window", "Indexed value"),
    ]:
        style_ax(_ax)
        for _strategy in ordered(_data.columns):
            _s = _data[_strategy].dropna()
            if _s.empty:
                continue
            _ax.plot(_s.index, _s.values, color=COLORS.get(_strategy), lw=2.0, label=_strategy)
            _ax.annotate(
                f"{_s.iloc[-1]:.0f}",
                xy=(_s.index[-1], _s.iloc[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                color=COLORS.get(_strategy, TEXT_COL),
                fontsize=8,
                va="center",
            )
        _ax.set_title(_title, fontsize=11, pad=8)
        _ax.set_ylabel(_ylabel)
        _ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"{value:.0f}"))
        _ax.legend(fontsize=8, facecolor="#21262d", edgecolor="#30363d", labelcolor=TEXT_COL)

    _axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(_axes[1].xaxis.get_majorticklabels(), rotation=20, ha="right")
    plt.tight_layout(pad=1.4)
    _fig
    return


@app.cell
def plot_drawdowns(daily, dd_events):
    mo.md("## Core Risk Story: Drawdowns")

    _drawdowns = daily.pivot(index="Date", columns="Strategy", values="Drawdown (%)")
    _fig, _ax = plt.subplots(figsize=(13, 4.8))
    _fig.patch.set_facecolor(DARK_BG)
    style_ax(_ax)

    for _strategy in ordered(_drawdowns.columns):
        _s = _drawdowns[_strategy].dropna()
        if not _s.empty:
            _ax.plot(_s.index, _s.values, color=COLORS.get(_strategy), lw=1.9, label=_strategy)

    _ax.axhline(0, color="#8b949e", lw=0.8)
    _ax.set_title("Underwater drawdown curve", fontsize=11, pad=8)
    _ax.set_ylabel("Drawdown (%)")
    _ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _ax.legend(fontsize=8, facecolor="#21262d", edgecolor="#30363d", labelcolor=TEXT_COL)
    plt.tight_layout(pad=1.2)

    mo.vstack([
        mo.as_html(_fig),
        mo.ui.table(dd_events, label="Worst Drawdown Events"),
    ])
    return


@app.cell
def plot_calendar_profile(calendar):
    mo.md("## Calendar-Year Profile")

    _metrics = [
        ("Return (%)", "Calendar return"),
        ("Max Drawdown (%)", "Max drawdown"),
        ("Max DD Duration (days)", "Max DD duration"),
        ("Calmar", "Calmar"),
    ]
    _metrics = [(_metric, _title) for _metric, _title in _metrics if _metric in calendar.columns]
    _strategies = ordered(calendar["Strategy"].unique())
    _years = sorted(calendar["Year"].dropna().unique())
    _x = np.arange(len(_years))
    _width = 0.8 / max(len(_strategies), 1)

    _fig, _axes = plt.subplots(len(_metrics), 1, figsize=(13, 3.3 * len(_metrics)), sharex=True)
    _fig.patch.set_facecolor(DARK_BG)
    _axes = np.atleast_1d(_axes)

    for _ax, (_metric, _title) in zip(_axes, _metrics):
        style_ax(_ax)
        for _idx, _strategy in enumerate(_strategies):
            _data = calendar[calendar["Strategy"] == _strategy].set_index("Year")
            _values = [_data.loc[_year, _metric] if _year in _data.index else np.nan for _year in _years]
            _offset = (_idx - (len(_strategies) - 1) / 2) * _width
            _ax.bar(_x + _offset, _values, width=_width, color=COLORS.get(_strategy), alpha=0.85, label=_strategy)
        _ax.axhline(0, color="#8b949e", lw=0.7)
        _ax.set_title(_title, fontsize=10, pad=6)
        _ax.set_ylabel(_metric, fontsize=8)

    _axes[0].legend(fontsize=8, facecolor="#21262d", edgecolor="#30363d", labelcolor=TEXT_COL)
    _axes[-1].set_xticks(_x)
    _axes[-1].set_xticklabels([str(_year) for _year in _years], rotation=25, ha="right")
    plt.tight_layout(pad=1.3)
    _fig
    return


@app.cell
def plot_rolling_behaviour(rolling):
    mo.md("## Rolling Behaviour")

    _panels = [
        ("Rolling CAGR (%)", "Rolling CAGR"),
        ("Rolling Max Drawdown (%)", "Rolling max drawdown"),
        ("Rolling Corr to SPY", "Rolling correlation to SPY"),
        ("Rolling Beta to SPY", "Rolling beta to SPY"),
    ]

    _fig, _axes = plt.subplots(2, 2, figsize=(13, 8))
    _fig.patch.set_facecolor(DARK_BG)

    for _ax, (_metric, _title) in zip(_axes.flat, _panels):
        style_ax(_ax)
        for _strategy in ordered(rolling["Strategy"].unique()):
            for _window, _linestyle in [("1Y", "-"), ("3Y", "--")]:
                _data = rolling[(rolling["Strategy"] == _strategy) & (rolling["Window"] == _window)]
                if _data.empty or _metric not in _data:
                    continue
                _ax.plot(
                    _data["Date"],
                    _data[_metric],
                    color=COLORS.get(_strategy),
                    linestyle=_linestyle,
                    lw=1.6,
                    alpha=0.9,
                    label=f"{_strategy} {_window}",
                )
        _ax.axhline(0, color="#8b949e", lw=0.7)
        _ax.set_title(_title, fontsize=10, pad=6)
    _axes[0, 0].legend(fontsize=7, facecolor="#21262d", edgecolor="#30363d", labelcolor=TEXT_COL, ncol=2)
    plt.tight_layout(pad=1.2)
    _fig
    return


@app.cell
def plot_monthly_returns(monthly):
    mo.md("## Return Distribution and Monthly Heatmaps")

    _strategies = ordered(monthly["Strategy"].unique())
    _fig, _axes = plt.subplots(len(_strategies), 2, figsize=(13, max(4, len(_strategies) * 2.6)))
    _fig.patch.set_facecolor(DARK_BG)
    if len(_strategies) == 1:
        _axes = np.array([_axes])

    _month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    for _row, _strategy in enumerate(_strategies):
        _data = monthly[monthly["Strategy"] == _strategy]
        _ax_hm, _ax_hist = _axes[_row]
        _pivot = _data.pivot(index="Year", columns="Month", values="Return (%)").reindex(columns=range(1, 13))
        _ax_hm.set_facecolor(PANEL_BG)
        _arr = _pivot.values
        _valid = _arr[~np.isnan(_arr)]
        _vmax = max(abs(_valid).max(), 1) if len(_valid) else 1
        _ax_hm.imshow(_arr, aspect="auto", cmap="RdYlGn", vmin=-_vmax, vmax=_vmax)
        _ax_hm.set_xticks(range(12))
        _ax_hm.set_xticklabels(_month_labels, fontsize=7, color=TEXT_COL)
        _ax_hm.set_yticks(range(len(_pivot.index)))
        _ax_hm.set_yticklabels(_pivot.index.astype(str), fontsize=7, color=TEXT_COL)
        _ax_hm.set_title(f"{_strategy} monthly returns", color="white", fontsize=9)
        for _spine in _ax_hm.spines.values():
            _spine.set_color(GRID_COL)

        _ax_hist.set_facecolor(PANEL_BG)
        _ax_hist.hist(_data["Return (%)"].dropna(), bins=24, color=COLORS.get(_strategy), alpha=0.85)
        _ax_hist.axvline(0, color="#8b949e", lw=0.8)
        _ax_hist.tick_params(colors=TEXT_COL, labelsize=7)
        _ax_hist.set_title(f"{_strategy} monthly distribution", color="white", fontsize=9)
        for _spine in _ax_hist.spines.values():
            _spine.set_color(GRID_COL)

    plt.tight_layout(pad=1.2)
    _fig
    return


@app.cell
def plot_risk_diagnostics(summary):
    mo.md("## Institutional Risk Diagnostics")

    _overlap = summary[summary["Window"] == "ALLW Overlap"]
    if _overlap.empty:
        _overlap = summary[summary["Window"] == "Full History"]

    _metrics = [
        "CAGR (%)", "Volatility (%)", "Sharpe", "Sortino",
        "Calmar", "Ulcer Index", "Max Drawdown (%)", "Max DD Duration (days)",
        "VaR 5% Daily (%)", "CVaR 5% Daily (%)", "Downside Beta", "Up Capture (%)",
    ]
    _strategies = ordered(_overlap["Strategy"].unique())
    _x = np.arange(len(_strategies))

    _fig, _axes = plt.subplots(3, 4, figsize=(14, 9))
    _fig.patch.set_facecolor(DARK_BG)

    for _ax, _metric in zip(_axes.flat, _metrics):
        style_ax(_ax)
        _vals = [
            _overlap.loc[_overlap["Strategy"] == _strategy, _metric].iloc[0]
            if _metric in _overlap and not _overlap.loc[_overlap["Strategy"] == _strategy, _metric].empty
            else np.nan
            for _strategy in _strategies
        ]
        _ax.bar(_x, _vals, color=[COLORS.get(_strategy) for _strategy in _strategies], alpha=0.85)
        _ax.set_title(_metric, fontsize=8, pad=4)
        _ax.set_xticks(_x)
        _ax.set_xticklabels([_strategy.replace(" ", "\n", 1) for _strategy in _strategies], fontsize=6, color=TEXT_COL)

    plt.tight_layout(pad=1.0)
    mo.vstack([
        mo.as_html(_fig),
        mo.ui.table(summary, label="Summary Metrics"),
    ])
    return


@app.cell
def plot_implementation_realism(risk_contrib, stress, turnover):
    mo.md("## Implementation Realism")

    _fig, _axes = plt.subplots(1, 2, figsize=(13, 4.6))
    _fig.patch.set_facecolor(DARK_BG)

    style_ax(_axes[0])
    if not risk_contrib.empty:
        _axes[0].bar(
            risk_contrib["Asset"],
            risk_contrib["Risk Contribution (%)"],
            color="#58a6ff",
            alpha=0.85,
        )
    _axes[0].set_title("Risk contribution by asset", fontsize=10, pad=6)
    _axes[0].set_ylabel("Risk contribution (%)")

    style_ax(_axes[1])
    if not turnover.empty:
        _axes[1].plot(
            turnover["Date"],
            turnover["Cumulative Cost Drag (%)"],
            color="#f0b429",
            lw=2.0,
        )
    _axes[1].set_title("Estimated cumulative transaction-cost drag", fontsize=10, pad=6)
    _axes[1].set_ylabel("Cost drag (%)")

    plt.tight_layout(pad=1.2)
    mo.vstack([
        mo.as_html(_fig),
        mo.hstack([
            mo.ui.table(risk_contrib, label="Risk Contribution"),
            mo.ui.table(turnover.tail(24), label="Recent Turnover and Costs"),
        ], gap=2),
        mo.ui.table(stress, label="Stress Period Metrics"),
    ])
    return


if __name__ == "__main__":
    app.run()
