import marimo

__generated_with = "0.23.5"
app = marimo.App(width="full")

with app.setup:
    import json
    import sys
    from pathlib import Path

    import marimo as mo
    import pandas as pd

    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from research.strategy_plotting import (
        COLORS,
        STRATEGY_ORDER,
        latest_bundle,
        plot_calendar_profile,
        plot_drawdowns,
        plot_growth,
        plot_implementation_realism,
        plot_monthly_returns,
        plot_risk_diagnostics,
        plot_rolling_behaviour,
    )

    BUNDLE_ROOTS = [
        ROOT / "results" / "production_validation",
        ROOT / "results" / "strategy_comparison",
    ]


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
    """
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
    ])"""
    return


@app.cell
def plot_growth_cell(daily):
    mo.md("## Overview: Growth of Money")
    plot_growth(daily)
    return


@app.cell
def plot_drawdowns_cell(daily, dd_events):
    mo.md("## Core Risk Story: Drawdowns")
    mo.vstack([
        mo.as_html(plot_drawdowns(daily)),
        mo.ui.table(dd_events, label="Worst Drawdown Events"),
    ])
    return


@app.cell
def plot_calendar_profile_cell(calendar):
    mo.md("## Calendar-Year Profile")
    plot_calendar_profile(calendar)
    return


@app.cell
def plot_rolling_behaviour_cell(rolling):
    mo.md("## Rolling Behaviour")
    plot_rolling_behaviour(rolling)
    return


@app.cell
def plot_monthly_returns_cell(monthly):
    mo.md("## Return Distribution and Monthly Heatmaps")
    plot_monthly_returns(monthly)
    return


@app.cell
def plot_risk_diagnostics_cell(summary):
    mo.md("## Institutional Risk Diagnostics")
    mo.vstack([
        mo.as_html(plot_risk_diagnostics(summary)),
        mo.ui.table(summary, label="Summary Metrics"),
    ])
    return


@app.cell
def plot_implementation_realism_cell(risk_contrib, stress, turnover):
    mo.md("## Implementation Realism")
    mo.vstack([
        mo.as_html(plot_implementation_realism(risk_contrib, turnover)),
        mo.hstack([
            mo.ui.table(risk_contrib, label="Risk Contribution"),
            mo.ui.table(turnover.tail(24), label="Recent Turnover and Costs"),
        ], gap=2),
        mo.ui.table(stress, label="Stress Period Metrics"),
    ])
    return


if __name__ == "__main__":
    app.run()
