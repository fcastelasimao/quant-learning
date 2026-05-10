import marimo

"""Archived marimo notebook for browsing legacy timestamped backtest runs."""

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import sys
    from pathlib import Path

    import matplotlib.pyplot as plt
    import pandas as pd

    import marimo as mo

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from engine.stats import compute_calendar_year_metrics

    return PROJECT_ROOT, compute_calendar_year_metrics, mo, pd, plt


@app.cell
def _(PROJECT_ROOT, mo):
    result_dirs = sorted(
        [p for p in (PROJECT_ROOT / "results").glob("20*") if p.is_dir()],
        reverse=True,
    )
    options = [str(p.relative_to(PROJECT_ROOT)) for p in result_dirs]
    selected_run = mo.ui.dropdown(options=options, value=options[0], label="Backtest run")
    selected_run
    return (selected_run,)


@app.cell
def _(PROJECT_ROOT, compute_calendar_year_metrics, pd, selected_run):
    run_dir = PROJECT_ROOT / selected_run.value
    backtest = pd.read_csv(run_dir / "backtest_history.csv", parse_dates=["Date"], index_col="Date")
    annual_path = run_dir / "annual_metrics.csv"
    annual = (
        pd.read_csv(annual_path)
        if annual_path.exists()
        else compute_calendar_year_metrics(backtest)
    )
    annual
    return annual, run_dir


@app.cell
def _(mo, run_dir):
    mo.md(f"""
    ## Year-by-year metrics\n\nSource: `{run_dir}`
    """)
    return


@app.cell
def _(annual, mo):
    mo.ui.table(annual, label="Annual PnL and metrics")
    return


@app.cell
def _(mo):
    metric = mo.ui.dropdown(
        options=["PnL ($)", "Return (%)", "Max Drawdown (%)"],
        value="PnL ($)",
        label="Metric",
    )
    metric
    return (metric,)


@app.cell
def _(annual, metric, plt):
    pivot = annual.pivot(index="Year", columns="Strategy", values=metric.value)
    fig, ax = plt.subplots(figsize=(11, 5))
    pivot.plot(kind="bar", ax=ax, width=0.82)
    ax.set_title(f"{metric.value} by year")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig
    return (pivot,)


@app.cell
def _(mo, pivot):
    mo.ui.table(pivot.round(2).reset_index(), label="Pivot table")
    return


if __name__ == "__main__":
    app.run()
