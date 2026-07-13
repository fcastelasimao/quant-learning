"""Interactive TQQQ indicator workbench for visual alpha discovery.

Run with: marimo edit notebooks/05_indicator_workbench.py
"""

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="wide")


@app.cell
def _imports():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

    import marimo as mo
    import pandas as pd

    from data import load_panel, SYMBOLS_ALL
    from metrics import REGISTRY
    from viz import DEFAULT_WORKBENCH_METRICS, indicator_workbench

    from quantcore import config as _qc_config
    DATA_DIR = _qc_config.data_dir()
    return (
        DATA_DIR,
        DEFAULT_WORKBENCH_METRICS,
        REGISTRY,
        SYMBOLS_ALL,
        indicator_workbench,
        load_panel,
        mo,
    )


@app.cell
def _load(DATA_DIR, SYMBOLS_ALL, load_panel):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        panel = load_panel(SYMBOLS_ALL, data_dir=DATA_DIR, warn_missing=True)
    print(f"Panel: {len(panel)} rows, {panel.index[0].date()} -> {panel.index[-1].date()}")
    return (panel,)


@app.cell
def _metric_controls(DEFAULT_WORKBENCH_METRICS, REGISTRY, mo):
    metric_names = list(REGISTRY.keys())
    default_metrics = [name for name in DEFAULT_WORKBENCH_METRICS if name in REGISTRY]
    metrics_ui = mo.ui.multiselect(
        options=metric_names,
        value=default_metrics,
        label="Metrics",
    )
    return (metrics_ui,)


@app.cell
def _date_controls(mo, panel):
    start_ui = mo.ui.text(
        value=str(panel.index[0].date()),
        label="Start date",
    )
    end_ui = mo.ui.text(
        value=str(panel.index[-1].date()),
        label="End date",
    )
    show_qqq_ui = mo.ui.checkbox(
        value=True,
        label="Show QQQ",
    )
    mo.hstack([start_ui, end_ui, show_qqq_ui])
    return end_ui, show_qqq_ui, start_ui


@app.cell
def _plot(
    end_ui,
    indicator_workbench,
    metrics_ui,
    mo,
    panel,
    show_qqq_ui,
    start_ui,
):
    fig = None
    skipped = {}
    try:
        fig, _, skipped = indicator_workbench(
            panel,
            selected_metrics=list(metrics_ui.value),
            start=start_ui.value,
            end=end_ui.value,
            show_qqq=show_qqq_ui.value,
        )
    except Exception as exc:
        _output = mo.vstack([metrics_ui, mo.md(f"Workbench error: `{exc}`")])
    else:
        if skipped:
            print("Skipped metrics:")
            for name, reason in skipped.items():
                print(f"- {name}: {reason}")
        _output = mo.vstack([metrics_ui, fig])
    _output
    return


if __name__ == "__main__":
    app.run()
