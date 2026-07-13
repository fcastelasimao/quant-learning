"""Per-metric inspection: 4-panel deep-dive for every registered metric.

Run with: marimo edit notebooks/02_metric_inspection.py
"""

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="wide")


@app.cell
def _imports():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

    import numpy as np
    import pandas as pd
    import marimo as mo

    from data import load_panel, SYMBOLS_ALL
    from metrics import REGISTRY, compute_all, vote_all
    from viz import inspect

    from quantcore import config as _qc_config
    DATA_DIR = _qc_config.data_dir()
    return DATA_DIR, REGISTRY, SYMBOLS_ALL, inspect, load_panel, mo, np, pd


@app.cell
def _load(DATA_DIR, SYMBOLS_ALL, load_panel):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        panel = load_panel(SYMBOLS_ALL, data_dir=DATA_DIR, warn_missing=True)
    print(f"Panel: {len(panel)} rows, {panel.index[0].date()} → {panel.index[-1].date()}")
    return (panel,)


@app.cell
def _metric_selector(REGISTRY, mo):
    metric_names = list(REGISTRY.keys())
    selector = mo.ui.dropdown(
        options=metric_names,
        value=metric_names[0],
        label="Select metric",
    )
    selector
    return (selector,)


@app.cell
def _show_metric(inspect, panel, selector):
    fig = inspect(selector.value, panel)
    fig
    return


@app.cell
def _ic_table(REGISTRY, np, panel, pd):
    """Information Coefficient table: Spearman IC vs 1/5/20-day forward returns."""
    from scipy.stats import spearmanr

    rows = []
    close = panel.get("QQQ_close")
    if close is None:
        close = panel.get("TQQQ_close")
    fwd_rets = {
        "ic_1d":  close.pct_change(1, fill_method=None).shift(-1),
        "ic_5d":  close.pct_change(5, fill_method=None).shift(-5),
        "ic_20d": close.pct_change(20, fill_method=None).shift(-20),
    }

    for name, m in REGISTRY.items():
        s = m.compute(panel).dropna()
        row = {"metric": name, "family": m.family, "status": m.status}
        for ic_name, fwd in fwd_rets.items():
            common = s.index.intersection(fwd.dropna().index)
            if len(common) < 30:
                row[ic_name] = np.nan
                row[ic_name + "_pval"] = np.nan
            else:
                ic, pval = spearmanr(s.loc[common].values, fwd.loc[common].values)
                row[ic_name] = round(ic, 4)
                row[ic_name + "_pval"] = round(pval, 4)
        rows.append(row)

    ic_df = pd.DataFrame(rows).set_index("metric")
    print("IC table (Spearman, forward QQQ returns):")
    ic_df
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
