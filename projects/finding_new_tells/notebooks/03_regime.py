"""HSMM 5-state regime model: fitting, visualization, state statistics.

Run with: marimo edit notebooks/03_regime.py
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
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import marimo as mo

    from data import load_panel, SYMBOLS_ALL
    from regime import HSMM5, build_regime_features, walk_forward_states

    from quantcore import config as _qc_config
    DATA_DIR = _qc_config.data_dir()
    COLORS = ["#2196F3", "#4CAF50", "#9E9E9E", "#FF9800", "#F44336"]
    LABELS = ["strong_bull", "weak_bull", "sideways", "weak_bear", "strong_bear"]
    return (
        COLORS,
        DATA_DIR,
        LABELS,
        SYMBOLS_ALL,
        go,
        load_panel,
        make_subplots,
        np,
        pd,
        walk_forward_states,
    )


@app.cell
def _load(DATA_DIR, SYMBOLS_ALL, load_panel):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        panel = load_panel(SYMBOLS_ALL, data_dir=DATA_DIR, warn_missing=True)
    print(f"Panel: {len(panel)} rows")
    return (panel,)


@app.cell
def _fit_regime(panel, walk_forward_states):
    """Fit HSMM walk-forward (may take a few minutes for 20y of data)."""
    print("Fitting HSMM5 walk-forward... (min_train_years=3, refit annually)")
    state_series, proba_df = walk_forward_states(panel, min_train_years=3)
    print("Done.")
    print(f"State distribution:\n{state_series.value_counts().sort_index()}")
    return proba_df, state_series


@app.cell
def _regime_plot(
    COLORS,
    LABELS,
    go,
    make_subplots,
    panel,
    proba_df,
    state_series,
):
    """Plot QQQ price with regime color bands + state probability time-series."""
    _qqq = panel.get("QQQ_close")

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.4],
        subplot_titles=["QQQ price (colored by regime state)", "State probabilities"],
    )

    if _qqq is not None:
        fig.add_trace(go.Scatter(
            x=_qqq.index, y=_qqq.values,
            name="QQQ", line_color="#BDBDBD",
        ), row=1, col=1)

    # Color bands
    if state_series is not None:
        prev = state_series.iloc[0]
        start = state_series.index[0]
        for date, state in state_series.items():
            if state != prev:
                fig.add_vrect(
                    x0=str(start), x1=str(date),
                    fillcolor=COLORS[int(prev) % len(COLORS)],
                    opacity=0.25, layer="below", line_width=0, row=1, col=1,
                )
                start = date
                prev = state
        fig.add_vrect(
            x0=str(start), x1=str(state_series.index[-1]),
            fillcolor=COLORS[int(prev) % len(COLORS)],
            opacity=0.25, layer="below", line_width=0, row=1, col=1,
        )

    # Probability curves
    if proba_df is not None:
        for _i, _label in enumerate(LABELS):
            col_name = f"p_state_{_i}"
            if col_name in proba_df.columns:
                fig.add_trace(go.Scatter(
                    x=proba_df.index, y=proba_df[col_name].values,
                    stackgroup="probs", name=_label,
                    line_color=COLORS[_i],
                ), row=2, col=1)

    fig.update_layout(height=700, title_text="5-state HSMM regime model")
    fig
    return


@app.cell
def _state_stats(LABELS, np, panel, pd, state_series):
    """Conditional statistics per regime state."""
    if state_series is None:
        print("No state series available.")
    else:
        _qqq = panel.get("QQQ_close")
        if _qqq is None:
            print("QQQ_close not available.")
        else:
            rets = _qqq.pct_change(fill_method=None)
            rows = []
            for _i, _label in enumerate(LABELS):
                mask = state_series == _i
                r = rets[mask]
                rows.append({
                    "state": _i,
                    "label": _label,
                    "count": mask.sum(),
                    "mean_ret_%": round(r.mean() * 100, 4),
                    "std_ret_%": round(r.std() * 100, 4),
                    "sharpe_ann": round(r.mean() / r.std() * np.sqrt(252), 2) if r.std() > 0 else np.nan,
                })
            pd.DataFrame(rows).set_index("state")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
