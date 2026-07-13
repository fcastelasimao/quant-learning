"""Data sanity checks: alignment, splits, coverage, basic stats.

Run with: marimo edit notebooks/01_data_sanity.py
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

    from data import load_panel, SYMBOLS_ALL, SYMBOLS_CORE

    from quantcore import config as _qc_config
    DATA_DIR = _qc_config.data_dir()
    return DATA_DIR, SYMBOLS_ALL, go, load_panel, make_subplots


@app.cell
def _load(DATA_DIR, SYMBOLS_ALL, load_panel):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        panel = load_panel(SYMBOLS_ALL, data_dir=DATA_DIR, warn_missing=True)
    print(f"Panel shape: {panel.shape}")
    print(f"Date range: {panel.index[0].date()} → {panel.index[-1].date()}")
    print(f"Missing values per column (in %):")
    na_pct = panel.isna().mean() * 100
    print(na_pct[na_pct > 0].to_string())
    return (panel,)


@app.cell
def _coverage(go, make_subplots, panel):
    """Show data coverage timeline per symbol."""
    _close_cols = [_c for _c in panel.columns if _c.endswith("_close")]
    syms = [_c.replace("_close", "") for _c in _close_cols]

    _fig = make_subplots(rows=1, cols=1)
    for _i, (_sym, _col) in enumerate(zip(syms, _close_cols)):
        valid = panel[_col].notna()
        x = panel.index[valid]
        if len(x) == 0:
            continue
        _fig.add_trace(go.Scatter(
            x=[x[0], x[-1]], y=[_i, _i],
            mode="lines+text",
            text=[_sym, ""], textposition="middle right",
            line={"width": 6},
            name=_sym,
        ))

    _fig.update_layout(
        title_text="Data coverage by symbol",
        height=max(300, len(syms) * 30),
        showlegend=False,
        yaxis={"showticklabels": False},
    )
    _fig
    return


@app.cell
def _split_check(go, make_subplots, panel):
    """Verify equity/ETF symbols have no >50% single-day jumps (split-adjustment sanity).

    Indices (^VIX, ^TNX, etc.) are intentionally excluded: volatility indices can
    legitimately spike >50% during crises, and rate indices are in % units where a
    50% relative change is plausible near-zero (e.g. IRX).
    """
    _EQUITY_SYMS = ["TQQQ", "QQQ", "SPY", "SPXL", "HYG", "LQD"]
    _available = [(_sym, panel[f"{_sym}_close"]) for _sym in _EQUITY_SYMS if f"{_sym}_close" in panel.columns]

    for _sym, series in _available:
        bad = series.pct_change(fill_method=None).abs()
        bad = bad[bad > 0.5]
        if len(bad) == 0:
            print(f"PASS: {_sym} — no >50% single-day jumps")
        else:
            print(f"WARN: {_sym} — {len(bad)} jump(s) >50%:")
            print(bad.to_string())

    if _available:
        _n = len(_available)
        _fig = make_subplots(rows=_n, cols=1, shared_xaxes=True,
                             subplot_titles=[_sym for _sym, _ in _available],
                             vertical_spacing=0.04)
        for _i, (_sym, series) in enumerate(_available, start=1):
            _fig.add_trace(go.Scatter(x=series.index, y=series.values, name=_sym), row=_i, col=1)
            _fig.update_yaxes(type="log", row=_i, col=1)
        _fig.update_layout(title_text="Equity/ETF close prices (log scale)",
                           height=220 * _n, showlegend=False)
        _fig
    return


@app.cell
def _correlation(go, panel):
    """Close-price return correlation matrix for core symbols."""
    _close_cols = [c for c in panel.columns if c.endswith("_close")]
    rets = panel[_close_cols].pct_change(fill_method=None).dropna()
    corr = rets.corr()

    _fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu",
        zmid=0,
        text=corr.round(2).values.astype(str),
        texttemplate="%{text}",
    ))
    _fig.update_layout(title_text="Return correlation matrix", height=500)
    _fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
