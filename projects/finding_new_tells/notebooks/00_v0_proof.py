"""V0 proof-of-concept: 3 metrics, majority vote, walk-forward backtest vs TQQQ BaH.

Run with: marimo edit notebooks/00_v0_proof.py
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

    from data import load_panel, SYMBOLS_ALL
    from metrics import REGISTRY
    from backtest import run_backtest, _perf_stats

    from quantcore import config as _qc_config
    DATA_DIR = _qc_config.data_dir()
    return DATA_DIR, REGISTRY, SYMBOLS_ALL, go, load_panel, make_subplots, pd


@app.cell
def _load_data(DATA_DIR, SYMBOLS_ALL, load_panel):
    """Load all available symbols from local SQLite databases."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        panel = load_panel(SYMBOLS_ALL, data_dir=DATA_DIR, warn_missing=True)
    print(f"Panel: {len(panel)} rows × {len(panel.columns)} columns")
    print(f"Date range: {panel.index[0].date()} → {panel.index[-1].date()}")
    panel
    return (panel,)


@app.cell
def _v0_metrics(REGISTRY, panel, pd):
    """Compute 3 V0 metrics and derive majority-vote signal."""
    V0_METRICS = ["qqq_sma50_200_regime", "qqq_rsi2", "qqq_yz_vol_20d"]

    votes = {}
    for name in V0_METRICS:
        m = REGISTRY[name]
        s = m.compute(panel)
        votes[name] = m.vote(s)

    vote_df = pd.DataFrame(votes)
    score = vote_df.mean(axis=1)
    signal = (score > 0).astype(int)   # majority buy → 1

    print("Vote counts:")
    for name in V0_METRICS:
        vc = vote_df[name].value_counts().sort_index()
        print(f"  {name}: {dict(vc)}")
    return (signal,)


@app.cell
def _v0_backtest(go, make_subplots, panel, signal):
    """Simple walk-forward backtest using the v0 majority-vote signal."""
    COST_BPS = 5.0

    tqqq_open  = panel.get("TQQQ_open")
    tqqq_close = panel.get("TQQQ_close")
    qqq_close  = panel.get("QQQ_close")

    if tqqq_open is None:
        print("TQQQ_open not available — cannot run backtest.")
        fig = go.Figure()
    else:
        # Position = yesterday's signal (fill at today's open)
        position = signal.shift(1).fillna(0).astype(int)
        pos_change = position.diff().abs().fillna(0)
        cost = pos_change * (COST_BPS / 10_000)

        daily_ret = tqqq_open.pct_change(fill_method=None).fillna(0)
        strat_ret = position * daily_ret - cost
        equity = (1 + strat_ret).cumprod()
        bah    = (1 + daily_ret).cumprod()

        p = _perf_stats(equity)
        p_bah = _perf_stats(bah)

        print(f"\n{'='*55}")
        print(f"  V0 Strategy  |  CAGR: {p['cagr']:.2%}  |  Sharpe: {p['sharpe']:.2f}")
        print(f"  MaxDD: {p['maxdd_pct']:.1f}%  |  DD Duration: {p['maxdd_duration_days']}d")
        print(f"  Exposure: {p['exposure_pct']:.1f}%")
        print(f"\n  TQQQ BaH     |  CAGR: {p_bah['cagr']:.2%}  |  Sharpe: {p_bah['sharpe']:.2f}")
        print(f"  MaxDD: {p_bah['maxdd_pct']:.1f}%  |  DD Duration: {p_bah['maxdd_duration_days']}d")
        print(f"{'='*55}\n")

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.7, 0.3],
                            subplot_titles=["Equity (log scale)", "Drawdown (%)"])
        fig.add_trace(go.Scatter(x=equity.index, y=equity.values,
                                 name="V0 Strategy", line_color="#2196F3"), row=1, col=1)
        fig.add_trace(go.Scatter(x=bah.index, y=bah.values,
                                 name="TQQQ BaH", line_color="#F44336"), row=1, col=1)

        dd_strat = (equity / equity.cummax() - 1) * 100
        dd_bah   = (bah / bah.cummax() - 1) * 100
        fig.add_trace(go.Scatter(x=dd_strat.index, y=dd_strat.values,
                                 fill="tozeroy", name="Strat DD",
                                 fillcolor="rgba(33,150,243,0.3)", line_color="#2196F3"), row=2, col=1)
        fig.add_trace(go.Scatter(x=dd_bah.index, y=dd_bah.values,
                                 fill="tozeroy", name="BaH DD",
                                 fillcolor="rgba(244,67,54,0.2)", line_color="#F44336"), row=2, col=1)

        fig.update_yaxes(type="log", row=1, col=1)
        fig.update_layout(height=600, title_text="V0 proof: 3-metric majority vote vs TQQQ BaH")

    fig
    return


if __name__ == "__main__":
    app.run()
