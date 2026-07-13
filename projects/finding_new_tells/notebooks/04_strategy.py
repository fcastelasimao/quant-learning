"""Full strategy: vote ensemble + regime-conditional softmax + walk-forward backtest.

Run with: marimo edit notebooks/04_strategy.py
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
    import marimo as mo

    from data import load_panel, SYMBOLS_ALL
    from metrics import REGISTRY
    from regime import walk_forward_states
    from strategy import decide_series, _DEFAULT_ALPHA, _DEFAULT_TAU
    from backtest import run_backtest, append_master_log
    from viz import backtest_report, dashboard

    from quantcore import config as _qc_config
    DATA_DIR = _qc_config.data_dir()
    return (
        DATA_DIR,
        SYMBOLS_ALL,
        backtest_report,
        dashboard,
        load_panel,
        mo,
        np,
        run_backtest,
    )


@app.cell
def _load(DATA_DIR, SYMBOLS_ALL, load_panel):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        panel = load_panel(SYMBOLS_ALL, data_dir=DATA_DIR, warn_missing=True)
    print(f"Panel: {len(panel)} rows, {panel.index[0].date()} → {panel.index[-1].date()}")
    return (panel,)


@app.cell
def _split_selector(mo):
    split_ui = mo.ui.dropdown(
        options=["train", "val"],  # "test" intentionally excluded until strategy is final
        value="val",
        label="Evaluation split (DO NOT select test until strategy is final)",
    )
    split_ui
    return (split_ui,)


@app.cell
def _tau_slider(mo):
    tau_ui = mo.ui.slider(start=0.1, stop=3.0, step=0.1, value=1.0, label="Temperature τ")
    tau_ui
    return (tau_ui,)


@app.cell
def _run(panel, run_backtest, split_ui, tau_ui):
    """Run the walk-forward backtest on the selected split."""
    from backtest import _DEFAULT_ALPHA
    print(f"Running backtest — split={split_ui.value}, τ={tau_ui.value}")
    result = run_backtest(
        panel,
        tau=tau_ui.value,
        alpha=_DEFAULT_ALPHA,
        use_regime=True,
        split=split_ui.value,
    )
    p = result.perf
    print(f"\nSplit: {result.split}")
    print(f"CAGR: {p['cagr']:.2%}  |  Sharpe: {p['sharpe']:.2f}  |  Sortino: {p['sortino']:.2f}")
    print(f"Calmar: {p['calmar']:.2f}  |  MaxDD: {p['maxdd_pct']:.1f}%  ({p['maxdd_duration_days']}d)")
    print(f"Exposure: {p['exposure_pct']:.1f}%  |  Hit rate: {p.get('hit_rate', 0):.1%}")
    print(f"vs TQQQ BaH: {p['vs_tqqq_bh_excess_cagr']:+.2%}")
    print(f"Gap: {result.gap_return_pct:.2f}%  |  Intraday: {result.intraday_return_pct:.2f}%")
    return (result,)


@app.cell
def _report(backtest_report, result):
    """Full 5-row backtest report."""
    _fig = backtest_report(result)
    _fig
    return


@app.cell
def _dashboard_cell(dashboard, np, panel, result):
    """Dashboard at the most recent date."""
    latest_date = panel.index[-1]
    regime_state = int(result.regime_states.iloc[-1]) if result.regime_states is not None else 2
    regime_probs = result.regime_probs.iloc[-1].values if result.regime_probs is not None else np.array([0.2]*5)

    _fig = dashboard(panel, latest_date, regime_state, regime_probs, result.regime_states)
    _fig
    return


@app.cell
def _log_to_master():
    """Append this run to MASTER_LOG.csv.

    Uncomment the append call when you're happy with the result.
    NEVER log test-split results unless strategy is finalized.
    """
    # append_master_log(
    #     result,
    #     strategy_version="v1",
    #     tau=tau_ui.value,
    #     split=result.split,
    #     notes=f"Full v1 strategy, tau={tau_ui.value}",
    # )
    print("Append to MASTER_LOG.csv is commented out — uncomment when ready.")
    return


if __name__ == "__main__":
    app.run()
