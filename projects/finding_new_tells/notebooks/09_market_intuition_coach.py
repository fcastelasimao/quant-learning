"""Prediction-first market intuition coach.

Run with: marimo edit notebooks/09_market_intuition_coach.py
"""

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="full")


@app.cell
def _imports():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

    import warnings
    import numpy as np
    import pandas as pd
    import marimo as mo
    import plotly.graph_objects as go

    from data import load_panel
    from metrics import REGISTRY
    from learning_coach import (
        Prediction,
        build_case_packet,
        eligible_case_dates,
        journal_row,
        postmortem,
        sample_case_date,
    )

    from quantcore import config as _qc_config
    DATA_DIR = _qc_config.data_dir()
    SYMBOLS_COACH = ["TQQQ", "QQQ", "SPY", "^VIX", "^VIX3M", "^TNX", "^IRX", "HYG", "LQD"]
    return (
        DATA_DIR,
        Prediction,
        REGISTRY,
        SYMBOLS_COACH,
        build_case_packet,
        eligible_case_dates,
        go,
        journal_row,
        load_panel,
        mo,
        pd,
        postmortem,
        sample_case_date,
        warnings,
    )


@app.cell
def _load(DATA_DIR, SYMBOLS_COACH, load_panel, warnings):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        panel = load_panel(SYMBOLS_COACH, data_dir=DATA_DIR, warn_missing=True)
    print(f"Panel: {len(panel)} rows, {panel.index[0].date()} -> {panel.index[-1].date()}")
    return (panel,)


@app.cell
def _state(mo):
    journal_state, set_journal = mo.state([])
    return journal_state, set_journal


@app.cell
def _intro(mo):
    mo.md("""
    # Market Intuition Coach

    This app is a deliberate-practice layer for the TQQQ/QQQ research repo.
    First read the historical chart, make a forecast, and write down the market mechanism.
    Then reveal the forward return and metric evidence. The point is not to be right every time;
    the point is to notice what your eye missed before changing strategy code.

    Weekly routine: do 20 random cases, review misses, inspect one metric in
    `06_metric_research.py`, then write a short memo before changing a strategy rule.
    """)
    return


@app.cell
def _base_controls(REGISTRY, mo):
    split_ui = mo.ui.dropdown(
        options=["train", "val", "train+val"],
        value="train+val",
        label="Research split",
    )
    horizon_ui = mo.ui.dropdown(
        options=[1, 2, 5, 10, 20],
        value=5,
        label="Forward horizon",
    )
    lookback_ui = mo.ui.slider(
        start=42,
        stop=252,
        step=21,
        value=126,
        show_value=True,
        label="Chart lookback",
    )
    seed_ui = mo.ui.number(value=7, start=0, stop=100_000, label="Random seed")

    families = sorted({metric.family for metric in REGISTRY.values()})
    default_families = [family for family in families if not family.startswith("physics_")]
    family_ui = mo.ui.multiselect(
        options=families,
        value=default_families,
        label="Metric families revealed after forecast",
    )
    include_watch_ui = mo.ui.checkbox(value=True, label="Include watch metrics on reveal")

    mo.vstack([
        mo.md("## 1. Choose A Case"),
        mo.md(
            "The frozen test split is not available here. Forward outcomes are computed inside the selected "
            "research split, so validation-end cases cannot peek into 2022+."
        ),
        mo.hstack([split_ui, horizon_ui, lookback_ui, seed_ui]),
        mo.hstack([include_watch_ui]),
        family_ui,
    ])
    return (
        family_ui,
        horizon_ui,
        include_watch_ui,
        lookback_ui,
        seed_ui,
        split_ui,
    )


@app.cell
def _date_control(
    eligible_case_dates,
    horizon_ui,
    lookback_ui,
    mo,
    panel,
    sample_case_date,
    seed_ui,
    split_ui,
):
    eligible_dates = eligible_case_dates(
        panel,
        split=split_ui.value,
        horizon=int(horizon_ui.value),
        lookback=int(lookback_ui.value),
    )
    sampled_date = sample_case_date(
        panel,
        split=split_ui.value,
        horizon=int(horizon_ui.value),
        lookback=int(lookback_ui.value),
        random_state=int(seed_ui.value),
    )
    date_options = [date.date().isoformat() for date in eligible_dates]
    date_ui = mo.ui.dropdown(
        options=date_options,
        value=sampled_date.date().isoformat(),
        searchable=True,
        label=f"Case date ({len(date_options)} eligible)",
    )
    date_ui
    return (date_ui,)


@app.cell
def _case(
    build_case_packet,
    date_ui,
    family_ui,
    horizon_ui,
    include_watch_ui,
    lookback_ui,
    panel,
    pd,
    split_ui,
):
    case = build_case_packet(
        panel,
        split=split_ui.value,
        horizon=int(horizon_ui.value),
        lookback=int(lookback_ui.value),
        case_date=pd.Timestamp(date_ui.value),
        families=list(family_ui.value),
        include_watch=include_watch_ui.value,
    )
    return (case,)


@app.cell
def _price_context(case, go, mo, pd):
    history = case.history.copy()
    fig = go.Figure()
    for price_col, price_color, price_opacity in [
        ("TQQQ_close", "#F44336", 1.0),
        ("QQQ_close", "#607D8B", 0.75),
    ]:
        series = history[price_col].dropna()
        indexed = series / series.iloc[0] * 100
        fig.add_trace(go.Scatter(
            x=indexed.index,
            y=indexed.values,
            name=price_col.replace("_close", " indexed"),
            line={"color": price_color},
            opacity=price_opacity,
        ))
    fig.add_vline(x=case.case_date, line_dash="dot", line_color="#212121")
    fig.update_layout(
        title=f"Known history through {case.case_date.date()}",
        height=460,
        hovermode="x unified",
        margin={"l": 60, "r": 30, "t": 70, "b": 50},
        legend={"orientation": "h", "y": -0.12},
    )
    fig.update_yaxes(title_text="Indexed to 100 (log)", type="log")

    rows = []
    for days in [5, 20, 63]:
        if len(history) > days:
            rows.append({
                "lookback_days": days,
                "TQQQ_%": round((history["TQQQ_close"].iloc[-1] / history["TQQQ_close"].iloc[-days - 1] - 1) * 100, 2),
                "QQQ_%": round((history["QQQ_close"].iloc[-1] / history["QQQ_close"].iloc[-days - 1] - 1) * 100, 2),
            })
    prior_returns = pd.DataFrame(rows)

    mo.vstack([
        mo.md("## 2. Read The Tape Before Looking At Metrics"),
        fig,
        mo.md("Recent returns into the decision date:"),
        prior_returns,
    ])
    return


@app.cell
def _prediction_controls(REGISTRY, mo):
    direction_ui = mo.ui.dropdown(
        options=["bullish", "neutral", "bearish"],
        value="neutral",
        label="What happens next?",
    )
    confidence_ui = mo.ui.slider(
        start=1,
        stop=5,
        step=1,
        value=3,
        show_value=True,
        label="Confidence",
    )
    regime_ui = mo.ui.dropdown(
        options=["strong bull", "weak bull", "sideways", "weak bear", "strong bear", "unclear"],
        value="unclear",
        label="Likely regime",
    )
    family_options = ["price action only"] + sorted({metric.family for metric in REGISTRY.values()})
    expected_family_ui = mo.ui.dropdown(
        options=family_options,
        value="price action only",
        label="Metric family I expect to matter",
    )
    mechanism_ui = mo.ui.text_area(
        rows=3,
        label="Market mechanism I am relying on",
        placeholder="Example: short-term oversold bounce inside a broader uptrend.",
    )
    prove_wrong_ui = mo.ui.text_area(
        rows=3,
        label="What would prove this read wrong?",
        placeholder="Example: volatility expands and QQQ loses the recent support area.",
    )
    reveal_ui = mo.ui.checkbox(value=False, label="Reveal outcome and metric evidence")

    mo.vstack([
        mo.md("## 3. Commit Before Reveal"),
        mo.md(
            "Answer before checking the metric table. This is the useful friction: forecast, reason, then learn."
        ),
        mo.hstack([direction_ui, confidence_ui, regime_ui, expected_family_ui]),
        mechanism_ui,
        prove_wrong_ui,
        reveal_ui,
    ])
    return (
        confidence_ui,
        direction_ui,
        expected_family_ui,
        mechanism_ui,
        prove_wrong_ui,
        regime_ui,
        reveal_ui,
    )


@app.cell
def _prediction(
    Prediction,
    confidence_ui,
    direction_ui,
    expected_family_ui,
    mechanism_ui,
    prove_wrong_ui,
    regime_ui,
):
    prediction = Prediction(
        direction=direction_ui.value,
        confidence=int(confidence_ui.value),
        likely_regime=regime_ui.value,
        mechanism=mechanism_ui.value.strip(),
        expected_family=expected_family_ui.value,
        prove_wrong=prove_wrong_ui.value.strip(),
    )
    return (prediction,)


@app.cell
def _reveal(case, mo, pd, postmortem, prediction, reveal_ui):
    if not reveal_ui.value:
        revealed = None
        reveal_output = mo.md(
            "## 4. Reveal\n\nKeep the outcome hidden until you have made a prediction and written a mechanism."
        )
    else:
        revealed = postmortem(case, prediction)
        snapshot = case.metric_snapshot.copy()
        for snapshot_col in ["value", "strength"]:
            snapshot[snapshot_col] = snapshot[snapshot_col].astype(float).round(4)
        snapshot = snapshot.sort_values(
            ["vote", "strength"],
            ascending=[False, False],
            kind="stable",
        ).reset_index(drop=True)

        agreeing = revealed["agreeing_metrics"].copy()
        disagreeing = revealed["disagreeing_metrics"].copy()
        for reveal_table in [agreeing, disagreeing]:
            for reveal_col in ["value", "strength"]:
                if reveal_col in reveal_table.columns:
                    reveal_table[reveal_col] = reveal_table[reveal_col].astype(float).round(4)

        outcome_table = pd.DataFrame([{
            "case_date": case.case_date.date().isoformat(),
            "entry_date": pd.Timestamp(case.outcome["entry_date"]).date().isoformat(),
            "exit_date": pd.Timestamp(case.outcome["exit_date"]).date().isoformat(),
            "horizon": case.horizon,
            "TQQQ_fwd_%": round(float(case.outcome["tqqq_forward_return"]) * 100, 3),
            "QQQ_fwd_%": round(float(case.outcome["qqq_forward_return"]) * 100, 3),
            "actual": revealed["actual_direction"],
            "prediction": revealed["predicted_direction"],
            "score": revealed["direction_score"],
            "pattern": revealed["pattern"],
        }])

        reveal_output = mo.vstack([
            mo.md("## 4. Reveal"),
            outcome_table,
            mo.md(f"**Postmortem:** {revealed['critique']}"),
            mo.md("### Metrics Most Aligned With The Outcome"),
            agreeing,
            mo.md("### Metrics Most Against The Outcome"),
            disagreeing,
            mo.md("### Full Revealed Metric Snapshot"),
            snapshot,
        ])
    reveal_output
    return (revealed,)


@app.cell
def _journal_button(
    case,
    journal_row,
    journal_state,
    mo,
    prediction,
    reveal_ui,
    revealed,
    set_journal,
):
    if reveal_ui.value and revealed is not None:
        row = journal_row(case, prediction, revealed).iloc[0].to_dict()
        add_button = mo.ui.button(
            label="Add revealed case to session journal",
            kind="success",
            on_click=lambda _: set_journal(journal_state() + [row]),
        )
        reset_button = mo.ui.button(
            label="Clear session journal",
            kind="warn",
            on_click=lambda _: set_journal([]),
        )
        controls = mo.hstack([add_button, reset_button])
    else:
        controls = mo.md("Reveal a case to enable journal logging.")
    controls
    return


@app.cell
def _journal(journal_state, mo, pd):
    journal = pd.DataFrame(journal_state())
    if journal.empty:
        journal_output = mo.md("## Session Journal\n\nNo cases logged yet.")
    else:
        journal_output = mo.vstack([
            mo.md("## Session Journal"),
            mo.ui.table(journal, selection=None, show_download=True, page_size=20),
        ])
    journal_output
    return


if __name__ == "__main__":
    app.run()
