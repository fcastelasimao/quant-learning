"""Client-facing research status review.

Run with: marimo edit notebooks/08_progress_review.py
"""

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="wide")


@app.cell
def _imports():
    import sys
    import warnings
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import marimo as mo

    from backtest import COST_BPS, _perf_stats as perf_stats
    from data import SYMBOLS_ALL, load_panel
    from metrics import REGISTRY
    from signal_diagnostics import (
        TRAIN_END,
        VAL_END,
        VAL_START,
        apply_cost_model,
        metric_decision_table,
        regime_conditional_edge_table,
        rolling_edge_decay,
        split_panel,
        vote_dynamics_report,
    )
    from strategy_v2 import (
        configs_from_decision_table,
        run_v2_backtest,
        select_threshold_on_train,
    )
    from viz import (
        regime_conditional_heatmap,
        rolling_edge_decay_small_multiples,
    )

    from quantcore import config as _qc_config
    ROOT = Path(__file__).parents[1]
    DATA_DIR = _qc_config.data_dir()
    OUTPUT_DIR = ROOT / "outputs"
    return (
        COST_BPS,
        DATA_DIR,
        OUTPUT_DIR,
        REGISTRY,
        SYMBOLS_ALL,
        apply_cost_model,
        configs_from_decision_table,
        go,
        load_panel,
        make_subplots,
        metric_decision_table,
        mo,
        pd,
        perf_stats,
        regime_conditional_edge_table,
        regime_conditional_heatmap,
        rolling_edge_decay,
        rolling_edge_decay_small_multiples,
        run_v2_backtest,
        select_threshold_on_train,
        split_panel,
        vote_dynamics_report,
        warnings,
    )


@app.cell
def _load(DATA_DIR, SYMBOLS_ALL, load_panel, warnings):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        panel = load_panel(SYMBOLS_ALL, data_dir=DATA_DIR, warn_missing=True)
    research_panel = panel.loc[:"2021-12-31"].copy()
    return panel, research_panel


@app.cell
def _helpers(COST_BPS, go, make_subplots, pd, perf_stats):
    def round_df(df: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
        out = df.copy()
        for col in out.select_dtypes(include=["float"]).columns:
            out[col] = out[col].round(digits)
        return out

    def nav_from_exposure(panel_eval: pd.DataFrame, exposure: pd.Series, *, cost_bps: float = COST_BPS) -> pd.Series:
        exposure = exposure.reindex(panel_eval.index).fillna(0.0).astype(float)
        pos_change = exposure.diff().abs().fillna(0)
        if len(pos_change):
            pos_change.iloc[0] = abs(float(exposure.iloc[0]))
        ret = panel_eval["TQQQ_open"].pct_change(fill_method=None).fillna(0)
        interval_exposure = exposure.shift(1).fillna(0.0)
        strategy_ret = interval_exposure * ret - pos_change * (cost_bps / 10_000)
        return (1 + strategy_ret).cumprod()

    def perf_row(name: str, equity: pd.Series, *, exposure: pd.Series | None = None, split: str = "val") -> dict:
        row = {"split": split, "name": name}
        row.update(perf_stats(equity))
        if exposure is not None:
            pos_change = exposure.diff().abs().fillna(0)
            if len(pos_change):
                pos_change.iloc[0] = abs(float(exposure.iloc[0]))
            row["exposure_pct"] = float(exposure.mean() * 100)
            row["trade_count"] = int((pos_change > 0).sum())
            row["turnover"] = float(pos_change.mean())
        return row

    def comparison_figure(series_by_name: dict[str, pd.Series]):
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.65, 0.35],
            vertical_spacing=0.05,
            subplot_titles=["Validation NAV, indexed to 1", "Drawdown"],
        )
        colors = {
            "corrected_v2": "#2563EB",
            "v1_saved": "#64748B",
            "v0_simple": "#059669",
            "qqq_buy_hold": "#7C3AED",
            "tqqq_buy_hold": "#DC2626",
            "tqqq_fixed_25": "#F59E0B",
            "tqqq_fixed_50": "#EA580C",
        }
        for name, series in series_by_name.items():
            fig.add_trace(
                go.Scatter(
                    x=series.index,
                    y=series.values / series.dropna().iloc[0],
                    name=name,
                    line_color=colors.get(name),
                ),
                row=1,
                col=1,
            )
            dd = series / series.cummax() - 1
            fig.add_trace(
                go.Scatter(
                    x=dd.index,
                    y=dd.values * 100,
                    name=f"{name} DD",
                    line_color=colors.get(name),
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
        fig.update_yaxes(type="log", title_text="NAV log", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown %", row=2, col=1)
        fig.update_layout(height=740, hovermode="x unified", legend={"orientation": "h"})
        return fig

    return comparison_figure, nav_from_exposure, perf_row, round_df


@app.cell
def _executive_summary(mo):
    mo.md("""
    # TQQQ Signal Research: Results Review

    This notebook is a results review, not a history of the work and not an investment recommendation.

    <div style="display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px;">
      <div style="border:1px solid #d7dde8; border-radius:8px; padding:14px;">
        <h3>Question</h3>
        <p>Which daily signals have enough train and validation evidence to justify continued
        investigation for TQQQ timing?</p>
      </div>
      <div style="border:1px solid #d7dde8; border-radius:8px; padding:14px;">
        <h3>Current Answer</h3>
        <p>A small group of mean-reversion, volatility-stress, trend-regime, credit, and leveraged-ETF
        residual signals pass the current train/validation evidence screen.</p>
      </div>
      <div style="border:1px solid #d7dde8; border-radius:8px; padding:14px;">
        <h3>Decision</h3>
        <p>Continue signal research. Do not deploy a strategy yet. The corrected v2 candidate is
        positive on validation but still weak versus simple benchmarks.</p>
      </div>
    </div>

    **Bottom line:** continue investigating the promising signals listed below; do not inspect the
    frozen 2022+ test set until the validation rules and strategy specification are frozen.
    """)
    return


@app.cell
def _data_assumptions(mo, panel, pd, research_panel, split_panel):
    close_cols = [c for c in panel.columns if c.endswith("_close")]
    coverage_rows = []
    for col in close_cols:
        sym = col.replace("_close", "")
        valid = panel[col].dropna()
        if valid.empty:
            continue
        coverage_rows.append({
            "symbol": sym,
            "first_date": valid.index.min().date(),
            "last_date": valid.index.max().date(),
            "rows": int(valid.shape[0]),
            "missing_pct": round(float(panel[col].isna().mean() * 100), 2),
        })
    coverage = pd.DataFrame(coverage_rows).sort_values("symbol").reset_index(drop=True)
    split_summary = pd.DataFrame([
        {"split": "train", "window": "<= 2017-12-31", "rows": len(split_panel(panel, "train"))},
        {"split": "validation", "window": "2018-01-01 to 2021-12-31", "rows": len(split_panel(panel, "val"))},
        {"split": "frozen test", "window": "2022-01-01 onward", "rows": "held out; not evaluated"},
    ])

    mo.vstack([
        mo.md("""
        ## Data, Splits, And Assumptions

        Data comes from local daily OHLCV SQLite databases. The research target is TQQQ long/flat
        exposure, with QQQ and cross-asset data used to form signals.

        **Execution assumption:** a signal observed after close[t] can first trade at open[t+1].
        Strategy PnL is recorded on the date the open-to-open return is realized. Costs are charged
        in basis points on position changes.
        """),
        mo.md("### Split policy"),
        split_summary,
        mo.md("### Symbol coverage"),
        coverage,
        mo.md(f"Research calculations below use only train + validation rows: `{research_panel.index.min().date()}` to `{research_panel.index.max().date()}`."),
    ])
    return


@app.cell
def _pipeline_overview(REGISTRY, mo, pd):
    registry_summary = (
        pd.DataFrame([
            {"family": m.family, "status": m.status, "metric": name}
            for name, m in REGISTRY.items()
        ])
        .groupby(["family", "status"])
        .size()
        .reset_index(name="metric_count")
        .sort_values(["family", "status"])
    )

    mo.vstack([
        mo.md("""
        ## Method: How Metrics Were Inspected

        Each metric is inspected as a causal daily signal:

        `daily data -> metric value at close[t] -> vote (-1/0/+1) -> tradable TQQQ return from open[t+1]`

        The decision screen used here:

        1. Select direction and weight on train only.
        2. Evaluate whether validation has the same directed edge.
        3. Require at least 30 bull and 30 bear validation observations.
        4. Continue investigating only signals with positive directed validation edge and train/validation sign agreement.

        **Voting metrics** can enter a strategy candidate. **Watch metrics** are research hypotheses and are
        excluded from v2 configs by default.
        """),
        registry_summary,
    ])
    return


@app.cell
def _signal_evidence(
    configs_from_decision_table,
    metric_decision_table,
    mo,
    research_panel,
    round_df,
):
    decisions = metric_decision_table(research_panel, horizon=5)
    configs = configs_from_decision_table(decisions)
    selected_names = [cfg.metric for cfg in configs]

    evidence_cols = [
        "metric",
        "family",
        "status",
        "decision",
        "direction",
        "weight",
        "tqqq_edge_bps_train",
        "tqqq_edge_bps_val",
        "min_directional_obs_val",
        "edge_sign_agrees",
        "selection_basis",
    ]
    evidence = decisions[evidence_cols].copy()
    evidence["directed_train_edge_bps"] = evidence["direction"] * evidence["tqqq_edge_bps_train"]
    evidence["directed_val_edge_bps"] = evidence["direction"] * evidence["tqqq_edge_bps_val"]
    evidence["investigation_decision"] = "do_not_prioritize"
    _promising_mask = (
        evidence["decision"].isin(["keep", "invert"])
        & evidence["edge_sign_agrees"]
        & (evidence["directed_val_edge_bps"] > 0)
        & (evidence["min_directional_obs_val"] >= 30)
    )
    _selected_but_failed_mask = evidence["decision"].isin(["keep", "invert"]) & ~_promising_mask
    evidence.loc[_selected_but_failed_mask, "investigation_decision"] = "selected_train_only_but_validation_failed"
    evidence.loc[_promising_mask, "investigation_decision"] = "continue_investigating"

    signal_review_cols = [
        "metric",
        "family",
        "decision",
        "investigation_decision",
        "directed_train_edge_bps",
        "directed_val_edge_bps",
        "min_directional_obs_val",
        "edge_sign_agrees",
        "weight",
    ]
    signal_review = evidence[signal_review_cols].copy()
    signal_review["decision_rank"] = signal_review["investigation_decision"].map({
        "continue_investigating": 0,
        "selected_train_only_but_validation_failed": 1,
        "do_not_prioritize": 2,
    })
    signal_review = (
        signal_review
        .sort_values(["decision_rank", "directed_val_edge_bps"], ascending=[True, False])
        .drop(columns=["decision_rank"])
        .reset_index(drop=True)
    )
    promising_signals = signal_review.loc[
        signal_review["investigation_decision"] == "continue_investigating"
    ].copy()

    mo.vstack([
        mo.md("""
        ## Signal Evidence And Investigation Decision

        This is the main results table. It answers: after inspecting each metric, should we continue
        investigating it?

        **Decision rule:** continue investigating if the signal was selected on train, validation has
        positive directed edge, validation has enough bull/bear observations, and train/validation
        edge signs agree.

        - **Edge bps:** average 5-day TQQQ tradable-open return after bull votes minus bear votes.
        - **Directed edge bps:** edge after applying the train-selected keep/invert direction.
        - **Min directional obs:** the smaller of bull/bear vote counts; low values are weak evidence.
        - **continue_investigating:** promising enough for deeper diagnostics.
        """),
        round_df(signal_review, 3),
        mo.md("### Promising signals only"),
        round_df(promising_signals, 3),
    ])
    return configs, decisions, promising_signals


@app.cell
def _rsi2_spotlight(decisions, mo, round_df):
    _rsi = decisions.loc[decisions["metric"] == "qqq_rsi2"].copy()
    if _rsi.empty:
        _content = mo.md("""
        ## Signal Spotlight: RSI2

        RSI2 is not present in the current corrected decision table. That would be unexpected and
        should be investigated before presenting the notebook.
        """)
    else:
        _rsi["directed_val_edge_bps"] = _rsi["direction"] * _rsi["tqqq_edge_bps_val"]
        _rsi["why_it_matters"] = (
            "High validation sample count, stable train/validation sign, simple finance intuition: "
            "short-term QQQ oversold conditions can precede TQQQ rebounds."
        )
        _view = _rsi[[
            "metric",
            "family",
            "decision",
            "weight",
            "tqqq_edge_bps_train",
            "tqqq_edge_bps_val",
            "directed_val_edge_bps",
            "min_directional_obs_val",
            "edge_sign_agrees",
            "why_it_matters",
        ]]
        _content = mo.vstack([
            mo.md("""
            ## Signal Spotlight: RSI2

            RSI2 is still one of the best **clean, explainable** signals in the current research set.
            It is not always the top row by raw edge, because signals like Bollinger z-score and
            Williams VIX Fix show larger train/validation edge. But RSI2 has a strong practical case:
            it is simple, intuitive, selected train-only, and has many more validation observations
            than some higher-edge signals.

            So the accurate wording is: **RSI2 is one of our best and most defensible signals, not
            necessarily the single highest-edge signal after the methodology correction.**
            """),
            round_df(_view, 3),
        ])
    _content
    return


@app.cell
def _readiness_costs(
    apply_cost_model,
    decisions,
    mo,
    pd,
    promising_signals,
    research_panel,
    round_df,
    vote_dynamics_report,
):
    finalist_names = promising_signals["metric"].tolist()
    dynamics = vote_dynamics_report(
        research_panel,
        split="train+val",
        metric_names=finalist_names,
        include_watch=True,
    )
    dyn_idx = dynamics.set_index("metric") if not dynamics.empty else pd.DataFrame()
    dec_idx = decisions.set_index("metric")
    readiness_rows = []
    for _name in finalist_names:
        if _name not in dyn_idx.index or _name not in dec_idx.index:
            continue
        dyn = dyn_idx.loc[_name]
        dec = dec_idx.loc[_name]
        directed_val_edge = float(dec["direction"]) * float(dec["tqqq_edge_bps_val"])
        cost = apply_cost_model(
            gross_edge_bps_per_trade=directed_val_edge,
            flips_per_year=float(dyn["flips_per_year"]),
            long_fraction=float(dyn["avg_long_fraction"]),
            one_way_spread_bps=2.0,
            one_way_commission_bps=0.0,
            annual_expense_bps=86.0,
        )
        readiness_rows.append({
            "metric": _name,
            "directed_val_edge_bps": directed_val_edge,
            "flips_per_year": dyn["flips_per_year"],
            "mean_run_length": dyn["mean_run_length"],
            "net_annual_bps": cost["net_annual_bps"],
            "breakeven_edge_bps_per_trade": cost["breakeven_edge_bps_per_trade"],
            "passes_cost_filter": cost["net_annual_bps"] > 0,
        })
    cost_readiness = pd.DataFrame(readiness_rows)

    mo.vstack([
        mo.md("""
        ## Readiness Diagnostics: Costs

        This stress test asks whether promising signals still have enough validation edge after
        basic friction assumptions. `directed_val_edge_bps` applies the train-selected keep/invert
        direction to validation edge. `passes_cost_filter=False` means the validation edge is not
        large enough at a 2 bps one-way spread and 86 bps annual TQQQ expense.
        """),
        round_df(cost_readiness, 2),
    ])
    return


@app.cell
def _readiness_regime_decay(
    OUTPUT_DIR,
    mo,
    pd,
    promising_signals,
    regime_conditional_edge_table,
    regime_conditional_heatmap,
    research_panel,
    rolling_edge_decay,
    rolling_edge_decay_small_multiples,
    warnings,
):
    top_for_plots = promising_signals["metric"].head(5).tolist()
    states_path = OUTPUT_DIR / "regime_states.csv"
    if states_path.exists():
        states = pd.read_csv(states_path, index_col=0, parse_dates=True).iloc[:, 0]
        states = states.reindex(research_panel.index)
    else:
        from regime import walk_forward_states
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            states, _ = walk_forward_states(research_panel)

    regime_reports = {}
    decay_reports = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _name in top_for_plots:
            regime_reports[_name] = regime_conditional_edge_table(
                research_panel,
                _name,
                states,
                split="train+val",
                horizon=5,
            )
            decay_reports[_name] = rolling_edge_decay(
                research_panel,
                _name,
                split="train+val",
                window=504,
                step=21,
                horizon=5,
            )

    regime_fig = regime_conditional_heatmap(regime_reports, top_metric_order=top_for_plots)
    decay_fig = rolling_edge_decay_small_multiples(decay_reports)

    mo.vstack([
        mo.md("""
        ## Readiness Diagnostics: Regimes And Decay

        The regime heatmap asks whether a signal only works in one market state. The rolling-edge
        chart asks whether the signal has decayed over time. These are research diagnostics, not a
        production rule yet.
        """),
        regime_fig,
        decay_fig,
    ])
    return


@app.cell
def _strategy_experiments(
    OUTPUT_DIR,
    REGISTRY,
    comparison_figure,
    configs,
    mo,
    nav_from_exposure,
    pd,
    perf_row,
    research_panel,
    round_df,
    run_v2_backtest,
    select_threshold_on_train,
    split_panel,
):
    train_panel = split_panel(research_panel, "train")
    val_panel = split_panel(research_panel, "val")
    selected_rule = select_threshold_on_train(train_panel, configs)
    v2_result = run_v2_backtest(
        research_panel,
        configs,
        mode=str(selected_rule["mode"]),
        threshold=float(selected_rule["threshold"]),
        medium_threshold=float(selected_rule["medium_threshold"]),
        split="val",
        evaluation_index=val_panel.index,
    )

    v0_metrics = ["qqq_sma50_200_regime", "qqq_rsi2", "qqq_yz_vol_20d"]
    votes = pd.DataFrame({
        name: REGISTRY[name].vote(REGISTRY[name].compute(research_panel))
        for name in v0_metrics
    }, index=research_panel.index)
    v0_signal = (votes.mean(axis=1) > 0).astype(float)
    v0_exposure = v0_signal.shift(1).fillna(0.0).reindex(val_panel.index)
    v0_equity = nav_from_exposure(val_panel, v0_exposure, cost_bps=5.0).rename("v0_simple")

    v1_path = OUTPUT_DIR / "strategy_val.csv"
    if v1_path.exists():
        v1_df = pd.read_csv(v1_path, parse_dates=["date"], index_col="date")
        v1_equity = v1_df["equity"].rename("v1_saved")
        v1_position = v1_df["position"] if "position" in v1_df else None
    else:
        v1_equity = pd.Series(dtype=float, name="v1_saved")
        v1_position = None

    series_by_name = {
        "corrected_v2": v2_result.equity,
        "v0_simple": v0_equity,
        "qqq_buy_hold": v2_result.benchmark_qqq,
        "tqqq_buy_hold": v2_result.benchmark_tqqq,
        "tqqq_fixed_25": v2_result.fixed_25_tqqq,
        "tqqq_fixed_50": v2_result.fixed_50_tqqq,
    }
    if not v1_equity.empty:
        series_by_name = {"v1_saved": v1_equity, **series_by_name}

    rows = [
        perf_row("corrected_v2", v2_result.equity, exposure=v2_result.exposure),
        perf_row("v0_simple", v0_equity, exposure=v0_exposure),
        perf_row("qqq_buy_hold", v2_result.benchmark_qqq),
        perf_row("tqqq_buy_hold", v2_result.benchmark_tqqq),
        perf_row("tqqq_fixed_25", v2_result.fixed_25_tqqq),
        perf_row("tqqq_fixed_50", v2_result.fixed_50_tqqq),
    ]
    if not v1_equity.empty:
        rows.insert(0, perf_row("v1_saved", v1_equity, exposure=v1_position))
    comparison = pd.DataFrame(rows)
    comparison_fig = comparison_figure(series_by_name)
    _strategy_meta = {
        "v1_saved": {
            "strategy_type": "legacy ensemble",
            "signal_source": "v1 vote ensemble + HSMM regime gating",
            "selection_basis": "legacy saved output",
            "exposure_rule": "binary long/flat",
            "costs_included": "yes",
            "test_status": "2022+ not shown",
            "readout": "Useful baseline, but not the cleaned methodology.",
        },
        "corrected_v2": {
            "strategy_type": "corrected research candidate",
            "signal_source": "train-selected weighted metric votes",
            "selection_basis": "train only",
            "exposure_rule": f"{selected_rule['mode']} threshold={selected_rule['threshold']}",
            "costs_included": "yes",
            "test_status": "2022+ not shown",
            "readout": "Main candidate, still not investment-ready.",
        },
        "v0_simple": {
            "strategy_type": "simple proof of concept",
            "signal_source": "3-metric majority vote",
            "selection_basis": "hand-built baseline",
            "exposure_rule": "binary long/flat",
            "costs_included": "yes",
            "test_status": "2022+ not shown",
            "readout": "Sanity check, not the final approach.",
        },
        "qqq_buy_hold": {
            "strategy_type": "benchmark",
            "signal_source": "none",
            "selection_basis": "market benchmark",
            "exposure_rule": "100% QQQ",
            "costs_included": "no trading costs",
            "test_status": "2022+ not shown",
            "readout": "Lower leverage comparison baseline.",
        },
        "tqqq_buy_hold": {
            "strategy_type": "benchmark",
            "signal_source": "none",
            "selection_basis": "market benchmark",
            "exposure_rule": "100% TQQQ",
            "costs_included": "no trading costs",
            "test_status": "2022+ not shown",
            "readout": "High-risk leverage baseline.",
        },
        "tqqq_fixed_25": {
            "strategy_type": "benchmark",
            "signal_source": "none",
            "selection_basis": "fixed exposure benchmark",
            "exposure_rule": "25% TQQQ",
            "costs_included": "no trading costs",
            "test_status": "2022+ not shown",
            "readout": "Risk-scaled leverage benchmark.",
        },
        "tqqq_fixed_50": {
            "strategy_type": "benchmark",
            "signal_source": "none",
            "selection_basis": "fixed exposure benchmark",
            "exposure_rule": "50% TQQQ",
            "costs_included": "no trading costs",
            "test_status": "2022+ not shown",
            "readout": "Risk-scaled leverage benchmark.",
        },
    }
    strategy_aspects = (
        pd.DataFrame([
            {"name": row["name"], **_strategy_meta.get(row["name"], {})}
            for row in rows
        ])
        .merge(
            comparison[[
                "name",
                "cagr",
                "sharpe",
                "maxdd_pct",
                "maxdd_duration_days",
                "exposure_pct",
                "trade_count",
            ]],
            on="name",
            how="left",
        )
    )

    mo.vstack([
        mo.md(f"""
        ## Strategy Experiments

        Corrected v2 uses train-only metric selection and selected this rule on train only:
        `{selected_rule}`.

        Validation is still not the final test. The 2022+ frozen test set remains untouched.
        """),
        mo.md("### Strategy summary table"),
        round_df(strategy_aspects, 4),
        mo.md("### Performance table"),
        round_df(comparison, 4),
        comparison_fig,
    ])
    return


@app.cell
def _review_findings(mo, pd):
    findings = pd.DataFrame([
        {
            "finding": "Validation used for selection",
            "status": "fixed in v2 path",
            "notebook_treatment": "Decision/direction/weight are train-only; validation is context.",
            "residual_risk": "Need final go/no-go thresholds before test.",
        },
        {
            "finding": "Validation cold-started rolling metrics",
            "status": "fixed in diagnostics",
            "notebook_treatment": "Metrics are computed on full causal train+val history, then sliced.",
            "residual_risk": "Assumes each metric implementation remains causal.",
        },
        {
            "finding": "V2 PnL date labeling",
            "status": "fixed in v2 backtest",
            "notebook_treatment": "Open-to-open PnL is recorded on realization date.",
            "residual_risk": "V1 saved outputs remain legacy comparison artifacts.",
        },
        {
            "finding": "Stale generated v2 artifacts",
            "status": "fixed after rerun",
            "notebook_treatment": "Notebook computes corrected v2 live and generated outputs are regenerated separately.",
            "residual_risk": "Regenerate outputs after future methodology changes.",
        },
    ])

    mo.vstack([
        mo.md("""
        ## Review Findings And Next Steps

        The review improved the research discipline. The current stance is intentionally conservative:
        keep building, but do not deploy or inspect the frozen test set yet.
        """),
        findings,
        mo.md("""
        **Next steps**

        1. Finalize train-only selection and validation acceptance thresholds.
        2. Decide the benchmark that matters most: QQQ, TQQQ, fixed-exposure TQQQ, or drawdown-controlled return.
        3. Freeze the strategy spec.
        4. Evaluate the 2022+ test set exactly once.
        5. If the test passes, write the production rules and monitoring checklist.
        """),
    ])
    return


if __name__ == "__main__":
    app.run()
