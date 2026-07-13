"""Pre-strategy diagnostic battery — stress-testing leaderboard finalists.

Run with: marimo edit notebooks/07_strategy_readiness.py
"""

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="wide")


@app.cell
def _imports():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

    import warnings
    import numpy as np
    import pandas as pd
    import marimo as mo

    from data import SYMBOLS_ALL, load_panel
    from metrics import REGISTRY
    from regime import walk_forward_states
    from signal_diagnostics import (
        apply_cost_model,
        multi_horizon_credibility_report,
        regime_conditional_edge_table,
        rolling_edge_decay,
        vote_dynamics_report,
    )
    from viz import (
        regime_conditional_heatmap,
        rolling_edge_decay_small_multiples,
    )

    from quantcore import config as _qc_config
    DATA_DIR = _qc_config.data_dir()
    return (
        DATA_DIR,
        REGISTRY,
        SYMBOLS_ALL,
        apply_cost_model,
        load_panel,
        mo,
        multi_horizon_credibility_report,
        np,
        pd,
        regime_conditional_edge_table,
        regime_conditional_heatmap,
        rolling_edge_decay,
        rolling_edge_decay_small_multiples,
        vote_dynamics_report,
        walk_forward_states,
        warnings,
    )


@app.cell
def _load(DATA_DIR, SYMBOLS_ALL, load_panel, warnings):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        panel = load_panel(SYMBOLS_ALL, data_dir=DATA_DIR, warn_missing=True)
    print(f"Panel: {len(panel)} rows, {panel.index[0].date()} -> {panel.index[-1].date()}")
    return (panel,)


@app.cell
def _intro(mo):
    mo.md("""
    # Strategy Readiness Diagnostics

    This notebook stress-tests the leaderboard finalists before strategy construction.
    It answers: do these signals survive transaction costs, hold across regimes, and stay stable in time?
    The test set (2022+) is not loaded.

    **Cell order:** Global Filters → Leaderboard Finalists → Vote Dynamics → Cost Model → Net Edge Table
    → Regime Fit → Regime Heatmap → Rolling Edge Decay → Threshold Audit → Summary.
    """)
    return


@app.cell
def _controls(REGISTRY, mo):
    _all_families = sorted({m.family for m in REGISTRY.values()})
    _default_families = [f for f in _all_families if not f.startswith("physics_")]

    min_obs_ui = mo.ui.slider(
        start=10, stop=200, step=10, value=30,
        label="Min directional obs (val)",
    )
    family_ui = mo.ui.multiselect(
        options=_all_families,
        value=_default_families,
        label="Families",
    )
    include_watch_ui = mo.ui.checkbox(value=False, label="Include watch metrics")
    top_n_for_diagnostics_ui = mo.ui.slider(
        start=3, stop=20, step=1, value=8,
        label="Top N for deep-dive diagnostics",
    )

    mo.vstack([
        mo.md("### Global Filters"),
        mo.hstack([min_obs_ui, include_watch_ui, top_n_for_diagnostics_ui]),
        family_ui,
    ])
    return family_ui, include_watch_ui, min_obs_ui, top_n_for_diagnostics_ui


@app.cell
def _leaderboard_finalists(
    family_ui,
    include_watch_ui,
    min_obs_ui,
    mo,
    multi_horizon_credibility_report,
    panel,
    top_n_for_diagnostics_ui,
    warnings,
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _full_report = multi_horizon_credibility_report(
            panel,
            horizons=(1, 2, 5, 10, 20),
            target_symbol="TQQQ",
            target_kind="tradable_open",
            include_watch=include_watch_ui.value,
            min_directional_obs=int(min_obs_ui.value),
        )

    finalists = (
        _full_report[
            _full_report["family"].isin(family_ui.value)
            & _full_report["passes_min_obs"]
        ]
        .head(int(top_n_for_diagnostics_ui.value))
        .reset_index(drop=True)
    )
    finalist_names = finalists["metric"].tolist()

    _view = finalists[["metric", "family", "edge_val_5d", "raw_ic_q_val_5d", "n_horizons_edge_sign_agree"]].copy()
    for _col in _view.select_dtypes(include=["float"]).columns:
        _view[_col] = _view[_col].round(4)

    mo.vstack([
        mo.md("## Leaderboard Finalists"),
        mo.md(
            "Top-N metrics by credibility score that pass the min-obs filter. "
            "These are the metrics subject to the four diagnostics below."
        ),
        _view,
    ])
    return finalist_names, finalists


@app.cell
def _vote_dynamics(finalist_names, mo, panel, vote_dynamics_report, warnings):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dynamics = vote_dynamics_report(
            panel,
            split="train+val",
            metric_names=finalist_names,
            include_watch=True,
        )

    _view = dynamics.copy()
    for _col in _view.select_dtypes(include=["float"]).columns:
        _view[_col] = _view[_col].round(3)

    mo.vstack([
        mo.md("## Vote Dynamics"),
        mo.md(
            "`flips_per_year` × `~2 bps` ≈ annual friction cost (see next cell). "
            "High `vote_autocorr` means the vote is sticky — the metric is making fewer but longer trades. "
            "**Columns:** `flips_per_year` = how often the vote changes direction per year; "
            "`mean_run_length` = average consecutive same-vote days; "
            "`vote_autocorr_1d/5d` = how predictable the vote is from recent history."
        ),
        _view,
    ])
    return (dynamics,)


@app.cell
def _cost_model_inputs(mo):
    spread_bps_ui = mo.ui.slider(
        start=0.5, stop=10.0, step=0.5, value=2.0,
        label="One-way spread (bps)",
    )
    commission_bps_ui = mo.ui.slider(
        start=0.0, stop=5.0, step=0.5, value=0.0,
        label="One-way commission (bps)",
    )
    expense_bps_ui = mo.ui.slider(
        start=0, stop=200, step=5, value=86,
        label="Annual ETF expense (bps)",
    )

    mo.vstack([
        mo.md("## Cost Model Assumptions"),
        mo.md(
            "- **Spread:** half the bid-ask spread paid on each side of a round-trip trade. "
            "TQQQ is highly liquid; 2 bps is a conservative estimate.\n"
            "- **Commission:** broker commission per leg. Zero for most modern retail accounts.\n"
            "- **ETF expense:** TQQQ expense ratio (86 bps/year). Charged proportional to time held long."
        ),
        mo.hstack([spread_bps_ui, commission_bps_ui, expense_bps_ui]),
    ])
    return commission_bps_ui, expense_bps_ui, spread_bps_ui


@app.cell
def _net_edge_table(
    apply_cost_model,
    commission_bps_ui,
    dynamics,
    expense_bps_ui,
    finalist_names,
    finalists,
    mo,
    pd,
    spread_bps_ui,
):
    _rows = []
    _dyn_idx = dynamics.set_index("metric")
    _fin_idx = finalists.set_index("metric")

    for _name in finalist_names:
        if _name not in _dyn_idx.index or _name not in _fin_idx.index:
            continue
        _dyn_row = _dyn_idx.loc[_name]
        _gross_edge = float(_fin_idx.loc[_name, "edge_val_5d"])
        _flips = float(_dyn_row["flips_per_year"])
        _long_frac = float(_dyn_row["avg_long_fraction"])

        _result = apply_cost_model(
            gross_edge_bps_per_trade=_gross_edge,
            flips_per_year=_flips,
            long_fraction=_long_frac,
            one_way_spread_bps=float(spread_bps_ui.value),
            one_way_commission_bps=float(commission_bps_ui.value),
            annual_expense_bps=float(expense_bps_ui.value),
        )
        _rows.append({
            "metric": _name,
            "gross_edge_bps_per_trade": round(_gross_edge, 1),
            "trades_per_year": round(_result["trades_per_year"], 1),
            "gross_annual_bps": round(_result["gross_annual_bps"], 1),
            "friction_annual_bps": round(_result["friction_annual_bps"], 1),
            "etf_drag_annual_bps": round(_result["etf_drag_annual_bps"], 1),
            "net_annual_bps": round(_result["net_annual_bps"], 1),
            "breakeven_edge_bps_per_trade": round(_result["breakeven_edge_bps_per_trade"], 1),
            "passes_cost_filter": _result["net_annual_bps"] > 0,
        })

    net_df = pd.DataFrame(_rows)

    mo.vstack([
        mo.md("## Net Edge After Costs"),
        mo.md(
            "A signal with negative `net_annual_bps` is gross-positive but cost-negative — "
            "i.e. real but not tradable at these friction assumptions. "
            "Adjust the sliders above to explore break-even conditions.\n\n"
            "`gross_edge_bps_per_trade` is the val-set bull-minus-bear edge at 5d horizon. "
            "`trades_per_year ≈ flips_per_year / 2` (each round trip = two flips)."
        ),
        net_df,
    ])
    return (net_df,)


@app.cell
def _regime_fit(mo, panel, walk_forward_states, warnings):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        states, regime_proba = walk_forward_states(panel)

    _n_warmup = int((states == -1).sum())
    _n_changes = int((states != states.shift(1)).sum())
    print(
        f"Regime model fit. {_n_warmup} warmup rows excluded. "
        f"{_n_changes} state transitions."
    )

    mo.md(
        f"**Regime model fit.** {_n_warmup} warmup rows excluded (state = −1). "
        f"{_n_changes} state transitions detected. States: 0=strong_bull, 1=weak_bull, "
        f"2=sideways, 3=weak_bear, 4=strong_bear."
    )
    return (states,)


@app.cell
def _regime_conditional(
    finalist_names,
    mo,
    panel,
    regime_conditional_edge_table,
    regime_conditional_heatmap,
    states,
    warnings,
):
    regime_reports = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _name in finalist_names:
            regime_reports[_name] = regime_conditional_edge_table(
                panel,
                _name,
                states,
                split="train+val",
                horizon=5,
            )

    _fig = regime_conditional_heatmap(regime_reports, top_metric_order=finalist_names)

    mo.vstack([
        mo.md("## Regime-Conditional Edge"),
        mo.md(
            "Edge per HSMM regime state. **Red columns** = state in which the signal works. "
            "A metric that is red everywhere is regime-robust. "
            "A metric that is strong only in `strong_bear` is a crash detector, not a general signal."
        ),
        _fig,
    ])
    return (regime_reports,)


@app.cell
def _edge_decay(
    finalist_names,
    mo,
    panel,
    rolling_edge_decay,
    rolling_edge_decay_small_multiples,
    warnings,
):
    decay_reports = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _name in finalist_names:
            decay_reports[_name] = rolling_edge_decay(
                panel,
                _name,
                split="train+val",
                window=504,
                step=21,
                horizon=5,
            )

    _fig = rolling_edge_decay_small_multiples(decay_reports)

    mo.vstack([
        mo.md("## Rolling Edge Decay"),
        mo.md(
            "Rolling 2-year edge through time. A flat trace means stationary signal; "
            "a downward slope means decay. Compare the most recent values to the leaderboard's "
            "val edge — if the trace has fallen below it, the val-set number overstates current edge. "
            "The dotted line is an OLS trend fit."
        ),
        _fig,
    ])
    return (decay_reports,)


@app.cell
def _threshold_audit(mo):
    mo.md("""
    ## Threshold Audit

    Each finalist's vote thresholds must have been set without val data for the leaderboard numbers
    to be unbiased. The table below documents the source of each finalist's thresholds.

    | Metric | Threshold source | Category | Leakage-free? |
    |---|---|---|---|
    | qqq_rsi2 | Fixed: oversold < 10, overbought > 90 | fixed | ✓ |
    | qqq_bb_z20 | Fixed: z < −1.0 bear, z > +1.0 bull | fixed | ✓ |
    | qqq_rv_20d | Rolling 252-day percentile (causal, window ends at t) | rolling_percentile | ✓ |
    | qqq_williams_vix_fix | Fixed: WVF > 0 bull (volatility spike = opportunity) | fixed | ✓ |
    | qqq_sma50_200_regime | Fixed: SMA50 < SMA200 = bearish | fixed | ✓ |
    | qqq_hv_rv_ratio | Rolling 126-day percentile | rolling_percentile | ✓ |
    | qqq_sign_entropy_20d | Rolling 126-day percentile | rolling_percentile | ✓ |
    | qqq_mom_12_1 | Fixed: positive = bull, negative = bear | fixed | ✓ |

    **Categories:**
    - `fixed` — threshold is a constant, immune to any data leakage.
    - `rolling_percentile` — threshold is computed causally on a rolling window ending at *t*, immune to leakage.
    - `tuned_on_train` — threshold was selected by inspecting train performance (immune if train-only).

    If a finalist not listed above appears, add it here before proceeding to strategy construction.
    """)
    return


@app.cell
def _summary(decay_reports, mo, net_df, np, pd, regime_reports):
    _cost_pass = net_df[["metric", "net_annual_bps", "passes_cost_filter"]].copy()

    _regime_rows = []
    for _name, _df in regime_reports.items():
        _signs = _df["edge_bps"].dropna().apply(lambda x: 1 if x > 0 else -1)
        _n_pos = int((_signs > 0).sum())
        _n_neg = int((_signs < 0).sum())
        _regime_robust = max(_n_pos, _n_neg) >= 4
        _regime_rows.append({"metric": _name, "regime_robust (>=4/5 same-sign)": _regime_robust})
    _regime_summary = pd.DataFrame(_regime_rows)

    _decay_rows = []
    for _name, _df in decay_reports.items():
        if _df.empty or _df["edge_bps"].isna().all():
            _decay_rows.append({"metric": _name, "decay_slope_bps_per_window": np.nan, "decay_flagged": True})
            continue
        _valid = _df.dropna(subset=["edge_bps"])
        _t = np.arange(len(_valid))
        _coeffs = np.polyfit(_t, _valid["edge_bps"].values, 1)
        _slope = float(_coeffs[0])
        _decay_rows.append({"metric": _name, "decay_slope_bps_per_window": round(_slope, 2), "decay_flagged": _slope < -5})
    _decay_summary = pd.DataFrame(_decay_rows)

    _combined = (
        _cost_pass
        .merge(_regime_summary, on="metric", how="outer")
        .merge(_decay_summary, on="metric", how="outer")
    )

    mo.vstack([
        mo.md("## Summary"),
        mo.md(
            "Pass/fail summary across the three diagnostic dimensions. "
            "`passes_cost_filter` = net_annual_bps > 0 at current friction assumptions. "
            "`regime_robust` = >=4 of 5 HSMM states have the same-sign edge. "
            "`decay_flagged` = OLS slope on rolling edge < -5 bps per 2-year window.\n\n"
            "**Next decisions, blocked until these are reviewed:**\n"
            "- Aggregation rule (majority / weighted / AND-logic)\n"
            "- Position sizing (binary / scaled / vol-target)\n"
            "- bb_z20 vs williams_vix_fix redundancy resolution (use net edge comparison)\n"
            "- Selection-bias correction (permutation test on the metric-survival pipeline)\n"
            "- Benchmark / good-enough definition (CAGR / Sharpe / DD targets vs TQQQ-BaH)\n"
            "- Backtest engine / MASTER_LOG writes (after strategy spec is frozen)"
        ),
        _combined,
    ])
    return


if __name__ == "__main__":
    app.run()
