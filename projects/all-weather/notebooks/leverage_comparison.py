import marimo

__generated_with = "0.23.5"
app = marimo.App(width="full")

with app.setup:
    import json
    import sys
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd

    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from engine.leverage import selected_window_metrics
    from research._shared.leverage_analysis import (
        BASE,
        SELECTOR_LABELS,
        broker_filter_impact,
        broker_profile_options,
        build_candidate_funnel,
        build_is_vs_oos_comparison,
        compute_heatmap_pivot,
        default_focus_strategies,
        derive_leverage_tables,
        filter_broker_limits,
        is_control_candidate,
        label_selector,
        latest_bundle,
        latest_bundle_for_profile,
        maybe_filter_benchmark,
        max_allowed_leverage_pct,
        portfolio_view_strategies,
        presentation_table,
        scale_overlay_leverage,
        slice_dates,
        strategy_label,
        visible_strategies,
    )
    from research._shared.leverage_plotting import (
        colour_map,
        plot_all_etf_rsi_figure,
        plot_calmar_maxdd_scatter_figure,
        plot_default_overlay_figure,
        plot_etf_oos_bars_figure,
        plot_exposure_timeline_figure,
        plot_grid_heatmap,
        plot_grid_heatmaps_figure,
        plot_growth_and_drawdown_figure,
        plot_mixed_growth_figure,
        plot_mixed_is_vs_oos_figure,
        plot_mixed_oos_delta_bars_figure,
        plot_oos_validation_figure,
        plot_threshold_heatmap_figure,
        plot_trade_episode_distribution_figure,
        plot_walk_forward_heatmap_figure,
        plot_yearly_figure,
    )

    _RSI_RESULTS = ROOT / "research" / "rsi_leverage_overlay" / "results"
    BUNDLE_ROOT = _RSI_RESULTS / "leverage_comparison"
    OOS_BUNDLE_ROOT = _RSI_RESULTS / "leverage_oos_validation"
    MIXED_OOS_BUNDLE_ROOT = _RSI_RESULTS / "mixed_leverage_oos_validation"
    MIXED_SWEEP_BUNDLE_ROOT = _RSI_RESULTS / "mixed_leverage_sweep"
    MIXED_FULL_GRID_BUNDLE_ROOT = _RSI_RESULTS / "mixed_leverage_full_grid_oos"

    # These constants must remain here for the presentation-only regression test.
    BENCHMARK = "S&P 500 (SPY)"
    PLOT_EXCLUDED = {BENCHMARK}
    BROKER_LIMIT_PROFILE_LABELS = ["IBKR Safe", "Strict Pilot", "Research Unrestricted"]


@app.cell
def choose_bundles():
    bundle_path = mo.ui.text(
        value=latest_bundle(BUNDLE_ROOT),
        label="Main leverage result bundle",
        full_width=True,
    )
    oos_bundle_path = mo.ui.text(
        value=latest_bundle(OOS_BUNDLE_ROOT),
        label="OOS validation bundle",
        full_width=True,
    )
    mo.vstack([
        mo.md(
            """
            # ETF Leverage Overlay Research

            This notebook is a presentation view of the leverage research. It reads
            exported CSV artifacts only; no prices are fetched and no backtests are
            recomputed here.
            """
        ),
        bundle_path,
        oos_bundle_path,
    ])
    return bundle_path, oos_bundle_path


@app.cell
def load_bundle(bundle_path):
    from pathlib import Path as _Path

    _raw_bundle_path = bundle_path.value.strip().strip("'\"")
    if not _raw_bundle_path:
        mo.stop(True, mo.md("Generate a leverage comparison bundle first."))

    bundle = _Path(_raw_bundle_path).expanduser()
    if not bundle.is_absolute():
        bundle = ROOT / bundle
    _manifest_path = bundle / "manifest.json"
    if not bundle.exists() or not _manifest_path.exists():
        mo.stop(True, mo.md(f"Could not find a valid result bundle at `{bundle}`."))

    with open(_manifest_path, "r", encoding="utf-8") as _handle:
        manifest = json.load(_handle)

    daily = pd.read_csv(bundle / "daily_series.csv", parse_dates=["Date"])
    summary = pd.read_csv(bundle / "summary_metrics.csv")
    calendar = pd.read_csv(bundle / "calendar_year_metrics.csv")
    rolling = pd.read_csv(bundle / "rolling_metrics.csv", parse_dates=["Date"])
    dd_events = pd.read_csv(bundle / "drawdown_events.csv")
    stress = pd.read_csv(bundle / "stress_period_metrics.csv")
    signals = pd.read_csv(bundle / "signal_history.csv", parse_dates=["Date"])
    diagnostics = pd.read_csv(bundle / "overlay_diagnostics.csv", parse_dates=["Date"])
    threshold_grid = pd.read_csv(bundle / "threshold_grid.csv")
    capital = pd.read_csv(bundle / "capital_view.csv")
    overlay_summary = pd.read_csv(bundle / "overlay_summary.csv")
    leverage_summary = pd.read_csv(bundle / "leverage_summary.csv")
    yearly_overlay = pd.read_csv(bundle / "yearly_overlay_metrics.csv")
    return (
        calendar,
        capital,
        daily,
        dd_events,
        diagnostics,
        manifest,
        overlay_summary,
        rolling,
        signals,
        stress,
        summary,
        threshold_grid,
        yearly_overlay,
    )


@app.cell
def load_oos_bundle(oos_bundle_path):
    from pathlib import Path as _Path

    _raw = oos_bundle_path.value.strip().strip("'\"")
    _empty = (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    if not _raw:
        is_grid, oos_summary, pass_fail, selected_rules, trade_episodes = _empty
    else:
        oos_bundle = _Path(_raw).expanduser()
        if not oos_bundle.is_absolute():
            oos_bundle = ROOT / oos_bundle
        if not oos_bundle.exists() or not (oos_bundle / "manifest.json").exists():
            is_grid, oos_summary, pass_fail, selected_rules, trade_episodes = _empty
        else:
            oos_summary = pd.read_csv(oos_bundle / "oos_summary.csv")
            pass_fail = pd.read_csv(oos_bundle / "pass_fail_summary.csv")
            selected_rules = pd.read_csv(oos_bundle / "selected_rules.csv")
            trade_episodes = pd.read_csv(oos_bundle / "oos_trade_episodes.csv")
            is_grid = pd.read_csv(oos_bundle / "is_threshold_grid.csv")

    for _df in [oos_summary, pass_fail, selected_rules, trade_episodes]:
        if not _df.empty and "Selector Label" not in _df:
            _df["Selector Label"] = _df["Selector"].map(label_selector)
    return oos_summary, pass_fail, selected_rules, trade_episodes


@app.cell
def broker_limit_profile_control():
    broker_profile = mo.ui.dropdown(
        options=BROKER_LIMIT_PROFILE_LABELS,
        value="IBKR Safe",
        label="Broker limit profile",
    )
    return (broker_profile,)


@app.cell
def choose_full_grid_bundle():
    full_grid_bundle_path = mo.ui.text(
        value=latest_bundle(MIXED_FULL_GRID_BUNDLE_ROOT),
        label="Full-grid SPY+GLD OOS + walk-forward bundle",
        full_width=True,
    )
    return (full_grid_bundle_path,)


@app.cell
def load_full_grid_bundle(full_grid_bundle_path):
    from pathlib import Path as _Path

    _raw = full_grid_bundle_path.value.strip().strip("'\"")
    full_grid_manifest = {}
    full_grid_leaderboard = pd.DataFrame()
    full_grid_annual_leaderboard = pd.DataFrame()
    full_grid_summary = pd.DataFrame()
    if _raw:
        _bundle = _Path(_raw).expanduser()
        if not _bundle.is_absolute():
            _bundle = ROOT / _bundle
        _manifest_path = _bundle / "manifest.json"
        if _manifest_path.exists():
            with open(_manifest_path, "r", encoding="utf-8") as _handle:
                full_grid_manifest = json.load(_handle)

            def _read_csv(name):
                _path = _bundle / name
                return pd.read_csv(_path) if _path.exists() else pd.DataFrame()

            full_grid_leaderboard = _read_csv("structural_full_grid_leaderboard.csv")
            full_grid_annual_leaderboard = _read_csv("annual_full_grid_leaderboard.csv")
            full_grid_summary = _read_csv("structural_full_grid_summary.csv")
    return full_grid_leaderboard, full_grid_manifest


@app.cell
def controls(broker_profile, daily, manifest, threshold_grid):
    _strategies = visible_strategies(daily["Strategy"].dropna().unique())
    strategy_selector = mo.ui.multiselect(
        options=_strategies,
        value=default_focus_strategies(_strategies),
        label="Main chart strategies",
        full_width=True,
    )
    include_spy_benchmark = mo.ui.checkbox(
        value=True,
        label="Include SPY benchmark in Portfolio View",
    )
    start_date = mo.ui.date(
        value=daily["Date"].min().date(),
        label="Initial date",
    )
    end_date = mo.ui.date(
        value=daily["Date"].max().date(),
        label="Final date",
    )
    _entries = [float(x) for x in sorted(threshold_grid["Entry Threshold"].dropna().unique())]
    _exits = [float(x) for x in sorted(threshold_grid["Exit Threshold"].dropna().unique())]
    _profiled_grid = filter_broker_limits(threshold_grid, broker_profile.value)
    _leverages = [float(x) for x in sorted(_profiled_grid["Overlay Weight (%)"].dropna().unique())]
    if not _leverages:
        _leverages = [float(x) for x in sorted(threshold_grid["Overlay Weight (%)"].dropna().unique())]
    portfolio_entry = mo.ui.dropdown(
        options=_entries,
        value=30.0 if 30.0 in _entries else _entries[0],
        label="Entry RSI",
    )
    portfolio_exit = mo.ui.dropdown(
        options=_exits,
        value=50.0 if 50.0 in _exits else _exits[0],
        label="Exit RSI",
    )
    portfolio_leverage = mo.ui.slider(
        steps=_leverages,
        value=max_allowed_leverage_pct(_leverages, broker_profile.value),
        show_value=True,
        include_input=True,
        label="Overlay leverage %",
    )
    mo.md(
        f"""
        **Research bundle:** `{manifest['strategy_id']}` |
        **Generated:** {manifest['generated_at']} |
        **Range:** {manifest['date_range']['actual_start']} to {manifest['date_range']['actual_end']} |
        **Default overlay cap:** {manifest['global_overlay_cap']:.0%}
        """
    )
    return (
        end_date,
        include_spy_benchmark,
        portfolio_entry,
        portfolio_exit,
        portfolio_leverage,
        start_date,
        strategy_selector,
    )


@app.cell
def derived_tables(manifest, overlay_summary, pass_fail):
    _tables = derive_leverage_tables(manifest, overlay_summary, pass_fail)
    allocation = _tables["allocation"]
    cards = _tables["cards"]
    default_rank = _tables["default_rank"]
    overlay_specs = _tables["overlay_specs"]
    pass_table = _tables["pass_table"]
    verdict_table = _tables["verdict_table"]
    return (
        allocation,
        cards,
        default_rank,
        overlay_specs,
        pass_table,
        verdict_table,
    )


@app.cell
def executive_summary(
    broker_profile,
    cards,
    default_rank,
    full_grid_leaderboard,
    full_grid_manifest,
    manifest,
    mixed_pass_fail,
    mixed_sweep_pass_fail,
    mixed_sweep_stability,
    pass_table,
    threshold_grid,
    verdict_table,
):
    _candidate_funnel = build_candidate_funnel(
        mixed_pass_fail,
        mixed_sweep_pass_fail,
        mixed_sweep_stability,
        broker_profile.value,
    )
    _full_grid_filtered = filter_broker_limits(full_grid_leaderboard, broker_profile.value)
    _full_grid_top = _full_grid_filtered.head(1).copy()
    _impact = pd.concat([
        broker_filter_impact(threshold_grid, broker_profile.value, label="Single-ETF threshold grid"),
        broker_filter_impact(mixed_sweep_pass_fail, broker_profile.value, label="Disciplined sweep candidates"),
        broker_filter_impact(full_grid_leaderboard, broker_profile.value, label="Full-grid brute-force leaderboard"),
    ], ignore_index=True)
    _generated = manifest.get("generated_at", "n/a")
    _end = manifest.get("date_range", {}).get("actual_end", "n/a")
    _full_grid_generated = full_grid_manifest.get("generated_at", "n/a")
    _full_grid_grid = full_grid_manifest.get("mixed_grid", {})
    _entry_values = [float(x) for x in _full_grid_grid.get("entry_thresholds", [])]
    _exit_values = [float(x) for x in _full_grid_grid.get("exit_thresholds", [])]
    _valid_rules = sum(1 for _entry in _entry_values for _exit in _exit_values if _exit > _entry)
    _full_grid_rows = (
        _valid_rules
        * _valid_rules
        * len(_full_grid_grid.get("spy_weight_grid", []))
        * len(_full_grid_grid.get("gld_weight_grid", []))
        * len(_full_grid_grid.get("cap_grid", []))
    )
    _leader_cols = [
        "Source", "Candidate", "Benchmark", "Global Cap", "SPY Rule", "GLD Rule",
        "Average OOS Calmar", "Average OOS Calmar Delta", "Worst OOS Calmar Delta",
        "Worst OOS MaxDD Delta (%)", "Min OOS Trade Episodes",
        "Average OOS Exposure (%)", "RF Cost Pass Splits", "Overall Pass", "Caveat", "Promotion Tier",
    ]
    _best = _candidate_funnel.head(1)
    _best_name = _best["Candidate"].iloc[0] if not _best.empty else "No promoted candidate loaded"
    _best_calmar = _best["Average OOS Calmar"].iloc[0] if not _best.empty and "Average OOS Calmar" in _best else "n/a"
    _full_grid_cols = [
        "Rank", "Overall Pass", "Broker Profile", "SPY Rule", "GLD Rule",
        "Global Cap", "Average OOS Calmar", "Average OOS Calmar Delta",
        "Worst OOS Calmar Delta", "Worst OOS MaxDD Delta (%)",
        "Average OOS CAGR Delta (%)", "RF Cost Pass Splits",
        "Min OOS Trade Episodes", "Average OOS Exposure (%)",
        "Annual Calmar Improvement Years", "Annual Years Tested",
    ]
    _fig_frontier = plot_calmar_maxdd_scatter_figure(_candidate_funnel)
    mo.vstack([
        mo.md(
            f"""
            ## 1. Audit Summary / Executive Summary

            **Best Calmar Candidate vs Base:** `{_best_name}`.
            Average OOS Calmar: `{_best_calmar}`.

            The SPY+GLD mixed research is reviewed through the active broker
            limit profile: **{broker_profile.value}**. Rows above the selected
            cap/sleeve limit are filtered from promotion views; the unrestricted
            profile keeps them visible and tags aggressive research. Default
            SPY+GLD 30/50/20 is **control only** and excluded from promotion.

            **Data freshness:** main bundle generated `{_generated}` through `{_end}`.
            Full-grid brute-force validation generated `{_full_grid_generated}` across
            `{_full_grid_rows:,}` configs per period.
            """
        ),
        broker_profile,
        mo.ui.table(presentation_table(_candidate_funnel, _leader_cols), label="Calmar-First Candidate Leaderboard"),
        mo.ui.table(presentation_table(_full_grid_top, _full_grid_cols), label="Full-Grid Top Calmar Row"),
        mo.as_html(_fig_frontier),
        mo.ui.table(_impact, label="Broker Filter Impact"),
        mo.ui.table(cards, label="Headline Cards"),
        mo.hstack([
            mo.ui.table(
                presentation_table(default_rank, [
                    "Ticker", "CAGR (%)", "RF Opportunity Cost CAGR (%)",
                    "Calmar", "Max Drawdown (%)", "Active Days (%)",
                    "Average Overlay Exposure (%)",
                ]),
                label="Default 20% Overlay Ranking",
            ),
            mo.ui.table(verdict_table, label="ETF-Level Verdict"),
        ], gap=2),
        mo.ui.table(
            presentation_table(pass_table, [
                "Ticker", "Selector Label", "Verdict", "Splits Passed",
                "Low Trade Count Splits", "Worst OOS Calmar Delta",
                "Worst OOS MaxDD Delta (%)", "Average OOS CAGR Delta (%)",
                "Caveat",
            ]),
            label="OOS Pass/Fail Summary",
        ),
    ])
    return


@app.cell
def methodology(allocation, overlay_specs):
    mo.vstack([
        mo.md(
            """
            ## 2. Methodology

            **What is being tested?** The base risk-parity portfolio is left
            unchanged. Each overlay adds temporary extra exposure when an ETF's
            own RSI-14 becomes oversold, then removes the exposure after RSI
            recovers.

            **How to read the results:** positive CAGR and Calmar deltas are good.
            A positive MaxDD delta means the overlay had a shallower drawdown
            than the base portfolio. RF opportunity cost charges the overlay for
            capital that could otherwise earn the risk-free rate.

            **Why OOS matters:** threshold grids are easy to overfit. The OOS
            validation selects rules using only pre-2018, pre-2020, or pre-2022
            data, then tests them after the split.
            """
        ),
        mo.hstack([
            mo.ui.table(allocation, label="Base Allocation"),
            mo.ui.table(overlay_specs, label="Default Overlay Specs"),
        ], gap=2),
    ])
    return


@app.cell
def filtered_data(
    broker_profile,
    daily,
    dd_events,
    diagnostics,
    end_date,
    include_spy_benchmark,
    manifest,
    portfolio_entry,
    portfolio_exit,
    portfolio_leverage,
    start_date,
    strategy_selector,
    threshold_grid,
):
    _start, _end = start_date.value, end_date.value
    filtered_daily = slice_dates(
        scale_overlay_leverage(daily, manifest, float(portfolio_leverage.value)),
        _start, _end,
    )
    filtered_diagnostics = slice_dates(diagnostics, _start, _end)
    selected_portfolio_strategies = portfolio_view_strategies(
        strategy_selector.value,
        include_spy_benchmark.value,
        filtered_daily["Strategy"].dropna().unique(),
    )
    selected_metrics = selected_window_metrics(filtered_daily, filtered_diagnostics)
    selected_metrics = selected_metrics[
        selected_metrics["Strategy"].isin(selected_portfolio_strategies)
    ].sort_values(["Calmar", "CAGR (%)"], ascending=[False, False])
    selected_drawdown_events = dd_events[
        dd_events["Strategy"].isin(selected_portfolio_strategies)
    ].copy()

    _profiled_grid = filter_broker_limits(threshold_grid, broker_profile.value)
    _grid_cols = [
        "Ticker", "Entry Threshold", "Exit Threshold", "Overlay Weight (%)",
        "Calmar", "CAGR (%)", "Max Drawdown (%)", "Sharpe",
        "Active Days (%)", "RF Opportunity Cost CAGR (%)",
        "Incremental CAGR (%)", "Incremental Calmar", "Incremental MaxDD (%)",
        "Broker Profile", "Broker Allowed", "Promotion Tier",
    ]
    _mask = (
        np.isclose(_profiled_grid["Entry Threshold"], float(portfolio_entry.value), atol=1e-9)
        & np.isclose(_profiled_grid["Exit Threshold"], float(portfolio_exit.value), atol=1e-9)
        & np.isclose(_profiled_grid["Overlay Weight (%)"], float(portfolio_leverage.value), atol=1e-9)
    )
    selected_grid_rows = (
        _profiled_grid[_mask][[c for c in _grid_cols if c in _profiled_grid.columns]]
        .sort_values("Calmar", ascending=False)
        .reset_index(drop=True)
    )
    return (
        filtered_daily,
        selected_drawdown_events,
        selected_grid_rows,
        selected_metrics,
        selected_portfolio_strategies,
    )


@app.cell
def plot_growth_and_drawdown(
    broker_profile,
    end_date,
    filtered_daily,
    include_spy_benchmark,
    manifest,
    portfolio_entry,
    portfolio_exit,
    portfolio_leverage,
    selected_drawdown_events,
    selected_grid_rows,
    selected_metrics,
    selected_portfolio_strategies,
    start_date,
    strategy_selector,
):
    _fig = plot_growth_and_drawdown_figure(filtered_daily, selected_portfolio_strategies)
    _grid_label = (
        f"Rule Metrics — Entry RSI {portfolio_entry.value:.0f}, "
        f"Exit RSI {portfolio_exit.value:.0f}, "
        f"Leverage {portfolio_leverage.value:.0f}%  "
        f"(full-period grid metrics, not date-filtered)"
    )
    _default_lev = float(
        manifest["overlay_specs"][0].get("overlay_weight", 0.20) * 100
        if manifest.get("overlay_specs") else 20.0
    )
    _lev_note = (
        f"\n\n> **Leverage scaled to {portfolio_leverage.value:.0f}%** "
        f"(bundle default {_default_lev:.0f}%). "
        "Entry/exit signal days are unchanged — only overlay size is adjusted. "
        "Regenerate the bundle to explore different entry/exit thresholds."
        if abs(portfolio_leverage.value - _default_lev) > 0.5 else ""
    )
    mo.vstack([
        mo.md("## 3. Portfolio View"),
        mo.hstack([broker_profile, start_date, end_date, strategy_selector, include_spy_benchmark], gap=2),
        mo.hstack([portfolio_entry, portfolio_exit, portfolio_leverage], gap=2),
        mo.md(
            "The top chart shows cumulative growth for the base portfolio and the "
            "main overlay candidates, filtered to the selected window. "
            "The bottom chart shows whether the extra return came with deeper drawdowns."
            + _lev_note
        ),
        mo.as_html(_fig),
        mo.ui.table(selected_grid_rows, label=_grid_label),
        mo.ui.table(selected_metrics, label="Selected-Window Ranking"),
        mo.ui.table(selected_drawdown_events, label="Worst Drawdown Events"),
    ])
    return


@app.cell
def plot_default_overlay_comparison(default_rank, overlay_summary):
    _fig = plot_default_overlay_figure(default_rank, overlay_summary)
    mo.vstack([
        mo.md(
            """
            ## 4. Default Overlay Results, All ETFs

            This compares the simple default rule for every ETF: RSI-14 entry
            below 30, exit above 50, +20% overlay, one-day execution lag.
            """
        ),
        mo.as_html(_fig),
    ])
    return


@app.cell
def oos_validation_view(oos_summary, pass_table):
    if oos_summary.empty:
        _view = mo.md("## 5. OOS Validation\n\nNo OOS validation bundle loaded yet.")
    else:
        _fig = plot_oos_validation_figure(oos_summary)
        _view = mo.vstack([
            mo.md(
                """
                ## 5. OOS Validation

                A rule passes a split when it improves Calmar, preserves or
                improves CAGR, avoids a material MaxDD deterioration, and still
                beats the base portfolio after RF opportunity cost.
                """
            ),
            mo.as_html(_fig),
            mo.ui.table(
                presentation_table(pass_table, [
                    "Ticker", "Selector Label", "Verdict", "Splits Passed",
                    "Worst OOS Calmar Delta", "Worst OOS MaxDD Delta (%)",
                    "Average OOS CAGR Delta (%)", "Caveat",
                ]),
                label="ETF x Selector OOS Verdicts",
            ),
        ])
    _view
    return


@app.cell
def gld_heatmap_controls(broker_profile, threshold_grid):
    _gld = filter_broker_limits(threshold_grid[threshold_grid["Ticker"] == "GLD"], broker_profile.value)
    if _gld.empty:
        _gld = threshold_grid[threshold_grid["Ticker"] == "GLD"]
    _g_entries = [float(x) for x in sorted(_gld["Entry Threshold"].dropna().unique())]
    _g_exits = [float(x) for x in sorted(_gld["Exit Threshold"].dropna().unique())]
    _g_leverages = [float(x) for x in sorted(_gld["Overlay Weight (%)"].dropna().unique())]
    _metric_opts = [
        "Calmar", "CAGR (%)", "Sharpe", "Max Drawdown (%)",
        "Worst Month (%)", "Active Days (%)", "Average Overlay Weight (%)",
        "RF Opportunity Cost CAGR (%)", "Incremental CAGR (%)",
        "Incremental Calmar", "Incremental MaxDD (%)",
        "Incremental CAGR per Avg Overlay",
    ]
    gld_metric = mo.ui.dropdown(options=_metric_opts, value="Calmar", label="Metric")
    gld_heatmap_mode = mo.ui.dropdown(
        options=["Leverage x Exit", "Leverage x Entry", "Exit x Entry"],
        value="Exit x Entry",
        label="Heatmap axes",
    )
    gld_entry = mo.ui.dropdown(
        options=_g_entries,
        value=22.0 if 22.0 in _g_entries else _g_entries[0],
        label="Entry RSI (fixed axis)",
    )
    gld_exit = mo.ui.dropdown(
        options=_g_exits,
        value=46.0 if 46.0 in _g_exits else _g_exits[0],
        label="Exit RSI (fixed axis)",
    )
    gld_leverage = mo.ui.slider(
        steps=_g_leverages,
        value=max_allowed_leverage_pct(_g_leverages, broker_profile.value),
        show_value=True,
        include_input=True,
        label="Leverage % (fixed axis)",
    )
    return gld_entry, gld_exit, gld_heatmap_mode, gld_leverage, gld_metric


@app.cell
def gld_deep_dive(
    broker_profile,
    gld_entry,
    gld_exit,
    gld_heatmap_mode,
    gld_leverage,
    gld_metric,
    oos_summary,
    selected_rules,
    threshold_grid,
):
    if oos_summary.empty:
        _view = mo.md("## 6. GLD Deep Dive\n\nNo OOS validation bundle loaded yet.")
    else:
        _gld_oos = oos_summary[
            (oos_summary["Ticker"] == "GLD")
            & (oos_summary["Selector"].isin(["default_30_50_20", "robust_calmar_region"]))
        ].copy()
        _gld_oos["Rule"] = _gld_oos["Selector"].map(label_selector)
        _fig = plot_etf_oos_bars_figure(_gld_oos, "GLD", [
            ("OOS CAGR Delta (%)", "GLD OOS CAGR delta"),
            ("OOS Calmar Delta", "GLD OOS Calmar delta"),
        ])
        _gld_grid = filter_broker_limits(threshold_grid[threshold_grid["Ticker"] == "GLD"], broker_profile.value)
        if _gld_grid.empty:
            _gld_grid = threshold_grid[threshold_grid["Ticker"] == "GLD"]
        _pivot, _xlabel, _ylabel, _title_suffix = compute_heatmap_pivot(
            _gld_grid,
            gld_heatmap_mode.value,
            float(gld_entry.value),
            float(gld_exit.value),
            float(gld_leverage.value),
            gld_metric.value,
        )
        _fig_heatmap = plot_threshold_heatmap_figure(
            _pivot, "GLD", gld_metric.value, _xlabel, _ylabel, _title_suffix,
        )
        _high_gld = selected_rules[
            (selected_rules["Ticker"] == "GLD") & (selected_rules["Overlay Weight"] > 0.50)
        ].sort_values(["Split", "Selector Label"])
        _low_gld = selected_rules[
            (selected_rules["Ticker"] == "GLD") & (selected_rules["Overlay Weight"] <= 0.50)
        ].sort_values(["Split", "Selector Label"])
        _top_gld = _gld_grid.sort_values(
            ["Calmar", "CAGR (%)"], ascending=[False, False]
        ).head(25)
        _heatmap_controls = mo.vstack([
            gld_metric, gld_heatmap_mode, gld_entry, gld_exit, gld_leverage,
        ])
        _view = mo.vstack([
            mo.md(
                """
                ## 6. GLD Deep Dive

                The default GLD rule is the cleanest result. The high-leverage
                robust GLD rule is interesting, but it carries a low-trade-count
                warning and should stay in research.

                Use the controls beside the heatmap to explore the threshold grid
                interactively. The fixed-axis control is ignored when the heatmap
                mode does not use it.
                """
            ),
            mo.as_html(_fig),
            mo.hstack([_heatmap_controls, mo.as_html(_fig_heatmap)]),
            mo.ui.table(
                presentation_table(_gld_oos, [
                    "Split", "Rule", "Entry Threshold", "Exit Threshold",
                    "Overlay Weight (%)", "OOS CAGR Delta (%)",
                    "OOS Calmar Delta", "OOS MaxDD Delta (%)",
                    "OOS Trade Episodes", "Pass Notes",
                ]),
                label="GLD Default vs Robust High-Leverage Candidate",
            ),
            mo.ui.table(
                presentation_table(_high_gld, [
                    "Split", "Selector Label", "Entry Threshold", "Exit Threshold",
                    "Overlay Weight (%)", "Calmar", "CAGR (%)",
                    "Max Drawdown (%)", "Active Days (%)", "Robust Avg Calmar",
                ]),
                label="Selected GLD Rules Above 50% Leverage",
            ),
            mo.ui.table(
                presentation_table(_low_gld, [
                    "Split", "Selector Label", "Entry Threshold", "Exit Threshold",
                    "Overlay Weight (%)", "Calmar", "CAGR (%)",
                    "Max Drawdown (%)", "Active Days (%)", "Robust Avg Calmar",
                ]),
                label="Selected GLD Rules At or Below 50% Leverage",
            ),
            mo.ui.table(
                presentation_table(_top_gld, [
                    "Ticker", "Entry Threshold", "Exit Threshold",
                    "Overlay Weight (%)", "Calmar", "CAGR (%)",
                    "Max Drawdown (%)", "Active Days (%)",
                    "RF Opportunity Cost CAGR (%)",
                ]),
                label="GLD Threshold Grid CSV Rows: Top by Calmar",
            ),
        ])
    _view
    return


@app.cell
def spy_heatmap_controls(broker_profile, threshold_grid):
    _spy = filter_broker_limits(threshold_grid[threshold_grid["Ticker"] == "SPY"], broker_profile.value)
    if _spy.empty:
        _spy = threshold_grid[threshold_grid["Ticker"] == "SPY"]
    _s_entries = [float(x) for x in sorted(_spy["Entry Threshold"].dropna().unique())]
    _s_exits = [float(x) for x in sorted(_spy["Exit Threshold"].dropna().unique())]
    _s_leverages = [float(x) for x in sorted(_spy["Overlay Weight (%)"].dropna().unique())]
    _s_metric_opts = [
        "Calmar", "CAGR (%)", "Sharpe", "Max Drawdown (%)",
        "Worst Month (%)", "Active Days (%)", "Average Overlay Weight (%)",
        "RF Opportunity Cost CAGR (%)", "Incremental CAGR (%)",
        "Incremental Calmar", "Incremental MaxDD (%)",
        "Incremental CAGR per Avg Overlay",
    ]
    spy_metric = mo.ui.dropdown(options=_s_metric_opts, value="Calmar", label="Metric")
    spy_heatmap_mode = mo.ui.dropdown(
        options=["Leverage x Exit", "Leverage x Entry", "Exit x Entry"],
        value="Exit x Entry",
        label="Heatmap axes",
    )
    spy_entry = mo.ui.dropdown(
        options=_s_entries,
        value=22.0 if 22.0 in _s_entries else _s_entries[0],
        label="Entry RSI (fixed axis)",
    )
    spy_exit = mo.ui.dropdown(
        options=_s_exits,
        value=42.0 if 42.0 in _s_exits else _s_exits[0],
        label="Exit RSI (fixed axis)",
    )
    spy_leverage = mo.ui.slider(
        steps=_s_leverages,
        value=max_allowed_leverage_pct(_s_leverages, broker_profile.value),
        show_value=True,
        include_input=True,
        label="Leverage % (fixed axis)",
    )
    return spy_entry, spy_exit, spy_heatmap_mode, spy_leverage, spy_metric


@app.cell
def spy_deep_dive(
    broker_profile,
    oos_summary,
    spy_entry,
    spy_exit,
    spy_heatmap_mode,
    spy_leverage,
    spy_metric,
    threshold_grid,
):
    if oos_summary.empty:
        _view = mo.md("## 7. SPY Deep Dive\n\nNo OOS validation bundle loaded yet.")
    else:
        _spy_oos = oos_summary[
            (oos_summary["Ticker"] == "SPY")
            & (oos_summary["Selector"].isin(["default_30_50_20", "best_maxdd_preservation"]))
        ].copy()
        _spy_oos["Rule"] = _spy_oos["Selector"].map(label_selector)
        _fig = plot_etf_oos_bars_figure(_spy_oos, "SPY", [
            ("OOS CAGR Delta (%)", "SPY OOS CAGR delta"),
            ("OOS Calmar Delta", "SPY OOS Calmar delta"),
        ])
        _spy_grid = filter_broker_limits(threshold_grid[threshold_grid["Ticker"] == "SPY"], broker_profile.value)
        if _spy_grid.empty:
            _spy_grid = threshold_grid[threshold_grid["Ticker"] == "SPY"]
        _pivot, _xlabel, _ylabel, _title_suffix = compute_heatmap_pivot(
            _spy_grid,
            spy_heatmap_mode.value,
            float(spy_entry.value),
            float(spy_exit.value),
            float(spy_leverage.value),
            spy_metric.value,
        )
        _fig_heatmap = plot_threshold_heatmap_figure(
            _pivot, "SPY", spy_metric.value, _xlabel, _ylabel, _title_suffix,
        )
        _top_spy = _spy_grid.sort_values(
            ["Calmar", "CAGR (%)"], ascending=[False, False]
        ).head(25)
        _heatmap_controls = mo.vstack([
            spy_metric, spy_heatmap_mode, spy_entry, spy_exit, spy_leverage,
        ])
        _view = mo.vstack([
            mo.md(
                """
                ## 7. SPY Deep Dive

                SPY is the useful equity-return overlay, but it is not as clean
                as GLD. The default rule passes OOS, while the in-sample optimized
                rules can add return at the cost of deeper drawdowns.

                Use the controls beside the heatmap to explore the threshold grid
                interactively. The fixed-axis control is ignored when the heatmap
                mode does not use it.
                """
            ),
            mo.as_html(_fig),
            mo.hstack([_heatmap_controls, mo.as_html(_fig_heatmap)]),
            mo.ui.table(
                presentation_table(_spy_oos, [
                    "Split", "Rule", "Entry Threshold", "Exit Threshold",
                    "Overlay Weight (%)", "OOS CAGR Delta (%)",
                    "OOS Calmar Delta", "OOS MaxDD Delta (%)",
                    "OOS Trade Episodes", "Pass Notes",
                ]),
                label="SPY Default vs Drawdown-Preservation Candidate",
            ),
            mo.ui.table(
                presentation_table(_top_spy, [
                    "Ticker", "Entry Threshold", "Exit Threshold",
                    "Overlay Weight (%)", "Calmar", "CAGR (%)",
                    "Max Drawdown (%)", "Active Days (%)",
                    "RF Opportunity Cost CAGR (%)",
                ]),
                label="SPY Threshold Grid CSV Rows: Top by Calmar",
            ),
        ])
    _view
    return


@app.cell
def plot_all_etf_rsi_small_multiples(
    diagnostics,
    end_date,
    signals,
    start_date,
):
    _start, _end = start_date.value, end_date.value
    _sig = slice_dates(signals, _start, _end)
    _diag = slice_dates(diagnostics, _start, _end)
    _fig = plot_all_etf_rsi_figure(_sig, _diag)
    mo.vstack([
        mo.md(
            """
            ## 8. RSI Signals Across All ETFs

            Each panel shows the ETF's own RSI-14. The blue shading is applied
            overlay exposure after the one-day execution lag. This replaces the
            old ETF toggle so the reviewer can compare signal behavior directly.
            """
        ),
        mo.as_html(_fig),
    ])
    return


@app.cell
def plot_yearly(end_date, start_date, strategy_selector, yearly_overlay):
    _start_year = pd.Timestamp(start_date.value).year
    _end_year = pd.Timestamp(end_date.value).year
    _data = yearly_overlay[
        (yearly_overlay["Year"] >= _start_year)
        & (yearly_overlay["Year"] <= _end_year)
        & (yearly_overlay["Strategy"].isin(strategy_selector.value))
    ].copy()
    _fig = plot_yearly_figure(_data, strategy_selector.value)
    mo.vstack([
        mo.md(
            """
            ## 9. Year-by-Year Behavior

            These charts show whether the overlay improvement is consistent by
            year or driven by a small number of episodes.
            """
        ),
        mo.as_html(_fig),
    ])
    return


@app.cell
def appendix_threshold_controls(broker_profile, threshold_grid):
    _profiled_grid = filter_broker_limits(threshold_grid, broker_profile.value)
    if _profiled_grid.empty:
        _profiled_grid = threshold_grid.copy()
    _ticker_options = sorted(_profiled_grid["Ticker"].dropna().unique())
    inspect_etf = mo.ui.dropdown(
        options=_ticker_options,
        value="GLD" if "GLD" in _ticker_options else _ticker_options[0],
        label="Inspect ETF",
    )
    metric = mo.ui.dropdown(
        options=[
            "Calmar", "CAGR (%)", "Sharpe", "Max Drawdown (%)",
            "Worst Month (%)", "Active Days (%)", "Average Overlay Weight (%)",
            "RF Opportunity Cost CAGR (%)", "Incremental CAGR (%)",
            "Incremental Calmar", "Incremental MaxDD (%)",
            "Incremental CAGR per Avg Overlay",
        ],
        value="Calmar",
        label="Grid metric",
    )
    _entries = [float(x) for x in sorted(_profiled_grid["Entry Threshold"].dropna().unique())]
    _exits = [float(x) for x in sorted(_profiled_grid["Exit Threshold"].dropna().unique())]
    _leverages = [float(x) for x in sorted(_profiled_grid["Overlay Weight (%)"].dropna().unique())]
    entry_selector = mo.ui.dropdown(
        options=_entries, value=30.0 if 30.0 in _entries else _entries[0], label="Entry"
    )
    exit_selector = mo.ui.dropdown(
        options=_exits, value=50.0 if 50.0 in _exits else _exits[0], label="Exit"
    )
    leverage_selector = mo.ui.slider(
        steps=_leverages,
        value=max_allowed_leverage_pct(_leverages, broker_profile.value),
        show_value=True,
        include_input=True,
        label="Leverage %",
    )
    heatmap_mode = mo.ui.dropdown(
        options=["Leverage x Exit", "Leverage x Entry", "Exit x Entry"],
        value="Exit x Entry",
        label="Heatmap",
    )
    return (
        entry_selector,
        exit_selector,
        heatmap_mode,
        inspect_etf,
        leverage_selector,
        metric,
    )


@app.cell
def show_threshold_grid(
    broker_profile,
    entry_selector,
    exit_selector,
    heatmap_mode,
    inspect_etf,
    leverage_selector,
    metric,
    threshold_grid,
):
    _filtered = filter_broker_limits(threshold_grid, broker_profile.value)
    if _filtered.empty:
        _filtered = threshold_grid.copy()
    _data = _filtered[_filtered["Ticker"] == inspect_etf.value]
    _pivot, _xlabel, _ylabel, _title_suffix = compute_heatmap_pivot(
        _data,
        heatmap_mode.value,
        float(entry_selector.value),
        float(exit_selector.value),
        float(leverage_selector.value),
        metric.value,
    )
    _fig = plot_threshold_heatmap_figure(
        _pivot, inspect_etf.value, metric.value, _xlabel, _ylabel, _title_suffix
    )

    _cols = [
        "Ticker", "Entry Threshold", "Exit Threshold", "Overlay Weight (%)", "Calmar",
        "CAGR (%)", "Sharpe", "Max Drawdown (%)", "Worst Month (%)",
        "Active Days (%)", "Incremental CAGR (%)", "Incremental Calmar",
        "Incremental MaxDD (%)", "Incremental CAGR per Avg Overlay", "Turnover",
        "Broker Profile", "Broker Allowed", "Aggressive Research", "Promotion Tier",
    ]
    _best_calmar = _data.sort_values(["Calmar", "CAGR (%)"], ascending=[False, False]).head(20)
    _best_cagr = _data.sort_values(["CAGR (%)", "Calmar"], ascending=[False, False]).head(20)
    _safest = _data.sort_values(["Max Drawdown (%)", "Calmar"], ascending=[False, False]).head(20)
    _default = _data[
        np.isclose(_data["Overlay Weight"], 0.20, atol=1e-9)
        & np.isclose(_data["Entry Threshold"], 30.0, atol=1e-9)
        & np.isclose(_data["Exit Threshold"], 50.0, atol=1e-9)
    ]

    _controls = mo.vstack([
        inspect_etf,
        metric,
        heatmap_mode,
        entry_selector,
        exit_selector,
        leverage_selector,
    ])
    mo.vstack([
        mo.md(
            """
            ## Appendix: Threshold Grid Inspection

            This is the research workbench. It is useful when someone asks why a
            specific ETF or threshold was not selected. The controls are placed
            beside the heatmap so the selected slice is always visible.
            """
        ),
        mo.hstack([_controls, mo.as_html(_fig)]),
        mo.ui.table(broker_filter_impact(threshold_grid, broker_profile.value, label="Threshold grid")),
        mo.accordion({
            "Best by Calmar": mo.ui.table(_best_calmar[_cols]),
            "Best by CAGR": mo.ui.table(_best_cagr[_cols]),
            "Safest by MaxDD": mo.ui.table(_safest[_cols]),
            "Default 20% 30/50": mo.ui.table(_default[_cols]),
        }),
    ])
    return


@app.cell
def choose_mixed_oos_bundle(broker_profile):
    mixed_bundle_path = mo.ui.text(
        value=latest_bundle_for_profile(MIXED_OOS_BUNDLE_ROOT, broker_profile.value),
        label="Mixed leverage OOS bundle",
        full_width=True,
    )
    return (mixed_bundle_path,)


@app.cell
def load_mixed_oos_bundle(mixed_bundle_path):
    from pathlib import Path as _Path

    _raw = mixed_bundle_path.value.strip().strip("'\"")
    _empty = (pd.DataFrame(),) * 8
    if not _raw:
        mixed_fixed, mixed_selectors, mixed_pass_fail, mixed_selected, mixed_daily, mixed_diagnostics, mixed_episodes, mixed_fixed_walk_forward = _empty
    else:
        _bundle = _Path(_raw).expanduser()
        if not _bundle.is_absolute():
            _bundle = ROOT / _bundle
        if not _bundle.exists() or not (_bundle / "manifest.json").exists():
            mixed_fixed, mixed_selectors, mixed_pass_fail, mixed_selected, mixed_daily, mixed_diagnostics, mixed_episodes, mixed_fixed_walk_forward = _empty
        else:
            def _read_csv(name):
                _path = _bundle / name
                return pd.read_csv(_path) if _path.exists() else pd.DataFrame()

            mixed_fixed = pd.read_csv(_bundle / "fixed_candidates_oos.csv")
            mixed_selectors = pd.read_csv(_bundle / "oos_summary.csv")
            mixed_pass_fail = pd.read_csv(_bundle / "pass_fail_summary.csv")
            mixed_selected = pd.read_csv(_bundle / "selected_rules.csv")
            mixed_daily = pd.read_csv(_bundle / "oos_daily_series.csv")
            mixed_diagnostics = pd.read_csv(_bundle / "oos_overlay_diagnostics.csv")
            mixed_episodes = pd.read_csv(_bundle / "oos_trade_episodes.csv")
            mixed_fixed_walk_forward = _read_csv("fixed_candidate_walk_forward_summary.csv")
    return (
        mixed_daily,
        mixed_diagnostics,
        mixed_episodes,
        mixed_fixed,
        mixed_fixed_walk_forward,
        mixed_pass_fail,
        mixed_selected,
        mixed_selectors,
    )


@app.cell
def choose_mixed_oos_split(mixed_daily):
    _splits = sorted(mixed_daily["Split"].dropna().unique()) if not mixed_daily.empty else []
    split_picker = mo.ui.dropdown(
        options=[str(s) for s in _splits],
        value=str(_splits[-1]) if _splits else None,
        label="OOS split for growth chart",
    )
    return (split_picker,)


@app.cell
def choose_mixed_exposure_strategy(mixed_diagnostics):
    _strategies = (
        sorted(mixed_diagnostics["Overlay Strategy"].dropna().unique())
        if not mixed_diagnostics.empty and "Overlay Strategy" in mixed_diagnostics
        else []
    )
    mixed_exposure_strategy = mo.ui.dropdown(
        options=_strategies,
        value=_strategies[0] if _strategies else None,
        label="Exposure candidate",
    )
    return (mixed_exposure_strategy,)


@app.cell
def mixed_oos_results(
    broker_profile,
    mixed_daily,
    mixed_diagnostics,
    mixed_episodes,
    mixed_exposure_strategy,
    mixed_fixed,
    mixed_fixed_walk_forward,
    mixed_pass_fail,
    mixed_selected,
    mixed_selectors,
    split_picker,
):
    if mixed_pass_fail.empty:
        _output = mo.vstack([
            mo.md("## Mixed SPY+GLD OOS Validation\n\n_No mixed leverage OOS bundle loaded._"),
        ])
    else:
        _mixed_fixed = filter_broker_limits(mixed_fixed, broker_profile.value)
        _mixed_fixed["Control Only"] = _mixed_fixed.get("Control Only", _mixed_fixed["Candidate Name"].map(is_control_candidate))
        _main_mixed_fixed = _mixed_fixed[~_mixed_fixed["Control Only"].fillna(False).astype(bool)].copy()
        _control_mixed_fixed = _mixed_fixed[_mixed_fixed["Control Only"].fillna(False).astype(bool)].copy()
        _mixed_selectors = filter_broker_limits(mixed_selectors, broker_profile.value)
        if "Selector" in _mixed_selectors:
            _mixed_selectors = _mixed_selectors[~_mixed_selectors["Selector"].map(is_control_candidate)].copy()
        _mixed_selected = filter_broker_limits(mixed_selected, broker_profile.value)
        if "Selector" in _mixed_selected:
            _mixed_selected = _mixed_selected[~_mixed_selected["Selector"].map(is_control_candidate)].copy()
        _mixed_pass_fail = filter_broker_limits(mixed_pass_fail, broker_profile.value)
        if "Name" in _mixed_pass_fail:
            _mixed_pass_fail = _mixed_pass_fail[~_mixed_pass_fail["Name"].map(is_control_candidate)].copy()
        _fixed_walk_forward = filter_broker_limits(mixed_fixed_walk_forward, broker_profile.value)
        if not _fixed_walk_forward.empty and "Candidate Name" in _fixed_walk_forward:
            _fixed_walk_forward = _fixed_walk_forward[~_fixed_walk_forward["Candidate Name"].map(is_control_candidate)].copy()
        _pf_cols = [
            "Source", "Name", "Benchmark", "Global Cap", "SPY Rule", "GLD Rule",
            "Splits Tested", "Splits Passed", "Average OOS Calmar",
            "Average OOS Calmar Delta", "Worst OOS Calmar Delta",
            "Worst OOS MaxDD Delta (%)", "Min OOS Trade Episodes",
            "Average OOS Exposure (%)", "RF Cost Pass Splits", "Average OOS CAGR Delta (%)",
            "Overall Pass", "MaxDD Breach >3pp",
            "Broker Profile", "Broker Allowed", "Aggressive Research", "Promotion Tier",
        ]

        _fig_fixed = plot_mixed_oos_delta_bars_figure(
            _main_mixed_fixed, name_col="Candidate Name", title_prefix="Fixed Candidates",
        )
        _fig_selectors = plot_mixed_oos_delta_bars_figure(
            _mixed_selectors, name_col="Selector", title_prefix="Grid Selectors",
        )

        _fig_growth = (
            plot_mixed_growth_figure(mixed_daily, split=str(split_picker.value))
            if split_picker.value and not mixed_daily.empty
            else None
        )

        _fig_is_oos = plot_mixed_is_vs_oos_figure(_mixed_selectors, name_col="Selector")
        _fig_exposure = plot_exposure_timeline_figure(
            mixed_diagnostics, strategy=mixed_exposure_strategy.value,
        )
        _fig_episodes = plot_trade_episode_distribution_figure(mixed_episodes)

        _is_oos_fixed = build_is_vs_oos_comparison(_main_mixed_fixed, "Candidate Name")
        _is_oos_selectors = build_is_vs_oos_comparison(_mixed_selectors, "Selector")
        _is_oos = pd.concat([_is_oos_fixed, _is_oos_selectors], ignore_index=True)

        _fixed_cols = [
            "Split", "Candidate Name", "Global Cap",
            "SPY Entry", "SPY Exit", "SPY Weight",
            "GLD Entry", "GLD Exit", "GLD Weight",
            "OOS Overlay Calmar", "OOS Calmar Delta",
            "OOS Overlay Max Drawdown (%)", "OOS MaxDD Delta (%)",
            "OOS CAGR Delta (%)", "OOS Active Days (%)",
            "OOS Trade Episodes", "Pass Split",
            "Broker Profile", "Broker Allowed", "Aggressive Research", "Promotion Tier",
        ]
        _sel_cols = [
            "Split", "Selector", "Global Cap",
            "SPY Entry", "SPY Exit", "SPY Weight",
            "GLD Entry", "GLD Exit", "GLD Weight",
            "OOS Overlay Calmar", "OOS Calmar Delta",
            "OOS Overlay Max Drawdown (%)", "OOS MaxDD Delta (%)",
            "OOS CAGR Delta (%)", "OOS Active Days (%)",
            "OOS Trade Episodes", "Pass Split",
            "Broker Profile", "Broker Allowed", "Aggressive Research", "Promotion Tier",
        ]

        _growth_section = (
            [split_picker, mo.as_html(_fig_growth)] if _fig_growth else [split_picker]
        )

        _output = mo.vstack([
            mo.md(
                """
                ## Mixed SPY+GLD OOS Validation

                Out-of-sample results for capped multi-ETF (SPY+GLD) leverage overlays,
                compared against the base strategy. Default SPY+GLD 30/50/20 is
                kept as a **control only** row in the appendix, not as a promoted
                candidate.
                Fixed candidates use pre-set parameters without re-selection; grid selectors
                pick the best IS-only config per split, then evaluate OOS.
                """
            ),
            mo.md("### Pass/Fail Summary"),
            mo.ui.table(broker_filter_impact(mixed_fixed, broker_profile.value, label="Mixed fixed candidates")),
            mo.ui.table(presentation_table(_mixed_pass_fail, _pf_cols)),
            mo.md("### Fixed Candidates: OOS Deltas by Split"),
            mo.as_html(_fig_fixed),
            mo.md("### Grid Selectors: OOS Deltas by Split"),
            mo.as_html(_fig_selectors),
            mo.md("### OOS Growth & Drawdown"),
            *_growth_section,
            mo.md("### IS vs OOS Calmar: Overfitting Check"),
            mo.as_html(_fig_is_oos),
            mo.ui.table(_is_oos, label="IS vs OOS Calmar Comparison"),
            mo.md("### Daily Exposure and Trade Episodes"),
            mixed_exposure_strategy,
            mo.as_html(_fig_exposure),
            mo.as_html(_fig_episodes),
            mo.md("### Detailed Tables"),
            mo.accordion({
                "Fixed candidates per split": mo.ui.table(presentation_table(_main_mixed_fixed, _fixed_cols)),
                "Control only: default SPY+GLD 30/50/20": mo.ui.table(presentation_table(_control_mixed_fixed, _fixed_cols)),
                "Fixed candidate walk-forward": mo.ui.table(_fixed_walk_forward),
                "Selector winners per split": mo.ui.table(presentation_table(_mixed_selectors, _sel_cols)),
                "Selected rules (IS picks)": mo.ui.table(_mixed_selected),
            }),
        ])
    _output
    return


@app.cell
def choose_mixed_sweep_bundle(broker_profile):
    mixed_sweep_bundle_path = mo.ui.text(
        value=latest_bundle_for_profile(MIXED_SWEEP_BUNDLE_ROOT, broker_profile.value),
        label="Disciplined SPY+GLD sweep bundle",
        full_width=True,
    )
    return (mixed_sweep_bundle_path,)


@app.cell
def load_mixed_sweep_bundle(mixed_sweep_bundle_path):
    from pathlib import Path as _Path

    _raw = mixed_sweep_bundle_path.value.strip().strip("'\"")
    _empty_manifest = {}
    _empty_df = pd.DataFrame()
    if not _raw:
        mixed_sweep_manifest = _empty_manifest
        mixed_sweep_oos = _empty_df
        mixed_sweep_pass_fail = _empty_df
        mixed_sweep_selected = _empty_df
        mixed_sweep_stability = _empty_df
        mixed_sweep_walk_forward = _empty_df
        mixed_sweep_heatmaps = _empty_df
    else:
        _bundle = _Path(_raw).expanduser()
        if not _bundle.is_absolute():
            _bundle = ROOT / _bundle
        _manifest_path = _bundle / "manifest.json"
        if not _bundle.exists() or not _manifest_path.exists():
            mixed_sweep_manifest = _empty_manifest
            mixed_sweep_oos = _empty_df
            mixed_sweep_pass_fail = _empty_df
            mixed_sweep_selected = _empty_df
            mixed_sweep_stability = _empty_df
            mixed_sweep_walk_forward = _empty_df
            mixed_sweep_heatmaps = _empty_df
        else:
            with open(_manifest_path, "r", encoding="utf-8") as _handle:
                mixed_sweep_manifest = json.load(_handle)

            def _read_csv(name):
                _path = _bundle / name
                return pd.read_csv(_path) if _path.exists() else pd.DataFrame()

            mixed_sweep_oos = _read_csv("oos_summary.csv")
            mixed_sweep_pass_fail = _read_csv("pass_fail_summary.csv")
            mixed_sweep_selected = _read_csv("selected_rules.csv")
            mixed_sweep_stability = _read_csv("parameter_stability.csv")
            mixed_sweep_walk_forward = _read_csv("walk_forward_summary.csv")
            mixed_sweep_heatmaps = _read_csv("sweep_heatmap_tables.csv")
    return (
        mixed_sweep_heatmaps,
        mixed_sweep_manifest,
        mixed_sweep_oos,
        mixed_sweep_pass_fail,
        mixed_sweep_selected,
        mixed_sweep_stability,
        mixed_sweep_walk_forward,
    )


@app.cell
def choose_mixed_sweep_heatmap(mixed_sweep_heatmaps):
    _dims = (
        sorted(mixed_sweep_heatmaps["Dimension"].dropna().unique())
        if not mixed_sweep_heatmaps.empty and "Dimension" in mixed_sweep_heatmaps
        else []
    )
    _splits = (
        sorted(str(s) for s in mixed_sweep_heatmaps["Split"].dropna().unique())
        if not mixed_sweep_heatmaps.empty and "Split" in mixed_sweep_heatmaps
        else []
    )
    mixed_sweep_dimension = mo.ui.dropdown(
        options=_dims,
        value="SPY Threshold" if "SPY Threshold" in _dims else (_dims[0] if _dims else None),
        label="Sweep heatmap",
    )
    mixed_sweep_split = mo.ui.dropdown(
        options=_splits,
        value=_splits[-1] if _splits else None,
        label="Structural split",
    )
    mixed_sweep_metric = mo.ui.dropdown(
        options=["Max_Calmar", "Avg_Calmar", "Avg_CAGR", "Avg_MaxDD"],
        value="Max_Calmar",
        label="Heatmap metric",
    )
    return mixed_sweep_dimension, mixed_sweep_metric, mixed_sweep_split


@app.cell
def mixed_sweep_results(
    broker_profile,
    mixed_pass_fail,
    mixed_sweep_dimension,
    mixed_sweep_heatmaps,
    mixed_sweep_manifest,
    mixed_sweep_metric,
    mixed_sweep_oos,
    mixed_sweep_pass_fail,
    mixed_sweep_selected,
    mixed_sweep_split,
    mixed_sweep_stability,
    mixed_sweep_walk_forward,
):
    if mixed_sweep_pass_fail.empty:
        _output = mo.vstack([
            mo.md("## Disciplined SPY+GLD Sweep\n\n_No disciplined mixed sweep bundle loaded._"),
        ])
    else:
        _sweep_pass_fail = filter_broker_limits(mixed_sweep_pass_fail, broker_profile.value)
        if "Selector" in _sweep_pass_fail:
            _sweep_pass_fail = _sweep_pass_fail[~_sweep_pass_fail["Selector"].map(is_control_candidate)].copy()
        _sweep_oos = filter_broker_limits(mixed_sweep_oos, broker_profile.value)
        if "Selector" in _sweep_oos:
            _sweep_oos = _sweep_oos[~_sweep_oos["Selector"].map(is_control_candidate)].copy()
        _sweep_selected = filter_broker_limits(mixed_sweep_selected, broker_profile.value)
        if "Selector" in _sweep_selected:
            _sweep_selected = _sweep_selected[~_sweep_selected["Selector"].map(is_control_candidate)].copy()
        _candidate_funnel = build_candidate_funnel(
            mixed_pass_fail,
            mixed_sweep_pass_fail,
            mixed_sweep_stability,
            broker_profile.value,
        )
        _broker_impact = pd.concat([
            broker_filter_impact(mixed_sweep_pass_fail, broker_profile.value, label="Sweep candidate summary"),
            broker_filter_impact(mixed_sweep_oos, broker_profile.value, label="Sweep structural OOS rows"),
            broker_filter_impact(mixed_sweep_selected, broker_profile.value, label="Sweep selected rules"),
        ], ignore_index=True)
        _pf_cols = [
            "Selector", "Benchmark", "Global Cap", "SPY Rule", "GLD Rule",
            "Structural Splits Tested", "Structural Splits Passed",
            "Average OOS Calmar", "Average OOS Calmar Delta",
            "Worst OOS Calmar Delta", "Worst OOS MaxDD Delta (%)",
            "Average OOS CAGR Delta (%)", "Min OOS Trade Episodes",
            "Average OOS Exposure (%)", "RF Cost Pass Splits",
            "Annual Years Tested", "Annual Calmar Improvement Years",
            "Stable Neighborhood Pass", "Aggressive Cap >30%", "Promotion Tier",
            "Overall Pass",
        ]
        _selected_cols = [
            "Split", "Selector", "Global Cap",
            "SPY Entry", "SPY Exit", "SPY Weight",
            "GLD Entry", "GLD Exit", "GLD Weight",
            "Calmar", "CAGR (%)", "Max Drawdown (%)",
            "Average Overlay Exposure (%)", "Robust Avg Calmar",
            "Robust Neighborhood Size",
            "Broker Profile", "Broker Allowed", "Aggressive Research", "Promotion Tier",
        ]
        _oos_cols = [
            "Split", "Selector", "Global Cap",
            "SPY Entry", "SPY Exit", "SPY Weight",
            "GLD Entry", "GLD Exit", "GLD Weight",
            "OOS Overlay Calmar", "OOS Calmar Delta",
            "OOS Overlay Max Drawdown (%)", "OOS MaxDD Delta (%)",
            "OOS CAGR Delta (%)", "OOS Trade Episodes", "Pass Split",
            "Broker Profile", "Broker Allowed", "Aggressive Research", "Promotion Tier",
        ]
        _stability_cols = [
            "Selector", "Structural Selections", "Unique Configs",
            "Most Common Config Count", "Most Common Config Share",
            "Average Robust Neighborhood Size", "Stable Neighborhood Pass",
            "Annual Years Tested", "Annual Calmar Improvement Years",
            "Most Common Config",
        ]

        if not mixed_sweep_walk_forward.empty:
            _walk_forward = mixed_sweep_walk_forward.copy()
            if "Calmar Improvement" in _walk_forward:
                _walk_forward["Calmar Improvement"] = (
                    _walk_forward["Calmar Improvement"].astype(str).str.lower().isin(["true", "1"])
                )
            _wf_rollup = (
                _walk_forward
                .groupby("Selector", as_index=False)
                .agg(
                    Years=("Year", "nunique"),
                    Calmar_Improvement_Years=("Calmar Improvement", "sum"),
                    Avg_OOS_Calmar_Delta=("OOS Calmar Delta", "mean"),
                    Worst_OOS_MaxDD_Delta=("OOS MaxDD Delta (%)", "min"),
                    Avg_OOS_CAGR_Delta=("OOS CAGR Delta (%)", "mean"),
                    Min_Trade_Episodes=("OOS Trade Episodes", "min"),
                )
            )
            _wf_rollup["Calmar Improvement Rate (%)"] = (
                _wf_rollup["Calmar_Improvement_Years"] / _wf_rollup["Years"] * 100
            ).round(2)
        else:
            _wf_rollup = pd.DataFrame()

        _fig_heatmap = None
        _heatmap_table = pd.DataFrame()
        if (
            not mixed_sweep_heatmaps.empty
            and mixed_sweep_dimension.value
            and mixed_sweep_split.value
            and mixed_sweep_metric.value in mixed_sweep_heatmaps.columns
        ):
            _heatmap_table = mixed_sweep_heatmaps[
                (mixed_sweep_heatmaps["Dimension"] == mixed_sweep_dimension.value)
                & (mixed_sweep_heatmaps["Split"].astype(str) == str(mixed_sweep_split.value))
            ].copy()
            if not _heatmap_table.empty:
                _pivot = _heatmap_table.pivot_table(
                    index="Y", columns="X", values=mixed_sweep_metric.value, aggfunc="mean"
                ).sort_index(ascending=True)
                _fig_heatmap = plot_threshold_heatmap_figure(
                    _pivot,
                    str(mixed_sweep_dimension.value),
                    str(mixed_sweep_metric.value).replace("_", " "),
                    "Entry RSI" if str(mixed_sweep_dimension.value) in {"SPY Threshold", "GLD Threshold"} else "X parameter",
                    "Exit RSI" if str(mixed_sweep_dimension.value) in {"SPY Threshold", "GLD Threshold"} else "Y parameter",
                    f"split {mixed_sweep_split.value}",
                )

        _manifest_table = pd.DataFrame([
            {"Field": "Strategy", "Value": mixed_sweep_manifest.get("strategy_id", "")},
            {"Field": "Generated at", "Value": mixed_sweep_manifest.get("generated_at", "")},
            {"Field": "IS grid rows", "Value": "Stored in is_sweep_grid.parquet"},
            {"Field": "Walk-forward years", "Value": ", ".join(map(str, mixed_sweep_manifest.get("walk_forward_years", [])))},
        ])

        _heatmap_section = (
            [mo.hstack([mixed_sweep_dimension, mixed_sweep_split, mixed_sweep_metric]), mo.as_html(_fig_heatmap)]
            if _fig_heatmap is not None
            else [mo.hstack([mixed_sweep_dimension, mixed_sweep_split, mixed_sweep_metric])]
        )

        _output = mo.vstack([
            mo.md(
                """
                ## 2. Candidate Funnel and Acceptance Gate Evidence

                Second-stage mixed leverage surface map for SPY and GLD. The notebook
                reads compact CSV summaries for review; the full IS grid is kept in
                `is_sweep_grid.parquet` inside the selected bundle.
                """
            ),
            mo.ui.table(_manifest_table, label="Sweep bundle"),
            mo.md("### Candidate Funnel"),
            mo.ui.table(_candidate_funnel),
            mo.as_html(plot_calmar_maxdd_scatter_figure(_candidate_funnel)),
            mo.md("### Broker Filter Impact"),
            mo.ui.table(_broker_impact),
            mo.md("### Acceptance Gate"),
            mo.ui.table(presentation_table(_sweep_pass_fail, _pf_cols)),
            mo.md("### Selector Stability"),
            mo.ui.table(presentation_table(mixed_sweep_stability, _stability_cols)),
            mo.md("### Annual Walk-Forward Rollup"),
            mo.ui.table(_wf_rollup),
            mo.as_html(plot_walk_forward_heatmap_figure(mixed_sweep_walk_forward)),
            mo.md("### Parameter Surface"),
            *_heatmap_section,
            mo.accordion({
                "Structural OOS selector results": mo.ui.table(presentation_table(_sweep_oos, _oos_cols)),
                "IS selected rules": mo.ui.table(presentation_table(_sweep_selected, _selected_cols)),
                "Annual walk-forward rows": mo.ui.table(mixed_sweep_walk_forward),
                "Heatmap table rows": mo.ui.table(_heatmap_table),
            }),
        ])
    _output
    return


@app.cell
def appendix_raw_tables(
    calendar,
    capital,
    include_spy_benchmark,
    rolling,
    stress,
    summary,
    trade_episodes,
):
    mo.vstack([
        mo.md("## Appendix: Raw Tables"),
        mo.accordion({
            "Summary metrics": mo.ui.table(maybe_filter_benchmark(summary, include_spy_benchmark.value)),
            "Capital view": mo.ui.table(capital),
            "Stress period metrics": mo.ui.table(maybe_filter_benchmark(stress, include_spy_benchmark.value)),
            "Calendar-year metrics": mo.ui.table(maybe_filter_benchmark(calendar, include_spy_benchmark.value)),
            "Recent rolling metrics": mo.ui.table(maybe_filter_benchmark(rolling, include_spy_benchmark.value).tail(80)),
            "OOS trade episodes": mo.ui.table(trade_episodes),
        }),
    ])
    return


if __name__ == "__main__":
    app.run()
