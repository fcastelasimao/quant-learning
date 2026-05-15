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
    from research.leverage_analysis import (
        BASE,
        SELECTOR_LABELS,
        compute_heatmap_pivot,
        default_focus_strategies,
        derive_leverage_tables,
        label_selector,
        latest_bundle,
        maybe_filter_benchmark,
        portfolio_view_strategies,
        presentation_table,
        slice_dates,
        strategy_label,
        visible_strategies,
    )
    from research.leverage_plotting import (
        colour_map,
        plot_all_etf_rsi_figure,
        plot_default_overlay_figure,
        plot_etf_oos_bars_figure,
        plot_grid_heatmap,
        plot_grid_heatmaps_figure,
        plot_growth_and_drawdown_figure,
        plot_oos_validation_figure,
        plot_threshold_heatmap_figure,
        plot_yearly_figure,
    )

    BUNDLE_ROOT = ROOT / "results" / "leverage_comparison"
    OOS_BUNDLE_ROOT = ROOT / "results" / "leverage_oos_validation"

    # These constants must remain here for the presentation-only regression test.
    BENCHMARK = "S&P 500 (SPY)"
    PLOT_EXCLUDED = {BENCHMARK}


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

    with open(_manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

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
def controls(daily, manifest):
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
    date_range = mo.ui.date_range(
        start=daily["Date"].min().date(),
        stop=daily["Date"].max().date(),
        value=(daily["Date"].min().date(), daily["Date"].max().date()),
        label="Date range",
    )
    mo.vstack([
        mo.md(
            f"""
            **Research bundle:** `{manifest['strategy_id']}` |
            **Generated:** {manifest['generated_at']} |
            **Range:** {manifest['date_range']['actual_start']} to {manifest['date_range']['actual_end']} |
            **Default overlay cap:** {manifest['global_overlay_cap']:.0%}
            """
        ),
        mo.hstack([date_range, strategy_selector, include_spy_benchmark], gap=2),
    ])
    return date_range, include_spy_benchmark, strategy_selector


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
def executive_summary(cards, default_rank, pass_table, verdict_table):
    mo.vstack([
        mo.md(
            """
            ## 1. Executive Summary

            **Main conclusion:** GLD is the cleanest leverage overlay candidate.
            The default GLD rule improves CAGR, Calmar, and drawdown across the
            OOS windows. SPY and QQQ defaults also pass, but they add equity/growth
            risk. TLT and GSG are not priorities under the current gates.
            """
        ),
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
    daily,
    date_range,
    dd_events,
    diagnostics,
    include_spy_benchmark,
    strategy_selector,
):
    _start, _end = date_range.value
    filtered_daily = slice_dates(daily, _start, _end)
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
    return (
        filtered_daily,
        selected_drawdown_events,
        selected_metrics,
        selected_portfolio_strategies,
    )


@app.cell
def plot_growth_and_drawdown(
    filtered_daily,
    selected_drawdown_events,
    selected_metrics,
    selected_portfolio_strategies,
):
    _fig = plot_growth_and_drawdown_figure(filtered_daily, selected_portfolio_strategies)
    mo.vstack([
        mo.md(
            """
            ## 3. Portfolio View

            The top chart shows cumulative growth for the base portfolio and the
            main overlay candidates. The bottom chart shows whether the extra
            return came with deeper drawdowns.
            """
        ),
        mo.as_html(_fig),
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
def gld_deep_dive(oos_summary, selected_rules, threshold_grid):
    if oos_summary.empty:
        _view = mo.md("## 6. GLD Deep Dive\n\nNo OOS validation bundle loaded yet.")
    else:
        _gld = oos_summary[
            (oos_summary["Ticker"] == "GLD")
            & (oos_summary["Selector"].isin(["default_30_50_20", "robust_calmar_region"]))
        ].copy()
        _gld["Rule"] = _gld["Selector"].map(label_selector)
        _fig = plot_etf_oos_bars_figure(_gld, "GLD", [
            ("OOS CAGR Delta (%)", "GLD OOS CAGR delta"),
            ("OOS Calmar Delta", "GLD OOS Calmar delta"),
        ])
        _fig2 = plot_grid_heatmaps_figure(threshold_grid, "GLD", [
            ("Calmar", 22.0, None),
            ("Calmar", None, 46.0),
        ])
        _high_gld = selected_rules[
            (selected_rules["Ticker"] == "GLD") & (selected_rules["Overlay Weight"] > 0.50)
        ].sort_values(["Split", "Selector Label"])
        _low_gld = selected_rules[
            (selected_rules["Ticker"] == "GLD") & (selected_rules["Overlay Weight"] <= 0.50)
        ].sort_values(["Split", "Selector Label"])
        _top_gld = threshold_grid[threshold_grid["Ticker"] == "GLD"].sort_values(
            ["Calmar", "CAGR (%)"], ascending=[False, False]
        ).head(25)
        _view = mo.vstack([
            mo.md(
                """
                ## 6. GLD Deep Dive

                The default GLD rule is the cleanest result. The high-leverage
                robust GLD rule is interesting, but it carries a low-trade-count
                warning and should stay in research.

                The heatmaps and top-row table below come directly from
                `threshold_grid.csv`, where each row is one ETF x entry RSI x
                exit RSI x leverage percentage test.
                """
            ),
            mo.as_html(_fig),
            mo.as_html(_fig2),
            mo.ui.table(
                presentation_table(_gld, [
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
def spy_deep_dive(oos_summary, threshold_grid):
    if oos_summary.empty:
        _view = mo.md("## 7. SPY Deep Dive\n\nNo OOS validation bundle loaded yet.")
    else:
        _spy = oos_summary[
            (oos_summary["Ticker"] == "SPY")
            & (oos_summary["Selector"].isin(["default_30_50_20", "best_maxdd_preservation"]))
        ].copy()
        _spy["Rule"] = _spy["Selector"].map(label_selector)
        _fig = plot_etf_oos_bars_figure(_spy, "SPY", [
            ("OOS CAGR Delta (%)", "SPY OOS CAGR delta"),
            ("OOS Calmar Delta", "SPY OOS Calmar delta"),
        ])
        _fig2 = plot_grid_heatmaps_figure(threshold_grid, "SPY", [
            ("Calmar", 22.0, None),
            ("Calmar", None, 42.0),
        ])
        _top_spy = threshold_grid[threshold_grid["Ticker"] == "SPY"].sort_values(
            ["Calmar", "CAGR (%)"], ascending=[False, False]
        ).head(25)
        _view = mo.vstack([
            mo.md(
                """
                ## 7. SPY Deep Dive

                SPY is the useful equity-return overlay, but it is not as clean
                as GLD. The default rule passes OOS, while the in-sample optimized
                rules can add return at the cost of deeper drawdowns.

                The heatmaps and top-row table below come directly from
                `threshold_grid.csv`, where each row is one ETF x entry RSI x
                exit RSI x leverage percentage test.
                """
            ),
            mo.as_html(_fig),
            mo.as_html(_fig2),
            mo.ui.table(
                presentation_table(_spy, [
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
def plot_all_etf_rsi_small_multiples(date_range, diagnostics, signals):
    _start, _end = date_range.value
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
def plot_yearly(date_range, strategy_selector, yearly_overlay):
    _start, _end = date_range.value
    _start_year = pd.Timestamp(_start).year
    _end_year = pd.Timestamp(_end).year
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
def appendix_threshold_controls(threshold_grid):
    _ticker_options = sorted(threshold_grid["Ticker"].dropna().unique())
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
    _entries = [float(x) for x in sorted(threshold_grid["Entry Threshold"].dropna().unique())]
    _exits = [float(x) for x in sorted(threshold_grid["Exit Threshold"].dropna().unique())]
    _leverages = [float(x) for x in sorted(threshold_grid["Overlay Weight (%)"].dropna().unique())]
    entry_selector = mo.ui.dropdown(
        options=_entries, value=30.0 if 30.0 in _entries else _entries[0], label="Entry"
    )
    exit_selector = mo.ui.dropdown(
        options=_exits, value=50.0 if 50.0 in _exits else _exits[0], label="Exit"
    )
    leverage_selector = mo.ui.slider(
        steps=_leverages,
        value=20.0 if 20.0 in _leverages else _leverages[0],
        show_value=True,
        include_input=True,
        label="Leverage %",
    )
    heatmap_mode = mo.ui.dropdown(
        options=["Leverage x Exit", "Leverage x Entry", "Exit x Entry"],
        value="Leverage x Exit",
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
    entry_selector,
    exit_selector,
    heatmap_mode,
    inspect_etf,
    leverage_selector,
    metric,
    threshold_grid,
):
    _data = threshold_grid[threshold_grid["Ticker"] == inspect_etf.value]
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
        mo.accordion({
            "Best by Calmar": mo.ui.table(_best_calmar[_cols]),
            "Best by CAGR": mo.ui.table(_best_cagr[_cols]),
            "Safest by MaxDD": mo.ui.table(_safest[_cols]),
            "Default 20% 30/50": mo.ui.table(_default[_cols]),
        }),
    ])
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
