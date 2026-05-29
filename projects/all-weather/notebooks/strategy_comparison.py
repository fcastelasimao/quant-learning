import marimo

__generated_with = "0.23.5"
app = marimo.App(width="full")

with app.setup:
    import json
    import sys
    from pathlib import Path

    import marimo as mo
    import pandas as pd

    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from research.strategy_plotting import (
        COLORS,
        DROPPED_LEGACY_STRATEGIES,
        LEGACY_STRATEGY_RENAMES,
        STRATEGY_ORDER,
        TARGET_LEVERAGE_STRATEGIES,
        clean_strategy_labels,
        latest_bundle,
        plot_calendar_profile,
        plot_drawdowns,
        plot_growth,
        plot_implementation_realism,
        plot_monthly_returns,
        plot_risk_diagnostics,
        plot_rolling_behaviour,
    )
    from research.rebalance_thresholds import (
        SERIES_LABELS as POLICY_SERIES_LABELS,
        SERIES_ORDER as POLICY_SERIES_ORDER,
    )

    BUNDLE_ROOTS = [
        ROOT / "results" / "production_validation",
        ROOT / "results" / "strategy_comparison",
    ]
    FULL_GRID_BUNDLE_ROOT = ROOT / "results" / "mixed_leverage_full_grid_oos"
    REBALANCE_BUNDLE_ROOT = ROOT / "results" / "rebalance_thresholds"


@app.cell
def choose_bundle():
    def _latest_rebalance_bundle(root):
        if not root.exists():
            return ""
        candidates = [
            p for p in root.iterdir()
            if p.is_dir()
            and (p / "run_config.json").exists()
            and (p / "threshold_summary.csv").exists()
        ]
        return str(max(candidates, key=lambda p: p.stat().st_mtime)) if candidates else ""

    bundle_path = mo.ui.text(
        value=latest_bundle(BUNDLE_ROOTS),
        label="Result bundle path",
        full_width=True,
    )
    full_grid_bundle_path = mo.ui.text(
        value=latest_bundle([FULL_GRID_BUNDLE_ROOT]),
        label="Full-grid leverage validation bundle",
        full_width=True,
    )
    rebalance_bundle_path = mo.ui.text(
        value=_latest_rebalance_bundle(REBALANCE_BUNDLE_ROOT),
        label="Rebalance thresholds bundle",
        full_width=True,
    )
    mo.vstack([
        mo.md("# Bank-Facing Strategy Comparison"),
        bundle_path,
        full_grid_bundle_path,
        rebalance_bundle_path,
    ])
    return bundle_path, full_grid_bundle_path, rebalance_bundle_path


@app.cell
def load_bundle(bundle_path):
    from pathlib import Path as _Path

    _raw_bundle_path = bundle_path.value.strip().strip("'\"")
    if not _raw_bundle_path:
        mo.stop(
            True,
            mo.md(
                "Generate a strategy comparison bundle first, then paste its path above. "
                "Example: `results/production_validation/<timestamp>_<strategy_id>`"
            ),
        )

    bundle = _Path(_raw_bundle_path).expanduser()
    if not bundle.is_absolute():
        bundle = ROOT / bundle
    _manifest_path = bundle / "manifest.json"
    if not bundle.exists() or not _manifest_path.exists():
        mo.stop(
            True,
            mo.md(
                f"Could not find a valid result bundle at `{bundle}`. "
                "The selected folder must contain `manifest.json` and the exported CSV files. "
                "Generate one with `python -m research.production_validation`."
            ),
        )

    with open(_manifest_path, "r", encoding="utf-8") as _handle:
        manifest = json.load(_handle)
    _provenance_path = bundle / "price_provenance.json"
    price_provenance = {}
    if _provenance_path.exists():
        with open(_provenance_path, "r", encoding="utf-8") as _handle:
            price_provenance = json.load(_handle)

    daily = pd.read_csv(bundle / "daily_series.csv", parse_dates=["Date"])
    monthly = pd.read_csv(bundle / "monthly_returns.csv", parse_dates=["Date"])
    summary = pd.read_csv(bundle / "summary_metrics.csv")
    calendar = pd.read_csv(bundle / "calendar_year_metrics.csv")
    rolling = pd.read_csv(bundle / "rolling_metrics.csv", parse_dates=["Date"])
    dd_events = pd.read_csv(bundle / "drawdown_events.csv")
    stress = pd.read_csv(bundle / "stress_period_metrics.csv")
    risk_contrib = pd.read_csv(bundle / "risk_contribution.csv")
    turnover = pd.read_csv(bundle / "turnover_costs.csv", parse_dates=["Date"])
    _event_path = bundle / "leverage_signal_events.csv"
    leverage_events = (
        pd.read_csv(_event_path, parse_dates=["Date"])
        if _event_path.exists()
        else pd.DataFrame()
    )

    daily = clean_strategy_labels(daily)
    monthly = clean_strategy_labels(monthly)
    summary = clean_strategy_labels(summary)
    calendar = clean_strategy_labels(calendar)
    rolling = clean_strategy_labels(rolling)
    dd_events = clean_strategy_labels(dd_events)
    stress = clean_strategy_labels(stress)
    leverage_events = clean_strategy_labels(leverage_events)
    if "fees" in manifest:
        manifest["fees"] = {
            LEGACY_STRATEGY_RENAMES.get(key, key): value
            for key, value in manifest["fees"].items()
            if key not in DROPPED_LEGACY_STRATEGIES
        }
    for key in ("leverage_candidate",):
        candidate = manifest.get(key)
        if candidate and candidate.get("name") in LEGACY_STRATEGY_RENAMES:
            candidate["name"] = LEGACY_STRATEGY_RENAMES[candidate["name"]]
    if manifest.get("leverage_candidates"):
        _candidates = []
        for candidate in manifest["leverage_candidates"]:
            name = candidate.get("name")
            if name in DROPPED_LEGACY_STRATEGIES:
                continue
            if name in LEGACY_STRATEGY_RENAMES:
                candidate = dict(candidate)
                candidate["name"] = LEGACY_STRATEGY_RENAMES[name]
            _candidates.append(candidate)
        manifest["leverage_candidates"] = _candidates
    return (
        calendar,
        daily,
        dd_events,
        leverage_events,
        manifest,
        monthly,
        price_provenance,
        risk_contrib,
        rolling,
        stress,
        summary,
        turnover,
    )


@app.cell
def load_full_grid_leaderboard(full_grid_bundle_path):
    from pathlib import Path as _Path

    _raw = full_grid_bundle_path.value.strip().strip("'\"")
    full_grid_manifest = {}
    full_grid_leaderboard = pd.DataFrame()
    if _raw:
        _bundle = _Path(_raw).expanduser()
        if not _bundle.is_absolute():
            _bundle = ROOT / _bundle
        _manifest_path = _bundle / "manifest.json"
        _leaderboard_path = _bundle / "structural_full_grid_leaderboard.csv"
        if _manifest_path.exists() and _leaderboard_path.exists():
            with open(_manifest_path, "r", encoding="utf-8") as _handle:
                full_grid_manifest = json.load(_handle)
            full_grid_leaderboard = pd.read_csv(_leaderboard_path)
    return full_grid_leaderboard, full_grid_manifest


@app.cell
def show_manifest(manifest, price_provenance):
    allocation = pd.DataFrame([
        {"Asset": asset, "Weight": weight, "Weight (%)": f"{weight:.1%}"}
        for asset, weight in manifest["allocation"].items()
    ])
    fees = pd.DataFrame([
        {"Strategy": key, "Fee (%)": f"{value:.2%}"}
        for key, value in manifest["fees"].items()
    ])
    provenance = price_provenance or {}
    provenance_rows = [
        ("Source", provenance.get("source", manifest.get("data_source", "Unavailable"))),
        ("Price column", provenance.get("price_column", "Unavailable")),
        ("Pricing model", provenance.get("pricing_model") or "Unavailable"),
        ("Retrieved on", provenance.get("retrieved_on", "Unavailable")),
        ("Actual start", provenance.get("actual_start", manifest["date_range"].get("actual_start"))),
        ("Actual end", provenance.get("actual_end", manifest["date_range"].get("actual_end"))),
        ("Returned columns", ", ".join(provenance.get("returned_columns", [])) or "Unavailable"),
    ]
    provenance_df = pd.DataFrame(provenance_rows, columns=["Field", "Value"])
    missingness = provenance.get("missing_fraction_by_column", {})
    missingness_df = pd.DataFrame([
        {"Ticker": ticker, "Missing fraction": value}
        for ticker, value in missingness.items()
    ])
    """
    mo.vstack([
        mo.md(
            f"**Strategy:** `{manifest['strategy_id']}`  |  "
            f"**Generated:** {manifest['generated_at']}  |  "
            f"**Data:** {manifest['data_source']}  |  "
            f"**Actual range:** {manifest['date_range']['actual_start']} to "
            f"{manifest['date_range']['actual_end']}"
        ),
        mo.hstack([
            mo.ui.table(allocation, label="Target Allocation"),
            mo.ui.table(fees, label="Fee Assumptions"),
        ], gap=2),
        mo.hstack([
            mo.ui.table(provenance_df, label="Price Provenance"),
            mo.ui.table(missingness_df, label="Missingness by Column"),
        ], gap=2),
    ])"""
    return


@app.cell
def leverage_candidate_status(
    daily,
    full_grid_leaderboard,
    full_grid_manifest,
    manifest,
    summary,
):
    candidate_meta = [
        item
        for item in (manifest.get("leverage_candidates") or [])
        if item.get("name") in TARGET_LEVERAGE_STRATEGIES
    ]
    if not candidate_meta and manifest.get("leverage_candidate"):
        _candidate = manifest["leverage_candidate"]
        candidate_meta = [_candidate] if _candidate.get("name") in TARGET_LEVERAGE_STRATEGIES else []
    candidate_names = list(TARGET_LEVERAGE_STRATEGIES)
    present_names = [name for name in candidate_names if name in set(daily["Strategy"].dropna().unique())]
    full_history_summary = summary[summary["Window"] == "Full History"].copy()
    allw_overlap_summary = summary[summary["Window"] == "ALLW Overlap"].copy()
    specs = pd.concat(
        [
            pd.DataFrame(item.get("specs", [])).assign(
                Candidate=item.get("name"),
                **{
                    "Global Cap": item.get("global_cap"),
                    "Notes": item.get("notes", ""),
                },
            )
            for item in candidate_meta
            if item.get("specs")
        ],
        ignore_index=True,
    ) if candidate_meta else pd.DataFrame()

    top_cols = [
        "Rank", "Overall Pass", "Broker Profile", "SPY Rule", "GLD Rule",
        "Global Cap", "Average OOS Calmar", "Average OOS Calmar Delta",
        "Worst OOS Calmar Delta", "Worst OOS MaxDD Delta (%)",
        "Annual Calmar Improvement Years", "Annual Years Tested",
        "Average OOS Exposure (%)",
    ]
    full_grid_top = full_grid_leaderboard.head(1).copy()
    full_grid_generated = full_grid_manifest.get("generated_at", "n/a")
    missing_names = [name for name in candidate_names if name not in present_names]
    warning = (
        mo.callout(
            mo.md(
                "This result bundle is missing: "
                + ", ".join(f"`{name}`" for name in missing_names)
                + ". Regenerate the strategy comparison bundle to include every exported candidate."
            ),
            kind="warn",
        )
        if missing_names
        else mo.md("")
    )

    mo.vstack([
        mo.md(
            f"""
            ## Added Leverage Candidates

            The strategy comparison now includes `SPY 34/42 @ 30% cap`,
            `GLD 32/64 @ 30% cap`, and
            `SPY 32/42 + GLD 36/52 @ 30% cap` as strategy series.
            The full-grid validation bundle was generated `{full_grid_generated}`.
            """
        ),
        warning,
        mo.ui.table(full_grid_top[[col for col in top_cols if col in full_grid_top]], label="Full-Grid Top Leaderboard Row"),
        mo.ui.table(full_history_summary, label="Full History Summary Metrics"),
        mo.ui.table(allw_overlap_summary, label="ALLW Overlap Summary Metrics"),
        mo.ui.table(specs, label="Candidate RSI Overlay Specs"),
    ])
    return


@app.cell
def growth_controls(daily):
    available_strategies = [
        strategy
        for strategy in STRATEGY_ORDER
        if strategy in set(daily["Strategy"].dropna().unique())
    ]
    available_strategies.extend(
        strategy
        for strategy in sorted(daily["Strategy"].dropna().unique())
        if strategy not in available_strategies
    )
    growth_strategies = mo.ui.multiselect(
        options=available_strategies,
        value=available_strategies,
        label="Strategies",
        full_width=True,
    )
    full_history_scale = mo.ui.dropdown(
        options=["log", "linear"],
        value="log",
        label="Full history y-axis",
    )
    overlap_scale = mo.ui.dropdown(
        options=["linear", "log"],
        value="linear",
        label="ALLW overlap y-axis",
    )
    show_leverage_events = mo.ui.checkbox(
        value=True,
        label="Show SPY/GLD leverage entry and exit markers",
    )
    mo.vstack([
        mo.md("## Overview: Growth of Money"),
        mo.hstack([full_history_scale, overlap_scale], gap=2),
        show_leverage_events,
        growth_strategies,
    ])
    return (
        available_strategies,
        full_history_scale,
        growth_strategies,
        overlap_scale,
        show_leverage_events,
    )


@app.cell
def plot_growth_cell(
    available_strategies,
    daily,
    full_history_scale,
    growth_strategies,
    leverage_events,
    overlap_scale,
    show_leverage_events,
):
    selected_strategies = growth_strategies.value or available_strategies
    mo.as_html(plot_growth(
        daily,
        strategies=selected_strategies,
        full_history_scale=full_history_scale.value,
        leverage_events=leverage_events,
        overlap_scale=overlap_scale.value,
        show_leverage_events=show_leverage_events.value,
    ))
    return


@app.cell
def plot_drawdowns_cell(daily, dd_events):
    mo.md("## Core Risk Story: Drawdowns")
    mo.vstack([
        mo.as_html(plot_drawdowns(daily)),
        mo.ui.table(dd_events, label="Worst Drawdown Events"),
    ])
    return


@app.cell
def plot_calendar_profile_cell(calendar):
    mo.md("## Calendar-Year Profile")
    plot_calendar_profile(calendar)
    return


@app.cell
def plot_rolling_behaviour_cell(rolling):
    mo.md("## Rolling Behaviour")
    plot_rolling_behaviour(rolling)
    return


@app.cell
def plot_monthly_returns_cell(monthly):
    mo.md("## Return Distribution and Monthly Heatmaps")
    plot_monthly_returns(monthly)
    return


@app.cell
def plot_risk_diagnostics_cell(summary):
    mo.md("## Institutional Risk Diagnostics")
    mo.vstack([
        mo.as_html(plot_risk_diagnostics(summary)),
        mo.ui.table(summary, label="Summary Metrics"),
    ])
    return


@app.cell
def plot_implementation_realism_cell(risk_contrib, stress, turnover):
    mo.md("## Implementation Realism")
    mo.vstack([
        mo.as_html(plot_implementation_realism(risk_contrib, turnover)),
        mo.hstack([
            mo.ui.table(risk_contrib, label="Risk Contribution"),
            mo.ui.table(turnover.tail(24), label="Recent Turnover and Costs"),
        ], gap=2),
        mo.ui.table(stress, label="Stress Period Metrics"),
    ])
    return


@app.cell
def load_rebalance_bundle(rebalance_bundle_path):
    from pathlib import Path as _Path

    _raw = rebalance_bundle_path.value.strip().strip("'\"")
    rebalance_bundle_dir = None
    rebalance_manifest = {}
    rebalance_summary = pd.DataFrame()
    rebalance_values = pd.DataFrame()
    if _raw:
        _bundle = _Path(_raw).expanduser()
        if not _bundle.is_absolute():
            _bundle = ROOT / _bundle
        _config_path = _bundle / "run_config.json"
        _summary_path = _bundle / "threshold_summary.csv"
        _values_path = _bundle / "threshold_values.csv"
        if _config_path.exists() and _summary_path.exists():
            rebalance_bundle_dir = _bundle
            with open(_config_path, "r", encoding="utf-8") as _handle:
                rebalance_manifest = json.load(_handle)
            rebalance_summary = pd.read_csv(_summary_path)
            if _values_path.exists():
                rebalance_values = pd.read_csv(
                    _values_path, parse_dates=["Date"]
                ).set_index("Date")
    return (
        rebalance_bundle_dir,
        rebalance_manifest,
        rebalance_summary,
        rebalance_values,
    )


@app.cell
def rebalancing_policy_section(
    rebalance_bundle_dir,
    rebalance_manifest,
    rebalance_summary,
):
    if rebalance_bundle_dir is None or rebalance_summary.empty:
        blocks = [
            mo.md("## Rebalancing Policy"),
            mo.callout(
                mo.md(
                    "No rebalance thresholds bundle loaded. Generate one with "
                    "`python -m research.rebalance_thresholds`, then paste the path above."
                ),
                kind="warn",
            ),
        ]
    else:
        summary_view = rebalance_summary.copy()
        summary_view.insert(
            0,
            "Policy Label",
            summary_view.apply(
                lambda row: POLICY_SERIES_LABELS.get(
                    f"{row['Policy']} | {row['Rebalance Action']}", row["Policy"]
                ),
                axis=1,
            ),
        )
        summary_view = summary_view.sort_values(
            by="Policy Label",
            key=lambda col: col.map(
                lambda label: POLICY_SERIES_ORDER.index(label)
                if label in POLICY_SERIES_ORDER
                else len(POLICY_SERIES_ORDER)
            ),
        )
        summary_cols = [
            "Policy Label",
            "CAGR (%)",
            "Max Drawdown (%)",
            "Calmar",
            "Sharpe",
            "Rebalance Count",
            "Avg Annual Turnover $",
            "Max Relative Drift Before (%)",
        ]
        summary_view = summary_view[[col for col in summary_cols if col in summary_view.columns]]

        blocks = [
            mo.md("## Rebalancing Policy"),
            mo.md(
                f"Threshold policies generated over "
                f"{rebalance_manifest.get('start_date', '?')} to "
                f"{rebalance_manifest.get('end_date', '?')}."
            ),
        ]
        growth_png = rebalance_bundle_dir / "threshold_growth.png"
        if growth_png.exists():
            blocks.append(mo.image(str(growth_png), alt="Threshold policies growth"))
        blocks.append(mo.ui.table(summary_view, label="Threshold Policy Summary"))

        overlap_png = rebalance_bundle_dir / "threshold_allw_overlap.png"
        if overlap_png.exists():
            blocks.append(mo.md("### ALLW overlap window (daily resolution)"))
            blocks.append(mo.image(str(overlap_png), alt="Threshold policies ALLW overlap"))

        for window_png in sorted(rebalance_bundle_dir.glob("threshold_rolling_*.png")):
            blocks.append(mo.image(str(window_png), alt=window_png.stem))

    mo.vstack(blocks)
    return


if __name__ == "__main__":
    app.run()
