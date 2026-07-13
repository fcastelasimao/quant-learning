"""Metric-by-metric research lab for TQQQ signal discovery.

Run with: marimo edit notebooks/06_metric_research.py
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
    from signal_diagnostics import (
        bh_qvalues,
        metric_forward_profile,
        metric_redundancy_table,
        multi_horizon_credibility_report,
        pairwise_redundancy_table,
        quantile_monotonicity,
        signal_credibility_table,
        train_val_credibility_report,
    )
    from viz import (
        edge_heatmap_figure,
        metric_forward_profile_figure,
        pairwise_redundancy_heatmap_figure,
        quantile_shape_small_multiples,
    )

    from quantcore import config as _qc_config
    DATA_DIR = _qc_config.data_dir()
    HORIZONS = (1, 2, 5, 10, 20)
    return (
        DATA_DIR,
        HORIZONS,
        REGISTRY,
        SYMBOLS_ALL,
        edge_heatmap_figure,
        load_panel,
        metric_forward_profile,
        metric_forward_profile_figure,
        metric_redundancy_table,
        mo,
        multi_horizon_credibility_report,
        np,
        pairwise_redundancy_heatmap_figure,
        pairwise_redundancy_table,
        pd,
        quantile_shape_small_multiples,
        signal_credibility_table,
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
    # Metric Research Lab

    **How to read this notebook:**

    - The notebook is structured leaderboard-first, microscope second.
    - Evidence comes only from **train** (≤ 2017) and **val** (2018–2021). The frozen **test** window (2022+) is never loaded here.
    - A metric is useful if its bull-minus-bear edge is consistent across both splits and across multiple horizons — not just in one split or one horizon.
    - Statistical significance is reported using **BH-corrected q-values** (Benjamini-Hochberg FDR), which account for testing many metrics simultaneously.

    Scroll down: Global Filters → Today's Votes → Leaderboard → Edge Heatmap → Quantile Shape → Redundancy Heatmap → Single-Metric Microscope.
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
    include_watch_ui = mo.ui.checkbox(value=True, label="Include watch metrics")

    mo.vstack([
        mo.md("### Global Filters"),
        mo.md(
            "- **Min directional obs (val):** minimum count of *both* bull (+1) and bear (−1) votes "
            "in the validation window. Filters out metrics that fire too one-sidedly to measure a "
            "reliable directional edge. Default = 30.\n"
            "- **Families:** which strategy families to show. `physics_*` families are excluded by "
            "default — they are hypothesis-stage metrics that fire too rarely to be evaluated.\n"
            "- **Include watch metrics:** `watch` status means not yet promoted to production voting."
        ),
        mo.hstack([min_obs_ui, include_watch_ui]),
        family_ui,
    ])
    return family_ui, include_watch_ui, min_obs_ui


@app.cell
def _today_votes(REGISTRY, family_ui, mo, np, panel, pd):
    _rows = []
    for _name, _m in REGISTRY.items():
        if _m.family not in family_ui.value:
            continue
        _vals = _m.compute(panel)
        _votes = _m.vote(_vals)
        _valid = _vals.dropna()
        _latest_date = _valid.index.max() if not _valid.empty else None
        _latest_vote = int(_votes.loc[_latest_date]) if _latest_date is not None else 0
        _rows.append({
            "metric": _name,
            "family": _m.family,
            "status": _m.status,
            "latest_date": _latest_date,
            "latest_value": float(_valid.iloc[-1]) if not _valid.empty else np.nan,
            "latest_vote": _latest_vote,
        })
    _today_df = (
        pd.DataFrame(_rows)
        .sort_values(["family", "metric"])
        .reset_index(drop=True)
    )
    mo.vstack([
        mo.md("## Today's Votes"),
        mo.md(
            "Current state of every metric in the selected families. "
            "Columns: **metric** = registry name; **family** = strategy family (colour-coded throughout); "
            "**status** = `voting` (used in production) or `watch` (under evaluation); "
            "**latest_date** = last date the metric has a value (some need warmup periods); "
            "**latest_value** = raw metric output on that date; "
            "**latest_vote** = current directional signal: +1 buy, 0 neutral, −1 sell."
        ),
        _today_df,
    ])
    return


@app.cell
def _leaderboard(
    family_ui,
    include_watch_ui,
    min_obs_ui,
    mo,
    multi_horizon_credibility_report,
    panel,
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

    leaderboard = _full_report[
        _full_report["family"].isin(family_ui.value)
        & _full_report["passes_min_obs"]
    ].reset_index(drop=True)

    _view = leaderboard.copy()
    for _col in _view.select_dtypes(include=["float"]).columns:
        _view[_col] = _view[_col].round(4)

    mo.vstack([
        mo.md("## Leaderboard"),
        mo.md(
            "One row per metric. Sorted by `credibility_score` (higher = more reliable). "
            "All forward-return targets are **TQQQ tradable open-to-open** (signal at close[t], "
            "earliest trade at open[t+1]). Primary horizon is **5 trading days**.\n\n"
            "**Column glossary:**\n"
            "- `edge_train_5d / edge_val_5d` — mean forward 5d return on bull days minus bear days, "
            "in basis points (bps). Positive = bull votes see higher returns than bear votes.\n"
            "- `raw_ic_*_5d` — Spearman rank correlation between raw metric values and 5d forward returns.\n"
            "- `raw_ic_p_val_5d` — p-value for the val IC (two-sided). "
            "`raw_ic_q_val_5d` — same, BH-FDR-corrected across this filtered set of metrics (use this, not p).\n"
            "- `vote_ic_val_5d` — Spearman IC but between votes (−1/0/+1) and 5d returns.\n"
            "- `monotonicity_val_5d` — Spearman corr between the 10 decile-bucket indices and their "
            "mean 5d return. +1 = perfect upward slope; −1 = perfect downward slope; ~0 = noisy/U-shaped.\n"
            "- `n_horizons_edge_sign_agree` — out of 5 horizons (1, 2, 5, 10, 20d), how many have the "
            "same bull-minus-bear sign in both train and val. 5/5 = very stable across time scales.\n"
            "- `credibility_score` — composite: |val edge| × sign-agreement multiplier × IC multiplier "
            "× obs-sufficiency multiplier. Not a return forecast — a reliability ranking.\n"
            "- `credibility_label` — `promising` = positive score + sign agrees + enough obs; "
            "`mixed` = sign disagrees or negative score; `weak` = insufficient evidence."
        ),
        mo.md("Sorted by `credibility_score`. `raw_ic_q_val_5d` is BH-corrected across this filtered set."),
        _view,
    ])
    return (leaderboard,)


@app.cell
def _heatmap_controls(leaderboard, mo):
    _max_n = max(5, len(leaderboard))
    heatmap_top_n_ui = mo.ui.slider(
        start=5, stop=_max_n, step=1, value=_max_n,
        label=f"Limit to top N (of {_max_n} passing metrics)",
    )
    mo.vstack([
        mo.md("## Edge Heatmap"),
        mo.md(
            "Bull-minus-bear edge in bps at each forward-return horizon (1, 2, 5, 10, 20 days), "
            "computed on the **val split**. "
            "**Red** = bull votes outperformed bear votes at that horizon. "
            "**Blue** = the reverse. "
            "**✓ overlay** = the sign of the edge at that horizon is the same in train and val "
            "(a stability check — ✓ on every cell of a row means the metric agrees with itself "
            "across all time scales). "
            "Metrics are sorted by leaderboard rank (best credibility at top)."
        ),
        heatmap_top_n_ui,
    ])
    return (heatmap_top_n_ui,)


@app.cell
def _heatmap(
    edge_heatmap_figure,
    heatmap_top_n_ui,
    leaderboard,
    mo,
    np,
    panel,
    pd,
    signal_credibility_table,
    warnings,
):
    _top_metrics = leaderboard.head(int(heatmap_top_n_ui.value))["metric"].tolist()

    _long_frames = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _h in (1, 2, 5, 10, 20):
            _val_ct = signal_credibility_table(
                panel,
                split="val",
                horizons=(_h,),
                target_symbol="TQQQ",
                target_kind="tradable_open",
                metric_names=_top_metrics,
                include_watch=True,
            )
            _train_ct = signal_credibility_table(
                panel,
                split="train",
                horizons=(_h,),
                target_symbol="TQQQ",
                target_kind="tradable_open",
                metric_names=_top_metrics,
                include_watch=True,
            )
            _merged = _val_ct.rename(columns={"bull_minus_bear_bps": "bull_minus_bear_bps_val"}).merge(
                _train_ct[["metric", "bull_minus_bear_bps"]].rename(
                    columns={"bull_minus_bear_bps": "bull_minus_bear_bps_train"}
                ),
                on="metric",
                how="left",
            )
            _merged["edge_sign_agrees"] = (
                np.sign(_merged["bull_minus_bear_bps_train"].fillna(0))
                == np.sign(_merged["bull_minus_bear_bps_val"].fillna(0))
            )
            _long_frames.append(
                _merged[["metric", "horizon", "bull_minus_bear_bps_val", "edge_sign_agrees", "family"]].copy()
            )

    _long_df = pd.concat(_long_frames, ignore_index=True)
    _cat = pd.CategoricalDtype(categories=_top_metrics, ordered=True)
    _long_df["metric"] = _long_df["metric"].astype(_cat)
    _long_df = _long_df.sort_values("metric").reset_index(drop=True)

    _fig_heatmap = edge_heatmap_figure(
        _long_df,
        value_col="bull_minus_bear_bps_val",
        agree_col="edge_sign_agrees",
        top_n=int(heatmap_top_n_ui.value),
    )
    mo.vstack([_fig_heatmap])
    return


@app.cell
def _quantile_controls(leaderboard, mo):
    _max_sm = min(24, max(4, len(leaderboard)))
    sm_top_n_ui = mo.ui.slider(
        start=4, stop=_max_sm, step=1, value=min(12, _max_sm),
        label="Top N metrics to show",
    )
    mo.vstack([
        mo.md("## Quantile Shape — Top Metrics (val split, horizon 5d)"),
        mo.md(
            "Each subplot covers one metric. "
            "Its raw values are sorted and split into 10 equal-count buckets "
            "(bucket 1 = the 10% of days with the *lowest* raw value, bucket 10 = the highest 10%). "
            "The y-axis shows the mean forward 5-day TQQQ return (in bps) for days in each bucket. "
            "A clean upward or downward slope means raw metric values predict returns monotonically — "
            "this is what you want. A flat line or U-shape means the vote thresholds are probably "
            "capturing the signal better than a linear mapping would."
        ),
        sm_top_n_ui,
    ])
    return (sm_top_n_ui,)


@app.cell
def _quantile_small_multiples(
    leaderboard,
    metric_forward_profile,
    mo,
    panel,
    quantile_shape_small_multiples,
    sm_top_n_ui,
    warnings,
):
    _top_metrics_q = leaderboard.head(int(sm_top_n_ui.value))["metric"].tolist()
    _profiles = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _name in _top_metrics_q:
            _profiles[_name] = metric_forward_profile(
                panel,
                _name,
                split="val",
                horizons=(5,),
                bins=10,
            )
    _fig_sm = quantile_shape_small_multiples(_profiles, horizon=5, ncols=4)
    mo.vstack([_fig_sm])
    return


@app.cell
def _redundancy_heatmap_cell(
    REGISTRY,
    family_ui,
    include_watch_ui,
    mo,
    pairwise_redundancy_heatmap_figure,
    pairwise_redundancy_table,
    panel,
    warnings,
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _names = [
            n for n, m in REGISTRY.items()
            if m.family in family_ui.value
            and (include_watch_ui.value or m.status == "voting")
        ]
        _corr = pairwise_redundancy_table(
            panel,
            split="train+val",
            metric_names=_names,
            include_watch=include_watch_ui.value,
            kind="raw",
        )
        _family_map = {n: REGISTRY[n].family for n in _corr.index}
        _fig_redund = pairwise_redundancy_heatmap_figure(_corr, family_map=_family_map)

    mo.vstack([
        mo.md("## Redundancy Heatmap (all metrics)"),
        mo.md(
            "Pairwise Spearman correlation of raw metric values computed on **train+val**. "
            "**Red** = metrics measuring the same thing (high positive correlation — consider dropping one). "
            "**Blue** = metrics moving in opposite directions (could be used as a pair or one may be the "
            "inverse of the other). "
            "Rows and columns are grouped by strategy **family** (colour prefix on labels). "
            "A hot diagonal block within one family is expected — it just means the family is internally "
            "consistent."
        ),
        _fig_redund,
    ])
    return


@app.cell
def _microscope_intro(mo):
    mo.md("""
    ---
    ## Single-Metric Microscope
    Deep dive on one metric. Defaults to the leaderboard top. Use this to inspect the full
    time-series, rolling IC stability, vote-bucket returns, and raw-value quantile shape
    for any metric in the registry.
    """)
    return


@app.cell
def _microscope_controls(REGISTRY, leaderboard, mo):
    _leaderboard_metrics = leaderboard["metric"].tolist()
    _rest = [m for m in REGISTRY if m not in _leaderboard_metrics]
    _all_metric_options = _leaderboard_metrics + _rest
    _default_metric = _leaderboard_metrics[0] if _leaderboard_metrics else _all_metric_options[0]

    metric_ui = mo.ui.dropdown(
        options=_all_metric_options,
        value=_default_metric,
        label="Metric",
    )
    split_ui = mo.ui.dropdown(
        options=["train", "val", "train+val"],
        value="val",
        label="Split",
    )
    microscope_horizon_ui = mo.ui.dropdown(
        options=[1, 2, 5, 10, 20],
        value=5,
        label="Horizon (days)",
    )
    bins_ui = mo.ui.slider(start=4, stop=20, step=1, value=10, label="Raw-value buckets")

    mo.vstack([
        mo.md("### Microscope controls"),
        mo.md(
            "- **Split:** which date window to compute on. "
            "`train` = rows on or before 2017-12-31; "
            "`val` = 2018-01-01 to 2021-12-31; "
            "`train+val` = both combined. "
            "The test window (2022 onward) is intentionally unavailable here to avoid lookahead bias.\n"
            "- **Horizon (days):** forward-return horizon used in the IC and vote-bucket tables below. "
            "The full multi-horizon IC table is always shown regardless of this setting.\n"
            "- **Raw-value buckets:** how many equal-count quantile bins to split the raw metric value "
            "into for the Raw-Value Buckets table. 10 = deciles. More bins = finer resolution but noisier estimates."
        ),
        mo.hstack([metric_ui, split_ui, microscope_horizon_ui, bins_ui]),
    ])
    return bins_ui, metric_ui, microscope_horizon_ui, split_ui


@app.cell
def _profile(
    HORIZONS,
    bins_ui,
    metric_forward_profile,
    metric_ui,
    panel,
    split_ui,
    warnings,
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        profile = metric_forward_profile(
            panel,
            metric_ui.value,
            split=split_ui.value,
            horizons=HORIZONS,
            target_symbol="TQQQ",
            target_kind="tradable_open",
            bins=int(bins_ui.value),
        )
    return (profile,)


@app.cell
def _profile_summary(bins_ui, microscope_horizon_ui, mo, profile):
    ic = profile.ic_table.copy()
    vote = profile.vote_bucket_table.loc[
        profile.vote_bucket_table["horizon"] == int(microscope_horizon_ui.value)
    ].copy()
    quantile = profile.quantile_table.loc[
        profile.quantile_table["horizon"] == int(microscope_horizon_ui.value)
    ].copy()

    for _df in [ic, vote, quantile]:
        for _col in _df.select_dtypes(include=["float"]).columns:
            _df[_col] = _df[_col].round(4)

    _warning_text = "\n".join(f"- {w}" for w in profile.warnings) if profile.warnings else "No warnings."
    mo.vstack([
        mo.md("## Selected Metric Summary"),
        mo.md(f"Latest: `{profile.latest}`"),
        mo.md(f"Warnings:\n{_warning_text}"),
        mo.md("### IC Table"),
        mo.md(
            "Spearman rank correlation between raw metric values (or votes) and forward TQQQ returns "
            "at each horizon. `raw_ic` uses the continuous metric value; `vote_ic` uses the discretised "
            "vote (−1/0/+1). `raw_ic_p` is the two-sided p-value — treat as indicative only; "
            "use the BH-corrected q-values in the leaderboard for rigorous significance testing."
        ),
        ic,
        mo.md("### Vote Buckets"),
        mo.md(
            f"Mean forward {microscope_horizon_ui.value}d TQQQ return for each vote bucket at the selected horizon. "
            "`bull_minus_bear` (row label) is the headline edge metric in bps: "
            "positive means bullish votes see higher returns than bearish votes."
        ),
        vote,
        mo.md("### Raw-Value Buckets"),
        mo.md(
            f"Like vote buckets, but the metric's raw values are split into {int(bins_ui.value)} equal-count quantile bins "
            "rather than by vote threshold. Use this to check whether the vote thresholds are well-placed — "
            "if returns are monotone across the buckets, the threshold captures the extremes correctly. "
            "If the best-returning bucket is in the middle, the current thresholds may be cutting off signal."
        ),
        quantile,
    ])
    return


@app.cell
def _profile_plot(
    metric_forward_profile_figure,
    microscope_horizon_ui,
    panel,
    profile,
):
    fig = metric_forward_profile_figure(
        panel,
        profile,
        horizon=int(microscope_horizon_ui.value),
    )
    fig
    return


@app.cell
def _redundancy(
    metric_redundancy_table,
    metric_ui,
    mo,
    panel,
    split_ui,
    warnings,
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        redundancy = metric_redundancy_table(
            panel,
            metric_ui.value,
            split=split_ui.value,
            include_watch=True,
        )
    redundancy_view = redundancy.head(15).copy()
    for _col in redundancy_view.select_dtypes(include=["float"]).columns:
        redundancy_view[_col] = redundancy_view[_col].round(4)
    mo.vstack([
        mo.md("## Redundancy Check (single metric)"),
        mo.md(
            f"How correlated is **{metric_ui.value}** with every other metric in the registry? "
            "`raw_corr` = Spearman correlation between raw values; `vote_corr` = agreement of votes. "
            "A metric is flagged `is_redundant` if |raw_corr| ≥ 0.80 or |vote_corr| ≥ 0.80 — "
            "meaning it provides largely the same information and one of the two could be dropped. "
            "Top 15 shown."
        ),
        redundancy_view,
    ])
    return


if __name__ == "__main__":
    app.run()
