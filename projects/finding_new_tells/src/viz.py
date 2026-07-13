"""Visualization for the TQQQ research framework.

Public functions:
  inspect(metric_name, panel, regime_states=None)
      → 4-panel plotly figure: time-series, histogram, ACF (matplotlib), hex-bin vs fwd return
  dashboard(panel, date, regime_state, regime_probs, regime_states=None)
      → plotly figure with per-metric percentile bars + vote chips + regime header
  backtest_report(result)
      → 5-row stacked plotly: equity, DD depth, DD duration, p_buy/hold/sell, position
  prepare_indicator_workbench(...)
      → aligned data, rule masks, and rule summary for the indicator workbench
  indicator_workbench(...)
      → TradingView-style stacked price + indicator figure
  metric_forward_profile_figure(...)
      → metric research plot: price, metric, votes, IC, buckets, event paths
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from backtest import BacktestResult

# ---------------------------------------------------------------------------
# Lazy imports for optional heavy deps
# ---------------------------------------------------------------------------

def _go():
    import plotly.graph_objects as go
    return go


def _make_subplots(**kwargs):
    from plotly.subplots import make_subplots
    return make_subplots(**kwargs)


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

_FAMILY_COLORS = {
    "trend":       "#2196F3",
    "mean_rev":    "#FF9800",
    "vol":         "#9C27B0",
    "volatility":  "#9C27B0",
    "lev_etf":     "#F44336",
    "leveraged":   "#F44336",
    "cross":       "#4CAF50",
    "cross_asset": "#4CAF50",
    "micro":       "#795548",
    "microstructure": "#795548",
    "calendar":    "#607D8B",
    "watch":       "#9E9E9E",
}

_STATE_COLORS = [
    "rgba(33,150,243,0.15)",   # 0 strong_bull
    "rgba(76,175,80,0.15)",    # 1 weak_bull
    "rgba(158,158,158,0.15)",  # 2 sideways
    "rgba(255,152,0,0.15)",    # 3 weak_bear
    "rgba(244,67,54,0.15)",    # 4 strong_bear
]

DEFAULT_WORKBENCH_METRICS = [
    "qqq_20d_slope",
    "qqq_rsi2",
    "qqq_rv_20d",
    "vix_term_structure",
    "qqq_spy_ratio_slope",
]


@dataclass(frozen=True)
class IndicatorWorkbenchData:
    """Prepared data for the visual indicator workbench."""

    panel: pd.DataFrame
    price: pd.Series
    qqq: pd.Series | None
    metrics: dict[str, pd.Series]
    skipped: dict[str, str]
    rule_metric: str | None
    buy_mask: pd.Series
    sell_mask: pd.Series
    summary: pd.DataFrame


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _add_regime_shading(fig, regime_states: pd.Series, row: int = 1, col: int = 1) -> None:
    """Add colored background bands per regime state to a plotly subplot."""
    go = _go()
    if regime_states is None or regime_states.empty:
        return
    prev_state = regime_states.iloc[0]
    start_date = regime_states.index[0]
    for date, state in regime_states.items():
        if state != prev_state:
            fig.add_vrect(
                x0=str(start_date), x1=str(date),
                fillcolor=_STATE_COLORS[prev_state % len(_STATE_COLORS)],
                opacity=1.0, layer="below", line_width=0,
                row=row, col=col,
            )
            start_date = date
            prev_state = state
    fig.add_vrect(
        x0=str(start_date), x1=str(regime_states.index[-1]),
        fillcolor=_STATE_COLORS[prev_state % len(_STATE_COLORS)],
        opacity=1.0, layer="below", line_width=0,
        row=row, col=col,
    )


def _fwd_return(close: pd.Series, n: int = 5) -> pd.Series:
    return close.pct_change(n, fill_method=None).shift(-n)


def _rgba(hex_str: str, a: float) -> str:
    r, g, b = (int(hex_str[i:i+2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{a})"


def _date_slice(panel: pd.DataFrame, start: pd.Timestamp | str | None, end: pd.Timestamp | str | None) -> pd.DataFrame:
    start_ts = pd.Timestamp(start) if start is not None else None
    end_ts = pd.Timestamp(end) if end is not None else None
    return panel.loc[start_ts:end_ts]


def _mask_starts(mask: pd.Series) -> pd.DatetimeIndex:
    if mask.empty:
        return pd.DatetimeIndex([])
    starts = mask.fillna(False) & ~mask.fillna(False).shift(1, fill_value=False)
    return pd.DatetimeIndex(mask.index[starts])


def _mask_segments(mask: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    segments: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    if mask.empty:
        return segments
    flags = mask.fillna(False).to_numpy()
    idx = mask.index
    i = 0
    while i < len(flags):
        if not flags[i]:
            i += 1
            continue
        start = i
        while i < len(flags) and flags[i]:
            i += 1
        end = i if i < len(flags) else len(flags) - 1
        segments.append((idx[start], idx[end]))
    return segments


def _candidate_rule_summary(price: pd.Series, buy_mask: pd.Series, sell_mask: pd.Series) -> pd.DataFrame:
    fwd_5d = price.pct_change(5, fill_method=None).shift(-5)
    fwd_20d = price.pct_change(20, fill_method=None).shift(-20)
    rows = []
    for action, mask in [("buy", buy_mask), ("sell", sell_mask)]:
        starts = _mask_starts(mask)
        rows.append({
            "action": action,
            "zones": len(starts),
            "avg_fwd_5d_%": round(float(fwd_5d.reindex(starts).mean() * 100), 3) if len(starts) else np.nan,
            "avg_fwd_20d_%": round(float(fwd_20d.reindex(starts).mean() * 100), 3) if len(starts) else np.nan,
        })
    return pd.DataFrame(rows).set_index("action")


def _mask_sample(mask: pd.Series, max_points: int = 750) -> pd.Series:
    hits = mask[mask.fillna(False)]
    if len(hits) <= max_points:
        return mask
    sampled_idx = hits.index[np.linspace(0, len(hits) - 1, max_points, dtype=int)]
    sampled = pd.Series(False, index=mask.index, dtype=bool)
    sampled.loc[sampled_idx] = True
    return sampled


def _indexed_to_100(series: pd.Series) -> pd.Series:
    clean = series.dropna()
    if clean.empty:
        return series
    base = clean.iloc[0]
    if base == 0 or np.isnan(base):
        return series
    return series / base * 100.0


def _robust_zscore(series: pd.Series) -> pd.Series:
    clean = series.dropna()
    if clean.empty:
        return series
    median = clean.median()
    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0 or np.isnan(iqr):
        std = clean.std()
        scale = std if std and not np.isnan(std) else 1.0
    else:
        scale = iqr / 1.349
    return ((series - median) / scale).clip(-5, 5)


def _robust_zvalue(series: pd.Series, value: float) -> float:
    clean = series.dropna()
    if clean.empty:
        return value
    median = clean.median()
    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0 or np.isnan(iqr):
        std = clean.std()
        scale = std if std and not np.isnan(std) else 1.0
    else:
        scale = iqr / 1.349
    return float(np.clip((value - median) / scale, -5, 5))


def _apply_threshold(
    target: pd.Series,
    *,
    threshold: float | None,
    comparison: str,
    action: str,
    buy_mask: pd.Series,
    sell_mask: pd.Series,
) -> None:
    if threshold is None:
        return
    hits = target <= threshold if comparison == "low" else target >= threshold
    if action == "buy":
        buy_mask.loc[hits.fillna(False)] = True
    elif action == "sell":
        sell_mask.loc[hits.fillna(False)] = True


# ---------------------------------------------------------------------------
# indicator_workbench() — TradingView-style visual research surface
# ---------------------------------------------------------------------------

def prepare_indicator_workbench(
    panel: pd.DataFrame,
    selected_metrics: list[str] | tuple[str, ...] | None = None,
    *,
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
    rule_metric: str | None = None,
    low_threshold: float | None = None,
    high_threshold: float | None = None,
    low_action: str = "buy",
    high_action: str = "sell",
) -> IndicatorWorkbenchData:
    """Prepare aligned metric series and threshold-zone summaries.

    Metrics are computed on the full panel before date slicing so rolling
    indicators keep their proper warmup history.
    """
    from metrics import REGISTRY

    if "TQQQ_close" not in panel.columns:
        raise KeyError("Panel must contain TQQQ_close for the indicator workbench.")

    metric_names = list(selected_metrics or DEFAULT_WORKBENCH_METRICS)
    if rule_metric and rule_metric not in metric_names:
        metric_names.append(rule_metric)
    metric_names = list(dict.fromkeys(metric_names))

    filtered = _date_slice(panel, start, end)
    price = panel["TQQQ_close"]
    price_view = price.reindex(filtered.index)
    qqq = panel["QQQ_close"].reindex(filtered.index) if "QQQ_close" in panel.columns else None

    metrics: dict[str, pd.Series] = {}
    skipped: dict[str, str] = {}
    for name in metric_names:
        metric = REGISTRY.get(name)
        if metric is None:
            skipped[name] = "not registered"
            continue
        try:
            series = metric.compute(panel)
        except Exception as exc:
            skipped[name] = f"compute failed: {exc}"
            continue
        if series.empty:
            skipped[name] = "empty series"
            continue
        aligned = series.reindex(filtered.index)
        if aligned.dropna().empty:
            skipped[name] = "no values in selected date range"
            continue
        metrics[name] = aligned

    buy_mask = pd.Series(False, index=filtered.index, dtype=bool)
    sell_mask = pd.Series(False, index=filtered.index, dtype=bool)
    active_rule_metric = rule_metric if rule_metric in metrics else None

    if active_rule_metric is not None:
        target = metrics[active_rule_metric]
        _apply_threshold(
            target,
            threshold=low_threshold,
            comparison="low",
            action=low_action,
            buy_mask=buy_mask,
            sell_mask=sell_mask,
        )
        _apply_threshold(
            target,
            threshold=high_threshold,
            comparison="high",
            action=high_action,
            buy_mask=buy_mask,
            sell_mask=sell_mask,
        )

    summary = _candidate_rule_summary(price, buy_mask, sell_mask)
    return IndicatorWorkbenchData(
        panel=filtered,
        price=price_view,
        qqq=qqq,
        metrics=metrics,
        skipped=skipped,
        rule_metric=active_rule_metric,
        buy_mask=buy_mask,
        sell_mask=sell_mask,
        summary=summary,
    )


def indicator_workbench(
    panel: pd.DataFrame,
    selected_metrics: list[str] | tuple[str, ...] | None = None,
    *,
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
    show_qqq: bool = True,
    rule_metric: str | None = None,
    low_threshold: float | None = None,
    high_threshold: float | None = None,
    low_action: str = "buy",
    high_action: str = "sell",
    show_rules: bool = False,
) -> tuple["go.Figure", pd.DataFrame, dict[str, str]]:
    """Return indexed price + raw indicator figure, rule summary, and skipped metrics."""
    from metrics import REGISTRY

    go = _go()
    data = prepare_indicator_workbench(
        panel,
        selected_metrics,
        start=start,
        end=end,
        rule_metric=rule_metric,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        low_action=low_action,
        high_action=high_action,
    )

    metric_names = list(data.metrics)
    rows = 2 if metric_names else 1
    row_heights = [0.62, 0.38] if metric_names else [1.0]
    fig = _make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        vertical_spacing=0.04,
        subplot_titles=["TQQQ / QQQ indexed price", "Metrics (raw values)"] if metric_names else ["TQQQ / QQQ indexed price"],
    )

    fig.add_trace(go.Scatter(
        x=data.price.index,
        y=_indexed_to_100(data.price).values,
        name="TQQQ indexed",
        line_color="#F44336",
    ), row=1, col=1)
    if show_qqq and data.qqq is not None:
        fig.add_trace(go.Scatter(
            x=data.qqq.index,
            y=_indexed_to_100(data.qqq).values,
            name="QQQ indexed",
            line_color="#9E9E9E",
            opacity=0.55,
        ), row=1, col=1)

    if show_rules:
        for mask, color, label in [
            (data.buy_mask, "rgba(76,175,80,0.18)", "buy zone"),
            (data.sell_mask, "rgba(244,67,54,0.16)", "sell zone"),
        ]:
            segments = _mask_segments(mask)
            for start_date, end_date in segments[:250]:
                fig.add_vrect(
                    x0=str(start_date),
                    x1=str(end_date),
                    fillcolor=color,
                    opacity=1.0,
                    layer="below",
                    line_width=0,
                    row=1,
                    col=1,
                )
            if mask.any():
                marker_mask = _mask_sample(mask)
                fig.add_trace(go.Scatter(
                    x=data.price.index[marker_mask],
                    y=_indexed_to_100(data.price).loc[marker_mask].values,
                    mode="markers",
                    marker={"size": 7, "color": color.replace("0.18", "0.8").replace("0.16", "0.8")},
                    name=label,
                    showlegend=True,
                ), row=1, col=1)

    for name in metric_names:
        series = data.metrics[name]
        metric = REGISTRY.get(name)
        color = _FAMILY_COLORS.get(metric.family if metric else "watch", "#607D8B")
        fig.add_trace(go.Scatter(
            x=series.index,
            y=series.values,
            name=name,
            line_color=color,
        ), row=2, col=1)

        if show_rules and data.rule_metric == name:
            if low_threshold is not None:
                fig.add_hline(y=_robust_zvalue(series, low_threshold), line_dash="dot", line_color="#4CAF50", row=2, col=1)
            if high_threshold is not None:
                fig.add_hline(y=_robust_zvalue(series, high_threshold), line_dash="dot", line_color="#F44336", row=2, col=1)
            for mask, color_name, label in [
                (data.buy_mask, "#4CAF50", "rule buy"),
                (data.sell_mask, "#F44336", "rule sell"),
            ]:
                if mask.any():
                    marker_mask = _mask_sample(mask)
                    fig.add_trace(go.Scatter(
                        x=series.index[marker_mask],
                        y=series.loc[marker_mask].values,
                        mode="markers",
                        marker={"size": 6, "color": color_name},
                        name=label,
                        showlegend=False,
                    ), row=2, col=1)

    if metric_names:
        fig.add_hline(y=0, line_dash="dot", line_color="#BDBDBD", row=2, col=1)

    fig.update_layout(
        title_text="TQQQ Indicator Workbench",
        height=780 if metric_names else 560,
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.05},
        margin={"l": 70, "r": 30, "t": 80, "b": 70},
    )
    fig.update_yaxes(title_text="Indexed to 100 (log)", type="log", row=1, col=1)
    if metric_names:
        fig.update_yaxes(title_text="Raw metric value", row=2, col=1)
    fig.update_xaxes(rangeslider_visible=False)
    return fig, data.summary, data.skipped


# ---------------------------------------------------------------------------
# metric_forward_profile_figure() — single-metric research notebook plot
# ---------------------------------------------------------------------------

def metric_forward_profile_figure(
    panel: pd.DataFrame,
    profile,
    *,
    horizon: int | None = None,
) -> "go.Figure":
    """Plot a single-metric forward profile returned by signal_diagnostics."""
    go = _go()
    horizon = int(horizon or profile.horizons[min(2, len(profile.horizons) - 1)])
    aligned = profile.aligned.copy()
    price_col = f"{profile.target_symbol}_close"
    if price_col not in panel.columns:
        price_col = "TQQQ_close"
    price = panel[price_col].reindex(aligned.index) if price_col in panel.columns else pd.Series(index=aligned.index, dtype=float)

    metric_name = profile.metric_name
    fwd_col = f"fwd_{horizon}d"
    color = "#607D8B"
    try:
        from metrics import REGISTRY
        color = _FAMILY_COLORS.get(REGISTRY[metric_name].family, color)
    except Exception:
        pass

    fig = _make_subplots(
        rows=7,
        cols=1,
        shared_xaxes=False,
        row_heights=[0.20, 0.16, 0.10, 0.13, 0.13, 0.13, 0.15],
        vertical_spacing=0.045,
        subplot_titles=[
            f"{profile.target_symbol} indexed close price",
            f"{metric_name} raw value",
            "Metric vote (+1 / 0 / -1)",
            "Rolling raw IC",
            f"Future {horizon}d return by vote",
            f"Future {horizon}d return by raw-value quantile",
            "Average event path after bullish / bearish votes",
        ],
    )

    fig.add_trace(go.Scatter(
        x=price.index,
        y=_indexed_to_100(price).values,
        name=f"{profile.target_symbol} indexed",
        line_color="#F44336",
    ), row=1, col=1)
    fig.update_yaxes(type="log", title_text="Indexed log", row=1, col=1)

    fig.add_trace(go.Scatter(
        x=aligned.index,
        y=aligned["metric"],
        name=metric_name,
        line_color=color,
    ), row=2, col=1)
    fig.update_yaxes(title_text="Raw", row=2, col=1)

    vote_colors = aligned["vote"].map({1: "#4CAF50", 0: "#9E9E9E", -1: "#F44336"}).fillna("#9E9E9E")
    fig.add_trace(go.Scatter(
        x=aligned.index,
        y=aligned["vote"],
        mode="markers",
        marker={"size": 4, "color": vote_colors},
        name="vote",
    ), row=3, col=1)
    fig.update_yaxes(title_text="Vote", tickvals=[-1, 0, 1], row=3, col=1)

    rolling = profile.rolling_ic
    if not rolling.empty:
        rolling_h = rolling.loc[rolling["horizon"] == horizon]
        fig.add_trace(go.Scatter(
            x=rolling_h["date"],
            y=rolling_h["rolling_raw_ic"],
            name=f"{horizon}d rolling IC",
            line_color="#2196F3",
        ), row=4, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#9E9E9E", row=4, col=1)
    fig.update_yaxes(title_text="Spearman", row=4, col=1)

    vote_table = profile.vote_bucket_table
    vote_h = vote_table.loc[
        (vote_table["horizon"] == horizon) & (vote_table["vote"].isin([1, 0, -1]))
    ].copy()
    if not vote_h.empty:
        fig.add_trace(go.Bar(
            x=vote_h["label"],
            y=vote_h["mean_fwd_bps"],
            text=vote_h["n"].astype(int).astype(str),
            textposition="outside",
            marker_color=["#4CAF50" if v == 1 else "#9E9E9E" if v == 0 else "#F44336" for v in vote_h["vote"]],
            name="vote buckets",
        ), row=5, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#9E9E9E", row=5, col=1)
    fig.update_yaxes(title_text="bps", row=5, col=1)

    quantile = profile.quantile_table
    q_h = quantile.loc[quantile["horizon"] == horizon].dropna(subset=["quantile"])
    if not q_h.empty:
        fig.add_trace(go.Bar(
            x=q_h["quantile"].astype(int).astype(str),
            y=q_h["mean_fwd_bps"],
            text=q_h["n"].astype(int).astype(str),
            textposition="outside",
            marker_color=color,
            name="quantiles",
        ), row=6, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#9E9E9E", row=6, col=1)
    fig.update_yaxes(title_text="bps", row=6, col=1)

    events = profile.event_paths
    for label, line_color in [("bull", "#4CAF50"), ("bear", "#F44336")]:
        event_h = events.loc[events["label"] == label]
        if event_h.empty:
            continue
        n_events = int(event_h["n_events"].max()) if event_h["n_events"].notna().any() else 0
        fig.add_trace(go.Scatter(
            x=event_h["step"],
            y=event_h["mean_return"] * 100,
            mode="lines+markers",
            name=f"{label} events (n={n_events})",
            line_color=line_color,
        ), row=7, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#9E9E9E", row=7, col=1)
    fig.update_xaxes(title_text="Trading days after entry", row=7, col=1)
    fig.update_yaxes(title_text="%", row=7, col=1)

    latest = profile.latest
    latest_date = latest.get("latest_date")
    latest_date_text = latest_date.date() if hasattr(latest_date, "date") else latest_date
    title = (
        f"Metric research: {metric_name} | split={profile.split} | "
        f"target={profile.target_symbol} {profile.target_kind} | "
        f"latest={latest_date_text}, vote={latest.get('latest_vote')}"
    )
    fig.update_layout(
        title_text=title,
        height=1280,
        hovermode="x unified",
        showlegend=True,
        legend={"orientation": "h", "y": -0.04},
        margin={"l": 70, "r": 30, "t": 90, "b": 90},
    )
    return fig


# ---------------------------------------------------------------------------
# edge_heatmap_figure() and quantile_shape_small_multiples() — leaderboard views
# ---------------------------------------------------------------------------

def edge_heatmap_figure(
    report_long: "pd.DataFrame",
    *,
    value_col: str = "bull_minus_bear_bps_val",
    agree_col: str = "edge_sign_agrees",
    top_n: int = 20,
) -> "go.Figure":
    """Heatmap of edge (bps) for top-N metrics × horizons on the val split.

    report_long must have columns: metric, horizon, value_col, agree_col, family.
    """
    go = _go()

    # Limit to top_n metrics in the order they appear
    top_metrics = list(dict.fromkeys(report_long["metric"].tolist()))[:top_n]
    df = report_long[report_long["metric"].isin(top_metrics)].copy()

    pivoted_val = df.pivot_table(index="metric", columns="horizon", values=value_col, aggfunc="mean", observed=False)
    pivoted_agree = df.pivot_table(index="metric", columns="horizon", values=agree_col, aggfunc="first", observed=False)

    # Preserve metric order
    pivoted_val = pivoted_val.reindex(top_metrics)
    pivoted_agree = pivoted_agree.reindex(top_metrics)

    horizons = sorted(pivoted_val.columns.tolist())
    metrics = pivoted_val.index.tolist()

    z = [[pivoted_val.loc[m, h] if h in pivoted_val.columns else None for h in horizons] for m in metrics]

    # Family color prefix for y-axis labels
    family_map = df.drop_duplicates("metric").set_index("metric")["family"].to_dict()
    y_labels = []
    for m in metrics:
        fam = family_map.get(m, "")
        color = _FAMILY_COLORS.get(fam, "#607D8B")
        # Prepend colored block as fallback since plotly doesn't support per-tick colors
        r, g, b = (int(color[i:i+2], 16) for i in (1, 3, 5))
        y_labels.append(f"<span style='color:rgb({r},{g},{b})'>■</span> {m}")

    heatmap = go.Heatmap(
        z=z,
        x=[str(h) for h in horizons],
        y=y_labels,
        colorscale="RdBu",
        zmid=0,
        hovertemplate="metric: %{y}<br>horizon: %{x}d<br>edge: %{z:.1f} bps<extra></extra>",
        colorbar={"title": "bps"},
    )

    fig = go.Figure(data=[heatmap])

    # Overlay ✓ on cells where sign agrees
    for i, m in enumerate(metrics):
        for j, h in enumerate(horizons):
            if h in pivoted_agree.columns:
                agrees = pivoted_agree.loc[m, h]
                if agrees:
                    fig.add_annotation(
                        x=str(h),
                        y=y_labels[i],
                        text="✓",
                        showarrow=False,
                        font={"size": 11, "color": "black"},
                    )

    fig.update_layout(
        title="Edge (bps) by metric × horizon — val split",
        height=max(400, 22 * top_n),
        margin={"l": 200, "r": 30, "t": 60, "b": 60},
        xaxis_title="Horizon (days)",
    )
    return fig


def quantile_shape_small_multiples(
    profiles: "dict[str, object]",
    *,
    horizon: int,
    ncols: int = 4,
) -> "go.Figure":
    """Grid of quantile-shape charts for a set of MetricForwardProfile objects.

    profiles: {metric_name: MetricForwardProfile}
    """
    go = _go()
    from metrics import REGISTRY

    names = list(profiles.keys())
    n = len(names)
    if n == 0:
        return go.Figure()

    rows = (n + ncols - 1) // ncols
    fig = _make_subplots(
        rows=rows,
        cols=ncols,
        subplot_titles=names,
        shared_yaxes=False,
        vertical_spacing=0.12,
        horizontal_spacing=0.06,
    )

    for idx, name in enumerate(names):
        profile = profiles[name]
        row = idx // ncols + 1
        col = idx % ncols + 1

        q_table = profile.quantile_table
        q_h = q_table.loc[q_table["horizon"] == horizon].dropna(subset=["quantile"])

        color = "#607D8B"
        m = REGISTRY.get(name)
        if m is not None:
            color = _FAMILY_COLORS.get(m.family, color)

        if not q_h.empty:
            fig.add_trace(go.Scatter(
                x=q_h["quantile"].astype(int).tolist(),
                y=q_h["mean_fwd_bps"].tolist(),
                mode="lines+markers",
                name=name,
                line_color=color,
                showlegend=False,
            ), row=row, col=col)

        fig.add_hline(y=0, line_dash="dot", line_color="#BDBDBD", row=row, col=col)

    fig.update_layout(
        title=f"Quantile shape — top metrics, horizon {horizon}d (val split)",
        height=max(360, 220 * rows),
        showlegend=False,
        margin={"l": 50, "r": 30, "t": 80, "b": 60},
    )
    return fig


def pairwise_redundancy_heatmap_figure(
    corr: "pd.DataFrame",
    *,
    family_map: "dict[str, str] | None" = None,
) -> "go.Figure":
    """N×N Spearman-correlation heatmap for all metrics.

    corr: square DataFrame from pairwise_redundancy_table (index = columns = metric names).
    family_map: {metric_name: family} for color-coded labels.
    """
    go = _go()

    names = list(corr.index)
    n = len(names)

    def _colored_label(name: str) -> str:
        fam = (family_map or {}).get(name, "")
        color = _FAMILY_COLORS.get(fam, "#607D8B")
        r, g, b = (int(color[i:i+2], 16) for i in (1, 3, 5))
        return f"<span style='color:rgb({r},{g},{b})'>■</span> {name}"

    labels = [_colored_label(n) for n in names]

    z = corr.values.tolist()

    heatmap = go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        colorscale="RdBu",
        zmid=0,
        zmin=-1,
        zmax=1,
        hovertemplate="%{y} × %{x}<br>ρ = %{z:.2f}<extra></extra>",
        colorbar={"title": "ρ"},
    )

    fig = go.Figure(data=[heatmap])
    fig.update_layout(
        title="Pairwise Spearman correlation — raw metric values (train+val)",
        height=max(500, 22 * n),
        margin={"l": 200, "r": 30, "t": 60, "b": 200},
        xaxis={"tickangle": 45},
    )
    return fig


# ---------------------------------------------------------------------------
# inspect() — 4-panel metric deep-dive
# ---------------------------------------------------------------------------

def inspect(
    metric_name: str,
    panel: pd.DataFrame,
    regime_states: pd.Series | None = None,
) -> "go.Figure":
    """4-panel inspection plot for a single metric.

    Panels:
      1. Metric time-series + QQQ price (right axis) + regime shading
      2. Histogram with current-value marker
      3. Autocorrelation function (first 40 lags)
      4. Hex-bin scatter: metric vs forward 5-day return
    """
    from metrics import REGISTRY
    go = _go()

    if metric_name not in REGISTRY:
        raise KeyError(f"Metric {metric_name!r} not in REGISTRY.")

    m = REGISTRY[metric_name]
    series = m.compute(panel).dropna()
    if series.empty:
        raise ValueError(f"Metric {metric_name!r} returned empty series.")

    qqq = panel.get("QQQ_close")
    fwd = _fwd_return(qqq) if qqq is not None else pd.Series(dtype=float)

    current_val = float(series.iloc[-1])
    color = _FAMILY_COLORS.get(m.family, "#607D8B")

    fig = _make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            f"{metric_name} time-series",
            f"{metric_name} distribution",
            "Autocorrelation (40 lags)",
            "Metric vs forward 5d return",
        ],
        specs=[[{"secondary_y": True}, {}], [{}, {}]],
    )

    # 1. Time series
    fig.add_trace(go.Scatter(x=series.index, y=series.values,
                             name=metric_name, line_color=color), row=1, col=1)
    if qqq is not None:
        fig.add_trace(go.Scatter(x=qqq.index, y=qqq.values,
                                 name="QQQ", line_color="#BDBDBD", opacity=0.5),
                      row=1, col=1, secondary_y=True)
    _add_regime_shading(fig, regime_states, row=1, col=1)

    # 2. Histogram
    vals = series.values
    fig.add_trace(go.Histogram(x=vals, nbinsx=50, name="dist",
                               marker_color=color, opacity=0.7), row=1, col=2)
    fig.add_vline(x=current_val, line_color="black", line_dash="dash", row=1, col=2)

    # 3. ACF
    n_lags = min(40, len(series) // 2)
    acf_vals = [pd.Series(vals).autocorr(lag=lag) for lag in range(1, n_lags + 1)]
    fig.add_trace(go.Bar(x=list(range(1, n_lags + 1)), y=acf_vals,
                         name="ACF", marker_color=color), row=2, col=1)
    # 95% CI band
    ci = 1.96 / np.sqrt(len(series))
    fig.add_hline(y=ci, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=-ci, line_dash="dot", line_color="red", row=2, col=1)

    # 4. Hex-bin (scatter with marker size=2 as a proxy)
    common = series.index.intersection(fwd.dropna().index)
    if len(common) > 10:
        x_sc = series.loc[common].values
        y_sc = fwd.loc[common].values * 100
        fig.add_trace(go.Scatter(
            x=x_sc, y=y_sc, mode="markers",
            marker={"size": 3, "color": color, "opacity": 0.4},
            name="fwd 5d ret %",
        ), row=2, col=2)

    fig.update_layout(
        title_text=f"Metric inspection: {metric_name}  (family={m.family}, status={m.status})",
        height=700,
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# dashboard() — current state snapshot
# ---------------------------------------------------------------------------

def dashboard(
    panel: pd.DataFrame,
    date: pd.Timestamp | str,
    regime_state: int,
    regime_probs: np.ndarray,
    regime_states: pd.Series | None = None,
) -> "go.Figure":
    """Current-percentile dashboard for all metrics.

    Shows horizontal bars colored by family, per-metric vote chip,
    and a regime state header.
    """
    from metrics import REGISTRY
    go = _go()

    date = pd.Timestamp(date)
    panel_until = panel.loc[:date]

    rows = []
    for name, m in REGISTRY.items():
        s = m.compute(panel_until)
        if s.empty or s.isna().all():
            continue
        cur = s.iloc[-1]
        if np.isnan(cur):
            continue
        pct = float((s <= cur).mean() * 100)
        vote = int(m.vote(s).iloc[-1]) if m.status == "voting" else 0
        rows.append({"name": name, "pct": pct, "vote": vote, "family": m.family, "status": m.status})

    df = pd.DataFrame(rows).sort_values("family")

    state_labels = ["strong_bull", "weak_bull", "sideways", "weak_bear", "strong_bear"]
    state_label = state_labels[min(regime_state, 4)]
    title = f"Dashboard @ {date.date()}  |  Regime: {state_label}  |  Probs: {np.round(regime_probs, 2)}"

    colors = [_FAMILY_COLORS.get(row["family"], "#607D8B") for _, row in df.iterrows()]
    vote_symbols = {-1: "▼ Sell", 0: "— Neutral", 1: "▲ Buy"}
    labels = [f"{row['name']}  {vote_symbols.get(row['vote'], '')}" for _, row in df.iterrows()]

    fig = _go().Figure()
    fig.add_trace(go.Bar(
        x=df["pct"].values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{p:.0f}%" for p in df["pct"]],
        textposition="inside",
    ))
    fig.add_vline(x=50, line_dash="dash", line_color="black")
    fig.update_layout(
        title_text=title,
        xaxis_title="Current percentile (rolling history)",
        height=max(400, len(df) * 22),
        showlegend=False,
        margin={"l": 280},
    )
    return fig


# ---------------------------------------------------------------------------
# backtest_report() — 5-row stacked plot
# ---------------------------------------------------------------------------

def backtest_report(result: "BacktestResult") -> "go.Figure":
    """5-row stacked plotly figure for backtest inspection.

    Rows:
      1. Equity (log scale) vs TQQQ BaH vs QQQ BaH
      2. Drawdown depth (%)
      3. Drawdown duration (rolling)
      4. Stacked area: p_buy / p_hold / p_sell
      5. Position state (0/1)
    """
    from backtest import _drawdown_series, _max_dd_duration
    go = _go()

    equity   = result.equity
    bah_tqqq = result.benchmark_tqqq
    bah_qqq  = result.benchmark_qqq
    signals  = result.signals
    pos      = result.positions
    p        = result.perf

    fig = _make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        row_heights=[0.35, 0.15, 0.15, 0.20, 0.15],
        subplot_titles=[
            "Equity (log scale)",
            "Drawdown depth (%)",
            "DD duration (days)",
            "p_buy / p_hold / p_sell",
            "Position",
        ],
        vertical_spacing=0.04,
    )

    # 1. Equity
    for s, name, color in [
        (equity,   "Strategy", "#2196F3"),
        (bah_tqqq, "TQQQ BaH", "#F44336"),
        (bah_qqq,  "QQQ BaH",  "#4CAF50"),
    ]:
        if s is not None and not s.isna().all():
            fig.add_trace(go.Scatter(
                x=s.index, y=s.values, name=name, line_color=color
            ), row=1, col=1)
    fig.update_yaxes(type="log", row=1, col=1)

    # 2. Drawdown depth
    dd = _drawdown_series(equity) * 100
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values, fill="tozeroy",
        fillcolor="rgba(244,67,54,0.3)", line_color="#F44336",
        name="DD%",
    ), row=2, col=1)

    # 3. DD duration (O(T) single pass)
    peak = equity.iloc[0]
    dd_start = 0
    in_dd = False
    dd_dur = []
    for t, v in enumerate(equity.values):
        if v >= peak:
            peak = v
            in_dd = False
            dd_dur.append(0)
        else:
            if not in_dd:
                in_dd = True
                dd_start = t
            dd_dur.append(t - dd_start)
    dd_dur_s = pd.Series(dd_dur, index=equity.index)
    fig.add_trace(go.Scatter(
        x=dd_dur_s.index, y=dd_dur_s.values,
        fill="tozeroy", fillcolor="rgba(255,152,0,0.3)", line_color="#FF9800",
        name="DD days",
    ), row=3, col=1)

    # 4. Stacked area p_buy/p_hold/p_sell
    for col_name, color, label in [
        ("p_buy",  "#4CAF50", "p_buy"),
        ("p_hold", "#9E9E9E", "p_hold"),
        ("p_sell", "#F44336", "p_sell"),
    ]:
        if col_name in signals.columns:
            fig.add_trace(go.Scatter(
                x=signals.index, y=signals[col_name].values,
                stackgroup="probs", name=label,
                fillcolor=_rgba(color, 0.6),
                line_color=color,
            ), row=4, col=1)

    # 5. Position
    fig.add_trace(go.Scatter(
        x=pos.index, y=pos.values, fill="tozeroy",
        fillcolor="rgba(33,150,243,0.3)", line_color="#2196F3",
        name="Long",
    ), row=5, col=1)

    # Annotation box with perf stats
    ann_text = (
        f"CAGR {p.get('cagr', 0):.1%} | Sharpe {p.get('sharpe', 0):.2f} | "
        f"MaxDD {p.get('maxdd_pct', 0):.1f}% ({p.get('maxdd_duration_days', 0)}d) | "
        f"vs TQQQ BaH {p.get('vs_tqqq_bh_excess_cagr', 0):.1%}"
    )
    fig.add_annotation(
        text=ann_text, xref="paper", yref="paper",
        x=0.0, y=1.02, xanchor="left", showarrow=False,
        font={"size": 11},
    )

    fig.update_layout(
        height=900,
        showlegend=True,
        legend={"orientation": "h", "y": -0.05},
    )
    return fig


# ---------------------------------------------------------------------------
# rolling_edge_decay_small_multiples() and regime_conditional_heatmap()
# — pre-strategy diagnostic views
# ---------------------------------------------------------------------------

def rolling_edge_decay_small_multiples(
    reports: "dict[str, pd.DataFrame]",
    *,
    ncols: int = 3,
) -> "go.Figure":
    """Grid of rolling edge-decay charts, one per metric.

    reports: {metric_name: rolling_edge_decay_dataframe}
    Each subplot shows edge_bps vs window_end with a faint OLS trend line.
    """
    go = _go()
    from metrics import REGISTRY

    names = list(reports.keys())
    n = len(names)
    if n == 0:
        return go.Figure()

    rows = (n + ncols - 1) // ncols
    fig = _make_subplots(
        rows=rows,
        cols=ncols,
        subplot_titles=names,
        shared_yaxes=False,
        vertical_spacing=0.12,
        horizontal_spacing=0.06,
    )

    for idx, name in enumerate(names):
        df = reports[name]
        row = idx // ncols + 1
        col = idx % ncols + 1

        color = "#607D8B"
        m = REGISTRY.get(name)
        if m is not None:
            color = _FAMILY_COLORS.get(m.family, color)

        if df.empty or df["edge_bps"].isna().all():
            continue

        x_vals = df["window_end"]
        y_vals = df["edge_bps"]

        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines",
            name=name,
            line_color=color,
            showlegend=False,
        ), row=row, col=col)

        # OLS trend line
        valid = df.dropna(subset=["edge_bps"])
        if len(valid) >= 2:
            t = np.arange(len(valid))
            coeffs = np.polyfit(t, valid["edge_bps"].values, 1)
            trend = np.polyval(coeffs, t)
            fig.add_trace(go.Scatter(
                x=valid["window_end"],
                y=trend,
                mode="lines",
                line={"color": color, "dash": "dot", "width": 1},
                opacity=0.4,
                showlegend=False,
            ), row=row, col=col)

        fig.add_hline(y=0, line_dash="dot", line_color="#9E9E9E", row=row, col=col)

    fig.update_layout(
        title="Rolling 2-year edge through time (bps) — train+val, 5d horizon",
        height=max(300, 240 * rows),
        showlegend=False,
        margin={"l": 50, "r": 30, "t": 80, "b": 60},
    )
    return fig


def regime_conditional_heatmap(
    reports: "dict[str, pd.DataFrame]",
    *,
    top_metric_order: "list[str] | None" = None,
) -> "go.Figure":
    """Heatmap of edge (bps) by metric × HSMM regime state.

    reports: {metric_name: regime_conditional_edge_table_dataframe}
    """
    go = _go()
    from metrics import REGISTRY

    regime_order = ["strong_bull", "weak_bull", "sideways", "weak_bear", "strong_bear"]
    names = top_metric_order if top_metric_order is not None else list(reports.keys())

    rows_list = []
    for name in names:
        df = reports.get(name)
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            rows_list.append({
                "metric": name,
                "regime_label": row["regime_label"],
                "edge_bps": row["edge_bps"],
                "n_days": row["n_days"],
            })

    if not rows_list:
        return go.Figure()

    combined = pd.DataFrame(rows_list)
    pivoted = combined.pivot_table(
        index="metric",
        columns="regime_label",
        values="edge_bps",
        aggfunc="first",
        observed=False,
    ).reindex(index=names, columns=regime_order)

    n_days_pivot = combined.pivot_table(
        index="metric",
        columns="regime_label",
        values="n_days",
        aggfunc="first",
        observed=False,
    ).reindex(index=names, columns=regime_order)

    family_map = {
        name: (REGISTRY[name].family if name in REGISTRY else "")
        for name in names
    }

    def _colored_label(name: str) -> str:
        fam = family_map.get(name, "")
        color = _FAMILY_COLORS.get(fam, "#607D8B")
        r, g, b = (int(color[i:i+2], 16) for i in (1, 3, 5))
        return f"<span style='color:rgb({r},{g},{b})'>■</span> {name}"

    y_labels = [_colored_label(n) for n in names if n in pivoted.index]
    valid_names = [n for n in names if n in pivoted.index]
    pivoted = pivoted.reindex(valid_names)

    z = pivoted.values.tolist()
    hover_text = []
    for i, name in enumerate(valid_names):
        row_hover = []
        for j, regime in enumerate(regime_order):
            edge = pivoted.loc[name, regime] if regime in pivoted.columns else np.nan
            nd = n_days_pivot.loc[name, regime] if name in n_days_pivot.index and regime in n_days_pivot.columns else np.nan
            row_hover.append(f"edge: {edge:.0f} bps<br>n_days: {int(nd) if not np.isnan(nd) else 'n/a'}")
        hover_text.append(row_hover)

    heatmap = go.Heatmap(
        z=z,
        x=regime_order,
        y=y_labels,
        colorscale="RdBu",
        zmid=0,
        text=hover_text,
        hovertemplate="%{y} | %{x}<br>%{text}<extra></extra>",
        colorbar={"title": "bps"},
    )

    fig = go.Figure(data=[heatmap])
    fig.update_layout(
        title="Edge (bps) by metric × HSMM regime — train+val, 5d horizon",
        height=max(400, 22 * len(valid_names) + 100),
        margin={"l": 220, "r": 30, "t": 60, "b": 80},
    )
    return fig
