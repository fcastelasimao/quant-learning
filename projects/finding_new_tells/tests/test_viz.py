"""Tests for visual workbench data preparation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metrics import REGISTRY
from viz import (
    _candidate_rule_summary,
    _mask_starts,
    indicator_workbench,
    prepare_indicator_workbench,
)


def test_indicator_workbench_aligns_metrics_to_selected_dates(synthetic_panel):
    start = synthetic_panel.index[300]
    end = synthetic_panel.index[450]

    data = prepare_indicator_workbench(
        synthetic_panel,
        ["qqq_20d_slope", "not_a_metric"],
        start=start,
        end=end,
    )

    assert data.price.index.equals(synthetic_panel.loc[start:end].index)
    assert data.metrics["qqq_20d_slope"].index.equals(data.price.index)
    assert data.skipped["not_a_metric"] == "not registered"


def test_indicator_workbench_threshold_masks_and_zone_counts(synthetic_panel):
    data = prepare_indicator_workbench(
        synthetic_panel,
        ["qqq_rsi2"],
        rule_metric="qqq_rsi2",
        low_threshold=35.0,
        high_threshold=65.0,
    )

    rsi = data.metrics["qqq_rsi2"]
    assert data.buy_mask.equals((rsi <= 35.0).fillna(False))
    assert data.sell_mask.equals((rsi >= 65.0).fillna(False))
    assert data.summary.loc["buy", "zones"] == len(_mask_starts(data.buy_mask))
    assert data.summary.loc["sell", "zones"] == len(_mask_starts(data.sell_mask))


def test_candidate_rule_summary_does_not_forward_fill_returns():
    idx = pd.bdate_range("2024-01-01", periods=25)
    price = pd.Series(np.linspace(100.0, 124.0, 25), index=idx)
    price.iloc[5] = np.nan
    buy_mask = pd.Series(False, index=idx)
    sell_mask = pd.Series(False, index=idx)
    buy_mask.iloc[0] = True

    summary = _candidate_rule_summary(price, buy_mask, sell_mask)

    assert np.isnan(summary.loc["buy", "avg_fwd_5d_%"])
    assert summary.loc["buy", "zones"] == 1
    assert summary.loc["sell", "zones"] == 0


def test_indicator_workbench_raises_without_tqqq_close(synthetic_panel):
    panel = synthetic_panel.drop(columns=["TQQQ_close"])
    with pytest.raises(KeyError):
        prepare_indicator_workbench(panel, ["qqq_rsi2"])


def test_indicator_workbench_returns_figure_summary_and_skips(synthetic_panel):
    fig, summary, skipped = indicator_workbench(
        synthetic_panel,
        ["qqq_rsi2", "missing_metric"],
        rule_metric="qqq_rsi2",
        low_threshold=30.0,
        high_threshold=70.0,
    )

    assert len(fig.data) >= 2
    assert set(summary.index) == {"buy", "sell"}
    assert skipped["missing_metric"] == "not registered"


def test_indicator_workbench_default_display_uses_raw_metrics_without_signals(synthetic_panel):
    fig, _, _ = indicator_workbench(
        synthetic_panel,
        ["qqq_rsi2", "qqq_rv_20d"],
        rule_metric="qqq_rsi2",
        low_threshold=30.0,
        high_threshold=70.0,
    )

    tqqq_trace = fig.data[0]
    qqq_trace = fig.data[1]
    metric_traces = fig.data[2:4]

    assert tqqq_trace.name == "TQQQ indexed"
    assert tqqq_trace.line.color == "#F44336"
    assert tqqq_trace.y[0] == 100.0
    assert qqq_trace.name == "QQQ indexed"
    assert qqq_trace.line.color == "#9E9E9E"
    assert qqq_trace.y[0] == 100.0
    assert fig.layout.yaxis.type == "log"
    assert all(trace.yaxis == "y2" for trace in metric_traces)
    for trace in metric_traces:
        expected = REGISTRY[trace.name].compute(synthetic_panel).reindex(synthetic_panel.index)
        np.testing.assert_allclose(trace.y, expected.values, equal_nan=True)
    assert not any("buy" in trace.name.lower() or "sell" in trace.name.lower() for trace in fig.data)
