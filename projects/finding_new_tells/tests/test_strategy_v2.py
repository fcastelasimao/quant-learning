"""Tests for the v2 weighted signal strategy."""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest import COST_BPS
from strategy_v2 import (
    V2MetricConfig,
    configs_from_decision_table,
    exposure_from_score,
    run_train_val_research,
    run_v2_backtest,
    weighted_vote_signals,
)


def test_configs_from_decision_table_keeps_and_inverts_only():
    decisions = pd.DataFrame({
        "metric": ["qqq_rsi2", "qqq_rv_20d", "qqq_bb_z20"],
        "decision": ["keep", "invert", "drop"],
        "direction": [1, -1, 0],
        "weight": [0.7, 0.4, 0.0],
    })

    configs = configs_from_decision_table(decisions)

    assert [(c.metric, c.direction, c.weight) for c in configs] == [
        ("qqq_rsi2", 1, 0.7),
        ("qqq_rv_20d", -1, 0.4),
    ]


def test_exposure_from_score_uses_allowed_binary_and_ternary_sets():
    score = pd.Series([-0.2, 0.0, 0.05, 0.2])

    binary = exposure_from_score(score, mode="binary", threshold=0.05)
    ternary = exposure_from_score(
        score,
        mode="ternary",
        threshold=0.10,
        medium_threshold=0.0,
    )

    assert set(binary.unique()) <= {0.0, 1.0}
    assert set(ternary.unique()) <= {0.0, 0.5, 1.0}
    assert ternary.tolist() == [0.0, 0.5, 0.5, 1.0]


def test_metric_inversion_changes_vote_signs(synthetic_panel):
    positive = weighted_vote_signals(
        synthetic_panel,
        [V2MetricConfig("qqq_rsi2", 1, 1.0)],
    )
    inverted = weighted_vote_signals(
        synthetic_panel,
        [V2MetricConfig("qqq_rsi2", -1, 1.0)],
    )

    pd.testing.assert_series_equal(
        positive["score"],
        -inverted["score"],
        check_names=False,
    )


def test_close_signal_fills_next_open_not_same_day(synthetic_panel):
    result = run_v2_backtest(
        synthetic_panel,
        [V2MetricConfig("qqq_rsi2", 1, 1.0)],
        mode="binary",
        threshold=-1.0,
        split="unit",
    )

    assert result.signals["target_exposure"].iloc[0] == 1.0
    assert result.exposure.iloc[0] == 0.0
    assert result.exposure.iloc[1] == 1.0


def test_strategy_return_is_recorded_on_realization_date(synthetic_panel):
    result = run_v2_backtest(
        synthetic_panel,
        [V2MetricConfig("qqq_rsi2", 1, 1.0)],
        mode="binary",
        threshold=-1.0,
        split="unit",
    )

    open_ = synthetic_panel["TQQQ_open"]
    expected_entry_day_return = -COST_BPS / 10_000
    expected_first_realized_return = open_.pct_change(fill_method=None).iloc[2]

    actual_entry_day_return = result.equity.pct_change(fill_method=None).iloc[1]
    actual_first_realized_return = result.equity.pct_change(fill_method=None).iloc[2]

    assert np.isclose(actual_entry_day_return, expected_entry_day_return)
    assert np.isclose(actual_first_realized_return, expected_first_realized_return)
    assert result.perf["trade_count"] == int((result.exposure.diff().abs().fillna(0) > 0).sum())


def test_v2_evaluation_index_uses_prior_history_for_first_validation_signal(synthetic_panel):
    spanning = synthetic_panel.copy()
    spanning.index = pd.bdate_range("2015-01-01", periods=len(spanning), freq="B")
    val_index = spanning.loc["2018-01-01":"2021-12-31"].index

    result = run_v2_backtest(
        spanning.loc[:"2021-12-31"],
        [V2MetricConfig("qqq_sma50_200_regime", 1, 1.0)],
        mode="binary",
        threshold=-1.0,
        split="val",
        evaluation_index=val_index,
    )

    assert result.equity.index.equals(val_index)
    assert result.signals.index.equals(val_index)
    assert result.exposure.iloc[0] == 1.0
    assert np.isnan(result.equity.pct_change(fill_method=None).iloc[0])


def test_run_train_val_research_config_excludes_watch_metrics(synthetic_panel, tmp_path):
    spanning = synthetic_panel.copy()
    spanning.index = pd.bdate_range("2015-01-01", periods=len(spanning), freq="B")

    artifacts = run_train_val_research(
        spanning,
        output_dir=tmp_path,
        horizon=1,
    )

    config_metrics = {cfg.metric for cfg in artifacts["configs"]}
    assert "yield_curve_10y3m" not in config_metrics
    assert "qqq_mom_12_1" not in config_metrics

    config_text = (tmp_path / "strategy_v2_config.json").read_text()
    assert '"selection_basis": "train_only"' in config_text
