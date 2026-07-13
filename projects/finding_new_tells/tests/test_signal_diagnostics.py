"""Tests for signal credibility diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signal_diagnostics import (
    apply_cost_model,
    bh_qvalues,
    forward_returns,
    metric_forward_profile,
    metric_decision_table,
    multi_horizon_credibility_report,
    pair_metric_profile,
    pairwise_redundancy_table,
    quantile_monotonicity,
    research_split_panel,
    rolling_edge_decay,
    signal_credibility_table,
    split_panel,
    train_val_credibility_report,
    tradable_open_forward_returns,
    vote_dynamics_report,
)
from metrics import Metric, REGISTRY


def _panel_spanning_train_and_val(panel: pd.DataFrame) -> pd.DataFrame:
    spanning = panel.copy()
    spanning.index = pd.bdate_range("2015-01-01", periods=len(spanning), freq="B")
    return spanning


def test_split_panel_respects_holdout_dates(synthetic_panel):
    train = split_panel(synthetic_panel, "train")
    val = split_panel(synthetic_panel, "val")
    test = split_panel(synthetic_panel, "test")

    assert train.index.max() <= pd.Timestamp("2017-12-31") or train.empty
    assert val.index.min() >= pd.Timestamp("2018-01-01")
    assert val.index.max() <= pd.Timestamp("2021-12-31")
    assert test.index.min() >= pd.Timestamp("2022-01-01")


def test_split_panel_rejects_unknown_split(synthetic_panel):
    with pytest.raises(ValueError, match="Unknown split"):
        split_panel(synthetic_panel, "paper")


def test_forward_returns_uses_future_close(synthetic_panel):
    fwd = forward_returns(synthetic_panel, horizons=(5,))[5]
    expected = synthetic_panel["QQQ_close"].shift(-5) / synthetic_panel["QQQ_close"] - 1

    pd.testing.assert_series_equal(fwd, expected.rename("QQQ_fwd_5d"))


def test_tradable_open_forward_returns_enter_next_open(synthetic_panel):
    fwd = tradable_open_forward_returns(synthetic_panel, symbol="TQQQ", horizons=(5,))[5]
    expected = (
        synthetic_panel["TQQQ_open"].shift(-6) / synthetic_panel["TQQQ_open"].shift(-1) - 1
    )

    pd.testing.assert_series_equal(fwd, expected.rename("TQQQ_tradable_fwd_5d"))


def test_signal_credibility_table_has_expected_columns(synthetic_panel):
    df = signal_credibility_table(
        synthetic_panel,
        split="val",
        horizons=(5,),
        metric_names=["qqq_rsi2", "qqq_rv_20d"],
    )

    assert set(df["metric"]) == {"qqq_rsi2", "qqq_rv_20d"}
    assert set(df["horizon"]) == {5}
    assert {
        "raw_ic",
        "vote_ic",
        "n_bull",
        "n_bear",
        "mean_bull",
        "mean_bear",
        "bull_minus_bear_bps",
        "min_directional_obs",
    } <= set(df.columns)
    assert np.isfinite(df["n"]).all()


def test_signal_credibility_table_supports_tradable_target(synthetic_panel):
    df = signal_credibility_table(
        synthetic_panel,
        split="val",
        horizons=(5,),
        target_symbol="TQQQ",
        target_kind="tradable_open",
        metric_names=["qqq_rsi2"],
    )

    assert len(df) == 1
    assert df.loc[0, "metric"] == "qqq_rsi2"
    assert np.isfinite(df.loc[0, "n"])


def test_signal_credibility_table_can_include_watch_metrics(synthetic_panel):
    df = signal_credibility_table(
        synthetic_panel,
        split="val",
        horizons=(5,),
        metric_names=["qqq_sign_entropy_20d"],
        include_watch=True,
    )

    assert len(df) == 1
    assert df.loc[0, "metric"] == "qqq_sign_entropy_20d"
    assert df.loc[0, "status"] == "watch"
    assert np.isfinite(df.loc[0, "n"])


def test_signal_credibility_table_uses_pre_split_metric_history():
    dates = pd.bdate_range("2017-12-20", periods=20)
    price = pd.Series(np.linspace(100.0, 119.0, len(dates)), index=dates)
    panel = pd.DataFrame({
        "TQQQ_open": price,
        "TQQQ_close": price,
        "QQQ_close": price,
    }, index=dates)

    metric_name = "_unit_requires_presplit_warmup"

    def _compute(p):
        return p["QQQ_close"].rolling(5, min_periods=5).mean()

    def _vote(s):
        return pd.Series(np.where(s.notna(), 1, 0), index=s.index, dtype=int)

    REGISTRY[metric_name] = Metric(metric_name, "unit", _compute, _vote)
    try:
        df = signal_credibility_table(
            panel,
            split="val",
            horizons=(1,),
            target_symbol="QQQ",
            target_kind="close",
            metric_names=[metric_name],
        )
    finally:
        REGISTRY.pop(metric_name, None)

    val_rows_with_forward_return = len(split_panel(panel, "val")) - 1
    assert df.loc[0, "n_bull"] == pytest.approx(val_rows_with_forward_return)


def test_train_val_report_ranks_metrics(synthetic_panel):
    panel = _panel_spanning_train_and_val(synthetic_panel)
    df = train_val_credibility_report(
        panel,
        horizon=5,
        metric_names=["qqq_rsi2", "qqq_rv_20d"],
    )

    assert list(df.columns[:4]) == [
        "metric",
        "family",
        "credibility_label",
        "credibility_score",
    ]
    assert set(df["credibility_label"]) <= {"promising", "mixed", "weak"}
    assert df["credibility_score"].is_monotonic_decreasing


def test_metric_decision_table_excludes_test_and_labels_are_deterministic(synthetic_panel):
    panel = _panel_spanning_train_and_val(synthetic_panel)
    df = metric_decision_table(
        panel,
        horizon=5,
        metric_names=["qqq_rsi2", "qqq_rv_20d"],
    )

    assert set(df["decision"]) <= {"keep", "invert", "drop"}
    assert (df["n_train"] <= len(split_panel(panel, "train"))).all()
    assert (df["n_val"] <= len(split_panel(panel, "val"))).all()
    assert "tqqq_edge_bps_train" in df.columns
    assert "qqq_edge_bps_val" in df.columns


def test_metric_decision_table_uses_train_only_for_keep_invert_drop():
    dates = pd.bdate_range("2015-01-01", "2021-12-31")
    votes = pd.Series(np.where(np.arange(len(dates)) % 2 == 0, 1, -1), index=dates)
    desired_fwd = pd.Series(0.0, index=dates)
    train_mask = dates <= pd.Timestamp("2017-12-31")
    val_mask = (dates >= pd.Timestamp("2018-01-01")) & (dates <= pd.Timestamp("2021-12-31"))
    desired_fwd.loc[train_mask] = np.where(votes.loc[train_mask] == 1, 0.01, -0.01)
    desired_fwd.loc[val_mask] = np.where(votes.loc[val_mask] == 1, -0.01, 0.01)

    open_values = [100.0]
    for i in range(1, len(dates)):
        source_idx = i - 2
        interval_ret = desired_fwd.iloc[source_idx] if source_idx >= 0 else 0.0
        open_values.append(open_values[-1] * (1 + interval_ret))
    open_ = pd.Series(open_values, index=dates)
    panel = pd.DataFrame({
        "TQQQ_open": open_,
        "TQQQ_close": open_,
        "QQQ_close": open_,
    }, index=dates)

    metric_name = "_unit_train_good_val_bad"

    def _compute(p):
        return votes.reindex(p.index).astype(float)

    def _vote(s):
        return s.fillna(0).astype(int)

    REGISTRY[metric_name] = Metric(metric_name, "unit", _compute, _vote)
    try:
        df = metric_decision_table(
            panel,
            horizon=1,
            metric_names=[metric_name],
            min_directional_obs=30,
        )
    finally:
        REGISTRY.pop(metric_name, None)

    row = df.iloc[0]
    assert row["decision"] == "keep"
    assert row["direction"] == 1
    assert row["selection_basis"] == "train_only"
    assert row["tqqq_edge_bps_train"] > 0
    assert row["tqqq_edge_bps_val"] < 0
    assert not bool(row["edge_sign_agrees"])


def test_metric_decision_table_excludes_watch_metrics_by_default(synthetic_panel):
    panel = _panel_spanning_train_and_val(synthetic_panel)
    df = metric_decision_table(panel, horizon=5)

    assert "status" in df.columns
    assert set(df["status"]) == {"voting"}


def test_research_split_panel_never_returns_frozen_test(synthetic_panel):
    research = research_split_panel(synthetic_panel, "train+val")

    assert research.index.max() <= pd.Timestamp("2021-12-31")
    with pytest.raises(ValueError, match="Research diagnostics"):
        research_split_panel(synthetic_panel, "test")


def test_metric_forward_profile_contains_all_requested_diagnostics(synthetic_panel):
    panel = _panel_spanning_train_and_val(synthetic_panel)
    profile = metric_forward_profile(
        panel,
        "qqq_rsi2",
        split="val",
        horizons=(1, 5),
        bins=5,
    )

    assert set(profile.ic_table["horizon"]) == {1, 5}
    assert set(profile.vote_bucket_table["horizon"]) == {1, 5}
    assert set(profile.quantile_table["horizon"]) == {1, 5}
    assert {"metric", "vote", "fwd_1d", "fwd_5d"} <= set(profile.aligned.columns)
    assert profile.aligned.index.min() >= pd.Timestamp("2018-01-01")
    assert profile.aligned.index.max() <= pd.Timestamp("2021-12-31")
    assert not profile.event_paths.empty


def test_metric_forward_profile_vote_buckets_match_manual_calculation(synthetic_panel):
    panel = _panel_spanning_train_and_val(synthetic_panel)
    profile = metric_forward_profile(panel, "qqq_rsi2", split="val", horizons=(5,))
    aligned = profile.aligned.dropna(subset=["fwd_5d"])

    manual_bull = aligned.loc[aligned["vote"] == 1, "fwd_5d"].mean()
    table_bull = profile.vote_bucket_table.loc[
        (profile.vote_bucket_table["horizon"] == 5)
        & (profile.vote_bucket_table["vote"] == 1),
        "mean_fwd",
    ].iloc[0]

    assert np.isclose(table_bull, manual_bull, equal_nan=True)


def test_metric_forward_profile_event_paths_start_after_next_open():
    dates = pd.bdate_range("2015-01-01", periods=50)
    opens = pd.Series(np.linspace(100, 149, len(dates)), index=dates)
    panel = pd.DataFrame({
        "TQQQ_open": opens,
        "TQQQ_close": opens + 0.5,
        "QQQ_close": opens / 3,
    }, index=dates)

    metric_name = "_unit_first_day_bull"

    def _compute(p):
        s = pd.Series(0.0, index=p.index)
        s.iloc[0] = 1.0
        return s

    def _vote(s):
        return pd.Series(np.where(s == 1.0, 1, 0), index=s.index, dtype=int)

    REGISTRY[metric_name] = Metric(metric_name, "unit", _compute, _vote)
    try:
        profile = metric_forward_profile(panel, metric_name, split="train", horizons=(1,))
    finally:
        REGISTRY.pop(metric_name, None)

    bull_path = profile.event_paths.loc[profile.event_paths["label"] == "bull"].set_index("step")
    expected_step_1 = panel["TQQQ_open"].iloc[2] / panel["TQQQ_open"].iloc[1] - 1

    assert bull_path.loc[0, "mean_return"] == 0.0
    assert np.isclose(bull_path.loc[1, "mean_return"], expected_step_1)


def test_metric_forward_profile_rejects_test_split(synthetic_panel):
    with pytest.raises(ValueError, match="Research diagnostics"):
        metric_forward_profile(synthetic_panel, "qqq_rsi2", split="test")


def test_pair_metric_profile_combined_masks_match_expected_logic():
    dates = pd.bdate_range("2015-01-01", periods=50)
    base = pd.Series(np.linspace(100, 149, len(dates)), index=dates)
    panel = pd.DataFrame({
        "TQQQ_open": base,
        "TQQQ_close": base,
        "QQQ_close": base,
    }, index=dates)

    primary_name = "_unit_primary_vote"
    filter_name = "_unit_filter_vote"
    primary_votes = pd.Series([1, 1, -1, -1, 0] + [0] * 45, index=dates)
    filter_votes = pd.Series([1, -1, -1, 1, 0] + [0] * 45, index=dates)

    def _compute_primary(p):
        return primary_votes.reindex(p.index)

    def _compute_filter(p):
        return filter_votes.reindex(p.index)

    def _vote_identity(s):
        return s.fillna(0).astype(int)

    REGISTRY[primary_name] = Metric(primary_name, "unit", _compute_primary, _vote_identity)
    REGISTRY[filter_name] = Metric(filter_name, "unit", _compute_filter, _vote_identity)
    try:
        profile = pair_metric_profile(
            panel,
            primary_name,
            filter_name,
            split="train",
            horizons=(1,),
        )
    finally:
        REGISTRY.pop(primary_name, None)
        REGISTRY.pop(filter_name, None)

    assert profile.masks["primary_bull_and_filter_bull"].sum() == 1
    assert profile.masks["primary_bull_and_filter_not_bear"].sum() == 1
    assert profile.masks["primary_bear_and_filter_bear"].sum() == 1
    assert set(profile.condition_table["condition"]) == {
        "primary_bull_and_filter_bull",
        "primary_bull_and_filter_not_bear",
        "primary_bear_and_filter_bear",
    }


def test_pair_metric_profile_rejects_test_split(synthetic_panel):
    with pytest.raises(ValueError, match="Research diagnostics"):
        pair_metric_profile(synthetic_panel, "qqq_rsi2", "qqq_rv_20d", split="test")


def test_pairwise_redundancy_table_is_symmetric_with_unit_diagonal(synthetic_panel):
    panel = _panel_spanning_train_and_val(synthetic_panel)
    corr = pairwise_redundancy_table(
        panel,
        split="train",
        metric_names=["qqq_rsi2", "qqq_rv_20d", "qqq_bb_z20"],
        include_watch=True,
    )
    assert corr.shape == (3, 3)
    assert list(corr.index) == list(corr.columns)
    np.testing.assert_allclose(np.diag(corr.values), 1.0, atol=1e-9)
    np.testing.assert_allclose(corr.values, corr.values.T, atol=1e-9)


def test_bh_qvalues_matches_manual_formula():
    pvals = pd.Series([0.001, 0.01, 0.04, 0.5], index=["a", "b", "c", "d"])
    qvals = bh_qvalues(pvals)

    # BH formula: q_i = min over k>=i of (m * p_(k) / k) where p is sorted ascending
    m = 4
    sorted_pvals = sorted(pvals.values)
    bh_raw = [m * sorted_pvals[k] / (k + 1) for k in range(m)]
    # running minimum from right
    bh_min = []
    running_min = float("inf")
    for v in reversed(bh_raw):
        running_min = min(running_min, v)
        bh_min.insert(0, running_min)
    # map back to original order by rank
    ranks = sorted(range(m), key=lambda i: pvals.values[i])
    expected = np.empty(m)
    for rank_idx, orig_idx in enumerate(ranks):
        expected[orig_idx] = min(bh_min[rank_idx], 1.0)

    assert np.allclose(qvals.values, expected, atol=1e-6)

    # NaN passthrough
    pvals_nan = pd.Series([0.05, np.nan, 0.1])
    qvals_nan = bh_qvalues(pvals_nan)
    assert np.isnan(qvals_nan.iloc[1])
    assert np.isfinite(qvals_nan.iloc[0])
    assert np.isfinite(qvals_nan.iloc[2])


def test_quantile_monotonicity_clean_signal_returns_high_value():
    rng = np.random.default_rng(0)
    n = 300
    values = pd.Series(np.linspace(0, 1, n) + rng.normal(0, 0.01, n))
    fwd_clean = values * 10 + rng.normal(0, 0.5, n)   # near-linear positive relationship

    mono_clean = quantile_monotonicity(values, fwd_clean, bins=10)
    assert abs(mono_clean) > 0.9, f"Expected |monotonicity| > 0.9, got {mono_clean}"

    # U-shape: high values at extremes → not monotone
    fwd_u = -(values - 0.5) ** 2 * 40 + rng.normal(0, 0.5, n)
    mono_u = quantile_monotonicity(values, fwd_u, bins=10)
    assert abs(mono_u) < 0.3, f"Expected |monotonicity| < 0.3, got {mono_u}"


def test_multi_horizon_credibility_report_columns_and_horizon_count(synthetic_panel):
    panel = _panel_spanning_train_and_val(synthetic_panel)
    df = multi_horizon_credibility_report(
        panel,
        horizons=(1, 2, 5, 10, 20),
        target_symbol="TQQQ",
        target_kind="tradable_open",
        include_watch=True,
        metric_names=["qqq_rsi2", "qqq_rv_20d"],
    )

    expected_cols = {
        "metric", "family", "status",
        "edge_train_5d", "edge_val_5d",
        "raw_ic_train_5d", "raw_ic_val_5d",
        "raw_ic_p_val_5d", "raw_ic_q_val_5d",
        "vote_ic_val_5d",
        "monotonicity_val_5d",
        "n_horizons_edge_sign_agree",
        "min_directional_obs_val",
        "n_val",
        "credibility_score",
        "credibility_label",
        "passes_min_obs",
    }
    assert expected_cols == set(df.columns), f"Missing: {expected_cols - set(df.columns)}"
    assert (df["n_horizons_edge_sign_agree"].between(0, 5)).all()
    assert df["raw_ic_p_val_5d"].notna().any()
    assert df["raw_ic_q_val_5d"].notna().any()


def test_vote_dynamics_constant_vote_has_zero_flips(synthetic_panel):
    """A metric that always votes +1 has 0 flips and run_length == n_days."""
    panel = _panel_spanning_train_and_val(synthetic_panel)
    name = "_unit_constant_bull_vote"

    def _compute(p):
        return pd.Series(1.0, index=p.index)

    def _vote(s):
        return pd.Series(1, index=s.index, dtype=int)

    REGISTRY[name] = Metric(name, "unit", _compute, _vote)
    try:
        df = vote_dynamics_report(panel, split="train+val", metric_names=[name], include_watch=True)
    finally:
        REGISTRY.pop(name, None)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["n_flips_any"] == 0
    assert row["n_flips_directional"] == 0
    assert row["pct_bull"] == pytest.approx(1.0)
    assert row["avg_long_fraction"] == pytest.approx(1.0)
    from signal_diagnostics import research_split_panel
    n_days = len(research_split_panel(panel, "train+val"))
    assert row["mean_run_length"] == pytest.approx(n_days)


def test_vote_dynamics_alternating_vote_has_max_flips(synthetic_panel):
    """A metric that flips every day has n_flips_any ≈ n_days - 1."""
    panel = _panel_spanning_train_and_val(synthetic_panel)
    name = "_unit_alternating_vote"
    from signal_diagnostics import research_split_panel
    n_days = len(research_split_panel(panel, "train+val"))

    def _compute(p):
        return pd.Series(range(len(p)), index=p.index, dtype=float)

    def _vote(s):
        return pd.Series(np.where(s % 2 == 0, 1, -1), index=s.index, dtype=int)

    REGISTRY[name] = Metric(name, "unit", _compute, _vote)
    try:
        df = vote_dynamics_report(panel, split="train+val", metric_names=[name], include_watch=True)
    finally:
        REGISTRY.pop(name, None)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["n_flips_any"] == n_days - 1


def test_apply_cost_model_arithmetic():
    """Verify net_annual_bps formula on known inputs.

    gross 50 bps/trade × 20 trades/year = 1000 gross annual
    friction 2 × 2bps × 20 = 80
    drag 86 × 0.5 = 43
    net = 1000 - 80 - 43 = 877
    """
    result = apply_cost_model(
        gross_edge_bps_per_trade=50.0,
        flips_per_year=40.0,   # 40 flips → 20 trades
        long_fraction=0.5,
        one_way_spread_bps=2.0,
        one_way_commission_bps=0.0,
        annual_expense_bps=86.0,
    )
    assert result["trades_per_year"] == pytest.approx(20.0)
    assert result["gross_annual_bps"] == pytest.approx(1000.0)
    assert result["friction_bps_per_round_trip"] == pytest.approx(4.0)
    assert result["friction_annual_bps"] == pytest.approx(80.0)
    assert result["etf_drag_annual_bps"] == pytest.approx(43.0)
    assert result["net_annual_bps"] == pytest.approx(877.0)
    assert result["breakeven_edge_bps_per_trade"] == pytest.approx((80.0 + 43.0) / 20.0)


def test_rolling_edge_decay_returns_one_row_per_window(synthetic_panel):
    """With ~1500 aligned obs, window=504, step=21 → about 47-48 rows."""
    panel = _panel_spanning_train_and_val(synthetic_panel)
    # TODO: lightweight regime fixture — regime_conditional_edge_table not unit-tested
    # (fitting HSMM in the test suite is too slow; covered by notebook visual check)
    df = rolling_edge_decay(
        panel,
        "qqq_rsi2",
        split="train+val",
        window=504,
        step=21,
        horizon=5,
    )
    assert not df.empty
    assert set(df.columns) >= {"window_end", "edge_bps", "raw_ic", "n", "n_directional"}
    assert len(df) >= 40  # at least 40 windows in ~1500 rows with window=504, step=21
    assert len(df) <= 60  # upper bound sanity check
    assert (df["n"] == 504).all()
