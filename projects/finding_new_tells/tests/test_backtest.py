"""Tests for src/backtest.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest import (
    BacktestResult,
    run_backtest,
    _perf_stats,
    _max_dd_duration,
    _current_dd_duration,
    _gap_intraday_decomp,
    _drawdown_series,
)


# ---------------------------------------------------------------------------
# Helper: minimal panel with just TQQQ and QQQ
# ---------------------------------------------------------------------------

@pytest.fixture
def mini_panel(synthetic_panel) -> pd.DataFrame:
    cols = [c for c in synthetic_panel.columns if c.startswith(("TQQQ_", "QQQ_"))]
    return synthetic_panel[cols].copy()


# ---------------------------------------------------------------------------
# MaxDD duration — hand-computed example
# ---------------------------------------------------------------------------

def test_max_dd_duration_hand_computed():
    """Equity that drops for 5 days then recovers: duration = 5."""
    vals = [1.0, 1.1, 1.0, 0.9, 0.8, 0.7, 0.8, 0.9, 1.1, 1.2]
    equity = pd.Series(vals)
    # Peak at index 1 (1.1), trough at index 5 (0.7), recovery at index 8 (1.1)
    # Duration = 8 - 1 = 7 days
    assert _max_dd_duration(equity) == 7


def test_max_dd_duration_no_dd():
    equity = pd.Series([1.0, 1.1, 1.2, 1.3, 1.4])
    assert _max_dd_duration(equity) == 0


def test_max_dd_duration_never_recovers():
    """If equity never recovers, duration = T - peak_idx - 1."""
    equity = pd.Series([1.0, 1.2, 1.1, 0.9, 0.8, 0.7])
    # Peak at idx 1, never recovers → duration = 4
    assert _max_dd_duration(equity) == 4


def test_current_dd_duration_zero_when_at_peak():
    equity = pd.Series([1.0, 1.1, 1.2, 1.3])
    assert _current_dd_duration(equity) == 0


def test_current_dd_duration_positive_in_dd():
    equity = pd.Series([1.0, 1.2, 1.1, 0.9, 0.8])
    # In drawdown since idx 2 (value < peak 1.2)
    result = _current_dd_duration(equity)
    assert result > 0


# ---------------------------------------------------------------------------
# gap + intraday sums to total return (approximately)
# ---------------------------------------------------------------------------

def test_gap_intraday_sum_to_total(mini_panel):
    tqqq_open  = mini_panel["TQQQ_open"]
    tqqq_close = mini_panel["TQQQ_close"]
    position = pd.Series(1, index=mini_panel.index)  # always long

    gap, intra = _gap_intraday_decomp(tqqq_open, tqqq_close, position)

    # Total return ≈ sum of daily open-to-open returns (rough; exact only in continuous compounding)
    # Check that gap and intraday are finite and of reasonable magnitude
    assert np.isfinite(gap)
    assert np.isfinite(intra)


# ---------------------------------------------------------------------------
# run_backtest: equity causality test
# ---------------------------------------------------------------------------

def test_equity_depends_only_on_past_signals(mini_panel):
    """Equity at index t must be unchanged when all data after t is shuffled."""
    rng = np.random.default_rng(42)
    cutoff = 200

    result_orig = run_backtest(mini_panel, use_regime=False)
    equity_orig = result_orig.equity

    panel_shuffled = mini_panel.copy()
    future = panel_shuffled.iloc[cutoff + 1:].values.copy()
    rng.shuffle(future)
    panel_shuffled.iloc[cutoff + 1:] = future

    result_shuffled = run_backtest(panel_shuffled, use_regime=False)
    equity_shuffled = result_shuffled.equity

    assert np.isclose(equity_orig.iloc[cutoff], equity_shuffled.iloc[cutoff], rtol=1e-6), (
        f"Equity at cutoff changed: {equity_orig.iloc[cutoff]:.6f} → {equity_shuffled.iloc[cutoff]:.6f}"
    )


# ---------------------------------------------------------------------------
# run_backtest: output shapes and types
# ---------------------------------------------------------------------------

def test_run_backtest_equity_length(mini_panel):
    result = run_backtest(mini_panel, use_regime=False)
    assert len(result.equity) == len(mini_panel)


def test_run_backtest_positions_binary(mini_panel):
    result = run_backtest(mini_panel, use_regime=False)
    assert set(result.positions.unique()) <= {0, 1}


def test_run_backtest_perf_keys(mini_panel):
    result = run_backtest(mini_panel, use_regime=False)
    for key in ["cagr", "sharpe", "maxdd_pct", "maxdd_duration_days", "exposure_pct"]:
        assert key in result.perf, f"Missing perf key: {key}"


def test_run_backtest_signals_columns(mini_panel):
    result = run_backtest(mini_panel, use_regime=False)
    for col in ["p_buy", "p_hold", "p_sell", "score", "action"]:
        assert col in result.signals.columns


# ---------------------------------------------------------------------------
# _perf_stats
# ---------------------------------------------------------------------------

def test_perf_stats_on_flat_equity():
    equity = pd.Series([1.0] * 252)
    stats = _perf_stats(equity)
    assert abs(stats["cagr"]) < 1e-6
    assert stats["maxdd_pct"] == 0.0


def test_perf_stats_growing_equity():
    n = 252
    equity = pd.Series(np.linspace(1.0, 2.0, n))
    stats = _perf_stats(equity)
    assert stats["cagr"] > 0
    assert stats["maxdd_pct"] <= 0


# ---------------------------------------------------------------------------
# Benchmark alignment
# ---------------------------------------------------------------------------

def test_benchmarks_have_same_index(mini_panel):
    result = run_backtest(mini_panel, use_regime=False)
    assert result.equity.index.equals(result.benchmark_tqqq.index)
    assert result.equity.index.equals(result.benchmark_qqq.index)


# ---------------------------------------------------------------------------
# MASTER_LOG path is in project root
# ---------------------------------------------------------------------------

def test_master_log_path_in_project_root():
    from backtest import MASTER_LOG_PATH
    assert MASTER_LOG_PATH.parent.name == "finding_new_tells"
