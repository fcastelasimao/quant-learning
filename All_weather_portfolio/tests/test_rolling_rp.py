"""
tests/test_rolling_rp.py
========================
Tests for run_rolling_rp.py.

Unit tests (no network required):
  - save_weight_history  : CSV creation, columns, row count, values
  - _latest_weights      : last entry extraction, no 'date' key, sum-to-one
  - Rolling RP weights   : weights change over time, always sum to 1.0
  - Backtest output      : expected columns, positive portfolio values
  - Win-counter logic    : all-rolling, all-static, partial, tie

Run with:
    conda run -n allweather python3 -m pytest tests/test_rolling_rp.py -v
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_rolling_rp import _latest_weights, save_weight_history

TICKERS = ["SPY", "QQQ", "TLT", "TIP", "GLD", "GSG"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_weight_history():
    """Three quarterly weight records — weights are deliberately different
    so _latest_weights tests can assert on specific values."""
    return [
        {"date": pd.Timestamp("2020-01-31"),
         "SPY": 0.15, "QQQ": 0.12, "TLT": 0.20, "TIP": 0.30, "GLD": 0.13, "GSG": 0.10},
        {"date": pd.Timestamp("2020-04-30"),
         "SPY": 0.14, "QQQ": 0.11, "TLT": 0.22, "TIP": 0.31, "GLD": 0.12, "GSG": 0.10},
        {"date": pd.Timestamp("2020-07-31"),
         "SPY": 0.16, "QQQ": 0.13, "TLT": 0.19, "TIP": 0.28, "GLD": 0.15, "GSG": 0.09},
    ]


@pytest.fixture
def synthetic_prices():
    """
    Daily prices for 6 tickers over 10 years with two distinct volatility
    regimes (volumes swap halfway). This forces the rolling RP optimizer to
    find meaningfully different weights in each half, making weight evolution
    detectable without needing real market data.
    """
    np.random.seed(42)
    n_days = 10 * 252
    dates  = pd.date_range("2010-01-01", periods=n_days, freq="B")
    n      = len(TICKERS)

    # Regime 1: equities low-vol, bonds high-vol
    vols_1 = np.array([0.08, 0.10, 0.18, 0.12, 0.12, 0.15]) / np.sqrt(252)
    # Regime 2: equities high-vol, bonds low-vol  (opposite ranking)
    vols_2 = np.array([0.25, 0.30, 0.05, 0.04, 0.20, 0.22]) / np.sqrt(252)

    half    = n_days // 2
    returns = np.zeros((n_days, n))
    returns[:half] = np.random.randn(half,           n) * vols_1
    returns[half:] = np.random.randn(n_days - half,  n) * vols_2

    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    return pd.DataFrame(prices, index=dates, columns=TICKERS)


# ---------------------------------------------------------------------------
# save_weight_history
# ---------------------------------------------------------------------------

class TestSaveWeightHistory:
    def test_creates_csv_file(self, sample_weight_history, tmp_path):
        path = save_weight_history(sample_weight_history, str(tmp_path), "test")
        assert os.path.exists(path)

    def test_csv_has_ticker_columns(self, sample_weight_history, tmp_path):
        path = save_weight_history(sample_weight_history, str(tmp_path), "test")
        df   = pd.read_csv(path, index_col=0)
        assert set(df.columns) == set(TICKERS)

    def test_row_count_matches_input(self, sample_weight_history, tmp_path):
        path = save_weight_history(sample_weight_history, str(tmp_path), "test")
        df   = pd.read_csv(path)
        assert len(df) == len(sample_weight_history)

    def test_filename_contains_tag(self, sample_weight_history, tmp_path):
        path = save_weight_history(sample_weight_history, str(tmp_path), "split2020")
        assert "split2020" in os.path.basename(path)

    def test_first_row_spy_weight_preserved(self, sample_weight_history, tmp_path):
        path = save_weight_history(sample_weight_history, str(tmp_path), "test")
        df   = pd.read_csv(path, index_col=0)
        assert abs(df["SPY"].iloc[0] - 0.15) < 1e-6

    def test_returns_string_path(self, sample_weight_history, tmp_path):
        result = save_weight_history(sample_weight_history, str(tmp_path), "test")
        assert isinstance(result, str)
        assert result.endswith(".csv")


# ---------------------------------------------------------------------------
# _latest_weights
# ---------------------------------------------------------------------------

class TestLatestWeights:
    def test_returns_last_entry_values(self, sample_weight_history):
        last = _latest_weights(sample_weight_history)
        # Third record has SPY=0.16
        assert abs(last["SPY"] - 0.16) < 1e-9

    def test_does_not_contain_date_key(self, sample_weight_history):
        last = _latest_weights(sample_weight_history)
        assert "date" not in last

    def test_contains_all_tickers(self, sample_weight_history):
        last = _latest_weights(sample_weight_history)
        assert set(last.keys()) == set(TICKERS)

    def test_weights_sum_to_one(self, sample_weight_history):
        last = _latest_weights(sample_weight_history)
        assert abs(sum(last.values()) - 1.0) < 1e-4

    def test_single_entry_history(self):
        history = [{"date": pd.Timestamp("2022-01-31"),
                    "SPY": 0.2, "QQQ": 0.2, "TLT": 0.2,
                    "TIP": 0.2, "GLD": 0.1, "GSG": 0.1}]
        last = _latest_weights(history)
        assert abs(last["SPY"] - 0.2) < 1e-9
        assert "date" not in last


# ---------------------------------------------------------------------------
# Rolling RP weight evolution (uses run_backtest_rolling_rp directly)
# ---------------------------------------------------------------------------

class TestRollingRPBehaviour:
    """
    These tests call run_backtest_rolling_rp with synthetic prices and a
    short lookback so they complete quickly without network access.
    Annual recomputation keeps the test fast (~1-2 s).
    """

    @staticmethod
    def _run(prices):
        from backtest import run_backtest_rolling_rp
        return run_backtest_rolling_rp(
            prices, prices["SPY"], TICKERS,
            rp_lookback_years=1.0,
            rp_recompute_freq="YS",    # annual — fast
        )

    def test_weights_evolve_across_regimes(self, synthetic_prices):
        """Weight vector must shift when volatility regime changes."""
        _, weight_history = self._run(synthetic_prices)
        assert len(weight_history) >= 2

        weight_arrays = [
            np.array([w[t] for t in TICKERS])
            for w in weight_history
        ]
        max_diff = max(
            np.max(np.abs(wa - weight_arrays[0]))
            for wa in weight_arrays[1:]
        )
        assert max_diff > 1e-3, (
            f"Weights did not evolve — max change was {max_diff:.6f}. "
            "Check that synthetic prices have a real volatility regime shift."
        )

    def test_weights_sum_to_one_at_every_recomputation(self, synthetic_prices):
        _, weight_history = self._run(synthetic_prices)
        for record in weight_history:
            total = sum(v for k, v in record.items() if k != "date")
            # compute_risk_parity_weights rounds to 4 dp; 6 assets can
            # accumulate up to 6×0.0001 = 0.0006 rounding error.
            assert abs(total - 1.0) < 1e-3, (
                f"Weights sum to {total:.6f} at {record['date']}"
            )

    def test_backtest_has_required_columns(self, synthetic_prices):
        df, _ = self._run(synthetic_prices)
        for col in ["All Weather Value", "Buy & Hold All Weather", "S&P 500 Value"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_portfolio_value_stays_positive(self, synthetic_prices):
        df, _ = self._run(synthetic_prices)
        assert (df["All Weather Value"] > 0).all()

    def test_weight_history_dates_are_monotone(self, synthetic_prices):
        _, weight_history = self._run(synthetic_prices)
        dates = [pd.Timestamp(w["date"]) for w in weight_history]
        assert dates == sorted(dates), "Weight history dates are not monotonically increasing"

    def test_recomputation_count_matches_frequency(self, synthetic_prices):
        """Annual recompute over 10 years should give roughly 10 entries."""
        _, weight_history = self._run(synthetic_prices)
        # Allow ±2 for calendar edge effects
        assert 8 <= len(weight_history) <= 12, (
            f"Expected ~10 annual recomputations, got {len(weight_history)}"
        )


# ---------------------------------------------------------------------------
# OOS backtest filtering
# ---------------------------------------------------------------------------

class TestOOSFiltering:
    """Verify that filtering a full backtest to an OOS window gives valid stats."""

    def test_filtered_backtest_starts_at_oos_date(self, synthetic_prices):
        from backtest import run_backtest_rolling_rp
        oos_start = "2016-01-01"
        df, _ = run_backtest_rolling_rp(
            synthetic_prices, synthetic_prices["SPY"], TICKERS,
            rp_lookback_years=1.0, rp_recompute_freq="YS",
        )
        df_oos = df[df.index >= oos_start]
        assert str(df_oos.index[0].date()) >= oos_start

    def test_filtered_backtest_compute_stats(self, synthetic_prices):
        """compute_stats must succeed on an OOS-filtered backtest slice."""
        from backtest import compute_stats, run_backtest_rolling_rp
        oos_start = "2016-01-01"
        df, weight_history = run_backtest_rolling_rp(
            synthetic_prices, synthetic_prices["SPY"], TICKERS,
            rp_lookback_years=1.0, rp_recompute_freq="YS",
        )
        df_oos       = df[df.index >= oos_start].copy()
        last_weights = _latest_weights(weight_history)
        stats_list   = compute_stats(df_oos, prices=synthetic_prices,
                                     allocation=last_weights)
        aw = next(s for s in stats_list if s.name == "AW_R")
        assert isinstance(aw.calmar, float)
        assert aw.calmar > -100   # sanity: not an absurd number


# ---------------------------------------------------------------------------
# Win-counter logic (pure logic, no I/O)
# ---------------------------------------------------------------------------

def _count_rolling_wins(results: list[dict]) -> int:
    """Replicate the win-counting logic from main()."""
    wins = 0
    for i in range(0, len(results), 2):
        if results[i]["calmar"] > results[i + 1]["calmar"]:
            wins += 1
    return wins


class TestWinCounter:
    def test_rolling_wins_all_three(self):
        results = [
            {"split": "split2020", "method": "rolling_rp", "calmar": 1.5},
            {"split": "split2020", "method": "static_rp",  "calmar": 1.2},
            {"split": "split2018", "method": "rolling_rp", "calmar": 1.4},
            {"split": "split2018", "method": "static_rp",  "calmar": 1.1},
            {"split": "split2022", "method": "rolling_rp", "calmar": 1.6},
            {"split": "split2022", "method": "static_rp",  "calmar": 1.3},
        ]
        assert _count_rolling_wins(results) == 3

    def test_static_wins_all_three(self):
        results = [
            {"split": "split2020", "method": "rolling_rp", "calmar": 1.0},
            {"split": "split2020", "method": "static_rp",  "calmar": 1.2},
            {"split": "split2018", "method": "rolling_rp", "calmar": 0.9},
            {"split": "split2018", "method": "static_rp",  "calmar": 1.1},
            {"split": "split2022", "method": "rolling_rp", "calmar": 1.1},
            {"split": "split2022", "method": "static_rp",  "calmar": 1.3},
        ]
        assert _count_rolling_wins(results) == 0

    def test_rolling_wins_two_of_three(self):
        results = [
            {"split": "split2020", "method": "rolling_rp", "calmar": 1.5},
            {"split": "split2020", "method": "static_rp",  "calmar": 1.2},
            {"split": "split2018", "method": "rolling_rp", "calmar": 0.9},
            {"split": "split2018", "method": "static_rp",  "calmar": 1.1},
            {"split": "split2022", "method": "rolling_rp", "calmar": 1.6},
            {"split": "split2022", "method": "static_rp",  "calmar": 1.3},
        ]
        assert _count_rolling_wins(results) == 2

    def test_exact_tie_is_not_a_rolling_win(self):
        results = [
            {"split": "split2020", "method": "rolling_rp", "calmar": 1.5},
            {"split": "split2020", "method": "static_rp",  "calmar": 1.5},
        ]
        assert _count_rolling_wins(results) == 0

    def test_single_split_rolling_wins(self):
        results = [
            {"split": "split2020", "method": "rolling_rp", "calmar": 2.0},
            {"split": "split2020", "method": "static_rp",  "calmar": 1.8},
        ]
        assert _count_rolling_wins(results) == 1

    def test_single_split_static_wins(self):
        results = [
            {"split": "split2020", "method": "rolling_rp", "calmar": 1.5},
            {"split": "split2020", "method": "static_rp",  "calmar": 1.8},
        ]
        assert _count_rolling_wins(results) == 0
