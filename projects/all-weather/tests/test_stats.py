"""
tests/test_stats.py
===================
Unit tests for the four stat helper functions in backtest.py:
  - compute_cagr
  - compute_max_drawdown
  - compute_sharpe
  - compute_calmar

These functions are pure (no side effects, no network calls, no file I/O)
so they are the easiest to test and the highest-value place to start.

Run with:
    pytest tests/test_stats.py -v

Each test follows the Arrange / Act / Assert pattern:
    1. Arrange: set up known inputs with a mathematically verifiable answer
    2. Act:     call the function
    3. Assert:  verify the output matches the expected value

Shared test fixtures (series inputs) are defined in conftest.py and injected
automatically by pytest -- no import needed.
"""

import pandas as pd
import numpy as np
import pytest

from backtest import (
    compute_cagr,
    compute_max_drawdown,
    compute_sharpe,
    compute_calmar,
    compute_avg_drawdown,
    compute_max_drawdown_duration,
    compute_avg_recovery_time,
    compute_ulcer_index,
    compute_sortino,
)

# Single tolerance constant used by all assertions.
# Defined here so it can be tightened or loosened globally in one place.
TOLERANCE = 1e-3

# Import the helper from conftest (not a fixture, just a utility function)
from conftest import make_cagr_series


# ===========================================================================
# compute_cagr -- parametrised
# ===========================================================================
#
# Using pytest.mark.parametrize to express multiple cases as a table rather
# than duplicating the same boilerplate across four separate test functions.
# Each row (start, end, years, expected, description) becomes one named test
# in the pytest output.
#
# Expected values verified by hand:
#   Doubling:    2^(1/10) - 1        = 7.177%
#   No growth:   1^(1/10) - 1        = 0.0%
#   Quadrupling: 4^(1/10) - 1        = 14.869%
#   Halving:     0.5^(1/10) - 1      = -6.697%
#   Large vals:  same as doubling     = 7.177%

@pytest.mark.parametrize("start,end,years,expected,description", [
    (10_000,   20_000, 10.0,   7.177, "doubling over 10 years"),
    (10_000,   10_000, 10.0,   0.000, "no growth over 10 years"),
    (10_000,   40_000, 10.0,  14.869, "quadrupling over 10 years"),
    (10_000,    5_000, 10.0,  -6.697, "halving over 10 years"),
    (1_000_000, 2_000_000, 10.0, 7.177, "large values -- no overflow"),
])
def test_cagr_parametrised(start, end, years, expected, description):
    """
    Parametrised CAGR tests covering growth, no growth, decline, and
    large values. Each case has a mathematically verified expected value.
    """
    series = make_cagr_series(start, end)
    result = compute_cagr(series, years=years)
    assert abs(result - expected) < 0.01, \
        f"[{description}] Expected {expected:.3f}%, got {result:.4f}%"


def test_cagr_negative_growth_exact():
    """
    Halving over 10 years = -6.697% CAGR.
    Checks the exact value, not just the sign.
    Verified: 0.5^(1/10) - 1 = -0.06697 = -6.697%
    """
    series = make_cagr_series(10_000, 5_000)
    result = compute_cagr(series, years=10.0)
    assert abs(result - (-6.697)) < 0.01, \
        f"Expected -6.697%, got {result:.4f}%"


def test_cagr_short_period():
    """CAGR over 1 year should equal the simple return."""
    series = make_cagr_series(10_000, 11_000)
    result = compute_cagr(series, years=1.0)
    assert abs(result - 10.0) < TOLERANCE, \
        f"Expected 10.0%, got {result:.4f}%"


# ===========================================================================
# compute_max_drawdown
# ===========================================================================

def test_max_drawdown_monotonically_increasing(rising_series):
    """A portfolio that only ever rises should have 0% drawdown."""
    result = compute_max_drawdown(rising_series)
    assert result == 0.0, f"Expected 0.0%, got {result:.4f}%"


def test_max_drawdown_known_crash(crash_series):
    """
    Peak of 100, drops to 50, then recovers to 80.
    Max drawdown = (50 - 100) / 100 = -50%
    """
    result = compute_max_drawdown(crash_series)
    assert abs(result - (-50.0)) < TOLERANCE, \
        f"Expected -50.0%, got {result:.4f}%"


def test_max_drawdown_end_crash(end_crash_series):
    """Drawdown should be detected even if the trough is the final value."""
    result = compute_max_drawdown(end_crash_series)
    assert abs(result - (-50.0)) < TOLERANCE, \
        f"Expected -50.0%, got {result:.4f}%"


def test_max_drawdown_multiple_crashes(multiple_crash_series):
    """Returns the worst crash, not the most recent or the first one."""
    result = compute_max_drawdown(multiple_crash_series)
    # Peak = 110, trough = 55 -> (55-110)/110 = -50%
    assert abs(result - (-50.0)) < TOLERANCE, \
        f"Expected -50.0%, got {result:.4f}%"


def test_max_drawdown_dip_at_start(dip_at_start_series):
    """
    Drawdown should be detected even at position 1 (immediately after start).
    Peak = 100, trough = 60 -> (60-100)/100 = -40%
    Replaces the previous redundant test_max_drawdown_returns_negative_number.
    """
    result = compute_max_drawdown(dip_at_start_series)
    assert abs(result - (-40.0)) < TOLERANCE, \
        f"Expected -40.0%, got {result:.4f}%"


def test_max_drawdown_always_non_positive(rising_series):
    """Max drawdown should always be <= 0 for any real series."""
    result = compute_max_drawdown(rising_series)
    assert result <= 0.0, \
        f"Max drawdown should be <= 0, got {result:.4f}%"


def test_max_drawdown_exact_percentage():
    """
    Verify a non-round drawdown value precisely.
    Peak = 150, trough = 90 -> (90-150)/150 = -40%
    """
    series = pd.Series([100, 150, 90, 120])
    result = compute_max_drawdown(series)
    assert abs(result - (-40.0)) < TOLERANCE, \
        f"Expected -40.0%, got {result:.4f}%"


# ===========================================================================
# compute_sharpe
# ===========================================================================

def test_sharpe_positive_returns(positive_monthly_returns):
    """Mostly positive returns with variation should produce a positive Sharpe."""
    result = compute_sharpe(positive_monthly_returns)
    assert result > 0, f"Expected positive Sharpe, got {result:.4f}"


def test_sharpe_negative_returns(negative_monthly_returns):
    """Mostly negative returns with variation should produce a negative Sharpe."""
    result = compute_sharpe(negative_monthly_returns)
    assert result < 0, f"Expected negative Sharpe, got {result:.4f}"


def test_sharpe_zero_volatility_returns_zero(zero_volatility_returns):
    """
    Identical returns -> std = 0 (or near-zero due to floating point).
    Sharpe is undefined -- function should return 0.0 rather than crashing
    or returning a huge nonsense number.
    """
    result = compute_sharpe(zero_volatility_returns)
    assert result == 0.0, \
        f"Expected 0.0 (zero std edge case), got {result:.4f}"


def test_sharpe_all_nan_returns_zero(nan_returns):
    """
    All-NaN series -> empty after dropna().
    Function should return 0.0 rather than returning nan or crashing.
    """
    result = compute_sharpe(nan_returns)
    assert result == 0.0, \
        f"Expected 0.0 for all-NaN series, got {result}"


def test_sharpe_higher_return_means_higher_sharpe():
    """
    With identical volatility, higher mean return should produce higher Sharpe.
    This tests the directional relationship, not an exact value.
    """
    low_returns  = pd.Series([0.5, 0.6, 0.4, 0.7, 0.3] * 12)
    high_returns = pd.Series([1.0, 1.1, 0.9, 1.2, 0.8] * 12)
    assert compute_sharpe(high_returns) > compute_sharpe(low_returns), \
        "Higher mean return should produce higher Sharpe ratio"


def test_sharpe_annualisation():
    """
    Sharpe should be annualised by sqrt(12).
    With known mean and std we can verify the exact value.
    """
    # Returns: alternating 2% and 0%, mean=1%, std=1%
    returns = pd.Series([2.0, 0.0] * 30)
    r       = returns / 100
    expected = (r.mean() / r.std()) * np.sqrt(12)
    result   = compute_sharpe(returns)
    assert abs(result - expected) < TOLERANCE, \
        f"Expected {expected:.4f}, got {result:.4f}"


# ===========================================================================
# compute_calmar
# ===========================================================================

@pytest.mark.parametrize("cagr,drawdown,expected,description", [
    (10.0, -20.0,  0.5,  "basic: 10% CAGR / 20% DD = 0.5"),
    (15.0, -10.0,  1.5,  "high Calmar: 15% CAGR / 10% DD = 1.5"),
    (5.0,  -20.0,  0.25, "low Calmar: 5% CAGR / 20% DD = 0.25"),
    (-5.0, -20.0, -0.25, "negative CAGR produces negative Calmar"),
])
def test_calmar_parametrised(cagr, drawdown, expected, description):
    """
    Parametrised Calmar tests covering normal, high, low, and negative cases.
    Calmar = CAGR / |max_drawdown|
    """
    result = compute_calmar(cagr=cagr, max_drawdown=drawdown)
    assert abs(result - expected) < TOLERANCE, \
        f"[{description}] Expected {expected:.4f}, got {result:.4f}"


def test_calmar_zero_drawdown_returns_zero():
    """
    Zero drawdown makes Calmar undefined (division by zero).
    Function should return 0.0 rather than raising ZeroDivisionError.
    """
    result = compute_calmar(cagr=10.0, max_drawdown=0.0)
    assert result == 0.0, \
        f"Expected 0.0 (zero drawdown edge case), got {result:.4f}"


def test_calmar_proportional_to_cagr():
    """
    Doubling CAGR with fixed drawdown should exactly double the Calmar.
    Tests the linear relationship between CAGR and Calmar.
    """
    calmar_1 = compute_calmar(cagr=5.0,  max_drawdown=-20.0)
    calmar_2 = compute_calmar(cagr=10.0, max_drawdown=-20.0)
    assert abs(calmar_2 - 2 * calmar_1) < TOLERANCE, \
        f"Expected calmar_2 = 2 * calmar_1, got {calmar_1:.4f} and {calmar_2:.4f}"


def test_calmar_inversely_proportional_to_drawdown():
    """
    Doubling the drawdown magnitude with fixed CAGR should halve the Calmar.
    Tests the inverse relationship between drawdown and Calmar.
    """
    calmar_1 = compute_calmar(cagr=10.0, max_drawdown=-10.0)
    calmar_2 = compute_calmar(cagr=10.0, max_drawdown=-20.0)
    assert abs(calmar_2 - calmar_1 / 2) < TOLERANCE, \
        f"Expected calmar_2 = calmar_1 / 2, got {calmar_1:.4f} and {calmar_2:.4f}"


def test_calmar_positive_drawdown_input():
    """
    compute_calmar uses abs(max_drawdown) internally.
    Passing a positive drawdown value (incorrect usage) should still
    produce the same result as passing the equivalent negative value.
    """
    result_negative = compute_calmar(cagr=10.0, max_drawdown=-20.0)
    result_positive = compute_calmar(cagr=10.0, max_drawdown=20.0)
    assert abs(result_negative - result_positive) < TOLERANCE, \
        "Calmar should be the same whether drawdown is passed as positive or negative"


# ===========================================================================
# compute_avg_drawdown
# ===========================================================================

def test_avg_drawdown_never_draws_down(rising_series):
    """A portfolio that only rises should have avg drawdown of 0.0."""
    result = compute_avg_drawdown(rising_series)
    assert result == 0.0, f"Expected 0.0 for rising series, got {result}"


def test_avg_drawdown_known_values():
    """
    Series: 100, 80, 50, 70, 80
    Peak at each point: 100, 100, 100, 100, 100
    Drawdowns: 0%, -20%, -50%, -30%, -20%  => underwater: -20, -50, -30, -20
    Mean of underwater = (-20 + -50 + -30 + -20) / 4 = -30%
    """
    series = pd.Series([100, 80, 50, 70, 80])
    result = compute_avg_drawdown(series)
    assert abs(result - (-30.0)) < TOLERANCE, \
        f"Expected -30.0%, got {result}"


def test_avg_drawdown_is_negative(crash_series):
    """Avg drawdown must be <= 0 for any series with a crash."""
    result = compute_avg_drawdown(crash_series)
    assert result <= 0.0, f"Avg drawdown must be <= 0, got {result}"


def test_avg_drawdown_less_severe_than_max(crash_series):
    """Average drawdown magnitude should be <= max drawdown magnitude."""
    avg = compute_avg_drawdown(crash_series)
    mdd = compute_max_drawdown(crash_series)
    assert avg >= mdd, (
        f"Avg drawdown ({avg}) should be >= max drawdown ({mdd}) "
        f"since average is less extreme than the worst point"
    )


# ===========================================================================
# compute_max_drawdown_duration
# ===========================================================================

def test_max_dd_duration_never_draws_down(rising_series):
    """A portfolio that only rises should have duration of 0."""
    result = compute_max_drawdown_duration(rising_series)
    assert result == 0, f"Expected 0 for rising series, got {result}"


def test_max_dd_duration_known_crash(crash_series):
    """
    Series: 100, 80, 50, 70, 80
    Peak=100; underwater at positions 1,2,3,4 => max run = 4 consecutive periods.
    """
    result = compute_max_drawdown_duration(crash_series)
    assert result == 4, f"Expected 4, got {result}"


def test_max_dd_duration_two_separate_crashes():
    """
    Series: 100, 70, 90, 80, 100, 60, 80
    First underwater run: positions 1,2,3 (3 periods, recovers at pos 4)
    Second underwater run: position 5,6 (2 periods)
    Max duration = 3
    """
    series = pd.Series([100, 70, 90, 80, 100, 60, 80])
    result = compute_max_drawdown_duration(series)
    assert result == 3, f"Expected 3, got {result}"


def test_max_dd_duration_returns_integer(crash_series):
    """Return type must be int, not float."""
    result = compute_max_drawdown_duration(crash_series)
    assert isinstance(result, int), f"Expected int, got {type(result)}"


# ===========================================================================
# compute_avg_recovery_time
# ===========================================================================

def test_avg_recovery_time_never_draws_down(rising_series):
    """No drawdown episodes -> no recovery events -> return 0.0."""
    result = compute_avg_recovery_time(rising_series)
    assert result == 0.0, f"Expected 0.0, got {result}"


def test_avg_recovery_time_no_recovery():
    """
    Portfolio that crashes and never recovers.
    No completed recovery episodes -> return 0.0.
    """
    series = pd.Series([100, 80, 60, 50, 40])
    result = compute_avg_recovery_time(series)
    assert result == 0.0, f"Expected 0.0 (never recovered), got {result}"


def test_avg_recovery_time_known_episode():
    """
    Series: 100, 80, 50, 70, 100, 90, 100
    Peak is always 100 (never exceeded after start).

    Episode 1: positions 1, 2, 3 are underwater (3 periods); recovers at pos 4.
    Episode 2: position 5 is underwater (1 period); recovers at pos 6.

    Avg = (3 + 1) / 2 = 2.0
    Note: the function counts periods SPENT underwater, not including the
    recovery period itself.
    """
    series = pd.Series([100, 80, 50, 70, 100, 90, 100])
    result = compute_avg_recovery_time(series)
    assert abs(result - 2.0) < TOLERANCE, f"Expected 2.0, got {result}"


def test_avg_recovery_time_non_negative(crash_series):
    """Recovery time must always be >= 0."""
    result = compute_avg_recovery_time(crash_series)
    assert result >= 0.0, f"Recovery time must be >= 0, got {result}"


# ===========================================================================
# compute_ulcer_index
# ===========================================================================

def test_ulcer_index_never_draws_down(rising_series):
    """A portfolio that only rises has 0% drawdown everywhere -> Ulcer = 0.0."""
    result = compute_ulcer_index(rising_series)
    assert result == 0.0, f"Expected 0.0 for rising series, got {result}"


def test_ulcer_index_non_negative(crash_series):
    """Ulcer Index is an RMS value and must always be >= 0."""
    result = compute_ulcer_index(crash_series)
    assert result >= 0.0, f"Ulcer Index must be >= 0, got {result}"


def test_ulcer_index_known_value():
    """
    Series: 100, 80, 100
    Peak series: 100, 100, 100
    Drawdown pct at each step: 0%, -20%, 0%
    Ulcer = sqrt(mean([0**2, (-20)**2, 0**2])) = sqrt(400/3) ≈ 11.547
    """
    series = pd.Series([100, 80, 100])
    result = compute_ulcer_index(series)
    expected = float(np.sqrt((0**2 + 20**2 + 0**2) / 3))
    assert abs(result - expected) < TOLERANCE, \
        f"Expected {expected:.4f}, got {result}"


def test_ulcer_index_worse_for_deeper_crash():
    """A deeper crash should produce a higher Ulcer Index than a shallow one."""
    shallow = pd.Series([100, 90, 100])
    deep    = pd.Series([100, 50, 100])
    assert compute_ulcer_index(deep) > compute_ulcer_index(shallow), \
        "Deeper crash should produce higher Ulcer Index"


# ===========================================================================
# compute_sortino
# ===========================================================================

def test_sortino_no_negative_returns():
    """
    All returns positive -> no downside -> return 0.0 (undefined, not infinity).
    """
    returns = pd.Series([1.0, 2.0, 1.5, 3.0] * 6)
    result  = compute_sortino(returns)
    assert result == 0.0, f"Expected 0.0 (no downside), got {result}"


def test_sortino_empty_series():
    """Empty series after dropna -> return 0.0."""
    result = compute_sortino(pd.Series([float("nan")] * 10))
    assert result == 0.0, f"Expected 0.0 for all-NaN series, got {result}"


def test_sortino_positive_for_positive_mean():
    """
    Mostly positive returns with some downside should produce Sortino > 0.
    """
    returns = pd.Series([2.0, -0.5, 3.0, -0.3, 1.5] * 12)
    result  = compute_sortino(returns)
    assert result > 0, f"Expected Sortino > 0 for positive mean returns, got {result}"


def test_sortino_negative_for_negative_mean():
    """
    Mostly negative returns should produce Sortino < 0.
    """
    returns = pd.Series([-2.0, -1.5, -3.0, 0.5, -1.0] * 12)
    result  = compute_sortino(returns)
    assert result < 0, f"Expected Sortino < 0 for negative mean returns, got {result}"


def test_sortino_higher_than_sharpe_with_asymmetric_returns():
    """
    When upside volatility is higher than downside (left-skewed upside),
    Sortino should be higher than Sharpe because it doesn't penalise good months.
    """
    # Large positive months, small negative months: Sortino > Sharpe
    returns = pd.Series([5.0, -0.5, 6.0, -0.3, 4.0, -0.4] * 10)
    sortino = compute_sortino(returns)
    sharpe  = compute_sharpe(returns)
    assert sortino > sharpe, (
        f"Sortino ({sortino:.3f}) should exceed Sharpe ({sharpe:.3f}) "
        f"when positive months dominate volatility"
    )
