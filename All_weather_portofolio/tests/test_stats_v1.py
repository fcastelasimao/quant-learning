"""
tests/test_stats.py
===================
Unit tests for the stat helper functions in backtest.py.

These functions are pure (no side effects, no network calls, no file I/O)
so they are the easiest to test and the highest-value place to start.

Run with:
    pytest tests/test_stats.py -v

Each test follows the Arrange / Act / Assert pattern:
    1. Arrange: set up known inputs with a mathematically verifiable answer
    2. Act:     call the function
    3. Assert:  verify the output matches the expected value

The tolerance in assert statements (e.g. abs(result - expected) < 0.01)
accounts for floating-point rounding differences.
"""

import sys
import os

# Add the project root to sys.path so we can import from the parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import pytest

from backtest import (
    compute_cagr,
    compute_max_drawdown,
    compute_sharpe,
    compute_calmar,
)


# ===========================================================================
# compute_cagr
# ===========================================================================

def test_cagr_known_doubling():
    """
    $10,000 doubling to $20,000 over 10 years.
    Verified by hand: 2^(1/10) - 1 = 7.177%
    """
    series = pd.Series(
        [10_000, 20_000],
        index=pd.date_range("2010-01-01", "2020-01-01", periods=2)
    )
    result = compute_cagr(series, years=10.0)
    assert abs(result - 7.177) < 0.01, f"Expected ~7.177%, got {result:.4f}%"


def test_cagr_no_growth():
    """A portfolio that ends where it started should have 0% CAGR."""
    series = pd.Series(
        [10_000, 10_000],
        index=pd.date_range("2010-01-01", "2020-01-01", periods=2)
    )
    result = compute_cagr(series, years=10.0)
    assert abs(result - 0.0) < 0.001, f"Expected 0.0%, got {result:.4f}%"


def test_cagr_quadrupling():
    """
    $10,000 quadrupling to $40,000 over 10 years.
    Verified by hand: 4^(1/10) - 1 = 14.87%
    """
    series = pd.Series(
        [10_000, 40_000],
        index=pd.date_range("2010-01-01", "2020-01-01", periods=2)
    )
    result = compute_cagr(series, years=10.0)
    assert abs(result - 14.869) < 0.01, f"Expected ~14.87%, got {result:.4f}%"


def test_cagr_negative_growth():
    """A portfolio that loses half its value should have a negative CAGR."""
    series = pd.Series(
        [10_000, 5_000],
        index=pd.date_range("2010-01-01", "2020-01-01", periods=2)
    )
    result = compute_cagr(series, years=10.0)
    assert result < 0.0, f"Expected negative CAGR, got {result:.4f}%"


# ===========================================================================
# compute_max_drawdown
# ===========================================================================

def test_max_drawdown_monotonically_increasing():
    """A portfolio that only ever rises should have 0% drawdown."""
    series = pd.Series([100, 110, 120, 130, 140])
    result = compute_max_drawdown(series)
    assert result == 0.0, f"Expected 0.0%, got {result:.4f}%"


def test_max_drawdown_known_crash():
    """
    Peak of 100, drops to 50, then recovers to 80.
    Max drawdown = (50 - 100) / 100 = -50%
    """
    series = pd.Series([100, 80, 50, 70, 80])
    result = compute_max_drawdown(series)
    assert abs(result - (-50.0)) < 0.001, f"Expected -50.0%, got {result:.4f}%"


def test_max_drawdown_end_crash():
    """Drawdown should be detected even if the crash is at the end."""
    series = pd.Series([100, 110, 120, 60])
    result = compute_max_drawdown(series)
    # Peak = 120, trough = 60, drawdown = (60-120)/120 = -50%
    assert abs(result - (-50.0)) < 0.001, f"Expected -50.0%, got {result:.4f}%"


def test_max_drawdown_multiple_crashes():
    """Returns the worst crash, not the most recent one."""
    series = pd.Series([100, 70, 90, 110, 55, 80])
    result = compute_max_drawdown(series)
    # Peak = 110, trough = 55, drawdown = (55-110)/110 = -50%
    assert abs(result - (-50.0)) < 0.001, f"Expected -50.0%, got {result:.4f}%"


def test_max_drawdown_returns_negative_number():
    """Max drawdown should always be <= 0."""
    series = pd.Series([100, 80, 120, 90, 110])
    result = compute_max_drawdown(series)
    assert result <= 0.0, f"Max drawdown should be <= 0, got {result:.4f}%"


# ===========================================================================
# compute_sharpe
# ===========================================================================

def test_sharpe_positive_consistent_returns():
    """
    Mostly positive returns with small variation should produce a positive Sharpe.
    Using slight variation so std > 0 and Sharpe is well-defined.
    """
    monthly_returns = pd.Series([1.0, 1.1, 0.9, 1.2, 0.8] * 12)
    result = compute_sharpe(monthly_returns)
    assert result > 0, f"Expected positive Sharpe, got {result:.4f}"


def test_sharpe_negative_consistent_returns():
    """
    Mostly negative returns with small variation should produce a negative Sharpe.
    """
    monthly_returns = pd.Series([-1.0, -1.1, -0.9, -1.2, -0.8] * 12)
    result = compute_sharpe(monthly_returns)
    assert result < 0, f"Expected negative Sharpe, got {result:.4f}"


def test_sharpe_zero_volatility():
    """
    If all monthly returns are identical, std = 0.
    The function should return 0.0 rather than dividing by zero.
    """
    monthly_returns = pd.Series([1.0] * 12)
    result = compute_sharpe(monthly_returns)
    assert result == 0.0, f"Expected 0.0 (zero std edge case), got {result:.4f}"


def test_sharpe_empty_after_dropna():
    """All-NaN series should return 0.0 without crashing."""
    monthly_returns = pd.Series([float("nan")] * 12)
    result = compute_sharpe(monthly_returns)
    assert result == 0.0, f"Expected 0.0 for all-NaN series, got {result:.4f}"


# ===========================================================================
# compute_calmar
# ===========================================================================

def test_calmar_basic():
    """
    CAGR of 10%, max drawdown of -20%.
    Calmar = 10 / 20 = 0.5
    """
    result = compute_calmar(cagr=10.0, max_drawdown=-20.0)
    assert abs(result - 0.5) < 0.001, f"Expected 0.5, got {result:.4f}"


def test_calmar_high_value():
    """CAGR of 15%, drawdown of -10% -> Calmar = 1.5"""
    result = compute_calmar(cagr=15.0, max_drawdown=-10.0)
    assert abs(result - 1.5) < 0.001, f"Expected 1.5, got {result:.4f}"


def test_calmar_zero_drawdown_returns_zero():
    """
    If drawdown is 0, Calmar is undefined (division by zero).
    The function should return 0.0 rather than crashing.
    """
    result = compute_calmar(cagr=10.0, max_drawdown=0.0)
    assert result == 0.0, f"Expected 0.0 (zero drawdown edge case), got {result:.4f}"


def test_calmar_negative_cagr():
    """Negative CAGR with negative drawdown should produce a negative Calmar."""
    result = compute_calmar(cagr=-5.0, max_drawdown=-20.0)
    assert result < 0, f"Expected negative Calmar for negative CAGR, got {result:.4f}"


def test_calmar_proportional():
    """Doubling the CAGR while keeping drawdown fixed should double the Calmar."""
    calmar_1 = compute_calmar(cagr=5.0,  max_drawdown=-20.0)
    calmar_2 = compute_calmar(cagr=10.0, max_drawdown=-20.0)
    assert abs(calmar_2 - 2 * calmar_1) < 0.001, \
        f"Expected calmar_2 = 2 * calmar_1, got {calmar_1:.4f} and {calmar_2:.4f}"
