"""
tests/conftest.py
=================
Shared pytest fixtures available to all test files in the tests/ folder.
pytest discovers this file automatically -- no import needed in test files.

Fixtures are reusable test inputs defined once here rather than duplicated
across individual test functions. Each fixture is a function decorated with
@pytest.fixture that returns a value. Test functions that declare a fixture
as a parameter receive that value automatically when pytest runs them.
"""


def pytest_configure(config):
    """Register custom markers so pytest does not warn about unknown marks."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as requiring network access (yfinance); "
        "skip offline with: pytest -m 'not integration'",
    )

import sys
import os

# Add the project root to sys.path so all test files can import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import pytest


# ===========================================================================
# PRICE SERIES FIXTURES
# ===========================================================================

@pytest.fixture
def rising_series():
    """A portfolio that only ever rises -- no drawdown at any point."""
    return pd.Series([100, 110, 120, 130, 140])


@pytest.fixture
def crash_series():
    """
    A portfolio that drops 50% from its peak of 100 to a trough of 50,
    then partially recovers to 80.
    Max drawdown = (50 - 100) / 100 = -50%
    """
    return pd.Series([100, 80, 50, 70, 80])


@pytest.fixture
def end_crash_series():
    """
    A portfolio that rises to a peak of 120 then crashes to 60 at the end.
    Tests that drawdown is detected even when the trough is the final value.
    Max drawdown = (60 - 120) / 120 = -50%
    """
    return pd.Series([100, 110, 120, 60])


@pytest.fixture
def multiple_crash_series():
    """
    A portfolio with two distinct crashes. The second is worse.
    First crash:  100 -> 70 = -30%
    Second crash: 110 -> 55 = -50%  <-- this is the max drawdown
    """
    return pd.Series([100, 70, 90, 110, 55, 80])


@pytest.fixture
def dip_at_start_series():
    """
    A portfolio that dips immediately from the first value.
    Tests that drawdown at position 1 (not just interior positions) is detected.
    Max drawdown = (60 - 100) / 100 = -40%
    """
    return pd.Series([100, 60, 80, 90, 95])


# ===========================================================================
# RETURN SERIES FIXTURES
# ===========================================================================

@pytest.fixture
def positive_monthly_returns():
    """
    Mostly positive monthly returns with small variation.
    std > 0 so Sharpe is well-defined. Mean is positive so Sharpe > 0.
    """
    return pd.Series([1.0, 1.1, 0.9, 1.2, 0.8] * 12)


@pytest.fixture
def negative_monthly_returns():
    """
    Mostly negative monthly returns with small variation.
    std > 0 so Sharpe is well-defined. Mean is negative so Sharpe < 0.
    """
    return pd.Series([-1.0, -1.1, -0.9, -1.2, -0.8] * 12)


@pytest.fixture
def zero_volatility_returns():
    """
    All monthly returns are identical. std = 0 (or near-zero due to floats).
    Sharpe is undefined -- function should return 0.0 rather than crashing.
    """
    return pd.Series([1.0] * 12)


@pytest.fixture
def nan_returns():
    """
    All values are NaN. After dropna() the series is empty.
    Function should return 0.0 rather than crashing.
    """
    return pd.Series([float("nan")] * 12)


# ===========================================================================
# CAGR SERIES FACTORY
# ===========================================================================

def make_cagr_series(start: float, end: float) -> pd.Series:
    """
    Helper function (not a fixture) to create a two-element Series
    for CAGR tests. Not decorated with @pytest.fixture because parametrised
    tests need to call it with different values directly.
    """
    return pd.Series(
        [start, end],
        index=pd.date_range("2010-01-01", "2020-01-01", periods=2)
    )
