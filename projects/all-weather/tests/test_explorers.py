import numpy as np
import pandas as pd
import pytest

from engine.explorers import data_quality_report


def _frame_with_nans() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {
            "SPY": [100.0, np.nan, 102.0, np.nan, 104.0],
            "GLD": [50.0, 51.0, 52.0, 53.0, 54.0],
            "TLT": [np.nan, np.nan, np.nan, 80.0, 81.0],
        },
        index=idx,
    )


def test_missing_values_counted_correctly():
    report = data_quality_report(_frame_with_nans())
    assert report.loc["SPY", "Missing Values"] == 2
    assert report.loc["GLD", "Missing Values"] == 0
    assert report.loc["TLT", "Missing Values"] == 3


def test_first_and_last_dates_ignore_nans():
    report = data_quality_report(_frame_with_nans())
    assert str(report.loc["SPY", "First Date"]) == "2020-01-01"
    assert str(report.loc["SPY", "Last Date"]) == "2020-01-05"
    assert str(report.loc["TLT", "First Date"]) == "2020-01-04"
    assert str(report.loc["TLT", "Last Date"]) == "2020-01-05"


def test_all_nan_column_returns_none_dates():
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    df = pd.DataFrame({"A": [np.nan, np.nan, np.nan]}, index=idx)
    report = data_quality_report(df)
    assert report.loc["A", "First Date"] is None
    assert report.loc["A", "Last Date"] is None
    assert report.loc["A", "Missing Values"] == 3


def test_output_index_matches_input_columns():
    df = _frame_with_nans()
    report = data_quality_report(df)
    assert list(report.index) == list(df.columns)


def test_output_columns_present():
    report = data_quality_report(_frame_with_nans())
    assert {"Missing Values", "First Date", "Last Date"} <= set(report.columns)
