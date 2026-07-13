"""Tests for src/fnt/data.py."""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data import load_panel, load_symbol, _FIELDS


def _make_db(tmp_path: Path, symbol: str, n: int = 50) -> Path:
    """Write a minimal candles_1d SQLite DB for testing."""
    db_path = tmp_path / f"DB_{symbol}_historical_data.db"
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2023-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))

    rows = [
        (
            int(d.timestamp()),
            float(close[i] * 0.99),
            float(close[i] * 1.01),
            float(close[i] * 0.98),
            float(close[i]),
            float(1e6),
            d.strftime("%Y-%m-%d %H:%M:%S"),
            d.strftime("%Y-%m-%d %H:%M:%S"),
        )
        for i, d in enumerate(dates)
    ]

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE candles_1d "
            "(ts INTEGER PRIMARY KEY, open REAL, high REAL, low REAL, "
            "close REAL, volume REAL, utc_datetime TEXT, et_datetime TEXT)"
        )
        conn.executemany("INSERT INTO candles_1d VALUES (?,?,?,?,?,?,?,?)", rows)
        conn.execute("CREATE INDEX idx ON candles_1d(ts)")
    return db_path


def test_load_symbol_returns_correct_columns(tmp_path):
    _make_db(tmp_path, "QQQ")
    df = load_symbol("QQQ", data_dir=tmp_path)
    assert not df.empty
    assert all(f"QQQ_{f}" in df.columns for f in _FIELDS)


def test_load_symbol_missing_returns_empty(tmp_path):
    df = load_symbol("FAKE", data_dir=tmp_path)
    assert df.empty


def test_load_symbol_no_duplicate_dates(tmp_path):
    _make_db(tmp_path, "SPY")
    df = load_symbol("SPY", data_dir=tmp_path)
    assert df.index.is_unique, "Duplicate ET dates found"


def test_load_panel_wide_alignment(tmp_path):
    _make_db(tmp_path, "TQQQ", n=30)
    _make_db(tmp_path, "QQQ", n=50)
    panel = load_panel(["TQQQ", "QQQ"], data_dir=tmp_path, warn_missing=False)
    assert "TQQQ_close" in panel.columns
    assert "QQQ_close" in panel.columns
    assert panel.index.is_monotonic_increasing


def test_load_panel_missing_symbol_omitted(tmp_path):
    _make_db(tmp_path, "QQQ")
    with pytest.warns(UserWarning, match="SPY"):
        panel = load_panel(["QQQ", "SPY"], data_dir=tmp_path, warn_missing=True)
    assert "QQQ_close" in panel.columns
    assert not any("SPY" in c for c in panel.columns)


def test_load_panel_all_missing_raises(tmp_path):
    with pytest.raises(ValueError):
        load_panel(["FAKE1", "FAKE2"], data_dir=tmp_path, warn_missing=False)


def test_no_large_single_day_jump(tmp_path):
    """TQQQ data must not have >50% single-day gaps after split adjustment."""
    _make_db(tmp_path, "TQQQ", n=100)
    df = load_symbol("TQQQ", data_dir=tmp_path)
    daily_ret = df["TQQQ_close"].pct_change(fill_method=None).abs()
    assert (daily_ret > 0.5).sum() == 0, (
        "Possible unadjusted stock split detected in TQQQ data."
    )
