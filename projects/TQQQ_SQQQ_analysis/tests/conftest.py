"""Shared pytest fixtures and path constants for the TQQQ/SQQQ research repo.

Run from project root:
    ~/opt/anaconda3/envs/quant/bin/python -m pytest tests/ -v

Tests are integration-style: they load the cached CSVs the build scripts emit
and verify invariants + reproducibility of headline numbers. Catching real
bugs is the point; mocking is avoided.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


PROJ = Path(__file__).resolve().parent.parent
CANON = PROJ / "full_history_canonical"
RESEARCH = PROJ / "research"

# Make _rule_naming importable by tests that need it
sys.path.insert(0, str(RESEARCH))


@pytest.fixture(scope="session")
def proj_root() -> Path:
    return PROJ


@pytest.fixture(scope="session")
def canon_dir() -> Path:
    return CANON


@pytest.fixture(scope="session")
def research_dir() -> Path:
    return RESEARCH


@pytest.fixture(scope="session")
def canon_tqqq() -> pd.DataFrame:
    path = CANON / "TRADES_TQQQ_full_history.csv"
    if not path.exists():
        pytest.skip("canonical trade data not present")
    return pd.read_csv(path, parse_dates=["entry_time", "exit_time"])


@pytest.fixture(scope="session")
def canon_sqqq() -> pd.DataFrame:
    path = CANON / "TRADES_SQQQ_full_history.csv"
    if not path.exists():
        pytest.skip("canonical trade data not present")
    return pd.read_csv(path, parse_dates=["entry_time", "exit_time"])


@pytest.fixture(scope="session")
def regime_tqqq(canon_tqqq) -> pd.DataFrame:
    return canon_tqqq[canon_tqqq["regime_entry"].notna()].copy()


@pytest.fixture(scope="session")
def regime_sqqq(canon_sqqq) -> pd.DataFrame:
    return canon_sqqq[canon_sqqq["regime_entry"].notna()].copy()
