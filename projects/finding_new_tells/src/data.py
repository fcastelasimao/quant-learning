"""SQLite OHLCV loader → aligned wide DataFrame for fnt research.

Thin shim over ``quantcore.data``.  All heavy lifting is in the shared
engine; this module re-exports the public API that notebooks and src/
modules import, and defines the project-specific symbol groups.
"""
from __future__ import annotations

from pathlib import Path

from quantcore import config as _qc_config
from quantcore.data import load_panel, load_symbol  # noqa: F401 — re-export

DATA_DIR: Path = _qc_config.data_dir()

_FIELDS = ["open", "high", "low", "close", "volume"]

# Canonical symbol groups used across the project
SYMBOLS_CORE = ["TQQQ", "QQQ", "SPY", "SPXL"]
SYMBOLS_VOL = ["^VIX", "^VIX3M"]          # preferred; fallback: ["VIXY", "VXZ"]
SYMBOLS_RATES = ["^TNX", "^IRX"]           # preferred; fallback: ["TLT", "SHY"]
SYMBOLS_CREDIT = ["HYG", "LQD"]
SYMBOLS_ALL = SYMBOLS_CORE + SYMBOLS_VOL + SYMBOLS_RATES + SYMBOLS_CREDIT
