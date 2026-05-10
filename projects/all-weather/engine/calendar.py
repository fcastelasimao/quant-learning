"""
calendar.py
===========
Small date/frequency helpers shared by backtests, analytics, and research.
"""

from __future__ import annotations

from pandas.tseries.frequencies import to_offset


def pandas_resample_frequency(freq: str) -> str:
    """
    Return a pandas resample alias that works in the installed pandas version.

    pandas 3 uses "ME" for month-end. Older pandas versions use "M". Keeping
    "ME" in config is clearer, but callers should pass this helper's return
    value into DataFrame.resample().
    """
    try:
        to_offset(freq)
        return freq
    except ValueError:
        if freq == "ME":
            to_offset("M")
            return "M"
        if freq == "M":
            to_offset("ME")
            return "ME"
        raise


def frequency_annualisation(freq: str) -> int:
    """Return the return-series annualisation factor for a configured frequency."""
    if freq in {"ME", "M"}:
        return 12
    if freq == "W":
        return 52
    raise ValueError(f"Unsupported data frequency: {freq}")
