"""Shared fixtures for the fnt test suite."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def synthetic_panel() -> pd.DataFrame:
    """~1500 trading days of synthetic OHLCV data for TQQQ, QQQ, SPY.

    Uses a simple GBM so metrics that require positive prices work correctly.
    """
    rng = np.random.default_rng(42)
    n = 1500
    dates = pd.bdate_range("2018-01-01", periods=n, freq="B")

    def _gbm(s0: float, mu: float, sigma: float) -> np.ndarray:
        log_rets = rng.normal(mu / 252, sigma / np.sqrt(252), n)
        prices = s0 * np.exp(np.cumsum(log_rets))
        return prices

    qqq = _gbm(200.0, 0.15, 0.18)
    spy = _gbm(300.0, 0.10, 0.14)
    tqqq = _gbm(40.0, 0.30, 0.55)   # 3× leverage approximation
    spxl = _gbm(50.0, 0.25, 0.42)
    hyg = _gbm(80.0, 0.04, 0.06)
    lqd = _gbm(110.0, 0.03, 0.05)
    # Simple VIX-like series
    vix = np.abs(rng.normal(20, 7, n)).clip(9, 80)
    vix3m = vix * rng.uniform(0.85, 1.15, n)
    tnx = np.abs(rng.normal(3.0, 0.8, n)).clip(0.1, 8.0)
    irx = np.abs(rng.normal(1.5, 0.6, n)).clip(0.0, 6.0)

    def _ohlcv(close: np.ndarray, vol_base: float = 1e7) -> dict:
        noise = rng.uniform(0.98, 1.02, (n, 2))
        high = close * np.maximum(noise[:, 0], 1.0)
        low = close * np.minimum(noise[:, 1], 1.0)
        open_ = close * rng.uniform(0.99, 1.01, n)
        volume = rng.uniform(0.5, 1.5, n) * vol_base
        return {"open": open_, "high": high, "low": low, "close": close, "volume": volume}

    data: dict[str, np.ndarray] = {}
    for sym, arr in [("TQQQ", tqqq), ("QQQ", qqq), ("SPY", spy), ("SPXL", spxl),
                     ("HYG", hyg), ("LQD", lqd)]:
        for field, vals in _ohlcv(arr).items():
            data[f"{sym}_{field}"] = vals

    # VIX / rates are single-value series (close only for index tickers)
    for sym, arr in [("^VIX", vix), ("^VIX3M", vix3m), ("^TNX", tnx), ("^IRX", irx)]:
        for field in ("open", "high", "low", "close", "volume"):
            if field == "close":
                data[f"{sym}_{field}"] = arr
            elif field == "volume":
                data[f"{sym}_{field}"] = np.zeros(n)
            else:
                data[f"{sym}_{field}"] = arr * rng.uniform(0.995, 1.005, n)

    return pd.DataFrame(data, index=dates)
