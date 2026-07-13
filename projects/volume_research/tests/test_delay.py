"""Tests for the delay/timing-cost estimator.

Ground truth: forward returns over k minutes of a random walk with per-minute vol σ
should have std ≈ σ·√k (the √-time scaling). We build a synthetic intraday price
series with known σ and check the recovered std matches across horizons.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slippage.delay import delay_costs_bps, timing_risk_bps  # noqa: E402


def _synthetic_close(sigma_bps: float, n_days: int = 200, seed: int = 0) -> pd.Series:
    """One year-ish of 1-min closes: independent days, Gaussian per-minute returns."""
    rng = np.random.default_rng(seed)
    sessions = []
    for d in range(n_days):
        day = pd.Timestamp("2022-01-03") + pd.Timedelta(days=d)
        idx = pd.date_range(day + pd.Timedelta(hours=9, minutes=30),
                            periods=390, freq="1min")
        rets = rng.normal(0.0, sigma_bps / 1e4, len(idx))
        sessions.append(pd.Series(100.0 * np.exp(np.cumsum(rets)), index=idx))
    return pd.concat(sessions)


def test_std_scales_with_sqrt_horizon():
    sigma = 15.0  # bps per minute
    close = _synthetic_close(sigma)
    costs = delay_costs_bps(close, [1, 4, 9], decision_every_min=15)
    std = costs.std()
    for k in (1, 4, 9):
        assert std[k] == pytest.approx(sigma * np.sqrt(k), rel=0.10), (
            f"horizon {k}: std {std[k]:.1f} vs expected {sigma*np.sqrt(k):.1f}"
        )


def test_mean_is_approximately_zero():
    close = _synthetic_close(15.0)
    costs = delay_costs_bps(close, [1, 5])
    # mean drift should be small relative to the std (no systematic direction).
    summ = timing_risk_bps(costs)
    assert (summ["mean_bps"].abs() < 0.25 * summ["std_bps"]).all()


def test_no_window_crosses_overnight():
    # Forward return at the last decision of a day must not borrow the next day's open.
    close = _synthetic_close(15.0, n_days=3)
    costs = delay_costs_bps(close, [15], decision_every_min=15)
    # 15:45 decision + 15 min would land at 16:00 (past the 15:59 grid end) -> NaN/dropped.
    last_decisions = costs.index.time == pd.Timestamp("15:45").time()
    assert costs.loc[last_decisions, 15].isna().all() or last_decisions.sum() == 0


def test_summary_columns():
    close = _synthetic_close(15.0, n_days=20)
    summ = timing_risk_bps(delay_costs_bps(close, [1, 5]))
    assert list(summ.columns) == ["mean_bps", "std_bps", "p05_bps", "p95_bps"]
    assert (summ["std_bps"] > 0).all()
