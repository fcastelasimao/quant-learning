"""Tests for slippage/state.py — MarketState and estimate_state.

Two layers: synthetic recovery of the pure functions (no data needed), and a DB-guarded sanity
check against real data (skipped cleanly when the DBs aren't present), mirroring
test_calibrate_live.py's pattern.
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slippage.state import (  # noqa: E402
    MarketState, VolumeProfile, SpreadCurve, VolRegimeBounds,
    session_bin_label, sigma_now_bps, classify_regime, estimate_state,
)


# --------------------------------------------------------------------------- session_bin_label
def test_bin_label_floors_to_15min():
    assert session_bin_label(pd.Timestamp("2026-01-05 09:37:00")) == "09:30"
    assert session_bin_label(pd.Timestamp("2026-01-05 09:44:59")) == "09:30"
    assert session_bin_label(pd.Timestamp("2026-01-05 09:45:00")) == "09:45"
    assert session_bin_label(pd.Timestamp("2026-01-05 15:59:00")) == "15:45"


# --------------------------------------------------------------------------- sigma_now_bps
def test_sigma_now_recovers_known_std():
    rng = np.random.default_rng(0)
    true_std = 0.001   # 10 bps per-minute std
    r = rng.normal(0, true_std, size=5000)
    sigma = sigma_now_bps(pd.Series(r), minutes_per_day=390)
    expected = true_std * np.sqrt(390) * 1e4
    assert sigma == pytest.approx(expected, rel=0.05)


def test_sigma_now_nan_below_two_obs():
    assert np.isnan(sigma_now_bps(pd.Series([0.001])))
    assert np.isnan(sigma_now_bps(pd.Series([], dtype=float)))


# --------------------------------------------------------------------------- classify_regime
def test_classify_regime_boundaries():
    bounds = VolRegimeBounds(q1_bps=100.0, q2_bps=200.0)
    assert classify_regime(50.0, bounds) == "calm"
    assert classify_regime(100.0, bounds) == "calm"
    assert classify_regime(150.0, bounds) == "normal"
    assert classify_regime(250.0, bounds) == "stress"


def test_classify_regime_nan_falls_back_to_normal():
    bounds = VolRegimeBounds(q1_bps=100.0, q2_bps=200.0)
    assert classify_regime(float("nan"), bounds) == "normal"


# --------------------------------------------------------------------------- estimate_state
def test_estimate_state_recovers_synthetic_inputs():
    profile = VolumeProfile(
        bin_share_mean={"09:30": 0.10, "12:30": 0.03},
        bin_share_p10={"09:30": 0.06, "12:30": 0.01},
        bin_share_p20={"09:30": 0.08, "12:30": 0.015},
    )
    spread = SpreadCurve(bin_half_spread_bps={"09:30": 2.5, "12:30": 0.6})
    bounds = VolRegimeBounds(q1_bps=100.0, q2_bps=200.0)

    rng = np.random.default_rng(1)
    calm_returns = pd.Series(rng.normal(0, 0.0003, 200))   # ~50 bps daily-equiv -> calm

    st = estimate_state(
        pd.Timestamp("2026-01-05 09:37:00"), "TQQQ",
        recent_1min_returns=calm_returns, trailing_daily_volume=1_000_000.0,
        profile=profile, spread_curve=spread, vol_bounds=bounds,
    )
    assert isinstance(st, MarketState)
    assert st.bin_label == "09:30"
    assert st.expected_interval_volume == pytest.approx(0.10 * 1_000_000.0)
    assert st.thin_volume_p10 == pytest.approx(0.06 * 1_000_000.0)
    assert st.thin_volume_p20 == pytest.approx(0.08 * 1_000_000.0)
    assert st.regime == "calm"
    assert st.spread_bps == pytest.approx(2.5)


def test_estimate_state_live_spread_overrides_curve():
    profile = VolumeProfile(bin_share_mean={"09:30": 0.10}, bin_share_p10={"09:30": 0.06},
                            bin_share_p20={"09:30": 0.08})
    spread = SpreadCurve(bin_half_spread_bps={"09:30": 2.5})
    bounds = VolRegimeBounds(q1_bps=100.0, q2_bps=200.0)
    st = estimate_state(
        pd.Timestamp("2026-01-05 09:31:00"), "TQQQ",
        recent_1min_returns=pd.Series([0.0001, -0.0001, 0.0002]),
        trailing_daily_volume=1_000_000.0, profile=profile, spread_curve=spread,
        vol_bounds=bounds, live_spread_bps=0.9,
    )
    assert st.spread_bps == pytest.approx(0.9)


def test_estimate_state_stress_regime_from_high_vol():
    profile = VolumeProfile(bin_share_mean={"10:00": 0.05}, bin_share_p10={"10:00": 0.02},
                            bin_share_p20={"10:00": 0.03})
    spread = SpreadCurve(bin_half_spread_bps={"10:00": 1.0})
    bounds = VolRegimeBounds(q1_bps=100.0, q2_bps=200.0)
    rng = np.random.default_rng(2)
    stress_returns = pd.Series(rng.normal(0, 0.002, 200))   # ~400 bps daily-equiv -> stress
    st = estimate_state(
        pd.Timestamp("2026-01-05 10:05:00"), "TQQQ",
        recent_1min_returns=stress_returns, trailing_daily_volume=1_000_000.0,
        profile=profile, spread_curve=spread, vol_bounds=bounds,
    )
    assert st.regime == "stress"


# --------------------------------------------------------------------------- live-data sanity
try:
    from quantcore import config
    _DATA_DIR = config.data_dir()
except Exception:  # pragma: no cover - env without quantcore
    _DATA_DIR = None


def _db(sym: str) -> Path:
    return _DATA_DIR / f"DB_{sym}_historical_data.db" if _DATA_DIR else Path("/nonexistent")


pytestmark = pytest.mark.skipif(
    _DATA_DIR is None or not _db("TQQQ").exists(),
    reason="historical DBs not available (market-state live-data sanity check)",
)


def _load_15min(sym: str) -> pd.DataFrame:
    with sqlite3.connect(_db(sym)) as c:
        d = pd.read_sql("SELECT et_datetime, high, low, volume FROM candles_15min ORDER BY ts", c)
    d["dt"] = pd.to_datetime(d["et_datetime"])
    return d.set_index("dt")


def test_open_bin_beats_midday_bin_in_volume_share_and_spread():
    """Sanity check on real data: the open (09:30) bin carries more volume share AND a wider
    spread than the midday (12:30) lull — the textbook U-shape, for all three tickers."""
    from slippage.spread import corwin_schultz_intraday, half_spread_bps

    for sym in ("TQQQ", "SQQQ", "QQQ"):
        d = _load_15min(sym)
        d = d[(d.index.time >= pd.Timestamp("09:30").time()) &
              (d.index.time <= pd.Timestamp("15:45").time())].copy()
        d["date"] = d.index.normalize()
        d["bin"] = [session_bin_label(ts) for ts in d.index]

        daily_total = d.groupby("date")["volume"].transform("sum")
        d["share"] = d["volume"] / daily_total
        prof = d.groupby("bin")["share"].mean()
        assert prof.loc["09:30"] > prof.loc["12:30"], f"{sym}: open volume share not > midday"

        s = corwin_schultz_intraday(d[["high", "low"]], clamp_negative=False)
        hs = half_spread_bps(s)
        tmp = pd.DataFrame({"bin": d["bin"].reindex(hs.index), "hs": hs}).dropna()
        curve = tmp.groupby("bin")["hs"].mean()
        # CS is indexed by the second bar of each pair, so 09:45 is the earliest bin present.
        assert curve.loc["09:45"] > curve.loc["12:30"], f"{sym}: open spread not > midday"
