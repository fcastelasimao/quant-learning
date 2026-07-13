"""Live calibration check — `calibrate()` on the real DBs reproduces the published chain.

Complements test_calibrate.py (synthetic recovery): this asserts calibrate() reproduces the
findings_04 σ / ADV and findings_01 half-spread from the *actual* TQQQ/SQQQ/QQQ DBs. Skipped
cleanly when the data isn't present (CI / other machines), so it never breaks the suite — it
only bites when run against the real data and a regression has crept in.
"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slippage import calibrate  # noqa: E402

try:
    from quantcore import config
    _DATA_DIR = config.data_dir()
except Exception:  # pragma: no cover - env without quantcore
    _DATA_DIR = None


def _db(sym: str) -> Path:
    return _DATA_DIR / f"DB_{sym}_historical_data.db" if _DATA_DIR else Path("/nonexistent")


pytestmark = pytest.mark.skipif(
    _DATA_DIR is None or not _db("TQQQ").exists(),
    reason="historical DBs not available (calibrate live-data check)",
)

# Published references: σ median / stress + $ADV from findings_04; half-spread from findings_01.
EXPECTED = {
    "TQQQ": dict(sigma_median=339, sigma_stress=510, adv=4.9e9, adv_thin=2.9e9, half_spread=0.74),
    "SQQQ": dict(sigma_median=341, sigma_stress=510, adv=2.8e9, adv_thin=1.0e9, half_spread=1.00),
    "QQQ":  dict(sigma_median=113, sigma_stress=172, adv=25.8e9, adv_thin=12.6e9, half_spread=0.72),
}


def _load(sym, table, cols):
    with sqlite3.connect(_db(sym)) as c:
        d = pd.read_sql(f"SELECT et_datetime, {cols} FROM {table} ORDER BY ts", c)
    d["dt"] = pd.to_datetime(d["et_datetime"])
    return d.set_index("dt")


@pytest.fixture(scope="module", params=list(EXPECTED))
def cal(request):
    sym = request.param
    daily = _load(sym, "candles_1d", "close, volume")
    intraday = _load(sym, "candles_15min", "high, low")
    return sym, calibrate(daily, intraday)


def test_adv_reproduces(cal):
    sym, c = cal
    assert c.adv_usd == pytest.approx(EXPECTED[sym]["adv"], rel=0.05)
    assert c.adv_thin_usd == pytest.approx(EXPECTED[sym]["adv_thin"], rel=0.05)


def test_sigma_stress_reproduces(cal):
    sym, c = cal
    assert c.sigma_stress_bps == pytest.approx(EXPECTED[sym]["sigma_stress"], rel=0.05)


def test_sigma_median_reproduces(cal):
    """The typical-day σ must match the published median (339/341/113)."""
    sym, c = cal
    assert c.sigma_daily_median_bps == pytest.approx(EXPECTED[sym]["sigma_median"], rel=0.03)


def test_sigma_mean_is_the_expected_cost_headline(cal):
    """The default σ is the mean; right-skew ⇒ mean > median by a modest, sane margin."""
    sym, c = cal
    assert c.sigma_daily_bps == c.normal.sigma_daily_bps       # normal uses the mean
    assert c.sigma_daily_bps > c.sigma_daily_median_bps        # right-skewed
    assert c.sigma_daily_bps == pytest.approx(c.sigma_daily_median_bps, rel=0.20)


def test_half_spread_reproduces(cal):
    sym, c = cal
    assert c.half_spread_bps == pytest.approx(EXPECTED[sym]["half_spread"], rel=0.10)
