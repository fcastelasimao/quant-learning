"""
test_backtest_golden.py
=======================
GOLDEN REGRESSION GUARD for engine/backtest.run_backtest.

Purpose (D.15/D.16 gate): the tax-model and drift-trigger work adds new
parameters to the backtest engine. Those parameters MUST default to the
pre-existing behavior so production numbers reproduce byte-identical. This test
pins the engine's full output on a deterministic synthetic price series with
the production 6-asset allocation, and fails if the *default* code path ever
changes its numbers.

How it works
------------
* ``build_golden_prices`` generates a fixed (seeded) daily price history for the
  six production tickers — no network, no central data store, fully portable.
* ``run_golden`` calls ``run_backtest`` with the production weights and the
  standard cost assumptions, passing NO new parameters (so the engine uses its
  defaults).
* The expected output lives in ``tests/data/backtest_golden.csv``, captured from
  the engine BEFORE the D.15 refactor. The test asserts the live engine
  reproduces it exactly via ``pandas.testing.assert_frame_equal``.

Regenerating the fixture
------------------------
Only do this if you have *intentionally* changed default backtest behavior and
understand why the numbers move::

    python -m tests.test_backtest_golden --regen

(or run this file directly with ``--regen``). Review the diff before committing.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.backtest import run_backtest
from engine.stats import compute_cagr, compute_calmar, compute_max_drawdown

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "data", "backtest_golden.csv")

# Production 6-asset weights (6asset_tip_gsg_rpavg) — see CLAUDE.md.
PROD_WEIGHTS = {
    "SPY": 0.134,
    "QQQ": 0.103,
    "TLT": 0.175,
    "TIP": 0.348,
    "GLD": 0.142,
    "GSG": 0.098,
}

# Pinned headline metrics on the synthetic golden series (independent of the
# byte-for-byte CSV check; a second, human-readable guard rail).
GOLDEN_AW_FINAL = 21029.91


def build_golden_prices() -> pd.DataFrame:
    """Deterministic daily price history for the 6 production tickers.

    Seeded geometric random walk — identical on every machine and run. The
    drift/vol per ticker are arbitrary but fixed; this is a regression anchor,
    not a market model.
    """
    rng = np.random.default_rng(20260530)
    dates = pd.date_range("2006-01-03", "2024-12-31", freq="B")
    drift = {"SPY": 0.00030, "QQQ": 0.00040, "TLT": 0.00010,
             "TIP": 0.00008, "GLD": 0.00025, "GSG": 0.00005}
    vol = {"SPY": 0.011, "QQQ": 0.014, "TLT": 0.009,
           "TIP": 0.004, "GLD": 0.010, "GSG": 0.013}
    cols = {}
    for ticker in PROD_WEIGHTS:
        steps = rng.normal(drift[ticker], vol[ticker], len(dates))
        cols[ticker] = 100.0 * np.exp(np.cumsum(steps))
    return pd.DataFrame(cols, index=dates)


def run_golden() -> pd.DataFrame:
    """Run the engine on the golden inputs using DEFAULT behavior only.

    No D.15/D.16 parameters are passed — that is the whole point: the default
    path must reproduce the committed fixture.
    """
    prices = build_golden_prices()
    return run_backtest(
        prices=prices,
        benchmark_prices=prices["SPY"],
        allocation=PROD_WEIGHTS,
        portfolio_value=10_000.0,
        tlt_prices=prices["TLT"],
        transaction_cost_pct=0.001,
        tax_drag_pct=0.0,
    )


def _read_golden() -> pd.DataFrame:
    df = pd.read_csv(GOLDEN_PATH, index_col=0, parse_dates=True)
    df.index.name = "Date"
    return df


def regenerate() -> None:
    """Overwrite the committed fixture from the current engine output."""
    os.makedirs(os.path.dirname(GOLDEN_PATH), exist_ok=True)
    run_golden().to_csv(GOLDEN_PATH)
    print(f"Wrote golden fixture: {GOLDEN_PATH}")


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_golden_fixture_exists():
    assert os.path.exists(GOLDEN_PATH), (
        "Golden fixture missing — generate it with "
        "`python -m tests.test_backtest_golden --regen` before refactoring."
    )


def test_default_backtest_matches_golden_fixture():
    """The default code path must reproduce the pre-refactor output exactly."""
    expected = _read_golden()
    actual = run_golden()
    # Align dtypes/columns the way a CSV round-trip would, then compare exactly.
    pd.testing.assert_frame_equal(
        actual,
        expected,
        check_exact=False,
        rtol=0.0,
        atol=1e-9,
        check_like=False,
    )


def test_golden_headline_metrics_stable():
    out = run_golden()
    aw = out["All Weather Value"]
    assert round(float(aw.iloc[-1]), 2) == GOLDEN_AW_FINAL
    years = (out.index[-1] - out.index[0]).days / 365.25
    cagr = compute_cagr(aw, years)
    mdd = compute_max_drawdown(aw)
    calmar = compute_calmar(round(cagr, 2), round(mdd, 2))
    # sanity bands — these only move if default behavior changes
    assert 0.0 < cagr < 20.0
    assert mdd < 0.0
    assert calmar > 0.0


def test_expected_columns_present():
    out = run_golden()
    for col in [
        "All Weather Value", "Buy & Hold All Weather", "S&P 500 Value",
        "60/40 Value", "All Weather Value Monthly Ret (%)",
    ]:
        assert col in out.columns


if __name__ == "__main__":
    if "--regen" in sys.argv:
        regenerate()
    else:
        print("Pass --regen to (re)write the golden fixture.")
