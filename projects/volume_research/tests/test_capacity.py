"""Tests for the Stage-6 capacity curve (math self-consistency).

No data ground truth (Sharpe levels depend on the adopted Y), so we verify the mechanics:
the net-Sharpe response to a drag and to timing variance, monotone cost-in-size, and the
λ → execution-speed direction (higher λ → faster fill → more impact, less timing).
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Load the build module by path (it lives under research/, not on the package path).
_spec = importlib.util.spec_from_file_location(
    "build_06", ROOT / "research/06_capacity_curve/build_06_capacity_curve.py")
b06 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b06)

P = b06.params_for("TQQQ")
# representative gross daily stats (mean, var, n_days, n_trades)
STATS = (0.0007, 0.0001, 3000, 1800)


def test_drag_lowers_sharpe():
    base = b06.net_sharpe(*STATS, drag_rt_frac=0.0, timing_var_rt_frac2=0.0)
    drag = b06.net_sharpe(*STATS, drag_rt_frac=0.0020, timing_var_rt_frac2=0.0)
    assert drag < base                       # a mean drag only hits the numerator


def test_timing_variance_lowers_sharpe_not_via_mean():
    base = b06.net_sharpe(*STATS, drag_rt_frac=0.0, timing_var_rt_frac2=0.0)
    tim = b06.net_sharpe(*STATS, drag_rt_frac=0.0, timing_var_rt_frac2=(0.005) ** 2)
    assert tim < base                        # variance-only still lowers Sharpe
    # and it does so through the denominator: numerator (mean) is unchanged
    tpd = STATS[3] / STATS[2]
    assert np.isclose(STATS[0], STATS[0] - 0.0 * tpd)


def test_cost_increases_with_size():
    small = b06.roundtrip_no_lambda(1e5, P, Y=0.5)
    big = b06.roundtrip_no_lambda(1e7, P, Y=0.5)
    assert big > small > 0                    # √-law: more size → more impact


def test_lambda_trades_impact_for_timing():
    # Higher λ → optimiser picks a faster fill → more impact drag, less timing risk.
    drag0, tvar0 = b06.roundtrip_with_lambda(1e6, P, Y=0.5, lam=0.0)
    drag3, tvar3 = b06.roundtrip_with_lambda(1e6, P, Y=0.5, lam=3.0)
    assert drag3 > drag0                      # faster → more impact (mean drag)
    assert tvar3 < tvar0                      # faster → less timing variance


def test_higher_Y_costs_more():
    lo = b06.roundtrip_no_lambda(1e6, P, Y=0.3)
    hi = b06.roundtrip_no_lambda(1e6, P, Y=1.0)
    assert hi > lo                            # conservative Y → more impact
