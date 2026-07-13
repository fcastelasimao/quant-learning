"""Tests for the square-root impact law.

There is NO data ground truth for the constant Y (we never traded), so these tests can
only verify the *math*: self-consistency of impact↔capacity, the scaling exponent, and
reproduction of the SLIPPAGE_PLAN worked example.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slippage.impact import (  # noqa: E402
    impact_bps, capacity, participation_for_impact, almgren_permanent, almgren_temporary,
)


def test_capacity_inverts_impact():
    # Q at the capacity for a budget should produce exactly that budget of impact.
    V, sigma, Y, beta = 80e6, 350.0, 0.5, 0.5
    for budget in (5.0, 10.0, 25.0):
        Q = capacity(budget, V, sigma, Y=Y, beta=beta)
        assert impact_bps(Q, V, sigma, Y=Y, beta=beta) == pytest.approx(budget, rel=1e-9)


def test_sqrt_scaling():
    # 4x the size -> 2x the impact under beta=0.5.
    V, sigma = 80e6, 350.0
    i1 = impact_bps(1e6, V, sigma)
    i4 = impact_bps(4e6, V, sigma)
    assert i4 == pytest.approx(2.0 * i1, rel=1e-9)


def test_exponent_changes_curvature():
    V, sigma = 80e6, 350.0
    # at a small participation rate, a higher exponent (0.6) gives LESS impact than 0.5.
    half = impact_bps(1e6, V, sigma, beta=0.5)
    three_fifths = impact_bps(1e6, V, sigma, beta=0.6)
    assert three_fifths < half


def test_plan_worked_example():
    # SLIPPAGE_PLAN: Y=0.5, daily sigma=3.5% (=350 bps), V=ADV=80M, budget=10 bps
    # -> Q_max ~ 260k shares.
    Q = capacity(10.0, 80e6, 350.0, Y=0.5, beta=0.5)
    assert Q == pytest.approx(261_224, rel=0.02)


def test_participation_is_volume_independent():
    p = participation_for_impact(10.0, 350.0, Y=0.5, beta=0.5)
    # 10 bps budget -> Q/V = (10/175)^2 ~ 0.33%
    assert p == pytest.approx((10.0 / (0.5 * 350.0)) ** 2, rel=1e-9)
    # and capacity == participation * V
    assert capacity(10.0, 80e6, 350.0) == pytest.approx(p * 80e6, rel=1e-9)


def test_y_band_widens_capacity():
    # smaller Y (less impact) -> larger capacity for the same budget.
    big = capacity(10.0, 80e6, 350.0, Y=0.3)
    small = capacity(10.0, 80e6, 350.0, Y=1.0)
    assert big > small


# --------------------------------------------------------------------------- C01 golden test
# Almgren, Thum, Hauptmann & Li (2005) Table 3: realized cost J = I/2 + K (temporary), for a
# purchase of X/V = 10% of ADV, at three execution durations T (days). Reproduces the paper's
# published (rounded-to-integer-bp) IBM/DRI worked examples to within 1 bp.
_IBM = dict(V=6.561e6, theta=1728e6, sigma_bps=157.0)   # sigma = 1.57%/day
_DRI = dict(V=1.929e6, theta=168e6, sigma_bps=226.0)    # sigma = 2.26%/day


def _realized_J(stock, T_days):
    X = 0.1 * stock["V"]
    I = almgren_permanent(X, stock["V"], stock["theta"], stock["sigma_bps"])
    participation = (X / stock["V"]) / T_days   # X/(V*T)
    K = almgren_temporary(participation, stock["sigma_bps"])
    return I, I / 2.0 + K


@pytest.mark.parametrize("stock,T,expected_I,expected_J", [
    (_IBM, 0.1, 20, 32), (_IBM, 0.2, 20, 25), (_IBM, 0.5, 20, 18),
    (_DRI, 0.1, 22, 43), (_DRI, 0.2, 22, 32), (_DRI, 0.5, 22, 23),
])
def test_almgren_reproduces_paper_table_3(stock, T, expected_I, expected_J):
    I, J = _realized_J(stock, T)
    assert I == pytest.approx(expected_I, abs=1.0)
    assert J == pytest.approx(expected_J, abs=1.0)


def test_almgren_temporary_is_signed():
    assert almgren_temporary(0.5, 200.0) > 0
    assert almgren_temporary(-0.5, 200.0) < 0
    assert almgren_temporary(-0.5, 200.0) == pytest.approx(-almgren_temporary(0.5, 200.0))


def test_almgren_temporary_rejects_sqrt_law_shape():
    # beta=0.6 (Almgren) gives LESS impact than beta=0.5 (sqrt-law) at small participation,
    # and the gap direction flips above p=1 - the two laws are genuinely different shapes.
    sqrt_law = impact_bps(1.0, 10.0, 200.0, Y=1.0, beta=0.5)   # Q/V = 0.1
    almgren = almgren_temporary(0.1, 200.0, eta=1.0, beta=0.6)   # same eta=Y=1 for a clean compare
    assert almgren < sqrt_law
