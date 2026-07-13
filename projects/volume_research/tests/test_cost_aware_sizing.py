"""Tests for Stage-7 cost-aware sizing (math self-consistency).

Verify the sizing mechanics: f* shrinks with AUM, the optimal trade size saturates, higher
λ sizes down, and cost-aware sizing rescues Sharpe vs all-in at large AUM.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "build_07", ROOT / "research/07_cost_aware_sizing/build_07_cost_aware_sizing.py")
b07 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b07)

P = b07.b06.params_for("TQQQ")
MU, SIG = 0.0107, 0.0278     # TQQQ μ_edge, σ_trade (measured)


def test_fraction_shrinks_with_aum():
    f_small, *_ = b07.optimal_fraction(1e5, MU, SIG, P, lam=0.0)
    f_big, *_ = b07.optimal_fraction(1e8, MU, SIG, P, lam=0.0)
    assert f_small > f_big                      # bigger AUM → deploy a smaller fraction


def test_optimal_trade_size_saturates():
    # Beyond the capacity point Q*=f*·AUM is ~constant: a 10× AUM jump moves Q* little.
    f1, *_ = b07.optimal_fraction(1e8, MU, SIG, P, lam=0.0)
    f2, *_ = b07.optimal_fraction(1e9, MU, SIG, P, lam=0.0)
    q1, q2 = f1 * 1e8, f2 * 1e9
    assert 0.3 < q1 / q2 < 3.0                   # within a small band, not 10×


def test_higher_lambda_sizes_down():
    f0, *_ = b07.optimal_fraction(1e6, MU, SIG, P, lam=0.0)
    f20, *_ = b07.optimal_fraction(1e6, MU, SIG, P, lam=20.0)
    assert f20 < f0                              # more risk-averse → smaller position


def test_cost_aware_rescues_sharpe_vs_all_in():
    # All-in at $30M (Stage-6 style) vs cost-aware sizing at the same AUM.
    aum = 3e7
    drag_allin = b07.b06.roundtrip_no_lambda(0.95 * aum, P, Y=0.5)
    sh_allin = b07.b06.net_sharpe(7e-4, 1e-4, 3000, 1800, drag_allin, 0.0)
    f, cost, tvar = b07.optimal_fraction(aum, MU, SIG, P, lam=0.0)
    sh_sized = b07.b06.net_sharpe(7e-4, 1e-4, 3000, 1800, cost, tvar)
    assert sh_sized > sh_allin                   # not over-trading keeps the edge
