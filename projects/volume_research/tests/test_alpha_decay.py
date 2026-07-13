"""Tests for slippage/alpha_decay.py — alpha_forfeit_frac()."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slippage.alpha_decay import alpha_forfeit_frac  # noqa: E402


def test_zero_at_zero_delay():
    assert alpha_forfeit_frac(0.0, "TQQQ") == pytest.approx(0.0)


def test_negative_delay_clamped_to_zero():
    assert alpha_forfeit_frac(-5.0, "TQQQ") == pytest.approx(0.0)


def test_monotonically_increasing_in_h():
    hs = [1, 15, 60, 240, 780, 5000]
    vals = [alpha_forfeit_frac(h, "TQQQ") for h in hs]
    assert vals == sorted(vals)


def test_bounded_in_unit_interval():
    for h in (0, 15, 390, 10_000):
        v = alpha_forfeit_frac(h, "TQQQ")
        assert 0.0 <= v <= 1.0


def test_approaches_one_for_large_h():
    assert alpha_forfeit_frac(10_000, "TQQQ") == pytest.approx(1.0, abs=1e-3)


def test_matches_findings_reference_points():
    # findings_01_alpha_decay.md, TQQQ, unweighted stretched-exp fit, tau=256.9 min, k=2.480.
    assert alpha_forfeit_frac(15, "TQQQ") == pytest.approx(0.001, abs=0.002)
    assert alpha_forfeit_frac(120, "TQQQ") == pytest.approx(0.140, abs=0.005)
    assert alpha_forfeit_frac(390, "TQQQ") == pytest.approx(0.940, abs=0.005)


def test_unknown_symbol_falls_back_to_tqqq():
    assert alpha_forfeit_frac(100.0, "SPXL") == pytest.approx(alpha_forfeit_frac(100.0, "TQQQ"))


def test_pnl_weighted_variant_differs_from_unweighted():
    a = alpha_forfeit_frac(200.0, "SQQQ", pnl_weighted=False)
    b = alpha_forfeit_frac(200.0, "SQQQ", pnl_weighted=True)
    assert a != pytest.approx(b)


def test_symbols_have_distinct_curves():
    assert alpha_forfeit_frac(100.0, "TQQQ") != pytest.approx(alpha_forfeit_frac(100.0, "SQQQ"))
