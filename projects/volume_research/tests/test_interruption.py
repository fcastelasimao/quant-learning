"""Tests for slippage/interruption.py — interruption_hazard() and interruption_cost()."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slippage.interruption import interruption_hazard, interruption_cost  # noqa: E402
from slippage.state import MarketState  # noqa: E402


def _state(symbol="TQQQ", regime="normal"):
    return MarketState(ts=pd.Timestamp("2026-01-05 10:00:00"), symbol=symbol, bin_label="10:00",
                       expected_interval_volume=1e6, thin_volume_p10=4e5, thin_volume_p20=6e5,
                       sigma_now_bps=300.0, regime=regime, spread_bps=0.74)


# --------------------------------------------------------------------------- hazard
def test_hazard_zero_at_zero():
    assert interruption_hazard(0.0, _state()) == pytest.approx(0.0)


def test_hazard_monotonically_increasing():
    hs = [1, 15, 60, 240, 1000, 5000]
    vals = [interruption_hazard(h, _state()) for h in hs]
    assert vals == sorted(vals)


def test_hazard_bounded_in_unit_interval():
    for h in (0, 60, 780, 10_000):
        v = interruption_hazard(h, _state())
        assert 0.0 <= v <= 1.0


def test_hazard_matches_p25_context():
    # findings_02: p25 hold ~2.7-2.8h (~165min) for both tickers -> hazard near 0.25 there.
    v = interruption_hazard(165, _state())
    assert 0.15 < v < 0.35


def test_hazard_regime_conditioning_changes_value():
    calm = interruption_hazard(240, _state(regime="calm"))
    stress = interruption_hazard(240, _state(regime="stress"))
    assert calm != pytest.approx(stress)


def test_hazard_use_regime_false_uses_pooled_curve():
    pooled = interruption_hazard(240, _state(regime="calm"), use_regime=False)
    direct_pooled = interruption_hazard(240, _state(regime="unknown_regime"))
    assert pooled == pytest.approx(direct_pooled)


def test_hazard_unknown_symbol_falls_back_to_tqqq():
    a = interruption_hazard(100.0, _state(symbol="SPXL"))
    b = interruption_hazard(100.0, _state(symbol="TQQQ"))
    assert a == pytest.approx(b)


# --------------------------------------------------------------------------- cost
def test_cost_rejects_bad_mode():
    with pytest.raises(ValueError):
        interruption_cost(60, 0.5, "resume", "TQQQ")


def test_cancel_full_fill_equals_g_of_h():
    from slippage.alpha_decay import alpha_forfeit_frac
    c = interruption_cost(60, 1.0, "cancel", "TQQQ")
    assert c == pytest.approx(alpha_forfeit_frac(60, "TQQQ"))


def test_cancel_zero_fill_forfeits_everything():
    assert interruption_cost(60, 0.0, "cancel", "TQQQ") == pytest.approx(1.0)


def test_cancel_costs_more_than_complete_now_for_partial_fill():
    cancel = interruption_cost(60, 0.5, "cancel", "TQQQ")
    complete = interruption_cost(60, 0.5, "complete_now", "TQQQ")
    assert cancel > complete


def test_complete_now_is_phi_invariant():
    a = interruption_cost(120, 0.1, "complete_now", "TQQQ")
    b = interruption_cost(120, 0.9, "complete_now", "TQQQ")
    assert a == pytest.approx(b)


def test_phi_is_clamped_to_unit_interval():
    assert interruption_cost(60, 1.5, "cancel", "TQQQ") == pytest.approx(
        interruption_cost(60, 1.0, "cancel", "TQQQ"))
    assert interruption_cost(60, -0.5, "cancel", "TQQQ") == pytest.approx(
        interruption_cost(60, 0.0, "cancel", "TQQQ"))
