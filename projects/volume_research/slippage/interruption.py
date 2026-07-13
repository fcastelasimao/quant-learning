"""interruption_hazard / interruption_cost — the mid-fill hazard the scheduler must plan for.

Two hazards a working order faces before it finishes: the strategy's trailing stop fires, or the
signal flips, while the order is still being worked. Measured from the TRADES CSVs' own
`hold_days` (>99.9% of exits are TRAIL_STOP in this dataset — see
`research/scheduler/02_interruption_risk/findings_02_interruption_risk.md` for the caveat that
this is an *observed exit-timing* hazard, not separately decomposed by cause).

    interruption_hazard(h_min, state)          -> P(interrupted within h_min), baked empirical curve
    interruption_cost(h_min, phi, mode, symbol) -> fraction of the order's edge forfeited if
                                                    interrupted at filled-fraction phi

Design note for E04 (owner requirement): these two functions describe the *historical average*
hazard/cost — a Monte-Carlo or expected-value consumer. E04's replay harness is built to accept
an **event stream** (actual timestamps of stop-fires / signal-flips) instead, so a caller can
later feed real backtest or live events rather than these averages; this module doesn't need to
change for that — it's simply bypassed in favor of the real event timestamps.

Pure functions, no IO.
"""

from __future__ import annotations

import numpy as np

from .alpha_decay import alpha_forfeit_frac
from .state import MarketState

# Empirical P(exit within h_min), baked from research/scheduler/02_interruption_risk. A leading
# (0, 0.0) point anchors interpolation for h below the first measured grid point; values beyond
# the last grid point (3120 min, ~1 week) hold constant (np.interp's default extrapolation).
_HAZARD_GRID_MIN = [0, 15, 30, 60, 120, 240, 390, 780, 1560, 3120]
_HAZARD_OVERALL = {
    "TQQQ": [0.0, 0.009, 0.036, 0.102, 0.201, 0.327, 0.385, 0.385, 0.761, 0.831],
    "SQQQ": [0.0, 0.008, 0.028, 0.097, 0.202, 0.368, 0.417, 0.417, 0.817, 0.869],
}
_HAZARD_BY_REGIME = {
    "TQQQ": {
        "calm":   [0.0, 0.009, 0.035, 0.105, 0.210, 0.341, 0.400, 0.400, 0.767, 0.824],
        "normal": [0.0, 0.011, 0.038, 0.099, 0.187, 0.315, 0.382, 0.382, 0.768, 0.849],
        "stress": [0.0, 0.007, 0.036, 0.102, 0.209, 0.323, 0.361, 0.361, 0.731, 0.813],
    },
    "SQQQ": {
        "calm":   [0.0, 0.010, 0.034, 0.106, 0.223, 0.412, 0.450, 0.450, 0.843, 0.885],
        "normal": [0.0, 0.013, 0.035, 0.109, 0.211, 0.358, 0.424, 0.424, 0.827, 0.869],
        "stress": [0.0, 0.002, 0.016, 0.075, 0.169, 0.331, 0.375, 0.375, 0.781, 0.854],
    },
}


def interruption_hazard(h_min: float, state: MarketState, *, use_regime: bool = True) -> float:
    """P(the position is interrupted — trailing stop / signal flip — within h_min of entry).

    Uses the regime-conditioned curve when `state.regime` is one of calm/normal/stress and
    `use_regime=True` (default); otherwise falls back to the pooled overall curve. Regime makes
    little difference in practice (findings_02: <5pp spread at any horizon) — `use_regime=False`
    is available for a simpler, state-independent lookup.
    """
    symbol = state.symbol if state.symbol in _HAZARD_OVERALL else "TQQQ"
    if use_regime and state.regime in _HAZARD_BY_REGIME[symbol]:
        grid = _HAZARD_BY_REGIME[symbol][state.regime]
    else:
        grid = _HAZARD_OVERALL[symbol]
    return float(np.interp(max(h_min, 0.0), _HAZARD_GRID_MIN, grid))


def interruption_cost(h_min: float, phi: float, mode: str, symbol: str) -> float:
    """Fraction of the FULL order's potential edge forfeited if interrupted at time `h_min`
    having filled fraction `phi` (0..1) of it.

    mode="cancel": the unfilled residue (1-phi) is cancelled outright — it forfeits its edge
        entirely (weight 1.0), while the filled part phi only suffers the g(h) alpha-forfeiture
        already priced by `alpha_decay.alpha_forfeit_frac`:
            cost = phi * g(h) + (1 - phi) * 1.0
    mode="complete_now": the residue is rushed to completion immediately at the interruption
        instant — no ADDITIONAL delay-forfeiture beyond g(h) (the whole order ends up uniformly
        at the g(h) forfeiture level):
            cost = g(h)
        This does NOT include the extra impact cost of rushing an unsliced residual — that's
        `expected_slippage_bps`'s job elsewhere in the pipeline, not modeled here.
    """
    if mode not in ("cancel", "complete_now"):
        raise ValueError(f"mode must be 'cancel' or 'complete_now', got {mode!r}")
    phi = float(np.clip(phi, 0.0, 1.0))
    g = alpha_forfeit_frac(h_min, symbol)
    if mode == "cancel":
        return phi * g + (1.0 - phi) * 1.0
    return g
