"""predict_slippage — the client's "I set an order at X and filled at X + diff — predict diff."

Composes P01's MarketState (spread/impact-volume/vol regime) with P02's measured chase-cost
curve and S03's empirical (leptokurtic, not Gaussian) timing-risk quantile shape into one
pre-trade prediction:

    predict_slippage(notional_usd, side, order_type, state, price)
      -> {mean_bps, p50_bps, p90_bps, p95_bps,
          components: {spread_bps, impact_band_bps, drag_bps, timing_sigma_bps}}

`mean_bps` is the expected (mean) slippage: spread + impact (+ signed entry drag). Timing risk
is a separate variance channel, folded into the p50/p90/p95 quantiles around that mean rather
than added to it — consistent with the rest of the library (`expected_slippage_bps`,
`RoundTripCost.expected_slippage_bps`).

Pure function, no IO — `state` and `price` carry everything measured; this module only combines.
"""

from __future__ import annotations

import numpy as np

from .impact import impact_bps, almgren_temporary, ALMGREN_ETA, ALMGREN_ETA_SE
from .state import MarketState

ORDER_TYPES = ("cross", "limit_chase")
IMPACT_MODELS = ("sqrt", "almgren", "envelope")
Y_BAND = (0.3, 0.5, 1.0)          # the adopted impact band, same convention as impact.py/model.py
IMPACT_BETA = 0.5

# --------------------------------------------------------------------------- P02 chase-drag curve
# Style-B (passive limit, timeout T, then cross) mean cost, bps — measured on full history,
# findings_02_chase_simulation.md. "cross" drag is 0 (S08/P02: crossing is clean beyond spread).
# Linearly interpolated between these T points; symbols outside the table fall back to TQQQ's
# curve (documented — calibrated on TQQQ/SQQQ only).
_P02_CHASE_DRAG_BPS: dict[str, dict[int, float]] = {
    "TQQQ": {1: 6.67, 2: 6.75, 5: 7.50, 10: 8.05, 15: 8.66, 30: 8.32, 60: 8.44},
    "SQQQ": {1: 9.14, 2: 9.68, 5: 10.99, 10: 12.14, 15: 13.22, 30: 13.03, 60: 12.55},
}


def _chase_drag_bps(symbol: str, latency_min: float) -> float:
    curve = _P02_CHASE_DRAG_BPS.get(symbol, _P02_CHASE_DRAG_BPS["TQQQ"])
    ts = sorted(curve)
    if latency_min <= ts[0]:
        return curve[ts[0]]
    if latency_min >= ts[-1]:
        return curve[ts[-1]]
    return float(np.interp(latency_min, ts, [curve[t] for t in ts]))


# --------------------------------------------------------------------------- S03 timing quantiles
# Empirical |signed cost| / std ratios at the 15-min horizon (decision cadence), TQQQ, from S03's
# own delay_costs_bps/timing_risk_bps method (findings_03_delay_cost.md: "sharp central peak +
# fat tails, leptokurtic, not Gaussian"). Roughly stable across 1-60 min (checked); not further
# horizon-corrected here. SQQQ runs a few % higher at p95 (1.53 vs 1.44) — using TQQQ's slightly
# more conservative-toward-center values for both is a documented simplification, not a fit.
P50_TIMING_MULT = 0.0     # median of the mean-zero timing distribution
P90_TIMING_MULT = 0.98
P95_TIMING_MULT = 1.44


def _timing_sigma_bps(sigma_now_bps: float, latency_min: float, *, minutes_per_day: int = 390) -> float:
    """Timing risk (1sigma, bps) at `latency_min`, scaled from the daily-bps nowcast via the
    same sqrt(t) convention as MarketParams.sigma_1min_bps / cost.py."""
    sigma_1min = sigma_now_bps / np.sqrt(minutes_per_day)
    return sigma_1min * np.sqrt(latency_min)


# --------------------------------------------------------------------------- the predictor
def predict_slippage(notional_usd: float, side: str, order_type: str, state: MarketState,
                     price: float, *, latency_min: float = 15.0,
                     beta: float = IMPACT_BETA, impact_model: str = "sqrt") -> dict:
    """Predict the fill-price deviation for one order.

    Parameters
    ----------
    notional_usd : the order size, one-way.
    side : "buy" | "sell" — informational; magnitude is symmetric. The `drag` component is
        calibrated on BUYS only (P02/S08 — TQQQ/SQQQ are long-only books); sells are assumed
        clean per S08 (drag=0 either way this fires, since sells were never chased).
    order_type : "cross" (pay the spread, ~0 extra drag) or "limit_chase" (rest a passive limit,
        `latency_min` timeout, then cross — the measured P02 chase-cost curve).
    state : MarketState from `estimate_state()` — supplies spread, sigma nowcast, regime,
        interval-volume forecast.
    price : current price, needed only to convert `state.expected_interval_volume` (shares) into
        a dollar window-volume for the participation ratio.
    latency_min : execution horizon — for "cross" this only scales the impact-volume window; for
        "limit_chase" it's also the resting-limit timeout fed to the P02 curve.
    impact_model : "sqrt" (default — the library's standing default; flipping it is an owner/GATE
        decision, not made here), "almgren" (Almgren et al. 2005's fitted temporary term, banded
        by its own published standard error on eta), or "envelope" (the honest combined
        uncertainty: band = min/max across {sqrt at Y=0.3, sqrt at Y=1.0, Almgren point estimate}
        — the point/mean estimate stays the sqrt Y=0.5 central value; envelope only widens the
        reported band). See research/capacity/01_almgren_envelope.

    Returns
    -------
    dict with `mean_bps` (expected/mean slippage: spread + impact + drag — timing NOT included,
    it's a variance channel), `p50_bps`/`p90_bps`/`p95_bps` (mean_bps + the empirical, leptokurtic
    timing quantile — not a Gaussian z-score), and `components` (the itemized pieces, each tagged
    by its evidence tier in the docstring above).
    """
    if side not in ("buy", "sell"):
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
    if order_type not in ORDER_TYPES:
        raise ValueError(f"order_type must be one of {ORDER_TYPES}, got {order_type!r}")
    if impact_model not in IMPACT_MODELS:
        raise ValueError(f"impact_model must be one of {IMPACT_MODELS}, got {impact_model!r}")

    window_volume_usd = state.expected_interval_volume * price * (latency_min / 15.0)
    participation = min(notional_usd / window_volume_usd, 1.0) if window_volume_usd > 0 else 1.0

    def _sqrt(y):
        return (float(impact_bps(notional_usd, window_volume_usd, state.sigma_now_bps, Y=y, beta=beta))
                if window_volume_usd > 0 else 0.0)

    def _almgren(eta):
        return (float(almgren_temporary(participation, state.sigma_now_bps, eta=eta))
                if window_volume_usd > 0 else 0.0)

    if impact_model == "sqrt":
        impact_lo, impact_mid, impact_hi = _sqrt(Y_BAND[0]), _sqrt(Y_BAND[1]), _sqrt(Y_BAND[2])
    elif impact_model == "almgren":
        impact_lo = _almgren(ALMGREN_ETA - ALMGREN_ETA_SE)
        impact_mid = _almgren(ALMGREN_ETA)
        impact_hi = _almgren(ALMGREN_ETA + ALMGREN_ETA_SE)
    else:   # "envelope"
        candidates = (_sqrt(Y_BAND[0]), _sqrt(Y_BAND[2]), _almgren(ALMGREN_ETA))
        impact_mid = _sqrt(Y_BAND[1])   # the point/mean estimate is unchanged — envelope only widens the band
        impact_lo, impact_hi = min(candidates), max(candidates)

    if order_type == "cross":
        spread, drag = state.spread_bps, 0.0
    else:
        # The P02 curve is a blended empirical mean (fill-at-limit=0 mixed with
        # cross-if-unfilled=drift+spread) — it already contains the effective spread paid on
        # unfilled attempts, so `spread` is reported as 0 here to avoid double-counting it
        # against `drag`.
        spread, drag = 0.0, _chase_drag_bps(state.symbol, latency_min)

    timing_sigma = _timing_sigma_bps(state.sigma_now_bps, latency_min)
    mean_bps = spread + impact_mid + drag

    return {
        "mean_bps": mean_bps,
        "p50_bps": mean_bps + P50_TIMING_MULT * timing_sigma,
        "p90_bps": mean_bps + P90_TIMING_MULT * timing_sigma,
        "p95_bps": mean_bps + P95_TIMING_MULT * timing_sigma,
        "components": {
            "spread_bps": spread,
            "impact_band_bps": (impact_lo, impact_mid, impact_hi),
            "drag_bps": drag,
            "timing_sigma_bps": timing_sigma,
        },
        "participation": participation,
    }
