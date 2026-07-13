"""CostModel — the colleague-facing facade over the slippage library.

The three measured pieces (spread, impact, delay/timing) and the size↔time link all live in
`cost.py`. This wraps them in one object so a backtest can charge a trade without re-deriving
round-trip composition, the participation→fill-time mapping, or the expected-vs-risk split.

Typical use (replace a flat 20 bps round-trip):

    from slippage import calibrate, CostModel
    cal = calibrate(daily_ohlcv, intraday_15min)      # -> normal + stress MarketParams
    model = CostModel(cal.normal)                     # all components on
    rt = model.roundtrip(notional_usd=2_000_000)      # filled at the 15-min cadence
    net_return = gross_return - rt.expected_slippage_bps / 1e4    # impact+spread = mean drag
    # rt.timing_bps is a RISK (1σ, mean≈0) -> add to variance, do not subtract from return

`Y` is adopted from the literature (not fittable from OHLC), so impact is a **band**; use
`roundtrip_band()` to get low/central/high rather than pretending a single number.

Components are toggleable (`spread`/`impact`/`delay`) — the caller chooses what's on.
Pure: no IO. Calibration (which reads data) is separate, in `calibrate.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cost import MarketParams, optimal_participation, expected_slippage_bps

DEFAULT_HORIZON_MIN = 15   # the strategy's decision cadence; orders fill within this window
DEFAULT_Y_BAND = (0.3, 0.5, 1.0)


@dataclass
class RoundTripCost:
    """Cost of a full round trip (entry + exit) of `notional` USD.

    `spread_bps`/`impact_bps` are mean drags (both sides summed) and `entry_drag_bps` is the
    signed entry drift for signal-correlated execution (entry side only); `expected_slippage_bps`
    is their sum — subtract it from the trade's return. `timing_bps` is a 1σ **risk** (mean ≈ 0)
    — feed it to variance via `timing_var_frac2`, never subtract it from the return.
    """
    notional_usd: float
    participation: float
    fill_minutes: float
    spread_bps: float
    impact_bps: float
    timing_bps: float
    entry_drag_bps: float = 0.0

    @property
    def expected_slippage_bps(self) -> float:
        """Expected (mean) slippage: spread + impact (+ signed entry drag). Timing risk is a
        separate variance channel and is deliberately NOT included in this number."""
        return self.spread_bps + self.impact_bps + self.entry_drag_bps

    @property
    def timing_var_frac2(self) -> float:
        """Round-trip timing variance in fractional² units (for adding to return variance)."""
        return (self.timing_bps / 1e4) ** 2


@dataclass
class CostModel:
    """Size-aware execution-cost model for one instrument.

    Parameters
    ----------
    params : MarketParams
        Measured σ / ADV / half-spread (from `calibrate`, or built by hand).
    spread, impact, delay : bool
        Toggle each component. Default all on.
    entry_drag_bps : float
        Signed adverse entry drift (bps) for signal-correlated execution — the *mean* of the
        delay distribution, which is ≈0 for a symmetric strategy but positive for momentum
        (live TQQQ limit-chase entries: ~+14 bps, findings_08). Charged once per round trip
        (entry only; clean market exits add ≈0) as a mean drag on `expected_slippage_bps`, on top
        of the symmetric `timing_bps` variance — they are different moments of the same
        decision→fill move, not a double-count. Default 0 = the symmetric assumption. A
        strategy/execution-style input from live fills, NOT fittable from OHLC, so `calibrate`
        never sets it — the caller supplies it. Rides the `delay` toggle.
    """
    params: MarketParams
    spread: bool = True
    impact: bool = True
    delay: bool = True
    entry_drag_bps: float = 0.0

    def _compose(self, notional_usd, participation, Y, beta, impact_model="sqrt") -> RoundTripCost:
        c = expected_slippage_bps(notional_usd, participation, self.params, Y=Y, beta=beta,
                                  impact_model=impact_model)
        # Round trip = entry + exit. Spread/impact add (both cross/push); timing is two
        # independent draws, so its 1σ adds in quadrature -> √2 × per-side.
        spread = 2.0 * c["spread_bps"] if self.spread else 0.0
        impact = 2.0 * c["impact_bps"] if self.impact else 0.0
        timing = np.sqrt(2.0) * c["timing_risk_bps"] if self.delay else 0.0
        # Signed entry drift (momentum): charged once — entry only, exits fill clean — as a
        # mean drag distinct from the symmetric timing variance. Rides the delay toggle.
        entry_drag = self.entry_drag_bps if self.delay else 0.0
        return RoundTripCost(
            notional_usd=float(notional_usd),
            participation=float(participation),
            fill_minutes=float(c["fill_minutes"]),
            spread_bps=float(spread), impact_bps=float(impact), timing_bps=float(timing),
            entry_drag_bps=float(entry_drag),
        )

    def roundtrip(self, notional_usd, *, horizon_min: float = DEFAULT_HORIZON_MIN,
                  Y: float = 0.5, beta: float = 0.5, impact_model: str = "sqrt") -> RoundTripCost:
        """Round-trip cost when the order must fill within `horizon_min` (pins participation).

        This is the robust, λ-free view: filling in a fixed window forces a participation
        rate, and impact grows with √(participation). Default horizon = the 15-min cadence.

        `impact_model`: "sqrt" (default — the library's standing default; flipping it is an
        owner/GATE decision, not made here) or "almgren" (Almgren et al. 2005's fitted temporary
        term — see research/capacity/01_almgren_envelope).
        """
        part = (notional_usd / self.params.adv_usd) * (self.params.minutes_per_day / horizon_min)
        return self._compose(notional_usd, min(part, 1.0), Y, beta, impact_model=impact_model)

    def roundtrip_optimal(self, notional_usd, *, lam: float = 1.0,
                          Y: float = 0.5, beta: float = 0.5,
                          impact_model: str = "sqrt") -> RoundTripCost:
        """Round-trip cost at the Almgren–Chriss optimal speed for risk-aversion `lam`.

        `lam` trades impact (mean drag) against timing (variance); see `optimal_participation`.
        Use when execution speed is free to choose; `roundtrip` (fixed horizon) is the
        operative view for a strategy that must hold to its decision cadence.
        """
        part = optimal_participation(notional_usd, self.params, Y=Y, beta=beta, lam=lam,
                                     impact_model=impact_model)
        return self._compose(notional_usd, part, Y, beta, impact_model=impact_model)

    def roundtrip_band(self, notional_usd, *, horizon_min: float = DEFAULT_HORIZON_MIN,
                       Ys=DEFAULT_Y_BAND, beta: float = 0.5) -> dict[float, RoundTripCost]:
        """Expected cost across the adopted Y band — report a band, never a single line.

        Returns {Y: RoundTripCost} for each Y in `Ys` (default 0.3 / 0.5 / 1.0).
        """
        return {Y: self.roundtrip(notional_usd, horizon_min=horizon_min, Y=Y, beta=beta)
                for Y in Ys}
