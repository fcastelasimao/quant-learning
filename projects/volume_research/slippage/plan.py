"""plan_execution — turn a decided order into a *how fast / how to slice* recommendation.

Input: the trade to do (a notional, one-way). Output: the participation rate (% of volume), the
fill horizon, a slice plan, and the expected cost — i.e. what urgency to hand a broker POV/IS algo.

This is the pre-trade *planner*, not a live order router: brokers already slice and route child
orders; what they need from us is the **urgency** and a cost forecast. The speed is the
Almgren–Chriss optimum (trade faster → more impact, slower → more timing risk), subject to the
strategy's decision cadence (`horizon_cap_min`), since an order worked past the next decision is no
longer the position we decided on.

At retail size the answer is trivially "cross now" — this earns its keep at $M+, where impact bites.
Caveat: the √-law impact term is calibrated for sizeable *metaorders*; for small fast clips it
over-extrapolates (the live fills show retail cost is spread + timing, not the model's nominal
impact). Treat the expected-cost number as meaningful from ~$1M up.
"""

from __future__ import annotations

from dataclasses import dataclass

from .cost import MarketParams, fill_minutes, optimal_participation, expected_slippage_bps


@dataclass
class ExecutionPlan:
    """A pre-trade execution recommendation for one order (one-way).

    `expected_slippage_bps` is the expected (mean) slippage: spread + impact (+ signed entry
    drag). Timing risk is a separate variance channel and is deliberately NOT included in this
    number.
    """
    notional_usd: float
    participation: float          # POV: fraction of interval volume to trade
    horizon_min: float            # time to complete the fill
    n_slices: int                 # suggested number of child orders
    slice_notional_usd: float
    expected_slippage_bps: float  # one-way spread + impact, central Y (cost of taking liquidity)
    cost_band_bps: tuple[float, float]   # (optimistic Y=0.3, conservative Y=1.0)
    timing_risk_bps: float        # one-way 1σ timing
    feasible: bool                # False if it can't fill within the cap even at 100% POV
    entry_drag_bps: float         # passive-chase adverse drift supplied by caller (0 = not modelled)
    cross_entry: bool | None      # True = cross/work it, False = passive limit; None if no drag given
    note: str

    def __str__(self) -> str:
        return self.note


def plan_execution(notional_usd: float, params: MarketParams, *, lam: float = 1.0,
                   Y: float = 0.5, beta: float = 0.5, horizon_cap_min: float = 15.0,
                   slice_minutes: float = 1.0, entry_drag_bps: float = 0.0) -> ExecutionPlan:
    """Recommend how fast to trade `notional_usd` and how to slice it.

    Parameters
    ----------
    notional_usd : the order size to execute (one-way).
    params : MarketParams for the instrument (from `calibrate`).
    lam : risk aversion — higher trades faster (less timing risk, more impact). 0 = patient.
    horizon_cap_min : the order must finish within this window (the decision cadence).
    slice_minutes : child-order spacing used to suggest a slice count.
    entry_drag_bps : the measured adverse drift of a *passive* limit-chase entry (findings_08:
        ~14 bps for momentum). When >0 the plan recommends marketable-cross vs passive-limit via
        `cross_entry`: since impact is ~common to both styles it cancels, leaving the spread you'd
        cross vs the drift you'd chase — so this compares `entry_drag_bps` to `half_spread`, NOT to
        `expected_slippage_bps` (whose √-law impact over-extrapolates at retail). Default 0 = not modelled.

    Returns
    -------
    ExecutionPlan with the participation rate, horizon, slice plan, and expected cost (band).
    """
    pov = optimal_participation(notional_usd, params, Y=Y, beta=beta, lam=lam)
    horizon = fill_minutes(notional_usd, pov, params)

    feasible = True
    if horizon > horizon_cap_min:
        # too slow for the cadence — speed up to fill within the cap
        pov = (notional_usd / params.adv_usd) * (params.minutes_per_day / horizon_cap_min)
        if pov >= 1.0:                      # can't fill even taking all the volume
            pov, feasible = 1.0, False
        horizon = fill_minutes(notional_usd, pov, params)

    c = expected_slippage_bps(notional_usd, pov, params, Y=Y, beta=beta)
    lo = expected_slippage_bps(notional_usd, pov, params, Y=0.3, beta=beta)["expected_slippage_bps"]
    hi = expected_slippage_bps(notional_usd, pov, params, Y=1.0, beta=beta)["expected_slippage_bps"]
    n_slices = max(1, round(horizon / slice_minutes))

    note = (f"Trade ~{pov:.1%} of volume over ~{horizon:.0f} min "
            f"in {n_slices} slice(s) of ~${notional_usd / n_slices / 1e6:.2f}M. "
            f"Expected cost ~{c['expected_slippage_bps']:.1f} bps "
            f"(band {lo:.1f}–{hi:.1f}); timing ±{c['timing_risk_bps']:.0f} bps (1σ).")

    # Entry style: marketable cross vs a passive limit that chases and pays the measured adverse
    # drift. Impact is ~common to both styles (your size moves price over the fill either way), so
    # it cancels — the differential is the spread you cross vs the drift you chase. Cross when the
    # spread is the cheaper of the two. (This is why it's compared to half_spread, not
    # expected_slippage_bps: the latter's √-law impact over-extrapolates at retail, where crossing
    # is empirically clean.)
    cross_entry: bool | None = None
    if entry_drag_bps > 0:
        cross_cost = params.half_spread_bps
        cross_entry = cross_cost <= entry_drag_bps
        if cross_entry:
            note += (f" Entry: CROSS — the spread (~{cross_cost:.1f} bps) is cheaper than chasing "
                     f"with a passive limit (~{entry_drag_bps:.0f} bps adverse drift).")
        else:
            note += (f" Entry: rest a LIMIT — chasing drift (~{entry_drag_bps:.0f} bps) is cheaper "
                     f"than crossing the spread (~{cross_cost:.1f} bps).")

    if not feasible:
        note = ("⚠ Order too large to fill within the cadence even at 100% of volume — "
                "split across multiple decision windows or reduce size. " + note)

    return ExecutionPlan(
        notional_usd=float(notional_usd), participation=float(pov), horizon_min=float(horizon),
        n_slices=int(n_slices), slice_notional_usd=float(notional_usd / n_slices),
        expected_slippage_bps=float(c["expected_slippage_bps"]), cost_band_bps=(float(lo), float(hi)),
        timing_risk_bps=float(c["timing_risk_bps"]), feasible=feasible,
        entry_drag_bps=float(entry_drag_bps), cross_entry=cross_entry, note=note)
