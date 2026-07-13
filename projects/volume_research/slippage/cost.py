"""Stage 5 — the size-aware slippage cost function.

Composes the three measured pieces into one `cost(Q, urgency)`:

    spread   (Block 1) — a fixed floor, ~half-spread
  + impact   (Block 4) — √-law in participation; faster trading → more push
  + timing   (Block 3) — σ·√t risk; slower trading → more drift exposure

linked by the participation/time identity (volume sets the fill time):

    fill_time  t = (Q/ADV) / participation · day

Impact and timing pull in **opposite** directions through participation — that is the
Almgren–Chriss trade-off. Impact is an expected **drag**; timing is a **risk** (mean ≈ 0,
reported as 1σ). They are kept separate and must not be naively summed.

Consistency with Block 4: at participation = Q/ADV (i.e. working the order over a full
day, t = 1 day) the impact term reduces to the Block-4 √-law `Y·σ·√(Q/ADV)`.

`Y` is adopted from the literature (not fittable from OHLC), so every output carries the
same band as Block 4. Pure functions / a small dataclass; no IO.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .impact import almgren_temporary

IMPACT_MODELS = ("sqrt", "almgren")


@dataclass
class MarketParams:
    """Measured inputs for one instrument (from Blocks 1 and 4)."""
    sigma_daily_bps: float
    adv_usd: float
    half_spread_bps: float
    minutes_per_day: int = 390

    @property
    def sigma_1min_bps(self) -> float:
        return self.sigma_daily_bps / np.sqrt(self.minutes_per_day)


def fill_minutes(notional_usd, participation, p: MarketParams):
    """Volume sets the fill time: t = (Q/ADV)/participation · day."""
    return (np.asarray(notional_usd, float) / p.adv_usd) / participation * p.minutes_per_day


def expected_slippage_bps(notional_usd, participation, p: MarketParams, *, Y: float = 0.5, beta: float = 0.5,
                          impact_model: str = "sqrt", eta: float = 0.142, almgren_beta: float = 0.6):
    """Decomposed cost (bps) of trading `notional_usd` at a given `participation` rate.

    Returns a dict of arrays (vectorises over notional and participation). Keys:
    participation, fill_minutes, spread_bps, impact_bps, timing_risk_bps (1σ),
    expected_slippage_bps (spread+impact, the drags), risk_adjusted_bps (+1σ timing).

    Expected (mean) slippage: spread + impact (+ signed entry drag). Timing risk is a
    separate variance channel and is deliberately NOT included in this number.

    `impact_model`: "sqrt" (default, the adopted-Y band, `Y`/`beta` control it) or "almgren"
    (Almgren et al. 2005's fitted temporary-impact term, `eta`/`almgren_beta` control it — `Y`/
    `beta` are ignored in this branch). The library DEFAULT stays "sqrt"; switching it is an
    owner decision, not made here (see research/capacity/01_almgren_envelope).
    """
    if impact_model not in IMPACT_MODELS:
        raise ValueError(f"impact_model must be one of {IMPACT_MODELS}, got {impact_model!r}")
    t = fill_minutes(notional_usd, participation, p)
    if impact_model == "sqrt":
        impact = Y * p.sigma_daily_bps * np.power(participation, beta)
    else:
        impact = almgren_temporary(participation, p.sigma_daily_bps, eta=eta, beta=almgren_beta)
    timing = p.sigma_1min_bps * np.sqrt(t)
    spread = p.half_spread_bps
    return {
        "participation": participation,
        "fill_minutes": t,
        "spread_bps": spread,
        "impact_bps": impact,
        "timing_risk_bps": timing,
        "expected_slippage_bps": spread + impact,
        "risk_adjusted_bps": spread + impact + timing,
    }


def optimal_participation(notional_usd, p: MarketParams, *, Y: float = 0.5, beta: float = 0.5,
                          lam: float = 1.0, grid: np.ndarray | None = None,
                          impact_model: str = "sqrt") -> float:
    """Participation rate minimising expected slippage + λ·(1σ timing) — the A–C optimum.

    `lam` is risk aversion on the timing term (1.0 = weight the 1σ move as a cost).
    """
    grid = np.linspace(0.002, 0.6, 400) if grid is None else grid
    c = expected_slippage_bps(notional_usd, grid, p, Y=Y, beta=beta, impact_model=impact_model)
    obj = c["expected_slippage_bps"] + lam * c["timing_risk_bps"]
    return float(grid[int(np.argmin(obj))])


def capacity_at_horizon(budget_bps, p: MarketParams, exec_minutes: float, *,
                        Y: float = 0.5, beta: float = 0.5) -> float:
    """Largest notional ($) whose **expected** cost (spread+impact) stays within
    `budget_bps` when the order must be filled in `exec_minutes`.

    Filling within a fixed window pins participation to p = (Q/ADV)·(day/T), so impact
    grows with √Q. Solving spread + Y·σ·√p ≤ budget for Q gives a closed form.
    """
    room = budget_bps - p.half_spread_bps
    if room <= 0:
        return 0.0
    part = np.power(room / (Y * p.sigma_daily_bps), 1.0 / beta)   # max participation for budget
    f = exec_minutes / p.minutes_per_day
    return part * f * p.adv_usd   # Q = participation · (volume over the window)
