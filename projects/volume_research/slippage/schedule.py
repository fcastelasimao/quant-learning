"""schedule_order — the centerpiece: turn a decided order into a full child-order schedule.

Extends (does not replace) `plan.py::plan_execution`. Where `plan_execution` picks a single
participation rate for a fixed cadence, `schedule_order` picks the **horizon itself** — trading
off execution cost saved by going slower against the alpha forfeited by delaying (E01) and the
risk of a mid-fill interruption (E02) — then slices the order across that horizon shaped by
predicted interval volume (P01), never exceeding a POV cap per slice.

    schedule_order(notional_usd, side, state, price, edge_bps=..., pov_cap=0.10, mode="cancel")
      -> Schedule(slices=[...], horizon_min=h*, expected_slippage_bps, expected_slippage_band_bps,
                  alpha_forfeit_bps, interruption_summary, feasible, assumptions)

**Horizon choice:** h* = argmin over h of
    predict_slippage(notional, h).mean_bps
        + edge_bps * (1 - hazard(h)) * alpha_forfeit_frac(h)           # completes, delayed h
        + edge_bps *      hazard(h)  * interruption_cost(h, phi=0.5, mode)   # interrupted mid-fill
— impact/timing cost saved by going slower (predict_slippage) vs alpha forfeited (E01) vs
interruption risk (E02), the last two combined as one expectation over the interruption event
(see `alpha_interruption_bps`). This replaces the hard 15-min pin used elsewhere in scheduler-land: S12
found the strategy's edge is multi-day and the profitable trades hold >= 1 day, so a longer
horizon is not automatically wrong the way it would be for a strategy that must act within one
15-min bar. `phi=0.5` in the interruption term is a documented simplification — the average fill
fraction at a random interruption time within [0, h], not tracked precisely.

**Slice allocation:** proportional to P01's predicted per-bin volume (VWAP-shaped) if a
`VolumeProfile` is supplied; otherwise flat at `state.expected_interval_volume` (documented
fallback — MarketState only carries a point estimate, not a forward curve). Each slice is capped
at `pov_cap` of its bin's predicted volume; a waterfall re-distributes any capped excess across
the remaining bins, extending the schedule past h* if needed to fit the whole order, up to a
hard cap — beyond that, `feasible=False`.

Pure function, no IO.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .alpha_decay import alpha_forfeit_frac
from .interruption import interruption_cost, interruption_hazard
from .predict import predict_slippage
from .state import MarketState, VolumeProfile, session_bin_label

DEFAULT_POV_CAP = 0.10
DEFAULT_MODE = "cancel"
INTERRUPTION_PHI = 0.5              # documented simplification, see module docstring
MAX_SCHEDULE_BINS = 200             # ~2 days of 15-min bins; beyond this, infeasible
_H_GRID_MIN = np.concatenate([
    np.arange(1, 60, 1), np.arange(60, 780, 15), np.arange(780, 3120, 60),
]).astype(float)


@dataclass
class ScheduleSlice:
    """One child order in the schedule."""
    time_offset_min: float
    child_notional_usd: float
    order_style: str   # "cross" | "limit_chase"


@dataclass
class Schedule:
    """A full pre-trade execution schedule for one order (one-way)."""
    notional_usd: float
    horizon_min: float                       # h*, the chosen horizon
    slices: list[ScheduleSlice]
    expected_slippage_bps: float             # at h*, order_style="cross"
    expected_slippage_band_bps: tuple[float, float]
    alpha_forfeit_bps: float                 # edge_bps * (1-hazard(h*)) * g(h*), completion branch
    interruption_summary: dict               # {"hazard": P(interrupted by h*), "expected_cost_bps": ...}
    feasible: bool
    assumptions: str

    def __str__(self) -> str:
        return self.assumptions


def alpha_interruption_bps(h: float, state: MarketState, edge_bps: float,
                           mode: str) -> tuple[float, float]:
    """Expected alpha-forfeiture and interruption cost (bps) of delaying entry to horizon `h`,
    as a proper expectation over the interruption event:

        alpha     = edge * (1 - hazard(h)) * g(h)                    # order completes, delayed h
        interrupt = edge *      hazard(h)  * interruption_cost(h)    # stop/flip fires mid-fill

    The `(1 - hazard)` weight is the fix for a double-count: earlier code charged `edge * g(h)`
    unconditionally AND added the interruption term — but `interruption_cost` in "cancel" mode is
    `phi*g(h) + (1-phi)`, which already prices the filled part's g(h) on the interrupted branch,
    so the un-weighted `g(h)` counted the completion-branch forfeiture in both branches (audit fix
    2026-07-09). The two channels are returned separately so callers can report/plot them, but
    together they are the clean expectation the horizon search minimizes.

    Shared by `_objective`, `schedule_order`'s reporting, and the E03/C02 research builds so the
    decomposition can't drift out of sync between them.
    """
    haz = interruption_hazard(h, state)
    g = alpha_forfeit_frac(h, state.symbol)
    ic = interruption_cost(h, INTERRUPTION_PHI, mode, state.symbol)
    return edge_bps * (1.0 - haz) * g, edge_bps * haz * ic


def _objective(h: float, notional_usd: float, side: str, state: MarketState, price: float,
              edge_bps: float, mode: str) -> float:
    pred = predict_slippage(notional_usd, side, "cross", state, price, latency_min=h)
    alpha_cost, interrupt_cost = alpha_interruption_bps(h, state, edge_bps, mode)
    return pred["mean_bps"] + alpha_cost + interrupt_cost


def _choose_horizon(notional_usd, side, state, price, edge_bps, mode) -> float:
    objs = [_objective(h, notional_usd, side, state, price, edge_bps, mode) for h in _H_GRID_MIN]
    return float(_H_GRID_MIN[int(np.argmin(objs))])


def _bin_volume(i: int, state: MarketState, price: float, profile: VolumeProfile | None) -> float:
    """Predicted volume (dollars) for the i-th 15-min bin after `state.ts`."""
    if profile is None:
        return state.expected_interval_volume * price
    bin_ts = state.ts + np.timedelta64(15 * i, "m")
    label = session_bin_label(bin_ts)
    daily_total = state.expected_interval_volume / max(profile.bin_share_mean.get(state.bin_label, 1e-9), 1e-9)
    share = profile.bin_share_mean.get(label, np.mean(list(profile.bin_share_mean.values())))
    return share * daily_total * price


def _waterfall_allocate(notional_usd: float, caps: np.ndarray) -> tuple[np.ndarray, float]:
    """Distribute `notional_usd` across bins proportional to `caps` (the VWAP weight, and also
    the hard per-bin ceiling), capping each at its own `caps[i]` and redistributing any excess
    across bins with remaining room. Returns (allocation, unallocated_remainder)."""
    caps = np.asarray(caps, float)
    alloc = np.zeros_like(caps)
    free = np.ones_like(caps, dtype=bool)
    remaining = notional_usd
    for _ in range(50):
        if remaining <= 1e-6 or not free.any():
            break
        room = caps - alloc
        w = np.where(free, caps, 0.0)
        if w.sum() <= 0:
            break
        share = remaining * w / w.sum()
        take = np.minimum(share, np.where(free, room, 0.0))
        alloc += take
        remaining -= take.sum()
        free = free & (alloc < caps - 1e-9)
    return alloc, max(remaining, 0.0)


def _build_slices(notional_usd, state, price, pov_cap, profile, order_style, min_bins) -> tuple[list[ScheduleSlice], bool]:
    n_bins = max(1, min_bins)
    while n_bins <= MAX_SCHEDULE_BINS:
        vols = np.array([_bin_volume(i, state, price, profile) for i in range(n_bins)])
        caps = pov_cap * vols
        alloc, remainder = _waterfall_allocate(notional_usd, caps)
        if remainder <= 1e-6:
            slices = [ScheduleSlice(time_offset_min=15.0 * i, child_notional_usd=float(alloc[i]),
                                    order_style=order_style)
                      for i in range(n_bins) if alloc[i] > 1e-6]
            return slices, True
        n_bins += 1
    return [], False


def schedule_order(notional_usd: float, side: str, state: MarketState, price: float, *,
                   edge_bps: float, pov_cap: float = DEFAULT_POV_CAP, mode: str = DEFAULT_MODE,
                   volume_profile: VolumeProfile | None = None,
                   order_style: str = "cross") -> Schedule:
    """Build a full child-order schedule for `notional_usd` of `state.symbol`.

    Parameters
    ----------
    edge_bps : the strategy's expected per-trade edge (bps) — a caller-supplied, strategy-level
        input (not measured by this library, same pattern as `entry_drag_bps` elsewhere). Used to
        weight the alpha-forfeiture and interruption-cost terms in the horizon search.
    pov_cap : max fraction of a bin's predicted volume any one slice may take (default 10%).
    mode : "cancel" (default) or "complete_now" — the interruption-cost model (E02); "cancel" is
        the more realistic default for an ENTRY schedule (a firing exit stop means the position is
        being closed, so an unfilled entry residue should simply not be chased further).
    volume_profile : P01's `VolumeProfile` for VWAP-shaped slicing across the whole horizon; if
        omitted, slices are flat at `state.expected_interval_volume` (documented fallback).
    order_style : "cross" or "limit_chase" — applied uniformly to every slice (P02: "cross" is
        expected to win almost always; carried through as a caller override).
    """
    h_star = _choose_horizon(notional_usd, side, state, price, edge_bps, mode)
    min_bins = max(1, int(np.ceil(h_star / 15.0)))
    slices, feasible = _build_slices(notional_usd, state, price, pov_cap, volume_profile,
                                     order_style, min_bins)

    pred = predict_slippage(notional_usd, side, "cross", state, price, latency_min=h_star)
    alpha_bps, interrupt_bps = alpha_interruption_bps(h_star, state, edge_bps, mode)
    haz = interruption_hazard(h_star, state)

    actual_horizon = 15.0 * len(slices) if slices else h_star
    note = (f"h*={h_star:.0f} min ({len(slices)} slice(s) over ~{actual_horizon:.0f} min): "
           f"expected cost {pred['mean_bps']:.1f} bps, alpha forfeit {alpha_bps:.1f} bps, "
           f"interruption {interrupt_bps:.1f} bps (hazard {haz*100:.0f}%).")
    if not feasible:
        note = ("WARNING: could not fit the order within the POV cap in "
                f"{MAX_SCHEDULE_BINS} bins — reduce size or raise pov_cap. " + note)

    return Schedule(
        notional_usd=float(notional_usd), horizon_min=h_star, slices=slices,
        expected_slippage_bps=float(pred["mean_bps"]),
        expected_slippage_band_bps=(float(pred["components"]["impact_band_bps"][0] + pred["components"]["spread_bps"]),
                                    float(pred["components"]["impact_band_bps"][2] + pred["components"]["spread_bps"])),
        alpha_forfeit_bps=float(alpha_bps),
        interruption_summary={"hazard": float(haz), "expected_cost_bps": float(interrupt_bps)},
        feasible=feasible, assumptions=note,
    )
