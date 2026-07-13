"""Slippage / market-impact library for the volume_research project.

A small, dependency-light (numpy + pandas) pre-trade cost model: spread (Corwin–Schultz),
market impact (√-law), and delay/timing risk, composed into a size-aware round-trip cost.

Quickstart:

    from slippage import calibrate, CostModel
    cal = calibrate(daily_ohlcv, intraday_15min)
    model = CostModel(cal.normal)
    rt = model.roundtrip(notional_usd=2_000_000)
    net_return = gross_return - rt.expected_slippage_bps / 1e4   # impact+spread = mean drag

See `slippage/README.md` for the integration guide and the Y-band / capacity caveats.
"""

from .alpha_decay import alpha_forfeit_frac
from .calibrate import Calibration, calibrate
from .cost import (
    MarketParams, expected_slippage_bps, fill_minutes, optimal_participation, capacity_at_horizon,
)
from .delay import delay_costs_bps, timing_risk_bps
from .impact import capacity, impact_bps, participation_for_impact
from .interruption import interruption_hazard, interruption_cost
from .model import CostModel, RoundTripCost
from .plan import ExecutionPlan, plan_execution
from .predict import predict_slippage
from .schedule import schedule_order, Schedule, ScheduleSlice
from .spread import (
    corwin_schultz, corwin_schultz_intraday, edge, edge_intraday, half_spread_bps,
)
from .state import (
    MarketState, VolumeProfile, SpreadCurve, VolRegimeBounds, estimate_state,
)

__all__ = [
    # facade (start here)
    "CostModel", "RoundTripCost", "calibrate", "Calibration",
    "plan_execution", "ExecutionPlan",
    # building blocks
    "corwin_schultz", "corwin_schultz_intraday", "edge", "edge_intraday", "half_spread_bps",
    "delay_costs_bps", "timing_risk_bps",
    "impact_bps", "capacity", "participation_for_impact",
    "MarketParams", "expected_slippage_bps", "fill_minutes", "optimal_participation",
    "capacity_at_horizon",
    "MarketState", "VolumeProfile", "SpreadCurve", "VolRegimeBounds", "estimate_state",
    "predict_slippage",
    "alpha_forfeit_frac",
    "interruption_hazard", "interruption_cost",
    "schedule_order", "Schedule", "ScheduleSlice",
]
