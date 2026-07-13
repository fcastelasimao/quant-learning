"""Delay / timing cost — the price drift between a decision and its fill.

Our strategies decide at each 15-min bar close, but the order fills *after* that.
The delay cost is the signed move from the decision price to the fill price:

    buy:  cost = (P_fill - P_decision) / P_decision      (paying up if price rose)
    sell: cost = (P_decision - P_fill) / P_decision

Because the trade direction is (to first order) uncorrelated with the unpredictable
short-horizon drift, the *expected* delay cost is ≈ 0 — this is a **risk** (the std of
the move), not a systematic drag. The exception is signal-correlated execution
(momentum fills adversely, mean-reversion favourably); we measure the raw symmetric
distribution here and flag that caveat downstream.

These are pure functions (no IO). Part of the slippage library alongside `spread`.
"""

from __future__ import annotations

import pandas as pd


def delay_costs_bps(
    close: pd.Series,
    horizons_min: list[int],
    *,
    decision_every_min: int = 15,
) -> pd.DataFrame:
    """Signed forward returns (bps) from each decision point to several horizons.

    Parameters
    ----------
    close : pd.Series
        1-min close prices, DatetimeIndex (ET), chronological.
    horizons_min : list[int]
        Fill delays in minutes (e.g. [1, 2, 3, 5, 10, 15]).
    decision_every_min : int
        Decision cadence; decisions are taken on the bar grid aligned to the 09:30
        open (default 15 → 09:45, 10:00, …).

    Returns
    -------
    pd.DataFrame
        Indexed by decision datetime, one column per horizon, values = signed forward
        return in bps (mean ≈ 0; the per-column std is the timing risk). Computed
        **within session** (a fill window never crosses the overnight gap), on a
        gap-filled 1-min grid so a horizon of k means k *minutes*, not k *bars*.
    """
    open_min = 570  # 09:30
    parts = []
    for _, g in close.groupby(close.index.normalize()):
        # Reindex to a contiguous 1-min grid and forward-fill, so shift(-k) == k minutes
        # even when a thin minute had no trade.
        grid = pd.date_range(g.index.min(), g.index.max(), freq="1min")
        c = g.reindex(grid).ffill()
        mins = c.index.hour * 60 + c.index.minute - open_min
        decision = (mins >= 0) & (mins % decision_every_min == 0)
        cols = {k: (c.shift(-k) / c - 1.0) * 1e4 for k in horizons_min}
        df = pd.DataFrame(cols, index=c.index)[decision]
        parts.append(df)
    return pd.concat(parts).dropna(how="all")


def timing_risk_bps(costs: pd.DataFrame) -> pd.DataFrame:
    """Summarise a delay-cost frame into per-horizon stats (bps).

    Returns mean (≈0 check), std (the timing risk), and the adverse 5%/95% tails.
    """
    return pd.DataFrame({
        "mean_bps": costs.mean(),
        "std_bps": costs.std(),
        "p05_bps": costs.quantile(0.05),
        "p95_bps": costs.quantile(0.95),
    })
