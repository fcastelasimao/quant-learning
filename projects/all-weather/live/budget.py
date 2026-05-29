"""
live/budget.py
==============
Per-account strategy budget — virtual sub-account within a larger brokerage account.

Problem
-------
You hold $50k at Tastytrade but only want the All-Weather strategy to manage $10k of it.
Pass ``--budget 10000`` and the rebalancer sizes every order against the strategy budget
rather than the full account equity.

How managed_capital is computed
--------------------------------
    managed_capital = strategy_positions_value + reserved_cash

* strategy_positions_value  : sum of market_value for all strategy symbols
* reserved_cash             : the uninvested portion of the budget

On ``--initialize-budget`` (first setup):
    reserved_cash = max(0, budget_cap - strategy_positions_value)

After each rebalance the strategy may hold less cash than the cap because prices
move; managed_capital floats with the market.  It grows only from:
  1. Strategy-symbol dividends (DIV activities for strategy tickers)
  2. Market appreciation of existing positions

State file
----------
    logs/budget_<broker>_<mode>_<account>_<strategy>.json

Schema::

    {
        "budget_cap":        10000.00,
        "reserved_cash":       250.00,
        "activity_cursor":  "2025-12-01T00:00:00+00:00",   // fetch activities after this
        "last_updated":     "2026-01-31T16:45:00-05:00"
    }

Usage from rebalance.py
-----------------------
::

    from live.budget import BudgetSnapshot, initialise_budget, load_budget, update_budget

    snap = load_budget(state_path, broker, positions, allocation)
    managed_capital = snap.managed_capital
    # ...after successful run:
    save_budget(state_path, snap)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from live.brokers.base import Broker, PositionSnapshot

_LOG = logging.getLogger("rebalancer.budget")


# ---------------------------------------------------------------------------
# Snapshot type
# ---------------------------------------------------------------------------

@dataclass
class BudgetSnapshot:
    """Point-in-time view of the strategy budget."""

    budget_cap: float             # immutable dollar cap set by the user
    reserved_cash: float          # uninvested strategy cash
    positions_value: float        # sum of market_value for strategy symbols
    activity_cursor: datetime     # fetch activities after this timestamp
    last_updated: datetime

    @property
    def managed_capital(self) -> float:
        """Total virtual equity owned by this strategy."""
        return self.positions_value + self.reserved_cash


# ---------------------------------------------------------------------------
# State file helpers
# ---------------------------------------------------------------------------

def _budget_state_path(
    logs_dir: str,
    broker_name: str,
    trading_mode: str,
    account_label: str,
    strategy_id: str,
) -> str:
    safe_acct = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in account_label)
    safe_strat = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in strategy_id)
    fname = f"budget_{broker_name}_{trading_mode}_{safe_acct}_{safe_strat}.json"
    return os.path.join(logs_dir, fname)


def _positions_value(
    positions: dict[str, "PositionSnapshot"],
    allocation: dict[str, float],
) -> float:
    return sum(
        positions[sym].market_value
        for sym in allocation
        if sym in positions
    )


def load_state(state_path: str) -> dict | None:
    """Return raw state dict or None if the file does not exist."""
    if not os.path.exists(state_path):
        return None
    with open(state_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_state(state_path: str, snap: BudgetSnapshot) -> None:
    """Persist a BudgetSnapshot to disk."""
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    payload = {
        "budget_cap": snap.budget_cap,
        "reserved_cash": round(snap.reserved_cash, 6),
        "activity_cursor": snap.activity_cursor.isoformat(),
        "last_updated": snap.last_updated.isoformat(),
    }
    with open(state_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def initialise_budget(
    state_path: str,
    budget_cap: float,
    positions: dict[str, "PositionSnapshot"],
    allocation: dict[str, float],
    logger: logging.Logger | None = None,
) -> BudgetSnapshot:
    """Seed budget state from current positions (called on ``--initialize-budget``).

    Computes strategy_positions_value from current broker positions and sets
    reserved_cash = budget_cap - positions_value  (floored at 0).

    Overwrites any existing state file.
    """
    log = logger or _LOG
    pos_val = _positions_value(positions, allocation)
    reserved = max(0.0, budget_cap - pos_val)
    now = datetime.now(timezone.utc)
    snap = BudgetSnapshot(
        budget_cap=budget_cap,
        reserved_cash=reserved,
        positions_value=pos_val,
        activity_cursor=now,
        last_updated=now,
    )
    save_state(state_path, snap)
    log.info(
        f"Budget initialised: cap=${budget_cap:,.2f}  "
        f"positions=${pos_val:,.2f}  reserved_cash=${reserved:,.2f}  "
        f"managed_capital=${snap.managed_capital:,.2f}"
    )
    return snap


def load_budget(
    state_path: str,
    positions: dict[str, "PositionSnapshot"],
    allocation: dict[str, float],
    logger: logging.Logger | None = None,
) -> BudgetSnapshot | None:
    """Load the budget state and recompute positions_value from live positions.

    Returns None if no state file exists (budget not yet initialised).
    Call ``initialise_budget`` first or prompt the user to run
    ``--initialize-budget``.
    """
    log = logger or _LOG
    raw = load_state(state_path)
    if raw is None:
        return None

    pos_val = _positions_value(positions, allocation)
    cursor = datetime.fromisoformat(raw["activity_cursor"])
    last_updated = datetime.fromisoformat(raw["last_updated"])

    snap = BudgetSnapshot(
        budget_cap=float(raw["budget_cap"]),
        reserved_cash=float(raw["reserved_cash"]),
        positions_value=pos_val,
        activity_cursor=cursor,
        last_updated=last_updated,
    )
    log.debug(
        f"Budget loaded: cap=${snap.budget_cap:,.2f}  "
        f"positions=${pos_val:,.2f}  reserved_cash=${snap.reserved_cash:,.2f}  "
        f"managed_capital=${snap.managed_capital:,.2f}"
    )
    return snap


def update_from_dividends(
    snap: BudgetSnapshot,
    broker: "Broker",
    allocation: dict[str, float],
    logger: logging.Logger | None = None,
) -> BudgetSnapshot:
    """Fetch strategy-symbol dividends since the last cursor and credit reserved_cash.

    Returns an updated BudgetSnapshot (does NOT persist — call save_state after).
    """
    log = logger or _LOG
    try:
        events = broker.fetch_activities(
            types=["DIV"],
            since=snap.activity_cursor,
            symbols=list(allocation.keys()),
        )
    except Exception as exc:
        log.warning(f"Could not fetch dividend activities: {exc}")
        return snap

    total_div = 0.0
    for ev in events:
        if ev.amount > 0:
            total_div += ev.amount
            log.info(
                f"Dividend credit: {ev.symbol} ${ev.amount:.4f} "
                f"on {ev.occurred_at.date().isoformat()}"
            )

    if total_div:
        log.info(f"Total dividends credited to budget: ${total_div:.4f}")

    new_cursor = max(
        (ev.occurred_at for ev in events),
        default=snap.activity_cursor,
    )
    # Advance cursor slightly past the last seen event to avoid double-counting
    if events:
        from datetime import timedelta
        new_cursor = new_cursor + timedelta(seconds=1)

    return BudgetSnapshot(
        budget_cap=snap.budget_cap,
        reserved_cash=snap.reserved_cash + total_div,
        positions_value=snap.positions_value,
        activity_cursor=new_cursor,
        last_updated=datetime.now(timezone.utc),
    )


def reconcile_after_trades(
    snap: BudgetSnapshot,
    positions: dict[str, "PositionSnapshot"],
    allocation: dict[str, float],
    target_managed_capital: float,
) -> BudgetSnapshot:
    """Refresh budget state from final positions after an executed rebalance."""
    pos_val = _positions_value(positions, allocation)
    return BudgetSnapshot(
        budget_cap=snap.budget_cap,
        reserved_cash=max(0.0, target_managed_capital - pos_val),
        positions_value=pos_val,
        activity_cursor=snap.activity_cursor,
        last_updated=datetime.now(timezone.utc),
    )


def get_managed_capital(
    state_path: str,
    budget_cap: float | None,
    account_equity: float,
    positions: dict[str, "PositionSnapshot"],
    allocation: dict[str, float],
    broker: "Broker",
    ignore_budget: bool,
    logger: logging.Logger | None = None,
) -> tuple[float, BudgetSnapshot | None]:
    """Resolve the capital to size rebalance orders against.

    Priority:
        1. If ignore_budget: use account_equity
        2. If budget state exists: compute managed_capital from state + live dividends
        3. If budget_cap provided but state missing: warn, use min(budget_cap, equity)
        4. Fallback: use account_equity

    Returns (managed_capital, snapshot_or_None).
    The caller should call save_state(state_path, snap) after a successful run.
    """
    log = logger or _LOG

    if ignore_budget:
        log.info(f"Budget ignored; using full account equity ${account_equity:,.2f}")
        return account_equity, None

    snap = load_budget(state_path, positions, allocation, log)

    if snap is not None:
        snap = update_from_dividends(snap, broker, allocation, log)
        capital = snap.managed_capital
        log.info(
            f"Budget active: cap=${snap.budget_cap:,.2f}  "
            f"managed_capital=${capital:,.2f}  "
            f"(positions=${snap.positions_value:,.2f}  "
            f"reserved=${snap.reserved_cash:,.2f})"
        )
        return capital, snap

    if budget_cap is not None:
        capital = min(budget_cap, account_equity)
        log.warning(
            f"Budget state not found at {state_path}. "
            f"Run --initialize-budget first for accurate tracking. "
            f"Using min(budget_cap, equity) = ${capital:,.2f} this run."
        )
        return capital, None

    log.info(f"No budget configured; using full account equity ${account_equity:,.2f}")
    return account_equity, None
