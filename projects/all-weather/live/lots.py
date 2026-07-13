"""
live/lots.py
============
FIFO lot ledger with minimum-holding-period enforcement.

Why
---
US accounts holding ETFs as part of a tax-efficient strategy should hold each
lot for at least 31 days.  Some brokers (e.g. Tastytrade) don't expose lot-level
acquisition dates via their API, so we track them ourselves.

How
---
On every executed BUY a new lot is appended to the ledger.
On every executed SELL the oldest lots are removed first (FIFO).

Holding-period check
--------------------
Before building the rebalance plan, ``holding_period_blocked_symbols`` is
called.  Any symbol whose YOUNGEST lot is within ``min_hold_days`` of today is
added to the blocked set.  The rebalancer then marks those rows
``holding_blocked=True`` and skips sells for them.

Note: we block on the YOUNGEST lot (most recently acquired) because as long as
we hold shares acquired recently we are within the hold window.  Older lots in
the same symbol are fine on their own, but a FIFO sell would not touch them
while the younger lot exists.

Ledger file
-----------
    live/logs/lots_<broker>_<mode>_<account>_<strategy>.json

Schema::

    {
        "SPY": [
            {"qty": 5.123, "price": 510.45, "acquired_on": "2025-10-31"},
            {"qty": 1.000, "price": 525.10, "acquired_on": "2025-11-30"}
        ],
        ...
    }

First-time setup
----------------
Call ``initialise_lots`` (triggered by ``--initialize-lots``).  Existing
positions are added as a single lot with today's date minus
``assume_held_days`` (default 0 = immediately sellable).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from live.brokers.base import PositionSnapshot


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Lot:
    qty: float
    price: float          # average cost basis per share (informational)
    acquired_on: date

    def to_dict(self) -> dict:
        return {
            "qty": self.qty,
            "price": self.price,
            "acquired_on": self.acquired_on.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Lot":
        return cls(
            qty=float(d["qty"]),
            price=float(d["price"]),
            acquired_on=date.fromisoformat(d["acquired_on"]),
        )


# LotLedger maps symbol -> list of Lot (oldest first)
LotLedger = dict[str, list[Lot]]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def load_ledger(path: str) -> LotLedger:
    """Return the lot ledger from disk, or an empty dict if the file is absent."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        raw: dict[str, list[dict]] = json.load(fh)
    return {
        sym: [Lot.from_dict(lot) for lot in lots]
        for sym, lots in raw.items()
    }


def save_ledger(path: str, ledger: LotLedger) -> None:
    """Persist the lot ledger to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    serialised = {
        sym: [lot.to_dict() for lot in lots]
        for sym, lots in ledger.items()
        if lots  # skip empty lists
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(serialised, fh, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Ledger mutations
# ---------------------------------------------------------------------------

def add_lot(
    ledger: LotLedger,
    symbol: str,
    qty: float,
    price: float,
    acquired_on: date,
) -> None:
    """Record a new buy lot (mutates ledger in-place).

    Appends to the end (newest lot last).  Lists are kept oldest-first so
    FIFO removal starts at index 0.
    """
    if qty <= 0:
        return
    ledger.setdefault(symbol, []).append(Lot(qty=qty, price=price, acquired_on=acquired_on))


def remove_lots_fifo(
    ledger: LotLedger,
    symbol: str,
    qty_sold: float,
) -> float:
    """Remove qty_sold shares from the oldest lots (FIFO, mutates in-place).

    Returns the qty actually removed (may be less than qty_sold if the ledger
    has fewer shares than requested — ledger out-of-sync with broker).
    """
    lots = ledger.get(symbol, [])
    remaining = qty_sold
    while lots and remaining > 1e-9:
        oldest = lots[0]
        if oldest.qty <= remaining + 1e-9:
            remaining -= oldest.qty
            lots.pop(0)
        else:
            lots[0] = Lot(
                qty=oldest.qty - remaining,
                price=oldest.price,
                acquired_on=oldest.acquired_on,
            )
            remaining = 0.0
    if not lots:
        ledger.pop(symbol, None)
    return qty_sold - remaining  # actual qty removed


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def youngest_lot_date(ledger: LotLedger, symbol: str) -> date | None:
    """Return the acquisition date of the most recently added lot, or None."""
    lots = ledger.get(symbol, [])
    if not lots:
        return None
    return max(lot.acquired_on for lot in lots)


def oldest_lot_date(ledger: LotLedger, symbol: str) -> date | None:
    """Return the acquisition date of the oldest lot, or None."""
    lots = ledger.get(symbol, [])
    if not lots:
        return None
    return min(lot.acquired_on for lot in lots)


def days_since_youngest(ledger: LotLedger, symbol: str, today: date | None = None) -> int | None:
    """Return how many days since the most recently acquired lot, or None if not tracked."""
    yd = youngest_lot_date(ledger, symbol)
    if yd is None:
        return None
    return (today or date.today() - yd).days if isinstance(today, type(None)) else (today - yd).days


def is_holding_blocked(
    ledger: LotLedger,
    symbol: str,
    min_hold_days: int,
    today: date | None = None,
) -> bool:
    """True if any lot in the ledger is younger than min_hold_days.

    A blocked symbol will not be sold in the current rebalance cycle.
    """
    youngest = youngest_lot_date(ledger, symbol)
    if youngest is None:
        # Not in ledger → treat as untracked → allow sell (conservative default)
        return False
    ref = today or date.today()
    return (ref - youngest).days < min_hold_days


def holding_period_blocked_symbols(
    ledger: LotLedger,
    symbols: list[str],
    min_hold_days: int,
    today: date | None = None,
) -> set[str]:
    """Return the subset of symbols that are within the minimum holding window."""
    return {
        sym for sym in symbols
        if is_holding_blocked(ledger, sym, min_hold_days, today)
    }


def holding_period_summary(
    ledger: LotLedger,
    symbols: list[str],
    min_hold_days: int,
    today: date | None = None,
) -> list[dict]:
    """Return a list of dicts for display / logging."""
    ref = today or date.today()
    rows = []
    for sym in symbols:
        youngest = youngest_lot_date(ledger, sym)
        oldest = oldest_lot_date(ledger, sym)
        if youngest is None:
            rows.append({
                "symbol": sym,
                "oldest_lot": None,
                "youngest_lot": None,
                "days_held": None,
                "blocked": False,
                "note": "not in ledger",
            })
        else:
            days = (ref - youngest).days
            blocked = days < min_hold_days
            rows.append({
                "symbol": sym,
                "oldest_lot": oldest.isoformat() if oldest else None,
                "youngest_lot": youngest.isoformat(),
                "days_held": days,
                "blocked": blocked,
                "note": f"{min_hold_days - days}d remaining" if blocked else "eligible",
            })
    return rows


# ---------------------------------------------------------------------------
# First-time initialisation
# ---------------------------------------------------------------------------

def initialise_lots(
    ledger_path: str,
    positions: dict[str, "PositionSnapshot"],
    allocation: dict[str, float],
    assume_held_days: int = 0,
) -> LotLedger:
    """Seed the lot ledger from current broker positions.

    Parameters
    ----------
    ledger_path      : where to persist the ledger
    positions        : current broker positions (from broker.get_positions())
    allocation       : strategy target allocation (only these symbols are seeded)
    assume_held_days : backdates the acquisition date by this many days
                       (0 = today = immediately sellable if min_hold_days=0;
                        32 = backdated 32 days = immediately sellable under 31-day rule)

    Returns the new ledger (already persisted).
    """
    acquired = date.today() - timedelta(days=assume_held_days)
    ledger: LotLedger = {}

    for sym in allocation:
        if sym in positions and positions[sym].qty > 0:
            add_lot(
                ledger,
                symbol=sym,
                qty=positions[sym].qty,
                price=positions[sym].avg_entry_price or positions[sym].current_price,
                acquired_on=acquired,
            )

    save_ledger(ledger_path, ledger)
    return ledger


# ---------------------------------------------------------------------------
# Path helper (mirrors the pattern in budget.py / rebalance.py)
# ---------------------------------------------------------------------------

def lot_ledger_path(
    logs_dir: str,
    broker_name: str,
    trading_mode: str,
    account_label: str,
    strategy_id: str,
) -> str:
    safe_acct = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in account_label)
    safe_strat = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in strategy_id)
    return os.path.join(
        logs_dir,
        f"lots_{broker_name}_{trading_mode}_{safe_acct}_{safe_strat}.json",
    )
