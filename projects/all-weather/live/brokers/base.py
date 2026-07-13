"""
live/brokers/base.py
====================
Broker-agnostic types and the Broker protocol. Every concrete broker
(Alpaca, Tastytrade, ...) implements this protocol; the rebalancer imports
ONLY from this module.

The rebalancer does not know what broker it is talking to. All concrete
imports are confined to live/brokers/<broker>.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Protocol, runtime_checkable


# ===========================================================================
# Value types
# ===========================================================================

@dataclass(frozen=True)
class PositionSnapshot:
    """One position normalised for planning. qty_available reflects any
    settlement / hold restrictions the broker reports.

    oldest_acquisition_date is best-effort. Brokers that do not expose
    lot-level dates return None here; the rebalancer consults the local
    lot ledger (live/lots.py) when None.
    """

    symbol: str
    qty: float
    qty_available: float
    market_value: float
    current_price: float
    avg_entry_price: float = 0.0
    oldest_acquisition_date: date | None = None


@dataclass(frozen=True)
class AccountSnapshot:
    equity: float
    cash: float
    buying_power: float


@dataclass(frozen=True)
class AssetMetadata:
    symbol: str
    tradable: bool
    fractionable: bool


@dataclass(frozen=True)
class OrderResult:
    """Result of submitting or polling one order."""

    order_id: str
    symbol: str
    side: str                  # "BUY" | "SELL"
    qty: float | None
    notional: float | None
    submitted_at: datetime
    status: str                # "filled" | "canceled" | "rejected" | "expired" | "timeout" | "open" | "pending"
    filled_qty: float = 0.0
    filled_avg_price: float = 0.0
    filled_notional: float = 0.0


@dataclass(frozen=True)
class ActivityEvent:
    """Account activity event normalised across brokers.

    activity_type is one of:
        DIV         dividend or distribution
        DEPOSIT     external cash deposit (ACH, wire, journal-in)
        WITHDRAWAL  external cash withdrawal (ACH, wire, journal-out)
        FILL        trade fill (for reconciliation; informational)
        OTHER       broker-specific event we did not normalise
    """

    activity_type: str
    symbol: str | None
    amount: float              # positive = inflow to account, negative = outflow
    occurred_at: datetime
    raw: dict = field(default_factory=dict)


# ===========================================================================
# Broker protocol
# ===========================================================================

@runtime_checkable
class Broker(Protocol):
    """Minimal surface the rebalancer needs.

    Concrete brokers MUST set name, trading_mode, account_label as instance
    attributes after construction.
    """

    name: str           # "alpaca" | "tastytrade" | ...
    trading_mode: str   # "paper" | "live"
    account_label: str

    # ---- Read account state ----

    def get_account(self) -> AccountSnapshot: ...

    def get_positions(self) -> dict[str, PositionSnapshot]: ...

    def get_asset(self, symbol: str) -> AssetMetadata: ...

    def get_open_orders(self, symbols: list[str]) -> list[OrderResult]: ...

    # ---- Market state ----

    def is_market_open(self) -> bool: ...

    def last_trading_day_of_month(self, ref: date) -> date: ...

    def is_trading_day(self, when: date) -> bool: ...

    # ---- Activities (dividends, transfers, fills) ----

    def fetch_activities(
        self,
        *,
        types: list[str],
        since: datetime,
        symbols: list[str] | None = None,
    ) -> list[ActivityEvent]: ...

    # ---- Orders ----

    def submit_market_order(
        self,
        *,
        symbol: str,
        side: str,
        qty: float | None = None,
        notional: float | None = None,
    ) -> OrderResult: ...

    def get_order(self, order_id: str) -> OrderResult: ...


# ===========================================================================
# Helpers
# ===========================================================================

TERMINAL_ORDER_STATUSES = frozenset({"filled", "canceled", "rejected", "expired", "timeout"})


def is_terminal_status(status: str) -> bool:
    return status.lower() in TERMINAL_ORDER_STATUSES


def normalise_side(side: str) -> str:
    """Normalise broker side strings to the canonical 'BUY'/'SELL'."""
    s = side.strip().upper()
    if s in {"BUY", "B"}:
        return "BUY"
    if s in {"SELL", "S"}:
        return "SELL"
    raise ValueError(f"Unknown order side: {side!r}")


def normalise_activity_type(raw_type: str, *, mapping: dict[str, str]) -> str:
    """Map a broker-specific activity-type string to our canonical enum."""
    return mapping.get(raw_type.upper(), "OTHER")


def sanitize_activity_payload(raw: Any) -> dict:
    """Best-effort conversion of an arbitrary broker activity payload to a
    JSON-safe dict. PII / credentials are not expected in activity payloads,
    but we still strip anything that looks like a token.
    """
    if isinstance(raw, dict):
        out = {}
        for key, value in raw.items():
            lower_key = str(key).lower()
            if any(token in lower_key for token in ("password", "secret", "token", "auth")):
                continue
            out[str(key)] = sanitize_activity_payload(value)
        return out
    if isinstance(raw, (list, tuple)):
        return [sanitize_activity_payload(item) for item in raw]  # type: ignore[return-value]
    if isinstance(raw, (str, int, float, bool)) or raw is None:
        return raw  # type: ignore[return-value]
    if isinstance(raw, (date, datetime)):
        return raw.isoformat()
    return str(raw)
