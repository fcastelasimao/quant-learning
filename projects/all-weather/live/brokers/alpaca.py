"""
live/brokers/alpaca.py
======================
Alpaca implementation of the Broker protocol. Built on alpaca-py.

Credentials precedence (account_label="my_label"):
    1. BROKER_ALPACA_<MY_LABEL>_KEY / BROKER_ALPACA_<MY_LABEL>_SECRET
    2. APCA_API_KEY_ID_<MY_LABEL>  / APCA_API_SECRET_KEY_<MY_LABEL>   (legacy)
    3. APCA_API_KEY_ID             / APCA_API_SECRET_KEY              (legacy default)
    4. ALPACA_API_KEY              / ALPACA_SECRET_KEY                (api_keys.env default)

This ordering preserves the env-var convention from the original
live/_legacy/alpaca_rebalance.py while also loading the shared QuantFinance
api_keys.env file.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone
from typing import Any

from live.env import load_api_keys_env

from .base import (
    AccountSnapshot,
    ActivityEvent,
    AssetMetadata,
    OrderResult,
    PositionSnapshot,
    normalise_activity_type,
    normalise_side,
    sanitize_activity_payload,
)

load_api_keys_env()

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
    from alpaca.trading.requests import (
        GetCalendarRequest,
        GetOrdersRequest,
        MarketOrderRequest,
    )
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "alpaca-py is not installed. Install it with:\n\n"
        "    conda run -n allweather pip install alpaca-py\n"
    ) from exc

try:
    from alpaca.trading.requests import GetAccountActivitiesRequest
    _HAS_ACTIVITIES_REQUEST = True
except ImportError:  # pragma: no cover - older alpaca-py
    GetAccountActivitiesRequest = None  # type: ignore[assignment]
    _HAS_ACTIVITIES_REQUEST = False


# Mapping from Alpaca's activity_type codes to our canonical enum.
# See https://docs.alpaca.markets/reference/getaccountactivities
_ALPACA_ACTIVITY_MAP = {
    "DIV": "DIV",
    "DIVCGL": "DIV",
    "DIVCGS": "DIV",
    "DIVFEE": "DIV",
    "DIVFT": "DIV",
    "DIVNRA": "DIV",
    "DIVROC": "DIV",
    "DIVTW": "DIV",
    "DIVTXEX": "DIV",
    "CSD": "DEPOSIT",     # cash deposit
    "CSW": "WITHDRAWAL",
    "ACATC": "DEPOSIT",   # ACAT in
    "ACATS": "DEPOSIT",   # ACATS
    "JNLC": "DEPOSIT",    # journal cash in (positive) / out (negative — handled by sign)
    "JNLS": "DEPOSIT",
    "FILL": "FILL",
    "PTC": "FILL",
}


def _resolve_credentials(account_label: str) -> tuple[str, str]:
    """Read API key + secret from env vars following the documented precedence."""
    upper = account_label.upper()
    candidates = [
        (f"BROKER_ALPACA_{upper}_KEY", f"BROKER_ALPACA_{upper}_SECRET"),
        (f"APCA_API_KEY_ID_{upper}", f"APCA_API_SECRET_KEY_{upper}"),
    ]
    if account_label.lower() == "default":
        candidates.append(("APCA_API_KEY_ID", "APCA_API_SECRET_KEY"))
        candidates.append(("ALPACA_API_KEY", "ALPACA_SECRET_KEY"))

    for key_var, secret_var in candidates:
        key = os.getenv(key_var)
        secret = os.getenv(secret_var)
        if key and secret:
            return key, secret

    tried = ", ".join(f"{k}/{s}" for k, s in candidates)
    raise SystemExit(
        f"Missing Alpaca credentials for account '{account_label}'. "
        f"Set one of these env-var pairs: {tried}"
    )


def _coerce_date(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    return date.fromisoformat(str(value))


def _status_str(order: Any) -> str:
    """Normalise Alpaca order statuses into lowercase strings."""
    return str(order.status).split(".")[-1].lower()


def _to_iso_utc(when: datetime) -> str:
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class AlpacaBroker:
    """Concrete Broker for Alpaca paper or live accounts."""

    name = "alpaca"

    def __init__(self, *, trading_mode: str, account_label: str) -> None:
        if trading_mode not in {"paper", "live"}:
            raise ValueError(f"trading_mode must be 'paper' or 'live', got {trading_mode!r}")
        self.trading_mode = trading_mode
        self.account_label = account_label
        api_key, secret = _resolve_credentials(account_label)
        self._client = TradingClient(
            api_key=api_key, secret_key=secret, paper=(trading_mode == "paper")
        )

    # ---- account state ------------------------------------------------------

    def get_account(self) -> AccountSnapshot:
        account = self._client.get_account()
        return AccountSnapshot(
            equity=float(account.equity),
            cash=float(account.cash),
            buying_power=float(account.buying_power),
        )

    def get_positions(self) -> dict[str, PositionSnapshot]:
        positions = self._client.get_all_positions()
        out: dict[str, PositionSnapshot] = {}
        for position in positions:
            out[position.symbol] = PositionSnapshot(
                symbol=position.symbol,
                qty=float(position.qty),
                qty_available=float(position.qty_available or position.qty),
                market_value=abs(float(position.market_value or 0.0)),
                current_price=float(position.current_price or 0.0),
                avg_entry_price=float(getattr(position, "avg_entry_price", 0.0) or 0.0),
                oldest_acquisition_date=None,  # Alpaca does not expose lot-level dates
            )
        return out

    def get_asset(self, symbol: str) -> AssetMetadata:
        asset = self._client.get_asset(symbol)
        return AssetMetadata(
            symbol=symbol,
            tradable=bool(asset.tradable),
            fractionable=bool(getattr(asset, "fractionable", False)),
        )

    def get_open_orders(self, symbols: list[str]) -> list[OrderResult]:
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=list(symbols))
        orders = list(self._client.get_orders(filter=request))
        return [self._to_order_result(o) for o in orders]

    # ---- market state -------------------------------------------------------

    def is_market_open(self) -> bool:
        clock = self._client.get_clock()
        return bool(clock.is_open)

    def last_trading_day_of_month(self, ref: date) -> date:
        first_day = ref.replace(day=1)
        probe_end = first_day + timedelta(days=40)
        month_end = probe_end.replace(day=1) - timedelta(days=1)
        calendar = self._client.get_calendar(
            GetCalendarRequest(start=first_day, end=month_end)
        )
        trading_days = sorted(_coerce_date(day.date) for day in calendar)
        if not trading_days:
            raise RuntimeError("Alpaca calendar returned no trading days for this month.")
        return trading_days[-1]

    def is_trading_day(self, when: date) -> bool:
        calendar = self._client.get_calendar(GetCalendarRequest(start=when, end=when))
        return any(_coerce_date(day.date) == when for day in calendar)

    # ---- activities ---------------------------------------------------------

    def fetch_activities(
        self,
        *,
        types: list[str],
        since: datetime,
        symbols: list[str] | None = None,
    ) -> list[ActivityEvent]:
        """Pull account activities since the cursor and normalise them.

        `types` is the *canonical* enum ("DIV", "DEPOSIT", ...). We translate
        to the Alpaca-specific codes before calling the API. If the installed
        alpaca-py version does not expose GetAccountActivitiesRequest the
        method returns [] and the caller logs the limitation.
        """
        if not _HAS_ACTIVITIES_REQUEST:
            return []

        # Inverse map: canonical -> set of Alpaca codes to request
        inverse: dict[str, list[str]] = {}
        for code, canon in _ALPACA_ACTIVITY_MAP.items():
            inverse.setdefault(canon, []).append(code)

        wanted_codes: list[str] = []
        for canonical_type in types:
            wanted_codes.extend(inverse.get(canonical_type, []))
        wanted_codes = sorted(set(wanted_codes))

        results: list[ActivityEvent] = []
        for code in wanted_codes:
            try:
                req = GetAccountActivitiesRequest(
                    activity_types=[code],
                    after=since,
                )
                activities = self._client.get_account_activities(activity_filter=req)
            except Exception:
                # Some alpaca-py versions reject empty/old activity types — skip silently
                continue
            for raw in activities:
                event = self._activity_to_event(raw)
                if event is None:
                    continue
                if symbols and event.activity_type == "DIV":
                    if event.symbol not in set(symbols):
                        continue
                results.append(event)

        results.sort(key=lambda e: e.occurred_at)
        return results

    def _activity_to_event(self, raw: Any) -> ActivityEvent | None:
        raw_type = str(getattr(raw, "activity_type", "")).split(".")[-1]
        canonical = normalise_activity_type(raw_type, mapping=_ALPACA_ACTIVITY_MAP)

        net_amount = getattr(raw, "net_amount", None)
        amount_field = getattr(raw, "amount", None)
        cash_amount = net_amount if net_amount is not None else amount_field
        try:
            amount = float(cash_amount) if cash_amount is not None else 0.0
        except (TypeError, ValueError):
            amount = 0.0

        # Alpaca uses transaction_time for cash events; activity_id includes the date
        when_attr = (
            getattr(raw, "transaction_time", None)
            or getattr(raw, "date", None)
            or getattr(raw, "submitted_at", None)
        )
        if isinstance(when_attr, datetime):
            occurred_at = when_attr if when_attr.tzinfo else when_attr.replace(tzinfo=timezone.utc)
        elif when_attr is not None:
            try:
                occurred_at = datetime.fromisoformat(str(when_attr).replace("Z", "+00:00"))
                if occurred_at.tzinfo is None:
                    occurred_at = occurred_at.replace(tzinfo=timezone.utc)
            except ValueError:
                return None
        else:
            return None

        symbol = getattr(raw, "symbol", None)

        return ActivityEvent(
            activity_type=canonical,
            symbol=symbol,
            amount=amount,
            occurred_at=occurred_at,
            raw=sanitize_activity_payload(
                getattr(raw, "model_dump", lambda: getattr(raw, "__dict__", {}))()
                if hasattr(raw, "model_dump")
                else getattr(raw, "__dict__", {})
            ),
        )

    # ---- orders -------------------------------------------------------------

    def submit_market_order(
        self,
        *,
        symbol: str,
        side: str,
        qty: float | None = None,
        notional: float | None = None,
    ) -> OrderResult:
        canonical_side = normalise_side(side)
        side_enum = OrderSide.BUY if canonical_side == "BUY" else OrderSide.SELL
        if qty is None and notional is None:
            raise ValueError("Either qty or notional is required")
        if qty is not None and notional is not None:
            raise ValueError("Pass qty OR notional, not both")
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            notional=notional,
            side=side_enum,
            time_in_force=TimeInForce.DAY,
        )
        order = self._client.submit_order(req)
        return self._to_order_result(order)

    def get_order(self, order_id: str) -> OrderResult:
        order = self._client.get_order_by_id(order_id)
        return self._to_order_result(order)

    # ---- internal -----------------------------------------------------------

    def _to_order_result(self, order: Any) -> OrderResult:
        status = _status_str(order)
        submitted = getattr(order, "submitted_at", None) or getattr(order, "created_at", None)
        if isinstance(submitted, datetime):
            submitted_at = submitted if submitted.tzinfo else submitted.replace(tzinfo=timezone.utc)
        elif submitted is not None:
            try:
                submitted_at = datetime.fromisoformat(str(submitted).replace("Z", "+00:00"))
                if submitted_at.tzinfo is None:
                    submitted_at = submitted_at.replace(tzinfo=timezone.utc)
            except ValueError:
                submitted_at = datetime.now(timezone.utc)
        else:
            submitted_at = datetime.now(timezone.utc)

        side_raw = str(getattr(order, "side", "")).split(".")[-1]
        try:
            side = normalise_side(side_raw)
        except ValueError:
            side = "BUY"

        return OrderResult(
            order_id=str(getattr(order, "id", "")),
            symbol=str(getattr(order, "symbol", "")),
            side=side,
            qty=float(order.qty) if getattr(order, "qty", None) else None,
            notional=float(order.notional) if getattr(order, "notional", None) else None,
            submitted_at=submitted_at,
            status=status,
            filled_qty=float(getattr(order, "filled_qty", 0.0) or 0.0),
            filled_avg_price=float(getattr(order, "filled_avg_price", 0.0) or 0.0),
            filled_notional=(
                float(getattr(order, "filled_qty", 0.0) or 0.0)
                * float(getattr(order, "filled_avg_price", 0.0) or 0.0)
            ),
        )
