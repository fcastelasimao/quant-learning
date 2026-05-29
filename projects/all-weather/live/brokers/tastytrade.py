"""
live/brokers/tastytrade.py
==========================
Tastytrade implementation of the Broker protocol.

Backed by the community SDK published on PyPI as `tastytrade`
(https://github.com/tastyware/tastytrade). Install with:

    conda run -n allweather pip install tastytrade

This module is intentionally tolerant of SDK API drift: every Tastytrade call
is wrapped so a minor SDK rename does not crash the whole rebalancer. Verify
the SDK version pinned in requirements.txt against the README before any live
execution.

Authentication
--------------
The pinned `tastytrade==12.4.1` SDK uses OAuth credentials. Credentials are
loaded from the shared QuantFinance api_keys.env file and resolved from env
vars based on account_label "MY_LABEL":

    BROKER_TASTYTRADE_<MY_LABEL>_PROVIDER_SECRET
    BROKER_TASTYTRADE_<MY_LABEL>_REFRESH_TOKEN
    BROKER_TASTYTRADE_<MY_LABEL>_ACCOUNT_NUMBER   (optional; required when the
                                                    user has multiple TT accounts)
"""

from __future__ import annotations

import json
import os
import stat
import sys
from decimal import Decimal
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
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

_TASTYTRADE_INSTALLED = True
try:
    import tastytrade  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    _TASTYTRADE_INSTALLED = False
    tastytrade = None  # type: ignore[assignment]


_SESSION_DIR = Path.home() / ".allweather"


# Mapping from Tastytrade transaction types/sub-types to our canonical enum.
# Tastytrade's transaction "transaction_type" is broad ("Money Movement",
# "Trade", ...). We mostly rely on transaction_sub_type for finer events.
_TT_ACTIVITY_MAP = {
    "DIVIDEND": "DIV",
    "DISTRIBUTION": "DIV",
    "DEPOSIT": "DEPOSIT",
    "ACH DEPOSIT": "DEPOSIT",
    "WIRE DEPOSIT": "DEPOSIT",
    "WITHDRAWAL": "WITHDRAWAL",
    "ACH WITHDRAWAL": "WITHDRAWAL",
    "WIRE WITHDRAWAL": "WITHDRAWAL",
    "TRADE": "FILL",
    "FILL": "FILL",
}


def _require_sdk() -> None:
    if not _TASTYTRADE_INSTALLED:
        raise SystemExit(
            "tastytrade SDK is not installed. Install it with:\n\n"
            "    conda run -n allweather pip install tastytrade\n\n"
            "See https://github.com/tastyware/tastytrade for the latest version."
        )


def _resolve_credentials(account_label: str) -> tuple[str, str, str | None]:
    """Return (provider_secret, refresh_token, account_number)."""
    upper = account_label.upper()
    credential_pairs = [
        (
            f"BROKER_TASTYTRADE_{upper}_PROVIDER_SECRET",
            f"BROKER_TASTYTRADE_{upper}_REFRESH_TOKEN",
            f"BROKER_TASTYTRADE_{upper}_ACCOUNT_NUMBER",
        )
    ]
    if account_label.lower() == "default":
        credential_pairs.append(
            ("TASTYTRADE_PROVIDER_SECRET", "TASTYTRADE_REFRESH_TOKEN", "TASTYTRADE_ACCOUNT_NUMBER")
        )
    provider_secret = None
    refresh_token = None
    account_number = None
    provider_var = credential_pairs[0][0]
    token_var = credential_pairs[0][1]
    for candidate_provider, candidate_token, candidate_account in credential_pairs:
        provider_secret = os.getenv(candidate_provider)
        refresh_token = os.getenv(candidate_token)
        account_number = os.getenv(candidate_account)
        if provider_secret and refresh_token:
            provider_var = candidate_provider
            token_var = candidate_token
            break
    legacy_username = os.getenv(f"BROKER_TASTYTRADE_{upper}_USERNAME")
    legacy_password = os.getenv(f"BROKER_TASTYTRADE_{upper}_PASSWORD")
    if legacy_username or legacy_password:
        raise SystemExit(
            "tastytrade==12.4.1 uses OAuth credentials, not username/password. "
            f"Set {provider_var} and {token_var}."
        )
    if not provider_secret or not refresh_token:
        tried = ", ".join(f"{provider}/{token}" for provider, token, _ in credential_pairs)
        raise SystemExit(
            f"Missing Tastytrade credentials for account '{account_label}'. "
            f"Set one of these env-var pairs in api_keys.env: {tried}."
        )
    return provider_secret, refresh_token, account_number


def _session_cache_path(account_label: str) -> Path:
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in account_label)
    return _SESSION_DIR / f"tt_session_{safe}.json"


def _load_cached_session_token(account_label: str) -> str | None:
    path = _session_cache_path(account_label)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    expiry = data.get("expires_at")
    if expiry:
        try:
            when = datetime.fromisoformat(expiry)
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            if when <= datetime.now(timezone.utc):
                return None
        except ValueError:
            return None
    token = data.get("session_token")
    return token if isinstance(token, str) and token else None


def _persist_session_token(account_label: str, token: str, *, ttl_hours: int = 23) -> None:
    _SESSION_DIR.mkdir(parents=True, exist_ok=True)
    path = _session_cache_path(account_label)
    payload = {
        "session_token": token,
        "expires_at": (datetime.now(timezone.utc) + timedelta(hours=ttl_hours)).isoformat(),
    }
    path.write_text(json.dumps(payload))
    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except PermissionError:
        pass


def _to_iso_utc(when: datetime) -> str:
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when.astimezone(timezone.utc).isoformat()


def _parse_when(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time(), tzinfo=timezone.utc)
    try:
        out = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return out if out.tzinfo else out.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


# ===========================================================================
# Broker implementation
# ===========================================================================

class TastytradeBroker:
    """Concrete Broker for Tastytrade paper or live accounts.

    Tastytrade exposes paper trading through "sandbox" accounts. We map our
    trading_mode == "paper" to the sandbox base URL, and "live" to the
    production endpoint.
    """

    name = "tastytrade"

    def __init__(self, *, trading_mode: str, account_label: str) -> None:
        _require_sdk()
        if trading_mode not in {"paper", "live"}:
            raise ValueError(f"trading_mode must be 'paper' or 'live', got {trading_mode!r}")
        self.trading_mode = trading_mode
        self.account_label = account_label

        provider_secret, refresh_token, account_number_env = _resolve_credentials(account_label)
        self._session = self._open_session(provider_secret, refresh_token)
        self._account = self._select_account(account_number_env)

        # Cache the live account number string for logging
        self.account_number = getattr(self._account, "account_number", None) or account_number_env

    # ---- session management -------------------------------------------------

    @staticmethod
    def _await(fn, *args, **kwargs):  # noqa: ANN001
        import anyio

        async def _runner():
            return await fn(*args, **kwargs)

        return anyio.run(_runner)

    def _open_session(self, provider_secret: str, refresh_token: str) -> Any:
        is_test = self.trading_mode == "paper"
        Session = getattr(tastytrade, "Session", None)
        if Session is None:
            raise SystemExit(
                "tastytrade SDK does not expose a Session class. "
                "Upgrade the SDK: conda run -n allweather pip install -U tastytrade"
            )
        return Session(provider_secret, refresh_token, is_test=is_test)

    def _select_account(self, account_number_env: str | None) -> Any:
        Account = getattr(tastytrade, "Account", None)
        if Account is None:
            raise SystemExit(
                "tastytrade SDK does not expose an Account class. "
                "Upgrade the SDK: conda run -n allweather pip install -U tastytrade"
            )
        if account_number_env:
            return self._await(Account.get, self._session, account_number_env)
        accounts = self._await(Account.get, self._session)
        if not accounts:
            raise SystemExit("Tastytrade returned no accounts for this user.")
        if len(accounts) > 1:
            numbers = [getattr(a, "account_number", "?") for a in accounts]
            raise SystemExit(
                f"User has multiple Tastytrade accounts ({numbers}). "
                f"Set BROKER_TASTYTRADE_{self.account_label.upper()}_ACCOUNT_NUMBER to disambiguate."
            )
        return accounts[0]

    # ---- account state ------------------------------------------------------

    def get_account(self) -> AccountSnapshot:
        balances = self._await(self._account.get_balances, self._session)
        equity = float(
            getattr(balances, "equity_buying_power", 0.0)
            or getattr(balances, "net_liquidating_value", 0.0)
        )
        cash = float(getattr(balances, "cash_balance", 0.0) or 0.0)
        buying_power = float(
            getattr(balances, "derivative_buying_power", 0.0)
            or getattr(balances, "equity_buying_power", 0.0)
            or 0.0
        )
        return AccountSnapshot(equity=equity, cash=cash, buying_power=buying_power)

    def get_positions(self) -> dict[str, PositionSnapshot]:
        positions = self._await(self._account.get_positions, self._session, include_marks=True)
        out: dict[str, PositionSnapshot] = {}
        for position in positions:
            symbol = str(getattr(position, "symbol", ""))
            if not symbol:
                continue
            qty = float(getattr(position, "quantity", 0.0) or 0.0)
            avg_price = float(
                getattr(position, "average_open_price", 0.0)
                or getattr(position, "cost_effect", 0.0)
                or 0.0
            )
            market_value = float(
                getattr(position, "mark_price", None)
                or getattr(position, "mark", None)
                or getattr(position, "close_price", 0.0)
                or 0.0
            ) * abs(qty)
            current_price = market_value / abs(qty) if qty else 0.0
            acquired = _parse_when(
                getattr(position, "created_at", None)
                or getattr(position, "open_date", None)
            )
            oldest = acquired.date() if acquired else None
            out[symbol] = PositionSnapshot(
                symbol=symbol,
                qty=qty,
                qty_available=qty,  # Tastytrade does not split hold/available the same way
                market_value=abs(market_value),
                current_price=current_price,
                avg_entry_price=avg_price,
                oldest_acquisition_date=oldest,
            )
        return out

    def get_asset(self, symbol: str) -> AssetMetadata:
        # Tastytrade exposes instruments via the metadata endpoint. Fractionable
        # ETF support is limited to a curated list; we default to False.
        fractionable = False
        tradable = True
        try:
            instrument_lookup = getattr(tastytrade, "instruments", None)
            if instrument_lookup is not None:
                # Newer SDK shape: tastytrade.instruments.Equity.get_equity(session, symbol)
                Equity = getattr(instrument_lookup, "Equity", None)
                if Equity is not None and hasattr(Equity, "get_equity"):
                    eq = Equity.get_equity(self._session, symbol)
                    tradable = bool(getattr(eq, "is_active", True))
                    fractionable = bool(
                        getattr(eq, "is_fractional_quantity_eligible", False)
                    )
                elif hasattr(Equity, "get"):
                    eqs = self._await(Equity.get, self._session, [symbol])
                    eq = eqs[0] if isinstance(eqs, list) else eqs
                    tradable = bool(getattr(eq, "is_active", True))
                    fractionable = bool(
                        getattr(eq, "is_fractional_quantity_eligible", False)
                    )
        except Exception:
            pass
        return AssetMetadata(symbol=symbol, tradable=tradable, fractionable=fractionable)

    def get_open_orders(self, symbols: list[str]) -> list[OrderResult]:
        wanted = set(symbols)
        try:
            orders = self._await(self._account.get_live_orders, self._session)
        except AttributeError:
            try:
                orders = self._await(self._account.get_order_history, self._session)
            except Exception:
                return []
        results: list[OrderResult] = []
        for order in orders:
            symbol = self._extract_order_symbol(order)
            if symbol not in wanted:
                continue
            results.append(self._to_order_result(order))
        return results

    # ---- market state -------------------------------------------------------

    def is_market_open(self) -> bool:
        # Tastytrade SDK does not always expose a market-clock endpoint; fall
        # back to NYSE hours via a simple wall-clock check.
        from zoneinfo import ZoneInfo

        now_et = datetime.now(ZoneInfo("America/New_York"))
        if now_et.weekday() >= 5:
            return False
        open_t = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        close_t = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        return open_t <= now_et <= close_t

    def last_trading_day_of_month(self, ref: date) -> date:
        # Approximation via NYSE-like rules. For Tastytrade we conservatively
        # fall back to "last weekday of month". The 31-day cadence path
        # (live/cadence.py) is the primary rebalance gate, not month-end.
        from calendar import monthrange

        last_day = ref.replace(day=monthrange(ref.year, ref.month)[1])
        while last_day.weekday() >= 5:
            last_day -= timedelta(days=1)
        return last_day

    def is_trading_day(self, when: date) -> bool:
        return when.weekday() < 5

    # ---- activities ---------------------------------------------------------

    def fetch_activities(
        self,
        *,
        types: list[str],
        since: datetime,
        symbols: list[str] | None = None,
    ) -> list[ActivityEvent]:
        since_d = since.date() if isinstance(since, datetime) else since
        try:
            transactions = self._await(
                self._account.get_history,
                self._session,
                start_date=since_d,
            )
        except AttributeError:
            try:
                transactions = self._await(
                    self._account.get_history,
                    self._session,
                    start_date=since_d,
                )
            except Exception:
                return []
        except Exception:
            return []

        wanted = set(types)
        out: list[ActivityEvent] = []
        for txn in transactions:
            sub_type = str(
                getattr(txn, "transaction_sub_type", "")
                or getattr(txn, "sub_type", "")
                or getattr(txn, "transaction_type", "")
            ).upper()
            canonical = normalise_activity_type(sub_type, mapping=_TT_ACTIVITY_MAP)
            if canonical not in wanted:
                continue
            symbol = getattr(txn, "symbol", None) or getattr(txn, "underlying_symbol", None)
            if canonical == "DIV" and symbols and symbol not in set(symbols):
                continue
            net = (
                getattr(txn, "net_value", None)
                or getattr(txn, "value", None)
                or getattr(txn, "amount", None)
                or 0.0
            )
            try:
                amount = float(net)
            except (TypeError, ValueError):
                amount = 0.0
            when = _parse_when(
                getattr(txn, "transaction_date", None)
                or getattr(txn, "executed_at", None)
                or getattr(txn, "created_at", None)
            )
            if when is None:
                continue
            out.append(
                ActivityEvent(
                    activity_type=canonical,
                    symbol=symbol,
                    amount=amount,
                    occurred_at=when,
                    raw=sanitize_activity_payload(
                        txn.dict() if hasattr(txn, "dict") else getattr(txn, "__dict__", {})
                    ),
                )
            )
        out.sort(key=lambda e: e.occurred_at)
        return out

    # ---- orders -------------------------------------------------------------

    def submit_market_order(
        self,
        *,
        symbol: str,
        side: str,
        qty: float | None = None,
        notional: float | None = None,
    ) -> OrderResult:
        if notional is not None and qty is None:
            raise NotImplementedError(
                "Tastytrade does not support notional market orders. "
                "Pass qty instead."
            )
        if qty is None:
            raise ValueError("qty is required")

        canonical_side = normalise_side(side)
        from tastytrade.order import InstrumentType

        order_module = getattr(tastytrade, "order", None) or tastytrade
        OrderType = getattr(order_module, "OrderType", None)
        OrderAction = getattr(order_module, "OrderAction", None)
        OrderTimeInForce = (
            getattr(order_module, "OrderTimeInForce", None)
            or getattr(order_module, "TimeInForce", None)
        )
        NewOrder = (
            getattr(order_module, "NewOrder", None)
            or getattr(order_module, "Order", None)
        )
        if not all((OrderType, OrderAction, OrderTimeInForce, NewOrder)):
            raise SystemExit(
                "tastytrade SDK does not expose the expected order classes. "
                "Upgrade the SDK: conda run -n allweather pip install -U tastytrade"
            )

        action_name = "BUY_TO_OPEN" if canonical_side == "BUY" else "SELL_TO_CLOSE"
        action = getattr(OrderAction, action_name)

        Leg = (
            getattr(order_module, "Leg", None)
            or getattr(order_module, "OrderLeg", None)
        )
        if Leg is None:
            raise SystemExit("tastytrade SDK does not expose a Leg / OrderLeg class.")

        leg = Leg(
            instrument_type=InstrumentType.EQUITY,
            symbol=symbol,
            quantity=Decimal(str(qty)),
            action=action,
        )
        new_order = NewOrder(
            time_in_force=getattr(OrderTimeInForce, "DAY"),
            order_type=getattr(OrderType, "MARKET", getattr(OrderType, "Market", None)),
            legs=[leg],
        )
        placed = self._await(self._account.place_order, self._session, new_order, dry_run=False)
        order = getattr(placed, "order", placed)
        return self._to_order_result(order)

    def get_order(self, order_id: str) -> OrderResult:
        try:
            order = self._await(self._account.get_order, self._session, int(order_id))
        except AttributeError:
            try:
                order = self._await(self._account.get_order, self._session, int(order_id))
            except Exception as exc:
                raise SystemExit(f"Tastytrade get_order failed: {exc}")
        return self._to_order_result(order)

    # ---- internal -----------------------------------------------------------

    def _extract_order_symbol(self, order: Any) -> str:
        legs = getattr(order, "legs", None)
        if legs:
            first = legs[0]
            return str(getattr(first, "symbol", "") or "")
        return str(getattr(order, "symbol", "") or "")

    def _to_order_result(self, order: Any) -> OrderResult:
        status_raw = str(
            getattr(order, "status", "") or getattr(order, "order_status", "")
        ).split(".")[-1]
        status = self._map_status(status_raw)

        symbol = self._extract_order_symbol(order)
        leg = (order.legs[0] if getattr(order, "legs", None) else None)
        qty = float(getattr(leg, "quantity", 0.0)) if leg else 0.0

        action_raw = str(getattr(leg, "action", "") if leg else "").split(".")[-1]
        side = "BUY" if "BUY" in action_raw.upper() else "SELL"

        when = _parse_when(
            getattr(order, "received_at", None)
            or getattr(order, "created_at", None)
            or getattr(order, "submitted_at", None)
        )
        submitted_at = when or datetime.now(timezone.utc)

        return OrderResult(
            order_id=str(getattr(order, "id", "") or getattr(order, "order_id", "")),
            symbol=symbol,
            side=side,
            qty=qty or None,
            notional=None,
            submitted_at=submitted_at,
            status=status,
            filled_qty=float(getattr(order, "filled_quantity", 0.0) or 0.0),
            filled_avg_price=float(getattr(order, "average_fill_price", 0.0) or 0.0),
            filled_notional=(
                float(getattr(order, "filled_quantity", 0.0) or 0.0)
                * float(getattr(order, "average_fill_price", 0.0) or 0.0)
            ),
        )

    @staticmethod
    def _map_status(status: str) -> str:
        s = status.lower()
        if s in {"filled", "complete", "completed"}:
            return "filled"
        if s in {"cancelled", "canceled"}:
            return "canceled"
        if s in {"rejected", "expired"}:
            return s
        if s in {"received", "routed", "in_flight", "live", "working", "open", "pending"}:
            return "open"
        return s


# ===========================================================================
# CLI helper: `python -m live.brokers.tastytrade login --account my_label`
# ===========================================================================

def _cli_login() -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="live.brokers.tastytrade login")
    parser.add_argument("login", nargs="?")
    parser.add_argument("--account", default="default")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Validate live OAuth credentials instead of paper.",
    )
    args = parser.parse_args(sys.argv[1:])
    mode = "live" if args.live else "paper"
    broker = TastytradeBroker(trading_mode=mode, account_label=args.account)
    print(f"Tastytrade {mode} session opened for account '{args.account}'.")
    print("OAuth credentials validated.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_cli_login())
