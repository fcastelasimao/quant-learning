"""
tests/test_broker_protocol.py
==============================
Protocol-level tests for live/brokers/.

These tests:
  - Verify the Broker runtime-checkable Protocol
  - Verify make_broker raises for unknown brokers
  - Exercise the shared plan helpers (build_rebalance_plan, validate_order_guardrails,
    verify_post_trade_drift, is_cadence_due) using a lightweight FakeBroker

No real credentials or network calls are made.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from live.brokers.base import (
    AccountSnapshot,
    ActivityEvent,
    AssetMetadata,
    Broker,
    OrderResult,
    PositionSnapshot,
    is_terminal_status,
    normalise_activity_type,
    normalise_side,
    sanitize_activity_payload,
)
from live.brokers.factory import make_broker
from live.budget import BudgetSnapshot, reconcile_after_trades
from live.lots import initialise_lots, youngest_lot_date
from live.rebalance import (
    OrderExecutionError,
    RebalanceRow,
    build_rebalance_plan,
    execute_rebalance,
    has_prior_preview,
    is_cadence_due,
    plan_to_frame,
    save_cadence_state,
    validate_order_guardrails,
    verify_post_trade_drift,
)


# ===========================================================================
# FakeBroker — minimal Broker implementation for unit tests
# ===========================================================================

class FakeBroker:
    """An in-memory Broker that satisfies the Broker protocol without any
    network calls.  Tests can pre-load positions and account state."""

    name = "fake"
    trading_mode = "paper"
    account_label = "test"

    def __init__(
        self,
        *,
        equity: float = 100_000.0,
        cash: float = 1_000.0,
        positions: dict[str, PositionSnapshot] | None = None,
        order_statuses: dict[str, str] | None = None,
    ) -> None:
        self._equity = equity
        self._cash = cash
        self._positions: dict[str, PositionSnapshot] = positions or {}
        self._order_statuses: dict[str, str] = order_statuses or {}
        self._submitted_orders: list[OrderResult] = []
        self._order_counter = 0

    # ---- Account ----
    def get_account(self) -> AccountSnapshot:
        return AccountSnapshot(
            equity=self._equity,
            cash=self._cash,
            buying_power=self._cash,
        )

    def get_positions(self) -> dict[str, PositionSnapshot]:
        return dict(self._positions)

    def get_asset(self, symbol: str) -> AssetMetadata:
        return AssetMetadata(symbol=symbol, tradable=True, fractionable=True)

    def get_open_orders(self, symbols: list[str]) -> list[OrderResult]:
        return []

    # ---- Market ----
    def is_market_open(self) -> bool:
        return True

    def last_trading_day_of_month(self, ref: date) -> date:
        return ref.replace(day=28)

    def is_trading_day(self, when: date) -> bool:
        return when.weekday() < 5

    # ---- Activities ----
    def fetch_activities(
        self,
        *,
        types: list[str],
        since: datetime,
        symbols: list[str] | None = None,
    ) -> list[ActivityEvent]:
        return []

    # ---- Orders ----
    def submit_market_order(
        self,
        *,
        symbol: str,
        side: str,
        qty: float | None = None,
        notional: float | None = None,
    ) -> OrderResult:
        self._order_counter += 1
        oid = f"fake-order-{self._order_counter}"
        result = OrderResult(
            order_id=oid,
            symbol=symbol,
            side=side.upper(),
            qty=qty,
            notional=notional,
            submitted_at=datetime.now(timezone.utc),
            status="filled",
            filled_qty=qty or 1.0,
            filled_avg_price=100.0,
        )
        self._submitted_orders.append(result)
        return result

    def get_order(self, order_id: str) -> OrderResult:
        status = self._order_statuses.get(order_id, "filled")
        return OrderResult(
            order_id=order_id,
            symbol="FAKE",
            side="BUY",
            qty=1.0,
            notional=None,
            submitted_at=datetime.now(timezone.utc),
            status=status,
            filled_qty=1.0 if status == "filled" else 0.0,
            filled_avg_price=100.0 if status == "filled" else 0.0,
        )


# ===========================================================================
# Protocol conformance
# ===========================================================================

def test_fake_broker_satisfies_broker_protocol():
    """FakeBroker must be recognised as a Broker via isinstance."""
    assert isinstance(FakeBroker(), Broker)


def test_account_snapshot_fields():
    broker = FakeBroker(equity=50_000.0, cash=2_500.0)
    snap = broker.get_account()
    assert snap.equity == 50_000.0
    assert snap.cash == 2_500.0


def test_positions_round_trip():
    pos = PositionSnapshot(
        symbol="SPY", qty=10.0, qty_available=10.0,
        market_value=5_000.0, current_price=500.0,
    )
    broker = FakeBroker(positions={"SPY": pos})
    positions = broker.get_positions()
    assert "SPY" in positions
    assert positions["SPY"].qty == 10.0


def test_reconcile_budget_after_trades_refreshes_reserved_cash():
    snap = BudgetSnapshot(
        budget_cap=10_000.0,
        reserved_cash=2_000.0,
        positions_value=8_000.0,
        activity_cursor=datetime.now(timezone.utc),
        last_updated=datetime.now(timezone.utc),
    )
    positions = {
        "SPY": PositionSnapshot("SPY", qty=10, qty_available=10, market_value=6_100, current_price=610),
        "TLT": PositionSnapshot("TLT", qty=39, qty_available=39, market_value=3_800, current_price=100),
    }

    updated = reconcile_after_trades(
        snap,
        positions,
        {"SPY": 0.60, "TLT": 0.40},
        target_managed_capital=10_000.0,
    )

    assert updated.positions_value == 9_900
    assert updated.reserved_cash == 100


def test_initialise_lots_can_backdate_existing_positions_as_sellable(tmp_path):
    positions = {
        "SPY": PositionSnapshot("SPY", qty=10, qty_available=10, market_value=5_000, current_price=500),
    }
    ledger = initialise_lots(
        str(tmp_path / "lots.json"),
        positions,
        {"SPY": 1.0},
        assume_held_days=32,
    )

    assert (date.today() - youngest_lot_date(ledger, "SPY")).days >= 32


# ===========================================================================
# make_broker factory
# ===========================================================================

def test_make_broker_raises_for_unknown_broker():
    with pytest.raises(ValueError, match="Unknown broker"):
        make_broker(broker_name="fidelity", trading_mode="paper", account_label="default")


def test_make_broker_raises_for_tastytrade_without_credentials(monkeypatch):
    """make_broker("tastytrade") should raise SystemExit when SDK is missing
    or credentials are absent — never a bare ImportError / AttributeError."""
    monkeypatch.delenv("BROKER_TASTYTRADE_DEFAULT_USERNAME", raising=False)
    monkeypatch.delenv("BROKER_TASTYTRADE_DEFAULT_PASSWORD", raising=False)
    monkeypatch.delenv("TASTYTRADE_PROVIDER_SECRET", raising=False)
    monkeypatch.delenv("TASTYTRADE_REFRESH_TOKEN", raising=False)
    monkeypatch.delenv("BROKER_TASTYTRADE_DEFAULT_PROVIDER_SECRET", raising=False)
    monkeypatch.delenv("BROKER_TASTYTRADE_DEFAULT_REFRESH_TOKEN", raising=False)
    with pytest.raises((SystemExit, RuntimeError)):
        make_broker(broker_name="tastytrade", trading_mode="paper", account_label="default")


# ===========================================================================
# build_rebalance_plan
# ===========================================================================

_ALLOCATION = {"SPY": 0.60, "TLT": 0.40}
_ASSET_META = {
    "SPY": AssetMetadata(symbol="SPY", tradable=True, fractionable=True),
    "TLT": AssetMetadata(symbol="TLT", tradable=True, fractionable=True),
}

def _make_positions(spy_val: float, tlt_val: float) -> dict[str, PositionSnapshot]:
    return {
        "SPY": PositionSnapshot("SPY", qty=spy_val / 500, qty_available=spy_val / 500,
                                market_value=spy_val, current_price=500.0),
        "TLT": PositionSnapshot("TLT", qty=tlt_val / 100, qty_available=tlt_val / 100,
                                market_value=tlt_val, current_price=100.0),
    }


def test_build_rebalance_plan_hold_when_within_threshold():
    """No trades should be generated when positions are within drift threshold."""
    equity = 100_000.0
    positions = _make_positions(60_000, 40_000)   # exactly at target
    rows, warnings = build_rebalance_plan(
        equity=equity,
        positions=positions,
        allocation=_ALLOCATION,
        asset_meta=_ASSET_META,
        drift_threshold=0.05,
        min_trade_value=25.0,
        cash_buffer_pct=0.0,
        liquidate_other_positions=False,
    )
    actions = {r.symbol: r.action for r in rows}
    assert actions["SPY"] == "HOLD"
    assert actions["TLT"] == "HOLD"
    assert not warnings


def test_build_rebalance_plan_generates_buy_and_sell():
    """Drifted positions should produce BUY/SELL rows."""
    equity = 100_000.0
    positions = _make_positions(80_000, 20_000)   # SPY overweight, TLT underweight
    rows, _ = build_rebalance_plan(
        equity=equity,
        positions=positions,
        allocation=_ALLOCATION,
        asset_meta=_ASSET_META,
        drift_threshold=0.05,
        min_trade_value=25.0,
        cash_buffer_pct=0.0,
        liquidate_other_positions=False,
    )
    actions = {r.symbol: r.action for r in rows}
    assert actions["SPY"] == "SELL"
    assert actions["TLT"] == "BUY"


def test_build_rebalance_plan_zero_positions_all_buys():
    """Starting from scratch: every symbol should be a BUY."""
    rows, _ = build_rebalance_plan(
        equity=100_000.0,
        positions={},
        allocation=_ALLOCATION,
        asset_meta=_ASSET_META,
        drift_threshold=0.05,
        min_trade_value=25.0,
        cash_buffer_pct=0.0,
        liquidate_other_positions=False,
    )
    actions = {r.symbol: r.action for r in rows}
    assert actions["SPY"] == "BUY"
    assert actions["TLT"] == "BUY"


def test_build_rebalance_plan_non_fractionable_still_buys_by_notional():
    """Qty-only brokers still need BUY rows; conversion happens at execution."""
    rows, warnings = build_rebalance_plan(
        equity=10_000.0,
        positions={},
        allocation=_ALLOCATION,
        asset_meta={
            "SPY": AssetMetadata(symbol="SPY", tradable=True, fractionable=False),
            "TLT": AssetMetadata(symbol="TLT", tradable=True, fractionable=False),
        },
        drift_threshold=0.05,
        min_trade_value=25.0,
        cash_buffer_pct=0.0,
        liquidate_other_positions=False,
    )
    by_symbol = {r.symbol: r for r in rows}
    assert by_symbol["SPY"].action == "BUY"
    assert by_symbol["SPY"].notional == 6_000.0
    assert by_symbol["TLT"].action == "BUY"
    assert any("whole-share" in w for w in warnings)


def test_build_rebalance_plan_non_strategy_warning():
    """Extra positions not in allocation produce a warning."""
    positions = _make_positions(60_000, 40_000)
    positions["QQQ"] = PositionSnapshot("QQQ", qty=1, qty_available=1,
                                        market_value=500, current_price=500)
    _, warnings = build_rebalance_plan(
        equity=100_500.0,
        positions=positions,
        allocation=_ALLOCATION,
        asset_meta=_ASSET_META,
        drift_threshold=0.05,
        min_trade_value=25.0,
        cash_buffer_pct=0.0,
        liquidate_other_positions=False,
    )
    assert any("QQQ" in w for w in warnings)


# ===========================================================================
# validate_order_guardrails
# ===========================================================================

def test_guardrails_pass_normal_plan():
    rows = [
        RebalanceRow("SPY", 0.6, 0.0, 60_000, 0, 60_000, "BUY", notional=60_000),
    ]
    validate_order_guardrails(rows, equity=100_000, max_order_pct_equity=0.80,
                              max_total_trade_pct_equity=1.05)


def test_guardrails_block_oversized_single_order():
    rows = [
        RebalanceRow("SPY", 0.5, 0.0, 50_000, 0, 50_000, "BUY", notional=50_000),
    ]
    with pytest.raises(SystemExit, match="exceed"):
        validate_order_guardrails(rows, equity=100_000, max_order_pct_equity=0.40,
                                  max_total_trade_pct_equity=1.05)


def test_execute_rebalance_rebuilds_buys_against_managed_capital(monkeypatch):
    """Budgeted accounts must not resize buys from full broker account equity."""
    broker = FakeBroker(equity=100_000.0, cash=100_000.0, positions={})
    monkeypatch.setattr("live.rebalance.fetch_latest_prices", lambda symbols, logger: {})

    filled = execute_rebalance(
        broker=broker,
        initial_rows=[],
        allocation=_ALLOCATION,
        asset_meta=_ASSET_META,
        drift_threshold=0.05,
        min_trade_value=25.0,
        cash_buffer_pct=0.0,
        liquidate_other_positions=False,
        timeout_seconds=1,
        logger=_Logger(),
        sizing_equity=10_000.0,
    )

    notionals = {order.symbol: order.notional for order in broker._submitted_orders}
    assert notionals == {"SPY": 6_000.0, "TLT": 4_000.0}
    assert {order.order_id for order in filled} == {"fake-order-1", "fake-order-2"}


# ===========================================================================
# verify_post_trade_drift
# ===========================================================================

class _Logger:
    def info(self, *a, **kw): pass
    def warning(self, *a, **kw): pass
    def error(self, *a, **kw): pass


def test_post_trade_drift_passes_when_within_tolerance():
    positions = _make_positions(60_000, 40_000)
    result = verify_post_trade_drift(
        equity=100_000,
        positions=positions,
        allocation=_ALLOCATION,
        tolerance=0.02,
        min_extra_position_value=25,
        logger=_Logger(),
    )
    assert list(result["Symbol"]) == ["SPY", "TLT"]


def test_post_trade_drift_raises_on_excess_drift():
    positions = _make_positions(80_000, 20_000)   # large drift
    with pytest.raises(OrderExecutionError, match="SPY"):
        verify_post_trade_drift(
            equity=100_000,
            positions=positions,
            allocation=_ALLOCATION,
            tolerance=0.02,
            min_extra_position_value=25,
            logger=_Logger(),
        )


# ===========================================================================
# Cadence helpers
# ===========================================================================

def test_cadence_due_when_no_state_file(tmp_path):
    state_path = str(tmp_path / "cadence.json")
    due, reason = is_cadence_due(state_path, interval_days=31)
    assert due is True
    assert "No previous run" in reason


def test_cadence_not_due_when_run_was_recent(tmp_path):
    state_path = str(tmp_path / "cadence.json")
    recent = date.today() - timedelta(days=5)
    save_cadence_state(state_path, recent)
    due, reason = is_cadence_due(state_path, interval_days=31)
    assert due is False
    assert "Too early" in reason


def test_cadence_due_after_interval(tmp_path):
    state_path = str(tmp_path / "cadence.json")
    old_date = date.today() - timedelta(days=35)
    save_cadence_state(state_path, old_date)
    due, reason = is_cadence_due(state_path, interval_days=31)
    assert due is True


def test_has_prior_preview_matches_broker_mode_account_and_strategy(tmp_path):
    logs_dir = tmp_path
    payload = {
        "broker": "alpaca",
        "trading_mode": "live",
        "account_label": "main",
        "strategy_id": "6_asset_rp_baseline",
        "outcome": "preview",
    }
    (logs_dir / "run_summary.jsonl").write_text(json.dumps(payload) + "\n")

    assert has_prior_preview(str(logs_dir), "alpaca", "live", "main", "6asset_tip_gsg_rpavg")
    assert not has_prior_preview(str(logs_dir), "alpaca", "paper", "main", "6asset_tip_gsg_rpavg")


# ===========================================================================
# Base helpers
# ===========================================================================

def test_normalise_side_buy():
    assert normalise_side("buy") == "BUY"
    assert normalise_side("B") == "BUY"


def test_normalise_side_sell():
    assert normalise_side("SELL") == "SELL"
    assert normalise_side("s") == "SELL"


def test_normalise_side_invalid():
    with pytest.raises(ValueError):
        normalise_side("SHORT")


def test_normalise_activity_type():
    mapping = {"DIV": "DIV", "FILL": "FILL"}
    assert normalise_activity_type("DIV", mapping=mapping) == "DIV"
    assert normalise_activity_type("UNKNOWN", mapping=mapping) == "OTHER"


def test_is_terminal_status():
    assert is_terminal_status("filled")
    assert is_terminal_status("canceled")
    assert not is_terminal_status("open")
    assert not is_terminal_status("pending")


def test_sanitize_activity_payload_strips_secrets():
    payload = {"amount": 100.0, "token": "abc123", "description": "dividend"}
    result = sanitize_activity_payload(payload)
    assert "token" not in result
    assert result["amount"] == 100.0
    assert result["description"] == "dividend"


def test_plan_to_frame_includes_all_symbols():
    rows = [
        RebalanceRow("SPY", 0.6, 0.5, 60_000, 50_000, 10_000, "BUY", notional=10_000),
        RebalanceRow("TLT", 0.4, 0.5, 40_000, 50_000, -10_000, "SELL", qty=100.0),
    ]
    frame = plan_to_frame(rows)
    assert set(frame["Symbol"]) == {"SPY", "TLT"}
    assert "Action" in frame.columns
