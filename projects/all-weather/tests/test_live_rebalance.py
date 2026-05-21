from types import SimpleNamespace

import pytest

from live.alpaca_rebalance import (
    OrderExecutionError,
    PositionSnapshot,
    RebalanceRow,
    _assert_strategy_live_allowed,
    _load_strategy_payload,
    _performance_headers,
    _resolve_target_allocation,
    parse_args,
    record_performance_snapshot,
    validate_order_guardrails,
    verify_post_trade_drift,
    wait_for_orders,
)


class _Logger:
    def info(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass

    def error(self, *_args, **_kwargs):
        pass


class _OrderClient:
    def __init__(self, statuses):
        self.statuses = statuses

    def get_order_by_id(self, order_id):
        return SimpleNamespace(status=self.statuses[order_id])


def test_live_strategy_loader_reads_project_root_strategy_file():
    payload = _load_strategy_payload("6_asset_rp_baseline")

    assert payload["_strategy_id"] == "6asset_tip_gsg_rpavg"
    assert payload["display_name"] == "6 Asset RP Baseline"
    assert payload["allocation"]["SPY"] == 0.134
    assert payload["live_tickers"]["GLD"] == "GLDM"


def test_live_target_allocation_uses_live_tickers_by_default(monkeypatch):
    monkeypatch.setattr("sys.argv", ["alpaca_rebalance.py", "--paper"])
    args = parse_args()

    allocation, mapping = _resolve_target_allocation(args.strategy_id, args.use_live_tickers)

    assert args.use_live_tickers is True
    assert "GLDM" in allocation
    assert "GLD" not in allocation
    assert mapping["GLD"] == "GLDM"


def test_live_target_allocation_can_opt_out_to_backtest_tickers(monkeypatch):
    monkeypatch.setattr("sys.argv", ["alpaca_rebalance.py", "--paper", "--use-backtest-tickers"])
    args = parse_args()

    allocation, mapping = _resolve_target_allocation(args.strategy_id, args.use_live_tickers)

    assert args.use_live_tickers is False
    assert "GLD" in allocation
    assert "GLDM" not in allocation
    assert mapping["GLD"] == "GLD"


def test_live_strategy_guard_blocks_non_production_by_default():
    payload = _load_strategy_payload("6asset_tip_gsg")

    with pytest.raises(SystemExit, match="non-production"):
        _assert_strategy_live_allowed(
            strategy_id="6asset_tip_gsg",
            payload=payload,
            allow_non_production=False,
        )


def test_wait_for_orders_raises_on_rejected_order():
    client = _OrderClient({"order-1": "rejected"})

    with pytest.raises(OrderExecutionError, match="rejected"):
        wait_for_orders(client, ["order-1"], timeout_seconds=1, logger=_Logger())


def test_wait_for_orders_raises_on_timeout(monkeypatch):
    client = _OrderClient({"order-1": "accepted"})
    monkeypatch.setattr("live.alpaca_rebalance.time.sleep", lambda _seconds: None)

    with pytest.raises(OrderExecutionError, match="timeout"):
        wait_for_orders(client, ["order-1"], timeout_seconds=0, logger=_Logger())


def test_post_trade_verification_rejects_excess_drift():
    account = SimpleNamespace(equity="100000")
    positions = {
        "SPY": PositionSnapshot("SPY", qty=10, qty_available=10, market_value=50000, current_price=500),
        "TLT": PositionSnapshot("TLT", qty=10, qty_available=10, market_value=50000, current_price=100),
    }
    allocation = {"SPY": 0.60, "TLT": 0.40}

    with pytest.raises(OrderExecutionError, match="SPY"):
        verify_post_trade_drift(
            account,
            positions,
            allocation,
            tolerance=0.015,
            min_extra_position_value=25,
            logger=_Logger(),
        )


def test_order_guardrails_block_oversized_trade():
    rows = [
        RebalanceRow(
            symbol="SPY",
            target_weight=0.5,
            current_weight=0.0,
            target_value=50000,
            current_value=0,
            delta_value=50000,
            action="BUY",
            notional=50000,
        )
    ]

    with pytest.raises(SystemExit, match="exceed"):
        validate_order_guardrails(
            rows,
            equity=100000,
            max_order_pct_equity=0.40,
            max_total_trade_pct_equity=1.05,
        )


def test_performance_tracking_headers_follow_live_symbols(tmp_path, monkeypatch):
    allocation = {"SPY": 0.134, "GLDM": 0.142}
    account = SimpleNamespace(equity="100000")
    positions = {
        "SPY": PositionSnapshot("SPY", qty=10, qty_available=10, market_value=13400, current_price=1340),
        "GLDM": PositionSnapshot("GLDM", qty=10, qty_available=10, market_value=14200, current_price=1420),
    }
    csv_path = tmp_path / "tracking.csv"
    monkeypatch.setattr(
        "live.alpaca_rebalance.calculate_benchmark_returns",
        lambda _logger: {"SPY_Return%": 0.0, "ALLW_Return%": 0.0, "60_40_Return%": 0.0},
    )

    record_performance_snapshot(account, positions, allocation, _Logger(), str(csv_path))

    header = csv_path.read_text().splitlines()[0].split(",")
    assert header == _performance_headers(allocation)
    assert "GLDM_Weight%" in header
    assert "GLD_Weight%" not in header
