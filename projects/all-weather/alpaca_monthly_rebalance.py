"""
alpaca_monthly_rebalance.py
===========================
Monthly ETF rebalancer for an Alpaca paper-trading account.

What it does
------------
- Loads a target ETF allocation from strategies.json
- Connects to Alpaca paper trading via alpaca-py
- Checks whether today is the last US trading day of the month
- Reads account equity, cash, and open positions
- Builds a rebalance plan from current weights to target weights
- Optionally submits market orders during the regular session

Safety defaults
---------------
- Preview only unless --execute is passed
- Refuses to trade outside the last trading day of the month unless --force
- Refuses to trade when the regular market is closed
- Executes sells first, then refreshes the account and computes buys again
- Leaves non-strategy positions alone unless --liquidate-other-positions is passed

Environment
-----------
Set Alpaca paper keys in your shell before running:

    export APCA_API_KEY_ID="..."
    export APCA_API_SECRET_KEY="..."

For multiple accounts, add a suffix (e.g. --account live):

    export APCA_API_KEY_ID_LIVE="..."
    export APCA_API_SECRET_KEY_LIVE="..."

Examples
--------
Preview only (default account, backtest tickers):
    conda run -n allweather python alpaca_monthly_rebalance.py

Preview with live tickers on the "live" account:
    conda run -n allweather python alpaca_monthly_rebalance.py --account live --use-live-tickers

Execute on the last trading day:
    conda run -n allweather python alpaca_monthly_rebalance.py --execute
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

import config

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import GetCalendarRequest, MarketOrderRequest
except ImportError as exc:  # pragma: no cover - import guard for environments without alpaca-py
    raise SystemExit(
        "alpaca-py is not installed. Install it with:\n\n"
        "    pip install alpaca-py\n"
    ) from exc


NY_TZ = ZoneInfo("America/New_York")
DEFAULT_TIMEOUT_SECONDS = 60
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(_SCRIPT_DIR, "logs", "performance_tracking.csv")
HOLDINGS_PATH = os.path.join(_SCRIPT_DIR, "portfolio_holdings.json")
CSV_HEADERS = [
    "Date", "Portfolio_Equity", "SPY_Weight%", "QQQ_Weight%", "TLT_Weight%",
    "TIP_Weight%", "GLD_Weight%", "GSG_Weight%", "Portfolio_Return%",
    "SPY_Return%", "ALLW_Return%", "60_40_Return%",
]


def setup_logging() -> logging.Logger:
    """
    Configure logging to both console and file.
    
    Creates logs/ directory if it doesn't exist.
    Returns a logger instance that writes to both stdout and a timestamped log file.
    """
    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create timestamped log file
    now_et = datetime.now(NY_TZ)
    log_filename = now_et.strftime("%Y-%m-%d_%H-%M-%S_alpaca_rebalance.log")
    log_path = os.path.join(logs_dir, log_filename)
    
    # Set up logger
    logger = logging.getLogger("alpaca_rebalancer")
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # File handler (DEBUG level - capture everything)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler (INFO level - user-friendly output)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging session started. Full log: {log_path}")
    return logger


# ===========================================================================
# PERFORMANCE TRACKING
# ===========================================================================
def calculate_allocation_actual(
    positions: dict[str, PositionSnapshot],
    allocation: dict[str, float],
    equity: float,
) -> dict[str, float]:
    """Calculate actual allocation % for each asset in target allocation."""
    return {
        symbol: round((positions[symbol].market_value / equity * 100) if symbol in positions and equity > 0 else 0.0, 1)
        for symbol in allocation
    }


def _get_price_pct_change(current: float, previous: float) -> float:
    """Helper to calculate percentage change between two prices."""
    return round(((current - previous) / previous * 100), 2) if previous > 0 else 0.0


def calculate_benchmark_returns(logger: logging.Logger) -> dict[str, float]:
    """Calculate monthly returns for SPY, ALLW, 60/40 by fetching last month's prices."""
    try:
        tickers = yf.download(["SPY", "ALLW", "TLT"], period="2mo", progress=False)
        
        # Get last traded day (today) and previous month's last day
        today_close = {t: tickers[t].iloc[-1] for t in ["SPY", "ALLW", "TLT"]}
        prev_close = {t: tickers[t].iloc[-22] for t in ["SPY", "ALLW", "TLT"]}  # ~1 month back
        
        spy_ret = _get_price_pct_change(today_close["SPY"], prev_close["SPY"])
        allw_ret = _get_price_pct_change(today_close["ALLW"], prev_close["ALLW"])
        tlt_ret = _get_price_pct_change(today_close["TLT"], prev_close["TLT"])
        
        return {
            "SPY_Return%": spy_ret,
            "ALLW_Return%": allw_ret,
            "60_40_Return%": round((spy_ret * 0.60 + tlt_ret * 0.40), 2),
        }
    except Exception as exc:
        logger.warning(f"Could not calculate benchmark returns: {exc}")
        return {"SPY_Return%": 0.0, "ALLW_Return%": 0.0, "60_40_Return%": 0.0}


def _ensure_csv_header_exists() -> None:
    """Create CSV with header if it doesn't exist."""
    if not os.path.exists(CSV_PATH):
        os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
        with open(CSV_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_HEADERS).writeheader()


def record_performance_snapshot(
    account: Any,
    positions: dict[str, PositionSnapshot],
    allocation: dict[str, float],
    logger: logging.Logger,
) -> None:
    """Record monthly performance snapshot to CSV."""
    _ensure_csv_header_exists()
    
    equity = float(account.equity)
    actual_allocation = calculate_allocation_actual(positions, allocation, equity)
    benchmark_returns = calculate_benchmark_returns(logger)
    
    # Calculate portfolio return from previous month if available
    portfolio_return = 0.0
    try:
        prev_equity = float(pd.read_csv(CSV_PATH).iloc[-1]["Portfolio_Equity"].replace("$", "").replace(",", ""))
        portfolio_return = round((equity - prev_equity) / prev_equity * 100, 2) if prev_equity > 0 else 0.0
    except (IndexError, KeyError, ValueError):
        pass
    
    row = {
        "Date": date.today().isoformat(),
        "Portfolio_Equity": f"${equity:,.2f}",
        **{f"{s}_Weight%": actual_allocation.get(s, 0) for s in ["SPY", "QQQ", "TLT", "TIP", "GLD", "GSG"]},
        "Portfolio_Return%": portfolio_return,
        **benchmark_returns,
    }
    
    try:
        with open(CSV_PATH, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_HEADERS).writerow(row)
        logger.info(f"Performance snapshot: {date.today().isoformat()} | Portfolio ${equity:,.2f} ({portfolio_return:+.2f}%) | "
                    f"Benchmarks SPY {benchmark_returns['SPY_Return%']:+.2f}% | ALLW {benchmark_returns['ALLW_Return%']:+.2f}% | 60/40 {benchmark_returns['60_40_Return%']:+.2f}%")
    except Exception as exc:
        logger.error(f"Failed to record performance: {exc}")


def _save_portfolio_holdings(
    positions: dict[str, PositionSnapshot],
    logger: logging.Logger,
) -> None:
    """Write current Alpaca positions to portfolio_holdings.json.

    Format matches portfolio.py's save_holdings():
        {"SPY": {"shares": 2.30, "last_price": 653.18}, ...}
    """
    holdings = {
        snap.symbol: {
            "shares": round(snap.qty, 6),
            "last_price": round(snap.current_price, 2),
        }
        for snap in sorted(positions.values(), key=lambda s: s.symbol)
    }
    with open(HOLDINGS_PATH, "w", encoding="utf-8") as fh:
        json.dump(holdings, fh, indent=2)
    logger.info(f"Portfolio holdings saved to {HOLDINGS_PATH}")


@dataclass
class PositionSnapshot:
    """Normalized portfolio snapshot for one symbol."""

    symbol: str
    qty: float
    qty_available: float
    market_value: float
    current_price: float


@dataclass
class RebalanceRow:
    """One planned rebalance action."""

    symbol: str
    target_weight: float
    current_weight: float
    target_value: float
    current_value: float
    delta_value: float
    action: str
    qty: float | None = None
    notional: float | None = None
    reason: str = ""


def _load_strategy_payload(strategy_id: str) -> dict[str, Any]:
    """Load one strategy definition from strategies.json or the example fallback."""
    base_path = os.path.dirname(__file__)
    strategies_path = os.path.join(base_path, "strategies.json")
    example_path = os.path.join(base_path, "strategies.example.json")

    if not os.path.exists(strategies_path):
        strategies_path = example_path

    with open(strategies_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    strategies = data["strategies"]
    if strategy_id not in strategies:
        raise KeyError(
            f"Strategy '{strategy_id}' not found. Available: {list(strategies.keys())}"
        )
    return strategies[strategy_id]


def _resolve_target_allocation(
    strategy_id: str,
    use_live_tickers: bool,
) -> tuple[dict[str, float], dict[str, str]]:
    """
    Resolve the target allocation, optionally translating backtest tickers to live ETFs.

    Returns:
        allocation: dict of tradable Alpaca symbols to weights
        mapping:    dict of original ticker -> tradable symbol
    """
    payload = _load_strategy_payload(strategy_id)
    allocation = payload["allocation"]
    live_tickers = payload.get("live_tickers", {})

    if not use_live_tickers:
        return dict(allocation), {ticker: ticker for ticker in allocation}

    translated: dict[str, float] = {}
    mapping: dict[str, str] = {}
    for ticker, weight in allocation.items():
        tradable = live_tickers.get(ticker, ticker)
        translated[tradable] = translated.get(tradable, 0.0) + float(weight)
        mapping[ticker] = tradable

    total = sum(translated.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Translated allocation must sum to 1.0, got {total:.6f}. "
            "Check live_tickers in strategies.json."
        )

    return translated, mapping


def _today_et() -> datetime:
    """Current New York time."""
    return datetime.now(NY_TZ)


def _coerce_date(value: Any) -> date:
    """Convert Alpaca calendar date payloads into date objects."""
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    return date.fromisoformat(str(value))


def get_end_of_month_status(client: TradingClient) -> tuple[bool, date, bool]:
    """
    Check whether today is the last US trading day of the month.

    Returns:
        is_last_trading_day, last_trading_day, market_is_open
    """
    now_et = _today_et()
    first_day = now_et.date().replace(day=1)
    probe_end = first_day + timedelta(days=40)
    month_end = probe_end.replace(day=1) - timedelta(days=1)

    calendar = client.get_calendar(
        GetCalendarRequest(start=first_day, end=month_end)
    )
    trading_days = sorted(_coerce_date(day.date) for day in calendar)
    if not trading_days:
        raise RuntimeError("Alpaca calendar returned no trading days for this month.")

    last_trading_day = trading_days[-1]
    clock = client.get_clock()
    return now_et.date() == last_trading_day, last_trading_day, bool(clock.is_open)


def get_account_snapshot(client: TradingClient) -> tuple[Any, dict[str, PositionSnapshot]]:
    """Fetch account state and normalize positions for planning."""
    account = client.get_account()
    positions = client.get_all_positions()

    snapshots: dict[str, PositionSnapshot] = {}
    for position in positions:
        snapshots[position.symbol] = PositionSnapshot(
            symbol=position.symbol,
            qty=float(position.qty),
            qty_available=float(position.qty_available or position.qty),
            market_value=abs(float(position.market_value or 0.0)),
            current_price=float(position.current_price or 0.0),
        )

    return account, snapshots


def _validate_target_assets(client: TradingClient, symbols: list[str]) -> dict[str, Any]:
    """Fetch Alpaca asset metadata for all target symbols."""
    assets: dict[str, Any] = {}
    for symbol in symbols:
        asset = client.get_asset(symbol)
        if not asset.tradable:
            raise ValueError(f"{symbol} is not tradable in Alpaca.")
        assets[symbol] = asset
    return assets


def build_rebalance_plan(
    account: Any,
    positions: dict[str, PositionSnapshot],
    allocation: dict[str, float],
    asset_meta: dict[str, Any],
    drift_threshold: float,
    min_trade_value: float,
    cash_buffer_pct: float,
    liquidate_other_positions: bool,
) -> tuple[list[RebalanceRow], list[str]]:
    """
    Build a conservative monthly rebalance plan.

    Buys use notional market orders, so target assets should be fractionable.
    Sells use qty market orders derived from the current position and price.
    """
    equity = float(account.equity)
    investable_equity = equity * (1.0 - cash_buffer_pct)
    if investable_equity <= 0:
        raise ValueError("Investable equity is <= 0. Check account funding and cash buffer.")

    target_symbols = set(allocation)
    warnings: list[str] = []
    rows: list[RebalanceRow] = []

    for symbol, weight in allocation.items():
        current = positions.get(
            symbol,
            PositionSnapshot(symbol=symbol, qty=0.0, qty_available=0.0, market_value=0.0, current_price=0.0),
        )
        target_value = investable_equity * weight
        current_value = current.market_value
        delta_value = target_value - current_value
        current_weight = current_value / equity if equity > 0 else 0.0
        drift = current_weight - weight

        action = "HOLD"
        qty: float | None = None
        notional: float | None = None
        reason = ""

        if abs(drift) <= drift_threshold or abs(delta_value) < min_trade_value:
            reason = "within thresholds"
        elif delta_value < 0:
            action = "SELL"
            if current.current_price <= 0 or current.qty_available <= 0:
                reason = "no sellable quantity"
                action = "HOLD"
            else:
                qty = min(abs(delta_value) / current.current_price, current.qty_available)
                qty = round(qty, 6)
                if qty <= 0:
                    action = "HOLD"
                    reason = "sell qty rounded to zero"
        else:
            action = "BUY"
            if not asset_meta[symbol].fractionable:
                action = "HOLD"
                reason = "asset is not fractionable; buy notional disabled"
                warnings.append(
                    f"{symbol} is not fractionable in Alpaca. "
                    "This script uses notional buys for monthly rebalancing."
                )
            else:
                notional = round(delta_value, 2)
                if notional < min_trade_value:
                    action = "HOLD"
                    reason = "buy notional below minimum trade size"

        rows.append(
            RebalanceRow(
                symbol=symbol,
                target_weight=weight,
                current_weight=current_weight,
                target_value=target_value,
                current_value=current_value,
                delta_value=delta_value,
                action=action,
                qty=qty,
                notional=notional,
                reason=reason,
            )
        )

    extra_symbols = sorted(set(positions) - target_symbols)
    for symbol in extra_symbols:
        position = positions[symbol]
        if position.market_value < min_trade_value:
            continue
        if liquidate_other_positions:
            rows.append(
                RebalanceRow(
                    symbol=symbol,
                    target_weight=0.0,
                    current_weight=(position.market_value / equity if equity > 0 else 0.0),
                    target_value=0.0,
                    current_value=position.market_value,
                    delta_value=-position.market_value,
                    action="SELL",
                    qty=round(position.qty_available, 6),
                    reason="non-strategy position",
                )
            )
        else:
            warnings.append(
                f"Non-strategy position detected: {symbol} (${position.market_value:,.2f}). "
                "Use --liquidate-other-positions to sell it."
            )

    rows.sort(key=lambda row: abs(row.delta_value), reverse=True)
    return rows, warnings


def plan_to_frame(rows: list[RebalanceRow]) -> pd.DataFrame:
    """Render the rebalance plan as a DataFrame for readable terminal output."""
    return pd.DataFrame(
        [
            {
                "Symbol": row.symbol,
                "Action": row.action,
                "Target %": round(row.target_weight * 100, 2),
                "Current %": round(row.current_weight * 100, 2),
                "Target $": round(row.target_value, 2),
                "Current $": round(row.current_value, 2),
                "Delta $": round(row.delta_value, 2),
                "Qty": row.qty,
                "Notional $": row.notional,
                "Reason": row.reason,
            }
            for row in rows
        ]
    )


def _status_str(order: Any) -> str:
    """Normalize Alpaca order statuses into lowercase strings."""
    return str(order.status).split(".")[-1].lower()


def wait_for_orders(
    client: TradingClient,
    order_ids: list[str],
    timeout_seconds: int,
    logger: logging.Logger,
) -> None:
    """Poll submitted orders until they reach a terminal state or timeout."""
    if not order_ids:
        return

    pending = set(order_ids)
    started = time.time()

    while pending and (time.time() - started) < timeout_seconds:
        completed: set[str] = set()
        for order_id in pending:
            order = client.get_order_by_id(order_id)
            status = _status_str(order)
            if status in {"filled", "canceled", "rejected", "expired"}:
                msg = f"Order {order_id} -> {status}"
                print(f"  {msg}")
                logger.info(msg)
                completed.add(order_id)
        pending -= completed
        if pending:
            time.sleep(2)

    if pending:
        msg = f"WARNING: Timed out waiting for {len(pending)} orders"
        print(f"\n{msg}:")
        logger.warning(msg)
        for order_id in sorted(pending):
            order = client.get_order_by_id(order_id)
            status_msg = f"  {order_id} -> {_status_str(order)}"
            print(status_msg)
            logger.warning(status_msg)


def execute_rebalance(
    client: TradingClient,
    initial_rows: list[RebalanceRow],
    allocation: dict[str, float],
    asset_meta: dict[str, Any],
    drift_threshold: float,
    min_trade_value: float,
    cash_buffer_pct: float,
    liquidate_other_positions: bool,
    timeout_seconds: int,
    logger: logging.Logger,
) -> None:
    """Execute monthly rebalance: sells first, then refresh, then buys."""
    sells = [row for row in initial_rows if row.action == "SELL" and row.qty and row.qty > 0]
    if sells:
        print("\nSubmitting sell orders first...")
        logger.info(f"Submitting {len(sells)} sell orders...")
    sell_order_ids: list[str] = []
    for row in sells:
        order = client.submit_order(
            MarketOrderRequest(
                symbol=row.symbol,
                qty=row.qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
        )
        sell_order_ids.append(str(order.id))
        msg = f"SELL {row.symbol:<6} qty={row.qty}"
        print(f"  {msg}")
        logger.info(msg)
    logger.info(f"Waiting for {len(sell_order_ids)} sell orders to fill (timeout: {timeout_seconds}s)...")
    wait_for_orders(client, sell_order_ids, timeout_seconds, logger)

    logger.info("Refreshing account snapshot after sells...")
    account, positions = get_account_snapshot(client)
    logger.info(f"Account refreshed: equity=${float(account.equity):,.2f}, cash=${float(account.cash):,.2f}")
    
    refreshed_rows, warnings = build_rebalance_plan(
        account=account,
        positions=positions,
        allocation=allocation,
        asset_meta=asset_meta,
        drift_threshold=drift_threshold,
        min_trade_value=min_trade_value,
        cash_buffer_pct=cash_buffer_pct,
        liquidate_other_positions=liquidate_other_positions,
    )

    if warnings:
        print("\nWarnings after sells:")
        for warning in warnings:
            print(f"  - {warning}")
            logger.warning(warning)

    buys = [row for row in refreshed_rows if row.action == "BUY" and row.notional and row.notional > 0]
    if not buys:
        msg = "No buy orders required after refreshing the account."
        print(f"\n{msg}")
        logger.info(msg)
        return

    print("\nSubmitting buy orders after refresh...")
    logger.info(f"Submitting {len(buys)} buy orders...")
    buy_order_ids: list[str] = []
    for row in buys:
        order = client.submit_order(
            MarketOrderRequest(
                symbol=row.symbol,
                notional=row.notional,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
        )
        buy_order_ids.append(str(order.id))
        msg = f"BUY  {row.symbol:<6} notional=${row.notional:,.2f}"
        print(f"  {msg}")
        logger.info(msg)
    logger.info(f"Waiting for {len(buy_order_ids)} buy orders to fill (timeout: {timeout_seconds}s)...")
    wait_for_orders(client, buy_order_ids, timeout_seconds, logger)


def parse_args() -> argparse.Namespace:
    """CLI options for preview and execution."""
    parser = argparse.ArgumentParser(
        description="Monthly ETF rebalancer for an Alpaca paper account."
    )
    parser.add_argument(
        "--strategy-id",
        default=config.DEFAULT_STRATEGY,
        help=f"Strategy id from strategies.json (default: {config.DEFAULT_STRATEGY})",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually submit paper-trading orders. Without this flag the script is preview only.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass the end-of-month guard. Market-open guard still applies.",
    )
    parser.add_argument(
        "--drift-threshold",
        type=float,
        default=config.REBALANCE_THRESHOLD,
        help=f"Minimum absolute weight drift to trade (default: {config.REBALANCE_THRESHOLD:.2f})",
    )
    parser.add_argument(
        "--min-trade-value",
        type=float,
        default=25.0,
        help="Ignore trades smaller than this dollar amount (default: 25.0)",
    )
    parser.add_argument(
        "--cash-buffer-pct",
        type=float,
        default=0.005,
        help="Keep this fraction of equity in cash to reduce order rejects (default: 0.005)",
    )
    parser.add_argument(
        "--liquidate-other-positions",
        action="store_true",
        help="Sell positions that are not part of the selected strategy.",
    )
    parser.add_argument(
        "--use-live-tickers",
        action="store_true",
        help="Translate backtest tickers via strategy live_tickers when available.",
    )
    parser.add_argument(
        "--account",
        default=None,
        help=(
            "Named account suffix for env vars. E.g. --account live reads "
            "APCA_API_KEY_ID_LIVE / APCA_API_SECRET_KEY_LIVE. "
            "Omit to use the default APCA_API_KEY_ID / APCA_API_SECRET_KEY."
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"How long to wait for fills after submitting orders (default: {DEFAULT_TIMEOUT_SECONDS})",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    logger = setup_logging()
    args = parse_args()

    logger.info("=" * 72)
    logger.info("ALPACA MONTHLY PAPER REBALANCER")
    logger.info("=" * 72)
    logger.info(f"Strategy ID: {args.strategy_id}")
    logger.info(f"Mode: {'EXECUTE' if args.execute else 'PREVIEW ONLY'}")
    logger.info(f"Force flag: {args.force}")

    # Resolve credentials: --account <suffix> reads APCA_API_KEY_ID_<SUFFIX>
    if args.account:
        suffix = args.account.upper()
        key_var = f"APCA_API_KEY_ID_{suffix}"
        secret_var = f"APCA_API_SECRET_KEY_{suffix}"
    else:
        key_var = "APCA_API_KEY_ID"
        secret_var = "APCA_API_SECRET_KEY"

    api_key = os.getenv(key_var)
    secret_key = os.getenv(secret_var)
    if not api_key or not secret_key:
        logger.error(f"Missing Alpaca credentials. Set {key_var} and {secret_var}.")
        raise SystemExit(
            f"Missing Alpaca credentials. Set {key_var} and {secret_var}."
        )

    account_label = args.account or "default"
    logger.info(f"Alpaca credentials found for account '{account_label}' ({key_var}).")

    try:
        logger.info(f"Loading strategy: {args.strategy_id}")
        allocation, mapping = _resolve_target_allocation(
            strategy_id=args.strategy_id,
            use_live_tickers=args.use_live_tickers,
        )
        logger.info(f"Strategy loaded with {len(allocation)} assets")
        
        logger.info("Connecting to Alpaca paper trading...")
        client = TradingClient(api_key=api_key, secret_key=secret_key, paper=True)
        logger.info("Successfully connected to Alpaca")
        
        logger.info("Validating target assets...")
        asset_meta = _validate_target_assets(client, list(allocation.keys()))
        logger.info(f"All {len(asset_meta)} target assets validated as tradable")

        logger.info("Checking calendar and market status...")
        is_last_trading_day, last_trading_day, market_is_open = get_end_of_month_status(client)
        logger.info(f"Last trading day of month: {last_trading_day.isoformat()}")
        logger.info(f"Today is month-end: {is_last_trading_day}")
        logger.info(f"Market is open: {market_is_open}")
        
        logger.info("Fetching account snapshot...")
        account, positions = get_account_snapshot(client)
        logger.info(f"Account equity: ${float(account.equity):,.2f}")
        logger.info(f"Account cash: ${float(account.cash):,.2f}")
        logger.info(f"Current positions: {len(positions)}")
        
        logger.info("Building rebalance plan...")
        rows, warnings = build_rebalance_plan(
            account=account,
            positions=positions,
            allocation=allocation,
            asset_meta=asset_meta,
            drift_threshold=args.drift_threshold,
            min_trade_value=args.min_trade_value,
            cash_buffer_pct=args.cash_buffer_pct,
            liquidate_other_positions=args.liquidate_other_positions,
        )
        logger.info(f"Rebalance plan built: {len(rows)} positions analyzed")

        now_et = _today_et()
        print("=" * 72)
        print("ALPACA MONTHLY PAPER REBALANCER")
        print("=" * 72)
        print(f"Now (ET):             {now_et.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"Strategy:             {args.strategy_id}")
        print(f"Paper account status: {'market open' if market_is_open else 'market closed'}")
        print(f"Last trading day:     {last_trading_day.isoformat()}")
        print(f"Today is month-end:   {'yes' if is_last_trading_day else 'no'}")
        print(f"Execution mode:       {'EXECUTE' if args.execute else 'PREVIEW ONLY'}")
        print(f"Equity:               ${float(account.equity):,.2f}")
        print(f"Cash:                 ${float(account.cash):,.2f}")
        print(f"Buying power:         ${float(account.buying_power):,.2f}")

        if args.use_live_tickers:
            print("\nTicker mapping:")
            for backtest_ticker, tradable_symbol in mapping.items():
                if backtest_ticker == tradable_symbol:
                    continue
                print(f"  {backtest_ticker} -> {tradable_symbol}")

        if warnings:
            print("\nWarnings:")
            for warning in warnings:
                print(f"  - {warning}")
                logger.warning(warning)

        print("\nRebalance plan:")
        frame = plan_to_frame(rows)
        if frame.empty:
            print("  No positions or targets found.")
            logger.info("No positions or targets found in rebalance plan.")
        else:
            print(frame.to_string(index=False))
            logger.debug("\nRebalance plan:\n" + frame.to_string(index=False))

        if not args.execute:
            print("\nPreview complete. Re-run with --execute to submit paper orders.")
            logger.info("Preview mode: no orders submitted")
            # Record snapshot even in preview mode for tracking
            logger.info("Recording performance snapshot (preview mode)...")
            record_performance_snapshot(account, positions, allocation, logger)
            _save_portfolio_holdings(positions, logger)
            return

        if not is_last_trading_day and not args.force:
            msg = (f"Refusing to trade because today is not the last trading day of the month "
                   f"({last_trading_day.isoformat()}). Use --force to override.")
            logger.error(msg)
            raise SystemExit(msg)

        if not market_is_open:
            msg = "Refusing to submit market orders while the regular session is closed."
            logger.error(msg)
            raise SystemExit(msg)

        if warnings and not args.liquidate_other_positions:
            msg = "Refusing to execute with unresolved warnings. Review the preview output first."
            logger.error(msg)
            raise SystemExit(msg)

        logger.info("All pre-execution checks passed. Beginning execution...")
        execute_rebalance(
            client=client,
            initial_rows=rows,
            allocation=allocation,
            asset_meta=asset_meta,
            drift_threshold=args.drift_threshold,
            min_trade_value=args.min_trade_value,
            cash_buffer_pct=args.cash_buffer_pct,
            liquidate_other_positions=args.liquidate_other_positions,
            timeout_seconds=args.timeout_seconds,
            logger=logger,
        )
        logger.info("Execution complete. ✓")
        print("\nExecution complete.")
        
        # Record performance snapshot and holdings after execution
        logger.info("Recording performance snapshot...")
        final_account, final_positions = get_account_snapshot(client)
        record_performance_snapshot(final_account, final_positions, allocation, logger)
        _save_portfolio_holdings(final_positions, logger)
        print("Performance snapshot recorded to logs/performance_tracking.csv")
        print(f"Portfolio holdings saved to {HOLDINGS_PATH}")
        
    except SystemExit:
        raise
    except Exception as exc:
        logger.exception(f"Fatal error: {exc}")
        raise


if __name__ == "__main__":
    main()
