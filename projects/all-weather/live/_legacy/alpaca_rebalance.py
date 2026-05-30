"""
alpaca_rebalance.py
===================
Monthly ETF rebalancer for Alpaca paper or live accounts.

What it does
------------
- Loads a target ETF allocation from strategies.json
- Connects to Alpaca paper or live trading via alpaca-py
- Checks whether today is the last US trading day of the month
- Reads account equity, cash, and open positions
- Builds a rebalance plan from current weights to target weights
- Optionally submits market orders during the regular session

Safety defaults
---------------
- Preview only unless --execute is passed
- Refuses to trade outside the last trading day of the month unless --force
- Refuses to trade when the regular market is closed
- Refuses to trade with open target-symbol orders
- Refuses duplicate executions unless explicitly allowed
- Refuses non-production strategies unless explicitly allowed
- Uses strategy live tickers by default (e.g. GLD -> GLDM)
- Fails on rejected, canceled, expired, or timed-out orders
- Verifies final portfolio weights after execution
- Executes sells first, then refreshes the account and computes buys again
- Leaves non-strategy positions alone unless --liquidate-other-positions is passed

Environment
-----------
Set Alpaca keys in /Users/franciscosimao/Documents/QuantFinance/api_keys.env:

    ALPACA_API_KEY="..."
    ALPACA_SECRET_KEY="..."

For multiple accounts, add a suffix (e.g. --account live):

    export APCA_API_KEY_ID_LIVE="..."
    export APCA_API_SECRET_KEY_LIVE="..."

Examples
--------
Preview only (default account, live tickers):
    conda run -n allweather python -m live.alpaca_rebalance --paper

Preview on the "live" account without executing:
    conda run -n allweather python -m live.alpaca_rebalance --live --account live

Execute on the last trading day:
    conda run -n allweather python -m live.alpaca_rebalance --paper --strategy-id 6_asset_rp_baseline --execute
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

from engine import config
from live.env import load_api_keys_env

load_api_keys_env()

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
    from alpaca.trading.requests import GetCalendarRequest, GetOrdersRequest, MarketOrderRequest
except ImportError as exc:  # pragma: no cover - import guard for environments without alpaca-py
    raise SystemExit(
        "alpaca-py is not installed. Install it with:\n\n"
        "    pip install alpaca-py\n"
    ) from exc


NY_TZ = ZoneInfo("America/New_York")
DEFAULT_TIMEOUT_SECONDS = 60
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOGS_DIR = os.path.join(_PROJECT_ROOT, "logs")
HOLDINGS_PATH = os.path.join(_PROJECT_ROOT, "portfolio_holdings.json")
RUN_REGISTRY_PATH = os.path.join(LOGS_DIR, "rebalance_run_registry.json")


class OrderExecutionError(RuntimeError):
    """Raised when a submitted order is not safely filled."""


def setup_logging() -> logging.Logger:
    """
    Configure logging to both console and file.
    
    Creates logs/ directory if it doesn't exist.
    Returns a logger instance that writes to both stdout and a timestamped log file.
    """
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    # Create timestamped log file
    now_et = datetime.now(NY_TZ)
    log_filename = now_et.strftime("%Y-%m-%d_%H-%M-%S_alpaca_rebalance.log")
    log_path = os.path.join(LOGS_DIR, log_filename)
    
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


def _previous_month_end_closes(close_prices: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    """Return latest and previous calendar-month-end closes for each column."""
    close_prices = close_prices.dropna(how="all").ffill()
    if close_prices.empty:
        raise ValueError("No benchmark prices returned.")

    latest_date = close_prices.index[-1]
    previous_month = close_prices.index.to_period("M") < latest_date.to_period("M")
    previous = close_prices.loc[previous_month]
    if previous.empty:
        raise ValueError("Not enough benchmark history to find previous month-end.")

    latest_row = close_prices.iloc[-1]
    previous_row = previous.iloc[-1]
    latest = {ticker: float(latest_row[ticker]) for ticker in close_prices.columns}
    prior = {ticker: float(previous_row[ticker]) for ticker in close_prices.columns}
    return latest, prior


def calculate_benchmark_returns(logger: logging.Logger) -> dict[str, float]:
    """Calculate month-to-date benchmark returns using real month-end closes."""
    try:
        tickers = yf.download(["SPY", "ALLW", "TLT"], period="3mo", progress=False)
        close_prices = tickers["Close"]
        if isinstance(close_prices, pd.Series):
            close_prices = close_prices.to_frame()

        today_close, prev_close = _previous_month_end_closes(close_prices)
        
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


def _performance_csv_path(account_label: str, trading_mode: str) -> str:
    """Return the private audit CSV path for one Alpaca account/mode."""
    safe_account = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in account_label)
    return os.path.join(LOGS_DIR, f"performance_tracking_{trading_mode}_{safe_account}.csv")


def _performance_headers(allocation: dict[str, float]) -> list[str]:
    """Build CSV headers from actual tradable symbols, e.g. GLDM not GLD."""
    weight_cols = [f"{symbol}_Weight%" for symbol in allocation]
    drift_cols = [f"{symbol}_Drift%" for symbol in allocation]
    return [
        "Date", "Portfolio_Equity",
        *weight_cols,
        *drift_cols,
        "Portfolio_Return%",
        "SPY_Return%", "ALLW_Return%", "60_40_Return%",
    ]


def _ensure_csv_header_exists(csv_path: str, headers: list[str]) -> None:
    """Create CSV with header if it doesn't exist."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            existing = next(csv.reader(f), [])
        if existing and existing != headers:
            raise ValueError(
                f"Performance CSV header mismatch in {csv_path}. "
                "Use a separate account label or archive the old CSV before changing symbols."
            )
        return

    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=headers).writeheader()


def record_performance_snapshot(
    account: Any,
    positions: dict[str, PositionSnapshot],
    allocation: dict[str, float],
    logger: logging.Logger,
    csv_path: str,
) -> None:
    """Record monthly performance snapshot to CSV."""
    headers = _performance_headers(allocation)
    _ensure_csv_header_exists(csv_path, headers)
    
    equity = float(account.equity)
    actual_allocation = calculate_allocation_actual(positions, allocation, equity)
    benchmark_returns = calculate_benchmark_returns(logger)
    drift = {
        symbol: round(actual_allocation.get(symbol, 0.0) - weight * 100, 2)
        for symbol, weight in allocation.items()
    }
    
    # Calculate portfolio return from previous month if available
    portfolio_return = 0.0
    try:
        prev_equity = float(pd.read_csv(csv_path).iloc[-1]["Portfolio_Equity"].replace("$", "").replace(",", ""))
        portfolio_return = round((equity - prev_equity) / prev_equity * 100, 2) if prev_equity > 0 else 0.0
    except (IndexError, KeyError, ValueError):
        pass
    
    row = {
        "Date": date.today().isoformat(),
        "Portfolio_Equity": f"${equity:,.2f}",
        **{f"{s}_Weight%": actual_allocation.get(s, 0) for s in allocation},
        **{f"{s}_Drift%": drift.get(s, 0) for s in allocation},
        "Portfolio_Return%": portfolio_return,
        **benchmark_returns,
    }
    
    try:
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=headers).writerow(row)
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
    strategies_path = os.path.join(_PROJECT_ROOT, "strategies.json")
    example_path = os.path.join(_PROJECT_ROOT, "strategies.example.json")

    if not os.path.exists(strategies_path):
        strategies_path = example_path

    with open(strategies_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    strategies = data["strategies"]
    canonical_id = config.resolve_strategy_id(strategy_id)
    if canonical_id not in strategies:
        raise KeyError(
            f"Strategy '{strategy_id}' not found. Available: {list(strategies.keys())}"
        )
    payload = dict(strategies[canonical_id])
    payload["_strategy_id"] = canonical_id
    return payload


def _assert_strategy_live_allowed(
    strategy_id: str,
    payload: dict[str, Any],
    allow_non_production: bool,
) -> None:
    """Block accidental trading of research or archived strategy definitions."""
    if allow_non_production:
        return

    is_allowed = bool(payload.get("allow_live_trading")) or (
        payload.get("tier") == "Production" and bool(payload.get("is_production"))
    )
    if is_allowed:
        return

    display = payload.get("display_name") or payload.get("description") or strategy_id
    raise SystemExit(
        f"Refusing to trade non-production strategy '{strategy_id}' ({display}). "
        "Use --allow-non-production-strategy only after manual review."
    )


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


def get_open_orders(client: TradingClient, symbols: list[str]) -> list[Any]:
    """Return open Alpaca orders for the symbols this run might trade."""
    request = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=list(symbols))
    return list(client.get_orders(filter=request))


def assert_no_open_orders(client: TradingClient,
                          symbols: list[str],
                          logger: logging.Logger) -> None:
    """Refuse to trade when there are already open orders in target symbols."""
    open_orders = get_open_orders(client, symbols)
    if not open_orders:
        return

    details = []
    for order in open_orders:
        symbol = getattr(order, "symbol", "UNKNOWN")
        order_id = getattr(order, "id", "UNKNOWN")
        status = _status_str(order)
        details.append(f"{symbol}:{order_id}:{status}")
    msg = "Open Alpaca orders already exist for target symbols: " + ", ".join(details)
    logger.error(msg)
    raise SystemExit(msg)


def build_rebalance_plan(
    account: Any,
    positions: dict[str, PositionSnapshot],
    allocation: dict[str, float],
    asset_meta: dict[str, Any],
    drift_threshold: float,
    min_trade_value: float,
    cash_buffer_pct: float,
    liquidate_other_positions: bool,
    rebalance_mode: str = "per_asset",
) -> tuple[list[RebalanceRow], list[str]]:
    """
    Build a conservative monthly rebalance plan.

    Buys use notional market orders, so target assets should be fractionable.
    Sells use qty market orders derived from the current position and price.

    rebalance_mode:
        "per_asset"      — each asset checked independently; only breaching assets traded
        "full_on_breach" — if any asset breaches drift_threshold, ALL assets go to target
    """
    equity = float(account.equity)
    investable_equity = equity * (1.0 - cash_buffer_pct)
    if investable_equity <= 0:
        raise ValueError("Investable equity is <= 0. Check account funding and cash buffer.")

    # In full_on_breach mode: if any asset drifts past the threshold, rebalance everything.
    # Achieved by setting the effective threshold to 0 for this run.
    if rebalance_mode == "full_on_breach":
        any_breach = any(
            abs((positions[s].market_value / equity if equity > 0 else 0.0) - w) > drift_threshold
            for s, w in allocation.items()
            if s in positions
        )
        effective_threshold = 0.0 if any_breach else drift_threshold
    else:
        effective_threshold = drift_threshold

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

        if abs(drift) <= effective_threshold or abs(delta_value) < min_trade_value:
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


def validate_order_guardrails(rows: list[RebalanceRow],
                              equity: float,
                              max_order_pct_equity: float,
                              max_total_trade_pct_equity: float) -> None:
    """Apply conservative notional limits before any order is submitted."""
    trade_rows = [row for row in rows if row.action in {"BUY", "SELL"}]
    if not trade_rows or equity <= 0:
        return

    max_order_value = equity * max_order_pct_equity
    max_total_value = equity * max_total_trade_pct_equity
    total_trade_value = sum(abs(row.delta_value) for row in trade_rows)

    oversized = [
        row for row in trade_rows
        if abs(row.delta_value) > max_order_value
    ]
    if oversized:
        details = ", ".join(
            f"{row.symbol} ${abs(row.delta_value):,.2f}" for row in oversized
        )
        raise SystemExit(
            f"Refusing to trade: one or more orders exceed "
            f"{max_order_pct_equity:.0%} of equity (${max_order_value:,.2f}): {details}"
        )
    if total_trade_value > max_total_value:
        raise SystemExit(
            f"Refusing to trade: total planned turnover ${total_trade_value:,.2f} "
            f"exceeds {max_total_trade_pct_equity:.0%} of equity (${max_total_value:,.2f})."
        )


def verify_post_trade_drift(account: Any,
                            positions: dict[str, PositionSnapshot],
                            allocation: dict[str, float],
                            tolerance: float,
                            min_extra_position_value: float,
                            logger: logging.Logger) -> pd.DataFrame:
    """Validate final account weights against target allocation after execution."""
    equity = float(account.equity)
    rows = []
    breaches = []
    for symbol, target_weight in allocation.items():
        current = positions.get(
            symbol,
            PositionSnapshot(symbol=symbol, qty=0.0, qty_available=0.0, market_value=0.0, current_price=0.0),
        )
        current_weight = current.market_value / equity if equity > 0 else 0.0
        drift = current_weight - target_weight
        rows.append({
            "Symbol": symbol,
            "Target %": round(target_weight * 100, 2),
            "Actual %": round(current_weight * 100, 2),
            "Drift %": round(drift * 100, 2),
            "Market Value": round(current.market_value, 2),
        })
        if abs(drift) > tolerance:
            breaches.append(f"{symbol} drift {drift:+.2%}")

    extra_positions = sorted(set(positions) - set(allocation))
    for symbol in extra_positions:
        position = positions[symbol]
        if position.market_value >= min_extra_position_value:
            breaches.append(f"non-strategy {symbol} ${position.market_value:,.2f}")

    frame = pd.DataFrame(rows)
    logger.info("\nPost-trade verification:\n" + frame.to_string(index=False))
    if breaches:
        raise OrderExecutionError(
            "Post-trade verification failed: " + "; ".join(breaches)
        )
    return frame


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


def _run_registry_key(trading_day: date,
                      account_label: str,
                      trading_mode: str,
                      strategy_id: str) -> str:
    return f"{trading_day.isoformat()}|{trading_mode}|{account_label}|{strategy_id}"


def assert_not_duplicate_run(trading_day: date,
                             account_label: str,
                             trading_mode: str,
                             strategy_id: str,
                             allow_duplicate: bool,
                             logger: logging.Logger) -> None:
    """Prevent accidental second execution for the same account/date/strategy."""
    if allow_duplicate or not os.path.exists(RUN_REGISTRY_PATH):
        return

    with open(RUN_REGISTRY_PATH, "r", encoding="utf-8") as handle:
        registry = json.load(handle)
    key = _run_registry_key(trading_day, account_label, trading_mode, strategy_id)
    if key in registry:
        msg = (
            f"Refusing duplicate execution for {key}. "
            "Use --allow-duplicate-run only after manually confirming no orders are pending."
        )
        logger.error(msg)
        raise SystemExit(msg)


def record_successful_run(trading_day: date,
                          account_label: str,
                          trading_mode: str,
                          strategy_id: str,
                          logger: logging.Logger) -> None:
    """Persist a small private marker after a successful execution."""
    os.makedirs(os.path.dirname(RUN_REGISTRY_PATH), exist_ok=True)
    if os.path.exists(RUN_REGISTRY_PATH):
        with open(RUN_REGISTRY_PATH, "r", encoding="utf-8") as handle:
            registry = json.load(handle)
    else:
        registry = {}

    key = _run_registry_key(trading_day, account_label, trading_mode, strategy_id)
    registry[key] = {
        "recorded_at": datetime.now(NY_TZ).isoformat(),
        "account": account_label,
        "trading_mode": trading_mode,
        "strategy_id": strategy_id,
    }
    with open(RUN_REGISTRY_PATH, "w", encoding="utf-8") as handle:
        json.dump(registry, handle, indent=2, sort_keys=True)
    logger.info(f"Recorded successful rebalance run marker: {key}")


def wait_for_orders(
    client: TradingClient,
    order_ids: list[str],
    timeout_seconds: int,
    logger: logging.Logger,
) -> dict[str, str]:
    """Poll submitted orders until they reach a terminal state or timeout."""
    if not order_ids:
        return {}

    pending = set(order_ids)
    statuses: dict[str, str] = {}
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
                statuses[order_id] = status
        pending -= completed
        if pending:
            time.sleep(2)

    if pending:
        msg = f"Timed out waiting for {len(pending)} orders"
        print(f"\n{msg}:")
        logger.error(msg)
        for order_id in sorted(pending):
            order = client.get_order_by_id(order_id)
            status_msg = f"  {order_id} -> {_status_str(order)}"
            print(status_msg)
            logger.error(status_msg)
            statuses[order_id] = "timeout"

    failed = {
        order_id: status for order_id, status in statuses.items()
        if status != "filled"
    }
    if failed:
        details = ", ".join(f"{order_id}:{status}" for order_id, status in failed.items())
        raise OrderExecutionError(f"Order execution failed: {details}")
    return statuses


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
    rebalance_mode: str = "per_asset",
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
        rebalance_mode=rebalance_mode,
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
        description="Monthly ETF rebalancer for Alpaca paper or live accounts."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--paper",
        action="store_true",
        help="Use Alpaca paper trading. This is the default.",
    )
    mode.add_argument(
        "--live",
        action="store_true",
        help="Use Alpaca live trading. Requires live credentials and all safety checks.",
    )
    parser.add_argument(
        "--strategy-id",
        default=config.DEFAULT_STRATEGY,
        help=f"Strategy id from strategies.json (default: {config.DEFAULT_STRATEGY})",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually submit Alpaca orders. Without this flag the script is preview only.",
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
        "--rebalance-mode",
        choices=["per_asset", "full_on_breach"],
        default=config.REBALANCE_MODE,
        help=(
            "per_asset: only breaching assets are traded (default). "
            "full_on_breach: if any asset breaches the threshold, all assets are brought to target."
        ),
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
    ticker_mode = parser.add_mutually_exclusive_group()
    ticker_mode.add_argument(
        "--use-live-tickers",
        action="store_true",
        dest="use_live_tickers",
        help="Translate backtest tickers via strategy live_tickers when available. This is the default.",
    )
    ticker_mode.add_argument(
        "--use-backtest-tickers",
        action="store_false",
        dest="use_live_tickers",
        help="Trade the backtest tickers directly instead of strategy live_tickers.",
    )
    parser.set_defaults(use_live_tickers=True)
    parser.add_argument(
        "--allow-non-production-strategy",
        action="store_true",
        help="Allow preview/execution for a strategy not marked for live trading.",
    )
    parser.add_argument(
        "--account",
        default=None,
        help=(
            "Named account suffix for api_keys.env vars. E.g. --account live reads "
            "BROKER_ALPACA_LIVE_KEY / BROKER_ALPACA_LIVE_SECRET or "
            "APCA_API_KEY_ID_LIVE / APCA_API_SECRET_KEY_LIVE. "
            "Omit to use ALPACA_API_KEY / ALPACA_SECRET_KEY."
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"How long to wait for fills after submitting orders (default: {DEFAULT_TIMEOUT_SECONDS})",
    )
    parser.add_argument(
        "--post-trade-tolerance",
        type=float,
        default=0.015,
        help="Maximum allowed absolute post-trade weight drift (default: 0.015 = 1.5%%).",
    )
    parser.add_argument(
        "--max-order-pct-equity",
        type=float,
        default=0.40,
        help="Refuse any single planned trade above this fraction of equity (default: 0.40).",
    )
    parser.add_argument(
        "--max-total-trade-pct-equity",
        type=float,
        default=1.05,
        help="Refuse total planned turnover above this fraction of equity (default: 1.05).",
    )
    parser.add_argument(
        "--allow-duplicate-run",
        action="store_true",
        help="Allow a second execution for the same trading day/account/strategy after manual review.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    logger = setup_logging()

    logger.info("=" * 72)
    logger.info("ALPACA MONTHLY PAPER REBALANCER")
    logger.info("=" * 72)
    logger.info(f"Strategy ID: {args.strategy_id}")
    logger.info(f"Mode: {'EXECUTE' if args.execute else 'PREVIEW ONLY'}")
    logger.info(f"Force flag: {args.force}")
    paper_trading = not args.live
    trading_mode = "paper" if paper_trading else "live"
    logger.info(f"Trading environment: {trading_mode.upper()}")

    # Resolve credentials loaded from api_keys.env.
    if args.account:
        suffix = args.account.upper()
        credential_pairs = [
            (f"BROKER_ALPACA_{suffix}_KEY", f"BROKER_ALPACA_{suffix}_SECRET"),
            (f"APCA_API_KEY_ID_{suffix}", f"APCA_API_SECRET_KEY_{suffix}"),
        ]
    else:
        credential_pairs = [
            ("BROKER_ALPACA_DEFAULT_KEY", "BROKER_ALPACA_DEFAULT_SECRET"),
            ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY"),
            ("ALPACA_API_KEY", "ALPACA_SECRET_KEY"),
        ]

    api_key = None
    secret_key = None
    key_var = credential_pairs[0][0]
    for candidate_key, candidate_secret in credential_pairs:
        api_key = os.getenv(candidate_key)
        secret_key = os.getenv(candidate_secret)
        if api_key and secret_key:
            key_var = candidate_key
            break
    if not api_key or not secret_key:
        tried = ", ".join(f"{key}/{secret}" for key, secret in credential_pairs)
        logger.error(f"Missing Alpaca credentials. Set one of: {tried}.")
        raise SystemExit(
            f"Missing Alpaca credentials. Set one of these pairs in api_keys.env: {tried}."
        )

    account_label = args.account or "default"
    logger.info(f"Alpaca credentials found for account '{account_label}' ({key_var}).")

    try:
        logger.info(f"Loading strategy: {args.strategy_id}")
        payload = _load_strategy_payload(args.strategy_id)
        _assert_strategy_live_allowed(
            strategy_id=args.strategy_id,
            payload=payload,
            allow_non_production=args.allow_non_production_strategy,
        )
        allocation, mapping = _resolve_target_allocation(
            strategy_id=args.strategy_id,
            use_live_tickers=args.use_live_tickers,
        )
        display_name = payload.get("display_name") or payload.get("description") or args.strategy_id
        logger.info(f"Strategy loaded: {display_name} ({payload.get('_strategy_id')})")
        logger.info(f"Strategy loaded with {len(allocation)} assets")
        
        logger.info(f"Connecting to Alpaca {trading_mode} trading...")
        client = TradingClient(api_key=api_key, secret_key=secret_key, paper=paper_trading)
        logger.info("Successfully connected to Alpaca")
        
        logger.info("Validating target assets...")
        asset_meta = _validate_target_assets(client, list(allocation.keys()))
        logger.info(f"All {len(asset_meta)} target assets validated as tradable")

        logger.info("Checking calendar and market status...")
        is_last_trading_day, last_trading_day, market_is_open = get_end_of_month_status(client)
        logger.info(f"Last trading day of month: {last_trading_day.isoformat()}")
        logger.info(f"Today is month-end: {is_last_trading_day}")
        logger.info(f"Market is open: {market_is_open}")

        if args.execute:
            assert_not_duplicate_run(
                trading_day=last_trading_day,
                account_label=account_label,
                trading_mode=trading_mode,
                strategy_id=args.strategy_id,
                allow_duplicate=args.allow_duplicate_run,
                logger=logger,
            )
            assert_no_open_orders(client, list(allocation.keys()), logger)
        
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
            rebalance_mode=args.rebalance_mode,
        )
        logger.info(f"Rebalance plan built: {len(rows)} positions analyzed")

        now_et = _today_et()
        print("=" * 72)
        print(f"ALPACA MONTHLY {trading_mode.upper()} REBALANCER")
        print("=" * 72)
        print(f"Now (ET):             {now_et.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"Strategy:             {args.strategy_id}")
        print(f"Trading environment:  {trading_mode.upper()}")
        print(f"Account status:       {'market open' if market_is_open else 'market closed'}")
        print(f"Last trading day:     {last_trading_day.isoformat()}")
        print(f"Today is month-end:   {'yes' if is_last_trading_day else 'no'}")
        print(f"Execution mode:       {'EXECUTE' if args.execute else 'PREVIEW ONLY'}")
        print(f"Rebalance mode:       {args.rebalance_mode}")
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

        validate_order_guardrails(
            rows=rows,
            equity=float(account.equity),
            max_order_pct_equity=args.max_order_pct_equity,
            max_total_trade_pct_equity=args.max_total_trade_pct_equity,
        )

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
            rebalance_mode=args.rebalance_mode,
        )
        logger.info("Execution complete. ✓")
        print("\nExecution complete.")
        
        logger.info("Verifying final positions...")
        final_account, final_positions = get_account_snapshot(client)
        verification = verify_post_trade_drift(
            account=final_account,
            positions=final_positions,
            allocation=allocation,
            tolerance=args.post_trade_tolerance,
            min_extra_position_value=args.min_trade_value,
            logger=logger,
        )
        print("\nPost-trade verification:")
        print(verification.to_string(index=False))

        # Record performance snapshot and holdings after execution
        logger.info("Recording performance snapshot...")
        performance_path = _performance_csv_path(account_label, trading_mode)
        record_performance_snapshot(final_account, final_positions, allocation, logger, performance_path)
        _save_portfolio_holdings(final_positions, logger)
        record_successful_run(last_trading_day, account_label, trading_mode, args.strategy_id, logger)
        print(f"Performance snapshot recorded to {performance_path}")
        print(f"Portfolio holdings saved to {HOLDINGS_PATH}")
        
    except SystemExit:
        raise
    except Exception as exc:
        logger.exception(f"Fatal error: {exc}")
        raise


if __name__ == "__main__":
    main()
