"""
live/rebalance.py
=================
Broker-agnostic monthly / ≥31-day ETF rebalancer.

This module supersedes live/_legacy/alpaca_rebalance.py and talks ONLY to the Broker
protocol defined in live/brokers/base.py.  The concrete broker is chosen at
startup via --broker and constructed by live/brokers/factory.make_broker.

Usage
-----
Preview (default broker = alpaca, paper):
    conda run -n allweather python -m live.rebalance --paper

Execute against the Tastytrade account labelled "main":
    conda run -n allweather python -m live.rebalance --live --broker tastytrade --account main --execute

Dry-execute (simulate fills without real orders, useful for log testing):
    conda run -n allweather python -m live.rebalance --paper --dry-execute

Cadence notes
-------------
The old month-end calendar gate is replaced by a minimum-interval check:
  --min-rebalance-interval-days  (default 31)
After a successful --execute run the cadence state file is updated.
Use --force-cadence to bypass the interval check (e.g. first run after
account setup, or manual override).

Budget notes
------------
When --budget AMOUNT is set, the rebalancer sizes orders against that dollar
amount instead of the full account equity.  The budget grows automatically
from strategy-symbol dividends and realised PnL.
Phase 2 (live/budget.py) implements full budget persistence; until then,
pass --ignore-budget to fall back to account equity.

Holding-period notes (Phase 3 — live/lots.py)
---------------------------------------------
The rebalancer respects a 31-day minimum holding period enforced by the lot
ledger.  Positions acquired < 31 days ago are protected from sells.
Pass --initialize-lots on first run to seed the lot ledger from current
positions (assumes all positions are immediately sellable by default).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from engine import config
from live.brokers import Broker, make_broker
from live.brokers.base import (
    TERMINAL_ORDER_STATUSES,
    AccountSnapshot,
    OrderResult,
    PositionSnapshot,
)
from live.budget import (
    BudgetSnapshot,
    _budget_state_path,
    get_managed_capital,
    initialise_budget,
    load_budget,
    reconcile_after_trades,
    save_state as save_budget_state,
)
from live.lots import (
    LotLedger,
    add_lot,
    holding_period_blocked_symbols,
    holding_period_summary,
    initialise_lots,
    load_ledger,
    lot_ledger_path,
    remove_lots_fifo,
    save_ledger,
)
from live.notify import notify_run_complete
from live.runlog import (
    RunSummary,
    TradeRecord,
    finalise_summary,
    make_run_id,
    write_run,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NY_TZ = ZoneInfo("America/New_York")
DEFAULT_TIMEOUT_SECONDS = 90
DEFAULT_MIN_INTERVAL_DAYS = 31

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(_PROJECT_ROOT, "logs")


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class OrderExecutionError(RuntimeError):
    """Raised when a submitted order is not safely filled."""


# ---------------------------------------------------------------------------
# Plan dataclass
# ---------------------------------------------------------------------------

@dataclass
class RebalanceRow:
    """One planned rebalance action."""

    symbol: str
    target_weight: float
    current_weight: float
    target_value: float
    current_value: float
    delta_value: float
    action: str                 # "BUY" | "SELL" | "HOLD"
    qty: float | None = None
    notional: float | None = None
    reason: str = ""
    holding_blocked: bool = False   # Phase 3: True when lot is < 31 days old


# ---------------------------------------------------------------------------
# Strategy helpers
# ---------------------------------------------------------------------------

def _load_strategy_payload(strategy_id: str) -> dict[str, Any]:
    """Load one strategy definition from strategies.json."""
    strategies_path = os.path.join(_PROJECT_ROOT, "strategies.json")
    example_path = os.path.join(_PROJECT_ROOT, "strategies.example.json")
    if not os.path.exists(strategies_path):
        strategies_path = example_path
    with open(strategies_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
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
    """Block accidental trading of research / archived strategies."""
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
    """Return (allocation, ticker_mapping) for the strategy.

    allocation     : {tradable_symbol: weight}
    ticker_mapping : {backtest_ticker: tradable_symbol}
    """
    payload = _load_strategy_payload(strategy_id)
    allocation = payload["allocation"]
    live_tickers = payload.get("live_tickers", {})

    if not use_live_tickers:
        return dict(allocation), {t: t for t in allocation}

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


# ---------------------------------------------------------------------------
# Performance tracking
# ---------------------------------------------------------------------------

def _performance_csv_path(broker_name: str, trading_mode: str, account_label: str) -> str:
    safe_acct = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in account_label)
    return os.path.join(LOGS_DIR, f"performance_tracking_{broker_name}_{trading_mode}_{safe_acct}.csv")


def _performance_headers(allocation: dict[str, float]) -> list[str]:
    """Build CSV headers from tradable symbols."""
    weight_cols = [f"{s}_Weight%" for s in allocation]
    drift_cols = [f"{s}_Drift%" for s in allocation]
    return [
        "Date", "Portfolio_Equity",
        *weight_cols,
        *drift_cols,
        "Portfolio_Return%",
        "SPY_Return%", "ALLW_Return%", "60_40_Return%", "JEPQ_Return%",
    ]


def _ensure_csv_header_exists(csv_path: str, headers: list[str]) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as fh:
            existing = next(csv.reader(fh), [])
        if existing and existing != headers:
            raise ValueError(
                f"Performance CSV header mismatch in {csv_path}. "
                "Archive or migrate the existing CSV before changing symbols/broker."
            )
        return
    with open(csv_path, "w", newline="") as fh:
        csv.DictWriter(fh, fieldnames=headers).writeheader()


def _get_pct_change(current: float, previous: float) -> float:
    return round((current - previous) / previous * 100, 2) if previous > 0 else 0.0


def calculate_benchmark_returns(logger: logging.Logger) -> dict[str, float]:
    """Download MTD benchmark returns (SPY, ALLW, TLT, JEPQ) from yfinance."""
    _default = {"SPY_Return%": 0.0, "ALLW_Return%": 0.0, "60_40_Return%": 0.0, "JEPQ_Return%": 0.0}
    try:
        raw = yf.download(["SPY", "ALLW", "TLT", "JEPQ"], period="3mo", progress=False)
        closes = raw["Close"]
        if isinstance(closes, pd.Series):
            closes = closes.to_frame()
        closes = closes.dropna(how="all").ffill()
        if closes.empty:
            raise ValueError("Empty price data from yfinance")
        latest_date = closes.index[-1]
        prev_month = closes.loc[closes.index.to_period("M") < latest_date.to_period("M")]
        if prev_month.empty:
            raise ValueError("Not enough price history for benchmark returns")
        today_row = closes.iloc[-1]
        prior_row = prev_month.iloc[-1]

        def _safe_pct(col: str) -> float:
            try:
                return _get_pct_change(float(today_row[col]), float(prior_row[col]))
            except (KeyError, TypeError, ValueError):
                return 0.0

        spy = _safe_pct("SPY")
        allw = _safe_pct("ALLW")
        tlt = _safe_pct("TLT")
        jepq = _safe_pct("JEPQ")
        return {
            "SPY_Return%": spy,
            "ALLW_Return%": allw,
            "60_40_Return%": round(spy * 0.60 + tlt * 0.40, 2),
            "JEPQ_Return%": jepq,
        }
    except Exception as exc:
        logger.warning(f"Could not calculate benchmark returns: {exc}")
        return _default


def record_performance_snapshot(
    equity: float,
    positions: dict[str, PositionSnapshot],
    allocation: dict[str, float],
    logger: logging.Logger,
    csv_path: str,
) -> None:
    """Append one row to the performance-tracking CSV."""
    headers = _performance_headers(allocation)
    _ensure_csv_header_exists(csv_path, headers)

    actual = {
        sym: round(positions[sym].market_value / equity * 100 if sym in positions and equity > 0 else 0.0, 1)
        for sym in allocation
    }
    drift = {sym: round(actual.get(sym, 0.0) - w * 100, 2) for sym, w in allocation.items()}
    benchmarks = calculate_benchmark_returns(logger)

    portfolio_return = 0.0
    try:
        prev_equity = float(
            pd.read_csv(csv_path).iloc[-1]["Portfolio_Equity"]
            .replace("$", "").replace(",", "")
        )
        portfolio_return = round((equity - prev_equity) / prev_equity * 100, 2) if prev_equity > 0 else 0.0
    except (IndexError, KeyError, ValueError, AttributeError):
        pass

    row = {
        "Date": date.today().isoformat(),
        "Portfolio_Equity": f"${equity:,.2f}",
        **{f"{s}_Weight%": actual.get(s, 0) for s in allocation},
        **{f"{s}_Drift%": drift.get(s, 0) for s in allocation},
        "Portfolio_Return%": portfolio_return,
        **benchmarks,
    }
    try:
        with open(csv_path, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=headers).writerow(row)
        logger.info(
            f"Performance snapshot: {date.today().isoformat()} | "
            f"${equity:,.2f} ({portfolio_return:+.2f}%) | "
            f"SPY {benchmarks['SPY_Return%']:+.2f}% | "
            f"ALLW {benchmarks['ALLW_Return%']:+.2f}% | "
            f"60/40 {benchmarks['60_40_Return%']:+.2f}% | "
            f"JEPQ {benchmarks['JEPQ_Return%']:+.2f}%"
        )
    except Exception as exc:
        logger.error(f"Failed to record performance snapshot: {exc}")


# ---------------------------------------------------------------------------
# Cadence gate
# ---------------------------------------------------------------------------

def _cadence_state_path(
    broker_name: str, trading_mode: str, account_label: str, strategy_id: str
) -> str:
    safe_acct = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in account_label)
    safe_strat = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in strategy_id)
    fname = f"cadence_{broker_name}_{trading_mode}_{safe_acct}_{safe_strat}.json"
    return os.path.join(LOGS_DIR, fname)


def load_cadence_state(state_path: str) -> dict[str, Any]:
    if not os.path.exists(state_path):
        return {}
    with open(state_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_cadence_state(state_path: str, run_date: date) -> None:
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    state = {
        "last_run_date": run_date.isoformat(),
        "last_run_recorded_at": datetime.now(NY_TZ).isoformat(),
    }
    with open(state_path, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)


def is_cadence_due(state_path: str, interval_days: int) -> tuple[bool, str]:
    """Return (is_due, human-readable reason).

    is_due is True when enough days have elapsed since the last successful run
    or when no run has been recorded yet.
    """
    state = load_cadence_state(state_path)
    if not state:
        return True, "No previous run recorded — cadence is due immediately."

    last_run = date.fromisoformat(state["last_run_date"])
    next_due = last_run + timedelta(days=interval_days)
    today = date.today()

    if today >= next_due:
        return True, (
            f"Cadence due (last run: {last_run.isoformat()}, "
            f"interval: {interval_days}d, next: {next_due.isoformat()})"
        )
    days_left = (next_due - today).days
    return False, (
        f"Too early: {days_left}d remaining (last run: {last_run.isoformat()}, "
        f"next due: {next_due.isoformat()})"
    )


# ---------------------------------------------------------------------------
# Rebalance plan
# ---------------------------------------------------------------------------

def build_rebalance_plan(
    equity: float,
    positions: dict[str, PositionSnapshot],
    allocation: dict[str, float],
    asset_meta: dict[str, Any],
    drift_threshold: float,
    min_trade_value: float,
    cash_buffer_pct: float,
    liquidate_other_positions: bool,
    rebalance_mode: str = "per_asset",
) -> tuple[list[RebalanceRow], list[str]]:
    """Build the monthly rebalance plan.

    Parameters
    ----------
    equity                     : total equity to size against (account equity or managed budget)
    positions                  : {symbol: PositionSnapshot} from broker.get_positions()
    allocation                 : {symbol: target_weight}
    asset_meta                 : {symbol: AssetMetadata} from broker.get_asset()
    drift_threshold            : minimum absolute drift to generate a trade
    min_trade_value            : ignore trades below this USD amount
    cash_buffer_pct            : fraction of equity to keep in cash
    liquidate_other_positions  : sell non-strategy holdings
    rebalance_mode             : "per_asset" or "full_on_breach"

    Returns
    -------
    (rows, warnings)
    """
    investable = equity * (1.0 - cash_buffer_pct)
    if investable <= 0:
        raise ValueError("Investable equity is zero or negative. Check account funding.")

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

    _zero = lambda sym: PositionSnapshot(
        symbol=sym, qty=0.0, qty_available=0.0, market_value=0.0, current_price=0.0
    )

    for symbol, weight in allocation.items():
        current = positions.get(symbol, _zero(symbol))
        target_value = investable * weight
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
            # SELL
            if current.current_price <= 0 or current.qty_available <= 0:
                reason = "no sellable quantity"
            else:
                sell_qty = min(abs(delta_value) / current.current_price, current.qty_available)
                sell_qty = round(sell_qty, 6)
                if sell_qty <= 0:
                    reason = "sell qty rounded to zero"
                else:
                    action = "SELL"
                    qty = sell_qty
        else:
            # BUY
            meta = asset_meta.get(symbol)
            notional = round(delta_value, 2)
            if meta and not meta.fractionable:
                reason = "asset not fractionable — buy will convert to whole-share qty"
                warnings.append(
                    f"{symbol} is not fractionable. "
                    "Buy will be converted to whole-share quantity at execution."
                )
            if notional < min_trade_value:
                reason = "buy notional below minimum trade size"
            else:
                action = "BUY"

        rows.append(RebalanceRow(
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
        ))

    # Extra (non-strategy) positions
    for symbol in sorted(set(positions) - target_symbols):
        pos = positions[symbol]
        if pos.market_value < min_trade_value:
            continue
        if liquidate_other_positions:
            rows.append(RebalanceRow(
                symbol=symbol,
                target_weight=0.0,
                current_weight=pos.market_value / equity if equity > 0 else 0.0,
                target_value=0.0,
                current_value=pos.market_value,
                delta_value=-pos.market_value,
                action="SELL",
                qty=round(pos.qty_available, 6),
                reason="non-strategy position",
            ))
        else:
            warnings.append(
                f"Non-strategy position: {symbol} (${pos.market_value:,.2f}). "
                "Pass --liquidate-other-positions to sell."
            )

    rows.sort(key=lambda r: abs(r.delta_value), reverse=True)
    return rows, warnings


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------

def validate_order_guardrails(
    rows: list[RebalanceRow],
    equity: float,
    max_order_pct_equity: float,
    max_total_trade_pct_equity: float,
) -> None:
    """Reject plans where individual or total trade size is unreasonably large."""
    trade_rows = [r for r in rows if r.action in {"BUY", "SELL"}]
    if not trade_rows or equity <= 0:
        return
    max_order = equity * max_order_pct_equity
    max_total = equity * max_total_trade_pct_equity
    total = sum(abs(r.delta_value) for r in trade_rows)

    oversized = [r for r in trade_rows if abs(r.delta_value) > max_order]
    if oversized:
        details = ", ".join(f"{r.symbol} ${abs(r.delta_value):,.2f}" for r in oversized)
        raise SystemExit(
            f"Refusing to trade: orders exceed {max_order_pct_equity:.0%} of equity "
            f"(${max_order:,.2f}): {details}"
        )
    if total > max_total:
        raise SystemExit(
            f"Refusing to trade: total turnover ${total:,.2f} exceeds "
            f"{max_total_trade_pct_equity:.0%} of equity (${max_total:,.2f})."
        )


# ---------------------------------------------------------------------------
# Post-trade verification
# ---------------------------------------------------------------------------

def verify_post_trade_drift(
    equity: float,
    positions: dict[str, PositionSnapshot],
    allocation: dict[str, float],
    tolerance: float,
    min_extra_position_value: float,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Validate final weights against targets. Raises OrderExecutionError on breaches."""
    rows = []
    breaches = []

    _zero = lambda sym: PositionSnapshot(
        symbol=sym, qty=0.0, qty_available=0.0, market_value=0.0, current_price=0.0
    )

    for symbol, target_weight in allocation.items():
        current = positions.get(symbol, _zero(symbol))
        actual_weight = current.market_value / equity if equity > 0 else 0.0
        drift = actual_weight - target_weight
        rows.append({
            "Symbol": symbol,
            "Target %": round(target_weight * 100, 2),
            "Actual %": round(actual_weight * 100, 2),
            "Drift %": round(drift * 100, 2),
            "Market Value": round(current.market_value, 2),
        })
        if abs(drift) > tolerance:
            breaches.append(f"{symbol} drift {drift:+.2%}")

    for symbol in sorted(set(positions) - set(allocation)):
        pos = positions[symbol]
        if pos.market_value >= min_extra_position_value:
            breaches.append(f"non-strategy {symbol} ${pos.market_value:,.2f}")

    frame = pd.DataFrame(rows)
    logger.info("\nPost-trade verification:\n" + frame.to_string(index=False))
    if breaches:
        raise OrderExecutionError(
            "Post-trade verification failed: " + "; ".join(breaches)
        )
    return frame


# ---------------------------------------------------------------------------
# Plan display
# ---------------------------------------------------------------------------

def plan_to_frame(rows: list[RebalanceRow]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Symbol": r.symbol,
        "Action": r.action + (" [HOLD-BLOCKED]" if r.holding_blocked else ""),
        "Target %": round(r.target_weight * 100, 2),
        "Current %": round(r.current_weight * 100, 2),
        "Target $": round(r.target_value, 2),
        "Current $": round(r.current_value, 2),
        "Delta $": round(r.delta_value, 2),
        "Qty": r.qty,
        "Notional $": r.notional,
        "Reason": r.reason,
    } for r in rows])


# ---------------------------------------------------------------------------
# Order execution helpers
# ---------------------------------------------------------------------------

def _submit_buy_order(
    broker: Broker,
    symbol: str,
    notional: float,
    current_price: float,
    logger: logging.Logger,
) -> OrderResult | None:
    """Submit a buy order; fall back to qty if the broker rejects notional."""
    try:
        return broker.submit_market_order(symbol=symbol, side="BUY", notional=round(notional, 2))
    except NotImplementedError:
        logger.warning(
            f"Broker '{broker.name}' does not support notional orders; "
            f"converting {symbol} buy to whole-share qty"
        )
        if current_price <= 0:
            logger.error(f"Cannot compute qty for {symbol}: price is $0")
            return None
        qty = math.floor(notional / current_price)
        if qty <= 0:
            logger.warning(f"Buy qty for {symbol} rounds to 0 at ${current_price:.2f}; skipping")
            return None
        return broker.submit_market_order(symbol=symbol, side="BUY", qty=float(qty))


def fetch_latest_prices(symbols: list[str], logger: logging.Logger) -> dict[str, float]:
    """Fetch latest closes for quantity conversion when a broker rejects notional buys."""
    if not symbols:
        return {}
    try:
        unique = sorted(set(symbols))
        raw = yf.download(unique, period="5d", progress=False)
        closes = raw["Close"]
        if isinstance(closes, pd.Series):
            closes = closes.to_frame(name=unique[0])
        closes = closes.dropna(how="all").ffill()
        if closes.empty:
            return {}
        latest = closes.iloc[-1]
        out: dict[str, float] = {}
        for symbol in unique:
            if symbol in latest and pd.notna(latest[symbol]):
                price = float(latest[symbol])
                if price > 0:
                    out[symbol] = price
        return out
    except Exception as exc:
        logger.warning(f"Could not fetch latest prices for qty conversion: {exc}")
        return {}


def wait_for_fills(
    broker: Broker,
    order_ids: list[str],
    timeout_seconds: int,
    logger: logging.Logger,
) -> dict[str, OrderResult]:
    """Poll submitted orders until terminal state or timeout.

    Returns {order_id: final OrderResult}.  Raises OrderExecutionError if any
    order ended in a non-filled terminal status or timed out.
    """
    if not order_ids:
        return {}

    pending = set(order_ids)
    results: dict[str, OrderResult] = {}
    started = time.time()

    while pending and (time.time() - started) < timeout_seconds:
        completed: set[str] = set()
        for oid in list(pending):
            result = broker.get_order(oid)
            if result.status in TERMINAL_ORDER_STATUSES:
                logger.info(f"Order {oid} -> {result.status}")
                completed.add(oid)
                results[oid] = result
        pending -= completed
        if pending:
            time.sleep(2)

    for oid in pending:
        try:
            result = broker.get_order(oid)
            results[oid] = result
        except Exception:
            results[oid] = OrderResult(
                order_id=oid,
                symbol="",
                side="",
                qty=None,
                notional=None,
                submitted_at=datetime.now(NY_TZ),
                status="timeout",
            )
        logger.error(f"Order {oid} timed out (status: {results[oid].status})")

    failed = {oid: r.status for oid, r in results.items() if r.status != "filled"}
    if failed:
        details = ", ".join(f"{oid}:{s}" for oid, s in failed.items())
        raise OrderExecutionError(f"Order execution failed: {details}")
    return results


def _simulate_fills(
    rows: list[RebalanceRow],
    positions: dict[str, PositionSnapshot],
    logger: logging.Logger,
) -> None:
    """Print what dry-execute would do — no real orders are placed."""
    trade_rows = [r for r in rows if r.action in {"BUY", "SELL"} and not r.holding_blocked]
    if not trade_rows:
        logger.info("DRY-EXECUTE: no trades would be submitted.")
        return
    for r in trade_rows:
        if r.action == "SELL":
            price = positions[r.symbol].current_price if r.symbol in positions else 0.0
            logger.info(
                f"DRY-EXECUTE SELL  {r.symbol:<6}  qty={r.qty}  "
                f"~${(r.qty or 0) * price:,.2f}"
            )
        else:
            if r.notional:
                logger.info(f"DRY-EXECUTE BUY   {r.symbol:<6}  notional=${r.notional:,.2f}")
            elif r.qty:
                price = positions[r.symbol].current_price if r.symbol in positions else 0.0
                logger.info(
                    f"DRY-EXECUTE BUY   {r.symbol:<6}  qty={r.qty}  "
                    f"~${(r.qty or 0) * price:,.2f}"
                )


def execute_rebalance(
    broker: Broker,
    initial_rows: list[RebalanceRow],
    allocation: dict[str, float],
    asset_meta: dict[str, Any],
    drift_threshold: float,
    min_trade_value: float,
    cash_buffer_pct: float,
    liquidate_other_positions: bool,
    timeout_seconds: int,
    logger: logging.Logger,
    sizing_equity: float,
    rebalance_mode: str = "per_asset",
) -> list[OrderResult]:
    """Submit orders: sells first, refresh account, then buys."""
    sells = [r for r in initial_rows if r.action == "SELL" and r.qty and r.qty > 0
             and not r.holding_blocked]

    filled_orders: list[OrderResult] = []
    sell_ids: list[str] = []
    for r in sells:
        result = broker.submit_market_order(symbol=r.symbol, side="SELL", qty=r.qty)
        sell_ids.append(result.order_id)
        logger.info(f"SELL {r.symbol:<6} qty={r.qty}")
    if sell_ids:
        logger.info(f"Waiting for {len(sell_ids)} sell orders (timeout {timeout_seconds}s)...")
        filled_orders.extend(wait_for_fills(broker, sell_ids, timeout_seconds, logger).values())

    # Refresh account snapshot after sells
    logger.info("Refreshing account snapshot after sells...")
    account = broker.get_account()
    positions = broker.get_positions()
    logger.info(
        f"Refreshed: account equity=${account.equity:,.2f}  "
        f"sizing equity=${sizing_equity:,.2f}  cash=${account.cash:,.2f}"
    )

    price_lookup = {
        sym: pos.current_price
        for sym, pos in positions.items()
        if pos.current_price > 0
    }
    missing_prices = [sym for sym in allocation if sym not in price_lookup]
    price_lookup.update(fetch_latest_prices(missing_prices, logger))

    refreshed_rows, warnings = build_rebalance_plan(
        equity=sizing_equity,
        positions=positions,
        allocation=allocation,
        asset_meta=asset_meta,
        drift_threshold=drift_threshold,
        min_trade_value=min_trade_value,
        cash_buffer_pct=cash_buffer_pct,
        liquidate_other_positions=liquidate_other_positions,
        rebalance_mode=rebalance_mode,
    )
    for w in warnings:
        logger.warning(w)

    buys = [r for r in refreshed_rows if r.action == "BUY" and not r.holding_blocked
            and (r.notional and r.notional > 0 or r.qty and r.qty > 0)]

    buy_ids: list[str] = []
    for r in buys:
        current_price = price_lookup.get(r.symbol, 0.0)
        result = _submit_buy_order(
            broker=broker,
            symbol=r.symbol,
            notional=r.notional or 0.0,
            current_price=current_price,
            logger=logger,
        )
        if result is None:
            logger.warning(f"Skipping buy for {r.symbol} — could not compute valid order")
            continue
        buy_ids.append(result.order_id)
        if result.notional:
            logger.info(f"BUY  {r.symbol:<6} notional=${result.notional:,.2f}")
        else:
            logger.info(f"BUY  {r.symbol:<6} qty={result.qty}")

    if buy_ids:
        logger.info(f"Waiting for {len(buy_ids)} buy orders (timeout {timeout_seconds}s)...")
        filled_orders.extend(wait_for_fills(broker, buy_ids, timeout_seconds, logger).values())
    else:
        logger.info("No buy orders required after sells.")
    return filled_orders


# ---------------------------------------------------------------------------
# Run registry (duplicate-execution guard)
# ---------------------------------------------------------------------------

_RUN_REGISTRY_PATH = os.path.join(LOGS_DIR, "rebalance_run_registry.json")


def _run_key(run_date: date, broker: str, mode: str, account: str, strategy_id: str) -> str:
    return f"{run_date.isoformat()}|{broker}|{mode}|{account}|{strategy_id}"


def assert_not_duplicate_run(
    run_date: date,
    broker_name: str,
    trading_mode: str,
    account_label: str,
    strategy_id: str,
    allow_duplicate: bool,
    logger: logging.Logger,
) -> None:
    if allow_duplicate or not os.path.exists(_RUN_REGISTRY_PATH):
        return
    with open(_RUN_REGISTRY_PATH, "r", encoding="utf-8") as fh:
        registry = json.load(fh)
    key = _run_key(run_date, broker_name, trading_mode, account_label, strategy_id)
    if key in registry:
        msg = (
            f"Refusing duplicate execution for {key}. "
            "Use --allow-duplicate-run after confirming no orders are pending."
        )
        logger.error(msg)
        raise SystemExit(msg)


def record_successful_run(
    run_date: date,
    broker_name: str,
    trading_mode: str,
    account_label: str,
    strategy_id: str,
    logger: logging.Logger,
) -> None:
    os.makedirs(os.path.dirname(_RUN_REGISTRY_PATH), exist_ok=True)
    if os.path.exists(_RUN_REGISTRY_PATH):
        with open(_RUN_REGISTRY_PATH, "r", encoding="utf-8") as fh:
            registry = json.load(fh)
    else:
        registry = {}
    key = _run_key(run_date, broker_name, trading_mode, account_label, strategy_id)
    registry[key] = {
        "recorded_at": datetime.now(NY_TZ).isoformat(),
        "broker": broker_name,
        "trading_mode": trading_mode,
        "account": account_label,
        "strategy_id": strategy_id,
    }
    with open(_RUN_REGISTRY_PATH, "w", encoding="utf-8") as fh:
        json.dump(registry, fh, indent=2, sort_keys=True)
    logger.info(f"Run registry updated: {key}")


def has_prior_preview(
    logs_dir: str,
    broker_name: str,
    trading_mode: str,
    account_label: str,
    strategy_id: str,
) -> bool:
    """Return True if this live identity has already produced a preview/dry run."""
    path = os.path.join(logs_dir, "run_summary.jsonl")
    if not os.path.exists(path):
        return False
    try:
        canonical_strategy = config.resolve_strategy_id(strategy_id)
    except Exception:
        canonical_strategy = strategy_id
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row_strategy = row.get("strategy_id", "")
            try:
                row_strategy = config.resolve_strategy_id(row_strategy)
            except Exception:
                pass
            if (
                row.get("broker") == broker_name
                and row.get("trading_mode") == trading_mode
                and row.get("account_label") == account_label
                and row_strategy == canonical_strategy
                and row.get("outcome") in {"preview", "dry_execute"}
            ):
                return True
    return False


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(broker_name: str, trading_mode: str) -> logging.Logger:
    os.makedirs(LOGS_DIR, exist_ok=True)
    now_et = datetime.now(NY_TZ)
    log_filename = now_et.strftime(f"%Y-%m-%d_%H-%M-%S_{broker_name}_{trading_mode}_rebalance.log")
    log_path = os.path.join(LOGS_DIR, log_filename)

    logger = logging.getLogger("rebalancer")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S"))

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Log: {log_path}")
    return logger


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Broker-agnostic ETF rebalancer (≥31-day cadence).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Trading environment
    mode_grp = parser.add_mutually_exclusive_group()
    mode_grp.add_argument("--paper", action="store_true",
                          help="Use paper / simulation trading (default).")
    mode_grp.add_argument("--live", action="store_true",
                          help="Use live trading. Requires live credentials.")

    # Broker
    parser.add_argument("--broker", default="alpaca",
                        choices=["alpaca", "tastytrade"],
                        help="Broker to use.")
    parser.add_argument("--account", default=None,
                        help="Named account label for credential env vars.")

    # Strategy
    parser.add_argument("--strategy-id", default=config.DEFAULT_STRATEGY,
                        help="Strategy id from strategies.json.")
    parser.add_argument("--allow-non-production-strategy", action="store_true",
                        help="Allow trading a strategy not marked for live trading.")

    # Execution
    parser.add_argument("--execute", action="store_true",
                        help="Submit real orders. Without this flag: preview only.")
    parser.add_argument("--dry-execute", action="store_true",
                        help=(
                            "Simulate fills using current price without placing real orders. "
                            "Useful for testing logs and the rebalance plan."
                        ))
    parser.add_argument("--force", action="store_true",
                        help="Bypass the cadence gate. Market-open guard still applies.")
    parser.add_argument("--allow-duplicate-run", action="store_true",
                        help="Allow a second execution for the same date/account/strategy.")

    # Cadence
    parser.add_argument("--min-rebalance-interval-days", type=int,
                        default=DEFAULT_MIN_INTERVAL_DAYS,
                        help="Minimum days between consecutive executed runs.")
    parser.add_argument("--force-cadence", action="store_true",
                        help="Bypass the minimum-interval cadence check.")

    # Budget (Phase 2 stubs)
    parser.add_argument("--budget", type=float, default=None,
                        help=(
                            "Managed capital cap in USD. When set, sizes orders against this "
                            "amount rather than full account equity."
                        ))
    parser.add_argument("--initialize-budget", action="store_true",
                        help="Seed budget state from current positions (first-run setup).")
    parser.add_argument("--ignore-budget", action="store_true",
                        help="Always use full account equity, even if a budget is configured.")

    # Lots / holding period (Phase 3 stubs)
    parser.add_argument("--initialize-lots", action="store_true",
                        help=(
                            "Seed the lot ledger from current positions "
                            "(all treated as immediately sellable)."
                        ))

    # Rebalance parameters
    parser.add_argument("--drift-threshold", type=float,
                        default=config.REBALANCE_THRESHOLD,
                        help="Minimum absolute weight drift to trigger a trade.")
    parser.add_argument("--rebalance-mode",
                        choices=["per_asset", "full_on_breach"],
                        default=config.REBALANCE_MODE,
                        help=(
                            "per_asset: trade only breaching assets. "
                            "full_on_breach: trade all assets when any breaches."
                        ))
    parser.add_argument("--min-trade-value", type=float, default=25.0,
                        help="Ignore trades smaller than this USD amount.")
    parser.add_argument("--cash-buffer-pct", type=float, default=0.005,
                        help="Fraction of equity to keep as cash buffer.")
    parser.add_argument("--liquidate-other-positions", action="store_true",
                        help="Sell positions not in the selected strategy.")

    # Ticker mode
    ticker_grp = parser.add_mutually_exclusive_group()
    ticker_grp.add_argument("--use-live-tickers", action="store_true",
                             dest="use_live_tickers",
                             help="Translate backtest tickers via strategy live_tickers (default).")
    ticker_grp.add_argument("--use-backtest-tickers", action="store_false",
                             dest="use_live_tickers",
                             help="Trade backtest tickers directly.")
    parser.set_defaults(use_live_tickers=True)

    # Order parameters
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS,
                        help="Seconds to wait for order fills.")
    parser.add_argument("--post-trade-tolerance", type=float, default=0.015,
                        help="Maximum allowed absolute post-trade weight drift (1.5%%).")
    parser.add_argument("--max-order-pct-equity", type=float, default=0.40,
                        help="Refuse single orders above this fraction of equity.")
    parser.add_argument("--max-total-trade-pct-equity", type=float, default=1.05,
                        help="Refuse total turnover above this fraction of equity.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    trading_mode = "live" if args.live else "paper"
    account_label = args.account or "default"

    logger = setup_logging(args.broker, trading_mode)

    logger.info("=" * 72)
    logger.info(f"ALL-WEATHER REBALANCER  broker={args.broker}  mode={trading_mode.upper()}")
    logger.info("=" * 72)

    is_dry = args.dry_execute
    is_execute = args.execute

    if is_dry and is_execute:
        raise SystemExit("--dry-execute and --execute are mutually exclusive.")

    from datetime import datetime as _dt, timezone as _tz
    _started_at = _dt.now(_tz.utc)
    _run_id = make_run_id(_started_at)
    run_summary = RunSummary(
        run_id=_run_id,
        broker=args.broker,
        trading_mode=trading_mode,
        account_label=account_label,
        strategy_id=args.strategy_id,
        started_at=_started_at,
        is_execute=is_execute,
        is_dry_execute=is_dry,
    )
    del _dt, _tz  # cleanup locals

    try:
        # ---- Strategy -------------------------------------------------------
        logger.info(f"Loading strategy: {args.strategy_id}")
        payload = _load_strategy_payload(args.strategy_id)
        _assert_strategy_live_allowed(
            strategy_id=args.strategy_id,
            payload=payload,
            allow_non_production=args.allow_non_production_strategy,
        )
        allocation, mapping = _resolve_target_allocation(args.strategy_id, args.use_live_tickers)
        display_name = payload.get("display_name") or payload.get("description") or args.strategy_id
        logger.info(f"Strategy: {display_name} ({payload.get('_strategy_id')}), {len(allocation)} assets")

        # ---- Broker ---------------------------------------------------------
        logger.info(f"Connecting to {args.broker} ({trading_mode})...")
        broker = make_broker(
            broker_name=args.broker,
            trading_mode=trading_mode,
            account_label=account_label,
        )
        logger.info(f"Broker connected: {broker.name}")

        # ---- Cadence check --------------------------------------------------
        cadence_path = _cadence_state_path(
            args.broker, trading_mode, account_label, args.strategy_id
        )
        # Always compute cadence status so the preview block can display it.
        # Enforcement only fires on real --execute without override flags.
        cadence_due, cadence_msg = is_cadence_due(
            cadence_path, args.min_rebalance_interval_days
        )
        logger.info(f"Cadence: {cadence_msg}")
        if is_execute and not args.force_cadence and not args.force:
            if not cadence_due:
                raise SystemExit(
                    f"Refusing to execute: {cadence_msg}. "
                    "Use --force or --force-cadence to override."
                )
        elif is_dry:
            logger.info("Dry-execute: cadence gate bypassed (no cadence will be updated).")

        # ---- Asset metadata -------------------------------------------------
        logger.info("Validating target assets...")
        asset_meta = {}
        for symbol in allocation:
            meta = broker.get_asset(symbol)
            if not meta.tradable:
                raise SystemExit(f"Asset {symbol} is not tradable with {args.broker}.")
            asset_meta[symbol] = meta
        logger.info(f"All {len(asset_meta)} target assets validated.")

        # ---- Market status --------------------------------------------------
        market_open = broker.is_market_open()
        logger.info(f"Market open: {market_open}")

        # ---- Account snapshot -----------------------------------------------
        logger.info("Fetching account snapshot...")
        account = broker.get_account()
        positions = broker.get_positions()
        equity = account.equity

        # ---- Budget ---------------------------------------------------------
        budget_state_path = _budget_state_path(
            LOGS_DIR, args.broker, trading_mode, account_label, args.strategy_id
        )
        if args.initialize_budget:
            if args.budget is None:
                raise SystemExit("--initialize-budget requires --budget AMOUNT")
            if load_budget(budget_state_path, positions, allocation, logger) is not None and not args.force:
                raise SystemExit(
                    f"Budget state already exists at {budget_state_path}. "
                    "Use --force only after manual review if you intend to overwrite it."
                )
            snap = initialise_budget(budget_state_path, args.budget, positions, allocation, logger)
            logger.info("Budget initialised. Re-run without --initialize-budget to rebalance.")
            return
        budget_snap: BudgetSnapshot | None
        managed_capital, budget_snap = get_managed_capital(
            state_path=budget_state_path,
            budget_cap=args.budget,
            account_equity=equity,
            positions=positions,
            allocation=allocation,
            broker=broker,
            ignore_budget=args.ignore_budget,
            logger=logger,
        )

        logger.info(f"Sizing capital: ${managed_capital:,.2f}  cash=${account.cash:,.2f}  "
                    f"buying_power=${account.buying_power:,.2f}")

        # ---- Rebalance plan -------------------------------------------------
        logger.info("Building rebalance plan...")
        rows, warnings = build_rebalance_plan(
            equity=managed_capital,
            positions=positions,
            allocation=allocation,
            asset_meta=asset_meta,
            drift_threshold=args.drift_threshold,
            min_trade_value=args.min_trade_value,
            cash_buffer_pct=args.cash_buffer_pct,
            liquidate_other_positions=args.liquidate_other_positions,
            rebalance_mode=args.rebalance_mode,
        )
        # ---- Lot ledger / holding-period gate -------------------------------
        ledger_path = lot_ledger_path(
            LOGS_DIR, args.broker, trading_mode, account_label, args.strategy_id
        )
        if args.initialize_lots:
            initialise_lots(
                ledger_path,
                positions,
                allocation,
                assume_held_days=args.min_rebalance_interval_days + 1,
            )
            logger.info(f"Lot ledger initialised from current positions → {ledger_path}")

        ledger = load_ledger(ledger_path)
        blocked_syms = holding_period_blocked_symbols(
            ledger, list(allocation.keys()), min_hold_days=args.min_rebalance_interval_days
        )
        if blocked_syms:
            for row in rows:
                if row.symbol in blocked_syms and row.action == "SELL":
                    row.action = "HOLD"
                    row.holding_blocked = True
                    row.reason = (
                        f"holding-period blocked (<{args.min_rebalance_interval_days}d)"
                    )
            logger.info(
                f"Holding-period blocked sells: {', '.join(sorted(blocked_syms))}"
            )
        lot_summary = holding_period_summary(
            ledger, list(allocation.keys()), args.min_rebalance_interval_days
        )

        # ---- Print preview --------------------------------------------------
        now_et = datetime.now(NY_TZ)
        print("=" * 72)
        print(f"ALL-WEATHER REBALANCER  broker={args.broker.upper()}  {trading_mode.upper()}")
        print("=" * 72)
        print(f"Time (ET):   {now_et.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"Strategy:    {args.strategy_id}")
        print(f"Account:     {account_label}")
        print(f"Equity:      ${equity:,.2f}   Sizing capital: ${managed_capital:,.2f}")
        print(f"Cash:        ${account.cash:,.2f}   Buying power: ${account.buying_power:,.2f}")
        print(f"Market open: {market_open}")
        print(f"Mode:        {'DRY-EXECUTE' if is_dry else 'EXECUTE' if is_execute else 'PREVIEW ONLY'}")
        print(f"Drift thr:   {args.drift_threshold:.1%}   Rebalance mode: {args.rebalance_mode}")

        # Cadence gate
        cadence_label = "ELIGIBLE" if cadence_due else "BLOCKED"
        print(f"Cadence:     {cadence_label} — {cadence_msg}")

        # Budget snapshot
        if budget_snap is not None:
            print(
                f"Budget:      cap ${budget_snap.budget_cap:,.2f}  "
                f"positions ${budget_snap.positions_value:,.2f}  "
                f"reserved ${budget_snap.reserved_cash:,.2f}  "
                f"managed ${budget_snap.managed_capital:,.2f}"
            )
        elif args.budget is not None:
            print(f"Budget:      ${args.budget:,.2f} requested — state not initialised (run --initialize-budget)")
        else:
            print("Budget:      none — sizing against full account equity")

        # Lot-ledger summary (held days, blocked status)
        blocked_rows = [r for r in lot_summary if r.get("blocked")]
        unblocked_rows = [r for r in lot_summary if not r.get("blocked") and r.get("days_held") is not None]
        if blocked_rows:
            blocked_str = ", ".join(
                f"{r['symbol']} ({r['note']})" for r in blocked_rows
            )
            print(f"Held lots:   {len(blocked_rows)} BLOCKED — {blocked_str}")
        if unblocked_rows:
            ages = ", ".join(
                f"{r['symbol']} ({r['days_held']}d)" for r in unblocked_rows
            )
            print(f"             eligible — {ages}")
        if not blocked_rows and not unblocked_rows:
            print("Held lots:   no ledger entries yet (consider --initialize-lots)")

        if args.use_live_tickers:
            remap = {bt: tr for bt, tr in mapping.items() if bt != tr}
            if remap:
                print("\nTicker mapping:")
                for bt, tr in remap.items():
                    print(f"  {bt} -> {tr}")

        if warnings:
            print("\nWarnings:")
            for w in warnings:
                print(f"  - {w}")
                logger.warning(w)

        print("\nRebalance plan:")
        frame = plan_to_frame(rows)
        if frame.empty:
            print("  No positions found.")
        else:
            print(frame.to_string(index=False))

        # ---- Dry-execute ----------------------------------------------------
        if is_dry:
            print("\n[DRY-EXECUTE] Simulating fills — no real orders placed.")
            logger.info("[DRY-EXECUTE] Simulating fills — no real orders placed.")
            _simulate_fills(rows, positions, logger)
            print("\nDry-execute complete. Run with --execute to place real orders.")
            run_summary.equity_before = equity
            run_summary.managed_capital = managed_capital
            run_summary.cash_before = account.cash
            run_summary.n_buy = sum(1 for r in rows if r.action == "BUY")
            run_summary.n_sell = sum(1 for r in rows if r.action == "SELL")
            run_summary.n_hold = sum(1 for r in rows if r.action == "HOLD" and not r.holding_blocked)
            run_summary.n_holding_blocked = sum(1 for r in rows if r.holding_blocked)
            run_summary.total_buy_notional = sum(r.notional or 0 for r in rows if r.action == "BUY")
            run_summary.total_sell_notional = sum(abs(r.delta_value) for r in rows if r.action == "SELL")
            run_summary.warnings = warnings
            finalise_summary(run_summary, outcome="dry_execute", equity_after=equity)
            write_run(LOGS_DIR, run_summary)
            notify_run_complete(run_summary, logger=logger)
            return

        # ---- Preview-only exit ----------------------------------------------
        if not is_execute:
            print("\nPreview complete. Re-run with --execute to submit orders.")
            logger.info("Preview mode: no orders submitted.")
            run_summary.equity_before = equity
            run_summary.managed_capital = managed_capital
            run_summary.warnings = warnings
            finalise_summary(run_summary, outcome="preview", equity_after=equity)
            write_run(LOGS_DIR, run_summary)
            return

        # ---- Execute guards -------------------------------------------------
        if trading_mode == "live" and not has_prior_preview(
            LOGS_DIR, args.broker, trading_mode, account_label, args.strategy_id
        ):
            raise SystemExit(
                "Refusing live execution: run a preview or dry-execute first "
                "for this broker/account/strategy."
            )
        if args.budget is not None and budget_snap is None and not args.ignore_budget:
            raise SystemExit(
                f"Refusing budgeted execution without budget state at {budget_state_path}. "
                "Run --initialize-budget first."
            )
        if not market_open:
            raise SystemExit(
                "Refusing to submit orders: market is closed. "
                "Wait for the regular session or use --dry-execute."
            )

        assert_not_duplicate_run(
            run_date=date.today(),
            broker_name=args.broker,
            trading_mode=trading_mode,
            account_label=account_label,
            strategy_id=args.strategy_id,
            allow_duplicate=args.allow_duplicate_run,
            logger=logger,
        )

        open_orders = broker.get_open_orders(list(allocation.keys()))
        if open_orders:
            details = ", ".join(
                f"{o.symbol}:{o.order_id}:{o.status}" for o in open_orders
            )
            raise SystemExit(f"Open orders exist for target symbols: {details}")

        validate_order_guardrails(
            rows=rows,
            equity=managed_capital,
            max_order_pct_equity=args.max_order_pct_equity,
            max_total_trade_pct_equity=args.max_total_trade_pct_equity,
        )

        # ---- Execute --------------------------------------------------------
        logger.info("All pre-execution checks passed. Beginning execution...")
        filled_orders = execute_rebalance(
            broker=broker,
            initial_rows=rows,
            allocation=allocation,
            asset_meta=asset_meta,
            drift_threshold=args.drift_threshold,
            min_trade_value=args.min_trade_value,
            cash_buffer_pct=args.cash_buffer_pct,
            liquidate_other_positions=args.liquidate_other_positions,
            timeout_seconds=args.timeout_seconds,
            logger=logger,
            sizing_equity=managed_capital,
            rebalance_mode=args.rebalance_mode,
        )
        logger.info("Execution complete.")

        # ---- Post-trade verification ----------------------------------------
        final_account = broker.get_account()
        final_positions = broker.get_positions()
        final_equity = final_account.equity
        verification = verify_post_trade_drift(
            equity=managed_capital,
            positions=final_positions,
            allocation=allocation,
            tolerance=args.post_trade_tolerance,
            min_extra_position_value=args.min_trade_value,
            logger=logger,
        )
        print("\nPost-trade verification:")
        print(verification.to_string(index=False))

        # ---- Record performance + cadence -----------------------------------
        performance_path = _performance_csv_path(args.broker, trading_mode, account_label)
        record_performance_snapshot(
            equity=final_equity,
            positions=final_positions,
            allocation=allocation,
            logger=logger,
            csv_path=performance_path,
        )
        print(f"Performance snapshot: {performance_path}")

        save_cadence_state(cadence_path, date.today())
        logger.info(f"Cadence state updated: {cadence_path}")

        record_successful_run(
            run_date=date.today(),
            broker_name=args.broker,
            trading_mode=trading_mode,
            account_label=account_label,
            strategy_id=args.strategy_id,
            logger=logger,
        )

        # Update lot ledger: record buys, remove sells
        for order in filled_orders:
            qty = order.filled_qty or order.qty or 0.0
            if qty <= 0:
                continue
            if order.side == "BUY":
                price = order.filled_avg_price or final_positions.get(order.symbol, PositionSnapshot(
                    order.symbol, 0, 0, 0, 0)).current_price
                add_lot(ledger, order.symbol, qty, price, date.today())
            elif order.side == "SELL":
                remove_lots_fifo(ledger, order.symbol, qty)
        save_ledger(ledger_path, ledger)
        logger.info(f"Lot ledger updated: {ledger_path}")

        # Update budget state if active
        if budget_snap is not None:
            budget_snap = reconcile_after_trades(
                budget_snap,
                final_positions,
                allocation,
                managed_capital,
            )
            save_budget_state(budget_state_path, budget_snap)
            logger.info("Budget state saved.")

        # RunSummary + notifications
        run_summary.equity_before = equity
        run_summary.managed_capital = managed_capital
        run_summary.cash_before = account.cash
        run_summary.equity_after = final_equity
        run_summary.n_buy = sum(1 for r in rows if r.action == "BUY")
        run_summary.n_sell = sum(1 for r in rows if r.action == "SELL")
        run_summary.n_hold = sum(1 for r in rows if r.action == "HOLD" and not r.holding_blocked)
        run_summary.n_holding_blocked = sum(1 for r in rows if r.holding_blocked)
        run_summary.total_buy_notional = sum(r.notional or 0 for r in rows if r.action == "BUY")
        run_summary.total_sell_notional = sum(abs(r.delta_value) for r in rows if r.action == "SELL")
        run_summary.warnings = warnings
        run_summary.trades = [
            TradeRecord(
                symbol=o.symbol,
                side=o.side,
                action_type="notional" if o.notional is not None else "qty",
                notional=o.notional,
                qty=o.qty,
                filled_qty=o.filled_qty,
                filled_avg_price=o.filled_avg_price,
                order_id=o.order_id,
                status=o.status,
            )
            for o in filled_orders
        ]
        finalise_summary(run_summary, outcome="executed", equity_after=final_equity)
        write_run(LOGS_DIR, run_summary)
        notify_run_complete(run_summary, logger=logger)

        print("\n✓ Rebalance complete.")

    except SystemExit as exc:
        try:
            finalise_summary(run_summary, outcome="skipped", error_message=str(exc))
            write_run(LOGS_DIR, run_summary)
            notify_run_complete(run_summary, logger=logger)
        except Exception:
            pass
        raise
    except Exception as exc:
        logger.exception(f"Fatal error: {exc}")
        try:
            finalise_summary(run_summary, outcome="error", error_message=str(exc))
            write_run(LOGS_DIR, run_summary)
            notify_run_complete(run_summary, logger=logger)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
