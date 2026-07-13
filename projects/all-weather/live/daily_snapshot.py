"""
live/daily_snapshot.py
======================
Lightweight daily price + portfolio-drift monitor.

Fetches current market prices via yfinance (no broker credentials needed),
estimates portfolio weights by multiplying the last-known positions by today's
prices, and appends a row to ``live/logs/daily_snapshots.csv``.

Why this exists
---------------
The rebalancer's ``performance_tracking.csv`` is only written on execute/dry-execute
runs (monthly at best).  This script runs daily — weekdays after market close —
and builds a fine-grained time series of:
  * estimated portfolio equity
  * per-ticker current weight
  * per-ticker target weight
  * per-ticker weight drift (current − target)
  * benchmark prices (SPY, ALLW if available, JEPQ)

This data feeds the shadow-comparison analysis (research/shadow_comparison/).

Usage
-----
Preview (no writes):
    python -m live.daily_snapshot --paper --broker alpaca --dry-run

Append row (runs normally):
    python -m live.daily_snapshot --paper --broker alpaca

Flags
-----
  --paper / --live     Trading mode for broker auth (default: --paper)
  --broker             alpaca | tastytrade (default: alpaca)
  --account            account label (default: default)
  --strategy-id        strategy id (default: from config)
  --dry-run            Print the row that would be written; do not write
  --no-broker          Skip broker connection; estimate weights from last
                       snapshot instead.  Useful for quick price checks.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from engine import config

_NY_TZ = ZoneInfo("America/New_York")
_LOGS_DIR = _PROJECT_ROOT / "live" / "logs"
_SNAPSHOT_CSV = _LOGS_DIR / "daily_snapshots.csv"

_BENCHMARKS = ["SPY", "ALLW", "JEPQ"]

_CSV_COLUMNS = [
    "Date", "Time_ET", "Strategy",
    "Equity_Est", "Cash_Est",
    # per-ticker columns are appended dynamically: <TICKER>_price, <TICKER>_weight, <TICKER>_target, <TICKER>_drift
    "SPY_price", "ALLW_price", "JEPQ_price",
    "Broker_Connected", "Source",
]


# ---------------------------------------------------------------------------
# Price fetch
# ---------------------------------------------------------------------------

def _fetch_prices(tickers: list[str]) -> dict[str, float]:
    """Return the latest close (or intraday last) for each ticker via yfinance."""
    prices: dict[str, float] = {}
    try:
        data = yf.download(
            tickers, period="2d", auto_adjust=True, progress=False, threads=False
        )
        closes = data["Close"] if "Close" in data else data
        if isinstance(closes, pd.Series):
            closes = closes.to_frame(tickers[0])
        for t in tickers:
            if t in closes.columns:
                val = closes[t].dropna()
                if not val.empty:
                    prices[t] = float(val.iloc[-1])
    except Exception as exc:
        print(f"  [snapshot] price fetch warning: {exc}")
    return prices


# ---------------------------------------------------------------------------
# Broker-connected path: read actual positions
# ---------------------------------------------------------------------------

def _positions_from_broker(
    broker: str, trading_mode: str, account_label: str
) -> tuple[dict[str, float], float] | None:
    """Return ({ticker: qty}, cash) from the broker, or None on failure."""
    try:
        from live.env import load_api_keys_env
        from live.brokers.factory import make_broker
        load_api_keys_env()
        b = make_broker(
            broker_name=broker,
            trading_mode=trading_mode,
            account_label=account_label,
        )
        snap = b.get_account()
        positions = {p.symbol: p.qty for p in b.get_positions().values()}
        cash = snap.cash
        return positions, cash
    except Exception as exc:
        print(f"  [snapshot] broker connection failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# No-broker path: estimate from last snapshot row
# ---------------------------------------------------------------------------

def _positions_from_last_snapshot(
    tickers: list[str],
    prices: dict[str, float],
) -> tuple[dict[str, float], float] | None:
    """Estimate quantities from the most recent snapshot row using today's prices."""
    if not _SNAPSHOT_CSV.exists():
        return None
    try:
        df = pd.read_csv(_SNAPSHOT_CSV)
        if df.empty:
            return None
        last = df.iloc[-1]
        # We have <TICKER>_weight * Equity_Est → value per ticker → qty = value / price
        equity = float(last.get("Equity_Est", 0.0))
        if equity <= 0:
            return None
        positions: dict[str, float] = {}
        for t in tickers:
            w_col = f"{t}_weight"
            if w_col in last and pd.notna(last[w_col]) and t in prices and prices[t] > 0:
                val = float(last[w_col]) * equity
                positions[t] = val / prices[t]
        cash_est = float(last.get("Cash_Est", 0.0))
        return positions, cash_est
    except Exception as exc:
        print(f"  [snapshot] last-snapshot estimation failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Core snapshot logic
# ---------------------------------------------------------------------------

def build_snapshot_row(
    strategy_id: str,
    prices: dict[str, float],
    positions: dict[str, float],
    cash: float,
    allocation: dict[str, float],
    broker_connected: bool,
) -> dict:
    """Build one CSV row from prices + positions + allocation."""
    now = datetime.now(tz=_NY_TZ)

    # Estimate equity
    position_value = sum(
        positions.get(t, 0.0) * prices.get(t, 0.0) for t in allocation
    )
    equity_est = position_value + cash

    # Per-ticker columns
    row: dict = {
        "Date":              now.strftime("%Y-%m-%d"),
        "Time_ET":           now.strftime("%H:%M:%S"),
        "Strategy":          strategy_id,
        "Equity_Est":        round(equity_est, 2),
        "Cash_Est":          round(cash, 2),
        "Broker_Connected":  broker_connected,
        "Source":            "broker" if broker_connected else "estimated",
    }

    for t in allocation:
        price  = prices.get(t, float("nan"))
        val    = positions.get(t, 0.0) * price if pd.notna(price) else 0.0
        weight = (val / equity_est) if equity_est > 0 else 0.0
        target = allocation.get(t, 0.0)
        drift  = weight - target

        row[f"{t}_price"]  = round(price, 4) if pd.notna(price) else ""
        row[f"{t}_weight"] = round(weight, 4)
        row[f"{t}_target"] = round(target, 4)
        row[f"{t}_drift"]  = round(drift, 4)

    for bench in _BENCHMARKS:
        row[f"{bench}_price"] = round(prices.get(bench, float("nan")), 4) if bench in prices else ""

    return row


def print_snapshot(row: dict, allocation: dict[str, float]) -> None:
    print(f"\n  Date:      {row['Date']} {row['Time_ET']} ET")
    print(f"  Strategy:  {row['Strategy']}")
    print(f"  Equity:    ${row['Equity_Est']:>12,.2f}")
    print(f"  Cash:      ${row['Cash_Est']:>12,.2f}")
    print(f"  Source:    {row['Source']}")
    print()
    print(f"  {'Ticker':<8} {'Price':>8}  {'Current%':>9}  {'Target%':>8}  {'Drift pp':>9}")
    print(f"  {'-'*8} {'-'*8}  {'-'*9}  {'-'*8}  {'-'*9}")
    for t in allocation:
        price  = row.get(f"{t}_price", "")
        weight = row.get(f"{t}_weight", 0.0)
        target = row.get(f"{t}_target", 0.0)
        drift  = row.get(f"{t}_drift", 0.0)
        price_str = f"${price:.2f}" if isinstance(price, float) else ""
        drift_flag = "  ⚠" if abs(drift) > 0.05 else ""
        print(f"  {t:<8} {price_str:>8}  {weight*100:>8.1f}%  {target*100:>7.1f}%  "
              f"{drift*100:>+7.1f}pp{drift_flag}")
    print()
    print(f"  Benchmarks: SPY={row.get('SPY_price','?')}  "
          f"ALLW={row.get('ALLW_price','?')}  JEPQ={row.get('JEPQ_price','?')}")


def append_snapshot(row: dict) -> None:
    """Append (or create) the daily_snapshots.csv file."""
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    new_file = not _SNAPSHOT_CSV.exists()
    # Collect all columns dynamically from this row
    fieldnames = list(row.keys())

    if new_file:
        with open(_SNAPSHOT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)
        print(f"  Created: {_SNAPSHOT_CSV}")
    else:
        # Read existing header; add any new columns
        with open(_SNAPSHOT_CSV, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_fields = list(reader.fieldnames or [])
        all_fields = existing_fields + [c for c in fieldnames if c not in existing_fields]
        with open(_SNAPSHOT_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
            writer.writerow(row)
        print(f"  Appended: {_SNAPSHOT_CSV}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Daily price + portfolio-drift snapshot (no orders).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--paper", action="store_true")
    mode.add_argument("--live",  action="store_true")
    p.add_argument("--broker",      default="alpaca", choices=["alpaca", "tastytrade"])
    p.add_argument("--account",     default=None)
    p.add_argument("--strategy-id", default=config.DEFAULT_STRATEGY)
    p.add_argument("--dry-run",     action="store_true",
                   help="Print snapshot row without writing to CSV.")
    p.add_argument("--no-broker",   action="store_true",
                   help="Skip broker auth; estimate positions from last snapshot.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    trading_mode  = "live" if args.live else "paper"
    account_label = args.account or "default"
    strategy_id   = args.strategy_id

    print(f"\nAll-Weather daily snapshot  [{strategy_id}  {args.broker}/{trading_mode}]")
    print(f"  {datetime.now(tz=_NY_TZ).strftime('%Y-%m-%d %H:%M:%S ET')}")

    # 1. Load allocation
    allocation = {
        t: float(w)
        for t, w in config.load_strategy(strategy_id)["allocation"].items()
    }
    tickers = list(allocation)
    all_tickers = tickers + [b for b in _BENCHMARKS if b not in tickers]

    # 2. Fetch prices
    print("  Fetching prices …")
    prices = _fetch_prices(all_tickers)
    missing = [t for t in tickers if t not in prices]
    if missing:
        print(f"  WARNING: no price for {missing} — snapshot may be incomplete")

    # 3. Get positions
    positions: dict[str, float] = {}
    cash = 0.0
    broker_connected = False

    if not args.no_broker:
        result = _positions_from_broker(args.broker, trading_mode, account_label)
        if result is not None:
            positions, cash = result
            broker_connected = True
            # Map live tickers back to strategy tickers (GLD→GLDM etc.)
            try:
                strat_cfg = config.load_strategy(strategy_id)
                live_map = strat_cfg.get("live_tickers", {})
                reverse_map = {v: k for k, v in live_map.items()}
                positions = {reverse_map.get(t, t): qty for t, qty in positions.items()}
            except Exception:
                pass

    if not broker_connected:
        result = _positions_from_last_snapshot(tickers, prices)
        if result is not None:
            positions, cash = result
            print("  Broker unavailable — estimated from last snapshot row")
        else:
            print("  No positions available — equity/weight columns will be zero")

    # 4. Build + display row
    row = build_snapshot_row(
        strategy_id, prices, positions, cash, allocation, broker_connected
    )
    print_snapshot(row, allocation)

    # 5. Write (unless dry-run)
    if args.dry_run:
        print("  [dry-run] not written to CSV")
    else:
        append_snapshot(row)

    print("Done.")


if __name__ == "__main__":
    main()
