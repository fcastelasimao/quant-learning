"""
live/runlog.py
==============
Structured run logging: JSONL stream, per-run JSON archive, and a monthly
summary CSV that makes performance-at-a-glance easy.

Log layout
----------
::

    live/logs/
        run_summary.jsonl           — one JSON line per run (append-only)
        monthly_runs.csv            — aggregate monthly view (all runs)
        runs/
            <timestamp>_<broker>_<mode>_<account>.json   — full run detail

Retention
---------
``_prune_runs(max_files=200)`` keeps the ``live/logs/runs/`` folder from growing
unbounded.  Oldest files are deleted first when the count exceeds max_files.
The JSONL and CSV are never pruned (they are append-only audit trails).

RunSummary
----------
Every run — preview, dry-execute, or execute — produces a RunSummary.
Only executed runs that resulted in filled orders have ``trades``.
"""

from __future__ import annotations

import csv
import glob
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    """One submitted order captured in the run summary."""
    symbol: str
    side: str            # "BUY" | "SELL"
    action_type: str     # "notional" | "qty"
    notional: float | None
    qty: float | None
    filled_qty: float
    filled_avg_price: float
    order_id: str
    status: str


@dataclass
class RunSummary:
    """Complete description of a single rebalancer invocation."""

    # Identity
    run_id: str                    # ISO timestamp + random suffix
    broker: str
    trading_mode: str              # "paper" | "live"
    account_label: str
    strategy_id: str

    # Timing
    started_at: datetime
    finished_at: datetime | None = None

    # Mode
    is_execute: bool = False
    is_dry_execute: bool = False

    # Outcome
    outcome: str = "preview"       # "preview" | "dry_execute" | "executed" | "skipped" | "error"
    error_message: str | None = None

    # Account state at run time
    equity_before: float = 0.0
    managed_capital: float = 0.0   # budget-adjusted capital
    cash_before: float = 0.0

    # Plan summary
    n_hold: int = 0
    n_buy: int = 0
    n_sell: int = 0
    n_holding_blocked: int = 0     # rows skipped due to < 31-day hold
    total_buy_notional: float = 0.0
    total_sell_notional: float = 0.0

    # Post-trade state
    equity_after: float = 0.0

    # Individual trades
    trades: list[TradeRecord] = field(default_factory=list)

    # Benchmark snapshot at run time
    benchmarks: dict[str, float] = field(default_factory=dict)

    # Misc
    warnings: list[str] = field(default_factory=list)
    cadence_days_remaining: int | None = None

    def duration_seconds(self) -> float | None:
        if self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["started_at"] = self.started_at.isoformat()
        d["finished_at"] = self.finished_at.isoformat() if self.finished_at else None
        return d


# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------

def _runs_dir(logs_dir: str) -> str:
    return os.path.join(logs_dir, "runs")


def _jsonl_path(logs_dir: str) -> str:
    return os.path.join(logs_dir, "run_summary.jsonl")


def _monthly_csv_path(logs_dir: str) -> str:
    return os.path.join(logs_dir, "monthly_runs.csv")


def _per_run_json_path(logs_dir: str, summary: RunSummary) -> str:
    ts = summary.started_at.strftime("%Y-%m-%d_%H-%M-%S")
    safe_acct = "".join(
        c if c.isalnum() or c in ("-", "_") else "_"
        for c in summary.account_label
    )
    fname = f"{ts}_{summary.broker}_{summary.trading_mode}_{safe_acct}.json"
    return os.path.join(_runs_dir(logs_dir), fname)


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------

def _prune_runs(logs_dir: str, max_files: int = 200) -> int:
    """Delete oldest per-run JSON files when count exceeds max_files.

    Returns the number of files deleted.
    """
    pattern = os.path.join(_runs_dir(logs_dir), "*.json")
    files = sorted(glob.glob(pattern))
    n_deleted = 0
    while len(files) > max_files:
        oldest = files.pop(0)
        try:
            os.remove(oldest)
            n_deleted += 1
        except OSError:
            pass
    return n_deleted


# ---------------------------------------------------------------------------
# Monthly CSV helpers
# ---------------------------------------------------------------------------

_MONTHLY_CSV_HEADERS = [
    "run_id",
    "started_at",
    "broker",
    "trading_mode",
    "account_label",
    "strategy_id",
    "outcome",
    "equity_before",
    "managed_capital",
    "equity_after",
    "n_buy",
    "n_sell",
    "n_hold",
    "n_holding_blocked",
    "total_buy_notional",
    "total_sell_notional",
    "duration_seconds",
    "error_message",
]


def _ensure_monthly_csv_header(csv_path: str) -> None:
    if not os.path.exists(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=_MONTHLY_CSV_HEADERS).writeheader()


def _append_monthly_csv(csv_path: str, summary: RunSummary) -> None:
    _ensure_monthly_csv_header(csv_path)
    row = {
        "run_id": summary.run_id,
        "started_at": summary.started_at.isoformat(),
        "broker": summary.broker,
        "trading_mode": summary.trading_mode,
        "account_label": summary.account_label,
        "strategy_id": summary.strategy_id,
        "outcome": summary.outcome,
        "equity_before": round(summary.equity_before, 2),
        "managed_capital": round(summary.managed_capital, 2),
        "equity_after": round(summary.equity_after, 2),
        "n_buy": summary.n_buy,
        "n_sell": summary.n_sell,
        "n_hold": summary.n_hold,
        "n_holding_blocked": summary.n_holding_blocked,
        "total_buy_notional": round(summary.total_buy_notional, 2),
        "total_sell_notional": round(summary.total_sell_notional, 2),
        "duration_seconds": round(summary.duration_seconds() or 0, 1),
        "error_message": summary.error_message or "",
    }
    with open(csv_path, "a", newline="") as fh:
        csv.DictWriter(fh, fieldnames=_MONTHLY_CSV_HEADERS).writerow(row)


# ---------------------------------------------------------------------------
# Public write API
# ---------------------------------------------------------------------------

def write_run(logs_dir: str, summary: RunSummary, max_run_files: int = 200) -> None:
    """Persist a RunSummary to all three sinks.

    1. Append one JSON line to run_summary.jsonl
    2. Write full detail to live/logs/runs/<timestamp>_...json
    3. Append one row to monthly_runs.csv
    4. Prune live/logs/runs/ if it exceeds max_run_files
    """
    os.makedirs(_runs_dir(logs_dir), exist_ok=True)

    # 1. JSONL
    jsonl = _jsonl_path(logs_dir)
    with open(jsonl, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(summary.to_dict(), default=str) + "\n")

    # 2. Per-run JSON
    per_run = _per_run_json_path(logs_dir, summary)
    with open(per_run, "w", encoding="utf-8") as fh:
        json.dump(summary.to_dict(), fh, indent=2, default=str)

    # 3. Monthly CSV
    _append_monthly_csv(_monthly_csv_path(logs_dir), summary)

    # 4. Prune
    _prune_runs(logs_dir, max_files=max_run_files)


# ---------------------------------------------------------------------------
# RunSummary factory helpers used by rebalance.py
# ---------------------------------------------------------------------------

def make_run_id(started_at: datetime | None = None) -> str:
    """Generate a unique run ID: ISO timestamp + 6-char hex suffix."""
    import secrets
    ts = (started_at or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}-{secrets.token_hex(3)}"


def finalise_summary(
    summary: RunSummary,
    outcome: str,
    equity_after: float = 0.0,
    error_message: str | None = None,
) -> RunSummary:
    """Stamp the finished_at time and set outcome/equity_after."""
    summary.finished_at = datetime.now(timezone.utc)
    summary.outcome = outcome
    summary.equity_after = equity_after
    summary.error_message = error_message
    return summary
