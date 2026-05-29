# All Weather — Live Readiness Implementation Plan

> **STATUS: COMPLETED 2026-05-27.** All 10 phases shipped.
> Verification: 32/32 broker + legacy tests passing; end-to-end dry-execute
> validated against `FakeBroker`. See `docs/BROKER_SETUP.md` for the runbook
> and `README.md` "Live trading" section for entry points.
>
> Mapping from plan to shipped code:
>
> | Phase | Module(s) | Status |
> |---|---|---|
> | 1 — Broker abstraction | `live/brokers/{base,alpaca,tastytrade,factory}.py`, `live/rebalance.py` | ✅ |
> | 2 — Per-account budget | `live/budget.py` | ✅ |
> | 3 — 31-day hold + cadence | `live/lots.py` + cadence helpers in `live/rebalance.py` | ✅ |
> | 4 — Structured logging | `live/runlog.py` | ✅ |
> | 5 — Notifications | `live/notify.py` (no DD circuit breaker per user) | ✅ |
> | 6 — Healthcheck + launchd | `live/healthcheck.py`, `live/scheduler/`, `scripts/install_launchd.sh` | ✅ |
> | 7 — JEPQ benchmark | `research/compare_jepq.py` + JEPQ in `compare_allw.py` + `_performance_headers` | ✅ |
> | 8 — Backtest shadow | `research/backtest_shadow.py` | ✅ |
> | 9 — Docs + Makefile | `docs/BROKER_SETUP.md`, updated Makefile, CLAUDE.md, README.md, ToDo.md | ✅ |
> | 10 — Integration gate | All tests pass; smoke test of dry-execute with `FakeBroker` | ✅ |

Author: planning pass on 2026-05-27, revised same day with broker-abstraction + 31-day-rule additions.
Audience: Claude Sonnet executing the work end-to-end.
Project root: `projects/all-weather/`. **All commands must run inside the `allweather` conda env.** Use `conda run -n allweather python -m ...` for every script, every test, every install command. Never run with the system Python.

Read this whole document before writing any code. Do not skip the **Design notes** and **Acceptance criteria** sections inside each phase — they are load-bearing.

---

## User decisions already locked in

- **Dividend scope:** strategy-symbol DIVs only count toward the strategy budget. Other dividends are ignored.
- **Cash boundary:** pull broker activity API (deposits / withdrawals) to subtract external transfers from managed-capital.
- **JEPQ:** full backtest + add to ALLW comparison + add as a benchmark in the live monthly performance CSV.
- **Broker abstraction:** the rebalance code must NOT depend on Alpaca-specific types. Build a thin broker interface; ship two implementations (Alpaca, Tastytrade). The user's chosen live broker is **Tastytrade**, but the code must remain broker-agnostic.
- **Holding period rule:** the live US accounts enforce a **31-day minimum holding period on ETF shares**. The rebalancer must refuse to sell any position whose oldest lot is < 31 days old. Cadence is therefore "every ≥31 days since last sell-triggering execute," NOT calendar month-end.
- **No drawdown circuit breaker.** Do not implement one.
- **Conda environment:** every new dependency goes into both `requirements.txt` and `environment.yml`, and is installed via `conda run -n allweather pip install ...`.

---

## Switching from paper to live money

There is no single "paper" string to flip. The boundary is two things:

1. **CLI flag.** Today: `--paper` (default) vs `--live`. The plan keeps that. When you want real money, you pass `--live` and the broker abstraction picks the live endpoint for whichever broker is configured.
2. **Credentials.** The script reads credentials from env vars keyed by `--account <label>`. Convention going forward (broker-agnostic):
   - Credentials are loaded from `/Users/franciscosimao/Documents/QuantFinance/api_keys.env`.
   - `BROKER_<BROKER>_<ACCOUNT>_KEY` and `BROKER_<BROKER>_<ACCOUNT>_SECRET`
   - Example for Tastytrade live with the pinned `tastytrade==12.4.1` SDK: `BROKER_TASTYTRADE_LIVE_PROVIDER_SECRET`, `BROKER_TASTYTRADE_LIVE_REFRESH_TOKEN`. For Alpaca live: `BROKER_ALPACA_LIVE_KEY`, `BROKER_ALPACA_LIVE_SECRET`. Default-account shorthand is also supported: `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` and `TASTYTRADE_PROVIDER_SECRET` / `TASTYTRADE_REFRESH_TOKEN`. Document the exact required env vars per broker in `docs/BROKER_SETUP.md` (new file).
   - Backwards-compatibility shim: the existing `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY` env vars continue to work for the default Alpaca account so existing setups don't break.

The `trading_mode` field stored in JSON state files becomes `"paper"` or `"live"` and is derived from the CLI flag, not configured separately. The first time you run with `--live` against a Tastytrade live account, the run registry, budget state, and performance CSV all get separate files keyed by `(broker, trading_mode, account_label, strategy_id)` — they do not overwrite paper state.

Hard guard: live execution against a strategy budget that has never been previewed at least once on the same broker/account must fail with a clear error. We force you to do a preview-on-live before any real-money trade.

---

## Conventions

- All new modules live under `live/`, `research/`, or `engine/` — match the existing layout.
- No top-level scripts. Everything is `python -m <package>.<module>` runnable.
- Existing CLI flags and defaults do NOT change behavior — new features are opt-in. Exception: the structured run summary (Phase 4) is always written; the human log already runs on every invocation.
- All new file output goes under `logs/` (private, gitignored) unless part of a research bundle, in which case under `results/`.
- Add dependencies to both `requirements.txt` and `environment.yml`. Install only via `conda run -n allweather pip install <pkg>`. Never `pip install` outside the env.
- Type hints on new public functions. Keep the existing `from __future__ import annotations` + dataclass style.
- Tests under `tests/`. Each phase has a "tests" subsection — write them as part of the phase, not at the end. Run with `conda run -n allweather python -m pytest tests/ -v` after each phase.

Recommended execution order: **1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10**. Phases 7 and 8 can run in parallel after 6.

---

## Phase 1 — Broker abstraction layer

### Goal
Replace direct `alpaca.trading.client.TradingClient` usage in the rebalancer with a broker-agnostic `Broker` protocol. Ship two concrete implementations: `AlpacaBroker` (preserves current behavior) and `TastytradeBroker` (new).

### New module: `live/brokers/__init__.py` + `live/brokers/base.py`

```python
# live/brokers/base.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol, Iterable

@dataclass(frozen=True)
class PositionSnapshot:
    symbol: str
    qty: float
    qty_available: float
    market_value: float
    current_price: float
    avg_entry_price: float
    oldest_acquisition_date: date | None   # used by 31-day rule; None means unknown

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
    order_id: str
    symbol: str
    side: str            # "BUY" | "SELL"
    qty: float | None
    notional: float | None
    submitted_at: datetime
    status: str          # "filled" | "canceled" | "rejected" | "expired" | "timeout" | "open"

@dataclass(frozen=True)
class ActivityEvent:
    activity_type: str   # "DIV" | "DEPOSIT" | "WITHDRAWAL" | "FILL" | other broker-specific
    symbol: str | None
    amount: float        # positive for inflow to account, negative for outflow
    occurred_at: datetime
    raw: dict            # broker-specific payload for audit; sanitize PII before persisting

class Broker(Protocol):
    name: str            # "alpaca" | "tastytrade"
    trading_mode: str    # "paper" | "live"
    account_label: str

    def get_account(self) -> AccountSnapshot: ...
    def get_positions(self) -> dict[str, PositionSnapshot]: ...
    def get_asset(self, symbol: str) -> AssetMetadata: ...
    def get_open_orders(self, symbols: list[str]) -> list[OrderResult]: ...
    def is_market_open(self) -> bool: ...
    def last_trading_day_of_month(self, ref: date) -> date: ...
    def is_trading_day(self, when: date) -> bool: ...

    def fetch_activities(
        self,
        *,
        types: list[str],
        since: datetime,
        symbols: list[str] | None = None,
    ) -> list[ActivityEvent]: ...

    def submit_market_order(
        self,
        *,
        symbol: str,
        side: str,                  # "BUY" | "SELL"
        qty: float | None = None,
        notional: float | None = None,
    ) -> OrderResult: ...

    def get_order(self, order_id: str) -> OrderResult: ...
```

### `live/brokers/alpaca.py`

Move all current Alpaca-specific code from `live/alpaca_rebalance.py` into this module behind the `Broker` protocol. Key migrations:
- `get_account_snapshot` → `AlpacaBroker.get_account()` + `AlpacaBroker.get_positions()`.
- `_validate_target_assets` → `AlpacaBroker.get_asset()` per symbol.
- `get_end_of_month_status` → split into `is_market_open` + `last_trading_day_of_month`.
- `wait_for_orders` polling loop stays in the rebalancer (broker-agnostic), but uses `broker.get_order(order_id)`.
- For `oldest_acquisition_date`: Alpaca does not return lot-level acquisition dates on `get_position`. Reconstruct lots from the local lot ledger added in Phase 3 — `AlpacaBroker` returns `oldest_acquisition_date = None` from the position object, and the rebalancer relies on Phase 3's ledger.

Credentials resolution: load `/Users/franciscosimao/Documents/QuantFinance/api_keys.env`, then use `BROKER_ALPACA_<ACCOUNT_LABEL>_KEY` / `BROKER_ALPACA_<ACCOUNT_LABEL>_SECRET`, plus fallback to the legacy `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY` and shorthand `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` when `account_label == "default"`.

### `live/brokers/tastytrade.py`

Use the Tastytrade community Python SDK. Pin a version into `requirements.txt` after verifying the package exists on PyPI — do not invent a name. Install with `conda run -n allweather pip install <verified-package>==<version>`.

Implement each `Broker` method. Tastytrade specifics:
- Authentication: the pinned community SDK uses OAuth. Store `BROKER_TASTYTRADE_<ACCOUNT>_PROVIDER_SECRET` and `BROKER_TASTYTRADE_<ACCOUNT>_REFRESH_TOKEN` in `api_keys.env`; the default account can also use `TASTYTRADE_PROVIDER_SECRET` and `TASTYTRADE_REFRESH_TOKEN`. The `python -m live.brokers.tastytrade login --account live` helper validates that the OAuth credentials can open a session.
- Fractional shares: Tastytrade supports fractional shares on a limited symbol list. `AssetMetadata.fractionable` must reflect the real broker capability — query the API; do not assume.
- Calendar/market clock: Tastytrade may not expose a calendar endpoint as clean as Alpaca's. If not available, fall back to `pandas_market_calendars` (already a transitive dependency via yfinance). Install only if not present.
- Activities: map Tastytrade transaction types to the standardized `ActivityEvent.activity_type` enum: dividends → `"DIV"`, ACH deposits → `"DEPOSIT"`, ACH withdrawals → `"WITHDRAWAL"`, ETF trade fills → `"FILL"`.

### `live/brokers/factory.py`

```python
def make_broker(
    *,
    broker_name: str,     # "alpaca" | "tastytrade"
    trading_mode: str,    # "paper" | "live"
    account_label: str,
) -> Broker: ...
```

Resolves credentials from env vars and returns the right concrete broker.

### Refactor `live/alpaca_rebalance.py` → `live/rebalance.py`

Rename the module. The old name stays as a thin shim:
```python
# live/alpaca_rebalance.py
"""Deprecated. Use `python -m live.rebalance` instead."""
from live.rebalance import main
if __name__ == "__main__":
    main()
```

The new `live/rebalance.py`:
- Replaces all Alpaca-specific types with the Broker protocol types from `live/brokers/base.py`.
- Adds CLI flag: `--broker {alpaca,tastytrade}`, default `alpaca` to preserve the existing workflow.
- The Makefile gains a `BROKER` variable: `make rebalance-preview BROKER=tastytrade ACCOUNT=LIVE`.

### Acceptance criteria

- `python -m live.rebalance --broker alpaca --paper --account default` reproduces byte-for-byte the existing preview output (modulo the new end-of-run summary line added in Phase 4).
- `python -m live.rebalance --broker tastytrade --paper --account paper` returns a valid preview against a Tastytrade paper account (requires Tastytrade paper credentials in env).
- `live/rebalance.py` contains zero `from alpaca` imports. Those live only in `live/brokers/alpaca.py`.
- Switching brokers does NOT mix state files: the budget JSON, run registry, and performance CSV all include `<broker>` in their filenames.

### Tests

`tests/test_broker_protocol.py`:
- Construct a `FakeBroker` that conforms to the protocol; pass it through the rebalance plan builder; assert the plan is correct without any network calls.

`tests/test_alpaca_broker.py` (integration-marked, skipped by default):
- Smoke test against Alpaca paper using mocked `alpaca.trading.client.TradingClient`.

`tests/test_tastytrade_broker.py` (integration-marked, skipped by default):
- Smoke test against Tastytrade paper using mocked SDK calls.

---

## Phase 2 — Per-account strategy budget

### Goal
Cap the strategy to a fixed dollar amount of a broker account. The budget grows only from (a) PnL on strategy positions and (b) dividends paid by strategy tickers. External deposits/withdrawals and non-strategy dividends do not inflate the budget.

This works for any broker (Alpaca, Tastytrade, future brokers) because it relies on the `Broker` protocol from Phase 1, not on Alpaca-specific calls.

### New module: `live/budget.py`

State file path: `logs/budget_state_<broker>_<trading_mode>_<account_label>_<strategy_id>.json`. Example: `logs/budget_state_tastytrade_live_main_6_asset_rp_baseline.json`.

State schema (JSON):
```jsonc
{
  "schema_version": 1,
  "broker": "tastytrade",
  "strategy_id": "6asset_tip_gsg_rpavg",
  "account_label": "main",
  "trading_mode": "live",
  "initial_capital": 10000.00,
  "initialized_at": "2026-06-01T13:30:00-04:00",
  "managed_capital": 10245.32,
  "reserved_cash": 145.21,
  "last_activity_cursor": "2026-05-30T20:00:00Z",
  "lifetime_dividends_strategy": 152.10,
  "lifetime_external_transfers": 0.00,
  "history": [...]
}
```

Public API:
```python
@dataclass(frozen=True)
class BudgetSnapshot:
    managed_capital: float
    positions_value: float
    reserved_cash: float
    dividends_since_last: float
    transfers_since_last: float
    raw_history_entry: dict

def state_path(broker: str, trading_mode: str, account_label: str, strategy_id: str) -> str: ...

def load_state(path: str) -> dict | None: ...

def initialise_state(
    path: str,
    *,
    broker: str,
    initial_capital: float,
    strategy_id: str,
    account_label: str,
    trading_mode: str,
    now_iso: str,
) -> dict: ...

def compute_snapshot(
    broker: Broker,
    *,
    state: dict,
    strategy_symbols: list[str],
    positions: dict[str, PositionSnapshot],
) -> BudgetSnapshot:
    """
    Pulls DIV activities (filtered to strategy_symbols) since state['last_activity_cursor'].
    Pulls DEPOSIT/WITHDRAWAL activities since the same cursor.
    positions_value = sum(positions[symbol].market_value for symbol in strategy_symbols).
    reserved_cash = previous reserved_cash
                    + dividends_since_last
                    - transfers_since_last
                    + cash freed by sells since last run
                    - cash spent on buys since last run.
    Clamp reserved_cash >= 0; record discrepancy in raw_history_entry['discrepancy'] if clamped.
    managed_capital = positions_value + reserved_cash.
    """

def persist_snapshot(path: str, state: dict, snapshot: BudgetSnapshot, now_iso: str) -> None: ...
```

### Wiring into `live/rebalance.py`

New CLI args:
```
--budget INITIAL_CAPITAL    Cap the strategy to a fixed dollar budget.
--initialize-budget         First-time-only. Refuses to overwrite an existing state file.
--ignore-budget             Run against full account equity (current default behavior).
```

When `--budget` is in effect:

1. After `broker.get_positions()`, call `compute_snapshot(...)`. Substitute its `managed_capital` everywhere the code currently uses `account.equity`.
2. Hot paths to update: `build_rebalance_plan`, `verify_post_trade_drift`, `validate_order_guardrails`, the preview print block, the performance CSV writer.
3. Persist the snapshot at the end of a successful execute (right after `record_successful_run`). For preview-only runs, do NOT persist — preview must stay idempotent.
4. With `--budget` active, `--liquidate-other-positions` does NOT touch non-strategy positions even if passed. Override with `--budget-liquidate-other-strategy-positions` (only sells positions that were once part of this strategy but no longer match the current allocation).

### Acceptance criteria

- `--budget 10000 --initialize-budget --broker alpaca --paper` against a fresh account creates a state file with `managed_capital ≈ initial_capital`; preview plan Target $ values sum to ~$10k.
- Re-running without `--initialize-budget` succeeds; with `--initialize-budget` raises (refuses to overwrite).
- Depositing $5000 to the account between runs does not change `managed_capital`.
- A simulated DIV in TIP increases `managed_capital` by approximately the DIV amount on the next run.
- A DIV in a non-strategy symbol leaves `managed_capital` unchanged.

### Tests

`tests/test_budget.py`:
- `test_state_path_format`
- `test_initialise_state` (refuses to overwrite)
- `test_compute_snapshot_with_no_activities`
- `test_compute_snapshot_with_strategy_dividend`
- `test_compute_snapshot_with_nonstrategy_dividend`
- `test_compute_snapshot_with_external_transfer`
- `test_compute_snapshot_negative_clamp`

Use a `FakeBroker` fixture; no real network calls.

---

## Phase 3 — 31-day minimum holding period enforcement

### Goal
Refuse to sell any position whose oldest lot is < 31 days old, and shift the rebalance cadence from "last trading day of month" to "every ≥31 days since the last sell-triggering execute."

### Lot ledger

Brokers do not consistently expose lot-level acquisition dates via their public APIs, and even when they do (Tastytrade does, Alpaca generally does not), we want a single source of truth. Maintain a local lot ledger.

New module: `live/lots.py`.

State file: `logs/lots_<broker>_<trading_mode>_<account_label>_<strategy_id>.json`.

Schema:
```jsonc
{
  "schema_version": 1,
  "lots": {
    "SPY": [
      {"qty": 12.345, "acquired_at": "2026-04-30T15:45:00-04:00", "fill_price": 502.10, "order_id": "..."},
      {"qty": 3.210,  "acquired_at": "2026-05-31T15:45:00-04:00", "fill_price": 510.55, "order_id": "..."}
    ],
    "QQQ": [...]
  }
}
```

Public API:
```python
def load_ledger(path: str) -> dict: ...

def record_fill(
    path: str,
    *,
    symbol: str,
    side: str,             # "BUY" | "SELL"
    qty: float,
    fill_price: float,
    order_id: str,
    filled_at: datetime,
) -> None:
    """
    BUY appends a new lot.
    SELL decrements lots FIFO (oldest first). Refuses to decrement if any lot used
    in the decrement is younger than HOLDING_DAYS_MIN. Caller is expected to have
    pre-validated; this function raises if a SELL would actually violate.
    """

def oldest_acquisition_date(path: str, symbol: str) -> date | None: ...

def sellable_qty(path: str, symbol: str, *, now: datetime, min_days: int = 31) -> float:
    """Sum of qty across all lots whose acquired_at <= now - min_days."""

def holding_period_blocked_symbols(
    path: str,
    *,
    plan_sells: dict[str, float],   # symbol -> qty we want to sell
    now: datetime,
    min_days: int = 31,
) -> list[tuple[str, float, float, date]]:
    """
    Returns a list of (symbol, requested_qty, sellable_qty, oldest_acquired) for any
    plan_sells entry that exceeds sellable_qty.
    """
```

### Cadence enforcement

New module: `live/cadence.py`.

Replace the "is_last_trading_day" check in `live/rebalance.py` with a configurable cadence model.

Rules:
- Track `last_execute_at` in the run registry per `(broker, trading_mode, account_label, strategy_id)`.
- A new run computes `days_since_last_execute = (now - last_execute_at).days`.
- New flag: `--min-rebalance-interval-days N`, default **31**.
- If `days_since_last_execute < min_rebalance_interval_days`, refuse to `--execute` (override with `--force-cadence`, which is explicitly logged as an anomaly).
- Preview is always allowed regardless of cadence.
- The old `--force` flag (which bypassed month-end) becomes a no-op alias for backwards compatibility, with a deprecation warning routed through the logger.

### Pre-trade integration in `live/rebalance.py`

After `build_rebalance_plan`, before `validate_order_guardrails`:

1. Build `plan_sells = {row.symbol: row.qty for row in rows if row.action == "SELL"}`.
2. Call `holding_period_blocked_symbols(...)`.
3. If non-empty:
   - In preview mode: print a clear warning table showing which sells would be blocked and by how many days, but still print the full plan so the user sees the intent.
   - In execute mode: convert each blocked symbol's `SELL` action to `HOLD` with a `reason = "31-day holding period"`, and add an entry to `RunSummary.anomalies`. Recompute downstream buys accordingly (the freed cash that the original sell would have produced does not exist, so the buys must shrink). Simplest implementation: rerun `build_rebalance_plan` with the blocked symbols removed from the "sellable" universe — pass a new `sellable_qty_override: dict[str, float]` parameter that caps each symbol's sellable qty at the lot-allowed amount.

After fills:

4. On every BUY fill, call `record_fill(side="BUY", ...)` to append a lot.
5. On every SELL fill, call `record_fill(side="SELL", ...)` to FIFO-decrement lots.
6. Persist the ledger atomically (write-then-rename).

### Backfilling lots on first run

The first time the ledger is used against an existing account that already holds positions, lots are unknown. Initialization:

- New flag: `--initialize-lots {assume-immediate-sellable, assume-just-acquired, manual}`, default `assume-immediate-sellable`.
- `assume-immediate-sellable`: stamp each existing position as a single lot with `acquired_at = now - timedelta(days=min_days + 1)`. Net effect: nothing is holding-period-locked on day one. Logs a clear WARNING that the ledger was bootstrapped.
- `assume-just-acquired`: stamp `acquired_at = now`. Worst-case conservative: nothing is sellable for 31 days.
- `manual`: read `logs/initial_lots.json` provided by the user.

### Acceptance criteria

- Preview against a fresh account with no lot history shows no blocks.
- After a simulated BUY of SPY 10 days ago, a planned SELL on SPY is converted to HOLD with reason "31-day holding period" and the preview surfaces this in a warning table.
- After 32 days, the same SPY SELL goes through.
- Two consecutive `--execute` runs 5 days apart: the second refuses with a clear "cadence not met" message; `--force-cadence` lets it through and the anomaly is recorded.
- Running `--initialize-lots assume-immediate-sellable` against an existing portfolio creates a lot ledger and the first rebalance runs cleanly.

### Tests

`tests/test_lots.py`:
- `test_record_buy_appends_lot`
- `test_record_sell_decrements_fifo`
- `test_sellable_qty_excludes_young_lots`
- `test_holding_period_blocked_symbols_partial_block` (some lots old enough, some not)
- `test_sell_younger_than_min_raises`

`tests/test_cadence.py`:
- `test_first_run_allowed`
- `test_within_interval_blocked`
- `test_within_interval_with_force` (force flag passes and emits anomaly)
- `test_past_interval_allowed`

---

## Phase 4 — Structured logging

### Goal
Keep the existing human-readable log. Add:
1. Per-run summary JSON (`logs/runs/<timestamp>_<broker>_<account>_<mode>.json`).
2. Rolling JSONL (`logs/run_summary.jsonl`), one line per run.
3. `monthly_runs.csv` (one row per attempt, preview or execute).
4. End-of-run one-liner in the human log.

### How to test logs without waiting for the end of the month

This is now trivial because cadence (Phase 3) replaces month-end. Specifically:

- Preview mode (`python -m live.rebalance --broker alpaca --paper`) runs the full pipeline including all log writers without needing month-end. Every run, regardless of date, produces:
  - A timestamped human log under `logs/`
  - A summary JSON under `logs/runs/`
  - A line appended to `logs/run_summary.jsonl`
  - A line appended to `logs/monthly_runs.csv` (yes, the name is "monthly" but it records every attempt)
- Execute mode against the paper account: pass `--force-cadence` if you want to execute even when the 31-day interval has not elapsed, so you can exercise the full execute path including lot ledger writes and post-trade verification.
- A dedicated `python -m live.rebalance --dry-execute` mode that skips the actual `submit_market_order` but goes through every other step (fills are simulated using `current_price`, written to logs and lot ledger). Useful for end-to-end log testing without any broker side effects, including against live credentials. The dry-execute path tags everything in `RunSummary.anomalies` as `"dry_execute"` so these runs are never confused with real fills, and they do NOT advance the cadence cursor.

### Implementation

New module: `live/runlog.py`.

```python
@dataclass
class RunSummary:
    started_at: str
    finished_at: str
    broker: str
    trading_mode: str
    account_label: str
    strategy_id: str
    execution_mode: str          # "preview" | "execute" | "dry_execute"
    budget_active: bool
    equity_before: float
    cash_before: float
    managed_capital_before: float | None
    cadence_days_since_last: int | None
    is_trading_day: bool
    market_is_open: bool
    plan: list[dict]
    warnings: list[str]
    blocked_by_holding_period: list[dict]
    orders_submitted: list[dict]
    final_drift: dict[str, float] | None
    final_equity: float | None
    final_managed_capital: float | None
    anomalies: list[str]
    outcome: str                 # "preview_ok" | "execute_ok" | "execute_failed" | "blocked" | "dry_execute_ok"
    error_message: str | None
    elapsed_seconds: float

def open_run(...) -> RunSummary: ...
def write_run(summary: RunSummary) -> None: ...
def append_monthly_runs_csv(summary: RunSummary, path: str) -> None: ...
```

In `live/rebalance.py`:
- Create the `RunSummary` at the top of `main()`. Fill it incrementally.
- Wrap the main body in `try/finally`; in `finally`, call `write_run(summary)` regardless of outcome.
- Final log line:
  `logger.info(f"{outcome.upper()} | mode={execution_mode} | broker={broker} | trades={n_trades} | turnover=${turnover:,.2f} | drift_max={drift_max:.2%} | managed=${managed_capital:,.2f}")`

### `monthly_runs.csv` schema

```
run_id, started_at, finished_at, broker, trading_mode, account_label, strategy_id,
execution_mode, outcome, budget_active, managed_capital_before,
managed_capital_after, equity_before, equity_after, cash_before, cash_after,
cadence_days_since_last, n_planned_trades, n_filled_trades, n_blocked_by_holding,
turnover_dollars, drift_max_pct, n_warnings, n_anomalies, error_message, log_file
```

### Log rotation

Add `_prune_logs(max_files=200)` called from `setup_logging`. Removes oldest `*_alpaca_rebalance.log` and `*_rebalance.log`. JSON and CSV are never pruned.

### Acceptance criteria

- A single `python -m live.rebalance --broker alpaca --paper` invocation (no execute) writes a complete `RunSummary` JSON, appends to JSONL, appends to CSV. Verifiable by running it now without waiting for any market schedule.
- A failed run still produces a complete summary with `outcome != "*_ok"`.
- `--dry-execute` writes a summary with `outcome == "dry_execute_ok"` and the order list populated as if the fills happened, but no real orders are submitted to the broker.
- `logs/monthly_runs.csv` is parseable by `pandas.read_csv`.

### Tests

`tests/test_runlog.py`:
- `test_run_summary_round_trip`
- `test_append_monthly_runs_csv_creates_header`
- `test_append_monthly_runs_csv_appends`
- `test_prune_logs_keeps_newest`

---

## Phase 5 — Alerting (notifications only, no drawdown breaker)

### Goal
Push outcome notifications to Slack and/or email. **No drawdown circuit breaker** — explicitly removed per user request.

### New module: `live/notify.py`

```python
def notify(title: str, body_lines: list[str], *, severity: str = "info") -> None:
    """
    Reads NOTIFY_SLACK_WEBHOOK_URL and/or NOTIFY_EMAIL_TO + NOTIFY_EMAIL_SMTP_*.
    If neither is set, no-op (logs an INFO "notification skipped").
    Severity in {"info", "warning", "error"} controls slack color and email subject prefix.
    Wrapped in try/except — failures here MUST NOT break the rebalance run.
    """
```

Slack: `requests.post` to webhook URL with `{"text": ..., "attachments": [{"color": ..., "fields": [...]}]}`. Add `requests` to `requirements.txt` and `environment.yml` if not present.

Email: stdlib `smtplib` + `email.mime`. Env vars: `NOTIFY_EMAIL_TO`, `NOTIFY_EMAIL_FROM`, `NOTIFY_EMAIL_SMTP_HOST`, `NOTIFY_EMAIL_SMTP_PORT`, `NOTIFY_EMAIL_SMTP_USER`, `NOTIFY_EMAIL_SMTP_PASS`. Use STARTTLS on port 587.

### Wiring

In `live/rebalance.main()`:
- One `notify(severity="info", ...)` at run start.
- One `notify(...)` at run end. Severity:
  - `info` if outcome ends in `_ok` and no warnings.
  - `warning` if there are warnings or blocked symbols.
  - `error` if outcome is failed or blocked.

Every notify call is wrapped in try/except inside `notify` itself; the rebalancer just calls it.

### Acceptance criteria

- With no notify env vars set, all runs complete normally and the logger emits "notification skipped" twice (start, end).
- With `NOTIFY_SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...` set (mocked in tests), a successful preview posts one info message at end. A failure posts one error message.
- Any exception inside `notify` is caught and logged; the rebalance continues.

### Tests

`tests/test_notify.py`:
- `test_notify_noop_without_env`
- `test_notify_slack_payload_shape` (mock `requests.post`, assert JSON shape + severity color)
- `test_notify_swallows_exceptions`

---

## Phase 6 — Scheduling + health check

### Goal
Make it easy to leave the laptop alone. Daily health check catches credential / market-calendar / stuck-order issues before the next ≥31-day rebalance.

### Health check script

New module: `live/healthcheck.py`. Runnable as `python -m live.healthcheck --broker alpaca --paper --account default`.

Checks:
1. Broker credentials authenticate (`broker.get_account()` returns).
2. Market calendar is reachable for the current week.
3. No stuck orders (any open order > 24h old not in a terminal state).
4. Budget state file exists and parses (if `--require-budget` is passed).
5. Lot ledger exists and parses (if `--require-lots` is passed).
6. Cadence status: report `days_since_last_execute` and whether the next execute is allowed.
7. Portfolio drift is within a warning threshold (default 8%, configurable via `--drift-warning-threshold`). This is a warning surface only — does not block anything.
8. Calls `notify(severity="info")` on full pass, `severity="warning"` on any check fail. Exits 0 on full pass, 1 on any fail.

Every healthcheck run writes a `RunSummary` with `execution_mode="healthcheck"` to the same JSONL / CSV stream as rebalances.

### macOS launchd templates

`live/scheduler/com.fcastelasimao.allweather.healthcheck.plist`:
- Runs every weekday at 09:00 ET.
- Calls `python -m live.healthcheck --broker alpaca --paper --account default`.

`live/scheduler/com.fcastelasimao.allweather.preview.plist`:
- Runs every weekday at 15:30 ET.
- Calls `python -m live.rebalance --broker alpaca --paper --account default`.

**No execute plist.** Auto-execution of real-money orders is intentionally not scheduled. The user runs `--execute` manually after reviewing each preview.

`scripts/install_launchd.sh`:
1. Substitutes `${HOME}` and the conda env path into each plist.
2. Copies to `~/Library/LaunchAgents/`.
3. `launchctl load` each.
4. Requires `--install` flag. Without it, prints a "review then run" message.

### Acceptance criteria

- `python -m live.healthcheck --broker alpaca --paper` exits 0 on a healthy account and writes a JSONL entry.
- The plist templates use `${HOME}` placeholders, not hardcoded paths.
- `scripts/install_launchd.sh --install` exits 0; `launchctl list | grep allweather` shows both agents.

### Tests

`tests/test_healthcheck.py`:
- Mock broker; assert each check returns expected result for success and failure inputs.
- `test_healthcheck_exit_codes`.

---

## Phase 7 — JEPQ benchmark + backtest + ALLW chart integration

### Goal
1. Add JEPQ as a benchmark return column in the live performance CSV.
2. Full head-to-head: All Weather production strategy vs JEPQ buy-and-hold, over JEPQ's history.
3. Add JEPQ as a third line in the ALLW comparison chart.

### Performance CSV: add `JEPQ_Return%`

Edit `_performance_headers` and `calculate_benchmark_returns` (currently in `live/alpaca_rebalance.py:176`, will move to `live/rebalance.py` per Phase 1):
- Add "JEPQ" to the yfinance.download symbol list.
- Compute `jepq_ret` the same way as SPY / ALLW / TLT.
- Add `"JEPQ_Return%": jepq_ret`.

Header migration:
- Detect existing CSV with old header (no `JEPQ_Return%`).
- Rename old CSV to `<path>.preJEPQ.csv`, log at WARNING, start fresh.

### Research module: `research/compare_jepq.py`

Modeled on `research/compare_allw.py`. Key differences:
- JEPQ inception: 2022-05-03. Document the short window in the chart caption.
- Buy-and-hold JEPQ vs production All Weather backtested over the same window. Both use `auto_adjust=True` for total return.
- Outputs:
  - `results/jepq_comparison_growth.png` — equity-curve chart, dark theme matching ALLW.
  - `results/jepq_comparison_metrics.xlsx` — CAGR, MaxDD, Calmar, Sharpe, Sortino, Ulcer, JEPQ distribution yield.
  - `results/jepq_comparison_drawdown.png` — drawdown overlay.
- JEPQ-specific: trailing 12-month distribution yield via `yf.Ticker("JEPQ").dividends`.
- Makefile: `make compare-jepq`.

### ALLW chart integration

In `research/compare_allw.py`, add JEPQ as an optional third benchmark line behind `INCLUDE_JEPQ = True` constant. JEPQ inception (2022-05-03) precedes ALLW inception (2025-03-06), so the overlap window is fine.

### Acceptance criteria

- `make compare-jepq` produces three artifacts; metrics table populates.
- `make compare-allw` after the change adds a JEPQ row and a new line.
- The next live rebalance records `JEPQ_Return%`; the old CSV survives under `*.preJEPQ.csv`.

### Tests

`tests/test_compare_jepq.py` (integration-marked):
- Network test that downloads 30 days of JEPQ data.
- Local fixture test for CAGR / MaxDD / Calmar math.

---

## Phase 8 — Backtest shadow + reconciliation

### Goal
Each cadence cycle, compare actual live performance vs what the engine's backtest would have produced over the same window.

### New module: `research/backtest_shadow.py`

`python -m research.backtest_shadow --broker alpaca --paper --account default --strategy-id 6_asset_rp_baseline`

What it does:
1. Reads `logs/monthly_runs.csv` filtered to the broker/account/strategy.
2. For each consecutive pair of execute rows, computes:
   - Actual managed_capital return between runs.
   - Hypothetical return: rerun `engine.backtest.run_backtest` between the two dates with the same allocation and starting capital.
3. Writes `logs/backtest_shadow_<broker>_<account>_<strategy_id>.csv`:
   `period_start, period_end, actual_return_pct, hypothetical_return_pct, tracking_error_pct, contributing_factors`.
4. If `|tracking_error_pct| > 0.5%`, call `notify(severity="warning")`.

### Acceptance criteria

- Running on a populated `monthly_runs.csv` produces N-1 rows.
- A deterministic test fixture yields tracking error within 5 bps after fees.

### Tests

`tests/test_backtest_shadow.py`:
- Fixture with two execute rows and a deterministic price series.

---

## Phase 9 — Documentation + Makefile

### Makefile additions

```make
# Broker-agnostic targets. Default BROKER=alpaca; override for tastytrade.
BROKER ?= alpaca
ACCOUNT ?= default

rebalance-preview:
	conda run -n allweather python -m live.rebalance --broker $(BROKER) --paper --account $(ACCOUNT) --use-live-tickers

rebalance-execute:
	conda run -n allweather python -m live.rebalance --broker $(BROKER) --paper --account $(ACCOUNT) --use-live-tickers --execute

rebalance-dry-execute:
	conda run -n allweather python -m live.rebalance --broker $(BROKER) --paper --account $(ACCOUNT) --use-live-tickers --dry-execute

budget-init:
	conda run -n allweather python -m live.rebalance --broker $(BROKER) --paper --account $(ACCOUNT) --use-live-tickers --budget $(BUDGET) --initialize-budget

budget-preview:
	conda run -n allweather python -m live.rebalance --broker $(BROKER) --paper --account $(ACCOUNT) --use-live-tickers --budget $(BUDGET)

budget-execute:
	conda run -n allweather python -m live.rebalance --broker $(BROKER) --paper --account $(ACCOUNT) --use-live-tickers --budget $(BUDGET) --execute

lots-init:
	conda run -n allweather python -m live.rebalance --broker $(BROKER) --paper --account $(ACCOUNT) --initialize-lots assume-immediate-sellable

healthcheck:
	conda run -n allweather python -m live.healthcheck --broker $(BROKER) --paper --account $(ACCOUNT)

compare-jepq:
	conda run -n allweather python -m research.compare_jepq

backtest-shadow:
	conda run -n allweather python -m research.backtest_shadow --broker $(BROKER) --paper --account $(ACCOUNT) --strategy-id 6_asset_rp_baseline
```

### README updates

In `README.md`, add sections:

**"Choosing a broker"** — explain `--broker alpaca` vs `--broker tastytrade`, env-var conventions, paper vs live switching.

**"Running with a fixed budget"** — between "Paper trading" and "Project Structure":
```
make budget-init   ACCOUNT=PAPER BUDGET=10000
make budget-preview ACCOUNT=PAPER BUDGET=10000
make budget-execute ACCOUNT=PAPER BUDGET=10000
```

**"31-day holding period"** — explain the rule, the `--min-rebalance-interval-days` flag, lot ledger initialization, and `--force-cadence` as a deliberate override.

**"Logs and notifications"** — JSONL summary, monthly_runs CSV, dry-execute for testing logs without affecting the broker, notify env vars.

**"Switching paper to live"** — checklist: preview-on-live succeeds first, env vars set, cadence cursor reset if needed, lot ledger initialized, budget state file exists.

Update **Known Limitations**: auto-execution of live orders is intentionally not scheduled.

Update **ToDo.md**: move implemented items to "Completed"; add anything discovered to "Someday".

### `docs/BROKER_SETUP.md` (new)

Step-by-step credential setup for Alpaca (paper + live) and Tastytrade (paper + live). Mirrors the structure of the existing `ALPACA_SETUP_MAC.md`.

---

## Phase 10 — Final integration & gating

After all phases, run inside the conda env:

```bash
conda run -n allweather python -m pytest tests/ -v
conda run -n allweather make compare-allw
conda run -n allweather make compare-jepq
conda run -n allweather make rebalance-preview BROKER=alpaca ACCOUNT=PAPER
conda run -n allweather make budget-preview BROKER=alpaca ACCOUNT=PAPER BUDGET=10000
conda run -n allweather make healthcheck BROKER=alpaca ACCOUNT=PAPER
conda run -n allweather make rebalance-dry-execute BROKER=alpaca ACCOUNT=PAPER
```

Optional (requires Tastytrade paper credentials):
```bash
conda run -n allweather make rebalance-preview BROKER=tastytrade ACCOUNT=PAPER
```

Confirm:
- All existing tests still pass.
- All new tests pass.
- `make rebalance-preview BROKER=alpaca ACCOUNT=PAPER` produces output equivalent to the pre-change preview except for the new end-of-run summary line and the broker tag in headers. Anything else is a regression — fix before moving on.
- `make budget-preview` shows Target $ values summing to ~$10k.
- `make rebalance-dry-execute` writes a complete RunSummary with simulated fills, does NOT contact the broker order endpoint, and does NOT advance cadence.
- `make healthcheck` exits 0 and writes a JSONL entry.

---

## What to NOT do

- Do not change production allocation weights or strategies.json.
- Do not auto-execute live orders on a schedule. Preview is the only thing automation runs unsupervised.
- Do not implement a drawdown circuit breaker. Explicitly removed.
- Do not log broker API keys or activity payloads containing PII. Sanitize before persisting.
- Do not push to GitHub or open PRs unless the user explicitly asks.
- Do not delete the legacy `live/portfolio.py` — it is the JSON-backed model still referenced by tests and `main.py`. Phase 2 adds a parallel layer; it does not replace `portfolio.py`.
- Do not `pip install` outside the conda env. Every install uses `conda run -n allweather pip install ...` and updates both `requirements.txt` and `environment.yml`.

---

## Open items to flag back to the user before any real-money execution

1. Confirm the live ETF substitution decision (only GLD→GLDM remains active). README "Live ETF Mapping" documents the rationale.
2. Tax-lot tracking for tax reporting is **out of scope** here. The lot ledger added in Phase 3 enforces the 31-day rule but is NOT a substitute for broker-issued tax forms.
3. FCA / brokerage compliance review is product-side, not engine.
4. Verify the exact Tastytrade Python SDK name and version on PyPI before pinning. Do not invent a package name.
5. The 31-day holding period rule should be sanity-checked against Tastytrade's specific policy when the live account is opened — the default 31 may need to become 30 or different per broker. The CLI flag `--min-rebalance-interval-days` allows per-account override.
