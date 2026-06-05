"""
live/healthcheck.py
===================
Quick sanity check to run before the rebalancer, or on its own.

Checks:
  1. Python interpreter is the allweather conda env
  2. Required packages are importable (alpaca-py, yfinance, pandas, etc.)
  3. Tastytrade SDK availability (optional)
  4. Broker credentials are set in api_keys.env / the environment
  5. strategies.json is valid and the target strategy exists
  6. live/logs/ directory is writable
  7. Cadence state — how many days until the next run is due

Exit codes:
    0 — all checks passed
    1 — one or more checks failed

Usage::

    conda run -n allweather python -m live.healthcheck
    conda run -n allweather python -m live.healthcheck --broker tastytrade --account main
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from datetime import date, timedelta

from live.env import default_api_keys_env_path, load_api_keys_env

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

load_api_keys_env()

PASS = "✓"
FAIL = "✗"
WARN = "!"


def _check(label: str, ok: bool, detail: str = "") -> bool:
    mark = PASS if ok else FAIL
    print(f"  [{mark}] {label}" + (f": {detail}" if detail else ""))
    return ok


def check_conda_env() -> bool:
    expected_fragment = os.path.join("envs", "allweather")
    ok = expected_fragment in sys.executable
    return _check("conda env is allweather", ok, sys.executable)


def check_imports() -> bool:
    required = [
        "pandas", "numpy", "yfinance", "alpaca.trading.client",
    ]
    all_ok = True
    for pkg in required:
        try:
            importlib.import_module(pkg)
            ok = True
        except ImportError:
            ok = False
            all_ok = False
        _check(f"import {pkg}", ok)
    return all_ok


def check_tastytrade_sdk() -> bool:
    try:
        importlib.import_module("tastytrade")
        return _check("tastytrade SDK installed", True)
    except ImportError:
        return _check("tastytrade SDK installed", False,
                      "optional — install with: conda run -n allweather pip install tastytrade")


def check_alpaca_credentials(account_label: str) -> bool:
    upper = account_label.upper()
    pairs = [
        (f"BROKER_ALPACA_{upper}_KEY", f"BROKER_ALPACA_{upper}_SECRET"),
        (f"APCA_API_KEY_ID_{upper}", f"APCA_API_SECRET_KEY_{upper}"),
    ]
    if account_label.lower() == "default":
        pairs.append(("APCA_API_KEY_ID", "APCA_API_SECRET_KEY"))
        pairs.append(("ALPACA_API_KEY", "ALPACA_SECRET_KEY"))
    for key_var, sec_var in pairs:
        if os.getenv(key_var) and os.getenv(sec_var):
            return _check(f"Alpaca credentials ({account_label})", True, f"via {key_var}")
    return _check(f"Alpaca credentials ({account_label})", False,
                  f"Set one of: {', '.join(f'{k}/{s}' for k,s in pairs)}")


def check_tastytrade_credentials(account_label: str) -> bool:
    upper = account_label.upper()
    pairs = [
        (f"BROKER_TASTYTRADE_{upper}_PROVIDER_SECRET", f"BROKER_TASTYTRADE_{upper}_REFRESH_TOKEN"),
    ]
    if account_label.lower() == "default":
        pairs.append(("TASTYTRADE_PROVIDER_SECRET", "TASTYTRADE_REFRESH_TOKEN"))
    for secret_var, token_var in pairs:
        if os.getenv(secret_var) and os.getenv(token_var):
            return _check(f"Tastytrade credentials ({account_label})", True, f"via {secret_var}")
    return _check(f"Tastytrade credentials ({account_label})", False,
                  f"Set one of: {', '.join(f'{s}/{t}' for s,t in pairs)}")


def check_strategies_json(strategy_id: str) -> bool:
    strategies_path = os.path.join(_PROJECT_ROOT, "strategies.json")
    if not os.path.exists(strategies_path):
        return _check("strategies.json exists", False, strategies_path)
    _check("strategies.json exists", True)
    try:
        with open(strategies_path) as fh:
            data = json.load(fh)
        strategies = data.get("strategies", {})
        # Fuzzy match (same logic as config.resolve_strategy_id)
        if strategy_id in strategies:
            found = strategy_id
        else:
            # Try canonical resolution
            found = None
            for key in strategies:
                if key.replace("_", "").replace("-", "") == strategy_id.replace("_", "").replace("-", ""):
                    found = key
                    break
        ok = found is not None
        return _check(f"Strategy '{strategy_id}' in strategies.json", ok,
                      f"found as '{found}'" if ok else f"available: {list(strategies)[:5]}")
    except Exception as exc:
        return _check("strategies.json parseable", False, str(exc))


def check_logs_writable() -> bool:
    logs_dir = os.path.join(_PROJECT_ROOT, "live", "logs")
    os.makedirs(logs_dir, exist_ok=True)
    probe = os.path.join(logs_dir, ".healthcheck_probe")
    try:
        with open(probe, "w") as fh:
            fh.write("ok")
        os.remove(probe)
        return _check("live/logs/ directory writable", True)
    except Exception as exc:
        return _check("live/logs/ directory writable", False, str(exc))


def check_cadence(
    broker: str,
    trading_mode: str,
    account: str,
    strategy_id: str,
    interval_days: int,
) -> bool:
    """Preview cadence state without importing rebalance.py (avoids heavy deps)."""
    logs_dir = os.path.join(_PROJECT_ROOT, "live", "logs")
    safe_a = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in account)
    safe_s = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in strategy_id)
    path = os.path.join(logs_dir, f"cadence_{broker}_{trading_mode}_{safe_a}_{safe_s}.json")
    if not os.path.exists(path):
        return _check("Cadence state", True, "No prior run — first run is immediately due")
    with open(path) as fh:
        state = json.load(fh)
    last = date.fromisoformat(state["last_run_date"])
    next_due = last + timedelta(days=interval_days)
    today = date.today()
    days_left = (next_due - today).days
    if today >= next_due:
        return _check("Cadence state", True,
                      f"Due now (last: {last}, next was: {next_due})")
    return _check("Cadence state", True,
                  f"{days_left}d until next run (last: {last}, due: {next_due})",
                  )


def run_checks(
    broker: str = "alpaca",
    account: str = "default",
    trading_mode: str = "paper",
    strategy_id: str = "6asset_tip_gsg_rpavg",
    interval_days: int = 31,
) -> bool:
    print(f"\nAll-Weather Healthcheck  broker={broker}  {trading_mode}  account={account}")
    print("=" * 60)
    results = []
    key_file = os.getenv("ALLW_API_KEYS_ENV_LOADED") or str(default_api_keys_env_path())
    results.append(_check("api_keys.env loaded", bool(os.getenv("ALLW_API_KEYS_ENV_LOADED")), key_file))
    print()
    results.append(check_conda_env())
    print()
    results.append(check_imports())
    print()
    tastytrade_ok = check_tastytrade_sdk()
    if broker == "tastytrade":
        results.append(tastytrade_ok)
    print()
    if broker == "alpaca":
        results.append(check_alpaca_credentials(account))
    else:
        results.append(check_tastytrade_credentials(account))
    print()
    results.append(check_strategies_json(strategy_id))
    print()
    results.append(check_logs_writable())
    print()
    results.append(check_cadence(broker, trading_mode, account, strategy_id, interval_days))
    print()

    n_fail = sum(1 for r in results if not r)
    if n_fail == 0:
        print(f"All checks passed.\n")
    else:
        print(f"{n_fail} check(s) failed. Review the output above.\n")
    return n_fail == 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="All-Weather system health check.")
    p.add_argument("--broker", default="alpaca", choices=["alpaca", "tastytrade"])
    p.add_argument("--account", default="default")
    p.add_argument("--paper", action="store_true")
    p.add_argument("--live", action="store_true")
    p.add_argument("--strategy-id", default="6asset_tip_gsg_rpavg")
    p.add_argument("--min-rebalance-interval-days", type=int, default=31)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mode = "live" if args.live else "paper"
    ok = run_checks(
        broker=args.broker,
        account=args.account,
        trading_mode=mode,
        strategy_id=args.strategy_id,
        interval_days=args.min_rebalance_interval_days,
    )
    sys.exit(0 if ok else 1)
