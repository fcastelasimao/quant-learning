"""
research/tax_threshold_sweep.py (D.18)
======================================
Sweep drift threshold × tax regime × lot selector, score each configuration's
Calmar on the three OOS windows (2018/2020/2022 → end), and apply the
pre-registered kill criterion.

Grid
----
* rebalance policy:
    - monthly_unconditional (baseline)
    - drift_relative: 10 / 15 / 20 / 25 / 30 / 40 %
    - drift_absolute: 1 / 2 / 3 / 5 pp
* tax regime: us, none
* lot selector: fifo, tax_optimal

Kill criterion (pre-registered, see docs/research/tax_threshold_sweep.md)
------------------------------------------------------------------------
Under the US regime, if the best drift policy beats the monthly baseline on
Calmar by >= 5% on at least 2 of the 3 OOS windows (matched on lot selector),
PROPOSE a new production policy. Otherwise the tax model stays research-only and
monthly_unconditional remains production.

Artifacts (results/tax_threshold_sweep/<ts>_<strategy>/)
-------------------------------------------------------
* threshold_sweep_summary.csv  — one row per (policy, regime, selector, window)
* verdict.json                 — kill-criterion evaluation + decision
* run_config.json              — inputs + provenance

The marimo reads the CSV; it never recomputes.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine import config
from engine.backtest import RebalancePolicy
from engine.data import fetch_dividends, fetch_prices, get_price_provenance
from engine.lot_ledger import LotSelector
from engine.stats import compute_cagr, compute_calmar, compute_max_drawdown
from engine.tax import TaxRegime
from engine.tax_backtest import run_tax_aware_backtest

# OOS windows: (label, start). End is the backtest end for all.
OOS_WINDOWS = ("2018", "2020", "2022")
DAYS_PER_YEAR = 365.25
KILL_CRITERION_PCT = 0.05      # 5% Calmar improvement
KILL_CRITERION_WINDOWS = 2     # on at least 2 of 3 windows


def _policies() -> dict[str, RebalancePolicy]:
    pol = {"monthly_unconditional": RebalancePolicy.monthly_unconditional()}
    for pct in (10, 15, 20, 25, 30, 40):
        pol[f"drift_relative_{pct}pct"] = RebalancePolicy.drift_relative(pct / 100)
    for pp in (1, 2, 3, 5):
        pol[f"drift_absolute_{pp}pp"] = RebalancePolicy.drift_absolute(pp / 100)
    return pol


def _regimes() -> dict[str, TaxRegime]:
    return {"us": TaxRegime.us(), "none": TaxRegime.none()}


def _window_calmar(monthly: pd.DataFrame, start_year: str) -> dict[str, float]:
    """Calmar/CAGR/MDD on the value series from start_year-01-01 onward."""
    start = pd.Timestamp(f"{start_year}-01-01")
    sub = monthly.loc[monthly.index >= start, "Value"].dropna()
    if len(sub) < 2:
        return {"calmar": float("nan"), "cagr": float("nan"), "mdd": float("nan"), "months": len(sub)}
    years = (sub.index[-1] - sub.index[0]).days / DAYS_PER_YEAR
    cagr = compute_cagr(sub, years)
    mdd = compute_max_drawdown(sub)
    return {
        "calmar": round(compute_calmar(round(cagr, 4), round(mdd, 4)), 4),
        "cagr": round(cagr, 4),
        "mdd": round(mdd, 4),
        "months": len(sub),
    }


def run_sweep(strategy_id: str, start_date: str, end_date: str,
              transaction_cost_pct: float) -> tuple[pd.DataFrame, dict, dict]:
    canonical = config.resolve_strategy_id(strategy_id)
    allocation = {t: float(w) for t, w in config.load_strategy(strategy_id)["allocation"].items()}
    prices = fetch_prices(list(allocation), start_date, end_date)
    dividends = fetch_dividends(list(allocation), start_date, end_date)

    rows: list[dict] = []
    for selector in (LotSelector.FIFO, LotSelector.TAX_OPTIMAL):
        for regime_name, regime in _regimes().items():
            for pol_name, policy in _policies().items():
                res = run_tax_aware_backtest(
                    prices, allocation,
                    regime=regime, rebalance_policy=policy,
                    lot_selector=selector, dividends=dividends,
                    transaction_cost_pct=transaction_cost_pct,
                )
                final = float(res.monthly["Value"].iloc[-1])
                cum_tax = float(res.monthly["Cumulative Tax Paid"].iloc[-1])
                rebal = int(res.monthly["Rebalanced"].sum())
                for win in OOS_WINDOWS:
                    m = _window_calmar(res.monthly, win)
                    rows.append({
                        "selector": selector.value,
                        "regime": regime_name,
                        "policy": pol_name,
                        "window": win,
                        "calmar": m["calmar"],
                        "cagr": m["cagr"],
                        "mdd": m["mdd"],
                        "final_value": round(final, 2),
                        "cumulative_tax": round(cum_tax, 2),
                        "rebalances": rebal,
                    })
    summary = pd.DataFrame(rows)
    verdict = evaluate_kill_criterion(summary)
    run_cfg = {
        "strategy_id": canonical,
        "allocation": allocation,
        "start_date": start_date,
        "end_date": end_date,
        "transaction_cost_pct": transaction_cost_pct,
        "oos_windows": list(OOS_WINDOWS),
        "kill_criterion": {
            "calmar_improvement_pct": KILL_CRITERION_PCT,
            "min_windows": KILL_CRITERION_WINDOWS,
        },
        "price_provenance": get_price_provenance(prices),
    }
    return summary, verdict, run_cfg


def evaluate_kill_criterion(summary: pd.DataFrame) -> dict:
    """Apply the pre-registered kill criterion under the US regime.

    For each lot selector, compare every drift policy's Calmar against the
    monthly baseline on each OOS window. A policy "wins" a window if its Calmar
    exceeds the baseline's by >= KILL_CRITERION_PCT (relative). A policy passes
    if it wins on >= KILL_CRITERION_WINDOWS windows.
    """
    us = summary[summary["regime"] == "us"]
    result = {"decision": "research_only", "passing_policies": [], "detail": {}}

    for selector in sorted(us["selector"].unique()):
        sel = us[us["selector"] == selector]
        baseline = sel[sel["policy"] == "monthly_unconditional"].set_index("window")["calmar"]
        for policy in sorted(p for p in sel["policy"].unique() if p != "monthly_unconditional"):
            pol = sel[sel["policy"] == policy].set_index("window")["calmar"]
            wins = []
            per_window = {}
            for win in OOS_WINDOWS:
                base_c = baseline.get(win, float("nan"))
                pol_c = pol.get(win, float("nan"))
                improved = (
                    pd.notna(base_c) and pd.notna(pol_c) and base_c > 0
                    and (pol_c - base_c) / base_c >= KILL_CRITERION_PCT
                )
                per_window[win] = {
                    "baseline_calmar": None if pd.isna(base_c) else float(base_c),
                    "policy_calmar": None if pd.isna(pol_c) else float(pol_c),
                    "win": bool(improved),
                }
                if improved:
                    wins.append(win)
            passed = len(wins) >= KILL_CRITERION_WINDOWS
            key = f"{selector}/{policy}"
            result["detail"][key] = {"windows_won": wins, "passed": passed, "per_window": per_window}
            if passed:
                result["passing_policies"].append(key)

    if result["passing_policies"]:
        result["decision"] = "propose_new_production_policy"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Tax × drift-threshold × lot-selector sweep (D.18).")
    parser.add_argument("--strategy-id", default=config.DEFAULT_STRATEGY)
    parser.add_argument("--start-date", default=config.BACKTEST_START)
    parser.add_argument("--end-date", default=config.BACKTEST_END)
    parser.add_argument("--data-source", choices=("yfinance", "fmp"), default="fmp")
    parser.add_argument("--fmp-price-column", default="adj_close",
                        choices=("open", "high", "low", "close", "adj_close"))
    parser.add_argument("--transaction-cost-pct", type=float, default=config.TRANSACTION_COST_PCT)
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "results" / "tax_threshold_sweep"))
    args = parser.parse_args()

    config.DATA_SOURCE = args.data_source
    config.FMP_PRICE_COLUMN = args.fmp_price_column

    summary, verdict, run_cfg = run_sweep(
        args.strategy_id, args.start_date, args.end_date, args.transaction_cost_pct)

    canonical = config.resolve_strategy_id(args.strategy_id)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(args.output_root) / f"{ts}_{canonical}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "threshold_sweep_summary.csv", index=False)
    (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2))
    (out_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2))

    print(f"Sweep written to: {out_dir}")
    print(f"Decision: {verdict['decision']}")
    if verdict["passing_policies"]:
        print(f"Passing policies: {verdict['passing_policies']}")
    # show US Calmar by policy × window for FIFO (the production-relevant selector)
    fifo_us = summary[(summary["regime"] == "us") & (summary["selector"] == "fifo")]
    pivot = fifo_us.pivot(index="policy", columns="window", values="calmar")
    print("\nUS / FIFO Calmar by OOS window:")
    print(pivot.to_string())


if __name__ == "__main__":
    main()
