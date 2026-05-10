"""
compare_fmp_rp.py
=================
Recompute risk-parity weights from repo-local FMP SQLite ETF data and compare
them with the production yfinance-derived allocation in strategies.json.

Usage:
    conda run -n allweather python -m research.compare_fmp_rp
"""

from __future__ import annotations

import json
import os
import sys

import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from engine import config
from engine.data import fetch_prices_from_fmp_db
from engine.optimiser import compute_risk_parity_weights


RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")
BOUNDARIES = ["2018-01-01", "2020-01-01", "2022-01-01"]


def _load_production_allocation() -> dict[str, float]:
    path = os.path.join(_PROJECT_ROOT, "strategies.json")
    with open(path, "r") as f:
        data = json.load(f)
    return dict(data["strategies"][config.DEFAULT_STRATEGY]["allocation"])


def recompute_fmp_rp() -> tuple[pd.DataFrame, pd.DataFrame]:
    production = _load_production_allocation()
    tickers = list(production.keys())
    prices = fetch_prices_from_fmp_db(tickers, config.BACKTEST_START, config.BACKTEST_END)

    rows = []
    for boundary in BOUNDARIES:
        weights = compute_risk_parity_weights(
            prices,
            tickers,
            estimation_years=config.RP_LOOKBACK_YEARS,
            min_weight=config.RP_MIN_WEIGHT,
            end_date=boundary,
        )
        rows.append({"Boundary": boundary, **weights})

    boundary_weights = pd.DataFrame(rows).set_index("Boundary")
    avg_weights = boundary_weights.mean()
    avg_weights = avg_weights / avg_weights.sum()

    comparison = pd.DataFrame({
        "Production Weight": pd.Series(production),
        "FMP Avg RP Weight": avg_weights,
    })
    comparison["Diff"] = comparison["FMP Avg RP Weight"] - comparison["Production Weight"]
    comparison["Diff (bp)"] = comparison["Diff"] * 10_000
    comparison = comparison.reset_index(names="Ticker")

    return boundary_weights.reset_index(), comparison


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    boundary_weights, comparison = recompute_fmp_rp()

    boundary_path = os.path.join(RESULTS_DIR, "fmp_rp_boundary_weights.csv")
    comparison_path = os.path.join(RESULTS_DIR, "fmp_rp_weight_comparison.csv")
    boundary_weights.to_csv(boundary_path, index=False)
    comparison.to_csv(comparison_path, index=False)

    print("\nFMP RP boundary weights")
    print(boundary_weights.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nFMP average vs production")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nImportant: this diagnostic uses FMP close prices by default.")
    print("Use research.rerun_rp_validation's fmp_adj_close scenario for the")
    print("production cross-check against yfinance total-return results.")
    print(f"\nSaved:\n  {boundary_path}\n  {comparison_path}")


if __name__ == "__main__":
    main()
