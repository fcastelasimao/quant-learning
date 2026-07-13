"""
rerun_rp_validation.py
======================
Recompute static 5-year risk-parity weights across the historical OOS
boundaries, average those weights, then rerun each OOS split.

Run with:
    conda run -n allweather python -m research.rerun_rp_validation

Outputs:
  - standard per-split results folders under results/
  - rows appended to results/master_log.xlsx
  - batch summary CSVs under results/rp_rerun_<timestamp>/
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from datetime import datetime
from typing import Iterator

import matplotlib
matplotlib.use("Agg")

import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from engine import config
from engine.backtest import compute_stats, run_backtest
from engine.data import fetch_prices
from engine.optimiser import compute_risk_parity_weights
from engine.plotting import plot_backtest
from research._shared.export import (
    append_to_master_log,
    export_results,
    make_results_dir,
    print_header,
    print_stats,
    start_run_log,
    stop_run_log,
)


SPLITS = ["2018-01-01", "2020-01-01", "2022-01-01"]

SCENARIOS = [
    {
        "id": "yfinance_total_return",
        "data_source": "yfinance",
        "pricing_model": "total_return",
        "fmp_price_column": "close",
    },
    {
        "id": "yfinance_price_return",
        "data_source": "yfinance",
        "pricing_model": "price_return",
        "fmp_price_column": "close",
    },
    {
        "id": "fmp_close",
        "data_source": "fmp",
        "pricing_model": "price_return",
        "fmp_price_column": "close",
    },
    {
        "id": "fmp_adj_close",
        "data_source": "fmp",
        "pricing_model": "total_return",
        "fmp_price_column": "adj_close",
    },
]


@contextmanager
def preserve_config() -> Iterator[None]:
    """Restore module-level config values after the batch finishes."""
    keys = [
        "DATA_SOURCE",
        "PRICING_MODEL",
        "FMP_PRICE_COLUMN",
        "RUN_MODE",
        "RUN_TAG",
        "OOS_START",
        "TARGET_ALLOCATION",
    ]
    original = {k: getattr(config, k) for k in keys}
    try:
        yield
    finally:
        for key, value in original.items():
            setattr(config, key, value)


def _production_allocation() -> dict[str, float]:
    return dict(config.TARGET_ALLOCATION)


def _scenario_label(scenario: dict[str, str]) -> str:
    return scenario["id"]


def _configure_scenario(scenario: dict[str, str]) -> None:
    config.DATA_SOURCE = scenario["data_source"]
    config.PRICING_MODEL = scenario["pricing_model"]
    config.FMP_PRICE_COLUMN = scenario["fmp_price_column"]


def _fetch_scenario_prices(tickers: list[str]) -> pd.DataFrame:
    all_tickers = list(dict.fromkeys(tickers + [config.BENCHMARK_TICKER, "TLT"]))
    return fetch_prices(all_tickers, config.BACKTEST_START, config.BACKTEST_END)


def _average_boundary_weights(boundary_weights: pd.DataFrame) -> dict[str, float]:
    avg = boundary_weights.mean()
    avg = avg / avg.sum()
    return {ticker: float(weight) for ticker, weight in avg.items()}


def _run_oos_split(prices: pd.DataFrame,
                   allocation: dict[str, float],
                   scenario_id: str,
                   oos_start: str) -> dict[str, object]:
    """Run one averaged-weight OOS split and append it to the master log."""
    tickers = list(allocation.keys())
    config.OOS_START = oos_start
    config.RUN_MODE = "oos_evaluate"
    config.RUN_TAG = f"{scenario_id}_rpavg_{oos_start[:4]}oos"
    config.TARGET_ALLOCATION = allocation
    config.validate_config()

    price_start = oos_start
    price_end = config.BACKTEST_END

    port_prices = prices[tickers]
    port_prices = port_prices[
        (port_prices.index >= price_start) &
        (port_prices.index < price_end)
    ]
    bench_prices = prices[config.BENCHMARK_TICKER]
    bench_prices = bench_prices[
        (bench_prices.index >= price_start) &
        (bench_prices.index < price_end)
    ]
    tlt_prices = prices["TLT"][
        (prices["TLT"].index >= price_start) &
        (prices["TLT"].index < price_end)
    ]

    run_label = config._build_run_label(price_start, price_end)
    results_dir = make_results_dir(run_label)

    tee = start_run_log(results_dir)
    try:
        print_header(
            f"RP AVERAGE RERUN | {scenario_id} | OOS {price_start} to {price_end}"
        )
        backtest = run_backtest(
            port_prices,
            bench_prices,
            allocation,
            tlt_prices=tlt_prices,
            transaction_cost_pct=config.TRANSACTION_COST_PCT,
            tax_drag_pct=config.TAX_DRAG_PCT,
        )
        stats_list = compute_stats(backtest, prices=port_prices, allocation=allocation)
        print_stats(stats_list)
        export_results(
            backtest,
            pd.DataFrame(),
            stats_list,
            allocation,
            results_dir,
            run_label,
        )
        append_to_master_log(results_dir, stats_list, allocation, run_label)
        plot_backtest(backtest, stats_list, results_dir, run_label, allocation=allocation)
    finally:
        stop_run_log(tee)

    aw = next(s for s in stats_list if s.name == "AW_R")
    return {
        "Scenario": scenario_id,
        "OOS Start": oos_start,
        "Run Label": run_label,
        "Results Dir": results_dir,
        "AW_R Calmar": aw.calmar,
        "AW_R CAGR (%)": aw.cagr,
        "AW_R Max DD (%)": aw.max_drawdown,
        "AW_R Max DD Daily (%)": aw.max_drawdown_daily,
        "AW_R Final Value ($)": aw.final_value,
    }


def run_scenario(scenario: dict[str, str],
                 production: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    scenario_id = _scenario_label(scenario)
    tickers = list(production.keys())

    print_header(f"SCENARIO: {scenario_id}")
    _configure_scenario(scenario)
    prices = _fetch_scenario_prices(tickers)

    boundary_rows = []
    for boundary in SPLITS:
        print_header(f"COMPUTING RP WEIGHTS | {scenario_id} | boundary {boundary}")
        weights = compute_risk_parity_weights(
            prices=prices,
            tickers=tickers,
            estimation_years=config.RP_LOOKBACK_YEARS,
            min_weight=config.RP_MIN_WEIGHT,
            end_date=boundary,
        )
        boundary_rows.append({"Scenario": scenario_id, "Boundary": boundary, **weights})

    boundary_weights = pd.DataFrame(boundary_rows).set_index(["Scenario", "Boundary"])
    avg_weights = _average_boundary_weights(boundary_weights)

    avg_rows = []
    for ticker, weight in avg_weights.items():
        prod_weight = production.get(ticker, 0.0)
        avg_rows.append({
            "Scenario": scenario_id,
            "Ticker": ticker,
            "Production Weight": prod_weight,
            "Avg RP Weight": weight,
            "Diff": weight - prod_weight,
            "Diff (bp)": (weight - prod_weight) * 10_000,
        })
    avg_df = pd.DataFrame(avg_rows)

    print_header(f"AVERAGED RP WEIGHTS | {scenario_id}")
    print(avg_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    summary_rows = []
    for oos_start in SPLITS:
        summary_rows.append(_run_oos_split(prices, avg_weights, scenario_id, oos_start))

    return boundary_weights.reset_index(), avg_df, summary_rows


def main() -> None:
    os.chdir(_PROJECT_ROOT)
    production = _production_allocation()
    summary_dir = os.path.join(
        "results",
        f"rp_rerun_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    )
    os.makedirs(summary_dir, exist_ok=True)

    all_boundary = []
    all_average = []
    all_summary = []

    with preserve_config():
        for scenario in SCENARIOS:
            boundary_df, average_df, summary_rows = run_scenario(scenario, production)
            all_boundary.append(boundary_df)
            all_average.append(average_df)
            all_summary.extend(summary_rows)

    boundary_out = pd.concat(all_boundary, ignore_index=True)
    average_out = pd.concat(all_average, ignore_index=True)
    summary_out = pd.DataFrame(all_summary)

    boundary_path = os.path.join(summary_dir, "rp_boundary_weights.csv")
    average_path = os.path.join(summary_dir, "rp_average_weights.csv")
    summary_path = os.path.join(summary_dir, "rp_oos_summary.csv")

    boundary_out.to_csv(boundary_path, index=False)
    average_out.to_csv(average_path, index=False)
    summary_out.to_csv(summary_path, index=False)

    print_header("BATCH SUMMARY SAVED")
    print(f"  {boundary_path}")
    print(f"  {average_path}")
    print(f"  {summary_path}")
    print("  Per-split runs were also appended to results/master_log.xlsx")


if __name__ == "__main__":
    main()
