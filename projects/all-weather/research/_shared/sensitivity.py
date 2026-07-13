"""
sensitivity.py
==============
Small sensitivity suite for production validation.

The goal is not to optimise new weights. It checks whether the production
allocation is fragile to ordinary implementation assumptions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from engine import config
from engine.backtest import run_backtest
from engine.data import fetch_prices
from engine.stats import compute_stats


DEFAULT_OUTPUT_ROOT = Path("results") / "sensitivity"


def _stats_row(name: str,
               prices: pd.DataFrame,
               benchmark: pd.Series,
               allocation: dict[str, float],
               transaction_cost_pct: float) -> dict:
    bt = run_backtest(
        prices=prices[list(allocation)],
        benchmark_prices=benchmark,
        allocation=allocation,
        transaction_cost_pct=transaction_cost_pct,
        tax_drag_pct=config.TAX_DRAG_PCT,
    )
    aw = compute_stats(bt, prices=prices[list(allocation)], allocation=allocation)[0]
    return {
        "Scenario": name,
        "CAGR (%)": aw.cagr,
        "Max Drawdown (%)": aw.max_drawdown,
        "Max Drawdown Daily (%)": aw.max_drawdown_daily,
        "Calmar": aw.calmar,
        "Ulcer Index": aw.ulcer_index,
        "Final Value ($)": aw.final_value,
    }


def run_sensitivity(output_root: str | Path = DEFAULT_OUTPUT_ROOT) -> Path:
    """Run deterministic implementation-sensitivity checks."""
    allocation = dict(config.TARGET_ALLOCATION)
    tickers = list(allocation) + [config.BENCHMARK_TICKER]
    prices = fetch_prices(list(dict.fromkeys(tickers)), config.OOS_START, config.BACKTEST_END)
    benchmark = prices[config.BENCHMARK_TICKER]

    rows = []
    for cost in (0.0, 0.001, 0.0025, 0.005):
        rows.append(_stats_row(f"transaction_cost_{cost:.4f}", prices, benchmark, allocation, cost))

    # Weight perturbation: move 1% absolute weight from each asset to the next,
    # preserving long-only and sum-to-one constraints.
    symbols = list(allocation)
    for idx, source in enumerate(symbols):
        target = symbols[(idx + 1) % len(symbols)]
        perturbed = dict(allocation)
        shift = min(0.01, perturbed[source] - 0.001)
        if shift <= 0:
            continue
        perturbed[source] -= shift
        perturbed[target] += shift
        rows.append(_stats_row(f"shift_1pct_{source}_to_{target}", prices, benchmark, perturbed, config.TRANSACTION_COST_PCT))

    # Start-date dependence across recent stress starts.
    for start in ("2018-01-01", "2020-01-01", "2022-01-01"):
        window = prices.loc[prices.index >= pd.Timestamp(start)]
        if len(window) > 40:
            rows.append(_stats_row(f"start_date_{start}", window, window[config.BENCHMARK_TICKER], allocation, config.TRANSACTION_COST_PCT))

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    out_path = output_root / "production_sensitivity.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run production strategy sensitivity checks.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()
    path = run_sensitivity(args.output_root)
    print(f"Sensitivity results written to: {path}")


if __name__ == "__main__":
    main()
