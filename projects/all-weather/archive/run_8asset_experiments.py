"""
run_8asset_experiments.py
=========================
RP-optimised validation for two 8-asset universe candidates.

Experiment A: SPY, QQQ, IJR, TLT, IEF, GLD, CPER, DBA  (scan winner, no TIP)
Experiment B: SPY, QQQ, IJR, TLT, TIP, GLD, CPER, DBA  (keeps TIP, drops IEF)

For each experiment:
  1. Compute RP weights from IS-only data for 3 OOS splits (2018, 2020, 2022).
  2. Average the 3 sets of RP weights ("rpavg").
  3. Run OOS backtest on each split with the averaged weights.

This mirrors the methodology used for the production 6-asset rpavg strategy.

Run with:
    conda run -n allweather python3 run_8asset_experiments.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

import config
from data import fetch_prices
from backtest import run_backtest, compute_stats
from optimiser import compute_risk_parity_weights
from export import (make_results_dir, export_results,
                    append_to_master_log, start_run_log, stop_run_log)
from plotting import plot_backtest


# ---- Experiment definitions ----

EXPERIMENTS = {
    "A_8asset_no_tip": {
        "description": "8-asset scan winner (IEF instead of TIP)",
        "tickers": ["SPY", "QQQ", "IJR", "TLT", "IEF", "GLD", "CPER", "DBA"],
    },
    "B_8asset_with_tip": {
        "description": "8-asset with TIP (replaces IEF)",
        "tickers": ["SPY", "QQQ", "IJR", "TLT", "TIP", "GLD", "CPER", "DBA"],
    },
}

SPLITS = [
    {"oos_start": "2018-01-01", "tag": "2018oos"},
    {"oos_start": "2020-01-01", "tag": "2020oos"},
    {"oos_start": "2022-01-01", "tag": "2022oos"},
]

RP_ESTIMATION_YEARS = 5.0


# ---- Helpers ----

def compute_rpavg_weights(
    prices: pd.DataFrame,
    tickers: list[str],
    splits: list[dict],
) -> dict[str, float]:
    """Compute RP weights for each split, then average them."""
    all_weights: list[dict[str, float]] = []

    for split in splits:
        oos_start = split["oos_start"]
        print(f"\n  Computing RP weights (IS ends at {oos_start})...")
        rp_w = compute_risk_parity_weights(
            prices=prices,
            tickers=tickers,
            estimation_years=RP_ESTIMATION_YEARS,
            end_date=oos_start,
        )
        all_weights.append(rp_w)

    # Average across splits
    avg: dict[str, float] = {}
    for ticker in tickers:
        vals = [w[ticker] for w in all_weights]
        avg[ticker] = round(float(np.mean(vals)), 4)

    # Renormalise to sum to exactly 1.0
    total = sum(avg.values())
    avg = {t: round(v / total, 4) for t, v in avg.items()}
    # Fix rounding residual on the largest weight
    residual = 1.0 - sum(avg.values())
    if abs(residual) > 1e-8:
        biggest = max(avg, key=avg.get)
        avg[biggest] = round(avg[biggest] + residual, 4)

    print(f"\n  RP-averaged weights:")
    for t, w in sorted(avg.items(), key=lambda x: -x[1]):
        print(f"    {t:<6} {w:.1%}")

    return avg


def run_single_oos(
    prices: pd.DataFrame,
    tickers: list[str],
    allocation: dict[str, float],
    oos_start: str,
    tag: str,
) -> dict:
    """Run one OOS backtest with the given allocation. Return stats dict."""
    # Override config
    config.OOS_START = oos_start
    config.RUN_MODE = "oos_evaluate"
    config.RUN_TAG = tag
    config.TARGET_ALLOCATION = allocation
    config.validate_config()

    price_start = oos_start
    price_end = config.BACKTEST_END

    port_prices = prices[tickers]
    port_prices = port_prices[(port_prices.index >= price_start) &
                              (port_prices.index < price_end)]
    bench_prices = prices["SPY"][(prices["SPY"].index >= price_start) &
                                  (prices["SPY"].index < price_end)]
    tlt_prices = prices["TLT"][(prices["TLT"].index >= price_start) &
                                (prices["TLT"].index < price_end)]

    run_label = config._build_run_label(price_start, price_end)
    results_dir = make_results_dir(run_label)

    tee = start_run_log(results_dir)
    try:
        backtest = run_backtest(port_prices, bench_prices, allocation,
                                tlt_prices=tlt_prices)
        stats_list = compute_stats(backtest, prices=port_prices,
                                   allocation=allocation)

        export_results(backtest, pd.DataFrame(), stats_list, allocation,
                       results_dir, run_label)
        append_to_master_log(results_dir, stats_list, allocation, run_label)
        plot_backtest(backtest, stats_list, results_dir, run_label,
                      allocation=allocation)

        aw = next(s for s in stats_list if s.name == "AW_R")
        bh = next(s for s in stats_list if s.name == "B&H_AW")
        return {
            "split": tag,
            "cagr": aw.cagr,
            "max_dd": aw.max_drawdown,
            "max_dd_daily": aw.max_drawdown_daily,
            "calmar": aw.calmar,
            "ulcer": aw.ulcer_index,
            "sortino": aw.sortino,
            "sharpe": aw.sharpe,
            "bh_calmar": bh.calmar,
        }
    finally:
        stop_run_log(tee)


# ---- Main ----

def main() -> None:
    print("=" * 80)
    print("8-ASSET UNIVERSE EXPERIMENTS — RP-averaged weights, 3 OOS splits")
    print("=" * 80)

    all_results: dict[str, list[dict]] = {}

    for exp_name, exp_def in EXPERIMENTS.items():
        tickers = exp_def["tickers"]
        desc = exp_def["description"]
        all_tickers = list(dict.fromkeys(tickers + ["SPY", "TLT"]))

        print(f"\n{'#' * 80}")
        print(f"# EXPERIMENT: {exp_name}")
        print(f"# {desc}")
        print(f"# Tickers: {', '.join(tickers)}")
        print(f"{'#' * 80}")

        # Fetch prices
        print(f"\nFetching prices for {len(all_tickers)} tickers...")
        prices = fetch_prices(all_tickers, config.BACKTEST_START, config.BACKTEST_END)

        # Step 1: Compute RP-averaged weights
        print(f"\n--- Step 1: Compute RP-averaged weights ---")
        rpavg_weights = compute_rpavg_weights(prices, tickers, SPLITS)

        # Step 2: Run OOS on all 3 splits with averaged weights
        print(f"\n--- Step 2: OOS evaluation with rpavg weights ---")
        exp_results = []
        for split in SPLITS:
            oos_start = split["oos_start"]
            tag = f"{exp_name}_rpavg_{split['tag']}"
            print(f"\n  OOS from {oos_start} ({tag})...")
            result = run_single_oos(prices, tickers, rpavg_weights,
                                    oos_start, tag)
            exp_results.append(result)

        all_results[exp_name] = exp_results

    # ---- Summary ----
    print(f"\n\n{'=' * 80}")
    print("SUMMARY — All experiments")
    print(f"{'=' * 80}")

    # Also show the 6-asset production baseline for comparison
    print(f"\n{'Experiment':<25} {'Split':<10} {'CAGR':>6} {'MaxDD':>7} "
          f"{'Calmar':>7} {'Ulcer':>6} {'Sortino':>8} {'B&H Cal':>8}")
    print("-" * 80)

    for exp_name, results in all_results.items():
        for r in results:
            print(f"{exp_name:<25} {r['split']:<10} "
                  f"{r['cagr']:>5.2f}% {r['max_dd']:>6.2f}% "
                  f"{r['calmar']:>7.3f} {r['ulcer']:>6.2f} "
                  f"{r['sortino']:>8.3f} {r['bh_calmar']:>8.3f}")
        print()

    print("\nFor reference — 6-asset production (rpavg):")
    print("  2018oos: CAGR 7.51%, MaxDD -16.60%, Calmar 0.452, Ulcer 4.83, Sortino 0.726")
    print("  2020oos: CAGR 7.58%, MaxDD -16.60%, Calmar 0.457, Ulcer 5.46, Sortino 0.717")
    print("  2022oos: CAGR 5.99%, MaxDD -15.91%, Calmar 0.376, Ulcer 6.05, Sortino 0.426")

    # Verdict
    print(f"\n{'=' * 80}")
    print("VERDICT")
    print(f"{'=' * 80}")
    for exp_name, results in all_results.items():
        avg_calmar = np.mean([r["calmar"] for r in results])
        avg_ulcer = np.mean([r["ulcer"] for r in results])
        print(f"  {exp_name}: avg Calmar = {avg_calmar:.3f}, avg Ulcer = {avg_ulcer:.2f}")
    print(f"  6-asset production:  avg Calmar = {np.mean([0.452, 0.457, 0.376]):.3f}, "
          f"avg Ulcer = {np.mean([4.83, 5.46, 6.05]):.2f}")


if __name__ == "__main__":
    main()
