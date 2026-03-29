"""
run_rp_validation.py
====================
Automated RP vs Manual validation across three OOS splits.

Produces 6 rows in the master log:
  - 3 splits × manual weights
  - 3 splits × RP weights (computed from IS-only data)

Run with:
    conda run -n allweather python3 run_rp_validation.py

Takes ~5 minutes. No config editing needed.
"""
import matplotlib
matplotlib.use("Agg")
import config
from data import fetch_prices
from backtest import run_backtest, compute_stats
from optimiser import compute_risk_parity_weights
from export import (make_results_dir, export_results,
                    append_to_master_log, start_run_log, stop_run_log)
from plotting import plot_backtest
import pandas as pd

# ---- Configuration ----

MANUAL_WEIGHTS = {
    "SPY": 0.15, "QQQ": 0.15, "TLT": 0.30,
    "TIP": 0.15, "GLD": 0.15, "GSG": 0.10,
}

SPLITS = [
    {"oos_start": "2020-01-01", "tag": "split2020"},
    {"oos_start": "2018-01-01", "tag": "split2018"},
    {"oos_start": "2022-01-01", "tag": "split2022"},
]

RP_ESTIMATION_YEARS = 5.0
TICKERS = list(MANUAL_WEIGHTS.keys())
ALL_TICKERS = list(dict.fromkeys(TICKERS + ["SPY", "TLT"]))


def run_single(prices, allocation, oos_start, tag, run_mode="oos_evaluate"):
    """Run a single backtest and log it."""
    # Override config for this run
    config.OOS_START = oos_start
    config.RUN_MODE = run_mode
    config.RUN_TAG = tag
    config.TARGET_ALLOCATION = allocation
    config.validate_config()

    # Determine date window
    if run_mode == "oos_evaluate":
        price_start, price_end = oos_start, config.BACKTEST_END
    else:
        price_start, price_end = config.BACKTEST_START, config.BACKTEST_END

    # Slice prices
    port_prices = prices[TICKERS]
    port_prices = port_prices[(port_prices.index >= price_start) &
                              (port_prices.index < price_end)]
    bench_prices = prices["SPY"][(prices["SPY"].index >= price_start) &
                                  (prices["SPY"].index < price_end)]
    tlt_prices = prices["TLT"][(prices["TLT"].index >= price_start) &
                                (prices["TLT"].index < price_end)]

    # Build label and results dir
    run_label = config._build_run_label(price_start, price_end)
    results_dir = make_results_dir(run_label)

    tee = start_run_log(results_dir)
    try:
        backtest = run_backtest(port_prices, bench_prices, allocation,
                                tlt_prices=tlt_prices)
        stats_list = compute_stats(backtest, prices=port_prices, allocation=allocation)

        export_results(backtest, pd.DataFrame(), stats_list, allocation, results_dir, run_label)
        append_to_master_log(results_dir, stats_list, allocation, run_label)
        plot_backtest(backtest, stats_list, results_dir, run_label,
                      allocation=allocation)

        # Extract AW_R Calmar for summary
        aw_stats = next(s for s in stats_list if s.name == "AW_R")
        return aw_stats.calmar, aw_stats.max_drawdown_daily
    finally:
        stop_run_log(tee)


def main():
    print("=" * 70)
    print("RP VALIDATION — 3 splits × 2 weight methods = 6 experiments")
    print("=" * 70)

    # Fetch all prices once
    prices = fetch_prices(ALL_TICKERS, config.BACKTEST_START, config.BACKTEST_END)

    results = []

    for split in SPLITS:
        oos_start = split["oos_start"]
        tag = split["tag"]
        print(f"\n{'=' * 70}")
        print(f"SPLIT: OOS starts {oos_start}")
        print(f"{'=' * 70}")

        # --- Manual weights ---
        print(f"\n--- Manual weights (OOS from {oos_start}) ---")
        cal_m, mdd_m = run_single(prices, dict(MANUAL_WEIGHTS),
                                   oos_start, f"manual_{tag}")
        results.append({"split": tag, "method": "manual",
                        "calmar": cal_m, "mdd_daily": mdd_m})

        # --- RP weights (computed from IS-only data) ---
        print(f"\n--- RP weights (covariance from IS only, end_date={oos_start}) ---")
        rp_weights = compute_risk_parity_weights(
            prices=prices,
            tickers=TICKERS,
            estimation_years=RP_ESTIMATION_YEARS,
            end_date=oos_start,
        )
        cal_r, mdd_r = run_single(prices, rp_weights,
                                   oos_start, f"rp5yr_{tag}")
        results.append({"split": tag, "method": "rp_5yr",
                        "calmar": cal_r, "mdd_daily": mdd_r})

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Split':<12} {'Method':<10} {'OOS Calmar':>12} {'Daily MDD':>12}")
    print("-" * 50)
    for r in results:
        print(f"{r['split']:<12} {r['method']:<10} {r['calmar']:>12.3f} {r['mdd_daily']:>11.2f}%")

    # Does RP beat manual on all three splits?
    rp_wins = sum(1 for i in range(0, len(results), 2)
                  if results[i+1]["calmar"] > results[i]["calmar"])
    print(f"\nRP beats manual on {rp_wins}/3 splits.")
    if rp_wins == 3:
        print(">>> RP is robust. Proceed to universe scan.")
    elif rp_wins >= 2:
        print(">>> RP is promising but not conclusive. Investigate the failing split.")
    else:
        print(">>> RP does not consistently beat manual. Reconsider methodology.")


if __name__ == "__main__":
    main()