"""
run_rolling_rp.py
=================
Automated experiment: Rolling Risk Parity vs Static RP.

For each OOS split (2020, 2018, 2022), runs:
  - Rolling RP : weights recomputed quarterly from a trailing 5-year covariance
  - Static RP  : weights fixed at the IS boundary (same methodology as production)

Also runs an IS sanity check (BACKTEST_START → 2020-01-01) to print the
in-sample Calmar and verify the rolling mechanism is working.

Full price history is always passed to run_backtest_rolling_rp so the
covariance lookback has access to IS data, even when evaluating OOS windows.
The resulting backtest is then filtered to the relevant date range for stats.

Outputs per run:
  - results/<timestamp>_<label>/  folder with backtest CSV, stats CSV, chart
  - weight_history_<tag>.csv      so you can see how weights drift over time
  - A row appended to results/master_log.xlsx

Run with:
    conda run -n allweather python3 run_rolling_rp.py
"""

import matplotlib
matplotlib.use("Agg")

import os

import pandas as pd

import config
from backtest import run_backtest, run_backtest_rolling_rp, compute_stats
from data import fetch_prices
from export import (
    append_to_master_log,
    export_results,
    make_results_dir,
    start_run_log,
    stop_run_log,
)
from optimiser import compute_risk_parity_weights
from plotting import plot_backtest

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TICKERS           = ["SPY", "QQQ", "TLT", "TIP", "GLD", "GSG"]
ALL_TICKERS       = list(dict.fromkeys(TICKERS + ["SPY", "TLT"]))  # deduplicated
RP_LOOKBACK_YEARS = 5.0
RP_RECOMPUTE_FREQ = "QS"   # quarterly

IS_END = "2020-01-01"      # default IS/OOS boundary for IS sanity check

SPLITS = [
    {"oos_start": "2020-01-01", "tag": "split2020"},
    {"oos_start": "2018-01-01", "tag": "split2018"},
    {"oos_start": "2022-01-01", "tag": "split2022"},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_weight_history(weight_history: list[dict], results_dir: str, tag: str) -> str:
    """Persist rolling RP weight history to a CSV. Returns the file path."""
    df   = pd.DataFrame(weight_history).set_index("date")
    path = os.path.join(results_dir, f"weight_history_{tag}.csv")
    df.to_csv(path)
    print(f"  Weight history → {path}  ({len(df)} recomputations)")
    return path


def _latest_weights(weight_history: list[dict]) -> dict:
    """Return the weights dict from the last recomputation (strips the 'date' key)."""
    last = weight_history[-1]
    return {k: v for k, v in last.items() if k != "date"}


# ---------------------------------------------------------------------------
# Run functions
# ---------------------------------------------------------------------------

def run_rolling_is(prices: pd.DataFrame) -> tuple[float, float, str]:
    """
    Run rolling RP over the full IS window and report IS Calmar.

    Passes the complete price history to run_backtest_rolling_rp (correct),
    then filters the resulting backtest to [BACKTEST_START, IS_END) for stats.

    Returns (IS Calmar, IS daily MDD, results_dir).
    """
    config.OOS_START = IS_END
    config.RUN_MODE  = "full_backtest"
    config.RUN_TAG   = "rolling_rp_is"
    config.validate_config()

    run_label   = config._build_run_label(config.BACKTEST_START, IS_END)
    results_dir = make_results_dir(run_label)

    tee = start_run_log(results_dir)
    try:
        backtest_full, weight_history = run_backtest_rolling_rp(
            prices, prices["SPY"], TICKERS,
            tlt_prices=prices["TLT"],
            rp_lookback_years=RP_LOOKBACK_YEARS,
            rp_recompute_freq=RP_RECOMPUTE_FREQ,
        )

        # Filter to IS window
        backtest_is = backtest_full[backtest_full.index < IS_END].copy()
        wh_is = [w for w in weight_history
                 if pd.Timestamp(w["date"]) < pd.Timestamp(IS_END)]

        last_weights = _latest_weights(wh_is if wh_is else weight_history)
        stats_list   = compute_stats(backtest_is, prices=prices, allocation=last_weights)

        save_weight_history(wh_is if wh_is else weight_history, results_dir, "is")
        export_results(backtest_is, pd.DataFrame(), stats_list,
                       last_weights, results_dir, run_label)
        append_to_master_log(results_dir, stats_list, last_weights, run_label)
        plot_backtest(backtest_is, stats_list, results_dir, run_label,
                      allocation=last_weights)

        aw = next(s for s in stats_list if s.name == "AW_R")
        return aw.calmar, aw.max_drawdown_daily, results_dir
    finally:
        stop_run_log(tee)


def run_rolling_oos(prices: pd.DataFrame, oos_start: str, tag: str) -> tuple[float, float, str]:
    """
    Run rolling RP evaluated on the OOS window [oos_start, BACKTEST_END).

    Full price history is passed to run_backtest_rolling_rp so the 5-year
    covariance lookback can draw on IS data when computing initial weights.
    The backtest result is then filtered to [oos_start, ...) for stats.

    Returns (OOS Calmar, OOS daily MDD, results_dir).
    """
    config.OOS_START = oos_start
    config.RUN_MODE  = "oos_evaluate"
    config.RUN_TAG   = f"rolling_rp_{tag}"
    config.validate_config()

    run_label   = config._build_run_label(oos_start, config.BACKTEST_END)
    results_dir = make_results_dir(run_label)

    tee = start_run_log(results_dir)
    try:
        backtest_full, weight_history = run_backtest_rolling_rp(
            prices, prices["SPY"], TICKERS,
            tlt_prices=prices["TLT"],
            rp_lookback_years=RP_LOOKBACK_YEARS,
            rp_recompute_freq=RP_RECOMPUTE_FREQ,
        )

        # Filter to OOS window
        backtest_oos = backtest_full[backtest_full.index >= oos_start].copy()
        wh_oos = [w for w in weight_history
                  if pd.Timestamp(w["date"]) >= pd.Timestamp(oos_start)]

        last_weights = _latest_weights(wh_oos if wh_oos else weight_history)
        stats_list   = compute_stats(backtest_oos, prices=prices, allocation=last_weights)

        save_weight_history(wh_oos if wh_oos else weight_history, results_dir, tag)
        export_results(backtest_oos, pd.DataFrame(), stats_list,
                       last_weights, results_dir, run_label)
        append_to_master_log(results_dir, stats_list, last_weights, run_label)
        plot_backtest(backtest_oos, stats_list, results_dir, run_label,
                      allocation=last_weights)

        aw = next(s for s in stats_list if s.name == "AW_R")
        return aw.calmar, aw.max_drawdown_daily, results_dir
    finally:
        stop_run_log(tee)


def run_static_oos(prices: pd.DataFrame, oos_start: str, tag: str) -> tuple[float, float]:
    """
    Run static RP (weights fixed at IS boundary) on the OOS window.

    Identical IS/OOS discipline to run_rp_validation.py: covariance is
    computed from IS data only (end_date=oos_start), then applied to OOS prices.

    Returns (OOS Calmar, OOS daily MDD).
    """
    rp_weights = compute_risk_parity_weights(
        prices=prices,
        tickers=TICKERS,
        estimation_years=RP_LOOKBACK_YEARS,
        end_date=oos_start,
    )

    config.OOS_START         = oos_start
    config.RUN_MODE          = "oos_evaluate"
    config.RUN_TAG           = f"static_rp_{tag}"
    config.TARGET_ALLOCATION = rp_weights
    config.validate_config()

    port_prices  = prices[TICKERS]
    port_prices  = port_prices[(port_prices.index >= oos_start) &
                               (port_prices.index < config.BACKTEST_END)]
    bench_prices = prices["SPY"][(prices["SPY"].index >= oos_start) &
                                 (prices["SPY"].index < config.BACKTEST_END)]
    tlt_prices   = prices["TLT"][(prices["TLT"].index >= oos_start) &
                                 (prices["TLT"].index < config.BACKTEST_END)]

    run_label   = config._build_run_label(oos_start, config.BACKTEST_END)
    results_dir = make_results_dir(run_label)

    tee = start_run_log(results_dir)
    try:
        backtest   = run_backtest(port_prices, bench_prices, rp_weights,
                                  tlt_prices=tlt_prices)
        stats_list = compute_stats(backtest, prices=port_prices, allocation=rp_weights)

        export_results(backtest, pd.DataFrame(), stats_list,
                       rp_weights, results_dir, run_label)
        append_to_master_log(results_dir, stats_list, rp_weights, run_label)
        plot_backtest(backtest, stats_list, results_dir, run_label, allocation=rp_weights)

        aw = next(s for s in stats_list if s.name == "AW_R")
        return aw.calmar, aw.max_drawdown_daily
    finally:
        stop_run_log(tee)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("ROLLING RP vs STATIC RP")
    print(f"IS sanity: {config.BACKTEST_START} → {IS_END}")
    print(f"OOS splits: {[s['oos_start'] for s in SPLITS]}")
    print("=" * 70)

    prices = fetch_prices(ALL_TICKERS, config.BACKTEST_START, config.BACKTEST_END)

    # IS sanity check
    print(f"\n{'=' * 70}")
    print(f"IS SANITY CHECK: {config.BACKTEST_START} → {IS_END}")
    print(f"{'=' * 70}")
    cal_is, mdd_is, _ = run_rolling_is(prices)
    print(f"\n  IS Calmar: {cal_is:.3f}   IS Daily MDD: {mdd_is:.2f}%")

    # OOS splits
    results = []
    for split in SPLITS:
        oos_start = split["oos_start"]
        tag       = split["tag"]

        print(f"\n{'=' * 70}")
        print(f"OOS SPLIT: {oos_start} → {config.BACKTEST_END}")
        print(f"{'=' * 70}")

        print(f"\n--- Rolling RP (quarterly recompute, {RP_LOOKBACK_YEARS}yr lookback) ---")
        cal_roll, mdd_roll, _ = run_rolling_oos(prices, oos_start, tag)
        results.append({"split": tag, "method": "rolling_rp",
                        "calmar": cal_roll, "mdd_daily": mdd_roll})

        print(f"\n--- Static RP (IS-only covariance, weights fixed at {oos_start}) ---")
        cal_stat, mdd_stat = run_static_oos(prices, oos_start, tag)
        results.append({"split": tag, "method": "static_rp",
                        "calmar": cal_stat, "mdd_daily": mdd_stat})

    # Summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Split':<16} {'Method':<12} {'Calmar':>10} {'Daily MDD':>12}  Winner")
    print("-" * 62)

    rolling_wins = 0
    for i in range(0, len(results), 2):
        r_roll = results[i]
        r_stat = results[i + 1]
        winner = "ROLLING" if r_roll["calmar"] > r_stat["calmar"] else "STATIC"
        if winner == "ROLLING":
            rolling_wins += 1
        print(f"{r_roll['split']:<16} {'rolling_rp':<12} "
              f"{r_roll['calmar']:>10.3f} {r_roll['mdd_daily']:>11.2f}%")
        print(f"{'':16} {'static_rp':<12} "
              f"{r_stat['calmar']:>10.3f} {r_stat['mdd_daily']:>11.2f}%  {winner}")

    n = len(SPLITS)
    print(f"\nRolling RP beats Static RP on {rolling_wins}/{n} OOS splits.")
    if rolling_wins == n:
        print(">>> Rolling RP is robustly better. Consider for production.")
    elif rolling_wins >= 2:
        print(">>> Rolling RP has edge but not conclusive. Check the failing split.")
    else:
        print(">>> Rolling RP does not consistently beat Static RP. "
              "Static weights remain preferred.")


if __name__ == "__main__":
    main()
