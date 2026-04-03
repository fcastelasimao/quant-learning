"""
run_overlay_grid.py
===================
Grid search over SPY overlay parameters on IS data (2006-2020).
Then validate the best combination on OOS (2020-2026).

Tests: d_window × threshold × reduce_pct combinations.
Picks the one with the best IS Calmar, then runs it OOS.

Run with:
    conda run -n allweather python3 run_overlay_grid.py
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

import config
from data import fetch_prices
from backtest import run_backtest_with_overlay, compute_stats

# ---- Grid parameters (IS tuning) ----

D_WINDOWS   = [5, 10, 15, 20, 30, 40, 60]    # trading days
THRESHOLDS  = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]  # drawdown %
REDUCE_PCTS = [0.50, 0.75, 1.00]              # fraction to exit

# RP weights (averaged across 3 splits)
ALLOCATION = {
    "SPY": 0.13, "QQQ": 0.11, "TLT": 0.19,
    "TIP": 0.33, "GLD": 0.14, "GSG": 0.10,
}

IS_END  = "2020-01-01"
OOS_END = None  # uses BACKTEST_END (today)


def run_overlay_test(prices, bench, tlt, allocation, d_window, threshold, reduce_pct,
                     start_date, end_date):
    """Run a single overlay backtest and return AW_R Calmar."""
    # Set overlay config
    config.ASSET_OVERLAYS["SPY"]["enabled"] = True
    config.ASSET_OVERLAYS["SPY"]["threshold"] = threshold
    config.ASSET_OVERLAYS["SPY"]["d_window"] = d_window
    config.ASSET_OVERLAYS["SPY"]["reduce_pct"] = reduce_pct

    # Slice to date range
    mask = (prices.index >= start_date) & (prices.index < end_date)
    p = prices.loc[mask]
    b = bench.loc[mask]
    t = tlt.loc[mask]

    backtest = run_backtest_with_overlay(p, b, allocation, tlt_prices=t)
    stats = compute_stats(backtest, prices=p[list(allocation.keys())], allocation=allocation)
    aw = next(s for s in stats if s.name == "AW_R")
    return aw.calmar, aw.max_drawdown_daily, aw.cagr, aw.max_drawdown


def main():
    print("=" * 70)
    print("SPY OVERLAY GRID SEARCH — IS tuning, then OOS validation")
    print("=" * 70)

    # Fetch all prices once
    tickers = list(ALLOCATION.keys()) + ["TLT"]
    tickers = list(dict.fromkeys(tickers))
    prices = fetch_prices(tickers, config.BACKTEST_START, config.BACKTEST_END)

    port_prices = prices[list(ALLOCATION.keys())]
    bench = prices["SPY"]
    tlt = prices["TLT"]
    oos_end = config.BACKTEST_END

    # --- Baseline: no overlay on IS ---
    config.ASSET_OVERLAYS["SPY"]["enabled"] = False
    is_mask = (port_prices.index >= config.BACKTEST_START) & (port_prices.index < IS_END)
    bt_base = run_backtest_with_overlay(
        port_prices.loc[is_mask], bench.loc[is_mask], ALLOCATION,
        tlt_prices=tlt.loc[is_mask])
    stats_base = compute_stats(bt_base)
    aw_base = next(s for s in stats_base if s.name == "AW_R")
    print(f"\nIS Baseline (no overlay): Calmar={aw_base.calmar:.3f}  "
          f"CAGR={aw_base.cagr:.2f}%  MDD={aw_base.max_drawdown:.2f}%")

    # --- Grid search on IS ---
    total = len(D_WINDOWS) * len(THRESHOLDS) * len(REDUCE_PCTS)
    print(f"\nSearching {total} parameter combinations on IS (2006-2020)...\n")

    results = []
    count = 0
    for dw in D_WINDOWS:
        for th in THRESHOLDS:
            for rp in REDUCE_PCTS:
                count += 1
                try:
                    cal, mdd_d, cagr, mdd = run_overlay_test(
                        port_prices, bench, tlt, ALLOCATION,
                        dw, th, rp,
                        config.BACKTEST_START, IS_END)
                    results.append({
                        "d_window": dw, "threshold": th, "reduce_pct": rp,
                        "is_calmar": cal, "is_mdd": mdd, "is_cagr": cagr,
                    })
                    if count % 20 == 0 or count == total:
                        print(f"  [{count}/{total}] d={dw} th={th:.0%} rp={rp:.0%} → "
                              f"Calmar={cal:.3f}")
                except Exception as e:
                    print(f"  [{count}/{total}] d={dw} th={th:.0%} rp={rp:.0%} → ERROR: {e}")

    results.sort(key=lambda x: x["is_calmar"], reverse=True)

    # --- Print top 10 IS results ---
    print(f"\n{'=' * 70}")
    print(f"TOP 10 IS RESULTS (baseline Calmar={aw_base.calmar:.3f})")
    print(f"{'=' * 70}")
    print(f"{'Rank':<5} {'D_win':>5} {'Thresh':>7} {'Red%':>5} {'Calmar':>8} {'CAGR':>7} {'MDD':>8}")
    print("-" * 50)
    for i, r in enumerate(results[:10]):
        beat = "✓" if r["is_calmar"] > aw_base.calmar else " "
        print(f"{i+1:<5} {r['d_window']:>5} {r['threshold']:>6.0%} {r['reduce_pct']:>5.0%} "
              f"{r['is_calmar']:>7.3f}{beat} {r['is_cagr']:>6.2f}% {r['is_mdd']:>7.2f}%")

    # --- Count how many beat baseline ---
    n_beat = sum(1 for r in results if r["is_calmar"] > aw_base.calmar)
    print(f"\n{n_beat}/{len(results)} combinations beat the baseline on IS.")


    # Save full grid results to CSV
    import csv, os
    os.makedirs("results", exist_ok=True)
    out_path = "results/overlay_grid_results.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["d_window", "threshold", "reduce_pct",
                                           "is_calmar", "is_mdd", "is_cagr"])
        w.writeheader()
        w.writerows(results)
    print(f"\nGrid results saved to {out_path}")


    if n_beat == 0:
        print("\nNo overlay parameters beat the baseline. Overlay does not help.")
        return

    # --- 3-split OOS validation of top IS performer ---
    print(f"\n{'=' * 70}")
    print("3-SPLIT OOS VALIDATION — Best IS parameters")
    print(f"{'=' * 70}")

    best = results[0]
    print(f"\nBest IS params: d={best['d_window']} th={best['threshold']:.0%} "
          f"rp={best['reduce_pct']:.0%}  IS Calmar={best['is_calmar']:.3f}")

    splits = [
        {"oos_start": "2020-01-01", "tag": "2020"},
        {"oos_start": "2018-01-01", "tag": "2018"},
        {"oos_start": "2022-01-01", "tag": "2022"},
    ]

    for split in splits:
        oos_start = split["oos_start"]

        # Baseline (no overlay) for this split
        config.ASSET_OVERLAYS["SPY"]["enabled"] = False
        s_mask = (port_prices.index >= oos_start) & (port_prices.index < oos_end)
        bt_b = run_backtest_with_overlay(
            port_prices.loc[s_mask], bench.loc[s_mask], ALLOCATION,
            tlt_prices=tlt.loc[s_mask])
        st_b = compute_stats(bt_b, prices=port_prices.loc[s_mask],
                             allocation=ALLOCATION)
        aw_b = next(s for s in st_b if s.name == "AW_R")

        # Overlay for this split
        cal, mdd_d, cagr, mdd = run_overlay_test(
            port_prices, bench, tlt, ALLOCATION,
            best["d_window"], best["threshold"], best["reduce_pct"],
            oos_start, oos_end)

        beat = "✓ BEATS" if cal > aw_b.calmar else "✗ LOSES"
        print(f"\n  Split {split['tag']} (OOS from {oos_start}):")
        print(f"    Baseline: Calmar={aw_b.calmar:.3f}  MDD={aw_b.max_drawdown:.2f}%")
        print(f"    Overlay:  Calmar={cal:.3f}  MDD={mdd:.2f}%  CAGR={cagr:.2f}%  {beat}")


if __name__ == "__main__":
    main()