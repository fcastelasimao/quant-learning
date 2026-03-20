"""
run_experiment.py
=================
Batch experiment runner for the All Weather portfolio project.

Runs the full IS → optimise → walk-forward → OOS → full-backtest pipeline for
a configurable list of asset universe experiments. Never modifies config.py on
disk; config overrides are applied in-memory and restored in a finally block.

Steps per experiment
--------------------
  1. IS backtest      (backtest mode,     BACKTEST_START → OOS_START)
  2. IS optimise      (optimise mode,     BACKTEST_START → OOS_START)
  3. Walk-forward     (walk_forward mode, BACKTEST_START → OOS_START)
  4. Confirmation     (human gate: yes / skip / quit)
  5. OOS evaluate     (oos_evaluate mode, OOS_START → BACKTEST_END)
  6. Full backtest    (full_backtest mode, BACKTEST_START → BACKTEST_END)

Usage
-----
    python3 run_experiment.py                             # all experiments
    python3 run_experiment.py --dry-run                  # preview only
    python3 run_experiment.py --experiments NAME [NAME ...]
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import argparse
import os
import sys
import time
import traceback
from typing import Optional

import numpy as np
import pandas as pd

import config
from data      import fetch_prices
from backtest  import run_backtest, compute_stats
from optimiser import optimise_allocation
from validation import run_walk_forward
from plotting  import plot_backtest
from export    import (make_results_dir, export_results, append_to_master_log,
                       print_header, print_stats, start_run_log, stop_run_log)

# ===========================================================================
# BASELINE REFERENCE
# ===========================================================================

BASELINE_NAME    = "8asset_manual_v1"
BASELINE_IS_CAL  = 0.336
BASELINE_OOS_CAL = 0.441


# ===========================================================================
# EXPERIMENTS
# ===========================================================================

EXPERIMENTS = [
    # ---- 5-asset: Dalio's original All Weather --------------------------------
    {
        "name":        "5asset_dalio_original",
        "description": "Dalio's original 5-asset All Weather (SPY/TLT/IEF/GLD/GSG)",
        "allocation": {
            "SPY": 0.300,
            "TLT": 0.400,
            "IEF": 0.150,
            "GLD": 0.075,
            "GSG": 0.075,
        },
        "asset_class_groups": {
            "stocks":             ["SPY"],
            "long_bonds":         ["TLT"],
            "intermediate_bonds": ["IEF"],
            "gold":               ["GLD"],
            "commodities":        ["GSG"],
        },
        "asset_class_max_weight": {
            "stocks":             0.50,
            "long_bonds":         0.50,
            "intermediate_bonds": 0.30,
            "gold":               0.20,
            "commodities":        0.20,
        },
    },

    # ---- 5-asset: previous best from TIP analysis ----------------------------
    {
        "name":        "5asset_tip_previous_best",
        "description": "5-asset: SPY/QQQ/TLT/TIP/GLD — previous best variant",
        "allocation": {
            "SPY": 0.142,
            "QQQ": 0.203,
            "TLT": 0.300,
            "TIP": 0.142,
            "GLD": 0.213,
        },
        "asset_class_groups": {
            "stocks":             ["SPY", "QQQ"],
            "long_bonds":         ["TLT"],
            "intermediate_bonds": ["TIP"],
            "gold":               ["GLD"],
        },
        "asset_class_max_weight": {
            "stocks":             0.45,
            "long_bonds":         0.40,
            "intermediate_bonds": 0.30,
            "gold":               0.30,
        },
    },

    # ---- 6-asset: add TIP (inflation-protected) and GSG (commodities) --------
    {
        "name":        "6asset_tip_gsg",
        "description": "6-asset: SPY/QQQ + TLT + TIP + GLD + GSG",
        "allocation": {
            "SPY": 0.150,
            "QQQ": 0.150,
            "TLT": 0.300,
            "TIP": 0.150,
            "GLD": 0.150,
            "GSG": 0.100,
        },
        "asset_class_groups": {
            "stocks":             ["SPY", "QQQ"],
            "long_bonds":         ["TLT"],
            "intermediate_bonds": ["TIP"],
            "gold":               ["GLD"],
            "commodities":        ["GSG"],
        },
        "asset_class_max_weight": {
            "stocks":             0.40,
            "long_bonds":         0.40,
            "intermediate_bonds": 0.25,
            "gold":               0.25,
            "commodities":        0.20,
        },
    },

    # ---- 6-asset: bond duration ladder (no commodities) ---------------------
    {
        "name":        "6asset_duration_ladder",
        "description": "6-asset: SPY/QQQ + TLT/IEF/SHY duration ladder + GLD",
        "allocation": {
            "SPY": 0.200,
            "QQQ": 0.150,
            "TLT": 0.250,
            "IEF": 0.150,
            "SHY": 0.100,
            "GLD": 0.150,
        },
        "asset_class_groups": {
            "stocks":             ["SPY", "QQQ"],
            "long_bonds":         ["TLT"],
            "intermediate_bonds": ["IEF", "SHY"],
            "gold":               ["GLD"],
        },
        "asset_class_max_weight": {
            "stocks":             0.45,
            "long_bonds":         0.40,
            "intermediate_bonds": 0.35,
            "gold":               0.25,
        },
    },

    # ---- 6-asset: international equity added --------------------------------
    {
        "name":        "6asset_intl_equity",
        "description": "6-asset: SPY/EFA (intl) + TLT/IEF + GLD + GSG",
        "allocation": {
            "SPY": 0.150,
            "EFA": 0.100,
            "TLT": 0.250,
            "IEF": 0.150,
            "GLD": 0.200,
            "GSG": 0.150,
        },
        "asset_class_groups": {
            "stocks":             ["SPY", "EFA"],
            "long_bonds":         ["TLT"],
            "intermediate_bonds": ["IEF"],
            "gold":               ["GLD"],
            "commodities":        ["GSG"],
        },
        "asset_class_max_weight": {
            "stocks":             0.40,
            "long_bonds":         0.40,
            "intermediate_bonds": 0.25,
            "gold":               0.30,
            "commodities":        0.25,
        },
    },

    # ---- 7-asset: 8-asset baseline minus GSG (commodities removed) ----------
    {
        "name":        "7asset_no_gsg",
        "description": "7-asset: 8-asset baseline without GSG (commodities removed)",
        "allocation": {
            "SPY": 0.100,
            "QQQ": 0.150,
            "IWD": 0.100,
            "TLT": 0.250,
            "IEF": 0.150,
            "SHY": 0.050,
            "GLD": 0.200,
        },
        "asset_class_groups": {
            "stocks":             ["SPY", "QQQ", "IWD"],
            "long_bonds":         ["TLT"],
            "intermediate_bonds": ["IEF", "SHY"],
            "gold":               ["GLD"],
        },
        "asset_class_max_weight": {
            "stocks":             0.40,
            "long_bonds":         0.40,
            "intermediate_bonds": 0.25,
            "gold":               0.25,
        },
    },

    # ---- 7-asset: 8-asset baseline minus IWD (value equity removed) ---------
    {
        "name":        "7asset_no_iwd",
        "description": "7-asset: 8-asset baseline without IWD (US value equity removed)",
        "allocation": {
            "SPY": 0.150,
            "QQQ": 0.150,
            "TLT": 0.250,
            "IEF": 0.100,
            "SHY": 0.050,
            "GLD": 0.150,
            "GSG": 0.150,
        },
        "asset_class_groups": {
            "stocks":             ["SPY", "QQQ"],
            "long_bonds":         ["TLT"],
            "intermediate_bonds": ["IEF", "SHY"],
            "gold":               ["GLD"],
            "commodities":        ["GSG"],
        },
        "asset_class_max_weight": {
            "stocks":             0.40,
            "long_bonds":         0.40,
            "intermediate_bonds": 0.25,
            "gold":               0.20,
            "commodities":        0.20,
        },
    },

    # ---- 7-asset: TIP/GSG dual addition (SPY/QQQ + bond trio + GLD + GSG) --
    {
        "name":        "7asset_tip_gsg_dual",
        "description": "7-asset: SPY/QQQ + TLT/IEF/TIP + GLD + GSG",
        "allocation": {
            "SPY": 0.100,
            "QQQ": 0.150,
            "TLT": 0.250,
            "IEF": 0.100,
            "TIP": 0.100,
            "GLD": 0.200,
            "GSG": 0.100,
        },
        "asset_class_groups": {
            "stocks":             ["SPY", "QQQ"],
            "long_bonds":         ["TLT"],
            "intermediate_bonds": ["IEF", "TIP"],
            "gold":               ["GLD"],
            "commodities":        ["GSG"],
        },
        "asset_class_max_weight": {
            "stocks":             0.40,
            "long_bonds":         0.40,
            "intermediate_bonds": 0.25,
            "gold":               0.25,
            "commodities":        0.20,
        },
    },

    # ---- 8-asset: EFA replaces IWD ------------------------------------------
    {
        "name":        "8asset_efa_replaces_iwd",
        "description": "8-asset: EFA (intl developed equity) instead of IWD (US value)",
        "allocation": {
            "SPY": 0.100,
            "QQQ": 0.150,
            "EFA": 0.100,
            "TLT": 0.250,
            "IEF": 0.100,
            "SHY": 0.050,
            "GLD": 0.150,
            "GSG": 0.100,
        },
        "asset_class_groups": {
            "stocks":             ["SPY", "QQQ", "EFA"],
            "long_bonds":         ["TLT"],
            "intermediate_bonds": ["IEF", "SHY"],
            "gold":               ["GLD"],
            "commodities":        ["GSG"],
        },
        "asset_class_max_weight": {
            "stocks":             0.40,
            "long_bonds":         0.40,
            "intermediate_bonds": 0.25,
            "gold":               0.20,
            "commodities":        0.20,
        },
    },

    # ---- 8-asset: TIP replaces SHY ------------------------------------------
    {
        "name":        "8asset_tip_replaces_shy",
        "description": "8-asset: TIP (inflation-protected bonds) instead of SHY (short-term)",
        "allocation": {
            "SPY": 0.100,
            "QQQ": 0.150,
            "IWD": 0.100,
            "TLT": 0.250,
            "IEF": 0.100,
            "TIP": 0.050,
            "GLD": 0.150,
            "GSG": 0.100,
        },
        "asset_class_groups": {
            "stocks":             ["SPY", "QQQ", "IWD"],
            "long_bonds":         ["TLT"],
            "intermediate_bonds": ["IEF", "TIP"],
            "gold":               ["GLD"],
            "commodities":        ["GSG"],
        },
        "asset_class_max_weight": {
            "stocks":             0.40,
            "long_bonds":         0.40,
            "intermediate_bonds": 0.25,
            "gold":               0.20,
            "commodities":        0.20,
        },
    },
]


# ===========================================================================
# HELPERS
# ===========================================================================

def _read_wf_summary(results_dir: str) -> dict:
    """
    Read walk_forward.csv from results_dir and return a summary dict.

    Returns
    -------
    dict with keys: mean_ratio, median_ratio, opt_beat, n_windows, verdict
    """
    csv_path = os.path.join(results_dir, "walk_forward.csv")
    if not os.path.exists(csv_path):
        return {
            "mean_ratio": float("nan"), "median_ratio": float("nan"),
            "opt_beat": 0, "n_windows": 0, "verdict": "no_data",
        }

    df = pd.read_csv(csv_path)

    OVERFIT_CAP = 2.0
    if "Overfit Ratio Clamped" in df.columns:
        ratios = df["Overfit Ratio Clamped"]
    elif "Overfit Ratio" in df.columns:
        ratios = df["Overfit Ratio"].clip(upper=OVERFIT_CAP)
    else:
        return {
            "mean_ratio": float("nan"), "median_ratio": float("nan"),
            "opt_beat": 0, "n_windows": 0, "verdict": "no_data",
        }

    mean_ratio   = float(ratios.mean())
    median_ratio = float(ratios.median())

    if "Opt Beat Original" in df.columns:
        opt_beat = int((df["Opt Beat Original"] == True).sum())  # noqa: E712
    elif ("Optimised Test Calmar" in df.columns
          and "Original Test Calmar" in df.columns):
        opt_beat = int(
            (df["Optimised Test Calmar"] > df["Original Test Calmar"]).sum()
        )
    else:
        opt_beat = 0

    n_windows = len(df)

    if mean_ratio >= 0.7:
        verdict = "robust"
    elif mean_ratio >= 0.5:
        verdict = "moderate"
    else:
        verdict = "overfit"

    return {
        "mean_ratio":   mean_ratio,
        "median_ratio": median_ratio,
        "opt_beat":     opt_beat,
        "n_windows":    n_windows,
        "verdict":      verdict,
    }


def _confirmation_gate(exp_name: str,
                       is_stats: list,
                       opt_alloc: dict,
                       wf_summary: dict,
                       auto_yes: bool = False) -> str:
    """
    Print a pre-OOS summary and ask the user whether to proceed.

    Returns one of: "yes", "skip", "quit"

    If auto_yes is True the prompt is skipped and "yes" is returned
    automatically, enabling unattended overnight runs.
    """
    print("\n" + "=" * 68)
    print(f"  PRE-OOS REVIEW — {exp_name}")
    print("=" * 68)

    if is_stats:
        aw = is_stats[0]
        print(f"  IS Backtest:  CAGR={aw.cagr:.2f}%  "
              f"MaxDD={aw.max_drawdown:.2f}%  Calmar={aw.calmar:.3f}")

    if opt_alloc:
        print(f"\n  Optimised weights:")
        for t, w in opt_alloc.items():
            print(f"    {t:5s}  {w:.1%}")

    if wf_summary.get("n_windows", 0) > 0:
        print(f"\n  Walk-forward:  "
              f"mean_ratio={wf_summary['mean_ratio']:.3f}  "
              f"median_ratio={wf_summary['median_ratio']:.3f}  "
              f"opt_beat={wf_summary['opt_beat']}/{wf_summary['n_windows']}  "
              f"verdict={wf_summary['verdict']}")

    if auto_yes:
        print("\n  --auto-yes flag set: proceeding to OOS automatically.")
        return "yes"

    print("\n  Proceed to OOS evaluation?")
    print("  [yes / skip / quit]", flush=True)
    try:
        answer = input("  > ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n  (stdin closed / interrupted — proceeding to OOS automatically)")
        return "yes"

    if answer in ("yes", "y"):
        return "yes"
    if answer in ("skip", "s"):
        return "skip"
    if answer in ("quit", "q", "exit"):
        return "quit"

    print(f"  Unrecognised input '{answer}', treating as skip.")
    return "skip"


# ===========================================================================
# PER-EXPERIMENT RUNNER
# ===========================================================================

def run_experiment(exp: dict, auto_yes: bool = False) -> dict:
    """
    Run the full 6-step validated workflow for one experiment.

    Returns a result dict for inclusion in the summary table.

    Config overrides are applied in-memory and always restored in the finally
    block — config.py is never touched on disk.

    If auto_yes is True the confirmation gate is bypassed and OOS always runs.
    """
    name  = exp["name"]
    alloc = exp["allocation"]

    assert abs(sum(alloc.values()) - 1.0) < 1e-6, (
        f"Allocation for '{name}' sums to {sum(alloc.values()):.6f}, not 1.0"
    )

    row: dict = {
        "name":          name,
        "n_assets":      len(alloc),
        "is_calmar":     float("nan"),
        "opt_calmar":    float("nan"),
        "wf_mean_ratio": float("nan"),
        "wf_med_ratio":  float("nan"),
        "oos_calmar":    float("nan"),
        "status":        "pending",
        "elapsed_s":     0.0,
    }

    t_exp_start = time.time()

    # ---- Save original config values ----------------------------------------
    _orig_allocation    = config.TARGET_ALLOCATION
    _orig_groups        = config.ASSET_CLASS_GROUPS
    _orig_max_weight    = config.ASSET_CLASS_MAX_WEIGHT
    _orig_run_mode      = config.RUN_MODE

    try:
        # ---- Apply in-memory overrides --------------------------------------
        config.TARGET_ALLOCATION      = alloc
        config.ASSET_CLASS_GROUPS     = exp["asset_class_groups"]
        config.ASSET_CLASS_MAX_WEIGHT = exp["asset_class_max_weight"]

        # ------------------------------------------------------------------
        # Fetch prices once for the full date range, then slice per step
        # ------------------------------------------------------------------
        all_tickers = list(dict.fromkeys(
            list(alloc.keys()) + [config.BENCHMARK_TICKER, "TLT"]
        ))
        print(f"\nFetching prices: {' '.join(all_tickers)}")
        prices_full = fetch_prices(
            all_tickers, config.BACKTEST_START, config.BACKTEST_END
        )

        port_cols = list(alloc.keys())
        bench_col = config.BENCHMARK_TICKER

        is_mask   = ((prices_full.index >= config.BACKTEST_START) &
                     (prices_full.index <  config.OOS_START))
        oos_mask  = ((prices_full.index >= config.OOS_START) &
                     (prices_full.index <  config.BACKTEST_END))
        full_mask = ((prices_full.index >= config.BACKTEST_START) &
                     (prices_full.index <  config.BACKTEST_END))

        is_port   = prices_full.loc[is_mask,   port_cols]
        is_bench  = prices_full.loc[is_mask,   bench_col]
        is_tlt    = prices_full.loc[is_mask,   "TLT"]
        oos_port  = prices_full.loc[oos_mask,  port_cols]
        oos_bench = prices_full.loc[oos_mask,  bench_col]
        oos_tlt   = prices_full.loc[oos_mask,  "TLT"]
        full_port  = prices_full.loc[full_mask, port_cols]
        full_bench = prices_full.loc[full_mask, bench_col]
        full_tlt   = prices_full.loc[full_mask, "TLT"]

        # ------------------------------------------------------------------
        # Step 1: IS backtest
        # ------------------------------------------------------------------
        t_step = time.time()
        config.RUN_MODE = "backtest"
        step_label  = f"exp_{name}_s1_backtest"
        results_dir = make_results_dir(step_label)

        tee = None
        try:
            tee = start_run_log(results_dir)
            print_header(f"STEP 1/6  IS BACKTEST — {name}")
            backtest_is = run_backtest(
                is_port, is_bench, alloc,
                tlt_prices           = is_tlt,
                transaction_cost_pct = config.TRANSACTION_COST_PCT,
                tax_drag_pct         = config.TAX_DRAG_PCT,
            )
            is_stats = compute_stats(backtest_is)
            print_stats(is_stats)
            export_results(backtest_is, pd.DataFrame(), is_stats,
                           alloc, results_dir, step_label)
            append_to_master_log(results_dir, is_stats, alloc, step_label)
            plot_backtest(backtest_is, is_stats, results_dir, step_label, alloc)
            row["is_calmar"] = is_stats[0].calmar
        finally:
            if tee is not None:
                stop_run_log(tee)

        print(f"  Step 1 complete in {time.time() - t_step:.1f}s")

        # ------------------------------------------------------------------
        # Step 2: IS optimise
        # ------------------------------------------------------------------
        t_step = time.time()
        config.RUN_MODE = "optimise"
        step_label  = f"exp_{name}_s2_optimise"
        results_dir = make_results_dir(step_label)

        opt_alloc = dict(alloc)   # will be updated to optimised weights below
        tee = None
        try:
            tee = start_run_log(results_dir)
            print_header(f"STEP 2/6  IS OPTIMISE — {name}")
            optimised = optimise_allocation(
                prices           = is_port,
                benchmark_prices = is_bench,
                allocation       = dict(alloc),
                method           = config.OPT_METHOD,
                min_weight       = config.OPT_MIN_WEIGHT,
                max_weight       = config.OPT_MAX_WEIGHT,
                min_cagr         = config.OPT_MIN_CAGR,
                n_trials         = config.OPT_N_TRIALS,
                random_seed      = config.OPT_RANDOM_SEED,
            )
            opt_alloc.update(optimised)

            # Backtest the optimised weights to record IS Calmar
            bt_opt = run_backtest(
                is_port, is_bench, opt_alloc,
                tlt_prices           = is_tlt,
                transaction_cost_pct = config.TRANSACTION_COST_PCT,
                tax_drag_pct         = config.TAX_DRAG_PCT,
            )
            opt_stats = compute_stats(bt_opt)
            print_stats(opt_stats)
            export_results(bt_opt, pd.DataFrame(), opt_stats,
                           opt_alloc, results_dir, step_label)
            append_to_master_log(results_dir, opt_stats, opt_alloc, step_label)
            plot_backtest(bt_opt, opt_stats, results_dir, step_label, opt_alloc)
            row["opt_calmar"] = opt_stats[0].calmar
        finally:
            if tee is not None:
                stop_run_log(tee)

        print(f"  Step 2 complete in {time.time() - t_step:.1f}s")

        # ------------------------------------------------------------------
        # Step 3: Walk-forward validation
        # ------------------------------------------------------------------
        t_step = time.time()
        config.RUN_MODE = "walk_forward"
        step_label = f"exp_{name}_s3_walkforward"
        wf_dir     = make_results_dir(step_label)

        wf_summary: dict = {
            "mean_ratio": float("nan"), "median_ratio": float("nan"),
            "opt_beat": 0, "n_windows": 0, "verdict": "no_data",
        }
        tee = None
        try:
            tee = start_run_log(wf_dir)
            print_header(f"STEP 3/6  WALK-FORWARD — {name}")
            run_walk_forward(
                prices           = is_port,
                benchmark_prices = is_bench,
                allocation       = dict(alloc),
                train_years      = config.WF_TRAIN_YEARS,
                test_years       = config.WF_TEST_YEARS,
                step_years       = config.WF_STEP_YEARS,
                min_weight       = config.OPT_MIN_WEIGHT,
                max_weight       = config.OPT_MAX_WEIGHT,
                n_trials         = config.OPT_N_TRIALS,
                random_seed      = config.OPT_RANDOM_SEED,
                results_dir      = wf_dir,
                tlt_prices       = is_tlt,
            )
            wf_summary = _read_wf_summary(wf_dir)
            row["wf_mean_ratio"] = wf_summary["mean_ratio"]
            row["wf_med_ratio"]  = wf_summary["median_ratio"]
        finally:
            if tee is not None:
                stop_run_log(tee)

        print(f"  Step 3 complete in {time.time() - t_step:.1f}s")

        # ------------------------------------------------------------------
        # Step 4: Human confirmation gate
        # ------------------------------------------------------------------
        answer = _confirmation_gate(name, is_stats, opt_alloc, wf_summary,
                                    auto_yes=auto_yes)
        if answer == "quit":
            row["status"] = "quit"
            return row
        if answer == "skip":
            row["status"] = "skipped"
            return row

        # ------------------------------------------------------------------
        # Step 5: OOS evaluate (with optimised weights)
        # ------------------------------------------------------------------
        t_step = time.time()
        config.RUN_MODE = "oos_evaluate"
        step_label  = f"exp_{name}_s4_oos"
        results_dir = make_results_dir(step_label)

        tee = None
        try:
            tee = start_run_log(results_dir)
            print_header(f"STEP 5/6  OOS EVALUATE — {name}")
            backtest_oos = run_backtest(
                oos_port, oos_bench, opt_alloc,
                tlt_prices           = oos_tlt,
                transaction_cost_pct = config.TRANSACTION_COST_PCT,
                tax_drag_pct         = config.TAX_DRAG_PCT,
            )
            oos_stats = compute_stats(backtest_oos)
            print_stats(oos_stats)
            export_results(backtest_oos, pd.DataFrame(), oos_stats,
                           opt_alloc, results_dir, step_label)
            append_to_master_log(results_dir, oos_stats, opt_alloc, step_label)
            plot_backtest(backtest_oos, oos_stats, results_dir, step_label, opt_alloc)
            row["oos_calmar"] = oos_stats[0].calmar
        finally:
            if tee is not None:
                stop_run_log(tee)

        print(f"  Step 5 complete in {time.time() - t_step:.1f}s")

        # ------------------------------------------------------------------
        # Step 6: Full backtest (with optimised weights)
        # ------------------------------------------------------------------
        t_step = time.time()
        config.RUN_MODE = "full_backtest"
        step_label  = f"exp_{name}_s5_full"
        results_dir = make_results_dir(step_label)

        tee = None
        try:
            tee = start_run_log(results_dir)
            print_header(f"STEP 6/6  FULL BACKTEST — {name}")
            backtest_full = run_backtest(
                full_port, full_bench, opt_alloc,
                tlt_prices           = full_tlt,
                transaction_cost_pct = config.TRANSACTION_COST_PCT,
                tax_drag_pct         = config.TAX_DRAG_PCT,
            )
            full_stats = compute_stats(backtest_full)
            print_stats(full_stats)
            export_results(backtest_full, pd.DataFrame(), full_stats,
                           opt_alloc, results_dir, step_label)
            append_to_master_log(results_dir, full_stats, opt_alloc, step_label)
            plot_backtest(backtest_full, full_stats, results_dir, step_label, opt_alloc)
        finally:
            if tee is not None:
                stop_run_log(tee)

        print(f"  Step 6 complete in {time.time() - t_step:.1f}s")

        row["status"] = "done"

    except Exception as exc:
        row["status"] = f"error: {exc}"
        traceback.print_exc()

    finally:
        # Always restore config — even on early return or exception
        config.TARGET_ALLOCATION      = _orig_allocation
        config.ASSET_CLASS_GROUPS     = _orig_groups
        config.ASSET_CLASS_MAX_WEIGHT = _orig_max_weight
        config.RUN_MODE               = _orig_run_mode
        row["elapsed_s"] = time.time() - t_exp_start

    return row


# ===========================================================================
# SUMMARY
# ===========================================================================

def _print_summary(rows: list[dict], output_path: str) -> None:
    """Print and save the final experiment summary table."""

    def _f(v: float) -> str:
        return f"{v:5.3f}" if not (isinstance(v, float) and np.isnan(v)) else "  n/a"

    sep    = "=" * 80
    header = (
        f"{sep}\n"
        f"EXPERIMENT RESULTS SUMMARY\n"
        f"{sep}\n"
        f"Baseline ({BASELINE_NAME}):  "
        f"IS Cal={BASELINE_IS_CAL:.3f}   OOS Cal={BASELINE_OOS_CAL:.3f}\n"
        f"{sep}\n"
        f"{'Name':<32}  {'N':>2}  "
        f"{'IS(M)':>5}  {'IS(O)':>5}  "
        f"{'WF(mn)':>6}  {'WF(md)':>6}  "
        f"{'OOS':>5}  {'Status'}\n"
        f"{'-' * 80}"
    )

    lines = [header]
    for r in rows:
        line = (
            f"{r['name']:<32}  {r['n_assets']:>2}  "
            f"{_f(r['is_calmar'])}  {_f(r['opt_calmar'])}  "
            f"{_f(r['wf_mean_ratio'])}  {_f(r['wf_med_ratio'])}  "
            f"{_f(r['oos_calmar'])}  {r['status']}"
        )
        lines.append(line)

    lines.append(sep)
    table = "\n".join(lines)

    print("\n" + table)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(table + "\n")
    print(f"\nSummary saved to {output_path}")


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch experiment runner for All Weather portfolio variants."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview experiments without running anything, then exit.",
    )
    parser.add_argument(
        "--experiments", nargs="+", metavar="NAME",
        help="Run only these named experiments (default: run all).",
    )
    parser.add_argument(
        "--auto-yes", action="store_true",
        help="Automatically proceed to OOS for every experiment "
             "without prompting. Use for overnight unattended runs.",
    )
    args = parser.parse_args()

    # ---- Filter experiments -------------------------------------------------
    if args.experiments:
        requested = set(args.experiments)
        selected  = [e for e in EXPERIMENTS if e["name"] in requested]
        unknown   = requested - {e["name"] for e in selected}
        if unknown:
            print(f"WARNING: Unknown experiment name(s): {', '.join(sorted(unknown))}")
            print("Available names:")
            for e in EXPERIMENTS:
                print(f"  {e['name']}")
        if not selected:
            print("No matching experiments found. Exiting.")
            sys.exit(1)
    else:
        selected = EXPERIMENTS

    # ---- Dry run ------------------------------------------------------------
    if args.dry_run:
        print("DRY RUN — no experiments will be executed.\n")
        print(f"{'Name':<35}  {'N':>2}  {'Tickers'}")
        print("-" * 72)
        for exp in selected:
            tickers = " ".join(exp["allocation"].keys())
            n       = len(exp["allocation"])
            total   = sum(exp["allocation"].values())
            tag     = "" if abs(total - 1.0) < 1e-6 else "  *** BAD SUM ***"
            print(f"{exp['name']:<35}  {n:>2}  {tickers}{tag}")
        print(f"\n{len(selected)} experiment(s) would be run.")
        sys.exit(0)

    # ---- Run all selected experiments ---------------------------------------
    t_total = time.time()
    rows: list[dict] = []

    for i, exp in enumerate(selected, 1):
        print(f"\n{'#' * 72}")
        print(f"  EXPERIMENT {i}/{len(selected)}: {exp['name']}")
        print(f"  {exp.get('description', '')}")
        print(f"{'#' * 72}")

        row = run_experiment(exp, auto_yes=args.auto_yes)
        rows.append(row)

        elapsed_str = f"{row['elapsed_s']:.1f}s"
        if row["elapsed_s"] >= 60:
            elapsed_str += f"  ({row['elapsed_s'] / 60:.1f}min)"
        print(f"\n  Elapsed: {elapsed_str}  |  Status: {row['status']}")

        if row["status"] == "quit":
            print("\nUser requested quit. Stopping.")
            break

    # ---- Summary ------------------------------------------------------------
    total_elapsed = time.time() - t_total
    print(f"\nTotal elapsed: {total_elapsed:.1f}s  ({total_elapsed / 60:.1f}min)")

    summary_path = os.path.join("results", "experiment_summary.txt")
    _print_summary(rows, summary_path)


if __name__ == "__main__":
    main()
