"""
export.py
=========
Everything that writes files to disk.

Contains:
  - make_results_dir      create timestamped folder for this run
  - save_run_config       save all parameters as JSON for reproducibility
  - export_results        save backtest CSV, stats CSV, allocation CSV
  - build_log_row         build one summary row from a list of StrategyStats
  - append_to_master_log  append that row to results/master_log.csv
  - print_header          print a section divider to terminal
  - print_rebalancing     print formatted rebalancing instructions
  - print_stats           print formatted performance statistics

The print functions live here rather than in main.py because they are
output formatting -- closely related to export -- and keeping them here
avoids main.py becoming a dumping ground for miscellaneous functions.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import pandas as pd

from backtest import StrategyStats
import config


# ===========================================================================
# RESULTS DIRECTORY
# ===========================================================================

def make_results_dir(label: str) -> str:
    """
    Create a timestamped folder inside 'results/' for this run.
    Format: results/YYYY-MM-DD_HH-MM-SS_<label>/

    Every run gets its own folder so results are never overwritten.
    Returns the path to the created folder.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder    = os.path.join("results", f"{timestamp}_{label}")
    os.makedirs(folder, exist_ok=True)
    return folder


# ===========================================================================
# RUN CONFIG
# ===========================================================================

def save_run_config(allocation: dict, results_dir: str):
    """
    Save all user parameters and the final allocation to run_config.json.

    This file contains everything needed to reproduce the run exactly:
    copy these values back into config.py and re-run main.py.
    """
    config_data = {
        "run_label":               config.RUN_LABEL,
        "timestamp":               datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "backtest_start":          config.BACKTEST_START,
        "backtest_end":            config.BACKTEST_END,
        "initial_portfolio_value": config.INITIAL_PORTFOLIO_VALUE,
        "rebalance_threshold":     config.REBALANCE_THRESHOLD,
        "data_frequency":      config.DATA_FREQUENCY,
        "sharpe_annualisation": config.SHARPE_ANNUALISATION,
        "benchmark_ticker":        config.BENCHMARK_TICKER,
        "optimiser": {
            "run_optimiser": config.RUN_OPTIMISER,
            "method":        config.OPT_METHOD if config.RUN_OPTIMISER else "not run",
            "min_weight":    config.OPT_MIN_WEIGHT,
            "max_weight":    config.OPT_MAX_WEIGHT,
            "min_cagr":      config.OPT_MIN_CAGR,
            "n_trials":      config.OPT_N_TRIALS,
            "random_seed":   config.OPT_RANDOM_SEED,
        },
        "pareto": {
            "run_pareto":  config.RUN_PARETO,
            "cagr_range":  list(config.PARETO_CAGR_RANGE) if config.RUN_PARETO else [],
        },
        "walk_forward": {
            "run_walk_forward": config.RUN_WALK_FORWARD,
            "train_years":      config.WF_TRAIN_YEARS,
            "test_years":       config.WF_TEST_YEARS,
            "step_years":       config.WF_STEP_YEARS,
            "opt_method":       config.WF_OPT_METHOD,
        },
        "target_allocation": allocation,
    }

    path = os.path.join(results_dir, "run_config.json")
    with open(path, "w") as f:
        json.dump(config_data, f, indent=2)


# ===========================================================================
# RESULTS EXPORT
# ===========================================================================

def export_results(backtest: pd.DataFrame,
                   instructions: pd.DataFrame,
                   stats_list: list[StrategyStats],
                   allocation: dict,
                   results_dir: str):
    """
    Export all results to results_dir:
      backtest_history.csv        -- monthly portfolio values
      rebalancing_instructions.csv
      stats.csv                   -- CAGR, drawdown, Sharpe, Calmar per strategy
      allocation.csv              -- weights used in this run
      run_config.json             -- all parameters for reproduction
    """
    # Stats CSV -- built from StrategyStats dataclasses, no string parsing needed
    stats_rows = []
    for s in stats_list:
        stats_rows.extend([
            {"Strategy": s.name, "Metric": "Period (years)",   "Value": s.period_years},
            {"Strategy": s.name, "Metric": "CAGR (%)",         "Value": s.cagr},
            {"Strategy": s.name, "Metric": "Max Drawdown (%)", "Value": s.max_drawdown},
            {"Strategy": s.name, "Metric": "Sharpe Ratio",     "Value": s.sharpe},
            {"Strategy": s.name, "Metric": "Calmar Ratio",     "Value": s.calmar},
            {"Strategy": s.name, "Metric": "Final Value ($)",  "Value": s.final_value},
        ])

    pd.DataFrame(stats_rows).to_csv(
        os.path.join(results_dir, "stats.csv"), index=False)

    backtest.to_csv(
        os.path.join(results_dir, "backtest_history.csv"))

    instructions.to_csv(
        os.path.join(results_dir, "rebalancing_instructions.csv"), index=False)

    pd.DataFrame([
        {"Ticker": t, "Weight": w, "Weight (%)": f"{w:.1%}"}
        for t, w in allocation.items()
    ]).to_csv(os.path.join(results_dir, "allocation.csv"), index=False)

    save_run_config(allocation, results_dir)

    print(f"  backtest_history.csv")
    print(f"  rebalancing_instructions.csv")
    print(f"  stats.csv")
    print(f"  allocation.csv")
    print(f"  run_config.json")


# ===========================================================================
# MASTER LOG
# ===========================================================================

def build_log_row(results_dir: str,
                  stats_list: list[StrategyStats],
                  weights: dict,
                  label: str) -> dict:
    row = {
        "Timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Label":          label,
        "Backtest Start": config.BACKTEST_START,
        "Backtest End":   config.BACKTEST_END,
        "Data Frequency": config.DATA_FREQUENCY,
        "Tickers":        " | ".join(f"{t}={w:.1%}" for t, w in weights.items()),
    }
    for s in stats_list:
        prefix = s.name.replace(" ", "_").replace("&", "and")
        row[f"{prefix}_CAGR (%)"]        = s.cagr
        row[f"{prefix}_Max_DD (%)"]      = s.max_drawdown
        row[f"{prefix}_Sharpe"]          = s.sharpe
        row[f"{prefix}_Calmar"]          = s.calmar
        row[f"{prefix}_Final_Value ($)"] = s.final_value

    # Results Folder last so it doesn't push stat columns out of alignment
    row["Results Folder"] = results_dir
    return row

def append_to_master_log(results_dir: str,
                         stats_list: list[StrategyStats],
                         weights: dict,
                         label: str):
    """
    Append one row to results/master_log.csv.
    Creates the file with a header on the first run, appends on subsequent runs.
    Each row represents one complete run, making all runs comparable side by side.
    """
    log_path = os.path.join("results", "master_log.csv")
    row      = build_log_row(results_dir, stats_list, weights, label)
    log_df   = pd.DataFrame([row])

    if os.path.exists(log_path):
        log_df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        os.makedirs("results", exist_ok=True)
        log_df.to_csv(log_path, mode="w", header=True, index=False)

    print(f"  Master log updated -> {log_path}")


# ===========================================================================
# PRETTY PRINTING
# ===========================================================================

def print_header(title: str):
    """Print a clearly visible section divider to the terminal."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_rebalancing(instructions: pd.DataFrame, total_value: float):
    """Print formatted rebalancing instructions to the terminal."""
    print_header("MONTHLY REBALANCING INSTRUCTIONS")
    print(f"  Total Portfolio Value: ${total_value:,.2f}\n")

    needs_action = instructions[instructions["Action"] != "HOLD"]
    hold_tickers = instructions[instructions["Action"] == "HOLD"]["Ticker"].tolist()

    if needs_action.empty:
        print("  No rebalancing needed -- all allocations within threshold.\n")
    else:
        for _, row in needs_action.iterrows():
            tag = "SELL" if row["Action"] == "SELL" else "BUY "
            print(f"  {tag}  {row['Ticker']:5s}  ${row['$ Amount']:>8,.2f}"
                  f"   (current {row['Current Weight']:.1f}%  ->"
                  f"  target {row['Target Weight']:.1f}%)")

    if hold_tickers:
        print(f"\n  HOLD: {', '.join(hold_tickers)}")

    print(f"\n  Full breakdown:")
    print(instructions[["Ticker", "Current Weight", "Target Weight",
                         "Drift (%)", "Action", "$ Amount"]].to_string(index=False))


def print_stats(stats_list: list[StrategyStats]):
    """Print formatted performance statistics to the terminal."""
    print_header("PERFORMANCE STATISTICS")
    for s in stats_list:
        print(f"\n  --- {s.name} ---")
        print(f"  {'Period (years)':<25} {s.period_years}")
        print(f"  {'CAGR (%)':<25} {s.cagr}")
        print(f"  {'Max Drawdown (%)':<25} {s.max_drawdown}")
        print(f"  {'Sharpe Ratio':<25} {s.sharpe}")
        print(f"  {'Calmar Ratio':<25} {s.calmar}")
        print(f"  {'Final Value ($)':<25} {s.final_value}")
