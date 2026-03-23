"""
main.py
=======
Entry point. Orchestrates the full run -- no logic lives here.

Reads configuration from config.py, calls functions from other modules
in the correct order, and passes data between them. If you want to
understand what the program does at a high level, read this file.
If you want to understand HOW something works, read the relevant module.

Run with:
    python main.py
"""

import config
from config import validate_config

from data      import fetch_prices
from portfolio import (load_holdings, save_holdings, initialise_holdings,
                       rebalancing_instructions)
from backtest  import run_backtest, compute_stats
from optimiser import optimise_allocation
from validation import run_walk_forward, run_pareto_frontier
from plotting  import plot_backtest
from export import (make_results_dir, export_results,
                    append_to_master_log, print_header,
                    print_rebalancing, print_stats,
                    start_run_log, stop_run_log)


def main():

    # ---- Validate all parameters before doing any work ----
    validate_config()

    # ---- Determine IS/OOS date window for this run mode ----
    if config.RUN_MODE in ("backtest", "optimise", "walk_forward", "pareto"):
        price_start = config.BACKTEST_START
        price_end   = config.OOS_START
    elif config.RUN_MODE == "oos_evaluate":
        price_start = config.OOS_START
        price_end   = config.BACKTEST_END
    else:  # full_backtest
        price_start = config.BACKTEST_START
        price_end   = config.BACKTEST_END

    # ---- Build run label and create timestamped results folder ----
    run_label   = config._build_run_label(price_start, price_end)
    results_dir = make_results_dir(run_label)
    print(f"Results will be saved to: {results_dir}\n")

    # Start logging to file
    tee = start_run_log(results_dir)

    try:
        # ---- Fetch price data (always full range; sliced to mode window below) ----
        # Deduplicate in case benchmark ticker is already in target_allocation
        all_tickers = list(dict.fromkeys(
            list(config.TARGET_ALLOCATION.keys()) + [config.BENCHMARK_TICKER, "TLT"]
        ))
        prices = fetch_prices(all_tickers, config.BACKTEST_START, config.BACKTEST_END)

        # Slice all series to the correct IS/OOS window for this mode
        port_prices  = prices[list(config.TARGET_ALLOCATION.keys())]
        bench_prices = prices[config.BENCHMARK_TICKER]
        port_prices  = port_prices[(port_prices.index  >= price_start) &
                                    (port_prices.index  <  price_end)]
        bench_prices = bench_prices[(bench_prices.index >= price_start) &
                                     (bench_prices.index <  price_end)]
        tlt_prices   = prices["TLT"][(prices["TLT"].index >= price_start) &
                                      (prices["TLT"].index <  price_end)]

        # ---- Optimiser (optional) ----
        # Runs before backtest so the optimised weights are used throughout
        allocation = dict(config.TARGET_ALLOCATION)  # work on a copy, not the global

        if config.RUN_MODE == "optimise":
            optimised = optimise_allocation(
                prices           = port_prices,
                benchmark_prices = bench_prices,
                allocation       = allocation,
                method           = config.OPT_METHOD,
                min_weight       = config.OPT_MIN_WEIGHT,
                max_weight       = config.OPT_MAX_WEIGHT,
                min_cagr         = config.OPT_MIN_CAGR,
                n_trials         = config.OPT_N_TRIALS,
                random_seed      = config.OPT_RANDOM_SEED,
            )
            allocation.update(optimised)

        # ---- Pareto frontier (optional) ----
        if config.RUN_MODE == "pareto":
            run_pareto_frontier(
                prices           = port_prices,
                benchmark_prices = bench_prices,
                allocation       = allocation,
                cagr_targets     = config.PARETO_CAGR_RANGE,
                min_weight       = config.OPT_MIN_WEIGHT,
                max_weight       = config.OPT_MAX_WEIGHT,
                n_trials         = config.OPT_N_TRIALS,
                random_seed      = config.OPT_RANDOM_SEED,
                results_dir      = results_dir,
            )

        # ---- Walk-forward validation (optional) ----
        if config.RUN_MODE == "walk_forward":
            run_walk_forward(
                prices           = port_prices,
                benchmark_prices = bench_prices,
                tlt_prices       = tlt_prices,
                allocation       = allocation,
                train_years      = config.WF_TRAIN_YEARS,
                test_years       = config.WF_TEST_YEARS,
                step_years       = config.WF_STEP_YEARS,
                min_weight       = config.OPT_MIN_WEIGHT,
                max_weight       = config.OPT_MAX_WEIGHT,
                n_trials         = config.OPT_N_TRIALS,
                random_seed      = config.OPT_RANDOM_SEED,
                results_dir      = results_dir,
            )

        BACKTEST_MODES = {"backtest", "optimise", "oos_evaluate", "full_backtest"}

        if config.RUN_MODE in BACKTEST_MODES:

            # ---- Current holdings & rebalancing ----
            latest_prices = port_prices.iloc[-1]

            holdings = load_holdings()
            if holdings is None:
                print("No existing holdings found. Initialising with target allocation...\n")
                holdings = initialise_holdings(latest_prices, allocation,
                                               config.INITIAL_PORTFOLIO_VALUE)
                save_holdings(holdings)
            elif set(holdings.keys()) != set(allocation.keys()):
                print("Target allocation has changed -- resetting holdings...\n")
                print(f"  Old tickers: {sorted(holdings.keys())}")
                print(f"  New tickers: {sorted(allocation.keys())}\n")
                holdings = initialise_holdings(latest_prices, allocation,
                                               config.INITIAL_PORTFOLIO_VALUE)
                save_holdings(holdings)

            instructions, total_value = rebalancing_instructions(
                holdings, latest_prices, allocation, config.REBALANCE_THRESHOLD
            )
            print_rebalancing(instructions, total_value)

            # ---- Backtest ----
            print_header(
                f"RUNNING BACKTEST ({price_start} to {price_end})"
                f"  [MODE: {config.RUN_MODE}]"
            )
            backtest   = run_backtest(port_prices, bench_prices, allocation,
                                      tlt_prices=tlt_prices,
                                      transaction_cost_pct=config.TRANSACTION_COST_PCT,
                                      tax_drag_pct=config.TAX_DRAG_PCT)
            stats_list = compute_stats(backtest, prices=port_prices, allocation=allocation)
            print_stats(stats_list)

            # ---- Export ----
            print_header(f"SAVING RESULTS TO {results_dir}")
            export_results(backtest, instructions, stats_list, allocation, results_dir,
                           run_label)
            append_to_master_log(results_dir, stats_list, allocation, run_label)

            # ---- Plot ----
            plot_backtest(backtest, stats_list, results_dir, run_label,
                          allocation=allocation)
    finally:
        # Stop logging to file
        stop_run_log(tee)

if __name__ == "__main__":
    main()
