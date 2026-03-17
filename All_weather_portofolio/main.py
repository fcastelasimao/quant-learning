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
                       rebalancing_instructions, apply_rebalance)
from backtest  import run_backtest, compute_stats
from optimiser import optimise_allocation
from validation import run_walk_forward, run_pareto_frontier
from plotting  import plot_backtest
from export    import (make_results_dir, export_results,
                       append_to_master_log, print_header,
                       print_rebalancing, print_stats)


def main():

    # ---- Validate all parameters before doing any work ----
    validate_config()

    # ---- Create timestamped results folder ----
    results_dir = make_results_dir(config.RUN_LABEL)
    print(f"Results will be saved to: {results_dir}\n")

    # ---- Fetch price data ----
    # Deduplicate in case benchmark ticker is already in target_allocation
    all_tickers = list(dict.fromkeys(
        list(config.TARGET_ALLOCATION.keys()) + [config.BENCHMARK_TICKER]
    ))
    prices = fetch_prices(all_tickers, config.BACKTEST_START, config.BACKTEST_END)

    port_prices  = prices[list(config.TARGET_ALLOCATION.keys())]
    bench_prices = prices[config.BENCHMARK_TICKER]

    # ---- Optimiser (optional) ----
    # Runs before backtest so the optimised weights are used throughout
    allocation = dict(config.TARGET_ALLOCATION)  # work on a copy, not the global

    if config.RUN_OPTIMISER:
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
        print(f"\nTarget allocation updated to optimised weights.")

    # ---- Pareto frontier (optional) ----
    if config.RUN_PARETO:
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
    if config.RUN_WALK_FORWARD:
        run_walk_forward(
            prices           = port_prices,
            benchmark_prices = bench_prices,
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

    needs_rebalance = (instructions["Action"] != "HOLD").any()
    if needs_rebalance:
        answer = input("\n  Apply rebalancing now and save? (y/n): ").strip().lower()
        if answer == "y":
            holdings = apply_rebalance(holdings, latest_prices,
                                       allocation, total_value)
            save_holdings(holdings)
            print("  Portfolio rebalanced and saved.")

    # ---- Backtest ----
    print_header(f"RUNNING BACKTEST ({config.BACKTEST_START} to {config.BACKTEST_END})")
    backtest   = run_backtest(port_prices, bench_prices, allocation)
    stats_list = compute_stats(backtest)
    print_stats(stats_list)

    # ---- Export ----
    print_header(f"SAVING RESULTS TO {results_dir}")
    export_results(backtest, instructions, stats_list, allocation, results_dir)
    append_to_master_log(results_dir, stats_list, allocation, config.RUN_LABEL)

    # ---- Plot ----
    plot_backtest(backtest, stats_list, results_dir, config.RUN_LABEL)


if __name__ == "__main__":
    main()
