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

import argparse

import pandas as pd

from engine import config
from engine.config import RuntimeConfig, validate_config

from engine.data      import fetch_prices, get_price_provenance
from engine.backtest  import run_backtest, compute_stats
from engine.optimiser import optimise_allocation
from research._shared.export  import (make_results_dir, export_results,
                               append_to_master_log, print_header, print_stats,
                               start_run_log, stop_run_log)


def parse_args() -> argparse.Namespace:
    """Parse explicit runtime overrides for reproducible command-line runs."""
    defaults = config.current_runtime_config()
    parser = argparse.ArgumentParser(description="Run the All Weather backtest engine.")
    parser.add_argument("--run-mode", default=defaults.run_mode,
                        choices=["backtest", "optimise", "walk_forward", "pareto", "oos_evaluate", "full_backtest"])
    parser.add_argument("--strategy-id", default=defaults.strategy_id)
    parser.add_argument("--data-source", default=defaults.data_source, choices=["yfinance", "fmp"])
    parser.add_argument("--fmp-price-column", default=defaults.fmp_price_column,
                        choices=["open", "high", "low", "close", "adj_close"])
    parser.add_argument("--pricing-model", default=defaults.pricing_model,
                        choices=["total_return", "price_return"])
    parser.add_argument("--backtest-start", default=defaults.backtest_start)
    parser.add_argument("--backtest-end", default=defaults.backtest_end)
    parser.add_argument("--oos-start", default=defaults.oos_start)
    parser.add_argument("--transaction-cost-pct", type=float, default=defaults.transaction_cost_pct)
    parser.add_argument("--tax-drag-pct", type=float, default=defaults.tax_drag_pct)
    parser.add_argument("--run-tag", default=defaults.run_tag)
    return parser.parse_args()


def _runtime_from_args(args: argparse.Namespace) -> RuntimeConfig:
    return RuntimeConfig(
        run_mode=args.run_mode,
        strategy_id=args.strategy_id,
        data_source=args.data_source,
        fmp_price_column=args.fmp_price_column,
        pricing_model=args.pricing_model,
        backtest_start=args.backtest_start,
        backtest_end=args.backtest_end,
        oos_start=args.oos_start,
        transaction_cost_pct=args.transaction_cost_pct,
        tax_drag_pct=args.tax_drag_pct,
        run_tag=args.run_tag,
    )


def main():

    config.apply_runtime_config(_runtime_from_args(parse_args()))

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
            from research._shared.validation import run_pareto_frontier

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
            from research._shared.validation import run_walk_forward

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
            instructions = pd.DataFrame(columns=[
                "Ticker",
                "Current Weight",
                "Target Weight",
                "Drift (%)",
                "Action",
                "$ Amount",
                "Current Price",
                "Current Shares",
            ])

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
                           run_label, price_provenance=get_price_provenance(prices))
            append_to_master_log(results_dir, stats_list, allocation, run_label)

            # ---- Plot ----
            from engine.plotting import plot_backtest

            plot_backtest(backtest, stats_list, results_dir, run_label,
                          allocation=allocation)
    finally:
        # Stop logging to file
        stop_run_log(tee)

if __name__ == "__main__":
    main()
