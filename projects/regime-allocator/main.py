"""
Regime-Adaptive Multi-Asset Allocator
======================================
Uses a Hidden Markov Model trained on daily cross-asset features to detect
market regimes (risk-on vs risk-off), then allocates dynamically via
regime-conditional mean-variance optimization with soft probability blending.

Walk-forward: at each monthly rebalance, the HMM is retrained on an expanding
window of daily data (500+ observations minimum). No future information leaks.
"""
import argparse
import warnings

import numpy as np
import pandas as pd

from regime_allocator.config import StrategyConfig
from regime_allocator.data import download_prices, to_monthly
from regime_allocator.features import compute_hmm_features
from regime_allocator.backtest import (
    run_walkforward,
    benchmark_buyhold,
    benchmark_sixty_forty,
    benchmark_equal_weight,
    benchmark_pure_momentum,
)
from regime_allocator.analysis import (
    performance_metrics,
    comparison_table,
    plot_results,
    plot_drawdown,
    sensitivity_analysis,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    parser = argparse.ArgumentParser(description="Regime-Adaptive Multi-Asset Allocator")
    parser.add_argument("--start", default="2007-01-01", help="Backtest start date")
    parser.add_argument("--end", default=None, help="Backtest end date (default: today)")
    parser.add_argument("--regimes", type=int, default=2, help="Number of HMM regimes")
    parser.add_argument("--cost-bps", type=float, default=10, help="Round-trip transaction cost in bps")
    parser.add_argument("--risk-aversion", type=float, default=1.0, help="Risk aversion parameter")
    parser.add_argument("--no-plots", action="store_true", help="Skip plotting")
    parser.add_argument("--sensitivity", action="store_true", help="Run sensitivity analysis")
    parser.add_argument("--save-dir", default=None, help="Directory to save plots")
    args = parser.parse_args()

    config = StrategyConfig(
        start_date=args.start,
        n_regimes=args.regimes,
        cost_bps=args.cost_bps,
        risk_aversion=args.risk_aversion,
    )

    print("=" * 70)
    print("REGIME-ADAPTIVE MULTI-ASSET ALLOCATOR")
    print("=" * 70)
    print(f"Universe:       {', '.join(config.tickers)}")
    print(f"Period:         {config.start_date} → {args.end or 'today'}")
    print(f"Regimes:        {config.n_regimes}")
    print(f"Risk aversion:  {config.risk_aversion}")
    print(f"Cost:           {config.cost_bps} bps round-trip")
    print(f"DD cutback at:  {config.drawdown_cutback:.0%}")
    print(f"Turnover buf:   {config.turnover_buffer:.0%}")
    print()

    # --- Data ---
    print("Downloading price data...")
    prices = download_prices(config.tickers, config.start_date, args.end)
    print(f"  {len(prices)} daily observations, {prices.columns.tolist()}")
    print(f"  {prices.index[0].date()} → {prices.index[-1].date()}")

    # --- Features (daily) ---
    print("Computing daily HMM features...")
    daily_features = compute_hmm_features(prices, config)
    print(f"  {len(daily_features)} daily feature observations")

    # --- Walk-forward backtest (HMM on daily, allocation monthly) ---
    print(f"Running walk-forward backtest (min {config.min_train_days} daily obs)...")
    results = run_walkforward(prices, daily_features, config)
    print(f"  {len(results)} trading months")

    # --- Benchmarks ---
    monthly_prices = to_monthly(prices)
    spy_vals, spy_rets = benchmark_buyhold(monthly_prices, "SPY")
    sf_vals, sf_rets = benchmark_sixty_forty(monthly_prices)
    ew_vals, ew_rets = benchmark_equal_weight(monthly_prices)
    mom_vals, mom_rets = benchmark_pure_momentum(monthly_prices, cost_bps=config.cost_bps)

    common_idx = results.index
    benchmarks = {
        "SPY Buy & Hold": (spy_vals, spy_rets),
        "60/40 SPY/TLT": (sf_vals, sf_rets),
        "Equal Weight": (ew_vals, ew_rets),
        "Pure Momentum": (mom_vals, mom_rets),
    }

    # --- Performance ---
    strat_metrics = performance_metrics(results["return"], "Regime Allocator")
    spy_metrics = performance_metrics(
        spy_rets.reindex(common_idx).dropna(), "SPY Buy & Hold"
    )
    sf_metrics = performance_metrics(
        sf_rets.reindex(common_idx).dropna(), "60/40 SPY/TLT"
    )
    ew_metrics = performance_metrics(
        ew_rets.reindex(common_idx).dropna(), "Equal Weight"
    )
    mom_metrics = performance_metrics(
        mom_rets.reindex(common_idx).dropna(), "Pure Momentum"
    )

    print()
    print("=" * 70)
    print("PERFORMANCE COMPARISON")
    print("=" * 70)
    table = comparison_table(
        [strat_metrics, spy_metrics, sf_metrics, ew_metrics, mom_metrics]
    )
    print(table.to_string())
    print()

    avg_turnover = results["turnover"].mean()
    avg_cost = results["cost"].mean()
    dd_months = results["dd_triggered"].sum()
    print(f"Avg monthly turnover:  {avg_turnover:.1%}")
    print(f"Avg monthly cost:      {avg_cost:.4%}")
    print(f"Months DD guard fired: {dd_months}")

    # --- Plots ---
    if not args.no_plots:
        save_path = f"{args.save_dir}/backtest.png" if args.save_dir else None
        plot_results(results, benchmarks, config, save_path=save_path)
        dd_path = f"{args.save_dir}/drawdown.png" if args.save_dir else None
        plot_drawdown(results, save_path=dd_path)

    # --- Sensitivity ---
    if args.sensitivity:
        print()
        print("=" * 70)
        print("SENSITIVITY ANALYSIS — Number of Regimes")
        print("=" * 70)

        def regime_factory(n):
            return StrategyConfig(
                start_date=config.start_date,
                n_regimes=n,
                cost_bps=config.cost_bps,
                risk_aversion=config.risk_aversion,
            )

        sens = sensitivity_analysis(
            prices, daily_features, regime_factory, "n_regimes", [2, 3, 4]
        )
        for _, row in sens.iterrows():
            print(
                f"  {row['name']:20s}  CAGR={row['cagr']:6.1%}  "
                f"Sharpe={row['sharpe']:.2f}  MaxDD={row['max_drawdown']:7.1%}"
            )

    return results


if __name__ == "__main__":
    main()
