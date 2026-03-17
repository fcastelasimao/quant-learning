"""
validation.py
=============
Analysis tools that sit above the optimiser and backtest engine.

Contains:
  - run_walk_forward    tests whether optimised weights are robust or overfitted
  - run_pareto_frontier maps the full CAGR vs drawdown tradeoff curve

Both functions call optimise_random() and run_backtest() internally but
do not modify any state -- they are read-only analysis tools that produce
CSVs and plots as output.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest import run_backtest, compute_cagr, compute_max_drawdown, compute_calmar
from optimiser import optimise_random

import config

# ===========================================================================
# WALK-FORWARD VALIDATION
# ===========================================================================

def run_walk_forward(prices: pd.DataFrame,
                     benchmark_prices: pd.Series,
                     allocation: dict,
                     train_years: int,
                     test_years: int,
                     step_years: int,
                     min_weight: float,
                     max_weight: float,
                     n_trials: int,
                     random_seed: int,
                     results_dir: str):
    """
    Walk-forward validation to detect whether optimised weights are robust
    or simply overfitted to the historical training period.

    How it works
    ------------
    1. Divide the full price history into sliding windows. Each window
       has a training period and a test period that does not overlap.
    2. For each window, optimise weights using ONLY the training data.
    3. Evaluate those weights on the unseen test period.
    4. Compare:
         a. In-sample train Calmar vs out-of-sample test Calmar (overfitting check)
         b. Optimised test Calmar vs original allocation test Calmar (value-add check)

    Key output -- the overfit ratio
    --------------------------------
    overfit_ratio = test_calmar / train_calmar

      >= 0.8  low overfitting, allocation is robust
      0.6-0.8 moderate -- treat with caution
      < 0.6   high overfitting -- do not use for live trading

    If the optimised allocation also beats the original on the test period,
    the optimisation is adding genuine value beyond the training period.
    """
    print(f"\nRunning walk-forward validation...")
    print(f"  Train window: {train_years} years")
    print(f"  Test window:  {test_years} years")
    print(f"  Step size:    {step_years} years\n")

    all_dates    = prices.index
    start_date   = all_dates[0]
    end_date     = all_dates[-1]
    train_delta  = pd.DateOffset(years=train_years)
    test_delta   = pd.DateOffset(years=test_years)
    step_delta   = pd.DateOffset(years=step_years)

    # Build all windows
    windows      = []
    window_start = start_date
    while True:
        train_end = window_start + train_delta
        test_end  = train_end   + test_delta
        if test_end > end_date:
            break
        windows.append({
            "train_start": window_start,
            "train_end":   train_end,
            "test_start":  train_end,
            "test_end":    test_end,
        })
        window_start = window_start + step_delta

    if not windows:
        print("Not enough data for walk-forward validation. "
              "Try reducing WF_TRAIN_YEARS or WF_TEST_YEARS.")
        return

    print(f"  Found {len(windows)} windows:\n")
    for i, w in enumerate(windows):
        print(f"  Window {i+1}: "
              f"train {w['train_start'].strftime('%Y-%m')} -> "
              f"{w['train_end'].strftime('%Y-%m')}  |  "
              f"test  {w['test_start'].strftime('%Y-%m')} -> "
              f"{w['test_end'].strftime('%Y-%m')}")
    print()

    records = []
    for i, w in enumerate(windows):
        print(f"  --- Window {i+1} "
              f"({w['train_start'].strftime('%Y-%m')} - "
              f"{w['test_end'].strftime('%Y-%m')}) ---")

        # Slice train and test data
        train_prices = prices[(prices.index >= w["train_start"]) &
                               (prices.index <  w["train_end"])]
        train_bench  = benchmark_prices[
                           (benchmark_prices.index >= w["train_start"]) &
                           (benchmark_prices.index <  w["train_end"])]
        test_prices  = prices[(prices.index >= w["test_start"]) &
                               (prices.index <  w["test_end"])]
        test_bench   = benchmark_prices[
                           (benchmark_prices.index >= w["test_start"]) &
                           (benchmark_prices.index <  w["test_end"])]

        if train_prices.empty or test_prices.empty:
            print(f"    Skipping -- insufficient data.")
            continue

        # Optimise on training data only
        print(f"    Optimising on training data ({train_years} years)...")
        if config.WF_OPT_METHOD == "differential_evolution":
            opt_weights, _ = optimise_allocation(
                train_prices, train_bench, allocation,
                method      = "differential_evolution",
                min_weight  = min_weight,
                max_weight  = max_weight,
                min_cagr    = 0.0,
                n_trials    = n_trials,
                random_seed = random_seed,
            )
            opt_weights = np.array(list(opt_weights.values()))
        else:
            opt_weights, _ = optimise_random(
                train_prices, train_bench, allocation,
                min_weight, max_weight, 0.0, n_trials,
                config.WF_OPT_METHOD, random_seed
            )
        if opt_weights is None:
            print(f"    Optimiser failed -- skipping window.")
            continue

        opt_allocation = dict(zip(list(allocation.keys()), opt_weights))

        # Evaluate optimised weights in-sample (train)
        bt_train     = run_backtest(train_prices, train_bench, opt_allocation)
        series_tr    = bt_train["All Weather Value"]
        years_tr     = (series_tr.index[-1] - series_tr.index[0]).days / 365.25
        train_cagr   = round(compute_cagr(series_tr, years_tr), 2)
        train_mdd    = round(compute_max_drawdown(series_tr), 2)
        train_calmar = round(compute_calmar(train_cagr, train_mdd), 3)

        # Evaluate optimised weights out-of-sample (test)
        bt_test      = run_backtest(test_prices, test_bench, opt_allocation)
        series_te    = bt_test["All Weather Value"]
        years_te     = (series_te.index[-1] - series_te.index[0]).days / 365.25
        test_cagr    = round(compute_cagr(series_te, years_te), 2)
        test_mdd     = round(compute_max_drawdown(series_te), 2)
        test_calmar  = round(compute_calmar(test_cagr, test_mdd), 3)

        # Evaluate original allocation out-of-sample (baseline comparison)
        bt_orig      = run_backtest(test_prices, test_bench, allocation)
        series_or    = bt_orig["All Weather Value"]
        orig_cagr    = round(compute_cagr(series_or, years_te), 2)
        orig_mdd     = round(compute_max_drawdown(series_or), 2)
        orig_calmar  = round(compute_calmar(orig_cagr, orig_mdd), 3)

        overfit_ratio = round(test_calmar / train_calmar, 3) \
                        if train_calmar != 0 else 0.0

        print(f"    In-sample  (train): CAGR={train_cagr:>6.2f}%  "
              f"MaxDD={train_mdd:>7.2f}%  Calmar={train_calmar:.3f}")
        print(f"    Out-sample (test):  CAGR={test_cagr:>6.2f}%  "
              f"MaxDD={test_mdd:>7.2f}%  Calmar={test_calmar:.3f}")
        print(f"    Original   (test):  CAGR={orig_cagr:>6.2f}%  "
              f"MaxDD={orig_mdd:>7.2f}%  Calmar={orig_calmar:.3f}")
        print(f"    Overfit ratio: {overfit_ratio:.3f}  "
              f"{'ok' if overfit_ratio >= 0.6 else 'WARNING: possible overfit'}\n")

        records.append({
            "Window":                    i + 1,
            "Train Start":               w["train_start"].strftime("%Y-%m"),
            "Train End":                 w["train_end"].strftime("%Y-%m"),
            "Test Start":                w["test_start"].strftime("%Y-%m"),
            "Test End":                  w["test_end"].strftime("%Y-%m"),
            "Opt Weights":               " | ".join(f"{t}={v:.1%}"
                                                    for t, v in opt_allocation.items()),
            "Train CAGR (%)":            train_cagr,
            "Train MaxDD (%)":           train_mdd,
            "Train Calmar":              train_calmar,
            "Test CAGR (%)":             test_cagr,
            "Test MaxDD (%)":            test_mdd,
            "Test Calmar":               test_calmar,
            "Original Test CAGR (%)":    orig_cagr,
            "Original Test MaxDD (%)":   orig_mdd,
            "Original Test Calmar":      orig_calmar,
            "Overfit Ratio":             overfit_ratio,
        })

    if not records:
        print("No valid windows completed.")
        return

    df             = pd.DataFrame(records)
    mean_overfit   = df["Overfit Ratio"].mean()
    mean_test_cal  = df["Test Calmar"].mean()
    mean_orig_cal  = df["Original Test Calmar"].mean()
    opt_beats_orig = (df["Test Calmar"] > df["Original Test Calmar"]).sum()

    print(f"\n  === WALK-FORWARD SUMMARY ===")
    print(f"  Windows completed:               {len(records)}")
    print(f"  Mean overfit ratio:              {mean_overfit:.3f}  "
          f"(1.0 = no overfit, <0.6 = concerning)")
    print(f"  Mean test Calmar (optimised):    {mean_test_cal:.3f}")
    print(f"  Mean test Calmar (original):     {mean_orig_cal:.3f}")
    print(f"  Windows where opt beat original: {opt_beats_orig}/{len(records)}")

    if mean_overfit >= 0.7 and opt_beats_orig > len(records) / 2:
        print(f"\n  VERDICT: Allocation appears robust.")
    elif mean_overfit >= 0.5:
        print(f"\n  VERDICT: Moderate overfitting. Treat results with caution.")
    else:
        print(f"\n  VERDICT: High overfitting. Do not use for live trading.")

    # Save CSV
    csv_path = os.path.join(results_dir, "walk_forward.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Walk-forward results saved -> {csv_path}")

    # Plot
    _plot_walk_forward(df, mean_overfit, train_years, test_years,
                       step_years, results_dir)


def _plot_walk_forward(df: pd.DataFrame,
                       mean_overfit: float,
                       train_years: int,
                       test_years: int,
                       step_years: int,
                       results_dir: str):
    """Internal: plot walk-forward results and save to results_dir."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor("#0d1117")
    ax1, ax2 = axes

    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#c9d1d9", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.label.set_color("#c9d1d9")
        ax.xaxis.label.set_color("#c9d1d9")
        ax.title.set_color("white")
        ax.grid(axis="y", color="#30363d", alpha=0.4)

    windows_x = (df["Window"].astype(str) + "\n" +
                 df["Train Start"] + " - " + df["Test End"])
    x     = np.arange(len(df))
    width = 0.28

    ax1.bar(x - width, df["Train Calmar"],         width, color="#58a6ff",
            alpha=0.85, label="Optimised (in-sample train)")
    ax1.bar(x,          df["Test Calmar"],          width, color="#3fb950",
            alpha=0.85, label="Optimised (out-of-sample test)")
    ax1.bar(x + width,  df["Original Test Calmar"], width, color="#f0b429",
            alpha=0.85, label="Original allocation (test)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(windows_x, fontsize=7)
    ax1.axhline(0, color="#8b949e", lw=0.8)
    ax1.set_ylabel("Calmar Ratio", fontsize=10)
    ax1.set_title("Calmar Ratio: Train vs Test vs Original\n"
                  "Green close to blue = low overfitting",
                  fontsize=11, pad=8)
    ax1.legend(fontsize=8, facecolor="#21262d", edgecolor="#30363d",
               labelcolor="white")

    colors = ["#3fb950" if r >= 0.6 else "#f85149" for r in df["Overfit Ratio"]]
    ax2.bar(x, df["Overfit Ratio"], color=colors, alpha=0.85)
    ax2.axhline(1.0, color="#8b949e", lw=1.0, linestyle="--",
                label="1.0 = perfect (no overfit)")
    ax2.axhline(0.6, color="#f0b429", lw=1.0, linestyle=":",
                label="0.6 = warning threshold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(windows_x, fontsize=7)
    ax2.set_ylabel("Overfit Ratio (test Calmar / train Calmar)", fontsize=10)
    ax2.set_title("Overfit Ratio per Window\n"
                  "Green = acceptable  |  Red = concerning",
                  fontsize=11, pad=8)
    ax2.legend(fontsize=8, facecolor="#21262d", edgecolor="#30363d",
               labelcolor="white")
    ax2.set_ylim(0, max(df["Overfit Ratio"].max() * 1.2, 1.2))

    plt.suptitle(
        f"Walk-Forward Validation  |  "
        f"Train={train_years}yr  Test={test_years}yr  Step={step_years}yr  |  "
        f"Mean overfit ratio: {mean_overfit:.3f}",
        fontsize=12, color="white", y=1.02
    )
    plt.tight_layout()

    path = os.path.join(results_dir, "walk_forward.png")
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  Walk-forward plot saved -> {path}")
    plt.show()


# ===========================================================================
# PARETO FRONTIER
# ===========================================================================

def run_pareto_frontier(prices: pd.DataFrame,
                        benchmark_prices: pd.Series,
                        allocation: dict,
                        cagr_targets: np.ndarray,
                        min_weight: float,
                        max_weight: float,
                        n_trials: int,
                        random_seed: int,
                        results_dir: str):
    """
    Sweep across minimum CAGR constraints and map the full risk-return tradeoff.

    For each CAGR target, finds the allocation that maximises Calmar while
    meeting that CAGR floor. Plotting CAGR vs max drawdown for all these
    points produces the efficient frontier -- the set of portfolios where
    you cannot reduce drawdown without also reducing return.

    The current allocation is marked on the chart so you can see where it
    sits relative to the frontier and whether there is room for improvement.

    Also saves pareto_frontier.csv with the weights for each frontier point,
    so you can pick the allocation that matches your risk tolerance.
    """
    print(f"\nRunning Pareto frontier analysis...")
    print(f"  CAGR targets:      {', '.join(f'{x:.1f}%' for x in cagr_targets)}")
    print(f"  Trials per target: {n_trials}\n")

    tickers       = list(allocation.keys())
    frontier_pts  = []

    for min_cagr in cagr_targets:
        print(f"  Optimising for CAGR >= {min_cagr:.1f}%...")
        best_weights, _ = optimise_random(
            prices, benchmark_prices, allocation,
            min_weight, max_weight, min_cagr, n_trials, "calmar", random_seed
        )
        if best_weights is None:
            print(f"    No valid allocation found -- skipping.")
            continue

        opt_alloc   = dict(zip(tickers, best_weights))
        bt          = run_backtest(prices, benchmark_prices, opt_alloc)
        series      = bt["All Weather Value"]
        years       = (bt.index[-1] - bt.index[0]).days / 365.25
        actual_cagr = compute_cagr(series, years)
        actual_mdd  = compute_max_drawdown(series)

        print(f"    -> CAGR: {actual_cagr:.2f}%  |  Max DD: {actual_mdd:.2f}%  "
              f"|  Calmar: {compute_calmar(actual_cagr, actual_mdd):.3f}")

        frontier_pts.append((actual_cagr, actual_mdd,
                             dict(zip(tickers, best_weights))))

    if not frontier_pts:
        print("No frontier points found. Try a wider CAGR range or more trials.")
        return

    # Current allocation reference point
    bt_cur       = run_backtest(prices, benchmark_prices, allocation)
    series_cur   = bt_cur["All Weather Value"]
    years_cur    = (bt_cur.index[-1] - bt_cur.index[0]).days / 365.25
    current_cagr = compute_cagr(series_cur, years_cur)
    current_mdd  = compute_max_drawdown(series_cur)

    frontier_pts.sort(key=lambda x: x[0])
    cagrs     = [p[0] for p in frontier_pts]
    drawdowns = [abs(p[1]) for p in frontier_pts]

    # Save CSV
    frontier_df = pd.DataFrame([
        {"Actual CAGR (%)":     round(p[0], 2),
         "Max Drawdown (%)":    round(p[1], 2),
         "Calmar":              round(compute_calmar(p[0], p[1]), 3),
         "Weights":             " | ".join(f"{t}={w:.1%}"
                                           for t, w in p[2].items())}
        for p in frontier_pts
    ])
    frontier_df.to_csv(os.path.join(results_dir, "pareto_frontier.csv"),
                       index=False)

    # Plot
    _plot_pareto(cagrs, drawdowns, frontier_pts, current_cagr,
                 current_mdd, results_dir)


def _plot_pareto(cagrs: list,
                 drawdowns: list,
                 frontier_pts: list,
                 current_cagr: float,
                 current_mdd: float,
                 results_dir: str):
    """Internal: plot the Pareto frontier and save to results_dir."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#c9d1d9")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.label.set_color("#c9d1d9")
    ax.xaxis.label.set_color("#c9d1d9")
    ax.title.set_color("white")
    ax.grid(alpha=0.3, color="#30363d")

    ax.plot(drawdowns, cagrs, "o-", color="#58a6ff", lw=2, markersize=7,
            label="Efficient frontier (Calmar-optimised)")

    for cagr, mdd_abs in zip(cagrs, drawdowns):
        calmar = compute_calmar(cagr, -mdd_abs)
        ax.annotate(f"Calmar={calmar:.2f}",
                    xy=(mdd_abs, cagr), xytext=(6, 0),
                    textcoords="offset points",
                    color="#8b949e", fontsize=7.5)

    ax.scatter([abs(current_mdd)], [current_cagr],
               color="#f0b429", s=120, zorder=5,
               label=f"Current allocation  "
                     f"(Calmar={compute_calmar(current_cagr, current_mdd):.2f})")

    ax.set_xlabel("Max Drawdown (%) -- lower is better", fontsize=11)
    ax.set_ylabel("CAGR (%) -- higher is better", fontsize=11)
    ax.set_title("Pareto Frontier: CAGR vs Max Drawdown\n"
                 "Each point is the best Calmar achievable at that drawdown tolerance",
                 fontsize=12, pad=10)
    ax.legend(fontsize=9, facecolor="#21262d", edgecolor="#30363d",
              labelcolor="white")

    path = os.path.join(results_dir, "pareto_frontier.png")
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\n  Pareto frontier saved -> {path}")
    plt.show()
