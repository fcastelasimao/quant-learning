"""
optimiser.py
============
Portfolio weight optimisation. Four methods, one shared scoring function.

All methods share _score_allocation() so the objective is computed
consistently regardless of which method is chosen. The key difference
between methods is HOW they search the weight space:

  random / calmar        -- blindly sample random weight combinations
  differential_evolution -- intelligently evolve a population of candidates
  sharpe_slsqp           -- follow the gradient of the Sharpe ratio

All scipy optimisers minimise by convention, so objectives that should be
maximised (Calmar, Sharpe) are negated before being passed to scipy and
negated again when displaying results.

Dependency: backtest.py (for run_backtest and stat helpers)
            config.py   (for OPT_RANDOM_SEED, passed in as argument)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution

from backtest import run_backtest, compute_cagr, compute_max_drawdown, \
                     compute_sharpe, compute_calmar


# ===========================================================================
# SHARED SCORING FUNCTION
# ===========================================================================

def _score_allocation(weights_array: np.ndarray,
                      tickers: list[str],
                      prices: pd.DataFrame,
                      benchmark_prices: pd.Series,
                      method: str,
                      min_cagr: float) -> float:
    """
    Score a weight array for any optimiser. Returns a value where LOWER is better
    (all optimisers minimise by convention -- we negate objectives we want to maximise).

    Scores:
      "sharpe_slsqp"                    -> -Sharpe
      "random", "calmar", "differential" -> -Calmar

    A large penalty (1e6) is returned if the CAGR constraint is violated,
    acting as a soft constraint compatible with all methods including DE
    (which does not support hard constraints natively).
    """
    allocation = dict(zip(tickers, weights_array))
    bt         = run_backtest(prices, benchmark_prices, allocation)
    series     = bt["All Weather Value"]
    years      = (bt.index[-1] - bt.index[0]).days / 365.25

    cagr = compute_cagr(series, years)
    mdd  = compute_max_drawdown(series)

    if cagr < min_cagr:
        return 1e6      # penalty: violated minimum CAGR constraint

    if method == "sharpe_slsqp":
        ret_col = "All Weather Value Monthly Ret (%)"
        return -compute_sharpe(bt[ret_col])

    return -compute_calmar(cagr, mdd)


# ===========================================================================
# RANDOM SEARCH (shared by "random" and "calmar" methods)
# ===========================================================================

def optimise_random(prices: pd.DataFrame,
                    benchmark_prices: pd.Series,
                    allocation: dict,
                    min_weight: float,
                    max_weight: float,
                    min_cagr: float,
                    n_trials: int,
                    method: str,
                    random_seed: int) -> tuple[np.ndarray | None, float]:
    """
    Random search: try n_trials random weight combinations and keep the best.

    Each trial draws weights uniformly between min_weight and max_weight,
    normalises them to sum to 1, and scores them with _score_allocation.

    Returns (best_weights_array, best_score). best_weights is None if no
    valid allocation was found (all trials violated the CAGR constraint).
    """
    np.random.seed(random_seed)

    tickers      = list(allocation.keys())
    n            = len(tickers)
    best_score   = np.inf
    best_weights = None

    for i in range(n_trials):
        raw     = np.random.uniform(min_weight, max_weight, n)
        weights = raw / raw.sum()

        # Skip if normalisation pushed any weight above the max
        if weights.max() > max_weight:
            continue

        score = _score_allocation(weights, tickers, prices,
                                  benchmark_prices, method, min_cagr)

        if score < best_score:
            best_score   = score
            best_weights = weights

        if (i + 1) % 500 == 0:
            display = -best_score if best_score < 1e5 else float("nan")
            print(f"  Trial {i+1:>5}/{n_trials} | Best Calmar so far: {display:.3f}")

    return best_weights, best_score


# ===========================================================================
# MAIN OPTIMISER DISPATCHER
# ===========================================================================

def optimise_allocation(prices: pd.DataFrame,
                        benchmark_prices: pd.Series,
                        allocation: dict,
                        method: str,
                        min_weight: float,
                        max_weight: float,
                        min_cagr: float,
                        n_trials: int,
                        random_seed: int) -> dict:
    """
    Dispatch to the appropriate optimisation method and return optimised weights
    as a dict {ticker: weight}.

    Parameters
    ----------
    prices, benchmark_prices : price DataFrames from data.fetch_prices()
    allocation               : starting allocation (used as initial guess for SLSQP)
    method                   : "random" | "calmar" | "differential_evolution" | "sharpe_slsqp"
    min_weight, max_weight   : per-asset weight bounds
    min_cagr                 : minimum acceptable CAGR (percent)
    n_trials                 : number of trials for random/calmar methods
    random_seed              : numpy random seed for reproducibility
    """
    tickers = list(allocation.keys())
    n       = len(tickers)
    bounds  = [(min_weight, max_weight)] * n

    print(f"\nRunning optimiser -- method: {method}")
    print(f"  Assets:        {', '.join(tickers)}")
    print(f"  Weight bounds: [{min_weight:.0%}, {max_weight:.0%}] per asset")
    print(f"  Min CAGR:      {min_cagr}%")
    print(f"  Random seed:   {random_seed}\n")

    # ------------------------------------------------------------------
    if method in ("random", "calmar"):
        print(f"  Objective:  maximise Calmar ratio (CAGR / |max drawdown|)")
        print(f"  Trials:     {n_trials}\n")

        best_weights, best_score = optimise_random(
            prices, benchmark_prices, allocation,
            min_weight, max_weight, min_cagr, n_trials, method, random_seed
        )

        if best_weights is None:
            print("WARNING: No valid allocation found. Try lowering OPT_MIN_CAGR.")
            return allocation

    # ------------------------------------------------------------------
    elif method == "differential_evolution":
        # DE maintains a population of candidate solutions and evolves them
        # over generations by combining the best candidates. It does not need
        # gradients and explores the search space much more efficiently than
        # random search. Each candidate's weights are normalised before scoring
        # so the sum-to-1 constraint is always satisfied.
        print(f"  Objective:  maximise Calmar ratio (CAGR / |max drawdown|)")
        print(f"  Algorithm:  Differential Evolution (scipy)\n")

        def de_objective(w):
            w_norm = w / w.sum()    # normalise to sum to 1
            return _score_allocation(w_norm, tickers, prices,
                                     benchmark_prices, "calmar", min_cagr)

        result = differential_evolution(
            de_objective,
            bounds   = bounds,
            maxiter  = 200,
            popsize  = 10,
            tol      = 1e-6,
            seed     = random_seed,
            disp     = True,
        )

        if not result.success and result.fun >= 1e5:
            print(f"WARNING: DE did not converge cleanly: {result.message}")

        raw_weights  = result.x
        best_weights = raw_weights / raw_weights.sum()
        best_weights = np.clip(best_weights, min_weight, max_weight)
        best_weights = best_weights / best_weights.sum()
        best_score = de_objective(best_weights)

    # ------------------------------------------------------------------
    elif method == "sharpe_slsqp":
        # SLSQP follows the gradient of the objective downhill. This works
        # for Sharpe because Sharpe (mean/std of returns) is smooth and
        # differentiable -- small weight changes produce non-zero gradients.
        # It does NOT work for max drawdown (single worst moment, discontinuous).
        print(f"  Objective:  maximise Sharpe ratio (return / volatility)")
        print(f"  Algorithm:  SLSQP gradient-based (scipy)\n")

        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        w0          = np.array(list(allocation.values()))

        result = minimize(
            lambda w: _score_allocation(w, tickers, prices,
                                        benchmark_prices, "sharpe_slsqp", min_cagr),
            w0,
            method      = "SLSQP",
            bounds      = bounds,
            constraints = constraints,
            options     = {"maxiter": 1000, "ftol": 1e-9},
        )

        if not result.success:
            print(f"WARNING: SLSQP did not converge: {result.message}")
            print("Returning original allocation unchanged.")
            return allocation

        best_weights = result.x
        best_score   = result.fun

    else:
        raise ValueError(
            f"Unknown method: '{method}'. "
            f"Choose from: random, calmar, differential_evolution, sharpe_slsqp"
        )

    # ------------------------------------------------------------------
    # Print results summary
    optimised = dict(zip(tickers, best_weights))
    print("\nOptimised weights:")
    for t, w in optimised.items():
        original = allocation[t]
        change   = w - original
        sign     = "+" if change >= 0 else ""
        print(f"  {t:5s}  {w:.1%}   (was {original:.1%}, {sign}{change:.1%})")

    metric_label = "Sharpe" if method == "sharpe_slsqp" else "Calmar"
    print(f"\nBest {metric_label}: {-best_score:.3f}")

    return optimised
