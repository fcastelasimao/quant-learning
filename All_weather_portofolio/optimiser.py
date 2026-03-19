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
# WEIGHT PROJECTION
# ===========================================================================

def _project_weights(weights: np.ndarray,
                     tickers: list[str],
                     min_weight: float,
                     max_weight: float,
                     asset_class_groups: dict | None = None,
                     asset_class_max_weight: dict | None = None,
                     max_iter: int = 200) -> np.ndarray:
    """
    Project weights onto the simplex with per-asset and per-group box constraints.

    Iteratively clips individual weights, clips asset class groups, and
    renormalises until all constraints are satisfied simultaneously.

    Parameters
    ----------
    weights                : raw weight array to project
    tickers                : list of ticker names matching weights order
    min_weight             : minimum weight per individual asset
    max_weight             : maximum weight per individual asset
    asset_class_groups     : dict of {group_name: [ticker, ...]}
    asset_class_max_weight : dict of {group_name: max_fraction}
    max_iter               : maximum projection iterations
    """
    w = weights.copy()
    for _ in range(max_iter):
        # Step 1: clip individual asset bounds
        w = np.clip(w, min_weight, max_weight)
        w = w / w.sum()

        # Step 2: clip asset class group bounds
        if asset_class_groups and asset_class_max_weight:
            for group, group_tickers in asset_class_groups.items():
                cap     = asset_class_max_weight.get(group, 1.0)
                indices = [tickers.index(t) for t in group_tickers if t in tickers]
                if not indices:
                    continue
                group_total = w[indices].sum()
                if group_total > cap + 1e-8:
                    # Scale down group members proportionally
                    scale      = cap / group_total
                    w[indices] *= scale

        w = w / w.sum()

        # Check convergence
        converged = True
        if w.max() > max_weight + 1e-8:
            converged = False
        if w.min() < min_weight - 1e-8:
            converged = False
        if asset_class_groups and asset_class_max_weight:
            for group, group_tickers in asset_class_groups.items():
                cap     = asset_class_max_weight.get(group, 1.0)
                indices = [tickers.index(t) for t in group_tickers if t in tickers]
                if indices and w[indices].sum() > cap + 1e-8:
                    converged = False
        if converged:
            break

    return w


# ===========================================================================
# SHARED SCORING FUNCTION
# ===========================================================================

def _score_allocation(weights_array: np.ndarray,
                      tickers: list[str],
                      prices: pd.DataFrame,
                      benchmark_prices: pd.Series,
                      method: str,
                      min_cagr: float,
                      asset_class_groups: dict | None = None,
                      asset_class_max_weight: dict | None = None) -> float:
    """
    Score a weight array for any optimiser. Returns a value where LOWER is better
    (all optimisers minimise by convention -- we negate objectives we want to maximise).

    Scores:
      "sharpe_slsqp"                    -> -Sharpe
      "random", "calmar", "differential" -> -Calmar

    A large penalty (1e6) is returned if the CAGR constraint is violated
    or if any asset class group exceeds its maximum weight, acting as soft
    constraints compatible with all methods including DE.
    """
    # Check asset class group constraints before running backtest
    if asset_class_groups and asset_class_max_weight:
        for group, group_tickers in asset_class_groups.items():
            cap         = asset_class_max_weight.get(group, 1.0)
            indices     = [tickers.index(t) for t in group_tickers if t in tickers]
            group_total = sum(weights_array[i] for i in indices)
            if group_total > cap + 1e-6:
                return 1e6  # penalty: violated asset class cap

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
                    random_seed: int,
                    asset_class_groups: dict | None = None,
                    asset_class_max_weight: dict | None = None
                    ) -> tuple[np.ndarray | None, float]:
    """
    Random search: try n_trials random weight combinations and keep the best.

    Each trial draws weights uniformly between min_weight and max_weight,
    normalises them to sum to 1, and scores them with _score_allocation.

    Returns (best_weights_array, best_score). best_weights is None if no
    valid allocation was found (all trials violated the constraints).
    """
    np.random.seed(random_seed)

    tickers      = list(allocation.keys())
    n            = len(tickers)
    best_score   = np.inf
    best_weights = None

    for i in range(n_trials):
        raw     = np.random.uniform(min_weight, max_weight, n)
        weights = raw / raw.sum()

        # Skip if normalisation pushed any weight above the per-asset max
        if weights.max() > max_weight:
            continue

        score = _score_allocation(weights, tickers, prices,
                                  benchmark_prices, method, min_cagr,
                                  asset_class_groups, asset_class_max_weight)

        if score < best_score:
            best_score   = score
            best_weights = weights

    display = -best_score if best_score < 1e5 else float("nan")
    print(f"  Optimiser complete | Best Calmar: {display:.3f}")

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

    Asset class group constraints are read directly from config and applied
    automatically to all methods.

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
    import config as _config

    tickers = list(allocation.keys())
    n       = len(tickers)
    bounds  = [(min_weight, max_weight)] * n

    # Read asset class constraints from config
    asset_class_groups     = getattr(_config, "ASSET_CLASS_GROUPS",     None)
    asset_class_max_weight = getattr(_config, "ASSET_CLASS_MAX_WEIGHT", None)

    print(f"\nOptimising ({method}) | {' '.join(tickers)} | bounds [{min_weight:.0%}, {max_weight:.0%}]")
    if asset_class_groups and asset_class_max_weight:
        caps = ", ".join(f"{g}: {v:.0%}" for g, v in asset_class_max_weight.items())
        print(f"  Asset class caps: {caps}")
    print()

    # ------------------------------------------------------------------
    if method in ("random", "calmar"):
        best_weights, best_score = optimise_random(
            prices, benchmark_prices, allocation,
            min_weight, max_weight, min_cagr, n_trials, method, random_seed,
            asset_class_groups, asset_class_max_weight
        )

        if best_weights is None:
            print("WARNING: No valid allocation found. Try lowering OPT_MIN_CAGR.")
            return allocation

    # ------------------------------------------------------------------
    elif method == "differential_evolution":
        def de_objective(w):
            w_norm = w / w.sum()
            return _score_allocation(w_norm, tickers, prices,
                                     benchmark_prices, "calmar", min_cagr,
                                     asset_class_groups, asset_class_max_weight)

        result = differential_evolution(
            de_objective,
            bounds   = bounds,
            maxiter  = 200,
            popsize  = 10,
            tol      = 1e-6,
            seed     = random_seed,
            disp     = False,
        )

        if not result.success and result.fun >= 1e5:
            print(f"WARNING: DE did not converge cleanly: {result.message}")

        best_weights = _project_weights(result.x, tickers, min_weight, max_weight,
                                        asset_class_groups, asset_class_max_weight)
        best_score   = de_objective(best_weights)

    # ------------------------------------------------------------------
    elif method == "sharpe_slsqp":
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        w0          = np.array(list(allocation.values()))

        result = minimize(
            lambda w: _score_allocation(w, tickers, prices,
                                        benchmark_prices, "sharpe_slsqp", min_cagr,
                                        asset_class_groups, asset_class_max_weight),
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

    # Print asset class totals if caps are active
    if asset_class_groups and asset_class_max_weight:
        print("\nAsset class totals:")
        for group, group_tickers in asset_class_groups.items():
            total = sum(optimised.get(t, 0.0) for t in group_tickers)
            cap   = asset_class_max_weight.get(group, 1.0)
            print(f"  {group:<15s} {total:.1%}  (cap: {cap:.0%})")

    metric_label = "Sharpe" if method == "sharpe_slsqp" else "Calmar"
    print(f"\nBest {metric_label}: {-best_score:.3f}")

    return optimised