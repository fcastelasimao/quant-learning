"""
optimiser.py
============
Portfolio weight optimisation. Five methods, one shared scoring function.

All methods share _score_allocation() so the objective is computed
consistently regardless of which method is chosen. The key difference
between methods is HOW they search the weight space:

  random / calmar        -- blindly sample random weight combinations
  differential_evolution -- intelligently evolve a population of candidates
                            (uses Martin ratio objective by default)
  sharpe_slsqp           -- follow the gradient of the Sharpe ratio
  martin                 -- same as calmar but uses Martin ratio (CAGR / Ulcer)

DE-specific design decisions:
  - Per-asset bounds from ASSET_BOUNDS config (fallback: uniform OPT_MIN/MAX_WEIGHT)
  - Martin ratio objective (smoother than Calmar; not dominated by single worst day)
  - Weights are projected inside de_objective so DE sees the true landscape

All scipy optimisers minimise by convention, so objectives that should be
maximised (Calmar, Martin, Sharpe) are negated before being passed to scipy
and negated again when displaying results.

Dependency: backtest.py (for run_backtest and stat helpers)
            config.py   (for OPT_RANDOM_SEED, passed in as argument)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution

from backtest import run_backtest, compute_cagr, compute_max_drawdown, \
                     compute_sharpe, compute_calmar, compute_ulcer_index


# ===========================================================================
# WEIGHT PROJECTION
# ===========================================================================

def _project_weights(weights: np.ndarray,
                     tickers: list[str],
                     min_weight,
                     max_weight,
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
    min_weight             : minimum weight per asset -- scalar or 1-D array
    max_weight             : maximum weight per asset -- scalar or 1-D array
    asset_class_groups     : dict of {group_name: [ticker, ...]}
    asset_class_max_weight : dict of {group_name: max_fraction}
    max_iter               : maximum projection iterations

    Accepts scalars or per-asset arrays for min_weight / max_weight so it
    works correctly with both uniform and ASSET_BOUNDS-derived bounds.
    """
    n  = len(weights)
    lo = (np.full(n, min_weight, dtype=float) if np.isscalar(min_weight)
          else np.asarray(min_weight, dtype=float))
    hi = (np.full(n, max_weight, dtype=float) if np.isscalar(max_weight)
          else np.asarray(max_weight, dtype=float))

    w = weights.copy().astype(float)
    for _ in range(max_iter):
        # Step 1: clip individual asset bounds
        w = np.clip(w, lo, hi)
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

        # Check convergence (works for both scalar and array lo/hi)
        converged = True
        if np.any(w > hi + 1e-8):
            converged = False
        if np.any(w < lo - 1e-8):
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
      "sharpe_slsqp"                         -> -Sharpe
      "random", "calmar"                     -> -Calmar
      "martin", "differential_evolution"     -> -Martin  (CAGR / Ulcer Index)

    Martin ratio is preferred for DE because Calmar is dominated by a single
    worst data point and is discontinuous, making it a poor landscape for
    population-based search. Martin is smoother (RMS drawdown) and better
    reflects persistent drawdown risk.

    A large penalty (1e6) is returned if the CAGR constraint is violated
    or if any asset class group exceeds its maximum weight.
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

    if method in ("martin", "differential_evolution"):
        ulcer = compute_ulcer_index(series)
        if ulcer < 1e-10:
            return -cagr  # degenerate: no drawdowns at all, use CAGR directly
        return -(cagr / ulcer)

    # "random" and "calmar"
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
                    asset_class_max_weight: dict | None = None,
                    asset_bounds: dict | None = None,
                    ) -> tuple[np.ndarray | None, float]:
    """
    Random search: try n_trials random weight combinations and keep the best.

    Each trial draws weights uniformly within per-asset bounds (or uniform
    [min_weight, max_weight] if asset_bounds is not provided), projects the
    sample onto the feasible simplex, and scores with _score_allocation.

    Returns (best_weights_array, best_score). best_weights is None if no
    valid allocation was found (all trials violated the constraints).
    """
    rng = np.random.default_rng(random_seed)

    tickers = list(allocation.keys())
    n       = len(tickers)

    # Resolve per-asset sampling bounds
    if asset_bounds and all(t in asset_bounds for t in tickers):
        lo = np.array([asset_bounds[t][0] for t in tickers])
        hi = np.array([asset_bounds[t][1] for t in tickers])
    else:
        lo = np.full(n, min_weight)
        hi = np.full(n, max_weight)

    best_score   = np.inf
    best_weights = None

    for _ in range(n_trials):
        raw     = rng.uniform(lo, hi)
        weights = _project_weights(raw, tickers, lo, hi,
                                   asset_class_groups, asset_class_max_weight)

        score = _score_allocation(weights, tickers, prices,
                                  benchmark_prices, method, min_cagr,
                                  asset_class_groups, asset_class_max_weight)

        if score < best_score:
            best_score   = score
            best_weights = weights

    display      = -best_score if best_score < 1e5 else float("nan")
    metric_label = "Martin ratio" if method == "martin" else "Calmar"
    print(f"  Optimiser complete | Best {metric_label}: {display:.3f}")

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

    Asset class group constraints and per-asset bounds are read directly from
    config and applied automatically to all methods.

    Parameters
    ----------
    prices, benchmark_prices : price DataFrames from data.fetch_prices()
    allocation               : starting allocation (used as initial guess for SLSQP)
    method                   : "random" | "calmar" | "martin" |
                               "differential_evolution" | "sharpe_slsqp"
    min_weight, max_weight   : fallback uniform per-asset weight bounds
    min_cagr                 : minimum acceptable CAGR (percent)
    n_trials                 : number of trials for random/calmar methods
    random_seed              : numpy random seed for reproducibility
    """
    import config as _config

    tickers = list(allocation.keys())
    n       = len(tickers)

    # Read asset class constraints from config
    asset_class_groups     = getattr(_config, "ASSET_CLASS_GROUPS",     None)
    asset_class_max_weight = getattr(_config, "ASSET_CLASS_MAX_WEIGHT", None)

    # Resolve per-asset bounds; fall back to uniform if ASSET_BOUNDS is absent
    # or doesn't cover every ticker in the current allocation.
    asset_bounds_cfg = getattr(_config, "ASSET_BOUNDS", None)
    missing_from_bounds = (
        [t for t in tickers if t not in (asset_bounds_cfg or {})]
    )
    if asset_bounds_cfg and not missing_from_bounds:
        per_asset_lo = np.array([asset_bounds_cfg[t][0] for t in tickers])
        per_asset_hi = np.array([asset_bounds_cfg[t][1] for t in tickers])
        bounds       = [(asset_bounds_cfg[t][0], asset_bounds_cfg[t][1]) for t in tickers]
        bounds_label = "per-asset ASSET_BOUNDS"
    else:
        per_asset_lo = np.full(n, min_weight)
        per_asset_hi = np.full(n, max_weight)
        bounds       = [(min_weight, max_weight)] * n
        bounds_label = f"uniform [{min_weight:.0%}, {max_weight:.0%}]"
        if missing_from_bounds:
            print(f"WARNING: {missing_from_bounds} not in ASSET_BOUNDS "
                  f"— using uniform fallback [{min_weight:.0%}, {max_weight:.0%}]")

    print(f"\nOptimising ({method}) | {' '.join(tickers)} | bounds: {bounds_label}")
    if asset_class_groups and asset_class_max_weight:
        caps = ", ".join(f"{g}: {v:.0%}" for g, v in asset_class_max_weight.items())
        print(f"  Asset class caps: {caps}")
    print()

    # ------------------------------------------------------------------
    if method in ("random", "calmar", "martin"):
        best_weights, best_score = optimise_random(
            prices, benchmark_prices, allocation,
            min_weight, max_weight, min_cagr, n_trials, method, random_seed,
            asset_class_groups, asset_class_max_weight,
            asset_bounds=asset_bounds_cfg,
        )

        if best_weights is None:
            print("WARNING: No valid allocation found. Try lowering OPT_MIN_CAGR.")
            return allocation

    # ------------------------------------------------------------------
    elif method == "differential_evolution":
        # FIX 2 + FIX 3: project weights inside the objective so DE sees the
        # true landscape; use Martin ratio (smoother than Calmar for DE).
        def de_objective(w):
            w_proj = _project_weights(w, tickers, per_asset_lo, per_asset_hi,
                                      asset_class_groups, asset_class_max_weight)
            return _score_allocation(w_proj, tickers, prices,
                                     benchmark_prices, "martin", min_cagr,
                                     asset_class_groups, asset_class_max_weight)

        result = differential_evolution(
            de_objective,
            bounds   = bounds,
            maxiter  = 400,
            popsize  = 15,
            tol      = 1e-6,
            seed     = random_seed,
            disp     = False,
        )

        if not result.success and result.fun >= 1e5:
            print(f"WARNING: DE did not converge cleanly: {result.message}")

        # FIX 4: project result.x once -- consistent with what was evaluated
        # inside de_objective throughout the search.
        best_weights = _project_weights(result.x, tickers, per_asset_lo, per_asset_hi,
                                        asset_class_groups, asset_class_max_weight)
        best_score   = _score_allocation(best_weights, tickers, prices,
                                         benchmark_prices, "martin", min_cagr,
                                         asset_class_groups, asset_class_max_weight)

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
            f"Choose from: random, calmar, martin, differential_evolution, sharpe_slsqp"
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

    # Print objective metric(s); for DE also report Calmar alongside Martin
    if method == "differential_evolution":
        martin_ratio = -best_score
        # Compute Calmar separately for comparison
        bt_final = run_backtest(prices, benchmark_prices, optimised)
        series   = bt_final["All Weather Value"]
        years    = (bt_final.index[-1] - bt_final.index[0]).days / 365.25
        calmar   = compute_calmar(compute_cagr(series, years),
                                  compute_max_drawdown(series))
        print(f"\nBest Martin ratio: {martin_ratio:.3f}  |  Calmar: {calmar:.3f}")
    elif method == "sharpe_slsqp":
        print(f"\nBest Sharpe: {-best_score:.3f}")
    else:
        print(f"\nBest Calmar: {-best_score:.3f}")

    return optimised


# ===========================================================================
# RISK PARITY DIAGNOSTIC
# ===========================================================================

def compute_risk_parity_weights(prices: pd.DataFrame,
                                tickers: list[str],
                                estimation_years: float = 5.0,
                                min_weight: float = 0.02) -> dict[str, float]:
    """
    Compute risk contribution equalisation (risk parity) weights.

    Finds weights such that every asset contributes equally to total
    portfolio variance. This is the mathematical foundation of Dalio's
    All Weather strategy.

    Objective (minimise):
        sum_i sum_j (TRC_i - TRC_j)^2
    where:
        TRC_i = w_i * (cov @ w)[i] / (w @ cov @ w)   (% risk contribution of asset i)

    The covariance matrix is estimated from daily log returns over the
    most recent `estimation_years` of the provided price history.

    Parameters
    ----------
    prices           : daily price DataFrame (total return, from fetch_prices)
    tickers          : list of tickers to include
    estimation_years : lookback window for covariance estimation (default 5yr)
    min_weight       : minimum weight per asset (default 2% floor)

    Returns
    -------
    dict {ticker: weight} summing to 1.0, rounded to 4 decimal places.
    Also prints a comparison against equal weight for reference.

    Notes
    -----
    - This is a diagnostic tool, not an optimiser. Run it once to understand
      what risk parity suggests, then compare to manual weights.
    - Covariance is estimated from log returns, not arithmetic returns.
      Log returns are more symmetric and better suited to covariance estimation.
    - The objective is convex and smooth, so SLSQP converges reliably.
    - SLSQP is used (not DE) because the risk parity objective is smooth
      and gradient-based methods outperform population-based methods here.
    """
    available = [t for t in tickers if t in prices.columns]
    if not available:
        raise ValueError(f"None of {tickers} found in prices columns.")

    # Slice to estimation window
    cutoff     = prices.index[-1] - pd.DateOffset(years=estimation_years)
    px         = prices.loc[prices.index >= cutoff, available].dropna()
    log_ret    = np.log(px / px.shift(1)).dropna()
    cov        = log_ret.cov().values   # annualisation not needed for weight derivation
    n          = len(available)

    def _risk_contributions(w: np.ndarray) -> np.ndarray:
        """Percentage risk contributions for each asset."""
        portfolio_var = float(w @ cov @ w)
        if portfolio_var < 1e-12:
            return np.ones(n) / n
        marginal = cov @ w
        return (w * marginal) / portfolio_var

    def _objective(w: np.ndarray) -> float:
        """Sum of squared pairwise differences in risk contributions."""
        trc = _risk_contributions(w)
        total = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                total += (trc[i] - trc[j]) ** 2
        return total

    # Initial guess: equal weight
    w0          = np.full(n, 1.0 / n)
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds      = [(min_weight, 1.0)] * n

    result = minimize(
        _objective,
        w0,
        method      = "SLSQP",
        bounds      = bounds,
        constraints = constraints,
        options     = {"maxiter": 2000, "ftol": 1e-12},
    )

    if not result.success:
        print(f"WARNING: risk parity optimisation did not converge: {result.message}")

    w_rp  = result.x / result.x.sum()   # re-normalise for numerical safety
    trc   = _risk_contributions(w_rp)
    equal = np.full(n, 1.0 / n)

    print(f"\nRisk Parity Weights  (estimation window: {estimation_years:.0f}yr, "
          f"{len(log_ret)} daily obs)")
    print(f"  {'Ticker':<6}  {'RP Weight':>9}  {'Equal Wt':>8}  {'Risk Contrib':>12}")
    print(f"  {'-'*6}  {'-'*9}  {'-'*8}  {'-'*12}")
    for i, t in enumerate(available):
        print(f"  {t:<6}  {w_rp[i]:>8.1%}  {equal[i]:>8.1%}  {trc[i]:>11.1%}")

    portfolio_vol = np.sqrt(float(w_rp @ cov @ w_rp) * 252) * 100
    print(f"\n  Annualised portfolio vol (RP weights): {portfolio_vol:.2f}%")
    print(f"  Objective value (should be ~0): {result.fun:.6f}")
    print(f"  Converged: {result.success}")

    return {t: round(float(w_rp[i]), 4) for i, t in enumerate(available)}
