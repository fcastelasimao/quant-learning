"""
archive/optimiser_de.py
=======================
ARCHIVED 2026-03-26 — DE optimiser confirmed to fail across all 26 experiments.
Gate 1 closed (Phase 9). Kept here as reference only. Do not use.

Root cause: IS period 2006-2020 is a single falling-rates regime. DE finds
TLT-heavy weights that collapse in the 2022 rate shock (OOS). Structural —
not fixable by tuning DE parameters or changing the IS/OOS split.

Active methodology: risk parity (compute_risk_parity_weights in optimiser.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution

from backtest import run_backtest, compute_cagr, compute_max_drawdown, \
                     compute_sharpe, compute_calmar, compute_ulcer_index


def _project_weights(weights, tickers, min_weight, max_weight,
                     asset_class_groups=None, asset_class_max_weight=None,
                     max_iter=200):
    """Project weights onto the feasible simplex with per-asset and group caps."""
    n  = len(weights)
    lo = (np.full(n, min_weight, dtype=float) if np.isscalar(min_weight)
          else np.asarray(min_weight, dtype=float))
    hi = (np.full(n, max_weight, dtype=float) if np.isscalar(max_weight)
          else np.asarray(max_weight, dtype=float))
    w = weights.copy().astype(float)
    for _ in range(max_iter):
        w = np.clip(w, lo, hi)
        w = w / w.sum()
        if asset_class_groups and asset_class_max_weight:
            for group, group_tickers in asset_class_groups.items():
                cap     = asset_class_max_weight.get(group, 1.0)
                indices = [tickers.index(t) for t in group_tickers if t in tickers]
                if not indices:
                    continue
                group_total = w[indices].sum()
                if group_total > cap + 1e-8:
                    w[indices] *= cap / group_total
        w = w / w.sum()
        converged = not np.any(w > hi + 1e-8) and not np.any(w < lo - 1e-8)
        if converged and asset_class_groups and asset_class_max_weight:
            for group, group_tickers in asset_class_groups.items():
                cap     = asset_class_max_weight.get(group, 1.0)
                indices = [tickers.index(t) for t in group_tickers if t in tickers]
                if indices and w[indices].sum() > cap + 1e-8:
                    converged = False
        if converged:
            break
    return w


def _score_allocation(weights_array, tickers, prices, benchmark_prices,
                      method, min_cagr, asset_class_groups=None,
                      asset_class_max_weight=None):
    """Score a weight array. Lower is better (objectives are negated)."""
    if asset_class_groups and asset_class_max_weight:
        for group, group_tickers in asset_class_groups.items():
            cap     = asset_class_max_weight.get(group, 1.0)
            indices = [tickers.index(t) for t in group_tickers if t in tickers]
            if sum(weights_array[i] for i in indices) > cap + 1e-6:
                return 1e6
    allocation = dict(zip(tickers, weights_array))
    bt         = run_backtest(prices, benchmark_prices, allocation)
    series     = bt["All Weather Value"]
    years      = (bt.index[-1] - bt.index[0]).days / 365.25
    cagr = compute_cagr(series, years)
    if cagr < min_cagr:
        return 1e6
    if method == "sharpe_slsqp":
        return -compute_sharpe(bt["All Weather Value Monthly Ret (%)"])
    if method in ("martin", "differential_evolution"):
        ulcer = compute_ulcer_index(series)
        return -(cagr / ulcer) if ulcer >= 1e-10 else -cagr
    return -compute_calmar(cagr, compute_max_drawdown(series))


def optimise_de(prices, benchmark_prices, allocation, min_weight, max_weight,
                min_cagr, random_seed, asset_class_groups=None,
                asset_class_max_weight=None, asset_bounds_cfg=None):
    """
    Differential Evolution optimiser.
    ARCHIVED — confirmed to fail OOS across 26 experiments (Phase 9).
    """
    tickers = list(allocation.keys())
    n       = len(tickers)
    missing = [t for t in tickers if t not in (asset_bounds_cfg or {})]
    if asset_bounds_cfg and not missing:
        per_asset_lo = np.array([asset_bounds_cfg[t][0] for t in tickers])
        per_asset_hi = np.array([asset_bounds_cfg[t][1] for t in tickers])
        bounds = [(asset_bounds_cfg[t][0], asset_bounds_cfg[t][1]) for t in tickers]
    else:
        per_asset_lo = np.full(n, min_weight)
        per_asset_hi = np.full(n, max_weight)
        bounds = [(min_weight, max_weight)] * n

    def de_objective(w):
        w_proj = _project_weights(w, tickers, per_asset_lo, per_asset_hi,
                                  asset_class_groups, asset_class_max_weight)
        return _score_allocation(w_proj, tickers, prices, benchmark_prices,
                                 "martin", min_cagr,
                                 asset_class_groups, asset_class_max_weight)

    result = differential_evolution(
        de_objective, bounds=bounds, maxiter=400, popsize=15,
        tol=1e-6, seed=random_seed, disp=False,
    )
    best_weights = _project_weights(
        result.x, tickers, per_asset_lo, per_asset_hi,
        asset_class_groups, asset_class_max_weight,
    )
    return dict(zip(tickers, best_weights))
