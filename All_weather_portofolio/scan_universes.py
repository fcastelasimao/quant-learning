"""
scan_universes.py
=================
Scan all possible ETF subsets of size 6-8 from a broad candidate universe.
Score each subset by diversification ratio under risk parity weights.
Output the top candidates for manual review and OOS validation.

Run with:
    conda run -n allweather python3 scan_universes.py

Takes ~10-15 minutes (brute-force over ~18,000 subsets).
"""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from data import fetch_prices

# ---- Configuration ----

BACKTEST_START = "2006-07-01"   # GSG/DJP inception constraint
IS_END         = "2020-01-01"   # use IS-only data for selection
COV_YEARS      = 5.0            # covariance estimation window

# Every candidate must have data from BACKTEST_START.
CANDIDATE_UNIVERSE = [
    # Equity
    "SPY",   # US broad (1993)
    "QQQ",   # US tech/growth (1999)
    "IWD",   # US value (2000)
    "EFA",   # International developed (2001)
    "EEM",   # Emerging markets (2003)
    # Bonds
    "TLT",   # Long-term government (2002)
    "IEF",   # Intermediate government (2002)
    "SHY",   # Short-term government (2002)
    "TIP",   # Inflation-linked (2003)
    "LQD",   # Investment-grade corporate (2002)
    "AGG",   # US aggregate (2003)
    # Real assets
    "GLD",   # Gold (2004)
    "VNQ",   # US REITs (2004)
    "GSG",   # Broad commodities (2006)
    "DJP",   # Diversified commodities (2006)
]

# Every valid subset must include at least one from each role.
REQUIRED_ROLES = {
    "equity":    ["SPY", "QQQ", "IWD", "EFA", "EEM"],
    "bonds":     ["TLT", "IEF", "SHY", "TIP", "LQD", "AGG"],
    "real_assets": ["GLD", "VNQ", "GSG", "DJP"],
}

SUBSET_SIZES = [6, 7, 8]
MIN_PORTFOLIO_VOL = 4.0   # % annualised — reject cash-like portfolios
MAX_SINGLE_WEIGHT = 0.40  # no asset above 40%
TOP_N = 20   # how many top results to print


def solve_rp(cov_matrix: np.ndarray, min_weight: float = 0.02) -> np.ndarray:
    """Solve risk parity for a given covariance matrix. Returns weight array."""
    n = cov_matrix.shape[0]

    def _risk_contributions(w):
        pvar = float(w @ cov_matrix @ w)
        if pvar < 1e-12:
            return np.ones(n) / n
        return (w * (cov_matrix @ w)) / pvar

    def _objective(w):
        trc = _risk_contributions(w)
        total = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                total += (trc[i] - trc[j]) ** 2
        return total

    w0 = np.full(n, 1.0 / n)
    result = minimize(
        _objective, w0, method="SLSQP",
        bounds=[(min_weight, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    return result.x / result.x.sum()


def diversification_ratio(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """
    DR = (sum of w_i * sigma_i) / sigma_portfolio
    Higher = more diversification benefit from combining these assets.
    """
    individual_vols = np.sqrt(np.diag(cov_matrix))
    weighted_avg_vol = np.dot(weights, individual_vols)
    portfolio_vol = np.sqrt(float(weights @ cov_matrix @ weights))
    if portfolio_vol < 1e-12:
        return 0.0
    return weighted_avg_vol / portfolio_vol


def has_all_roles(subset: tuple, required_roles: dict) -> bool:
    """Check that the subset has at least one ETF from each required role."""
    subset_set = set(subset)
    return all(
        bool(subset_set & set(role_tickers))
        for role_tickers in required_roles.values()
    )


def main():
    print("=" * 70)
    print("ETF UNIVERSE SCAN — Diversification Ratio under Risk Parity")
    print("=" * 70)

    # Fetch daily prices for all candidates
    print(f"\nFetching prices for {len(CANDIDATE_UNIVERSE)} ETFs...")
    prices = fetch_prices(CANDIDATE_UNIVERSE, BACKTEST_START, IS_END)

    # Drop any tickers that failed to download
    available = [t for t in CANDIDATE_UNIVERSE if t in prices.columns]
    missing = set(CANDIDATE_UNIVERSE) - set(available)
    if missing:
        print(f"WARNING: No data for {missing}. Excluding from scan.")

    # Compute log returns and covariance (last COV_YEARS of IS period)
    cutoff = prices.index[-1] - pd.DateOffset(years=COV_YEARS)
    px = prices.loc[prices.index >= cutoff, available].dropna()
    log_ret = np.log(px / px.shift(1)).dropna()
    full_cov = log_ret.cov().values
    ticker_idx = {t: i for i, t in enumerate(available)}

    # Print correlation matrix for reference
    corr = log_ret.corr()
    print(f"\nCorrelation matrix ({len(available)} assets, {len(log_ret)} daily obs):")
    print(corr.round(2).to_string())

    # Scan all subsets
    results = []
    total_checked = 0
    total_skipped = 0

    for size in SUBSET_SIZES:
        for subset in combinations(available, size):
            # Filter: must have all roles
            if not has_all_roles(subset, REQUIRED_ROLES):
                total_skipped += 1
                continue

            total_checked += 1

            # Extract sub-covariance matrix
            idx = [ticker_idx[t] for t in subset]
            sub_cov = full_cov[np.ix_(idx, idx)]

            # Solve RP
            rp_w = solve_rp(sub_cov)

            # Score
            port_vol = np.sqrt(float(rp_w @ sub_cov @ rp_w) * 252) * 100

            # Filter: reject cash-like portfolios and single-asset concentration
            if port_vol < MIN_PORTFOLIO_VOL:
                total_skipped += 1
                continue
            if max(rp_w) > MAX_SINGLE_WEIGHT:
                total_skipped += 1
                continue

            dr = diversification_ratio(rp_w, sub_cov)

            results.append({
                "subset": subset,
                "size": size,
                "weights": {t: round(float(rp_w[i]), 4) for i, t in enumerate(subset)},
                "div_ratio": round(dr, 4),
                "port_vol_ann": round(port_vol, 2),
            })

    results.sort(key=lambda x: x["div_ratio"], reverse=True)

    print(f"\nScanned {total_checked} valid subsets ({total_skipped} skipped for missing roles)")
    print(f"\n{'=' * 70}")
    print(f"TOP {TOP_N} SUBSETS BY DIVERSIFICATION RATIO")
    print(f"{'=' * 70}")
    print(f"{'Rank':<5} {'Size':<5} {'DR':>6} {'Vol%':>6}  {'Assets & RP Weights'}")
    print("-" * 70)

    for i, r in enumerate(results[:TOP_N]):
        weights_str = "  ".join(f"{t}:{w:.0%}" for t, w in r["weights"].items())
        print(f"{i+1:<5} {r['size']:<5} {r['div_ratio']:>6.3f} {r['port_vol_ann']:>5.1f}%  {weights_str}")

    # Show bottom 5 for contrast
    print(f"\n{'BOTTOM 5 (worst diversification)':}")
    print("-" * 70)
    for r in results[-5:]:
        weights_str = "  ".join(f"{t}:{w:.0%}" for t, w in r["weights"].items())
        print(f"  {r['size']}  DR={r['div_ratio']:.3f}  Vol={r['port_vol_ann']:.1f}%  {weights_str}")

    # Analysis: which ETFs appear most in top 20?
    print(f"\n{'=' * 70}")
    print("ETF FREQUENCY IN TOP 20")
    print(f"{'=' * 70}")
    from collections import Counter
    ticker_counts = Counter()
    for r in results[:TOP_N]:
        ticker_counts.update(r["subset"])
    for t, count in ticker_counts.most_common():
        bar = "█" * count
        print(f"  {t:<5} {count:>3}  {bar}")

    # Save full results to CSV for further analysis
    import csv
    out_path = "results/universe_scan_results.csv"
    import os
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "size", "div_ratio", "port_vol_ann", "assets", "weights"])
        for i, r in enumerate(results):
            w.writerow([
                i + 1, r["size"], r["div_ratio"], r["port_vol_ann"],
                "|".join(r["subset"]),
                "|".join(f"{t}={v:.4f}" for t, v in r["weights"].items()),
            ])
    print(f"\nFull results saved to {out_path} ({len(results)} rows)")


if __name__ == "__main__":
    main()