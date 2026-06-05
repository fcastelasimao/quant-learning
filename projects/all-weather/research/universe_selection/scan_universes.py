"""
scan_universes.py
=================
Scan ETF subsets from a broad (~100) candidate universe.
Score each subset by diversification ratio under risk parity weights.

Strategy for large universes (>30 candidates):
  1. Download all available price data (no fixed start date).
  2. Remove ETFs with < MIN_HISTORY_DAYS of data before IS_END.
  3. De-duplicate: if two ETFs have correlation > DEDUP_CORR_THRESHOLD,
     keep the one with longer history (or cheaper if equal).
  4. Random-sample N_RANDOM_SAMPLES subsets instead of brute-force.
  5. For each subset, covariance is computed from the *common* overlap
     window (last COV_YEARS before IS_END, but only dates where all
     members of that subset have data).

Run with:
    conda run -n allweather python3 scan_universes.py

For the original brute-force on a small universe (<=30 after dedup):
    conda run -n allweather python3 scan_universes.py --exhaustive
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import warnings

import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", message=".*auto_adjust.*")

# ---- Configuration ----

FETCH_START    = "1993-01-01"   # fetch as far back as possible
IS_END         = "2020-01-01"   # use IS-only data for selection
COV_YEARS      = 5.0            # covariance estimation window

MIN_HISTORY_DAYS = 504          # ~2 trading years — minimum to include an ETF
DEDUP_CORR_THRESHOLD = 0.95     # merge ETFs with higher correlation
N_RANDOM_SAMPLES = 100_000      # random subsets to evaluate (if not exhaustive)
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Candidate universe — ~100 ETFs across 7 macro buckets
# Inception year in comment; the scanner handles variable start dates per
# subset so newer ETFs are not excluded a priori.
# ---------------------------------------------------------------------------

CANDIDATE_UNIVERSE = [
    # ── US Equity ──────────────────────────────────────────────────────────
    "SPY",   # S&P 500 large-cap (1993)
    "QQQ",   # Nasdaq-100 tech/growth (1999)
    #"IWD",   # Russell 1000 Value (2000)
    "IJR",   # S&P 600 Small-Cap (2000)
    #"VO",    # Vanguard Mid-Cap (2004)
    #"RSP",   # S&P 500 Equal Weight (2003)
    #"VIG",   # Dividend Appreciation (2006)
    #"SCHD",  # Schwab Dividend Equity (2011)
    #"USMV",  # MSCI USA Min Volatility (2011)
    "MTUM",  # MSCI USA Momentum (2013)
    #"QUAL",  # MSCI USA Quality (2013)
    #"VTI",   # Total US Stock Market (2001)

    # ── International Equity ───────────────────────────────────────────────
    #"EFA",   # MSCI EAFE Developed (2001)
    #"VWO",   # FTSE Emerging Markets (2005)
    #"EEM",   # MSCI Emerging Markets (2003)
    #"EZU",   # MSCI Eurozone (2000)
    #"EWJ",   # MSCI Japan (1996)
    #"EWU",   # MSCI UK (1996)
    #"MCHI",  # MSCI China (2011)
    #"INDA",  # MSCI India (2012)
    #"ILF",   # Latin America 40 (2001)
    #"VPL",   # FTSE Pacific (2005)
    #"FM",    # MSCI Frontier Markets (2012)
    #"VXUS",  # Total Intl ex-US (2011)

    # ── US Treasuries (full duration spectrum) ─────────────────────────────
    #"SHY",   # 1-3yr Treasury (2002)
    #"IEI",   # 3-7yr Treasury (2007)
    "IEF",   # 7-10yr Treasury (2002)
    "TLT",   # 20+yr Treasury (2002)
    #"EDV",   # Extended Duration STRIPS (2007)
    #"ZROZ",  # 25+yr Zero Coupon (2009)
    #"VGIT",  # Vanguard Intermediate Treasury (2009)
    #"VGLT",  # Vanguard Long Treasury (2009)
    #"GOVT",  # All-Duration Treasury Blend (2012)

    # ── Inflation-Protected ────────────────────────────────────────────────
    "TIP",   # TIPS all maturities (2003)
    #"SCHP",  # Schwab TIPS (2010)
    #"VTIP",  # Vanguard Short-Term TIPS (2012)

    # ── Credit / Corporate ─────────────────────────────────────────────────
    #"LQD",   # IG Corporate (2002)
    #"VCIT",  # Vanguard Intermediate Corp (2009)
    #"HYG",   # High-Yield Corporate (2007)
    #"JNK",   # High-Yield (alt) (2007)
    #"FLOT",  # Floating Rate IG (2011)
    #"SRLN",  # Senior Loans / Floating Rate (2013)

    # ── Aggregate / Other Fixed Income ─────────────────────────────────────
    #"AGG",   # US Aggregate Bond (2003)
    #"BND",   # Vanguard Total Bond Market (2007)
    #"BNDX",  # Total Intl Bond Hedged (2013)
    #"EMB",   # EM USD Sovereign Debt (2007)
    #"MBB",   # Mortgage-Backed Securities (2007)
    #"JAAA",  # AAA CLOs (2020)

    # ── Commodities ────────────────────────────────────────────────────────
    "GLD",   # Gold (2004)
    #"IAU",   # Gold cheaper (2005)
    "SLV",   # Silver (2006)
    #"GSG",   # S&P GSCI Broad Commodity (2006)
    #"DJP",   # Diversified Commodities (2006)
    #"PDBC",  # Optimum Yield Commodity no K-1 (2014)
    #"COM",   # Auspice Trend Commodities (2010)
    "DBA",   # Agriculture (2007)
    #"USO",   # Crude Oil WTI (2006)
    "CPER",  # Copper (2011)
    #"URA",   # Uranium Miners (2010)
    #"WOOD",  # Global Timber & Forestry (2008)
    #"KRBN",  # Carbon Credits (2020)

    # ── Real Assets ────────────────────────────────────────────────────────
    #"VNQ",   # US REITs (2004)
    #"SCHH",  # Schwab US REIT (2011)
    #"VNQI",  # International REITs (2010)
    #"IGF",   # Global Infrastructure (2007)

    # ── Alternatives ───────────────────────────────────────────────────────
    #"DBMF",  # Managed Futures (2019)
    #"KMLM",  # Mount Lucas Managed Futures (2020)
    #"CTA",   # Simplify Managed Futures (2022)
    #"MNA",   # Merger Arbitrage (2009)
    #"CWB",   # Convertible Bonds (2009)
    #"PFF",   # Preferred Securities (2007)
    #"SVXY",  # Short VIX (2011)
    #"IBIT",  # Bitcoin Spot (2024)
    #"BITO",  # Bitcoin Futures (2021)

    # ── Cash / Ultra-Short ─────────────────────────────────────────────────
    #"BIL",   # 1-3mo T-Bill (2007)
    #"SHV",   # Short Treasury (2007)
    #"SGOV",  # 0-3mo Treasury (2020)
    #"TFLO",  # Treasury Floating Rate (2014)
]

# Every valid subset must include at least one from each macro role.
REQUIRED_ROLES = {
    "equity": [
        "SPY", "QQQ", "IWD", "IJR", "VO", "RSP", "VIG", "SCHD",
        "USMV", "MTUM", "QUAL", "VTI",
        "EFA", "VWO", "EEM", "EZU", "EWJ", "EWU", "MCHI", "INDA",
        "ILF", "VPL", "FM", "VXUS",
    ],
    "bonds": [
        "SHY", "IEI", "IEF", "TLT", "EDV", "ZROZ", "VGIT", "VGLT",
        "GOVT", "TIP", "SCHP", "VTIP",
        "LQD", "VCIT", "HYG", "JNK", "FLOT", "SRLN",
        "AGG", "BND", "BNDX", "EMB", "MBB", "JAAA",
    ],
    "real_assets": [
        "GLD", "IAU", "SLV", "GSG", "DJP", "PDBC", "COM", "DBA",
        "USO", "CPER", "URA", "WOOD", "KRBN",
        "VNQ", "SCHH", "VNQI", "IGF",
        "DBMF", "KMLM", "CTA", "MNA", "CWB", "PFF", "SVXY",
        "IBIT", "BITO",
    ],
}

SUBSET_SIZES = [8, 9, 10]
MIN_PORTFOLIO_VOL = 4.0   # % annualised — reject cash-like portfolios
MAX_SINGLE_WEIGHT = 0.40  # no asset above 40%
TOP_N = 30   # how many top results to print


# ===========================================================================
# Data fetching — tolerant of missing/new ETFs
# ===========================================================================

def fetch_prices_tolerant(
    tickers: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Download adjusted close for all tickers, silently dropping any that fail.
    Unlike data.fetch_prices(), this never raises on missing data.
    """
    print(f"  Downloading {len(tickers)} tickers ({start} → {end})...")
    raw = yf.download(tickers, start=start, end=end,
                       progress=False, auto_adjust=True)
    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.dropna(how="all").ffill()

    # Drop columns that are entirely NaN (ticker didn't exist in range)
    prices = prices.dropna(axis=1, how="all")
    return prices


# ===========================================================================
# Risk parity solver
# ===========================================================================

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
    DR = (sum of w_i * sigma_i) / sigma_portfolio.
    Higher = more diversification benefit from combining these assets.
    """
    individual_vols = np.sqrt(np.diag(cov_matrix))
    weighted_avg_vol = np.dot(weights, individual_vols)
    portfolio_vol = np.sqrt(float(weights @ cov_matrix @ weights))
    if portfolio_vol < 1e-12:
        return 0.0
    return weighted_avg_vol / portfolio_vol


# ===========================================================================
# Data preparation helpers
# ===========================================================================

def filter_by_history(
    prices: pd.DataFrame,
    is_end: str,
    min_days: int,
) -> list[str]:
    """Return tickers that have at least min_days of data before is_end."""
    end = pd.Timestamp(is_end)
    kept = []
    for col in prices.columns:
        series = prices[col].dropna()
        series = series[series.index < end]
        if len(series) >= min_days:
            kept.append(col)
    return kept


def deduplicate_by_correlation(
    prices: pd.DataFrame,
    tickers: list[str],
    threshold: float,
) -> tuple[list[str], list[tuple[str, str, float]]]:
    """
    Remove near-duplicates: if corr(A, B) > threshold, drop the one with
    less history. Returns (kept_tickers, removed_pairs).
    """
    # Use the longest common window for correlation
    px = prices[tickers].dropna()
    if len(px) < 60:
        return tickers, []

    log_ret = np.log(px / px.shift(1)).dropna()
    corr = log_ret.corr()

    removed: set[str] = set()
    removed_pairs: list[tuple[str, str, float]] = []

    # History length per ticker (total non-NaN days)
    history_len = {t: prices[t].dropna().shape[0] for t in tickers}

    for i, t1 in enumerate(tickers):
        if t1 in removed:
            continue
        for t2 in tickers[i + 1:]:
            if t2 in removed:
                continue
            if t1 not in corr.columns or t2 not in corr.columns:
                continue
            c = corr.loc[t1, t2]
            if abs(c) > threshold:
                # Drop the one with less history
                drop = t2 if history_len[t1] >= history_len[t2] else t1
                removed.add(drop)
                removed_pairs.append((t1, t2, round(c, 3)))

    kept = [t for t in tickers if t not in removed]
    return kept, removed_pairs


def compute_subset_cov(
    prices: pd.DataFrame,
    subset: tuple[str, ...],
    is_end: str,
    cov_years: float,
) -> np.ndarray | None:
    """
    Compute covariance matrix for a subset using their common overlap window.
    Returns None if insufficient data (< 252 common trading days).
    """
    end = pd.Timestamp(is_end)
    start = end - pd.DateOffset(years=cov_years)
    px = prices.loc[start:end, list(subset)].dropna()
    if len(px) < 252:
        return None
    log_ret = np.log(px / px.shift(1)).dropna()
    return log_ret.cov().values


# ===========================================================================
# Subset validation
# ===========================================================================

def has_all_roles(subset: tuple, required_roles: dict) -> bool:
    """Check that the subset has at least one ETF from each required role."""
    subset_set = set(subset)
    return all(
        bool(subset_set & set(role_tickers))
        for role_tickers in required_roles.values()
    )


def _role_sets() -> dict[str, set[str]]:
    """Pre-compute role sets for fast lookup."""
    return {role: set(tickers) for role, tickers in REQUIRED_ROLES.items()}


# ===========================================================================
# Sampling strategies
# ===========================================================================

def generate_random_subsets(
    available: list[str],
    sizes: list[int],
    n_samples: int,
    rng: random.Random,
    role_sets: dict[str, set[str]],
) -> list[tuple[str, ...]]:
    """Generate n_samples random subsets that satisfy role constraints."""
    seen: set[tuple[str, ...]] = set()
    valid: list[tuple[str, ...]] = []
    attempts = 0
    max_attempts = n_samples * 20  # give up if too many misses

    while len(valid) < n_samples and attempts < max_attempts:
        attempts += 1
        size = rng.choice(sizes)
        subset = tuple(sorted(rng.sample(available, size)))
        if subset in seen:
            continue
        seen.add(subset)
        subset_set = set(subset)
        if all(bool(subset_set & rs) for rs in role_sets.values()):
            valid.append(subset)

    return valid


def generate_exhaustive_subsets(
    available: list[str],
    sizes: list[int],
    role_sets: dict[str, set[str]],
) -> list[tuple[str, ...]]:
    """Generate all valid subsets (only feasible for small universes)."""
    valid = []
    for size in sizes:
        for subset in combinations(available, size):
            subset_set = set(subset)
            if all(bool(subset_set & rs) for rs in role_sets.values()):
                valid.append(subset)
    return valid


# ===========================================================================
# Scoring engine
# ===========================================================================

def score_subsets(
    subsets: list[tuple[str, ...]],
    prices: pd.DataFrame,
    is_end: str,
    cov_years: float,
    min_vol: float,
    max_weight: float,
) -> list[dict]:
    """Score a list of subsets. Returns sorted results (best first)."""
    results = []
    skipped_data = 0
    skipped_vol = 0
    skipped_conc = 0

    total = len(subsets)
    report_every = max(1, total // 20)

    for i, subset in enumerate(subsets):
        if (i + 1) % report_every == 0:
            pct = (i + 1) / total * 100
            print(f"  [{pct:5.1f}%] scored {i + 1:,}/{total:,} subsets "
                  f"({len(results):,} valid so far)", flush=True)

        sub_cov = compute_subset_cov(prices, subset, is_end, cov_years)
        if sub_cov is None:
            skipped_data += 1
            continue

        rp_w = solve_rp(sub_cov)
        port_vol = np.sqrt(float(rp_w @ sub_cov @ rp_w) * 252) * 100

        if port_vol < min_vol:
            skipped_vol += 1
            continue
        if max(rp_w) > max_weight:
            skipped_conc += 1
            continue

        dr = diversification_ratio(rp_w, sub_cov)

        results.append({
            "subset": subset,
            "size": len(subset),
            "weights": {t: round(float(rp_w[j]), 4)
                        for j, t in enumerate(subset)},
            "div_ratio": round(dr, 4),
            "port_vol_ann": round(port_vol, 2),
        })

    results.sort(key=lambda x: x["div_ratio"], reverse=True)

    print(f"\n  Scoring complete: {len(results):,} valid | "
          f"skipped: {skipped_data:,} (data), {skipped_vol:,} (vol), "
          f"{skipped_conc:,} (concentration)")

    return results


# ===========================================================================
# Reporting
# ===========================================================================

def print_results(results: list[dict], top_n: int) -> None:
    """Print top and bottom subsets plus ETF frequency table."""
    print(f"\n{'=' * 80}")
    print(f"TOP {min(top_n, len(results))} SUBSETS BY DIVERSIFICATION RATIO")
    print(f"{'=' * 80}")
    print(f"{'Rank':<5} {'Size':<5} {'DR':>6} {'Vol%':>6}  {'Assets & RP Weights'}")
    print("-" * 80)

    for i, r in enumerate(results[:top_n]):
        weights_str = "  ".join(
            f"{t}:{w:.0%}" for t, w in r["weights"].items()
        )
        print(f"{i+1:<5} {r['size']:<5} {r['div_ratio']:>6.3f} "
              f"{r['port_vol_ann']:>5.1f}%  {weights_str}")

    if len(results) >= 5:
        print(f"\n{'BOTTOM 5 (worst diversification)':}")
        print("-" * 80)
        for r in results[-5:]:
            weights_str = "  ".join(
                f"{t}:{w:.0%}" for t, w in r["weights"].items()
            )
            print(f"  {r['size']}  DR={r['div_ratio']:.3f}  "
                  f"Vol={r['port_vol_ann']:.1f}%  {weights_str}")

    # ETF frequency in top results
    top_count = min(top_n, len(results))
    print(f"\n{'=' * 80}")
    print(f"ETF FREQUENCY IN TOP {top_count}")
    print(f"{'=' * 80}")
    ticker_counts: Counter = Counter()
    for r in results[:top_count]:
        ticker_counts.update(r["subset"])
    for t, count in ticker_counts.most_common():
        bar = "█" * count
        print(f"  {t:<6} {count:>3}  {bar}")


def save_results(results: list[dict]) -> str:
    """Save all results to CSV. Returns the output path."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "universe_scan_results_reduced_8-10.csv")

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "rank", "size", "div_ratio", "port_vol_ann", "assets", "weights",
        ])
        for i, r in enumerate(results):
            w.writerow([
                i + 1, r["size"], r["div_ratio"], r["port_vol_ann"],
                "|".join(r["subset"]),
                "|".join(f"{t}={v:.4f}" for t, v in r["weights"].items()),
            ])
    return out_path


# ===========================================================================
# Main
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan ETF subsets for diversification under risk parity."
    )
    parser.add_argument(
        "--exhaustive", action="store_true",
        help="Brute-force all subsets (only feasible if universe ≤ ~30 after dedup).",
    )
    parser.add_argument(
        "--samples", type=int, default=N_RANDOM_SAMPLES,
        help=f"Number of random subsets to evaluate (default: {N_RANDOM_SAMPLES:,}).",
    )
    parser.add_argument(
        "--no-dedup", action="store_true",
        help="Skip correlation-based de-duplication.",
    )
    parser.add_argument(
        "--dedup-threshold", type=float, default=DEDUP_CORR_THRESHOLD,
        help=f"Correlation threshold for de-duplication (default: {DEDUP_CORR_THRESHOLD}).",
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help=f"Random seed for reproducibility (default: {RANDOM_SEED}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 80)
    print("ETF UNIVERSE SCAN — Diversification Ratio under Risk Parity")
    print("=" * 80)

    # ── Step 1: Fetch all price data ──────────────────────────────────────
    n_candidates = len(CANDIDATE_UNIVERSE)
    print(f"\n[1/5] Fetching prices for {n_candidates} candidate ETFs...")
    prices = fetch_prices_tolerant(CANDIDATE_UNIVERSE, FETCH_START, IS_END)

    available = [t for t in CANDIDATE_UNIVERSE if t in prices.columns]
    missing = sorted(set(CANDIDATE_UNIVERSE) - set(available))
    if missing:
        print(f"  WARNING: No data for {missing}. Excluding from scan.")
    print(f"  Downloaded: {len(available)} ETFs")

    # ── Step 2: Filter by history length ──────────────────────────────────
    print(f"\n[2/5] Filtering: require ≥ {MIN_HISTORY_DAYS} trading days "
          f"before {IS_END}...")
    available = filter_by_history(prices, IS_END, MIN_HISTORY_DAYS)
    print(f"  Passed: {len(available)} ETFs")

    # Show earliest data date per ETF
    for t in sorted(available):
        first_date = prices[t].dropna().index[0].strftime("%Y-%m-%d")
        n_days = prices[t].dropna().shape[0]
        print(f"    {t:<6} from {first_date}  ({n_days:,} days)")

    # ── Step 3: De-duplicate ──────────────────────────────────────────────
    if not args.no_dedup:
        print(f"\n[3/5] De-duplicating (correlation > {args.dedup_threshold})...")
        available, removed_pairs = deduplicate_by_correlation(
            prices, available, args.dedup_threshold,
        )
        if removed_pairs:
            for t1, t2, c in removed_pairs:
                # Figure out which was dropped
                kept = t1 if t1 in available else t2
                dropped = t2 if kept == t1 else t1
                print(f"    {dropped:<6} dropped (corr={c:.3f} with {kept})")
        print(f"  After dedup: {len(available)} ETFs")
    else:
        print(f"\n[3/5] De-duplication skipped (--no-dedup).")

    # ── Step 4: Generate subsets ──────────────────────────────────────────
    role_sets = _role_sets()
    n_avail = len(available)

    # Estimate brute-force size
    from math import comb
    total_combos = sum(comb(n_avail, s) for s in SUBSET_SIZES)
    use_exhaustive = args.exhaustive or total_combos <= 500_000

    if use_exhaustive:
        print(f"\n[4/5] Generating subsets (exhaustive: {total_combos:,} "
              f"total combos)...")
        subsets = generate_exhaustive_subsets(available, SUBSET_SIZES, role_sets)
    else:
        print(f"\n[4/5] Generating {args.samples:,} random subsets "
              f"(exhaustive would be {total_combos:,} combos)...")
        rng = random.Random(args.seed)
        subsets = generate_random_subsets(
            available, SUBSET_SIZES, args.samples, rng, role_sets,
        )

    print(f"  Generated: {len(subsets):,} valid subsets with all roles")

    if not subsets:
        print("\nERROR: No valid subsets found. Check REQUIRED_ROLES coverage.")
        sys.exit(1)

    # ── Step 5: Score ─────────────────────────────────────────────────────
    print(f"\n[5/5] Scoring subsets...")
    t0 = time.time()
    results = score_subsets(
        subsets, prices, IS_END, COV_YEARS,
        MIN_PORTFOLIO_VOL, MAX_SINGLE_WEIGHT,
    )
    elapsed = time.time() - t0
    print(f"  Elapsed: {elapsed:.1f}s")

    if not results:
        print("\nNo valid results after filtering. Try relaxing constraints.")
        sys.exit(1)

    # ── Report ────────────────────────────────────────────────────────────
    print_results(results, TOP_N)

    out_path = save_results(results)
    print(f"\nFull results saved to {out_path} ({len(results):,} rows)")


if __name__ == "__main__":
    main()
