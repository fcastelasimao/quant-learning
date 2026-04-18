#!/usr/bin/env python3
"""
Batch re-run script:
1. Re-runs all existing implementations on the 449-stock universe
2. Backtests 14 hand-crafted factors (mean-reversion, volatility, quality, value)
3. Logs all results to the SQLite knowledge base

Performance notes:
- IC computation is vectorized (pandas rank, not scipy per-date loop): ~50-100x faster
- IC decay restricted to OOS dates only: ~2x faster
- Parallel execution via ProcessPoolExecutor: N_WORKERS x faster
  Set N_WORKERS=1 to disable parallelism (useful for debugging).
"""
import json
import sys
import traceback
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import sqlite3

from qframe.pipeline.executor import make_factor_fn, run_factor_with_timeout, validate_factor_output
from qframe.factor_harness.walkforward import WalkForwardValidator
from qframe.factor_harness.costs import DEFAULT_COST_PARAMS
from qframe.knowledge_base.db import KnowledgeBase

PRICES_PATH = "data/processed/sp500_close.parquet"
KB_PATH = "knowledge_base/qframe.db"
OOS_START = "2018-01-01"
MIN_STOCKS = 50
UNIVERSE = "sp500_449_survivorship_biased"
TIMEOUT = 360  # seconds per factor

# Number of parallel workers. Default: half the logical CPU count, min 1, max 6.
# Reduce to 1 if you hit memory issues or want serial debugging output.
N_WORKERS = min(6, max(1, os.cpu_count() // 2))


# ---------------------------------------------------------------------------
# Worker: compute one factor (runs in a subprocess, returns metrics dict)
# ---------------------------------------------------------------------------

def _worker(prices_path: str, fn_code: str, impl_id: int, factor_name: str):
    """
    Subprocess worker: load prices, compute factor, run walk-forward.
    Returns (impl_id, factor_name, metrics_dict) on success, or
    (impl_id, factor_name, None) on any error.
    Each worker loads prices independently so subprocesses don't share memory.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    import numpy as np
    import pandas as pd
    import traceback
    from qframe.pipeline.executor import make_factor_fn, run_factor_with_timeout, validate_factor_output
    from qframe.factor_harness.walkforward import WalkForwardValidator
    from qframe.factor_harness.costs import DEFAULT_COST_PARAMS

    try:
        prices = pd.read_parquet(prices_path).sort_index()
        fn = make_factor_fn(fn_code)
        factor_df = run_factor_with_timeout(fn, prices, timeout=TIMEOUT)
        validate_factor_output(factor_df, prices)

        validator = WalkForwardValidator(
            factor_fn=fn,
            oos_start=OOS_START,
            horizon=1,
            cost_params=DEFAULT_COST_PARAMS,
            min_stocks=MIN_STOCKS,
        )
        wf = validator.run(prices)
        s = wf.summary()

        passed = 1 if s["ic"] >= 0.015 and s["icir"] >= 0.15 else 0
        metrics = {
            "ic":             s["ic"],
            "icir":           s["icir"],
            "net_ic":         s["net_ic"],
            "sharpe":         s["sharpe"],
            "turnover":       s["turnover"],
            "decay_halflife": s["decay_halflife"],
            "slow_icir_21":   s.get("slow_icir_21"),
            "slow_icir_63":   s.get("slow_icir_63"),
            "ic_decay_json":  s.get("ic_decay_json"),
            "ic_horizon_1":   s.get("ic_horizon_1"),
            "ic_horizon_5":   s.get("ic_horizon_5"),
            "ic_horizon_21":  s.get("ic_horizon_21"),
            "ic_horizon_63":  s.get("ic_horizon_63"),
            "oos_start":      OOS_START,
            "oos_end":        s["oos_end"],
            "universe":       UNIVERSE,
            "passed_gate":    passed,
        }
        return impl_id, factor_name, metrics

    except Exception:
        tb_lines = traceback.format_exc().strip().splitlines()
        return impl_id, factor_name, None, tb_lines[-1][:120]


# ---------------------------------------------------------------------------
# Helper: run a batch of (impl_id, name, code) tuples in parallel, log to KB
# ---------------------------------------------------------------------------

def run_batch(items: list[tuple[int, str, str]], kb, label: str = ""):
    """
    Run a list of (impl_id, factor_name, fn_code) through the walk-forward
    harness using up to N_WORKERS parallel processes.  Results are logged to
    the KnowledgeBase from the main process (SQLite is not fork-safe).

    Returns (n_ok, n_err).
    """
    n_ok = n_err = 0
    prices_path = str(Path(PRICES_PATH).resolve())

    # Use 'fork' context on POSIX — avoids spawn re-import overhead and the
    # 'must be run from __main__' restriction. Safe here because we don't use
    # any fork-unsafe libraries (Tk, CoreFoundation, etc.) in the workers.
    mp_ctx = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=mp_ctx) as pool:
        futures = {
            pool.submit(_worker, prices_path, code, impl_id, name): (impl_id, name)
            for impl_id, name, code in items
        }
        for fut in as_completed(futures):
            result = fut.result()
            if len(result) == 4:  # error tuple
                impl_id, factor_name, _, err_msg = result
                print(f"  [ERROR] {factor_name:40s} | {err_msg}")
                n_err += 1
            else:
                impl_id, factor_name, metrics = result
                passed = metrics["passed_gate"]
                gate_str = "PASS" if passed else "FAIL"
                print(
                    f"  [{gate_str}] {factor_name:40s} | IC={metrics['ic']:+.4f} | "
                    f"ICIR={metrics['icir']:.3f} | Net_IC={metrics['net_ic']:+.4f} | "
                    f"Sharpe={metrics['sharpe']:.3f}"
                )
                kb.log_result(impl_id, metrics)
                n_ok += 1

    return n_ok, n_err


# ---------------------------------------------------------------------------
# 14 hand-crafted factor code strings
# ---------------------------------------------------------------------------

HAND_CRAFTED = [
    # -------------------------------------------------------------------------
    # 1. Mean-reversion: Bollinger Band z-score (negated for reversal signal)
    # -------------------------------------------------------------------------
    {
        "name": "bollinger_band_z",
        "description": "Negative z-score of price within 21-day Bollinger Bands (overbought = negative score)",
        "rationale": "Prices that deviate far above their 21-day mean tend to revert. Short overbought, long oversold.",
        "mechanism_score": 3,
        "factor_type": "mean_reversion",
        "code": """\
def factor(prices):
    roll_mean = prices.rolling(21, min_periods=21).mean()
    roll_std = prices.rolling(21, min_periods=21).std(ddof=1)
    z = (prices - roll_mean) / roll_std.replace(0, np.nan)
    return -z  # negate: overbought (high z) gets negative score
""",
    },
    # -------------------------------------------------------------------------
    # 2. Mean-reversion: RSI-14 contrarian (inverted)
    # -------------------------------------------------------------------------
    {
        "name": "rsi_14_contrarian",
        "description": "14-day RSI proxy, inverted: oversold (low RSI) = high positive score",
        "rationale": "Oversold conditions as measured by RSI tend to revert upward in the short run.",
        "mechanism_score": 3,
        "factor_type": "mean_reversion",
        "code": """\
def factor(prices):
    delta = prices.diff(1)
    up   = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    down = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    rsi  = 100 - 100 / (1 + up / down.replace(0, np.nan))
    return -rsi  # invert: low RSI (oversold) gets high score
""",
    },
    # -------------------------------------------------------------------------
    # 3. Mean-reversion: negative distance from 252-day MA
    # -------------------------------------------------------------------------
    {
        "name": "ma_distance_252",
        "description": "Negative of (price - 252-day MA) / 252-day MA — far above MA = short signal",
        "rationale": "Long-run mean reversion to 252-day moving average. Stocks far above their annual MA tend to underperform.",
        "mechanism_score": 3,
        "factor_type": "mean_reversion",
        "code": """\
def factor(prices):
    ma252 = prices.rolling(252, min_periods=252).mean()
    return -((prices - ma252) / ma252.replace(0, np.nan))
""",
    },
    # -------------------------------------------------------------------------
    # 4. Mean-reversion: 5-year reversal (De Bondt & Thaler 1985)
    # -------------------------------------------------------------------------
    {
        "name": "five_year_reversal",
        "description": "Negative of 1260-day (5-year) price return — long-run losers outperform long-run winners",
        "rationale": "De Bondt & Thaler (1985): investor overreaction to long-run trends leads to mean reversion over 3-5 year horizons.",
        "mechanism_score": 4,
        "factor_type": "mean_reversion",
        "code": """\
def factor(prices):
    return -prices.pct_change(1260)
""",
    },
    # -------------------------------------------------------------------------
    # 5. Mean-reversion: Williams %R (21-day) — oversold = high score
    # -------------------------------------------------------------------------
    {
        "name": "williams_r_21",
        "description": "Williams %R over 21-day window: (rolling_max - price) / (rolling_max - rolling_min); oversold = high score",
        "rationale": "Williams %R measures the position of price relative to its recent high-low range. High values indicate oversold conditions primed for reversal.",
        "mechanism_score": 3,
        "factor_type": "mean_reversion",
        "code": """\
def factor(prices):
    roll_max = prices.rolling(21, min_periods=21).max()
    roll_min = prices.rolling(21, min_periods=21).min()
    range_   = (roll_max - roll_min).replace(0, np.nan)
    wr = (roll_max - prices) / range_  # 0 = at top (overbought), 1 = at bottom (oversold)
    return wr  # high = oversold = positive mean-reversion signal
""",
    },
    # -------------------------------------------------------------------------
    # 6. Volatility: negative 21-day realized vol (low vol = high quality)
    # -------------------------------------------------------------------------
    {
        "name": "realized_vol_21",
        "description": "Negative 21-day realized volatility of log returns — low volatility stocks score higher",
        "rationale": "Low-volatility anomaly (Ang et al. 2006, Baker et al. 2011): low-vol stocks earn higher risk-adjusted returns due to leverage constraints and lottery preferences.",
        "mechanism_score": 4,
        "factor_type": "volatility",
        "code": """\
def factor(prices):
    log_ret = np.log(prices / prices.shift(1))
    rvol = log_ret.rolling(21, min_periods=21).std(ddof=1)
    return -rvol  # negate: low volatility = high score
""",
    },
    # -------------------------------------------------------------------------
    # 7. Volatility: negative vol-of-vol (stability of volatility)
    # -------------------------------------------------------------------------
    {
        "name": "vol_of_vol_21",
        "description": "Negative rolling std of 21-day realized vol over 126-day window — stable low vol = high score",
        "rationale": "Stocks with consistently low and stable volatility are preferred by risk-averse institutions, leading to relative overperformance.",
        "mechanism_score": 3,
        "factor_type": "volatility",
        "code": """\
def factor(prices):
    log_ret = np.log(prices / prices.shift(1))
    rvol_21 = log_ret.rolling(21, min_periods=21).std(ddof=1)
    vol_of_vol = rvol_21.rolling(126, min_periods=63).std(ddof=1)
    return -vol_of_vol  # negate: stable volatility = high score
""",
    },
    # -------------------------------------------------------------------------
    # 8. Volatility: negative max 1-day return over 21 days (MAX effect)
    # -------------------------------------------------------------------------
    {
        "name": "max_daily_return_21",
        "description": "Negative of maximum 1-day return over past 21 days (MAX effect, Bali et al. 2011)",
        "rationale": "Bali, Cakici & Whitelaw (2011): stocks with extreme positive returns (lottery stocks) are overpriced by retail investors seeking skewness, leading to subsequent underperformance.",
        "mechanism_score": 4,
        "factor_type": "volatility",
        "code": """\
def factor(prices):
    ret = prices.pct_change(1)
    max_ret = ret.rolling(21, min_periods=21).max()
    return -max_ret  # negate: high max return = lottery stock = negative expected return
""",
    },
    # -------------------------------------------------------------------------
    # 9. Quality: return consistency (fraction of up days in past 252 days)
    # -------------------------------------------------------------------------
    {
        "name": "return_consistency_252",
        "description": "Fraction of past 252 trading days with positive daily return — consistent positive momentum",
        "rationale": "Stocks with consistently positive daily returns are fundamentally sound with stable demand; this consistency predicts future performance beyond point-in-time momentum.",
        "mechanism_score": 3,
        "factor_type": "quality",
        "code": """\
def factor(prices):
    daily_ret = prices.pct_change(1)
    up_days = (daily_ret > 0).astype(float)
    return up_days.rolling(252, min_periods=126).mean()
""",
    },
    # -------------------------------------------------------------------------
    # 10. Quality: Calmar proxy (252-day return / max drawdown)
    # -------------------------------------------------------------------------
    {
        "name": "calmar_proxy_252",
        "description": "252-day return divided by absolute max drawdown over 252 days — risk-adjusted momentum",
        "rationale": "The Calmar ratio measures return per unit of drawdown. High Calmar stocks have strong returns with limited downside, indicating quality and trend persistence.",
        "mechanism_score": 3,
        "factor_type": "quality",
        "code": """\
def factor(prices):
    n = 252
    past_ret = prices.pct_change(n)
    roll_max = prices.rolling(n, min_periods=n).max()
    drawdown = (prices - roll_max) / roll_max.replace(0, np.nan)
    max_dd = drawdown.rolling(n, min_periods=n).min()  # most negative value
    max_dd_abs = max_dd.abs().replace(0, np.nan)
    calmar = past_ret / max_dd_abs
    return calmar
""",
    },
    # -------------------------------------------------------------------------
    # 11. Quality: trend linearity over 63 days (R² of log-price on time)
    # -------------------------------------------------------------------------
    {
        "name": "trend_linearity_63",
        "description": "R² of 63-day OLS fit of log-price on linear time index — quality of trend",
        "rationale": "Stocks whose price follows a clean linear trend (high R²) are more likely to continue that trend; noisy, mean-reverting stocks have low R².",
        "mechanism_score": 3,
        "factor_type": "quality",
        "code": """\
def factor(prices):
    n = 63
    t = np.arange(n, dtype=float)
    t_c = t - t.mean()
    t_ss = (t_c ** 2).sum()

    log_p = np.log(prices.clip(lower=1e-10))
    arr = log_p.values.astype(float)
    out = np.full_like(arr, np.nan)

    for j in range(arr.shape[1]):
        col = arr[:, j]
        for i in range(n - 1, len(col)):
            y = col[i - n + 1:i + 1]
            if np.isnan(y).any():
                continue
            y_c = y - y.mean()
            ss_yy = (y_c ** 2).sum()
            if ss_yy > 0:
                out[i, j] = np.dot(t_c, y_c) ** 2 / (t_ss * ss_yy)

    return pd.DataFrame(out, index=prices.index, columns=prices.columns)
""",
    },
    # -------------------------------------------------------------------------
    # 12. Value: distance from 52-week high (52WH effect, George & Hwang 2004)
    # -------------------------------------------------------------------------
    {
        "name": "distance_from_52w_high",
        "description": "1 - (price / 52-week rolling max close) — stocks near 52W high have momentum",
        "rationale": "George & Hwang (2004): distance from 52-week high is a strong predictor of future returns. Stocks near their annual high have positive momentum as investors anchor to this reference point.",
        "mechanism_score": 4,
        "factor_type": "value",
        "code": """\
def factor(prices):
    high_52w = prices.rolling(252, min_periods=126).max()
    dist = 1.0 - prices / high_52w.replace(0, np.nan)
    return -dist  # negate: near 52W high (low dist) = bullish = positive score
""",
    },
    # -------------------------------------------------------------------------
    # 13. Value: 52-week range position ((price - low) / (high - low))
    # -------------------------------------------------------------------------
    {
        "name": "range_position_52w",
        "description": "(price - 52W low) / (52W high - 52W low) — position within annual price range",
        "rationale": "Stocks near the top of their annual range have strong momentum; the range position captures relative strength over the annual horizon.",
        "mechanism_score": 3,
        "factor_type": "value",
        "code": """\
def factor(prices):
    high_52w = prices.rolling(252, min_periods=126).max()
    low_52w  = prices.rolling(252, min_periods=126).min()
    range_   = (high_52w - low_52w).replace(0, np.nan)
    pos = (prices - low_52w) / range_  # 0 = at 52W low, 1 = at 52W high
    return pos  # high = near 52W high = momentum signal
""",
    },
    # -------------------------------------------------------------------------
    # 14. Value: 5-year z-score of log returns
    # -------------------------------------------------------------------------
    {
        "name": "longrun_zscore_1260",
        "description": "Z-score of log return over 1260-day (5-year) rolling window — relative long-run value",
        "rationale": "A stock whose log return is low relative to its own 5-year history is cheap on a mean-reversion basis, while one that is high is a momentum/growth candidate.",
        "mechanism_score": 3,
        "factor_type": "value",
        "code": """\
def factor(prices):
    log_ret = np.log(prices / prices.shift(1))
    roll_mean = log_ret.rolling(1260, min_periods=630).mean()
    roll_std  = log_ret.rolling(1260, min_periods=630).std(ddof=1)
    z = (log_ret - roll_mean) / roll_std.replace(0, np.nan)
    return z  # high z = strong recent return relative to own history = momentum
""",
    },
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Batch computation: 449-stock universe re-run + 14 hand-crafted factors")
    print(f"Parallel workers: {N_WORKERS}")
    print("=" * 70)

    kb = KnowledgeBase(KB_PATH)

    # -----------------------------------------------------------------------
    # Part 1: Re-run all existing implementations on 449-stock universe
    # -----------------------------------------------------------------------
    print("\n=== Part 1: Re-running existing implementations on 449-stock universe ===\n")

    conn = sqlite3.connect(KB_PATH)
    impls = conn.execute("""
        SELECT i.id, i.hypothesis_id, i.code, i.notes, h.factor_name, h.description
        FROM implementations i
        JOIN hypotheses h ON h.id = i.hypothesis_id
        WHERE i.code IS NOT NULL AND i.code != ''
        ORDER BY i.id
    """).fetchall()

    existing_449 = {
        r[0] for r in conn.execute(
            "SELECT implementation_id FROM backtest_results WHERE universe = ?",
            (UNIVERSE,),
        ).fetchall()
    }
    conn.close()

    print(f"  Found {len(impls)} implementations in DB")
    print(f"  Already done: {len(existing_449)}  |  To run: {len(impls) - len(existing_449)}")
    print()

    items_1 = []
    for impl_id, hyp_id, code, notes, factor_name, description in impls:
        if impl_id in existing_449:
            print(f"  [SKIP] impl_{impl_id}")
            continue
        display_name = (factor_name or description or f"impl_{impl_id}")[:50]
        items_1.append((impl_id, display_name, code))

    rerun_ok, rerun_err = run_batch(items_1, kb, label="Part 1")
    print(f"\n  Part 1 done: {rerun_ok} succeeded, {rerun_err} failed")

    # -----------------------------------------------------------------------
    # Part 2: Hand-crafted factors
    # -----------------------------------------------------------------------
    print("\n=== Part 2: Hand-crafted factors (14 total) ===\n")

    items_2 = []
    for spec in HAND_CRAFTED:
        conn = sqlite3.connect(KB_PATH)
        existing = conn.execute(
            "SELECT id FROM hypotheses WHERE factor_name = ?", (spec["name"],)
        ).fetchone()
        conn.close()

        if existing:
            print(f"  [SKIP] {spec['name']} — already in DB (hyp_id={existing[0]})")
            continue

        hyp_id = kb.add_hypothesis(
            factor_name=spec["name"],
            description=spec["description"],
            rationale=spec["rationale"],
            mechanism_score=spec["mechanism_score"],
            status="active",
        )
        impl_id = kb.add_implementation(
            hypothesis_id=hyp_id,
            code=spec["code"],
            notes=f"factor_type={spec['factor_type']};hand_crafted=true",
        )
        items_2.append((impl_id, spec["name"], spec["code"]))

    craft_ok, craft_err = run_batch(items_2, kb, label="Part 2")
    print(f"\n  Part 2 done: {craft_ok} succeeded, {craft_err} failed")

    # -----------------------------------------------------------------------
    # Summary: all 449-stock results with multiple testing correction
    # -----------------------------------------------------------------------
    print("\n=== Summary: All 449-stock universe results ===\n")

    from qframe.factor_harness.multiple_testing import correct_ic_pvalues, print_correction_summary

    all_results = kb.get_all_results()
    results_449 = [r for r in all_results if r.get("universe") == UNIVERSE]
    print(f"Total 449-stock results logged: {len(results_449)}")

    if results_449:
        pass_count = sum(1 for r in results_449 if r.get("passed_gate") == 1)
        print(f"Passed gate (IC>=0.015 AND ICIR>=0.15): {pass_count}/{len(results_449)}")

        # Show top 10 by IC
        print("\nTop 15 by IC (449-stock universe):")
        print(f"  {'impl_id':>8} {'factor_name':<40} {'IC':>7} {'ICIR':>7} {'Net_IC':>8} {'Sharpe':>7} {'Pass':>5}")
        print(f"  {'-'*8} {'-'*40} {'-'*7} {'-'*7} {'-'*8} {'-'*7} {'-'*5}")
        sorted_449 = sorted(results_449, key=lambda r: r.get("ic") or -999, reverse=True)
        for r in sorted_449[:15]:
            fname = r.get("factor_name") or f"impl_{r.get('implementation_id')}"
            ic    = r.get("ic") or 0
            icir  = r.get("icir") or 0
            net_ic = r.get("net_ic") or 0
            sharpe = r.get("sharpe") or 0
            passed = "YES" if r.get("passed_gate") == 1 else "no"
            impl_id = r.get("implementation_id")
            print(f"  {impl_id:>8} {fname:<40} {ic:>+7.4f} {icir:>7.3f} {net_ic:>+8.4f} {sharpe:>7.3f} {passed:>5}")

        # Multiple testing correction
        corrected = correct_ic_pvalues(results_449, alpha=0.05, n_oos_days=1762)
        if not corrected.empty:
            print_correction_summary(corrected)
        else:
            print("\nNo positive-IC results for multiple testing correction.")
    else:
        print("No 449-stock results found — check for errors above.")

    print("\n=== Batch run complete ===")
    total_ok = rerun_ok + craft_ok
    total_err = rerun_err + craft_err
    print(f"  Total successful: {total_ok}, Total failed: {total_err}")


if __name__ == "__main__":
    main()
