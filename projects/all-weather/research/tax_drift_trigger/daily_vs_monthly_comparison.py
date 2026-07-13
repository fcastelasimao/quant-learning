"""
research/tax_drift_trigger/daily_vs_monthly_comparison.py (L.53)
================================================================
Compare daily-resolution and monthly-resolution engines across the K.50
policy ladder (monthly + abs 4–7pp + rel 40%), US/FIFO, OOS 2018/2020/2022.

Kill criterion
--------------
If the top-2 candidates from the monthly research (5.5pp and 7pp) both rank
in the top-3 of the daily engine on at least 2 of the 3 OOS windows, the
monthly engine is a valid proxy and F.26 may proceed.

Artifacts  (results/daily_vs_monthly/<ts>_<strategy>/)
---------------------------------------------------------
* calmar_comparison.csv   — (policy, engine, window, calmar, cagr, mdd, rebalances)
* rebalance_timing.csv    — (policy, engine, rebalance_date)
* summary.json            — kill-criterion verdict + per-window rank tables
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine import config
from engine.backtest import RebalancePolicy
from engine.daily_tax_backtest import run_daily_tax_backtest
from engine.data import fetch_dividends, fetch_prices, get_price_provenance
from engine.lot_ledger import LotSelector
from engine.stats import compute_calmar, compute_cagr, compute_max_drawdown
from engine.tax import TaxRegime
from engine.tax_backtest import run_tax_aware_backtest

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TICKERS = list(config.TARGET_ALLOCATION)
ALLOCATION = config.TARGET_ALLOCATION
OOS_WINDOWS = ("2018", "2020", "2022")

# K.50 candidates under review
TOP_CANDIDATES = {"drift_absolute(0.055)", "drift_absolute(0.07)"}
KILL_CRITERION_WINDOWS = 2   # pass on at least 2 of 3
TOP_N_THRESHOLD = 3          # candidates must rank in top-N of daily engine

DAYS_PER_YEAR = 365.25


# ---------------------------------------------------------------------------
# Policies (K.50 ladder)
# ---------------------------------------------------------------------------

def _policies() -> dict[str, RebalancePolicy]:
    return {
        "monthly_unconditional": RebalancePolicy.monthly_unconditional(),
        "drift_absolute_4pp":    RebalancePolicy.drift_absolute(0.04),
        "drift_absolute_5pp":    RebalancePolicy.drift_absolute(0.05),
        "drift_absolute_5.5pp":  RebalancePolicy.drift_absolute(0.055),
        "drift_absolute_6pp":    RebalancePolicy.drift_absolute(0.06),
        "drift_absolute_6.5pp":  RebalancePolicy.drift_absolute(0.065),
        "drift_absolute_7pp":    RebalancePolicy.drift_absolute(0.07),
        "drift_relative_40pct":  RebalancePolicy.drift_relative(0.40),
    }


# ---------------------------------------------------------------------------
# Calmar helper
# ---------------------------------------------------------------------------

def _window_calmar(monthly: pd.DataFrame, start_year: str) -> dict:
    """Calmar/CAGR/MDD on the Value series from start_year-01-01 onward."""
    start = pd.Timestamp(f"{start_year}-01-01")
    sub = monthly.loc[monthly.index >= start, "Value"].dropna()
    if len(sub) < 2:
        return {"calmar": None, "cagr": None, "mdd": None, "months": len(sub)}
    years = len(sub) / 12
    cagr = compute_cagr(sub, years)
    mdd = compute_max_drawdown(sub)
    return {
        "calmar": round(compute_calmar(cagr, mdd), 4),
        "cagr":   round(cagr, 4),
        "mdd":    round(mdd, 4),
        "months": len(sub),
    }


# ---------------------------------------------------------------------------
# Kill criterion evaluation
# ---------------------------------------------------------------------------

def _evaluate_kill_criterion(
    df: pd.DataFrame,
) -> dict:
    """
    For each OOS window, rank the drift policies by Calmar in the daily engine.
    Pass if TOP_CANDIDATES both appear in the top-N on >= KILL_CRITERION_WINDOWS.

    Returns a dict with ``passed``, per-window rank tables, and details.
    """
    daily_df = df[df["engine"] == "daily"].copy()
    window_results = {}
    windows_passed = 0

    for win in OOS_WINDOWS:
        win_df = daily_df[daily_df["window"] == win].copy()
        # Exclude monthly_unconditional from the drift-policy ranking
        drift_df = win_df[win_df["policy"] != "monthly_unconditional"].copy()
        drift_df = drift_df.dropna(subset=["calmar"])
        drift_df = drift_df.sort_values("calmar", ascending=False).reset_index(drop=True)
        drift_df["rank"] = drift_df.index + 1

        top_n_labels = set(drift_df.head(TOP_N_THRESHOLD)["policy_label"].tolist())
        # Map to canonical labels used in TOP_CANDIDATES
        cand_ranks = {}
        for _, row in drift_df.iterrows():
            lbl = row["policy_label"]
            if lbl in TOP_CANDIDATES:
                cand_ranks[lbl] = int(row["rank"])

        cands_in_top_n = all(
            cand_ranks.get(c, 999) <= TOP_N_THRESHOLD for c in TOP_CANDIDATES
        )
        if cands_in_top_n:
            windows_passed += 1

        window_results[win] = {
            "candidates_in_top_3": cands_in_top_n,
            "candidate_ranks": cand_ranks,
            "ranking": drift_df[["rank", "policy", "policy_label", "calmar"]].to_dict(orient="records"),
        }

    passed = windows_passed >= KILL_CRITERION_WINDOWS
    return {
        "passed": passed,
        "windows_passed": windows_passed,
        "windows_required": KILL_CRITERION_WINDOWS,
        "top_n_threshold": TOP_N_THRESHOLD,
        "top_candidates": list(TOP_CANDIDATES),
        "interpretation": (
            "Monthly research is a valid proxy for the daily live engine — "
            "F.26 may proceed." if passed else
            "Rank order NOT preserved in daily engine — re-evaluate before F.26."
        ),
        "per_window": window_results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    config.DATA_SOURCE = args.data_source
    config.FMP_PRICE_COLUMN = args.fmp_price_column

    start = args.start
    end = args.end or config.BACKTEST_END
    strategy_id = config.DEFAULT_STRATEGY

    print(f"Loading prices: {TICKERS}, {start} → {end} (source={config.DATA_SOURCE}) …")
    try:
        prices = fetch_prices(TICKERS, start, end)
        dividends = fetch_dividends(TICKERS, start, end)
        prov = get_price_provenance(prices)
    except Exception as exc:
        print(f"[ERROR] Could not load price data: {exc}")
        print("Run `quantcore-ingest` to refresh the central data store.")
        sys.exit(1)

    if prices.empty:
        print("[ERROR] Price data is empty. Check data store.")
        sys.exit(1)

    regime = TaxRegime.us()
    selector = LotSelector.FIFO
    policies = _policies()

    calmar_rows: list[dict] = []
    timing_rows: list[dict] = []

    for pol_name, policy in policies.items():
        print(f"  [{pol_name}] monthly engine …", end=" ", flush=True)
        m_res = run_tax_aware_backtest(
            prices, ALLOCATION,
            regime=regime, rebalance_policy=policy,
            lot_selector=selector, dividends=dividends,
            transaction_cost_pct=0.0,
        )
        print("daily engine …", end=" ", flush=True)
        d_res = run_daily_tax_backtest(
            prices, ALLOCATION,
            regime=regime, rebalance_policy=policy,
            lot_selector=selector, dividends=dividends,
            transaction_cost_pct=0.0,
            min_rebalance_days=31,
        )
        print("done")

        for win in OOS_WINDOWS:
            for engine_name, monthly in (("monthly", m_res.monthly), ("daily", d_res.monthly)):
                stats = _window_calmar(monthly, win)
                calmar_rows.append({
                    "policy":       pol_name,
                    "policy_label": policy.label,
                    "engine":       engine_name,
                    "window":       win,
                    "calmar":       stats["calmar"],
                    "cagr":         stats["cagr"],
                    "mdd":          stats["mdd"],
                    "months":       stats["months"],
                    "rebalances": (
                        int(m_res.events["Rebalanced"].sum()) if engine_name == "monthly"
                        else len(d_res.rebalance_dates)
                    ),
                })

        for rd in m_res.events[m_res.events["Rebalanced"]]["Date"].tolist():
            timing_rows.append({"policy": pol_name, "engine": "monthly", "rebalance_date": str(rd.date())})
        for rd in d_res.rebalance_dates:
            timing_rows.append({"policy": pol_name, "engine": "daily", "rebalance_date": str(rd)})

    calmar_df  = pd.DataFrame(calmar_rows)
    timing_df  = pd.DataFrame(timing_rows)
    verdict    = _evaluate_kill_criterion(calmar_df)

    # Write artifacts
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = PROJECT_ROOT / "results" / "daily_vs_monthly" / f"{ts}_{strategy_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    calmar_df.to_csv(out_dir / "calmar_comparison.csv", index=False)
    timing_df.to_csv(out_dir / "rebalance_timing.csv", index=False)

    summary = {
        "run_date":    datetime.now().isoformat(),
        "strategy_id": strategy_id,
        "price_start": start,
        "price_end":   end,
        "regime":      regime.name,
        "selector":    selector.value,
        "oos_windows": list(OOS_WINDOWS),
        "kill_criterion": verdict,
        "price_provenance": prov,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nArtifacts → {out_dir}")
    print(f"\n{'='*60}")
    print(f"KILL CRITERION: {'PASSED ✓' if verdict['passed'] else 'FAILED ✗'}")
    print(f"  {verdict['interpretation']}")
    print(f"  Windows passed: {verdict['windows_passed']} / 3 (need {KILL_CRITERION_WINDOWS})")

    for win, wr in verdict["per_window"].items():
        print(f"\n  OOS {win}: candidates in top-{TOP_N_THRESHOLD}? {'YES' if wr['candidates_in_top_3'] else 'NO'}")
        print(f"    ranks: {wr['candidate_ranks']}")
        top3 = [r for r in wr["ranking"] if r["rank"] <= TOP_N_THRESHOLD]
        for r in top3:
            print(f"      #{r['rank']} {r['policy_label']:30s}  Calmar {r['calmar']}")

    # Print summary table
    print(f"\n{'='*60}")
    print("Calmar by policy × engine (OOS 2018, FIFO, US tax):\n")
    pivot = calmar_df[calmar_df["window"] == "2018"].pivot(
        index="policy_label", columns="engine", values="calmar"
    ).round(4)
    print(pivot.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily vs monthly engine comparison (L.53)")
    parser.add_argument("--start", default="2006-01-01", help="Price history start")
    parser.add_argument("--end",   default=None,         help="Price history end (default: today)")
    parser.add_argument("--data-source",      choices=("yfinance", "fmp"), default="fmp")
    parser.add_argument("--fmp-price-column", default="adj_close")
    main(parser.parse_args())
