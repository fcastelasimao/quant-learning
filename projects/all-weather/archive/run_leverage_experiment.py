"""
run_leverage_experiment.py
==========================
Test the effect of bond leverage on the 6-asset production portfolio.

How institutional RP works: bonds have low vol (~7%) vs equities (~16%).
RP overweights bonds to equalise risk, but this caps returns. The fix is
to lever up bond positions so they contribute equal risk AND equal return.

Implementation: we scale bond returns by a leverage factor (1.0x to 2.5x).
At each rebalance, the portfolio holds:
  - Unleveraged assets at RP weights
  - Bond assets at RP weight * leverage (returns multiplied by factor)
  - Financing cost deducted (risk-free rate * excess notional)

This is equivalent to holding bond futures or using margin on TLT/TIP.

Run with:
    conda run -n allweather python3 run_leverage_experiment.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import copy
import numpy as np
import pandas as pd

import config
from data import fetch_prices
from backtest import (compute_cagr, compute_max_drawdown, compute_sharpe,
                      compute_calmar, compute_ulcer_index, compute_sortino)


# ---- Config ----

ALLOCATION = {
    "SPY": 0.134, "QQQ": 0.103, "TLT": 0.175,
    "TIP": 0.348, "GLD": 0.142, "GSG": 0.098,
}
BOND_TICKERS = ["TLT", "TIP"]
LEVERAGE_LEVELS = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
FINANCING_RATE = 0.035  # annual cost of leverage (approx risk-free rate)
TRANSACTION_COST_PCT = 0.001

SPLITS = [
    {"oos_start": "2018-01-01", "label": "2018 OOS"},
    {"oos_start": "2020-01-01", "label": "2020 OOS"},
    {"oos_start": "2022-01-01", "label": "2022 OOS"},
]


def run_leveraged_backtest(
    prices: pd.DataFrame,
    allocation: dict[str, float],
    bond_tickers: list[str],
    leverage: float,
    financing_rate: float,
    txn_cost: float,
    start_date: str,
    end_date: str,
    portfolio_value: float = 10_000.0,
) -> dict:
    """
    Simulate a leveraged RP portfolio.

    Bond positions get leverage * their RP weight in notional exposure.
    Excess notional is financed at `financing_rate`.
    Monthly rebalancing with transaction costs.
    """
    tickers = list(allocation.keys())
    monthly = prices[tickers].resample("ME").last().dropna()
    monthly = monthly[(monthly.index >= start_date) & (monthly.index < end_date)]

    if monthly.empty:
        raise ValueError(f"No data in range {start_date} to {end_date}")

    # Compute leveraged allocation
    # Bond weights are scaled; total notional > 1.0
    lev_alloc = {}
    for t, w in allocation.items():
        if t in bond_tickers:
            lev_alloc[t] = w * leverage
        else:
            lev_alloc[t] = w

    total_notional = sum(lev_alloc.values())
    excess_notional = total_notional - 1.0  # this is what we borrow
    monthly_financing = (1 + financing_rate) ** (1/12) - 1

    # Simulate
    value = portfolio_value
    values = [value]
    dates = [monthly.index[0]]

    for i in range(1, len(monthly)):
        prev_row = monthly.iloc[i - 1]
        curr_row = monthly.iloc[i]

        # Monthly return for each asset
        port_return = 0.0
        trade_turnover = 0.0
        for t, w_lev in lev_alloc.items():
            asset_ret = float(curr_row[t]) / float(prev_row[t]) - 1.0
            port_return += w_lev * asset_ret

        # Subtract financing cost on borrowed notional
        if excess_notional > 0:
            port_return -= excess_notional * monthly_financing

        # Apply return
        new_value = value * (1 + port_return)

        # Transaction costs (approximate: proportional to turnover from rebalance)
        # At each rebalance, drift from target ≈ spread of returns * weight
        # Simplified: charge txn_cost on total notional each month
        new_value -= value * total_notional * txn_cost * 0.5  # ~half turnover

        value = new_value
        values.append(value)
        dates.append(monthly.index[i])

    series = pd.Series(values, index=dates)
    years = len(series) / 12

    # Compute monthly returns for Sharpe/Sortino
    monthly_rets = series.pct_change().dropna() * 100

    return {
        "leverage": leverage,
        "total_notional": round(total_notional, 2),
        "excess_notional": round(excess_notional, 2),
        "cagr": round(compute_cagr(series, years), 2),
        "max_dd": round(compute_max_drawdown(series), 2),
        "calmar": round(compute_calmar(
            compute_cagr(series, years),
            compute_max_drawdown(series)), 3),
        "ulcer": round(compute_ulcer_index(series), 2),
        "sharpe": round(compute_sharpe(monthly_rets, config.RISK_FREE_RATE), 3),
        "sortino": round(compute_sortino(monthly_rets, config.RISK_FREE_RATE), 3),
        "final_value": round(values[-1], 2),
    }


def main() -> None:
    import csv
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "leverage_experiment.csv")

    print("=" * 90)
    print("LEVERAGE EXPERIMENT — Bond leverage on 6-asset production RP")
    print("=" * 90)
    print(f"Bonds leveraged: {', '.join(BOND_TICKERS)}")
    print(f"Financing rate: {FINANCING_RATE:.1%}")
    print(f"Leverage levels: {LEVERAGE_LEVELS}")

    tickers = list(ALLOCATION.keys())
    prices = fetch_prices(tickers, config.BACKTEST_START, config.BACKTEST_END)

    all_rows: list[dict] = []

    for split in SPLITS:
        oos_start = split["oos_start"]
        label = split["label"]

        print(f"\n{'=' * 90}")
        print(f"  {label} (from {oos_start})")
        print(f"{'=' * 90}")
        print(f"  {'Lev':>5} {'Notional':>9} {'CAGR':>7} {'MaxDD':>7} "
              f"{'Calmar':>7} {'Ulcer':>6} {'Sortino':>8} {'Final$':>9}")
        print(f"  {'-'*5} {'-'*9} {'-'*7} {'-'*7} {'-'*7} {'-'*6} {'-'*8} {'-'*9}")

        for lev in LEVERAGE_LEVELS:
            r = run_leveraged_backtest(
                prices, ALLOCATION, BOND_TICKERS, lev,
                FINANCING_RATE, TRANSACTION_COST_PCT,
                oos_start, config.BACKTEST_END,
            )
            r["split"] = label
            r["oos_start"] = oos_start
            all_rows.append(r)

            marker = " <-- ALLW-like" if abs(lev - 2.0) < 0.01 else ""
            print(f"  {lev:>5.2f} {r['total_notional']:>8.2f}x "
                  f"{r['cagr']:>6.2f}% {r['max_dd']:>6.2f}% "
                  f"{r['calmar']:>7.3f} {r['ulcer']:>6.2f} "
                  f"{r['sortino']:>8.3f} {r['final_value']:>9.2f}{marker}")

    # Save to CSV
    fieldnames = [
        "split", "oos_start", "leverage", "total_notional", "excess_notional",
        "cagr", "max_dd", "calmar", "ulcer", "sharpe", "sortino", "final_value",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nResults saved to {csv_path} ({len(all_rows)} rows)")

    # ALLW reference
    print(f"\n{'=' * 90}")
    print("REFERENCE — Bridgewater ALLW ETF (Mar 2025 – present)")
    print("  CAGR ~17.2%, MaxDD ~-8.8%, Calmar ~1.96")
    print("  Note: ALLW uses ~2x bond leverage + active management + alpha")
    print(f"{'=' * 90}")

    print("\nInterpretation:")
    print("  1.0x = your current unleveraged portfolio")
    print("  2.0x = ALLW-equivalent leverage on bonds")
    print("  Look for the leverage level that maximises Calmar, not CAGR")
    print("  Financing costs eat into returns — there's a sweet spot")


if __name__ == "__main__":
    main()
