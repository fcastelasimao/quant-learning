"""
run_rebalance_mode_comparison.py
================================
Compares three rebalancing modes over the full backtest period:

  1. Monthly      — rebalance all assets every month (no threshold)
  2. per_asset    — rebalance only assets that individually drift > threshold
  3. full_on_breach — if any asset drifts > threshold, rebalance ALL assets

Key question: does full_on_breach (which redistributes the sell proceeds
immediately) produce better risk-adjusted returns than per_asset (where
proceeds sit as uninvested cash until another asset also breaches)?

Run with:
    conda run -n allweather python3 run_rebalance_mode_comparison.py

Optional arguments:
    --threshold FLOAT   drift threshold (default: 0.05)
    --start YYYY-MM-DD  backtest start (default: config.BACKTEST_START)
    --end   YYYY-MM-DD  backtest end   (default: today)
    --cost  FLOAT       transaction cost per trade, e.g. 0.001 (default: 0)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from engine import config
from engine.data import fetch_prices
from engine.stats import compute_cagr, compute_max_drawdown, compute_sharpe, compute_calmar


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

@dataclass
class ModeResult:
    name: str
    equity: pd.Series       # monthly portfolio value
    rebalance_months: int   # months in which at least one trade occurred
    total_turnover: float   # total $ value traded (buys + sells)
    cash_series: pd.Series  # uninvested cash each month (useful for per_asset)


def _simulate(
    monthly: pd.DataFrame,
    allocation: dict[str, float],
    start_value: float,
    drift_threshold: float,
    rebalance_mode: str,
    transaction_cost_pct: float,
) -> ModeResult:
    """
    Simulate one rebalancing strategy. Returns a ModeResult.

    rebalance_mode:
        "monthly"       — always rebalance all assets (threshold ignored)
        "per_asset"     — rebalance only assets whose drift exceeds threshold;
                          proceeds from sells sit as uninvested cash
        "full_on_breach" — if any asset drifts > threshold, rebalance all;
                           otherwise no trades
    """
    tickers = list(allocation.keys())
    shares = {t: (start_value * w) / float(monthly.iloc[0][t])
              for t, w in allocation.items()}
    cash = 0.0

    equity_vals: list[tuple] = []
    cash_vals: list[tuple] = []
    rebalance_months = 0
    total_turnover = 0.0

    for date, row in monthly.iterrows():
        prices = {t: float(row[t]) for t in tickers}
        invested = sum(shares[t] * prices[t] for t in tickers)
        total = invested + cash

        current_weights = {t: shares[t] * prices[t] / total for t in tickers}

        # --- Determine which assets to trade ---
        if rebalance_mode == "monthly":
            to_rebalance = set(tickers)
        elif rebalance_mode == "full_on_breach":
            any_breach = any(
                abs(current_weights[t] - allocation[t]) > drift_threshold
                for t in tickers
            )
            to_rebalance = set(tickers) if any_breach else set()
        else:  # per_asset
            to_rebalance = {
                t for t in tickers
                if abs(current_weights[t] - allocation[t]) > drift_threshold
            }

        # --- Execute trades ---
        if to_rebalance:
            rebalance_months += 1

            # Sells first (overweight assets in to_rebalance)
            for t in to_rebalance:
                target_val = total * allocation[t]
                current_val = shares[t] * prices[t]
                if current_val > target_val:
                    sell_val = current_val - target_val
                    cost = sell_val * transaction_cost_pct
                    shares[t] -= sell_val / prices[t]
                    cash += sell_val - cost
                    total_turnover += sell_val

            # Refresh total after sells
            invested = sum(shares[t] * prices[t] for t in tickers)
            total = invested + cash

            # Buys (underweight assets in to_rebalance)
            for t in to_rebalance:
                target_val = total * allocation[t]
                current_val = shares[t] * prices[t]
                if current_val < target_val:
                    buy_val = min(target_val - current_val, cash)
                    if buy_val > 0:
                        cost = buy_val * transaction_cost_pct
                        shares[t] += (buy_val - cost) / prices[t]
                        cash -= buy_val
                        total_turnover += buy_val

        invested = sum(shares[t] * prices[t] for t in tickers)
        total = invested + cash
        equity_vals.append((date, total))
        cash_vals.append((date, cash))

    equity = pd.Series(dict(equity_vals))
    cash_series = pd.Series(dict(cash_vals))
    return ModeResult(
        name=rebalance_mode,
        equity=equity,
        rebalance_months=rebalance_months,
        total_turnover=total_turnover,
        cash_series=cash_series,
    )


# ---------------------------------------------------------------------------
# Stats + display
# ---------------------------------------------------------------------------

def _stats_row(result: ModeResult, rf: float) -> dict:
    eq = result.equity
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    monthly_ret = eq.pct_change().dropna() * 100

    cagr = compute_cagr(eq, years)
    mdd  = compute_max_drawdown(eq)
    calmar = compute_calmar(cagr, mdd)
    sharpe = compute_sharpe(monthly_ret, rf_annual=rf)

    avg_cash_pct = (result.cash_series / eq * 100).mean()

    return {
        "Mode":              result.name,
        "CAGR %":            round(cagr, 2),
        "Max DD %":          round(mdd, 2),
        "Calmar":            round(calmar, 3),
        "Sharpe":            round(sharpe, 3),
        "Rebal months":      result.rebalance_months,
        "Turnover $k":       round(result.total_turnover / 1_000, 1),
        "Avg cash %":        round(avg_cash_pct, 2),
    }


def _print_comparison(rows: list[dict]) -> None:
    df = pd.DataFrame(rows).set_index("Mode")
    col_w = max(len(c) for c in df.columns) + 2
    idx_w = max(len(str(i)) for i in df.index) + 2

    header = f"{'Mode':<{idx_w}}" + "".join(f"{c:>{col_w}}" for c in df.columns)
    print("\n" + "=" * len(header))
    print("REBALANCING MODE COMPARISON")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for mode, row in df.iterrows():
        line = f"{mode:<{idx_w}}" + "".join(f"{v:>{col_w}}" for v in row)
        print(line)
    print("=" * len(header))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare per_asset vs full_on_breach rebalancing modes."
    )
    parser.add_argument("--threshold", type=float, default=config.REBALANCE_THRESHOLD,
                        help=f"Drift threshold (default: {config.REBALANCE_THRESHOLD})")
    parser.add_argument("--start", default=config.BACKTEST_START,
                        help=f"Backtest start date (default: {config.BACKTEST_START})")
    parser.add_argument("--end", default=config.BACKTEST_END,
                        help=f"Backtest end date (default: today)")
    parser.add_argument("--cost", type=float, default=0.0,
                        help="Transaction cost per trade fraction (default: 0.0)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    allocation = dict(config.TARGET_ALLOCATION)
    tickers = list(allocation.keys())

    print(f"Fetching prices for {tickers} ({args.start} → {args.end})...")
    all_tickers = list(dict.fromkeys(tickers + [config.BENCHMARK_TICKER]))
    prices = fetch_prices(all_tickers, args.start, args.end)
    port_prices = prices[tickers]
    port_prices = port_prices[
        (port_prices.index >= args.start) & (port_prices.index < args.end)
    ]

    monthly = port_prices.resample(config.DATA_FREQUENCY).last().dropna()
    print(f"Period: {monthly.index[0].date()} → {monthly.index[-1].date()} "
          f"({len(monthly)} months)")
    print(f"Threshold: {args.threshold:.1%}  |  Transaction cost: {args.cost:.3%}\n")

    modes = ["monthly", "per_asset", "full_on_breach"]
    rows = []
    for mode in modes:
        result = _simulate(
            monthly=monthly,
            allocation=allocation,
            start_value=config.INITIAL_PORTFOLIO_VALUE,
            drift_threshold=args.threshold,
            rebalance_mode=mode,
            transaction_cost_pct=args.cost,
        )
        rows.append(_stats_row(result, rf=config.RISK_FREE_RATE))

    _print_comparison(rows)

    print("\nNotes:")
    print("  monthly      — baseline: rebalances every month regardless of drift")
    print("  per_asset    — only trades assets that individually exceed the threshold;")
    print("                 sell proceeds sit as uninvested cash until a buy is triggered")
    print("  full_on_breach — if any asset breaches, all are brought back to target")
    print("  Avg cash %   — mean uninvested cash as % of portfolio (cash drag indicator)")
    print("  Turnover $k  — total $ volume traded over the full period (cost proxy)\n")


if __name__ == "__main__":
    main()
