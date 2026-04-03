"""
Post-Session Analysis: load trade logs and generate performance metrics.
Adapted for Crypto CEX arbitrage trades.

Run: python analyse.py
"""
from __future__ import annotations
import json
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict

from config import STRATEGY


def load_trades(path: str) -> list[dict]:
    trades = []
    p = Path(path)
    if not p.exists():
        print(f"No trade log found at {path}")
        return trades
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    trades.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return trades


def analyse():
    trades = load_trades(STRATEGY.trades_log_path)
    if not trades:
        print("No trades to analyse. Run the scanner or simulation first.")
        return

    # Deduplicate (keep latest version of each trade ID)
    by_id = {}
    for t in trades:
        by_id[t["id"]] = t
    trades = list(by_id.values())

    resolved = [t for t in trades if t.get("status") != "open"]
    open_trades = [t for t in trades if t.get("status") == "open"]

    quote_currency = STRATEGY.quote_currency
    total_profit = sum(t.get("actual_profit_quote", 0) for t in resolved)
    profits = [t.get("actual_profit_quote", 0) for t in resolved]

    print("=" * 70)
    print("  CRYPTO CEX ARBITRAGE — TRADE ANALYSIS REPORT")
    print("=" * 70)
    print(f"  Total trades logged: {len(trades)}")
    print(f"  Resolved: {len(resolved)}")
    print(f"  Open: {len(open_trades)}")
    print()

    if resolved:
        winners = [p for p in profits if p > 0]
        losers = [p for p in profits if p <= 0]

        print(f"  PERFORMANCE")
        print(f"  Total P&L: {total_profit:.2f} {quote_currency}")
        print(f"  Avg profit per trade: {total_profit / len(resolved):.2f} {quote_currency}")
        print(f"  Winners: {len(winners)}  Losers: {len(losers)}")
        print(f"  Win rate: {len(winners) / len(resolved) * 100:.1f}%")
        if winners:
            print(f"  Avg win: {sum(winners) / len(winners):.2f} {quote_currency}")
            print(f"  Best trade: {max(winners):.2f} {quote_currency}")
        if losers:
            print(f"  Avg loss: {sum(losers) / len(losers):.2f} {quote_currency}")
            print(f"  Worst trade: {min(losers):.2f} {quote_currency}")
        print()

        # Equity curve
        print(f"  EQUITY CURVE")
        running = STRATEGY.initial_bankroll
        peak = running
        max_dd = 0
        for p in profits:
            running += p
            if running > peak:
                peak = running
            dd = (peak - running) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
        print(f"  Final bankroll: {running:.2f} {quote_currency}")
        print(f"  Peak bankroll: {peak:.2f} {quote_currency}")
        print(f"  Max drawdown: {max_dd * 100:.1f}%")
        print()

        # By trading pair
        print(f"  BY PAIR")
        by_pair = defaultdict(list)
        for t in resolved:
            by_pair[t.get("pair", "Unknown")].append(t.get("actual_profit_quote", 0))
        for pair, pnls in sorted(by_pair.items(), key=lambda x: -sum(x[1])):
            win_count = len([p for p in pnls if p > 0])
            print(
                f"  {pair:<12}: {len(pnls):3d} trades, "
                f"{sum(pnls):>8.2f} {quote_currency} total, "
                f"{sum(pnls) / len(pnls):>7.2f} {quote_currency} avg, "
                f"{win_count}/{len(pnls)} wins"
            )
        print()

        # By direction (which exchange bought vs sold)
        print(f"  BY DIRECTION")
        by_dir = defaultdict(list)
        for t in resolved:
            buy_ex = t.get("buy_exchange", "?")
            sell_ex = t.get("sell_exchange", "?")
            key = f"{buy_ex} → {sell_ex}"
            by_dir[key].append(t.get("actual_profit_quote", 0))
        for direction, pnls in sorted(by_dir.items(), key=lambda x: -sum(x[1])):
            win_count = len([p for p in pnls if p > 0])
            print(
                f"  {direction:<20}: {len(pnls):3d} trades, "
                f"{sum(pnls):>8.2f} {quote_currency} total, "
                f"{win_count}/{len(pnls)} wins"
            )
        print()

    print("=" * 70)


if __name__ == "__main__":
    analyse()
