"""
Offline log analyser for the pairs paper trader.

Reads logs/paper_trades.jsonl and prints a full performance report:
  - Trade-by-trade breakdown
  - Per-pair performance
  - Running equity curve (ASCII)
  - Win/loss streaks
  - Holding time distribution

Usage:
    python3 analyse.py
    python3 analyse.py --pair RENDERUSDT/FETUSDT   # filter one pair
    python3 analyse.py --since 2026-04-01          # trades from date onwards
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_LOG_PATH = Path("logs/paper_trades.jsonl")
_W = 72


def _load_trades(pair_filter: str | None, since: datetime | None) -> list[dict]:
    if not _LOG_PATH.exists():
        print(f"No trade log found at {_LOG_PATH}. Run python3 run_paper.py first.")
        sys.exit(0)

    trades = []
    with open(_LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event") != "close":
                continue
            if pair_filter:
                label = f"{rec['base']}/{rec['quote']}"
                if label != pair_filter:
                    continue
            if since:
                ts = datetime.fromisoformat(rec["timestamp"])
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts < since:
                    continue
            trades.append(rec)

    return trades


def _equity_curve(trades: list[dict], initial: float = 1000.0) -> list[float]:
    equity = initial
    curve = [equity]
    for t in trades:
        equity += t.get("pnl_usd", 0) or 0
        curve.append(equity)
    return curve


def _ascii_curve(curve: list[float], width: int = 50, height: int = 10) -> str:
    if len(curve) < 2:
        return "  (not enough data)"
    lo, hi = min(curve), max(curve)
    span = hi - lo or 1
    rows = []
    for row in range(height, -1, -1):
        threshold = lo + (row / height) * span
        line = ""
        for val in curve:
            line += "*" if val >= threshold else " "
        label = f"${threshold:>8.2f} |"
        if row == height:
            label = f"${hi:>8.2f} |"
        elif row == 0:
            label = f"${lo:>8.2f} |"
        elif row == height // 2:
            label = f"${(hi+lo)/2:>8.2f} |"
        else:
            label = f"{'':>9}|"
        rows.append(f"  {label}{line}")
    rows.append(f"  {'':>9}+" + "-" * len(curve))
    rows.append(f"  {'':>9} Trade 0{' '*(len(curve)-10)}Trade {len(curve)-1}")
    return "\n".join(rows)


def _max_streak(wins: list[bool]) -> tuple[int, int]:
    """Return (max_win_streak, max_loss_streak)."""
    max_w = max_l = cur_w = cur_l = 0
    for w in wins:
        if w:
            cur_w += 1
            cur_l = 0
        else:
            cur_l += 1
            cur_w = 0
        max_w = max(max_w, cur_w)
        max_l = max(max_l, cur_l)
    return max_w, max_l


def _sharpe(pnls: list[float]) -> float:
    if len(pnls) < 2:
        return 0.0
    arr = np.array(pnls)
    std = np.std(arr, ddof=1)
    return float(np.mean(arr) / std * np.sqrt(252)) if std > 0 else 0.0


def print_report(trades: list[dict], pair_filter: str | None, since: datetime | None):
    print("=" * _W)
    print("  PAIRS TRADING — PERFORMANCE REPORT")
    print("=" * _W)
    if pair_filter:
        print(f"  Pair filter:  {pair_filter}")
    if since:
        print(f"  Since:        {since.strftime('%Y-%m-%d')}")
    print(f"  Trades loaded: {len(trades)}")

    if not trades:
        print("\n  No closed trades found matching the filters.")
        return

    initial = 1000.0
    curve = _equity_curve(trades, initial)
    final = curve[-1]
    total_pnl = final - initial
    total_pnl_pct = total_pnl / initial * 100

    pnls = [t.get("pnl_usd", 0) or 0 for t in trades]
    pnl_pcts = [p / initial * 100 for p in pnls]
    wins = [p > 0 for p in pnls]
    n_wins = sum(wins)
    n_loss = len(trades) - n_wins
    wr = n_wins / len(trades) * 100

    avg_win = np.mean([p for p in pnls if p > 0]) if n_wins else 0.0
    avg_loss = np.mean([p for p in pnls if p <= 0]) if n_loss else 0.0
    profit_factor = abs(avg_win * n_wins / (avg_loss * n_loss)) if n_loss and avg_loss else float("inf")

    holding_hours = []
    for t in trades:
        try:
            entry = datetime.fromisoformat(t["entry_time"])
            exit_ = datetime.fromisoformat(t["timestamp"])
            holding_hours.append((exit_ - entry).total_seconds() / 3600)
        except Exception:
            pass

    max_dd = 0.0
    peak = initial
    for v in curve:
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100
        if dd > max_dd:
            max_dd = dd

    max_win_streak, max_loss_streak = _max_streak(wins)

    # ── Summary block ─────────────────────────────────────────────────────────
    sign = "+" if total_pnl >= 0 else ""
    print(f"\n  OVERALL PERFORMANCE")
    print(f"  {'-'*40}")
    print(f"  Final bankroll:    ${final:>10.2f}  (start: ${initial:.2f})")
    print(f"  Total P&L:         {sign}${total_pnl:>9.2f}  ({sign}{total_pnl_pct:.2f}%)")
    print(f"  Win rate:          {wr:.1f}%  ({n_wins}W / {n_loss}L)")
    print(f"  Avg winning trade: +${avg_win:.2f}")
    print(f"  Avg losing trade:  -${abs(avg_loss):.2f}")
    print(f"  Profit factor:     {profit_factor:.2f}x")
    print(f"  Sharpe ratio:      {_sharpe(pnl_pcts):.2f}  (annualised)")
    print(f"  Max drawdown:      {max_dd:.2f}%")
    print(f"  Max win streak:    {max_win_streak}")
    print(f"  Max loss streak:   {max_loss_streak}")
    if holding_hours:
        print(f"  Avg holding time:  {np.mean(holding_hours):.1f}h  "
              f"(min {min(holding_hours):.1f}h / max {max(holding_hours):.1f}h)")

    # ── Per-pair breakdown ────────────────────────────────────────────────────
    pairs = sorted({f"{t['base']}/{t['quote']}" for t in trades})
    if len(pairs) > 1:
        print(f"\n  PER-PAIR BREAKDOWN")
        print(f"  {'-'*60}")
        print(f"  {'Pair':<24} {'Trades':>6}  {'WR%':>5}  {'P&L $':>8}  {'P&L%':>7}  Sharpe")
        print(f"  {'-'*24} {'-'*6}  {'-'*5}  {'-'*8}  {'-'*7}  {'-'*6}")
        for pair in pairs:
            pt = [t for t in trades if f"{t['base']}/{t['quote']}" == pair]
            pp = [t.get("pnl_usd", 0) or 0 for t in pt]
            pw = sum(1 for p in pp if p > 0)
            pp_pct = [p / initial * 100 for p in pp]
            p_total = sum(pp)
            p_total_pct = p_total / initial * 100
            p_sign = "+" if p_total >= 0 else ""
            print(f"  {pair:<24} {len(pt):>6}  {pw/len(pt)*100:>4.1f}%  "
                  f"{p_sign}${p_total:>7.2f}  {p_sign}{p_total_pct:>6.2f}%  "
                  f"{_sharpe(pp_pct):>6.2f}")

    # ── Exit reason breakdown ─────────────────────────────────────────────────
    reasons = {}
    for t in trades:
        r = t.get("exit_reason", "unknown")
        reasons[r] = reasons.get(r, 0) + 1
    print(f"\n  EXIT REASONS")
    print(f"  {'-'*30}")
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        pct_r = count / len(trades) * 100
        print(f"  {reason:<15} {count:>4} trades  ({pct_r:.1f}%)")

    # ── Trade log ────────────────────────────────────────────────────────────
    print(f"\n  TRADE LOG")
    print(f"  {'-'*_W}")
    print(f"  {'#':>3}  {'Pair':<20} {'Dir':<14} {'Entry time':<18} "
          f"{'Hold':>5}  {'P&L':>8}  Reason")
    print(f"  {'---':>3}  {'-'*20} {'-'*14} {'-'*18} "
          f"{'-----':>5}  {'--------':>8}  {'-'*10}")
    for i, t in enumerate(trades, 1):
        pair = f"{t['base']}/{t['quote']}"
        pnl = t.get("pnl_usd", 0) or 0
        pnl_str = f"{'+' if pnl>=0 else ''}${pnl:.2f}"
        try:
            entry_ts = datetime.fromisoformat(t["entry_time"]).strftime("%Y-%m-%d %H:%M")
            exit_ts = datetime.fromisoformat(t["timestamp"])
            entry_dt = datetime.fromisoformat(t["entry_time"])
            hold = (exit_ts - entry_dt).total_seconds() / 3600
            hold_str = f"{hold:.1f}h"
        except Exception:
            entry_ts = "?"
            hold_str = "?"
        print(f"  {i:>3}  {pair:<20} {t.get('direction','?'):<14} "
              f"{entry_ts:<18} {hold_str:>5}  {pnl_str:>8}  {t.get('exit_reason','?')}")

    # ── Equity curve ─────────────────────────────────────────────────────────
    print(f"\n  EQUITY CURVE  (each * = one closed trade)")
    print(f"  {'-'*_W}")
    print(_ascii_curve(curve))
    print("=" * _W)


def main():
    parser = argparse.ArgumentParser(description="Pairs trading log analyser")
    parser.add_argument("--pair", type=str, default=None,
                        help="Filter by pair, e.g. RENDERUSDT/FETUSDT")
    parser.add_argument("--since", type=str, default=None,
                        help="Only show trades on or after this date (YYYY-MM-DD)")
    args = parser.parse_args()

    since_dt = None
    if args.since:
        try:
            since_dt = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            print(f"Invalid date format: {args.since}. Use YYYY-MM-DD.")
            sys.exit(1)

    trades = _load_trades(args.pair, since_dt)
    print_report(trades, args.pair, since_dt)


if __name__ == "__main__":
    main()
