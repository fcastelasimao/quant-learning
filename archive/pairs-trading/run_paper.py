"""
Entry point: run the live paper trader with a refreshing dashboard.

Usage:
    python3 run_paper.py

Scans every 5 minutes for new 1h candle closes, updates z-scores, and
opens/closes paper positions. The terminal clears and redraws a live
dashboard after every scan. Press Ctrl+C to stop and see a final summary.

Run the backtest first (python3 run_backtest.py) to confirm the pairs
are cointegrated before committing paper capital.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from config import STRATEGY
from paper_trader import RUNNING, LivePairsTrader
from models import Signal

Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/paper_trader.log"),
        # No StreamHandler — dashboard owns stdout
    ],
)
logger = logging.getLogger("run_paper")

_W = 70  # dashboard width


def _clear():
    os.system("cls" if os.name == "nt" else "clear")


def _bar(pct: float, width: int = 20) -> str:
    """Simple ASCII progress bar for P&L."""
    filled = int(abs(pct) / 20 * width)  # scale: ±20% = full bar
    filled = min(filled, width)
    char = "+" if pct >= 0 else "-"
    return f"[{char * filled}{'.' * (width - filled)}]"


def print_dashboard(trader: LivePairsTrader, scan_count: int, next_scan_in: float):
    _clear()
    port = trader.portfolio
    now = datetime.now(timezone.utc)

    # ── Header ────────────────────────────────────────────────────────────────
    print("=" * _W)
    print("  PAIRS TRADING — LIVE PAPER DASHBOARD")
    print("=" * _W)
    print(f"  Time (UTC):  {now.strftime('%Y-%m-%d %H:%M:%S')}  |  "
          f"Scan #{scan_count}  |  Next scan in {next_scan_in:.0f}s")
    print(f"  Pairs:       {', '.join(f'{b}/{q}' for b,q in STRATEGY.candidate_pairs)}")

    # ── Portfolio ─────────────────────────────────────────────────────────────
    print("-" * _W)
    pnl_sign = "+" if port.total_pnl >= 0 else ""
    pct = port.total_pnl_pct
    print(f"  PORTFOLIO")
    print(f"  Bankroll:    ${port.bankroll:>10.2f}   (start: ${port.initial_bankroll:.2f})")
    print(f"  Total P&L:   {pnl_sign}${port.total_pnl:>9.2f}   ({pnl_sign}{pct:.2f}%)  {_bar(pct)}")

    closed = port.closed_positions
    wins = [p for p in closed if (p.pnl_usd or 0) > 0]
    wr = len(wins) / len(closed) * 100 if closed else 0.0
    print(f"  Trades:      {len(closed)} closed  |  Win rate: {wr:.1f}%  "
          f"({len(wins)}W / {len(closed)-len(wins)}L)")

    # ── Open positions ────────────────────────────────────────────────────────
    print("-" * _W)
    print(f"  OPEN POSITIONS ({len(port.open_positions)})")
    if not port.open_positions:
        print("  None — waiting for entry signal")
    else:
        for pos in port.open_positions:
            holding = (now - pos.entry_time).total_seconds() / 3600
            direction_str = "LONG spread " if pos.direction == Signal.LONG_SPREAD else "SHORT spread"
            print(f"  {pos.base}/{pos.quote:<14}  {direction_str}  "
                  f"z_entry={pos.entry_z_score:+.2f}  holding={holding:.1f}h")
            print(f"  {'':30}entry: base=${pos.entry_price_base:.4f}  "
                  f"quote=${pos.entry_price_quote:.4f}")

    # ── Live z-scores ─────────────────────────────────────────────────────────
    print("-" * _W)
    print(f"  LIVE Z-SCORES")
    if not trader._last_states:
        print("  Fetching first scan...")
    else:
        print(f"  {'Pair':<24} {'Z-score':>8}  {'Signal':<16}  Bar")
        print(f"  {'-'*24} {'-'*8}  {'-'*16}  {'-'*22}")
        for key, state in trader._last_states.items():
            base, quote = key
            z = state.z_score
            sig = state.signal.value
            # Visual z-score bar centred at 0
            bar_width = 20
            centre = bar_width // 2
            pos_bar = int(z / STRATEGY.stop_loss_z_score * centre)
            pos_bar = max(-centre, min(centre, pos_bar))
            bar = ["."] * bar_width
            bar[centre] = "|"
            if pos_bar > 0:
                for i in range(centre + 1, centre + pos_bar + 1):
                    bar[i] = "+"
            elif pos_bar < 0:
                for i in range(centre + pos_bar, centre):
                    bar[i] = "-"
            bar_str = "".join(bar)
            print(f"  {base}/{quote:<14}  {z:>+8.3f}  {sig:<16}  [{bar_str}]")

    # ── Recent closed trades ──────────────────────────────────────────────────
    print("-" * _W)
    print("  RECENT TRADES (last 5)")
    recent = closed[-5:]
    if not recent:
        print("  No trades yet")
    else:
        print(f"  {'Pair':<20} {'Dir':<14} {'Hold':>6}  {'P&L':>8}  Reason")
        print(f"  {'-'*20} {'-'*14} {'-'*6}  {'-'*8}  {'-'*10}")
        for p in reversed(recent):
            hold = (p.exit_time - p.entry_time).total_seconds() / 3600
            pnl_str = f"{'+' if (p.pnl_usd or 0) >= 0 else ''}${p.pnl_usd:.2f}"
            print(f"  {p.base}/{p.quote:<14}  {p.direction.value:<14}  "
                  f"{hold:>5.1f}h  {pnl_str:>8}  {p.exit_reason}")

    # ── Footer ────────────────────────────────────────────────────────────────
    print("=" * _W)
    print("  Press Ctrl+C to stop  |  Full log: logs/paper_trader.log")
    print("=" * _W)


def main():
    trader = LivePairsTrader()

    # Print static header before first scan
    _clear()
    print("=" * _W)
    print("  PAIRS TRADING — LIVE PAPER DASHBOARD")
    print("=" * _W)
    print(f"  Pairs: {', '.join(f'{b}/{q}' for b,q in STRATEGY.candidate_pairs)}")
    print(f"  Interval: {STRATEGY.candle_interval}  |  "
          f"Entry |z|: {STRATEGY.entry_z_score}  |  "
          f"Stop |z|: {STRATEGY.stop_loss_z_score}")
    print(f"  Bankroll: ${STRATEGY.initial_bankroll:.2f}")
    print("=" * _W)
    print("  Initialising — fitting cointegration models and seeding trackers...")
    print("  (This takes ~30 seconds on first run)")

    while RUNNING:
        loop_start = time.monotonic()

        try:
            trader.scan()
        except Exception as e:
            logger.exception(f"Error during scan: {e}")

        elapsed = time.monotonic() - loop_start
        sleep_for = max(0.0, STRATEGY.scan_interval_seconds - elapsed)

        print_dashboard(trader, trader._scan_count, sleep_for)

        logger.info(
            f"Scan #{trader._scan_count} | elapsed={elapsed:.1f}s | "
            f"open={len(trader.portfolio.open_positions)} | "
            f"P&L=${trader.portfolio.total_pnl:+.2f}"
        )

        if RUNNING:
            time.sleep(sleep_for)

    # Final summary on exit
    _clear()
    print_dashboard(trader, trader._scan_count, 0)
    print("\n  Stopped. Final state above. Full trade log: logs/paper_trades.jsonl")
    logger.info("Paper trader stopped.")


if __name__ == "__main__":
    main()
