"""
Paper trader: simulates live pairs trading using real-time 1h candle closes.

Every `scan_interval_seconds` it fetches the latest closed candle for each
pair, updates the spread tracker, and opens/closes two-leg paper positions
based on z-score signals. All trades are logged to JSONL for analysis.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from cointegration import compute_spread, test_cointegration
from config import STRATEGY
from data_fetcher import fetch_candles_cached, fetch_latest_close
from models import CointegrationResult, PairPosition, Signal
from spread_tracker import SpreadTracker

logger = logging.getLogger(__name__)

RUNNING = True


def _signal_handler(sig, frame):
    global RUNNING
    logger.info("Shutdown signal received — finishing current scan then exiting")
    RUNNING = False


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


class PaperPortfolio:
    def __init__(self, bankroll: float = STRATEGY.initial_bankroll):
        self.initial_bankroll = bankroll
        self.bankroll = bankroll
        self.open_positions: list[PairPosition] = []
        self.closed_positions: list[PairPosition] = []
        self._log_path = Path(STRATEGY.log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def total_pnl(self) -> float:
        return sum(p.pnl_usd for p in self.closed_positions if p.pnl_usd is not None)

    @property
    def total_pnl_pct(self) -> float:
        return self.total_pnl / self.initial_bankroll * 100

    def open_position(
        self,
        base: str,
        quote: str,
        direction: Signal,
        price_base: float,
        price_quote: float,
        hedge_ratio: float,
        z_score: float,
    ) -> PairPosition:
        stake = self.bankroll * STRATEGY.position_size_pct
        base_qty = stake / price_base
        quote_qty = base_qty * hedge_ratio * (price_base / price_quote)

        pos = PairPosition(
            base=base,
            quote=quote,
            direction=direction,
            entry_z_score=z_score,
            hedge_ratio=hedge_ratio,
            base_qty=base_qty,
            quote_qty=quote_qty,
            entry_price_base=price_base,
            entry_price_quote=price_quote,
            entry_time=datetime.now(timezone.utc),
        )
        self.open_positions.append(pos)
        self._log_event("open", pos)
        logger.info(
            f"OPEN {direction.value} | {base}/{quote} | "
            f"z={z_score:+.2f} | base_qty={base_qty:.6f} | quote_qty={quote_qty:.6f}"
        )
        return pos

    def close_position(
        self,
        pos: PairPosition,
        price_base: float,
        price_quote: float,
        reason: str,
        z_score: float,
    ) -> float:
        """Close a position and return realised P&L in USD."""
        comm = STRATEGY.commission

        if pos.direction == Signal.LONG_SPREAD:
            # Bought base, sold quote on open
            pnl_base = (price_base - pos.entry_price_base) * pos.base_qty
            pnl_quote = -(price_quote - pos.entry_price_quote) * pos.quote_qty
        else:
            # Sold base, bought quote on open
            pnl_base = -(price_base - pos.entry_price_base) * pos.base_qty
            pnl_quote = (price_quote - pos.entry_price_quote) * pos.quote_qty

        # Commission on all four legs (open base, open quote, close base, close quote)
        total_notional = (
            pos.base_qty * pos.entry_price_base
            + pos.quote_qty * pos.entry_price_quote
            + pos.base_qty * price_base
            + pos.quote_qty * price_quote
        )
        commission_cost = total_notional * comm
        pnl_usd = pnl_base + pnl_quote - commission_cost

        pos.exit_price_base = price_base
        pos.exit_price_quote = price_quote
        pos.exit_time = datetime.now(timezone.utc)
        pos.exit_reason = reason
        pos.pnl_usd = round(pnl_usd, 4)

        self.bankroll += pnl_usd
        self.open_positions.remove(pos)
        self.closed_positions.append(pos)
        self._log_event("close", pos, extra={"exit_reason": reason, "exit_z": z_score})

        sign = "+" if pnl_usd >= 0 else ""
        logger.info(
            f"CLOSE {pos.direction.value} | {pos.base}/{pos.quote} | "
            f"z={z_score:+.2f} | reason={reason} | "
            f"P&L={sign}${pnl_usd:.2f}"
        )
        return pnl_usd

    def _log_event(self, event_type: str, pos: PairPosition, extra: dict = None):
        record = {
            "event": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "base": pos.base,
            "quote": pos.quote,
            "direction": pos.direction.value,
            "entry_z_score": pos.entry_z_score,
            "hedge_ratio": pos.hedge_ratio,
            "entry_price_base": pos.entry_price_base,
            "entry_price_quote": pos.entry_price_quote,
            "entry_time": pos.entry_time.isoformat() if pos.entry_time else None,
            "pnl_usd": pos.pnl_usd,
        }
        if extra:
            record.update(extra)
        with open(self._log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def print_summary(self):
        print("\n" + "=" * 65)
        print("  PAPER PORTFOLIO SUMMARY")
        print("=" * 65)
        print(f"  Bankroll:     ${self.bankroll:.2f}  (start: ${self.initial_bankroll:.2f})")
        sign = "+" if self.total_pnl >= 0 else ""
        print(f"  Total P&L:    {sign}${self.total_pnl:.2f}  ({sign}{self.total_pnl_pct:.2f}%)")
        print(f"  Closed trades: {len(self.closed_positions)}")
        wins = [p for p in self.closed_positions if (p.pnl_usd or 0) > 0]
        print(f"  Win rate:     {len(wins)}/{len(self.closed_positions)}")
        print(f"  Open positions: {len(self.open_positions)}")
        for pos in self.open_positions:
            holding = (datetime.now(timezone.utc) - pos.entry_time).total_seconds() / 3600
            print(f"    {pos.base}/{pos.quote} {pos.direction.value} | "
                  f"z_entry={pos.entry_z_score:+.2f} | holding={holding:.1f}h")
        print("=" * 65)


class LivePairsTrader:
    """
    Orchestrates live paper trading across multiple pairs.

    For each pair:
      - Re-fits the hedge ratio every `refit_interval_scans` scans.
      - Maintains a SpreadTracker for real-time z-score updates.
      - Manages open positions with stop-loss and timeout logic.
    """

    def __init__(self):
        self.portfolio = PaperPortfolio()
        self._trackers: dict[tuple, SpreadTracker] = {}
        self._coints: dict[tuple, CointegrationResult] = {}
        self._last_states: dict[tuple, object] = {}  # latest SpreadState per pair
        self._scan_count = 0
        self._refit_interval = 24  # Refit hedge ratio every 24 scans (24h)

    def _get_or_refit_tracker(
        self, base: str, quote: str
    ) -> Optional[SpreadTracker]:
        key = (base, quote)
        if key not in self._trackers or self._scan_count % self._refit_interval == 0:
            logger.info(f"Fitting cointegration for {base}/{quote}...")
            base_candles = fetch_candles_cached(
                base,
                interval=STRATEGY.candle_interval,
                limit=STRATEGY.lookback_candles,
                cache_dir=STRATEGY.data_cache_path,
                max_cache_age_minutes=55,
            )
            quote_candles = fetch_candles_cached(
                quote,
                interval=STRATEGY.candle_interval,
                limit=STRATEGY.lookback_candles,
                cache_dir=STRATEGY.data_cache_path,
                max_cache_age_minutes=55,
            )
            if not base_candles or not quote_candles:
                return None

            coint = test_cointegration(base_candles, quote_candles, base, quote)
            if coint is None or not coint.is_cointegrated:
                logger.warning(f"{base}/{quote} not cointegrated — no trading")
                self._trackers.pop(key, None)
                return None

            self._coints[key] = coint
            tracker = SpreadTracker(coint)
            # Seed with historical spread
            spread_history = compute_spread(
                base_candles, quote_candles, coint.hedge_ratio, coint.intercept
            )
            tracker.seed(list(spread_history))
            self._trackers[key] = tracker
            logger.info(f"{base}/{quote}: cointegrated (p={coint.p_value}, "
                        f"half_life={coint.half_life_hours}h)")

        return self._trackers.get(key)

    def scan(self):
        self._scan_count += 1

        for base, quote in STRATEGY.candidate_pairs:
            tracker = self._get_or_refit_tracker(base, quote)
            if tracker is None:
                continue

            coint = self._coints[(base, quote)]

            price_base = fetch_latest_close(base, interval=STRATEGY.candle_interval)
            price_quote = fetch_latest_close(quote, interval=STRATEGY.candle_interval)

            if price_base is None or price_quote is None:
                logger.warning(f"Could not fetch latest prices for {base}/{quote}")
                continue

            state = tracker.update(
                timestamp=datetime.now(timezone.utc),
                price_base=price_base,
                price_quote=price_quote,
            )
            if state is None:
                continue

            self._last_states[(base, quote)] = state

            logger.info(
                f"{base}/{quote} | price_base={price_base:.4f} | "
                f"price_quote={price_quote:.4f} | z={state.z_score:+.4f} | "
                f"signal={state.signal.value}"
            )

            # Check open positions for this pair
            open_pos = next(
                (p for p in self.portfolio.open_positions
                 if p.base == base and p.quote == quote),
                None,
            )

            if open_pos:
                holding_hours = (
                    datetime.now(timezone.utc) - open_pos.entry_time
                ).total_seconds() / 3600

                should_exit = (
                    state.signal == Signal.EXIT
                    or holding_hours >= STRATEGY.max_holding_hours
                )
                if should_exit:
                    reason = "timeout" if holding_hours >= STRATEGY.max_holding_hours else "signal"
                    self.portfolio.close_position(
                        open_pos, price_base, price_quote, reason, state.z_score
                    )

            elif (
                state.signal in (Signal.LONG_SPREAD, Signal.SHORT_SPREAD)
                and len(self.portfolio.open_positions) < STRATEGY.max_open_positions
            ):
                self.portfolio.open_position(
                    base=base,
                    quote=quote,
                    direction=state.signal,
                    price_base=price_base,
                    price_quote=price_quote,
                    hedge_ratio=coint.hedge_ratio,
                    z_score=state.z_score,
                )
