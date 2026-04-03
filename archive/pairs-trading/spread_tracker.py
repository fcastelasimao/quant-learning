"""
Real-time spread tracker and signal generator.

Takes a fitted CointegrationResult and a stream of new price observations
and emits SpreadState objects (including the current z-score and signal).

This is used both by the backtester (feeding historical candles) and by
the paper trader (feeding live candles one at a time).
"""
from __future__ import annotations

import logging
from collections import deque
from datetime import datetime

import numpy as np

from config import STRATEGY
from models import CointegrationResult, Signal, SpreadState

logger = logging.getLogger(__name__)


class SpreadTracker:
    """
    Maintains a rolling window of spread values and computes z-scores.

    Usage:
        tracker = SpreadTracker(coint_result)
        for candle_base, candle_quote in zip(base_candles, quote_candles):
            state = tracker.update(
                timestamp=candle_base.timestamp,
                price_base=candle_base.close,
                price_quote=candle_quote.close,
            )
            if state and state.signal != Signal.NONE:
                # act on signal
    """

    def __init__(
        self,
        coint: CointegrationResult,
        window: int | None = None,
        entry_z: float | None = None,
        exit_z: float | None = None,
        stop_z: float | None = None,
    ):
        self._coint = coint
        self._window = window or STRATEGY.zscore_window
        self._entry_z = entry_z or STRATEGY.entry_z_score
        self._exit_z = exit_z or STRATEGY.exit_z_score
        self._stop_z = stop_z or STRATEGY.stop_loss_z_score
        self._spread_buffer: deque[float] = deque(maxlen=self._window)

    def _compute_spread(self, price_base: float, price_quote: float) -> float:
        import math
        log_base = math.log(price_base)
        log_quote = math.log(price_quote)
        return log_base - self._coint.hedge_ratio * log_quote - self._coint.intercept

    def _classify_signal(self, z: float, current_signal: Signal) -> Signal:
        """
        Determine the signal given the current z-score.

        Entry logic:
          z > +entry_z  → SHORT_SPREAD (spread too high, expect reversion down)
          z < -entry_z  → LONG_SPREAD  (spread too low, expect reversion up)

        Exit logic (only relevant when there's an open position — handled
        externally, but we still emit EXIT here so the caller knows):
          |z| < exit_z  → EXIT
          |z| > stop_z  → EXIT (stop loss)
        """
        abs_z = abs(z)
        if abs_z > self._stop_z:
            return Signal.EXIT  # Stop loss
        if abs_z < self._exit_z:
            return Signal.EXIT  # Mean reversion complete
        if z > self._entry_z:
            return Signal.SHORT_SPREAD
        if z < -self._entry_z:
            return Signal.LONG_SPREAD
        return Signal.NONE

    def update(
        self,
        timestamp: datetime,
        price_base: float,
        price_quote: float,
    ) -> SpreadState | None:
        """
        Ingest one new price observation and return the current SpreadState.

        Returns None until the rolling window is full (first `window` calls).
        """
        spread = self._compute_spread(price_base, price_quote)
        self._spread_buffer.append(spread)

        if len(self._spread_buffer) < self._window:
            return None  # Not enough data yet

        arr = np.array(self._spread_buffer)
        mu = float(np.mean(arr))
        sigma = float(np.std(arr))

        if sigma < 1e-10:
            return None  # Degenerate spread (all prices identical)

        z = (spread - mu) / sigma
        signal = self._classify_signal(z, Signal.NONE)

        return SpreadState(
            timestamp=timestamp,
            price_base=price_base,
            price_quote=price_quote,
            spread=round(spread, 8),
            z_score=round(z, 4),
            hedge_ratio=self._coint.hedge_ratio,
            signal=signal,
        )

    def seed(self, spreads: list[float]) -> None:
        """
        Pre-populate the rolling window with historical spread values.

        Call this before the first live update to avoid a cold-start period.
        """
        for s in spreads[-self._window:]:
            self._spread_buffer.append(s)
