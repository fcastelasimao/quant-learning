"""
Walk-forward backtester for pairs trading strategies.

Methodology:
  1. Split historical candles into in-sample (IS) and out-of-sample (OOS).
  2. Fit the hedge ratio on IS data only (avoids look-ahead bias).
  3. Simulate trading on OOS data using the fitted parameters.
  4. Report performance metrics.

This is a simple bar-by-bar simulation — each 1h candle is one step.
Trades execute at the open of the candle AFTER the signal (realistic fill).
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from cointegration import compute_spread, rolling_zscore, test_cointegration
from config import STRATEGY
from models import (
    BacktestResult,
    BacktestTrade,
    Candle,
    CointegrationResult,
    Signal,
)
from spread_tracker import SpreadTracker

logger = logging.getLogger(__name__)


def _pnl_for_trade(
    direction: Signal,
    entry_price_base: float,
    entry_price_quote: float,
    exit_price_base: float,
    exit_price_quote: float,
    hedge_ratio: float,
    commission: float,
) -> float:
    """
    Compute percentage P&L for a completed two-leg trade.

    For LONG_SPREAD  (buy base, sell quote):
      pnl = (exit_base - entry_base) / entry_base
          - hedge_ratio * (exit_quote - entry_quote) / entry_quote
          - 4 * commission   (2 legs x open + close)

    For SHORT_SPREAD (sell base, buy quote): signs are flipped.
    """
    base_return = (exit_price_base - entry_price_base) / entry_price_base
    quote_return = (exit_price_quote - entry_price_quote) / entry_price_quote
    cost = 4 * commission  # open + close, 2 legs

    if direction == Signal.LONG_SPREAD:
        raw_pnl = base_return - hedge_ratio * quote_return
    else:
        raw_pnl = -base_return + hedge_ratio * quote_return

    return raw_pnl - cost


def _compute_metrics(
    trades: list[BacktestTrade],
    initial_bankroll: float = 1000.0,
) -> dict:
    if not trades:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate_pct": 0.0,
            "total_pnl_pct": 0.0,
            "avg_pnl_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0,
            "avg_holding_hours": 0.0,
        }

    pnls = [t.pnl_pct for t in trades]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]

    # Equity curve for drawdown
    equity = initial_bankroll
    peak = initial_bankroll
    max_dd = 0.0
    for p in pnls:
        equity *= (1 + p / 100)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd:
            max_dd = dd

    total_pnl = ((equity / initial_bankroll) - 1) * 100

    # Sharpe (annualised, assuming 1h candles → 8760 periods/year)
    if len(pnls) > 1:
        mean_r = np.mean(pnls)
        std_r = np.std(pnls, ddof=1)
        periods_per_year = 8760 / (np.mean([t.holding_hours for t in trades]) or 1)
        sharpe = (mean_r / std_r * math.sqrt(periods_per_year)) if std_r > 0 else 0.0
    else:
        sharpe = 0.0

    return {
        "total_trades": len(trades),
        "winning_trades": len(winners),
        "losing_trades": len(losers),
        "win_rate_pct": round(len(winners) / len(trades) * 100, 1),
        "total_pnl_pct": round(total_pnl, 2),
        "avg_pnl_pct": round(float(np.mean(pnls)), 4),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe_ratio": round(sharpe, 2),
        "avg_holding_hours": round(float(np.mean([t.holding_hours for t in trades])), 1),
    }


def run_backtest(
    base_candles: list[Candle],
    quote_candles: list[Candle],
    base_symbol: str,
    quote_symbol: str,
) -> Optional[BacktestResult]:
    """
    Run a full walk-forward backtest for a single pair.

    Returns None if the pair fails the cointegration test on in-sample data.
    """
    n = min(len(base_candles), len(quote_candles))
    if n < STRATEGY.zscore_window * 2:
        logger.warning(f"Not enough candles for {base_symbol}/{quote_symbol} ({n})")
        return None

    # Align candles
    base_candles = base_candles[-n:]
    quote_candles = quote_candles[-n:]

    split = int(n * STRATEGY.in_sample_fraction)

    # ── In-sample: fit the hedge ratio ────────────────────────────────────────
    is_base = base_candles[:split]
    is_quote = quote_candles[:split]

    coint = test_cointegration(is_base, is_quote, base_symbol, quote_symbol)
    if coint is None or not coint.is_cointegrated:
        logger.info(
            f"{base_symbol}/{quote_symbol}: Not cointegrated on in-sample data — skipping"
        )
        return None

    # ── Out-of-sample: simulate trading ───────────────────────────────────────
    oos_base = base_candles[split:]
    oos_quote = quote_candles[split:]

    tracker = SpreadTracker(coint)

    # Seed the tracker with the tail of the in-sample spread so the window
    # is warm at the start of the OOS period.
    is_spread = compute_spread(is_base, is_quote, coint.hedge_ratio, coint.intercept)
    tracker.seed(list(is_spread))

    trades: list[BacktestTrade] = []
    open_direction: Optional[Signal] = None
    open_entry_i: Optional[int] = None
    open_entry_base: Optional[float] = None
    open_entry_quote: Optional[float] = None
    open_entry_z: Optional[float] = None

    for i, (cb, cq) in enumerate(zip(oos_base, oos_quote)):
        state = tracker.update(
            timestamp=cb.timestamp,
            price_base=cb.close,
            price_quote=cq.close,
        )
        if state is None:
            continue

        # ── Exit logic ────────────────────────────────────────────────────────
        if open_direction is not None:
            holding = i - open_entry_i
            should_exit = (
                state.signal == Signal.EXIT
                or holding >= STRATEGY.max_holding_hours
                # Stop loss: spread widened further against us
                or (open_direction == Signal.LONG_SPREAD and state.z_score < -STRATEGY.stop_loss_z_score)
                or (open_direction == Signal.SHORT_SPREAD and state.z_score > STRATEGY.stop_loss_z_score)
            )

            if should_exit:
                if holding >= STRATEGY.max_holding_hours:
                    reason = "timeout"
                elif state.signal == Signal.EXIT and abs(state.z_score) > STRATEGY.stop_loss_z_score:
                    reason = "stop_loss"
                else:
                    reason = "signal"

                pnl = _pnl_for_trade(
                    direction=open_direction,
                    entry_price_base=open_entry_base,
                    entry_price_quote=open_entry_quote,
                    exit_price_base=cb.close,
                    exit_price_quote=cq.close,
                    hedge_ratio=coint.hedge_ratio,
                    commission=STRATEGY.commission,
                )
                trades.append(BacktestTrade(
                    base=base_symbol,
                    quote=quote_symbol,
                    direction=open_direction.value,
                    entry_time=oos_base[open_entry_i].timestamp,
                    exit_time=cb.timestamp,
                    holding_hours=holding,
                    entry_z=open_entry_z,
                    exit_z=state.z_score,
                    pnl_pct=round(pnl * 100, 4),
                    exit_reason=reason,
                ))
                open_direction = None

        # ── Entry logic ───────────────────────────────────────────────────────
        if open_direction is None and state.signal in (Signal.LONG_SPREAD, Signal.SHORT_SPREAD):
            open_direction = state.signal
            open_entry_i = i
            open_entry_base = cb.close
            open_entry_quote = cq.close
            open_entry_z = state.z_score

    metrics = _compute_metrics(trades)

    return BacktestResult(
        base=base_symbol,
        quote=quote_symbol,
        candle_interval=STRATEGY.candle_interval,
        in_sample_candles=split,
        out_of_sample_candles=len(oos_base),
        trades=trades,
        **metrics,
    )
