"""
Paper Trading Engine for Crypto CEX Arbitrage.

Simulates trade execution, tracks portfolio state, and logs everything
to JSONL files for later analysis.

For crypto CEX arbs, trades settle instantly (within seconds).
This does NOT place real trades. It records what WOULD have happened.
"""
from __future__ import annotations
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import STRATEGY
from models import (
    ArbOpportunity,
    PaperTrade,
    PortfolioState,
    TradeStatus,
)

logger = logging.getLogger(__name__)


class PaperTrader:
    """Simulates crypto CEX arbitrage trade execution and tracks P&L."""

    def __init__(self):
        self.portfolio = PortfolioState(bankroll=STRATEGY.initial_bankroll)
        self.portfolio.peak_bankroll = STRATEGY.initial_bankroll
        self.daily_loss = 0.0

        # Ensure data directory exists
        Path(STRATEGY.trades_log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(STRATEGY.snapshots_path).parent.mkdir(parents=True, exist_ok=True)

    def execute_paper_trade(self, opp: ArbOpportunity) -> Optional[PaperTrade]:
        """
        'Execute' an arbitrage opportunity as a paper trade.
        For crypto CEX arbs, we simulate instant settlement.
        
        Returns the PaperTrade if executed, None if rejected.
        """
        # Check if stopped for the day (daily loss limit)
        if self.portfolio.stopped_for_day:
            logger.warning("Daily loss limit hit, not executing new trades")
            return None
        
        # Check portfolio constraints
        if len(self.portfolio.open_positions) >= STRATEGY.max_open_positions:
            logger.warning("Max open positions reached, skipping trade")
            return None

        # Check bankroll
        total_cost = opp.buy_quantity * opp.buy_price
        max_cost = self.portfolio.bankroll * STRATEGY.max_position_pct
        if total_cost > max_cost:
            logger.warning(
                f"Trade cost {total_cost:.2f} {STRATEGY.quote_currency} exceeds "
                f"{STRATEGY.max_position_pct:.0%} of bankroll "
                f"({self.portfolio.bankroll:.2f} {STRATEGY.quote_currency}), skipping"
            )
            return None

        # Create paper trade
        trade = PaperTrade(
            opportunity_id=opp.id,
            pair=opp.pair,
            buy_exchange=opp.buy_exchange,
            buy_price=opp.buy_price,
            buy_quantity=opp.buy_quantity,
            sell_exchange=opp.sell_exchange,
            sell_price=opp.sell_price,
            sell_quantity=opp.sell_quantity,
            expected_profit_quote=opp.guaranteed_profit_quote,
            status=TradeStatus.OPEN,
        )

        self.portfolio.open_positions.append(trade)
        self.portfolio.total_trades += 1

        # Log trade
        self._log_trade(trade)

        logger.info(
            f"PAPER TRADE EXECUTED: {trade.id} | "
            f"{trade.pair} | "
            f"Buy {trade.buy_exchange.value}@{trade.buy_price:.8f} "
            f"(qty: {trade.buy_quantity:.8f}) | "
            f"Sell {trade.sell_exchange.value}@{trade.sell_price:.8f} "
            f"(qty: {trade.sell_quantity:.8f}) | "
            f"Expected profit: {trade.expected_profit_quote:.2f} {STRATEGY.quote_currency}"
        )

        return trade

    def auto_resolve_trade_immediately(self, trade: PaperTrade) -> Optional[PaperTrade]:
        """
        For crypto CEX arbs, the profit is guaranteed once both sides are executed.
        In reality this happens within milliseconds on-chain.
        We realize the profit immediately in paper mode.
        """
        if not any(t.id == trade.id for t in self.portfolio.open_positions):
            logger.warning(f"Trade {trade.id} not in open positions")
            return None

        # Profit is already calculated and guaranteed
        trade.actual_profit_quote = trade.expected_profit_quote
        trade.status = TradeStatus.CLOSED
        trade.resolved_at = datetime.utcnow()

        # Update portfolio
        self.portfolio.bankroll += trade.actual_profit_quote
        self.portfolio.total_pnl += trade.actual_profit_quote
        
        if trade.actual_profit_quote > 0:
            self.portfolio.winning_trades += 1
        else:
            self.portfolio.losing_trades += 1
            self.daily_loss += abs(trade.actual_profit_quote)
        
        self.portfolio.update_drawdown()

        # Check if daily loss limit exceeded
        if self.daily_loss > (STRATEGY.initial_bankroll * STRATEGY.daily_loss_limit_pct):
            logger.warning(
                f"Daily loss limit exceeded "
                f"({self.daily_loss:.2f} {STRATEGY.quote_currency}), stopping trades"
            )
            self.portfolio.stopped_for_day = True

        # Check if max drawdown exceeded
        if self.portfolio.max_drawdown > STRATEGY.max_drawdown_pct:
            logger.error(
                f"Max drawdown limit exceeded ({self.portfolio.max_drawdown*100:.1f}%), "
                f"HARD STOP - emergency halt"
            )
            self.portfolio.stopped_for_day = True

        # Move from open to closed
        self.portfolio.open_positions.remove(trade)
        self.portfolio.closed_positions.append(trade)

        # Log update
        self._log_trade(trade)

        logger.info(
            f"TRADE RESOLVED: {trade.id} | "
            f"{trade.pair} | "
            f"P&L: {trade.actual_profit_quote:.2f} {STRATEGY.quote_currency} | "
            f"Bankroll: {self.portfolio.bankroll:.2f} {STRATEGY.quote_currency}"
        )

        return trade

    def get_portfolio_summary(self) -> dict:
        """Return a summary of current portfolio state."""
        return {
            "quote_currency": STRATEGY.quote_currency,
            "bankroll": round(self.portfolio.bankroll, 2),
            "initial_bankroll": STRATEGY.initial_bankroll,
            "total_pnl": round(self.portfolio.total_pnl, 2),
            "total_pnl_pct": round(
                (self.portfolio.total_pnl / STRATEGY.initial_bankroll) * 100, 2
            ),
            "total_trades": self.portfolio.total_trades,
            "winning_trades": self.portfolio.winning_trades,
            "losing_trades": self.portfolio.losing_trades,
            "win_rate": (
                round(
                    self.portfolio.winning_trades
                    / max(
                        self.portfolio.winning_trades + self.portfolio.losing_trades,
                        1,
                    )
                    * 100,
                    1,
                )
            ) if (self.portfolio.winning_trades + self.portfolio.losing_trades) > 0 else 0,
            "open_positions": len(self.portfolio.open_positions),
            "peak_bankroll": round(self.portfolio.peak_bankroll, 2),
            "max_drawdown_pct": round(self.portfolio.max_drawdown * 100, 2),
            "daily_loss": round(self.daily_loss, 2),
            "stopped_for_day": self.portfolio.stopped_for_day,
        }

    def _log_trade(self, trade: PaperTrade):
        """Append trade to JSONL log file."""
        try:
            with open(STRATEGY.trades_log_path, "a") as f:
                f.write(trade.to_json() + "\n")
        except IOError as e:
            logger.error(f"Failed to log trade: {e}")

    def log_snapshot(self, data: dict):
        """Log an odds snapshot or scan result."""
        try:
            data["_timestamp"] = datetime.utcnow().isoformat()
            with open(STRATEGY.snapshots_path, "a") as f:
                f.write(json.dumps(data) + "\n")
        except IOError as e:
            logger.error(f"Failed to log snapshot: {e}")
