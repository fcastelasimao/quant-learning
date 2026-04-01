"""
Arbitrage Detection Engine for Crypto CEX.

Detects cross-CEX arbitrage opportunities where:
  - BUY price on exchange A < SELL price on exchange B
  - After accounting for commissions on both sides
  - With sufficient liquidity on both sides

Example:
  - BTC trades 67,100 on Bitstamp, 67,500 on Kraken
  - Buy cost = 67,100 * (1 + buy_commission)
  - Sell revenue = 67,500 * (1 - sell_commission)
  - Profit per unit = sell revenue - buy cost
"""
from __future__ import annotations
import logging
from typing import Optional

from config import EXCHANGE_CONFIGS, STRATEGY
from models import (
    PriceSnapshot,
    ArbOpportunity,
)

logger = logging.getLogger(__name__)


def calculate_arb(
    buy_prices: PriceSnapshot,
    sell_prices: PriceSnapshot,
    pair: str,
    bankroll: float,
) -> Optional[ArbOpportunity]:
    """
    Check if there's an arbitrage opportunity between two price snapshots.
    
    Args:
        buy_prices: PriceSnapshot to buy from (we use the ask price)
        sell_prices: PriceSnapshot to sell to (we use the bid price)
        pair: Trading pair (e.g., "BTC/GBP")
    
    Returns:
        ArbOpportunity if edge >= min_edge_pct, else None
    """
    # Check that we have the prices we need
    if not buy_prices.ask or not sell_prices.bid:
        return None
    
    # Can't make money if buy price >= sell price
    if buy_prices.ask >= sell_prices.bid:
        return None
    
    buy_price = buy_prices.ask
    sell_price = sell_prices.bid
    
    # ── Determine commission rates ──
    buy_comm = EXCHANGE_CONFIGS[buy_prices.exchange.value]["commission"]
    sell_comm = EXCHANGE_CONFIGS[sell_prices.exchange.value]["commission"]
    
    # ── Calculate gross edge (before commissions) ──
    gross_edge_pct = ((sell_price / buy_price) - 1) * 100
    
    # ── Calculate net edge (after commissions) ──
    # For BUY side: cost = quantity * buy_price * (1 + buy_comm)
    # For SELL side: revenue = quantity * sell_price * (1 - sell_comm)
    # Profit per unit = sell_price * (1 - sell_comm) - buy_price * (1 + buy_comm)
    # Net edge = Profit per unit / buy_price per unit
    
    profit_per_unit = sell_price * (1 - sell_comm) - buy_price * (1 + buy_comm)
    net_edge_pct = (profit_per_unit / buy_price) * 100
    
    # Check if edge meets minimum threshold
    if net_edge_pct < STRATEGY.min_edge_pct:
        return None
    
    # ── Calculate position sizing ──
    # Limit by liquidity on both sides
    max_from_buy_liquidity = buy_prices.ask_volume if buy_prices.ask_volume else 0
    max_from_sell_liquidity = sell_prices.bid_volume if sell_prices.bid_volume else 0
    
    # Take the minimum of available liquidity
    if max_from_buy_liquidity <= 0 or max_from_sell_liquidity <= 0:
        return None
    
    if buy_prices.ask_notional < STRATEGY.min_liquidity_quote:
        return None

    if sell_prices.bid_notional < STRATEGY.min_liquidity_quote:
        return None

    max_quantity_from_liquidity = min(max_from_buy_liquidity, max_from_sell_liquidity)

    # Limit by configured max stake.
    max_quantity_from_stake_buy = STRATEGY.max_stake_per_trade / buy_price
    max_quantity_from_stake_sell = STRATEGY.max_stake_per_trade / sell_price
    max_quantity_from_stake = min(max_quantity_from_stake_buy, max_quantity_from_stake_sell)

    # Limit by max position % of bankroll.
    max_quantity_from_bankroll = bankroll * STRATEGY.max_position_pct / buy_price

    quantity = min(
        max_quantity_from_liquidity,
        max_quantity_from_stake,
        max_quantity_from_bankroll,
    )

    if quantity <= 0:
        return None

    stake_required = quantity * buy_price
    if stake_required < STRATEGY.min_stake_per_trade:
        return None

    # Calculate profit.
    guaranteed_profit_quote = quantity * profit_per_unit

    quantity = round(quantity, 8)
    guaranteed_profit_quote = round(guaranteed_profit_quote, 2)

    opp = ArbOpportunity(
        pair=pair,
        buy_exchange=buy_prices.exchange,
        buy_price=round(buy_price, 8),
        buy_volume_available=buy_prices.ask_volume or 0,
        sell_exchange=sell_prices.exchange,
        sell_price=round(sell_price, 8),
        sell_volume_available=sell_prices.bid_volume or 0,
        gross_edge_pct=round(gross_edge_pct, 3),
        net_edge_pct=round(net_edge_pct, 3),
        buy_quantity=quantity,
        sell_quantity=quantity,
        guaranteed_profit_quote=guaranteed_profit_quote,
    )

    logger.info(
        f"ARB FOUND: {pair} | "
        f"Buy {opp.buy_exchange.value}@{buy_price:.8f} / "
        f"Sell {opp.sell_exchange.value}@{sell_price:.8f} | "
        f"Net edge: {net_edge_pct:.3f}% | "
        f"Profit: {guaranteed_profit_quote:.2f} {STRATEGY.quote_currency}"
    )

    return opp


def scan_for_arbs(
    exchange_price_maps: dict[str, dict[str, PriceSnapshot]],
    bankroll: float,
) -> list[ArbOpportunity]:
    """
    Scan all trading pairs across all available exchanges for arbitrage opportunities.

    Args:
        exchange_price_maps: dict of exchange name -> pair -> PriceSnapshot

    Returns:
        list of ArbOpportunity objects
    """
    opportunities = []
    all_pairs = set()
    for prices in exchange_price_maps.values():
        all_pairs |= set(prices.keys())

    for pair in all_pairs:
        available = [
            (exchange_name, prices[pair])
            for exchange_name, prices in exchange_price_maps.items()
            if pair in prices
        ]
        for buy_exchange_name, buy_snap in available:
            for sell_exchange_name, sell_snap in available:
                if buy_exchange_name == sell_exchange_name:
                    continue
                arb = calculate_arb(buy_snap, sell_snap, pair, bankroll)
                if arb:
                    opportunities.append(arb)

    return opportunities
