"""
WebSocket simulation mode: paper trading with streaming crypto data.

This keeps the REST-based simulator intact and adds a lower-overhead market
data path using Kraken WebSocket v2 ticker updates and Bitstamp order-book
stream updates.

Run: python simulate_cex_ws.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from arb_engine import scan_for_arbs
from binance_client import BinanceClient
from config import BITSTAMP_WS_URL, KRAKEN_WS_URL, STRATEGY
from cooldown import CooldownManager
from models import Exchange, PriceSnapshot
from paper_trader import PaperTrader

try:
    import websockets
except ImportError:
    websockets = None

Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/simulation_cex_ws.log"),
    ],
)
logger = logging.getLogger("simulator_ws")

RUNNING = True


def signal_handler(sig, frame):
    global RUNNING
    logger.info("Shutdown signal received")
    RUNNING = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def clear_terminal():
    """Clear without spawning an external shell process."""
    if os.name == "nt":
        os.system("cls")
        return
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def format_price(price: float, currency_symbol: str) -> str:
    """Display small-quote assets with enough precision to be readable."""
    if price >= 10:
        decimals = 2
    elif price >= 1:
        decimals = 4
    else:
        decimals = 5
    return f"{currency_symbol}{price:.{decimals}f}"


def print_sim_dashboard(
    trader,
    scan_count: int,
    arbs_total: int,
    arb_candidates: int,
    all_arbs,
    current_prices: dict,
):
    """Print a live dashboard to terminal."""
    if STRATEGY.quote_currency == "USD":
        currency_symbol = "$"
    elif STRATEGY.quote_currency == "GBP":
        currency_symbol = "£"
    else:
        currency_symbol = f"{STRATEGY.quote_currency} "
    clear_terminal()
    summary = trader.get_portfolio_summary()

    print("=" * 90)
    print("  CRYPTO CEX ARBITRAGE SCANNER — WEBSOCKET PAPER MODE")
    print("=" * 90)
    print(f"  Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"  Scan #{scan_count} | Trading pairs: {len(current_prices)}")
    print("-" * 90)

    pnl_symbol = "+" if summary["total_pnl"] >= 0 else ""
    print("  PORTFOLIO")
    print(
        f"  Bankroll: {currency_symbol}{summary['bankroll']:.2f}  "
        f"(start: {currency_symbol}{summary['initial_bankroll']:.2f})"
    )
    print(
        f"  P&L: {pnl_symbol}{currency_symbol}{summary['total_pnl']:.2f}  "
        f"({pnl_symbol}{summary['total_pnl_pct']:.1f}%)"
    )
    print(
        f"  Trades: {summary['total_trades']}  "
        f"(W:{summary['winning_trades']} / L:{summary['losing_trades']})  "
        f"Win rate: {summary['win_rate']}%"
    )
    print(
        f"  Open: {summary['open_positions']}  "
        f"Peak: {currency_symbol}{summary['peak_bankroll']:.2f}  "
        f"Max DD: {summary['max_drawdown_pct']:.1f}%"
    )
    print("-" * 90)

    print("  LIVE PRICES")
    print(
        f"  {'Pair':<10} {'Binance Mid':<14} {'Bitstamp Mid':<14} {'Kraken Mid':<14}"
    )
    for pair, prices in list(current_prices.items())[:8]:
        bn_prices = prices.get("binance")
        bs_prices = prices.get("bitstamp")
        kr_prices = prices.get("kraken")

        bn_mid = bn_prices["mid"] if bn_prices else 0.0
        bs_mid = bs_prices["mid"] if bs_prices else 0.0
        kr_mid = kr_prices["mid"] if kr_prices else 0.0

        print(
            f"  {pair:<10} "
            f"{format_price(bn_mid, currency_symbol):<14} "
            f"{format_price(bs_mid, currency_symbol):<14} "
            f"{format_price(kr_mid, currency_symbol):<14}"
        )

    print("-" * 90)
    print(f"  Candidate opportunities found: {arb_candidates}")
    print(f"  Executed paper trades: {arbs_total}")
    print(f"  Min edge threshold: {STRATEGY.min_edge_pct}%")

    if all_arbs:
        print()
        print(
            f"  {'Pair':<10} {'Direction':<25} "
            f"{'Edge%':>6} {'Qty':>10} {'Profit':>10}"
        )
        for arb in list(all_arbs)[-STRATEGY.dashboard_recent_arbs:]:
            direction = f"{arb.buy_exchange.value} → {arb.sell_exchange.value}"
            print(
                f"  {arb.pair:<10} {direction:<25} "
                f"{arb.net_edge_pct:>5.2f}% "
                f"{arb.buy_quantity:>10.8f} "
                f"{currency_symbol}{arb.guaranteed_profit_quote:>9.2f}"
            )

    print()
    print("=" * 90)
    print("  Press Ctrl+C to stop")
    print("=" * 90)


class MarketDataStore:
    """Shared in-memory top-of-book cache for both exchanges."""

    def __init__(self):
        self._quotes: dict[tuple[str, str], dict] = {}
        self._lock = asyncio.Lock()

    async def update(
        self,
        exchange: str,
        pair: str,
        bid: float,
        ask: float,
        bid_volume: float,
        ask_volume: float,
    ):
        quote = {
            "bid": bid,
            "ask": ask,
            "bid_volume": bid_volume,
            "ask_volume": ask_volume,
            "mid": (bid + ask) / 2,
            "received_at": time.time(),
        }
        async with self._lock:
            self._quotes[(exchange, pair)] = quote

    async def snapshot(self) -> dict[tuple[str, str], dict]:
        async with self._lock:
            return {key: value.copy() for key, value in self._quotes.items()}


def bitstamp_channel_for_pair(pair: str) -> str:
    return f"order_book_{pair.replace('/', '').lower()}"


async def run_bitstamp_ws(store: MarketDataStore):
    """Consume Bitstamp WebSocket order-book snapshots."""
    channel_to_pair = {
        bitstamp_channel_for_pair(pair): pair for pair in STRATEGY.trading_pairs
    }

    while RUNNING:
        try:
            async with websockets.connect(BITSTAMP_WS_URL, ping_interval=20, ping_timeout=20) as ws:
                logger.info("Connected to Bitstamp WebSocket")
                for channel in channel_to_pair:
                    await ws.send(json.dumps({
                        "event": "bts:subscribe",
                        "data": {"channel": channel},
                    }))

                async for raw_message in ws:
                    if not RUNNING:
                        break

                    message = json.loads(raw_message)
                    event = message.get("event")

                    if event == "bts:request_reconnect":
                        logger.warning("Bitstamp requested reconnect")
                        break

                    if event != "data":
                        continue

                    channel = message.get("channel")
                    pair = channel_to_pair.get(channel)
                    if not pair:
                        continue

                    data = message.get("data", {})
                    bids = data.get("bids") or []
                    asks = data.get("asks") or []
                    if not bids or not asks:
                        continue

                    best_bid = bids[0]
                    best_ask = asks[0]
                    await store.update(
                        "bitstamp",
                        pair,
                        float(best_bid[0]),
                        float(best_ask[0]),
                        float(best_bid[1]),
                        float(best_ask[1]),
                    )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"Bitstamp WebSocket disconnected: {e}")

        if RUNNING:
            await asyncio.sleep(STRATEGY.ws_reconnect_delay_seconds)


async def run_kraken_ws(store: MarketDataStore):
    """Consume Kraken WebSocket v2 ticker updates with BBO trigger."""
    while RUNNING:
        try:
            async with websockets.connect(KRAKEN_WS_URL, ping_interval=20, ping_timeout=20) as ws:
                logger.info("Connected to Kraken WebSocket")
                await ws.send(json.dumps({
                    "method": "subscribe",
                    "params": {
                        "channel": "ticker",
                        "symbol": STRATEGY.trading_pairs,
                        "event_trigger": "bbo",
                        "snapshot": True,
                    },
                }))

                async for raw_message in ws:
                    if not RUNNING:
                        break

                    message = json.loads(raw_message)
                    if message.get("channel") != "ticker":
                        continue

                    data = message.get("data") or []
                    if not data:
                        continue

                    ticker = data[0]
                    pair = ticker.get("symbol")
                    bid = ticker.get("bid")
                    ask = ticker.get("ask")
                    bid_qty = ticker.get("bid_qty")
                    ask_qty = ticker.get("ask_qty")
                    if not pair or bid is None or ask is None:
                        continue

                    await store.update(
                        "kraken",
                        pair,
                        float(bid),
                        float(ask),
                        float(bid_qty or 0.0),
                        float(ask_qty or 0.0),
                    )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"Kraken WebSocket disconnected: {e}")

        if RUNNING:
            await asyncio.sleep(STRATEGY.ws_reconnect_delay_seconds)


def build_price_maps(quotes: dict[tuple[str, str], dict]):
    """Build the scan engine inputs from the live quote cache."""
    now = time.time()
    current_prices = {}
    bitstamp_prices = {}
    kraken_prices = {}

    for pair in STRATEGY.trading_pairs:
        bs_quote = quotes.get(("bitstamp", pair))
        kr_quote = quotes.get(("kraken", pair))

        if bs_quote and now - bs_quote["received_at"] <= STRATEGY.market_data_stale_after_seconds:
            bitstamp_prices[pair] = PriceSnapshot(
                exchange=Exchange.BITSTAMP,
                pair=pair,
                bid=bs_quote["bid"],
                ask=bs_quote["ask"],
                bid_volume=bs_quote["bid_volume"],
                ask_volume=bs_quote["ask_volume"],
            )
            current_prices[pair] = {
                "bitstamp": {
                    "bid": bs_quote["bid"],
                    "ask": bs_quote["ask"],
                    "mid": bs_quote["mid"],
                }
            }

        if kr_quote and now - kr_quote["received_at"] <= STRATEGY.market_data_stale_after_seconds:
            kraken_prices[pair] = PriceSnapshot(
                exchange=Exchange.KRAKEN,
                pair=pair,
                bid=kr_quote["bid"],
                ask=kr_quote["ask"],
                bid_volume=kr_quote["bid_volume"],
                ask_volume=kr_quote["ask_volume"],
            )
            current_prices.setdefault(pair, {})["kraken"] = {
                "bid": kr_quote["bid"],
                "ask": kr_quote["ask"],
                "mid": kr_quote["mid"],
            }

    return bitstamp_prices, kraken_prices, current_prices


async def run_simulation_ws():
    if websockets is None:
        raise RuntimeError(
            "WebSocket simulator requires the 'websockets' package. "
            "Install it with: pip install websockets"
        )

    logger.info("Starting websocket paper trading mode")
    store = MarketDataStore()
    trader = PaperTrader()
    cooldown = CooldownManager(cooldown_seconds=15.0)
    binance = BinanceClient()
    logger.info("Testing Binance connection...")
    if not binance.test_connection():
        logger.warning("Binance connection test failed, proceeding anyway")

    feed_tasks = [
        asyncio.create_task(run_bitstamp_ws(store), name="bitstamp_ws"),
        asyncio.create_task(run_kraken_ws(store), name="kraken_ws"),
    ]

    scan_count = 0
    arbs_total = 0
    all_arbs = deque(maxlen=500)

    try:
        while RUNNING:
            loop_started = time.monotonic()
            scan_count += 1

            quotes = await store.snapshot()
            bitstamp_prices, kraken_prices, current_prices = build_price_maps(quotes)

            binance_prices = {}
            for pair in STRATEGY.trading_pairs:
                bn_quote = await asyncio.to_thread(
                    binance.get_top_of_book,
                    pair,
                    STRATEGY.order_book_depth,
                )
                if bn_quote:
                    binance_prices[pair] = PriceSnapshot(
                        exchange=Exchange.BINANCE,
                        pair=pair,
                        bid=bn_quote["bid"],
                        ask=bn_quote["ask"],
                        bid_volume=bn_quote["bid_volume"],
                        ask_volume=bn_quote["ask_volume"],
                    )
                    current_prices.setdefault(pair, {})["binance"] = {
                        "bid": bn_quote["bid"],
                        "ask": bn_quote["ask"],
                        "mid": bn_quote["mid"],
                    }

            opportunities, near_misses = scan_for_arbs(
                {
                    "binance": binance_prices,
                    "bitstamp": bitstamp_prices,
                    "kraken": kraken_prices,
                },
                bankroll=trader.portfolio.bankroll,
            )

            for opp in opportunities:
                key = (opp.pair, f"{opp.buy_exchange.value}-{opp.sell_exchange.value}")
                if cooldown.can_trade(key[0], key[1]):
                    trade = trader.execute_paper_trade(opp)
                    if trade:
                        trader.auto_resolve_trade_immediately(trade)
                        cooldown.record_trade(key[0], key[1])
                        arbs_total += 1
                        all_arbs.append(opp)

            if scan_count % STRATEGY.dashboard_update_interval_scans == 0:
                print_sim_dashboard(
                    trader,
                    scan_count,
                    arbs_total,
                    len(opportunities),
                    all_arbs,
                    current_prices,
                )

            if scan_count % STRATEGY.snapshot_log_interval_scans == 0:
                summary = trader.get_portfolio_summary()
                trader.log_snapshot({
                    "scan": scan_count,
                    "opportunities": len(opportunities),
                    "arbs_found": arbs_total,
                    "quote_currency": summary["quote_currency"],
                    "bankroll": summary["bankroll"],
                    "total_pnl": summary["total_pnl"],
                    "total_trades": summary["total_trades"],
                    "market_data_mode": "websocket",
                })

            elapsed = time.monotonic() - loop_started
            sleep_for = STRATEGY.scan_interval_seconds - elapsed
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
    finally:
        for task in feed_tasks:
            task.cancel()
        await asyncio.gather(*feed_tasks, return_exceptions=True)

        logger.info("WebSocket simulation ended")
        summary = trader.get_portfolio_summary()
        print("\n" + "=" * 90)
        print("  FINAL SUMMARY")
        print("=" * 90)
        print(f"  Total scans: {scan_count}")
        print(f"  Total arbitrages found: {arbs_total}")
        print(f"  Total trades executed: {summary['total_trades']}")
        print(f"  Winning trades: {summary['winning_trades']}")
        print(f"  Losing trades: {summary['losing_trades']}")
        print(f"  Win rate: {summary['win_rate']}%")
        print(
            f"  Final bankroll: {summary['bankroll']:.2f} {summary['quote_currency']}"
        )
        print(
            f"  Total P&L: {summary['total_pnl']:.2f} "
            f"{summary['quote_currency']} ({summary['total_pnl_pct']:.1f}%)"
        )
        print(f"  Max drawdown: {summary['max_drawdown_pct']:.1f}%")
        print("=" * 90)


def main():
    asyncio.run(run_simulation_ws())


if __name__ == "__main__":
    main()
