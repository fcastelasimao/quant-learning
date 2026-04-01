"""
Simulation Mode: Paper trading with live crypto data from Bitstamp + Kraken.

This fetches live prices from both exchanges and simulates arbitrage detection
without risking real money.
"""
from __future__ import annotations
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

from config import STRATEGY
from models import Exchange, PriceSnapshot
from binance_client import BinanceClient
from bitstamp_client import BitstampClient
from kraken_client import KrakenClient
from arb_engine import scan_for_arbs
from paper_trader import PaperTrader
from cooldown import CooldownManager

Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/simulation_cex.log"),
    ],
)
logger = logging.getLogger("simulator")

RUNNING = True


def signal_handler(sig, frame):
    global RUNNING
    logger.info("Shutdown signal received")
    RUNNING = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ──────────────────────────────────────────────
# Dashboard
# ──────────────────────────────────────────────
def print_sim_dashboard(
    trader,
    scan_count: int,
    arbs_total: int,
    all_arbs: list,
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
    print("  CRYPTO CEX ARBITRAGE SCANNER — PAPER TRADING MODE")
    print("=" * 90)
    print(f"  Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"  Scan #{scan_count} | Trading pairs: {len(current_prices)}")
    print("-" * 90)

    # Portfolio
    pnl_symbol = "+" if summary["total_pnl"] >= 0 else ""
    print(f"  PORTFOLIO")
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

    # Current prices
    print(f"  LIVE PRICES")
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
    print(f"  ARBITRAGE OPPORTUNITIES: {arbs_total}")
    print(f"  Min edge threshold: {STRATEGY.min_edge_pct}%")

    if all_arbs:
        print()
        print(
            f"  {'Pair':<10} {'Direction':<25} "
            f"{'Edge%':>6} {'Qty':>10} {'Profit':>10}"
        )
        for arb in all_arbs[-STRATEGY.dashboard_recent_arbs:]:
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


def clear_terminal():
    """Clear without spawning a shell process every dashboard refresh."""
    if os.name == "nt":
        os.system("cls")
        return
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def format_price(price: float, currency_symbol: str) -> str:
    """Display small-quote assets with enough precision to be readable."""
    if price >= 1000:
        decimals = 2
    elif price >= 10:
        decimals = 2
    elif price >= 1:
        decimals = 4
    else:
        decimals = 5
    return f"{currency_symbol}{price:.{decimals}f}"


# ──────────────────────────────────────────────
# Main simulation loop
# ──────────────────────────────────────────────
def run_simulation():
    logger.info("Starting paper trading mode with live prices")

    bitstamp = BitstampClient()
    kraken = KrakenClient()
    binance = BinanceClient()
    trader = PaperTrader()
    cooldown = CooldownManager(cooldown_seconds=15.0)

    # Check connections
    logger.info("Testing Binance connection...")
    if not binance.test_connection():
        logger.warning("Binance connection test failed, proceeding anyway")

    logger.info("Testing Bitstamp connection...")
    if not bitstamp.test_connection():
        logger.warning("Bitstamp connection test failed, proceeding anyway")

    logger.info("Testing Kraken connection...")
    if not kraken.test_connection():
        logger.warning("Kraken connection test failed, proceeding anyway")

    scan_count = 0
    arbs_total = 0
    all_arbs = []
    current_prices = {}

    try:
        while RUNNING:
            loop_started = time.monotonic()
            scan_count += 1

            # Fetch prices for all pairs
            binance_prices = {}
            bitstamp_prices = {}
            kraken_prices = {}
            current_prices = {}

            for pair in STRATEGY.trading_pairs:
                # Fetch from Binance
                bn_quote = binance.get_top_of_book(pair, depth=STRATEGY.order_book_depth)
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

                # Fetch from Bitstamp
                bs_quote = bitstamp.get_top_of_book(pair, depth=STRATEGY.order_book_depth)
                if bs_quote:
                    bitstamp_prices[pair] = PriceSnapshot(
                        exchange=Exchange.BITSTAMP,
                        pair=pair,
                        bid=bs_quote["bid"],
                        ask=bs_quote["ask"],
                        bid_volume=bs_quote["bid_volume"],
                        ask_volume=bs_quote["ask_volume"],
                    )
                    current_prices.setdefault(pair, {})["bitstamp"] = {
                        "bid": bs_quote["bid"],
                        "ask": bs_quote["ask"],
                        "mid": bs_quote["mid"],
                    }

                # Fetch from Kraken
                kr_quote = kraken.get_top_of_book(pair, depth=STRATEGY.order_book_depth)
                if kr_quote:
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

            # Scan for arbitrage across Binance, Bitstamp, and Kraken
            opportunities = scan_for_arbs(
                {
                    "binance": binance_prices,
                    "bitstamp": bitstamp_prices,
                    "kraken": kraken_prices,
                },
                bankroll=trader.portfolio.bankroll,
            )

            # Filter by cooldown
            for opp in opportunities:
                key = (opp.pair, f"{opp.buy_exchange.value}-{opp.sell_exchange.value}")
                if cooldown.can_trade(key[0], key[1]):
                    # Execute paper trade
                    trade = trader.execute_paper_trade(opp)
                    if trade:
                        # Immediately resolve it (crypto settles instantly)
                        trader.auto_resolve_trade_immediately(trade)
                        cooldown.record_trade(key[0], key[1])
                        arbs_total += 1
                        all_arbs.append(opp)

            # Print dashboard
            if scan_count % STRATEGY.dashboard_update_interval_scans == 0:
                print_sim_dashboard(trader, scan_count, arbs_total, all_arbs, current_prices)

            # Log snapshot
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
                })

            # Sleep before next scan
            elapsed = time.monotonic() - loop_started
            sleep_for = STRATEGY.scan_interval_seconds - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.exception(f"Error during simulation: {e}")
    finally:
        logger.info("Simulation ended")
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


if __name__ == "__main__":
    run_simulation()
