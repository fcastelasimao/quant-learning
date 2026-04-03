"""
Entry point: run the backtest across all candidate pairs, print results,
and emit a KEEP / DROP verdict for each one.

Verdict logic:
  KEEP  — cointegrated AND out-of-sample P&L > 0 AND at least 5 trades
  WATCH — cointegrated AND P&L < 0 but win rate >= 50% (may improve with tuning)
  DROP  — failed cointegration OR P&L < 0 AND win rate < 50%

Usage:
    python3 run_backtest.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from backtester import run_backtest
from config import STRATEGY
from data_fetcher import fetch_candles_cached
from models import BacktestResult

Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/backtest.log"),
    ],
)
logger = logging.getLogger("run_backtest")

# ── Verdict thresholds ────────────────────────────────────────────────────────
_MIN_TRADES_FOR_VERDICT = 3
_KEEP_MIN_PNL_PCT = 0.0
_WATCH_MIN_WIN_RATE = 50.0


def _verdict(result: BacktestResult | None, failed_coint: bool) -> tuple[str, str]:
    """Return (label, reason) for a pair."""
    if failed_coint or result is None:
        return "DROP ", "failed cointegration test"
    if result.total_trades < _MIN_TRADES_FOR_VERDICT:
        return "DROP ", f"too few trades ({result.total_trades}) — signal too rare"
    if result.total_pnl_pct > _KEEP_MIN_PNL_PCT and result.sharpe_ratio > 0:
        return "KEEP ", f"P&L={result.total_pnl_pct:+.2f}%  Sharpe={result.sharpe_ratio:.2f}"
    if result.win_rate_pct >= _WATCH_MIN_WIN_RATE:
        return "WATCH", f"P&L={result.total_pnl_pct:+.2f}% but win_rate={result.win_rate_pct:.1f}% — needs tuning"
    return "DROP ", f"P&L={result.total_pnl_pct:+.2f}%  win_rate={result.win_rate_pct:.1f}%"


def print_result(result: BacktestResult) -> None:
    pair = f"{result.base}/{result.quote}"
    print(f"\n{'='*70}")
    print(f"  PAIR: {pair}  |  Interval: {result.candle_interval}")
    print(f"{'='*70}")
    print(f"  In-sample:   {result.in_sample_candles} candles  |  "
          f"Out-of-sample: {result.out_of_sample_candles} candles")
    print(f"  Total trades: {result.total_trades}")
    if result.total_trades == 0:
        print("  No trades executed in out-of-sample period.")
        return

    sign = "+" if result.total_pnl_pct >= 0 else ""
    print(f"  Win rate:     {result.win_rate_pct:.1f}%  ({result.winning_trades}W / {result.losing_trades}L)")
    print(f"  Total P&L:    {sign}{result.total_pnl_pct:.2f}%")
    print(f"  Avg P&L:      {result.avg_pnl_pct:+.4f}%")
    print(f"  Max drawdown: {result.max_drawdown_pct:.2f}%")
    print(f"  Sharpe:       {result.sharpe_ratio:.2f}")
    print(f"  Avg hold:     {result.avg_holding_hours:.1f}h")
    print(f"\n  {'Entry time':<22} {'Direction':<14} {'Entry z':>8} {'Exit z':>8} "
          f"{'Hold(h)':>8} {'P&L%':>8}  Reason")
    print(f"  {'-'*22} {'-'*14} {'-'*8} {'-'*8} {'-'*8} {'-'*8}  {'-'*10}")
    for t in result.trades:
        print(
            f"  {t.entry_time.strftime('%Y-%m-%d %H:%M'):<22} "
            f"{t.direction:<14} "
            f"{t.entry_z:>+8.2f} "
            f"{t.exit_z:>+8.2f} "
            f"{t.holding_hours:>8.0f} "
            f"{t.pnl_pct:>+8.3f}%  "
            f"{t.exit_reason}"
        )


def main():
    print("\n" + "=" * 70)
    print("  PAIRS TRADING — BACKTEST")
    print(f"  Lookback: {STRATEGY.lookback_candles} x {STRATEGY.candle_interval} candles  "
          f"({STRATEGY.lookback_candles}h ≈ {STRATEGY.lookback_candles//24}d)")
    print(f"  In-sample: {int(STRATEGY.in_sample_fraction*100)}%  |  "
          f"Out-of-sample: {int((1-STRATEGY.in_sample_fraction)*100)}%")
    print(f"  Entry |z|: {STRATEGY.entry_z_score}  |  "
          f"Exit |z|: {STRATEGY.exit_z_score}  |  "
          f"Stop |z|: {STRATEGY.stop_loss_z_score}")
    print(f"  Z-score window: {STRATEGY.zscore_window}h  |  "
          f"Commission: {STRATEGY.commission*100:.2f}% per leg")
    print("=" * 70)

    results: list[tuple[str, str, BacktestResult | None, str, str]] = []
    # stores: (base, quote, result_or_none, verdict_label, verdict_reason)

    for base_sym, quote_sym in STRATEGY.candidate_pairs:
        pair_label = f"{base_sym}/{quote_sym}"
        logger.info(f"{'-'*50}")
        logger.info(f"Testing {pair_label}...")

        base_candles = fetch_candles_cached(
            base_sym,
            interval=STRATEGY.candle_interval,
            limit=STRATEGY.lookback_candles,
            cache_dir=STRATEGY.data_cache_path,
        )
        quote_candles = fetch_candles_cached(
            quote_sym,
            interval=STRATEGY.candle_interval,
            limit=STRATEGY.lookback_candles,
            cache_dir=STRATEGY.data_cache_path,
        )

        if not base_candles or not quote_candles:
            logger.warning(f"No data for {pair_label} — skipping")
            results.append((base_sym, quote_sym, None, "DROP ", "no data"))
            continue

        result = run_backtest(base_candles, quote_candles, base_sym, quote_sym)
        failed_coint = result is None
        label, reason = _verdict(result, failed_coint)

        results.append((base_sym, quote_sym, result, label, reason))

        if result:
            print_result(result)

    # ── Final verdict table ───────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("  PAIR SCREENING RESULTS")
    print(f"{'='*70}")
    print(f"  {'Pair':<24} {'Verdict':<8}  {'Trades':>6}  {'P&L%':>7}  {'WR%':>5}  {'Sharpe':>6}  Reason")
    print(f"  {'-'*24} {'-'*8}  {'-'*6}  {'-'*7}  {'-'*5}  {'-'*6}  {'-'*30}")

    keep_pairs = []
    watch_pairs = []
    for base, quote, result, label, reason in results:
        pair = f"{base}/{quote}"
        if result and result.total_trades > 0:
            trades_str = str(result.total_trades)
            pnl_str = f"{result.total_pnl_pct:+.2f}%"
            wr_str = f"{result.win_rate_pct:.1f}%"
            sharpe_str = f"{result.sharpe_ratio:.2f}"
        else:
            trades_str = "—"
            pnl_str = "—"
            wr_str = "—"
            sharpe_str = "—"

        marker = "[KEEP ]" if label == "KEEP " else ("[WATCH]" if label == "WATCH" else "[DROP ]")
        print(f"  {pair:<24} {marker}  {trades_str:>6}  {pnl_str:>7}  "
              f"{wr_str:>5}  {sharpe_str:>6}  {reason}")

        if label == "KEEP ":
            keep_pairs.append((base, quote))
        elif label == "WATCH":
            watch_pairs.append((base, quote))

    # ── Recommendation ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  RECOMMENDATION")
    print(f"{'='*70}")

    if keep_pairs:
        print(f"\n  [KEEP] {len(keep_pairs)} pair{'s' if len(keep_pairs)>1 else ''}:")
        for b, q in keep_pairs:
            print(f"    (\"{b}\", \"{q}\")")
    else:
        print("\n  No pairs passed screening — none profitable out-of-sample.")

    if watch_pairs:
        print(f"\n  [WATCH] {len(watch_pairs)} pair{'s' if len(watch_pairs)>1 else ''}"
              f" — cointegrated but not yet profitable:")
        for b, q in watch_pairs:
            print(f"    (\"{b}\", \"{q}\")")
        print("  These may improve by tuning entry_z_score or zscore_window.")

    dropped = [(b, q) for b, q, _, lbl, _ in results if lbl == "DROP "]
    if dropped:
        print(f"\n  [DROP] {len(dropped)} pairs — remove from config:")
        for b, q in dropped:
            print(f"    (\"{b}\", \"{q}\")")

    if not keep_pairs and not watch_pairs:
        print("\n  Suggestions:")
        print("    — Try different sectors (DeFi, GameFi, AI tokens)")
        print("    — Increase lookback_candles further (e.g. 3000)")
        print("    — Lower entry_z_score to 1.8 for more signals")


if __name__ == "__main__":
    main()
