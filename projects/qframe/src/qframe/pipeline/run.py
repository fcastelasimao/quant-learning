"""
CLI entry point for the agentic pipeline.

Usage:
    conda activate qframe
    python3 -m qframe.pipeline.run --domain momentum
    python3 -m qframe.pipeline.run --domain mean_reversion --n 3
    python3 -m qframe.pipeline.run --domain volatility --n 5 --oos 2019-01-01
    python3 -m qframe.pipeline.run --domain all --n 5   # 5 iterations × all 5 domains = 25 total
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from qframe.data.loader import load_sp500_tickers, load_ohlcv
from qframe.pipeline.loop import PipelineLoop
from qframe.pipeline.models import ResearchSpec

_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_PRICE_CACHE = _REPO_ROOT / "data" / "processed" / "sp500_close.parquet"
_KB_PATH = _REPO_ROOT / "knowledge_base" / "qframe.db"

ALL_DOMAINS = ["momentum", "mean_reversion", "volatility", "quality", "value"]


def load_prices() -> pd.DataFrame:
    if _PRICE_CACHE.exists():
        print(f"Loading prices from cache: {_PRICE_CACHE}")
        return pd.read_parquet(_PRICE_CACHE)

    print("No price cache found — fetching from yfinance (this takes a few minutes)...")
    tickers = load_sp500_tickers()
    ohlcv = load_ohlcv(tickers, start="2010-01-01", end="2024-12-31")
    close = ohlcv["Close"].sort_index()
    close = close.loc[:, close.isna().mean() < 0.20].ffill(limit=5)
    _PRICE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    close.to_parquet(_PRICE_CACHE)
    print(f"Prices cached to {_PRICE_CACHE}")
    return close


def main():
    parser = argparse.ArgumentParser(description="qframe agentic research pipeline")
    parser.add_argument(
        "--domain", "-d",
        default="momentum",
        help=(
            "Factor domain(s) to explore. "
            "Single domain: momentum, mean_reversion, volatility, quality, value. "
            "All domains: 'all' (runs --n iterations per domain, prices loaded once)."
        ),
    )
    parser.add_argument(
        "--n", "-n",
        type=int,
        default=1,
        help="Number of iterations per domain (default: 1)",
    )
    parser.add_argument(
        "--oos",
        default="2018-01-01",
        help="OOS start date (default: 2018-01-01)",
    )
    args = parser.parse_args()

    prices = load_prices()

    loop = PipelineLoop(
        prices=prices,
        kb_path=_KB_PATH,
        oos_start=args.oos,
    )

    domains = ALL_DOMAINS if args.domain == "all" else [args.domain]
    total = len(domains) * args.n

    if len(domains) > 1:
        print(f"\n{'='*60}")
        print(f"Running {args.n} iteration(s) × {len(domains)} domains = {total} total")
        print(f"Domains: {', '.join(domains)}")
        print(f"{'='*60}\n")

    for i, domain in enumerate(domains, 1):
        if len(domains) > 1:
            print(f"\n{'─'*60}")
            print(f"  Domain {i}/{len(domains)}: {domain}")
            print(f"{'─'*60}")
        spec = ResearchSpec(factor_domain=domain)
        if args.n == 1:
            result = loop.run_iteration(spec)
            result.print_summary()
        else:
            loop.run_n(spec, n=args.n)

    if len(domains) > 1:
        print(f"\n{'='*60}")
        print(f"All done — {total} iterations complete across {len(domains)} domains.")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
