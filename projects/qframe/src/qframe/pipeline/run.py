"""
CLI entry point for the agentic pipeline.

Usage:
    conda activate qframe
    python3 -m qframe.pipeline.run --domain momentum
    python3 -m qframe.pipeline.run --domain mean_reversion --n 3
    python3 -m qframe.pipeline.run --domain volatility --n 5 --oos 2019-01-01
    python3 -m qframe.pipeline.run --domain all --n 5   # 5 iterations × all 5 domains = 25 total

Dirty-worktree guard
--------------------
The CLI refuses to start if the git worktree has uncommitted changes.
This ensures that git_hash logged to the KB is meaningful — a dirty hash
cannot be reproduced and silently makes backtest traceability worthless.

To bypass in emergencies (NOT recommended for production runs):
    QFRAME_ALLOW_DIRTY=1 ./run_pipeline.sh --domain momentum

Hold-out enforcement
--------------------
By default the price data fed to the pipeline is silently truncated one day
before HOLDOUT_START (2024-06-01).  This keeps the sealed hold-out pristine
for the final go/no-go evaluation of a pre-registered strategy.

To include hold-out data (final validation only — never during exploration):
    QFRAME_UNSEAL_HOLDOUT=1 ./run_pipeline.sh --domain momentum
"""
from __future__ import annotations

import os
import subprocess
import sys
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


# ---------------------------------------------------------------------------
# Worktree cleanliness guard
# ---------------------------------------------------------------------------

def _get_dirty_files() -> list[str]:
    """
    Return a list of uncommitted (modified/staged/untracked) files.

    Returns an empty list if git is not available or the cwd is not in a repo.
    Untracked files in `logs/` and `__pycache__/` are excluded — they are
    runtime artefacts, not research state.
    """
    _IGNORE_PREFIXES = ("logs/", "__pycache__/", ".pytest_cache/")
    try:
        raw = subprocess.check_output(
            ["git", "status", "--porcelain"],
            text=True,
            cwd=str(_REPO_ROOT),
        ).strip()
    except Exception:
        return []  # git not available / not a repo — skip the guard

    dirty = []
    for line in raw.splitlines():
        # git --porcelain lines: "XY filename" where XY are status codes
        filename = line[3:].strip()
        if not any(filename.startswith(p) for p in _IGNORE_PREFIXES):
            dirty.append(filename)
    return dirty


def _enforce_clean_worktree() -> None:
    """
    Abort the CLI run if the git worktree has uncommitted changes.

    The guard protects traceability: every backtest result in the KB stores
    a git_hash. A dirty worktree means that hash cannot be reproduced — the
    KB entry is effectively unauditable.

    Override: set env var QFRAME_ALLOW_DIRTY=1 to skip (not recommended).
    """
    if os.environ.get("QFRAME_ALLOW_DIRTY", "0").strip() not in ("", "0"):
        print("⚠  QFRAME_ALLOW_DIRTY is set — skipping worktree check.")
        return

    dirty = _get_dirty_files()
    if not dirty:
        return  # clean — proceed

    print("\n" + "═" * 62)
    print("✗  DIRTY WORKTREE — pipeline aborted")
    print("═" * 62)
    print(
        "The git worktree has uncommitted changes. Running the pipeline\n"
        "on a dirty repo makes the git_hash logged to the KB meaningless:\n"
        "you cannot reproduce the exact code that produced any result.\n"
    )
    print(f"  {len(dirty)} file(s) with uncommitted changes:")
    for f in dirty[:10]:
        print(f"    • {f}")
    if len(dirty) > 10:
        print(f"    … and {len(dirty) - 10} more.")
    print(
        "\nFix: commit or stash your changes before running the pipeline.\n"
        "  git add <files> && git commit -m '...'\n"
        "  # or to skip this check (emergency only):\n"
        "  QFRAME_ALLOW_DIRTY=1 ./run_pipeline.sh --domain momentum\n"
    )
    print("═" * 62 + "\n")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Price loading
# ---------------------------------------------------------------------------

def load_prices() -> pd.DataFrame:
    """
    Load adjusted close prices from the parquet cache (or yfinance fallback).

    Hold-out enforcement
    --------------------
    Unless QFRAME_UNSEAL_HOLDOUT=1 is set, prices are silently truncated to the
    day before HOLDOUT_START (2024-06-01).  This prevents any research evaluation
    from accidentally consuming the sealed hold-out period.
    """
    if _PRICE_CACHE.exists():
        print(f"Loading prices from cache: {_PRICE_CACHE}")
        prices = pd.read_parquet(_PRICE_CACHE)
    else:
        print("No price cache found — fetching from yfinance (this takes a few minutes)...")
        tickers = load_sp500_tickers()
        ohlcv = load_ohlcv(tickers, start="2010-01-01", end="2024-12-31")
        prices = ohlcv["Close"].sort_index()
        prices = prices.loc[:, prices.isna().mean() < 0.20].ffill(limit=5)
        _PRICE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        prices.to_parquet(_PRICE_CACHE)
        print(f"Prices cached to {_PRICE_CACHE}")

    # Hold-out guard: truncate prices before the sealed hold-out date unless unsealed
    if os.environ.get("QFRAME_UNSEAL_HOLDOUT", "0").strip() in ("", "0"):
        from qframe.factor_harness.walkforward import HOLDOUT_START
        cutoff = pd.Timestamp(HOLDOUT_START) - pd.Timedelta(days=1)
        n_before = len(prices)
        prices = prices.loc[:cutoff]
        n_after = len(prices)
        if n_before != n_after:
            print(
                f"  ⚠  Hold-out enforced: truncated {n_before - n_after} trading days "
                f"(data past {HOLDOUT_START} sealed). Set QFRAME_UNSEAL_HOLDOUT=1 to include."
            )
    else:
        print("  ⚠  QFRAME_UNSEAL_HOLDOUT is set — hold-out data included.")

    return prices


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    # --- Worktree guard: must run before any KB writes ---
    _enforce_clean_worktree()

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
