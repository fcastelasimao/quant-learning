"""
production_validation.py
========================
Build the canonical customer/bank validation artifact bundle.

This script intentionally delegates the analytics to
research.build_strategy_comparison_report so the generated package is the same
one reviewed by notebooks and tests.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from engine import config
from research.build_strategy_comparison_report import build_from_yfinance


DEFAULT_OUTPUT_ROOT = Path("results") / "production_validation"


def build_production_validation(strategy_id: str = config.DEFAULT_STRATEGY,
                                start_date: str = config.BACKTEST_START,
                                end_date: str = config.BACKTEST_END,
                                output_root: str | Path = DEFAULT_OUTPUT_ROOT) -> Path:
    """Generate the canonical production validation bundle."""
    bundle = build_from_yfinance(
        strategy_id=strategy_id,
        start_date=start_date,
        end_date=end_date,
        output_root=output_root,
        apply_fees=True,
        include_6040=True,
    )
    manifest_path = bundle / "production_validation_manifest.json"
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "strategy_id": strategy_id,
        "start_date": start_date,
        "end_date": end_date,
        "canonical_bundle": str(bundle),
        "required_review_files": [
            "manifest.json",
            "price_provenance.json",
            "summary_metrics.csv",
            "stress_period_metrics.csv",
            "risk_contribution.csv",
            "turnover_costs.csv",
            "drawdown_events.csv",
        ],
        "claim_register": "docs/claim_register.md",
        "client_pack": "docs/customer_pack.md",
        "bank_pack": "docs/bank_pack.md",
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical production validation artifacts.")
    parser.add_argument("--strategy-id", default=config.DEFAULT_STRATEGY)
    parser.add_argument("--start-date", default=config.BACKTEST_START)
    parser.add_argument("--end-date", default=config.BACKTEST_END)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()

    bundle = build_production_validation(
        strategy_id=args.strategy_id,
        start_date=args.start_date,
        end_date=args.end_date,
        output_root=args.output_root,
    )
    print(f"Production validation bundle written to: {bundle}")


if __name__ == "__main__":
    main()
