"""
production_validation.py
========================
Build the canonical customer/bank validation artifact bundle.

This script intentionally delegates the headline analytics to
research.build_strategy_comparison_report so the generated package is the same
one reviewed by notebooks and tests.

D.17 addendum
-------------
On top of the comparison bundle it writes a tax-aware addendum:
  * rebalance_events.csv  — one row per rebalance/tax event (from the tax engine)
  * tax_summary.csv       — annual ST/LT/dividend/§1256 tax breakdown
  * tax_monthly_series.csv — month-end after-tax value + cumulative tax
  * tax_addendum_manifest.json — tax_regime + rebalance_policy blocks

The addendum uses the FMP total-return prices + central dividend store and the
FIFO lot selector by default — i.e. what is actually achievable on Alpaca
(see research/tax_drift_trigger/findings_alpaca_lot_selection.md). tax_optimal is available as a
research counterfactual via --lot-selector.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from engine import config
from engine.backtest import RebalancePolicy
from engine.data import fetch_dividends, fetch_prices, get_price_provenance
from engine.lot_ledger import LotSelector
from engine.tax import TaxRegime
from engine.tax_backtest import run_tax_aware_backtest
from research.production_validation.build_strategy_comparison_report import build_from_yfinance


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "results"


def _resolve_policy(spec: str) -> RebalancePolicy:
    """Parse a policy spec like 'monthly', 'drift_absolute:0.05', 'drift_relative:0.2'."""
    spec = spec.strip().lower()
    if spec in ("monthly", "monthly_unconditional"):
        return RebalancePolicy.monthly_unconditional()
    if ":" in spec:
        kind, val = spec.split(":", 1)
        v = float(val)
        if kind == "drift_absolute":
            return RebalancePolicy.drift_absolute(v)
        if kind == "drift_relative":
            return RebalancePolicy.drift_relative(v)
        if kind == "monthly_check_then_drift":
            return RebalancePolicy.monthly_check_then_drift(v)
    raise ValueError(f"Unrecognized rebalance policy spec: {spec!r}")


def build_tax_addendum(
    bundle: Path,
    strategy_id: str,
    start_date: str,
    end_date: str,
    *,
    regime_name: str = "us",
    lot_selector: str = "fifo",
    policy_spec: str = "monthly",
    transaction_cost_pct: float = 0.001,
) -> Path:
    """Write tax-aware artifacts into an existing bundle directory (D.17)."""
    canonical = config.resolve_strategy_id(strategy_id)
    allocation = {t: float(w) for t, w in config.load_strategy(strategy_id)["allocation"].items()}

    prices = fetch_prices(list(allocation), start_date, end_date)
    dividends = fetch_dividends(list(allocation), start_date, end_date)
    regime = TaxRegime.us() if regime_name == "us" else TaxRegime.none()
    policy = _resolve_policy(policy_spec)

    res = run_tax_aware_backtest(
        prices, allocation,
        regime=regime, rebalance_policy=policy,
        lot_selector=LotSelector.coerce(lot_selector), dividends=dividends,
        transaction_cost_pct=transaction_cost_pct,
    )

    res.events.to_csv(bundle / "rebalance_events.csv", index=False)
    res.tax_summary.to_csv(bundle / "tax_summary.csv", index=False)
    res.monthly.to_csv(bundle / "tax_monthly_series.csv")

    alt_regime = TaxRegime.none() if regime_name == "us" else TaxRegime.us()
    alt_res = run_tax_aware_backtest(
        prices, allocation,
        regime=alt_regime, rebalance_policy=policy,
        lot_selector=LotSelector.coerce(lot_selector), dividends=dividends,
        transaction_cost_pct=transaction_cost_pct,
    )
    regime_cmp = pd.DataFrame({
        "US Value": res.monthly["Value"] if regime_name == "us" else alt_res.monthly["Value"],
        "ISA Value": alt_res.monthly["Value"] if regime_name == "us" else res.monthly["Value"],
    }, index=res.monthly.index)
    regime_cmp.index.name = "Date"
    regime_cmp.to_csv(bundle / "tax_regime_comparison.csv")

    final_value = float(res.monthly["Value"].iloc[-1])
    total_tax = float(res.monthly["Cumulative Tax Paid"].iloc[-1])
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "strategy_id": canonical,
        "start_date": start_date,
        "end_date": end_date,
        "tax_regime": {
            "name": regime.name,
            "apply_niit": getattr(regime, "apply_niit", None),
            "allow_loss_offset": getattr(regime, "allow_loss_offset", None),
            "rates_source": "engine/tax_rates_us.yaml",
            "asset_class_overrides": {
                "GLD": "collectible (28% LT cap)",
                "GLDM": "collectible (28% LT cap)",
                "GSG": "section_1256 (60/40, year-end mark-to-market)",
            },
        },
        "rebalance_policy": {
            "label": policy.label,
            "mode": policy.mode,
            "relative_threshold": policy.relative_threshold,
            "absolute_threshold": policy.absolute_threshold,
        },
        "lot_selector": {
            "name": res.selector,
            "note": ("fifo = Alpaca broker reality; tax_optimal/hifo are research "
                     "counterfactuals (Alpaca cannot select lots on order). "
                     "See research/tax_drift_trigger/findings_alpaca_lot_selection.md."),
        },
        "transaction_cost_pct": transaction_cost_pct,
        "results": {
            "final_after_tax_value": round(final_value, 2),
            "cumulative_tax_paid": round(total_tax, 2),
            "rebalances": int(res.monthly["Rebalanced"].sum()),
        },
        "price_provenance": get_price_provenance(prices),
        "artifacts": ["rebalance_events.csv", "tax_summary.csv", "tax_monthly_series.csv",
                      "tax_regime_comparison.csv"],
    }
    (bundle / "tax_addendum_manifest.json").write_text(json.dumps(manifest, indent=2))
    return bundle


def build_production_validation(strategy_id: str = config.DEFAULT_STRATEGY,
                                start_date: str = config.BACKTEST_START,
                                end_date: str = config.BACKTEST_END,
                                output_root: str | Path = DEFAULT_OUTPUT_ROOT,
                                *,
                                with_tax_addendum: bool = True,
                                tax_regime: str = "us",
                                lot_selector: str = "fifo",
                                rebalance_policy: str = "monthly",
                                transaction_cost_pct: float = 0.001) -> Path:
    """Generate the canonical production validation bundle (+ optional tax addendum)."""
    bundle = build_from_yfinance(
        strategy_id=strategy_id,
        start_date=start_date,
        end_date=end_date,
        output_root=output_root,
        apply_fees=True,
        include_6040=True,
    )
    required = [
        "manifest.json",
        "price_provenance.json",
        "summary_metrics.csv",
        "stress_period_metrics.csv",
        "risk_contribution.csv",
        "turnover_costs.csv",
        "drawdown_events.csv",
    ]

    if with_tax_addendum:
        try:
            build_tax_addendum(
                bundle, strategy_id, start_date, end_date,
                regime_name=tax_regime, lot_selector=lot_selector,
                policy_spec=rebalance_policy,
                transaction_cost_pct=transaction_cost_pct,
            )
            required += ["rebalance_events.csv", "tax_summary.csv",
                         "tax_monthly_series.csv", "tax_regime_comparison.csv",
                         "tax_addendum_manifest.json"]
        except Exception as exc:  # noqa: BLE001 - addendum needs central FMP store
            print(f"Warning: tax addendum skipped ({exc}).")

    manifest_path = bundle / "production_validation_manifest.json"
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "strategy_id": strategy_id,
        "start_date": start_date,
        "end_date": end_date,
        "canonical_bundle": str(bundle),
        "required_review_files": required,
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
    parser.add_argument("--no-tax-addendum", action="store_true")
    parser.add_argument("--tax-regime", choices=("us", "none"), default="us")
    parser.add_argument("--lot-selector", choices=("fifo", "hifo", "tax_optimal"), default="fifo")
    parser.add_argument("--rebalance-policy", default="monthly",
                        help="monthly | drift_absolute:0.05 | drift_relative:0.2")
    parser.add_argument("--transaction-cost-pct", type=float, default=0.001)
    args = parser.parse_args()

    bundle = build_production_validation(
        strategy_id=args.strategy_id,
        start_date=args.start_date,
        end_date=args.end_date,
        output_root=args.output_root,
        with_tax_addendum=not args.no_tax_addendum,
        tax_regime=args.tax_regime,
        lot_selector=args.lot_selector,
        rebalance_policy=args.rebalance_policy,
        transaction_cost_pct=args.transaction_cost_pct,
    )
    print(f"Production validation bundle written to: {bundle}")


if __name__ == "__main__":
    main()
