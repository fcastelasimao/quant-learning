"""
build_mixed_leverage_report.py
==============================
Generate research artifacts for capped multi-ETF leverage overlays.

This is intentionally research-only. It reuses the overlay engine's existing
multi-spec support and keeps the production portfolio/backtest path unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine import config
from engine.analytics import (
    apply_annual_fee,
    build_monthly_rebalanced_series,
    monthly_returns,
    summary_metrics,
    stress_period_metrics,
)
from engine.data import fetch_prices, get_price_provenance
from engine.leverage import OverlaySpec, apply_overlay_to_base
from research.rsi_leverage_overlay.build_leverage_comparison_report import (
    BASE_LABEL,
    BENCHMARK_LABEL,
    DEFAULT_OUTPUT_ROOT as SINGLE_OUTPUT_ROOT,
    DIY_FEE,
    _available_stress_periods,
    _clean_prices,
    _daily_series,
    _write_json,
    load_strategy,
)


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "results" / "mixed_leverage"


@dataclass(frozen=True)
class MixedOverlayCandidate:
    """Named set of overlay specs evaluated as one capped overlay sleeve."""

    name: str
    specs: tuple[OverlaySpec, ...]
    global_cap: float
    notes: str = ""


def default_mixed_candidates() -> list[MixedOverlayCandidate]:
    """Return conservative SPY+GLD candidates for first-pass research."""
    spy_default = OverlaySpec("SPY", entry_threshold=30.0, exit_threshold=50.0, overlay_weight=0.20)
    gld_default = OverlaySpec("GLD", entry_threshold=30.0, exit_threshold=50.0, overlay_weight=0.20)

    return [
        MixedOverlayCandidate(
            name="SPY+GLD default 20% total cap",
            specs=(spy_default, gld_default),
            global_cap=0.20,
            notes="CONTROL ONLY: both default RSI sleeves compete for the same 20% overlay budget.",
        ),
        MixedOverlayCandidate(
            name="SPY 10% + GLD 20% total cap",
            specs=(
                OverlaySpec("SPY", entry_threshold=30.0, exit_threshold=50.0, overlay_weight=0.10),
                gld_default,
            ),
            global_cap=0.20,
            notes="Same total cap, with gold favoured when both assets signal.",
        ),
        MixedOverlayCandidate(
            name="SPY selective + GLD default 25% cap",
            specs=(
                OverlaySpec("SPY", entry_threshold=22.0, exit_threshold=42.0, overlay_weight=0.25),
                gld_default,
            ),
            global_cap=0.25,
            notes="Tests the selective SPY threshold against the default gold sleeve.",
        ),
        MixedOverlayCandidate(
            name="SPY default + GLD robust 20% cap",
            specs=(
                spy_default,
                OverlaySpec("GLD", entry_threshold=24.0, exit_threshold=46.0, overlay_weight=0.20),
            ),
            global_cap=0.20,
            notes="Uses the GLD threshold region that survived OOS validation most cleanly.",
        ),
        MixedOverlayCandidate(
            name="SPY34/42 + GLD32/64 30% cap",
            specs=(
                OverlaySpec("SPY", entry_threshold=34.0, exit_threshold=42.0, overlay_weight=0.25),
                OverlaySpec("GLD", entry_threshold=32.0, exit_threshold=64.0, overlay_weight=0.30),
            ),
            global_cap=0.30,
            notes="Heatmap-inspired Calmar candidate: SPY single-ETF peak with GLD 32/64 region.",
        ),
        MixedOverlayCandidate(
            name="SPY34/42 + GLD32/64 25% cap",
            specs=(
                OverlaySpec("SPY", entry_threshold=34.0, exit_threshold=42.0, overlay_weight=0.25),
                OverlaySpec("GLD", entry_threshold=32.0, exit_threshold=64.0, overlay_weight=0.25),
            ),
            global_cap=0.25,
            notes="Broker-safe heatmap candidate with equal 25% sleeve weights.",
        ),
        MixedOverlayCandidate(
            name="SPY34/42 + GLD32/64 20% cap",
            specs=(
                OverlaySpec("SPY", entry_threshold=34.0, exit_threshold=42.0, overlay_weight=0.20),
                OverlaySpec("GLD", entry_threshold=32.0, exit_threshold=64.0, overlay_weight=0.20),
            ),
            global_cap=0.20,
            notes="Strict-pilot heatmap candidate with 20% cap and sleeves.",
        ),
        MixedOverlayCandidate(
            name="Older mixed best_calmar 30% cap",
            specs=(
                OverlaySpec("SPY", entry_threshold=34.0, exit_threshold=42.0, overlay_weight=0.25),
                OverlaySpec("GLD", entry_threshold=22.0, exit_threshold=50.0, overlay_weight=0.25),
            ),
            global_cap=0.30,
            notes="Fixed version of the older mixed best_calmar selector that passed structural OOS.",
        ),
        MixedOverlayCandidate(
            name="Best MaxDD Preservation 30% cap",
            specs=(
                OverlaySpec("SPY", entry_threshold=22.0, exit_threshold=40.0, overlay_weight=0.25),
                OverlaySpec("GLD", entry_threshold=22.0, exit_threshold=64.0, overlay_weight=0.30),
            ),
            global_cap=0.30,
            notes="Fixed version of latest IBKR-safe disciplined sweep drawdown-preservation candidate.",
        ),
        MixedOverlayCandidate(
            name="Best MaxDD Preservation 20% cap",
            specs=(
                OverlaySpec("SPY", entry_threshold=22.0, exit_threshold=40.0, overlay_weight=0.20),
                OverlaySpec("GLD", entry_threshold=22.0, exit_threshold=64.0, overlay_weight=0.20),
            ),
            global_cap=0.20,
            notes="Strict-pilot fixed version of the disciplined sweep drawdown-preservation candidate.",
        ),
    ]


def build_mixed_report_bundle(
    prices: pd.DataFrame,
    strategy_id: str,
    allocation: dict[str, float],
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    start_date: str = "2006-01-01",
    end_date: str | None = None,
    candidates: list[MixedOverlayCandidate] | None = None,
    apply_fees: bool = True,
    data_source: str = "yfinance",
    generated_at: datetime | None = None,
) -> Path:
    """Build a SPY/GLD mixed-leverage research bundle from an in-memory price frame."""
    generated_at = generated_at or datetime.now()
    end_date = end_date or date.today().strftime("%Y-%m-%d")
    candidates = candidates or default_mixed_candidates()

    price_provenance = get_price_provenance(prices)
    prices = _clean_prices(prices)
    diy_prices = prices[list(allocation)].dropna()
    if diy_prices.empty:
        raise ValueError("No date has complete price history for the selected allocation.")

    base_gross = build_monthly_rebalanced_series(diy_prices, allocation, start_value=100.0)
    base = apply_annual_fee(base_gross, DIY_FEE) if apply_fees else base_gross
    base = base.rename(BASE_LABEL)

    controls = [
        MixedOverlayCandidate(
            name="SPY default only",
            specs=(OverlaySpec("SPY", entry_threshold=30.0, exit_threshold=50.0, overlay_weight=0.20),),
            global_cap=0.20,
            notes="Single-sleeve control.",
        ),
        MixedOverlayCandidate(
            name="GLD default only",
            specs=(OverlaySpec("GLD", entry_threshold=30.0, exit_threshold=50.0, overlay_weight=0.20),),
            global_cap=0.20,
            notes="Single-sleeve control.",
        ),
    ]

    values = pd.DataFrame({BASE_LABEL: base})
    results = {}
    for candidate in [*controls, *candidates]:
        result = apply_overlay_to_base(
            base_values=base,
            prices=prices,
            specs=list(candidate.specs),
            global_cap=candidate.global_cap,
            execution_lag=1,
            name=candidate.name,
        )
        results[candidate.name] = (candidate, result)
        values[candidate.name] = result.value_series

    if "SPY" in prices:
        spy = prices["SPY"].dropna()
        values[BENCHMARK_LABEL] = spy / spy.iloc[0] * 100.0
    values = values.dropna(how="all")

    bundle_dir = Path(output_root) / f"{generated_at.strftime('%Y-%m-%d_%H-%M-%S')}_{strategy_id}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        bundle_dir / "manifest.json",
        _manifest(
            strategy_id=strategy_id,
            allocation=allocation,
            values=values,
            start_date=start_date,
            end_date=end_date,
            candidates=[*controls, *candidates],
            apply_fees=apply_fees,
            data_source=data_source,
            generated_at=generated_at,
        ),
    )
    _write_json(bundle_dir / "price_provenance.json", price_provenance)
    _daily_series(values).to_csv(bundle_dir / "daily_series.csv", index=False)
    monthly_returns(values).to_csv(bundle_dir / "monthly_returns.csv", index=False)
    summary_metrics(values, benchmark=BENCHMARK_LABEL).to_csv(bundle_dir / "summary_metrics.csv", index=False)
    stress_period_metrics(values, _available_stress_periods(values)).to_csv(
        bundle_dir / "stress_period_metrics.csv", index=False
    )
    _diagnostics(results).to_csv(bundle_dir / "overlay_diagnostics.csv", index=False)
    _signals(results).to_csv(bundle_dir / "signal_history.csv", index=False)
    _positions(results).to_csv(bundle_dir / "position_history.csv", index=False)
    _mixed_summary(values, results).to_csv(bundle_dir / "mixed_overlay_summary.csv", index=False)
    return bundle_dir


def build_from_yfinance(
    strategy_id: str,
    start_date: str,
    end_date: str,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    apply_fees: bool = True,
) -> Path:
    """Fetch prices and write the mixed-leverage research bundle."""
    payload = load_strategy(strategy_id)
    allocation = payload["allocation"]
    tickers = sorted(set(allocation) | {"SPY", "GLD"})
    prices = fetch_prices(tickers, start_date, end_date)
    return build_mixed_report_bundle(
        prices=prices,
        strategy_id=strategy_id,
        allocation=allocation,
        output_root=output_root,
        start_date=start_date,
        end_date=end_date,
        apply_fees=apply_fees,
        data_source=config.DATA_SOURCE,
    )


def _diagnostics(results: dict[str, tuple[MixedOverlayCandidate, object]]) -> pd.DataFrame:
    frames = []
    for label, (candidate, result) in results.items():
        frame = result.daily_diagnostics.reset_index()
        frame.insert(1, "Overlay Strategy", label)
        frame.insert(2, "Tickers", "+".join(spec.ticker for spec in candidate.specs))
        frame.insert(3, "Global Cap", candidate.global_cap)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _signals(results: dict[str, tuple[MixedOverlayCandidate, object]]) -> pd.DataFrame:
    frames = []
    for label, (candidate, result) in results.items():
        frame = result.signal_history.copy()
        frame.insert(1, "Overlay Strategy", label)
        frame.insert(2, "Global Cap", candidate.global_cap)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _positions(results: dict[str, tuple[MixedOverlayCandidate, object]]) -> pd.DataFrame:
    frames = []
    for label, (candidate, result) in results.items():
        wide = result.positions.copy()
        wide.index.name = "Date"
        positions = wide.reset_index().melt(
            id_vars="Date",
            var_name="Ticker",
            value_name="Applied Overlay Weight",
        )
        positions.insert(1, "Overlay Strategy", label)
        positions.insert(2, "Global Cap", candidate.global_cap)
        frames.append(positions)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _mixed_summary(values: pd.DataFrame,
                   results: dict[str, tuple[MixedOverlayCandidate, object]]) -> pd.DataFrame:
    metrics = summary_metrics(values, benchmark=BENCHMARK_LABEL)
    full = metrics[metrics["Window"] == "Full History"].set_index("Strategy")
    base = full.loc[BASE_LABEL]
    rows = [{
        "Strategy": BASE_LABEL,
        "Tickers": "BASE",
        "Global Cap": 0.0,
        "CAGR (%)": base["CAGR (%)"],
        "Calmar": base["Calmar"],
        "Max Drawdown (%)": base["Max Drawdown (%)"],
        "Incremental CAGR (%)": 0.0,
        "Incremental Calmar": 0.0,
        "Incremental MaxDD (%)": 0.0,
        "Active Days (%)": 0.0,
        "Both Active Days (%)": 0.0,
        "Average Overlay Exposure (%)": 0.0,
        "Max Overlay Exposure (%)": 0.0,
        "Average SPY Weight (%)": 0.0,
        "Average GLD Weight (%)": 0.0,
        "Overlay Return Contribution (%)": 0.0,
        "Notes": "Production reference.",
    }]

    for label, (candidate, result) in results.items():
        row = full.loc[label]
        exposure = result.daily_diagnostics["Overlay Exposure"]
        positions = result.positions
        both_active = 0.0
        if {"SPY", "GLD"} <= set(positions.columns):
            both_active = float(((positions["SPY"] > 0) & (positions["GLD"] > 0)).mean() * 100)
        rows.append({
            "Strategy": label,
            "Tickers": "+".join(spec.ticker for spec in candidate.specs),
            "Global Cap": candidate.global_cap,
            "CAGR (%)": row["CAGR (%)"],
            "Calmar": row["Calmar"],
            "Max Drawdown (%)": row["Max Drawdown (%)"],
            "Incremental CAGR (%)": round(float(row["CAGR (%)"] - base["CAGR (%)"]), 4),
            "Incremental Calmar": round(float(row["Calmar"] - base["Calmar"]), 4),
            "Incremental MaxDD (%)": round(float(row["Max Drawdown (%)"] - base["Max Drawdown (%)"]), 4),
            "Active Days (%)": round(float((exposure > 0).mean() * 100), 4),
            "Both Active Days (%)": round(both_active, 4),
            "Average Overlay Exposure (%)": round(float(exposure.mean() * 100), 4),
            "Max Overlay Exposure (%)": round(float(exposure.max() * 100), 4),
            "Average SPY Weight (%)": round(float(positions.get("SPY", pd.Series(0.0, index=positions.index)).mean() * 100), 4),
            "Average GLD Weight (%)": round(float(positions.get("GLD", pd.Series(0.0, index=positions.index)).mean() * 100), 4),
            "Overlay Return Contribution (%)": round(float(result.daily_diagnostics["Overlay Return"].sum() * 100), 4),
            "Notes": candidate.notes,
        })

    return pd.DataFrame(rows).sort_values(["Calmar", "CAGR (%)"], ascending=[False, False])


def _manifest(
    strategy_id: str,
    allocation: dict[str, float],
    values: pd.DataFrame,
    start_date: str,
    end_date: str,
    candidates: list[MixedOverlayCandidate],
    apply_fees: bool,
    data_source: str,
    generated_at: datetime,
) -> dict:
    return {
        "generated_at": generated_at.isoformat(timespec="seconds"),
        "strategy_id": strategy_id,
        "source_report_root": str(SINGLE_OUTPUT_ROOT),
        "start_date_requested": start_date,
        "end_date_requested": end_date,
        "start_date_actual": values.index.min().date().isoformat(),
        "end_date_actual": values.index.max().date().isoformat(),
        "allocation": allocation,
        "apply_fees": apply_fees,
        "data_source": data_source,
        "candidates": [
            {
                "name": candidate.name,
                "global_cap": candidate.global_cap,
                "notes": candidate.notes,
                "specs": [
                    {
                        "ticker": spec.ticker,
                        "indicator": spec.indicator,
                        "lookback": spec.lookback,
                        "entry_threshold": spec.entry_threshold,
                        "exit_threshold": spec.exit_threshold,
                        "overlay_weight": spec.overlay_weight,
                    }
                    for spec in candidate.specs
                ],
            }
            for candidate in candidates
        ],
        "artifacts": [
            "manifest.json",
            "price_provenance.json",
            "daily_series.csv",
            "monthly_returns.csv",
            "summary_metrics.csv",
            "stress_period_metrics.csv",
            "overlay_diagnostics.csv",
            "signal_history.csv",
            "position_history.csv",
            "mixed_overlay_summary.csv",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build mixed SPY/GLD leverage overlay artifacts.")
    parser.add_argument("--strategy-id", default=config.DEFAULT_STRATEGY)
    parser.add_argument("--start-date", default=config.BACKTEST_START)
    parser.add_argument("--end-date", default=config.BACKTEST_END)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--no-fees", action="store_true", help="Do not apply DIY ETF expense-ratio drag.")
    args = parser.parse_args()

    bundle = build_from_yfinance(
        strategy_id=args.strategy_id,
        start_date=args.start_date,
        end_date=args.end_date,
        output_root=args.output_root,
        apply_fees=not args.no_fees,
    )
    print(f"Mixed leverage bundle written to: {bundle}")


if __name__ == "__main__":
    main()
