"""
build_strategy_comparison_report.py
===================================
Generate the bank-facing strategy comparison artifact bundle.

The Marimo notebook should read the files produced here; it should not fetch
prices, construct portfolios, apply fees, or compute analytics.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine import config
from engine.analytics import (
    apply_annual_fee,
    build_monthly_rebalanced_series,
    calendar_year_metrics,
    drawdown_events,
    drawdown_series,
    monthly_returns,
    risk_contribution,
    rolling_metrics,
    stress_period_metrics,
    summary_metrics,
    turnover_costs,
)
from engine.data import fetch_prices, get_price_provenance
from engine.leverage import OverlaySpec, apply_overlay_to_base


ALLW_LAUNCH_DATE = "2025-03-06"
DIY_FEE = 0.0012
ALLW_FEE = 0.0085
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "strategy_comparison"

DISPLAY_NAMES = {
    "DIY": "My Strategy (DIY)",
    "SPY": "S&P 500 (SPY)",
    "ALLW": "ALLW (Bridgewater)",
    "60/40": "60/40 (SPY/TLT)",
}

SPY_ONLY_CANDIDATE_NAME = "SPY 34/42 @ 30% cap"
SPY_ONLY_CANDIDATE_SPECS = (
    OverlaySpec("SPY", entry_threshold=34.0, exit_threshold=42.0, overlay_weight=0.30),
)
SPY_ONLY_CANDIDATE_GLOBAL_CAP = 0.30
GLD_ONLY_CANDIDATE_NAME = "GLD 32/64 @ 30% cap"
GLD_ONLY_CANDIDATE_SPECS = (
    OverlaySpec("GLD", entry_threshold=32.0, exit_threshold=64.0, overlay_weight=0.30),
)
GLD_ONLY_CANDIDATE_GLOBAL_CAP = 0.30
MIXED_CANDIDATE_NAME = "SPY 32/42 + GLD 36/52 @ 30% cap"
MIXED_CANDIDATE_SPECS = (
    OverlaySpec("SPY", entry_threshold=32.0, exit_threshold=42.0, overlay_weight=0.30),
    OverlaySpec("GLD", entry_threshold=36.0, exit_threshold=52.0, overlay_weight=0.30),
)
MIXED_CANDIDATE_GLOBAL_CAP = 0.30
LEVERAGE_CANDIDATES = (
    {
        "name": SPY_ONLY_CANDIDATE_NAME,
        "global_cap": SPY_ONLY_CANDIDATE_GLOBAL_CAP,
        "specs": SPY_ONLY_CANDIDATE_SPECS,
        "notes": "Strongest SPY-only 30% capped row from the single-ETF leverage grid.",
    },
    {
        "name": GLD_ONLY_CANDIDATE_NAME,
        "global_cap": GLD_ONLY_CANDIDATE_GLOBAL_CAP,
        "specs": GLD_ONLY_CANDIDATE_SPECS,
        "notes": "Strongest GLD-only 30% capped row from the single-ETF leverage grid.",
    },
    {
        "name": MIXED_CANDIDATE_NAME,
        "global_cap": MIXED_CANDIDATE_GLOBAL_CAP,
        "specs": MIXED_CANDIDATE_SPECS,
        "notes": "Strongest clean full-grid mixed row: SPY 32/42 @ 30%, GLD 36/52 @ 30%, cap 30%.",
    },
)
LEVERAGE_SIGNAL_HISTORY_COLUMNS = [
    "Candidate",
    "Date",
    "Ticker",
    "Indicator",
    "Lookback",
    "Entry Threshold",
    "Exit Threshold",
    "RSI",
    "Raw Signal",
    "Desired Overlay Weight",
    "Capped Overlay Weight",
    "Applied Overlay Weight",
]
LEVERAGE_SIGNAL_EVENT_COLUMNS = [
    "Candidate",
    "Date",
    "Ticker",
    "Event",
    "Entry Threshold",
    "Exit Threshold",
    "RSI",
    "Applied Overlay Weight",
]

STRESS_PERIODS = {
    "2018 Q4 equity selloff": ("2018-09-20", "2018-12-24"),
    "COVID crash": ("2020-02-19", "2020-03-23"),
    "COVID full shock": ("2020-02-19", "2020-12-31"),
    "2022 rate shock": ("2022-01-03", "2022-10-14"),
    "2022 inflation shock": ("2022-02-24", "2022-06-14"),
    "2022-2023 rising-rate cycle": ("2022-01-03", "2023-10-19"),
    "ALLW overlap": (ALLW_LAUNCH_DATE, date.today().strftime("%Y-%m-%d")),
}


def load_strategy(strategy_id: str) -> dict:
    """Load one strategy payload from strategies.json, accepting registry aliases."""
    with open(PROJECT_ROOT / "strategies.json", "r", encoding="utf-8") as handle:
        strategies = json.load(handle)["strategies"]
    canonical_id = config.resolve_strategy_id(strategy_id)
    if canonical_id not in strategies:
        raise KeyError(f"Unknown strategy_id '{strategy_id}'. Available: {sorted(strategies)}")
    return strategies[canonical_id]


def required_tickers(allocation: dict[str, float],
                     include_6040: bool = True,
                     include_leverage_candidate: bool = True) -> list[str]:
    """Return all tickers needed for the comparison bundle."""
    tickers = set(allocation) | {"SPY", "ALLW"}
    if include_6040:
        tickers.add("TLT")
    if include_leverage_candidate:
        tickers.update(
            spec.ticker
            for candidate in LEVERAGE_CANDIDATES
            for spec in candidate["specs"]
            if spec.enabled
        )
    return sorted(tickers)


def build_report_bundle(prices: pd.DataFrame,
                        strategy_id: str,
                        allocation: dict[str, float],
                        output_root: str | Path = DEFAULT_OUTPUT_ROOT,
                        start_date: str = "2006-01-01",
                        end_date: str | None = None,
                        apply_fees: bool = True,
                        include_6040: bool = True,
                        include_leverage_candidate: bool = True,
                        data_source: str = "yfinance",
                        generated_at: datetime | None = None,
                        transaction_cost_pct: float = 0.001) -> Path:
    """
    Build all presentation artifacts from an in-memory price DataFrame.

    This function is intentionally testable without network access.
    """
    generated_at = generated_at or datetime.now()
    end_date = end_date or date.today().strftime("%Y-%m-%d")
    price_provenance = get_price_provenance(prices)
    prices = _clean_prices(prices)

    diy_prices = prices[list(allocation)].dropna()
    if diy_prices.empty:
        raise ValueError("No date has complete price history for the selected DIY allocation.")

    values = pd.DataFrame(index=prices.index)
    diy = build_monthly_rebalanced_series(diy_prices, allocation, start_value=100.0)
    values["DIY"] = apply_annual_fee(diy, DIY_FEE) if apply_fees else diy
    leverage_signal_history_frames: list[pd.DataFrame] = []

    if include_leverage_candidate:
        for candidate in LEVERAGE_CANDIDATES:
            specs = tuple(candidate["specs"])
            overlay_tickers = [spec.ticker for spec in specs if spec.enabled]
            if not set(overlay_tickers).issubset(prices.columns):
                continue
            overlay_prices = prices[overlay_tickers].dropna(how="all")
            result = apply_overlay_to_base(
                values["DIY"],
                overlay_prices,
                list(specs),
                global_cap=float(candidate["global_cap"]),
                financing_cost_annual=config.RISK_FREE_RATE,
                name=str(candidate["name"]),
            )
            values[str(candidate["name"])] = result.value_series
            _history = result.signal_history.copy()
            _history.insert(0, "Candidate", str(candidate["name"]))
            leverage_signal_history_frames.append(_history)

    leverage_signal_history = (
        pd.concat(leverage_signal_history_frames, ignore_index=True)
        if leverage_signal_history_frames
        else pd.DataFrame(columns=LEVERAGE_SIGNAL_HISTORY_COLUMNS)
    )
    leverage_signal_events = _leverage_signal_events(leverage_signal_history)

    if "SPY" in prices:
        spy = prices["SPY"].dropna()
        values["SPY"] = spy / spy.iloc[0] * 100.0

    if "ALLW" in prices:
        allw_raw = prices["ALLW"].dropna()
        if not allw_raw.empty:
            allw = allw_raw / allw_raw.iloc[0] * 100.0
            values["ALLW"] = apply_annual_fee(allw, ALLW_FEE) if apply_fees else allw

    if include_6040 and {"SPY", "TLT"}.issubset(prices.columns):
        sixty_forty_prices = prices[["SPY", "TLT"]].dropna()
        if not sixty_forty_prices.empty:
            values["60/40"] = build_monthly_rebalanced_series(
                sixty_forty_prices,
                {"SPY": 0.60, "TLT": 0.40},
                start_value=100.0,
            )

    values = values.rename(columns=DISPLAY_NAMES).dropna(how="all")
    if values.empty:
        raise ValueError("No strategy value series could be built.")

    bundle_dir = _bundle_dir(output_root, generated_at, strategy_id)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    manifest = _manifest(
        strategy_id=strategy_id,
        allocation=allocation,
        values=values,
        start_date=start_date,
        end_date=end_date,
        apply_fees=apply_fees,
        include_6040=include_6040,
        include_leverage_candidate=include_leverage_candidate,
        data_source=data_source,
        generated_at=generated_at,
        transaction_cost_pct=transaction_cost_pct,
    )

    allw_name = DISPLAY_NAMES["ALLW"]
    allw_start = values[allw_name].first_valid_index() if allw_name in values else None
    windows = {"Full History": (None, None)}
    if allw_start is not None:
        windows["ALLW Overlap"] = (pd.Timestamp(allw_start), None)

    _write_json(bundle_dir / "manifest.json", manifest)
    _write_json(bundle_dir / "price_provenance.json", price_provenance)
    _daily_series(values, allw_start).to_csv(bundle_dir / "daily_series.csv", index=False)
    monthly_returns(values).to_csv(bundle_dir / "monthly_returns.csv", index=False)
    summary_metrics(values, benchmark=DISPLAY_NAMES["SPY"], windows=windows).to_csv(
        bundle_dir / "summary_metrics.csv", index=False
    )
    calendar_year_metrics(values).to_csv(bundle_dir / "calendar_year_metrics.csv", index=False)
    rolling_metrics(values, benchmark=DISPLAY_NAMES["SPY"]).to_csv(
        bundle_dir / "rolling_metrics.csv", index=False
    )
    drawdown_events(values, top_n=5).to_csv(bundle_dir / "drawdown_events.csv", index=False)
    stress_period_metrics(values, _available_stress_periods(values)).to_csv(
        bundle_dir / "stress_period_metrics.csv", index=False
    )
    risk_contribution(diy_prices, allocation).to_csv(bundle_dir / "risk_contribution.csv", index=False)
    turnover_costs(
        diy_prices,
        allocation,
        transaction_cost_pct=transaction_cost_pct,
        start_value=100.0,
    ).to_csv(bundle_dir / "turnover_costs.csv", index=False)
    leverage_signal_history.to_csv(bundle_dir / "leverage_signal_history.csv", index=False)
    leverage_signal_events.to_csv(bundle_dir / "leverage_signal_events.csv", index=False)

    return bundle_dir


def build_from_yfinance(strategy_id: str,
                        start_date: str,
                        end_date: str,
                        output_root: str | Path = DEFAULT_OUTPUT_ROOT,
                        apply_fees: bool = True,
                        include_6040: bool = True,
                        include_leverage_candidate: bool = True) -> Path:
    """Fetch prices and write a strategy comparison bundle."""
    payload = load_strategy(strategy_id)
    allocation = payload["allocation"]
    tickers = required_tickers(
        allocation,
        include_6040=include_6040,
        include_leverage_candidate=include_leverage_candidate,
    )
    prices = fetch_prices(tickers, start_date, end_date)
    return build_report_bundle(
        prices=prices,
        strategy_id=strategy_id,
        allocation=allocation,
        output_root=output_root,
        start_date=start_date,
        end_date=end_date,
        apply_fees=apply_fees,
        include_6040=include_6040,
        include_leverage_candidate=include_leverage_candidate,
        data_source="yfinance",
        transaction_cost_pct=config.TRANSACTION_COST_PCT,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build bank-facing strategy comparison artifacts.")
    parser.add_argument("--strategy-id", default=config.DEFAULT_STRATEGY)
    parser.add_argument("--start-date", default=config.BACKTEST_START)
    parser.add_argument("--end-date", default=config.BACKTEST_END)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--no-fees", action="store_true", help="Do not apply ETF expense ratios.")
    parser.add_argument("--no-6040", action="store_true", help="Exclude the 60/40 benchmark.")
    parser.add_argument(
        "--no-leverage-candidate",
        action="store_true",
        help="Exclude leverage overlay candidates.",
    )
    args = parser.parse_args()

    bundle = build_from_yfinance(
        strategy_id=args.strategy_id,
        start_date=args.start_date,
        end_date=args.end_date,
        output_root=args.output_root,
        apply_fees=not args.no_fees,
        include_6040=not args.no_6040,
        include_leverage_candidate=not args.no_leverage_candidate,
    )
    print(f"Strategy comparison bundle written to: {bundle}")


def _clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    out = prices.copy()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out = out.sort_index().dropna(how="all").ffill()
    return out


def _bundle_dir(output_root: str | Path, generated_at: datetime, strategy_id: str) -> Path:
    timestamp = generated_at.strftime("%Y-%m-%d_%H-%M-%S")
    return Path(output_root) / f"{timestamp}_{strategy_id}"


def _daily_series(values: pd.DataFrame, overlap_start: pd.Timestamp | None) -> pd.DataFrame:
    rows = []
    for strategy in values.columns:
        s = values[strategy].dropna()
        if s.empty:
            continue
        daily_return = s.pct_change() * 100
        drawdowns = drawdown_series(s)
        overlap_indexed = pd.Series(index=s.index, dtype=float)
        if overlap_start is not None:
            overlap = s.loc[s.index >= overlap_start]
            if not overlap.empty:
                overlap_indexed.loc[overlap.index] = overlap / overlap.iloc[0] * 100
        for date_idx, value in s.items():
            rows.append({
                "Date": date_idx,
                "Strategy": strategy,
                "Value": round(float(value), 8),
                "Indexed Value": round(float(value / s.iloc[0] * 100), 8),
                "Overlap Indexed Value": round(float(overlap_indexed.loc[date_idx]), 8)
                if pd.notna(overlap_indexed.loc[date_idx]) else pd.NA,
                "Daily Return (%)": round(float(daily_return.loc[date_idx]), 8)
                if pd.notna(daily_return.loc[date_idx]) else pd.NA,
                "Drawdown (%)": round(float(drawdowns.loc[date_idx]), 8),
            })
    return pd.DataFrame(rows)


def _available_stress_periods(values: pd.DataFrame) -> dict[str, tuple[str, str]]:
    start = values.index.min()
    end = values.index.max()
    periods = {}
    for name, (period_start, period_end) in STRESS_PERIODS.items():
        ps = pd.Timestamp(period_start)
        pe = pd.Timestamp(period_end)
        if pe >= start and ps <= end:
            periods[name] = (max(ps, start).date().isoformat(), min(pe, end).date().isoformat())
    periods["Full available history"] = (start.date().isoformat(), end.date().isoformat())
    return periods


def _leverage_signal_events(signal_history: pd.DataFrame) -> pd.DataFrame:
    """Return applied overlay entry/exit events from long-form signal history."""
    if signal_history.empty:
        return pd.DataFrame(columns=LEVERAGE_SIGNAL_EVENT_COLUMNS)

    rows = []
    for (_, ticker), group in signal_history.groupby(["Candidate", "Ticker"], sort=False):
        data = group.sort_values("Date").copy()
        active = data["Applied Overlay Weight"].fillna(0.0).astype(float) > 0
        previous = active.shift(1, fill_value=False)
        event_labels = pd.Series(pd.NA, index=data.index, dtype="object")
        event_labels.loc[active & ~previous] = "Entry"
        event_labels.loc[~active & previous] = "Exit"
        events = data.loc[event_labels.notna()].copy()
        for idx, event in events.iterrows():
            rows.append({
                "Candidate": event["Candidate"],
                "Date": event["Date"],
                "Ticker": event["Ticker"],
                "Event": event_labels.loc[idx],
                "Entry Threshold": event["Entry Threshold"],
                "Exit Threshold": event["Exit Threshold"],
                "RSI": event["RSI"],
                "Applied Overlay Weight": event["Applied Overlay Weight"],
            })

    return pd.DataFrame(rows, columns=LEVERAGE_SIGNAL_EVENT_COLUMNS)


def _manifest(strategy_id: str,
              allocation: dict[str, float],
              values: pd.DataFrame,
              start_date: str,
              end_date: str,
              apply_fees: bool,
              include_6040: bool,
              include_leverage_candidate: bool,
              data_source: str,
              generated_at: datetime,
              transaction_cost_pct: float) -> dict:
    fees = {
        "DIY": DIY_FEE if apply_fees else 0.0,
        "ALLW": ALLW_FEE if apply_fees else 0.0,
        "SPY": 0.0,
        "60/40": 0.0,
    }
    if include_leverage_candidate:
        for candidate in LEVERAGE_CANDIDATES:
            fees[str(candidate["name"])] = DIY_FEE if apply_fees else 0.0

    leverage_candidates = [
        {
            "name": str(candidate["name"]),
            "global_cap": float(candidate["global_cap"]),
            "financing_cost_annual": config.RISK_FREE_RATE,
            "notes": str(candidate["notes"]),
            "specs": [spec.__dict__ for spec in candidate["specs"]],
        }
        for candidate in LEVERAGE_CANDIDATES
    ] if include_leverage_candidate else []

    return {
        "generated_at": generated_at.isoformat(timespec="seconds"),
        "strategy_id": strategy_id,
        "strategy_label": "My Strategy (DIY)",
        "date_range": {
            "requested_start": start_date,
            "requested_end": end_date,
            "actual_start": values.index.min().date().isoformat(),
            "actual_end": values.index.max().date().isoformat(),
        },
        "data_source": data_source,
        "pricing_model": config.PRICING_MODEL,
        "apply_fees": apply_fees,
        "fees": fees,
        "transaction_cost_pct": transaction_cost_pct,
        "include_6040": include_6040,
        "include_leverage_candidate": include_leverage_candidate,
        "leverage_candidate": leverage_candidates[0] if leverage_candidates else None,
        "leverage_candidates": leverage_candidates,
        "full_grid_top_leaderboard_row": {
            "candidate": MIXED_CANDIDATE_NAME,
            "source": "results/mixed_leverage_full_grid_oos/*/structural_full_grid_leaderboard.csv",
            "rank": 1,
            "spy_rule": "32/42 @ 30%",
            "gld_rule": "36/52 @ 30%",
            "global_cap": MIXED_CANDIDATE_GLOBAL_CAP,
        } if include_leverage_candidate else None,
        "allocation": allocation,
        "strategies": list(values.columns),
        "artifacts": [
            "manifest.json",
            "daily_series.csv",
            "monthly_returns.csv",
            "summary_metrics.csv",
            "calendar_year_metrics.csv",
            "rolling_metrics.csv",
            "drawdown_events.csv",
            "stress_period_metrics.csv",
            "risk_contribution.csv",
            "turnover_costs.csv",
            "leverage_signal_history.csv",
            "leverage_signal_events.csv",
            "price_provenance.json",
        ],
    }


def _write_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
