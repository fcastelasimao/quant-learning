"""
build_leverage_comparison_report.py
===================================
Generate a research bundle for ETF-by-ETF RSI leverage comparisons.

The baseline portfolio remains unchanged. Each overlay strategy is an
independent variant: base portfolio plus one ETF RSI overlay at a time.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
    rolling_metrics,
    stress_period_metrics,
    summary_metrics,
)
from engine.data import fetch_prices, get_price_provenance
from engine.leverage import OverlaySpec, TRADING_DAYS_PER_YEAR, apply_overlay_to_base


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "results" / "leverage_comparison"
DIY_FEE = 0.0012
BASE_LABEL = "My Strategy (Base)"
BENCHMARK_LABEL = "S&P 500 (SPY)"

DEFAULT_ENTRY_GRID = tuple(float(value) for value in range(20, 37, 2))
DEFAULT_EXIT_GRID = tuple(float(value) for value in range(40, 71, 2))
DEFAULT_LEVERAGE_GRID = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)

STRESS_PERIODS = {
    "2018 Q4 equity selloff": ("2018-09-20", "2018-12-24"),
    "COVID crash": ("2020-02-19", "2020-03-23"),
    "COVID full shock": ("2020-02-19", "2020-12-31"),
    "2022 rate shock": ("2022-01-03", "2022-10-14"),
    "2022 inflation shock": ("2022-02-24", "2022-06-14"),
    "2022-2023 rising-rate cycle": ("2022-01-03", "2023-10-19"),
}


def overlay_strategy_name(ticker: str) -> str:
    """Display name for a one-ETF overlay strategy."""
    return f"My Strategy + {ticker} RSI Overlay"


def default_overlay_specs(allocation: dict[str, float],
                          tickers: list[str] | None = None) -> list[OverlaySpec]:
    """Create default RSI overlay specs for the requested allocation tickers."""
    selected = tickers or list(allocation)
    return [
        OverlaySpec(
            ticker=ticker,
            indicator="rsi",
            lookback=14,
            entry_threshold=30.0,
            exit_threshold=50.0,
            overlay_weight=0.20,
            enabled=True,
        )
        for ticker in selected
        if ticker in allocation
    ]


def load_strategy(strategy_id: str) -> dict:
    """Load one strategy payload from strategies.json."""
    with open(PROJECT_ROOT / "strategies.json", "r", encoding="utf-8") as handle:
        strategies = json.load(handle)["strategies"]
    if strategy_id not in strategies:
        raise KeyError(f"Unknown strategy_id '{strategy_id}'. Available: {sorted(strategies)}")
    return strategies[strategy_id]


def required_tickers(allocation: dict[str, float], overlay_specs: list[OverlaySpec]) -> list[str]:
    """Return all tickers needed for baseline, overlays, and benchmark."""
    overlay_tickers = {spec.ticker for spec in overlay_specs if spec.enabled}
    return sorted(set(allocation) | overlay_tickers | {"SPY"})


def build_report_bundle(prices: pd.DataFrame,
                        strategy_id: str,
                        allocation: dict[str, float],
                        output_root: str | Path = DEFAULT_OUTPUT_ROOT,
                        start_date: str = "2006-01-01",
                        end_date: str | None = None,
                        overlay_specs: list[OverlaySpec] | None = None,
                        overlay_tickers: list[str] | None = None,
                        global_cap: float = 0.20,
                        financing_cost_annual: float = 0.0,
                        apply_fees: bool = True,
                        data_source: str = "yfinance",
                        generated_at: datetime | None = None,
                        entry_grid: tuple[float, ...] = DEFAULT_ENTRY_GRID,
                        exit_grid: tuple[float, ...] = DEFAULT_EXIT_GRID,
                        leverage_grid: tuple[float, ...] = DEFAULT_LEVERAGE_GRID) -> Path:
    """Build all leverage comparison artifacts from an in-memory price DataFrame."""
    generated_at = generated_at or datetime.now()
    end_date = end_date or date.today().strftime("%Y-%m-%d")
    overlay_specs = overlay_specs or default_overlay_specs(allocation, overlay_tickers)

    price_provenance = get_price_provenance(prices)
    prices = _clean_prices(prices)
    diy_prices = prices[list(allocation)].dropna()
    if diy_prices.empty:
        raise ValueError("No date has complete price history for the selected allocation.")

    base_gross = build_monthly_rebalanced_series(diy_prices, allocation, start_value=100.0)
    base = apply_annual_fee(base_gross, DIY_FEE) if apply_fees else base_gross
    base = base.rename(BASE_LABEL)

    values = pd.DataFrame({BASE_LABEL: base})
    overlay_results = {}
    for spec in [s for s in overlay_specs if s.enabled]:
        label = overlay_strategy_name(spec.ticker)
        result = apply_overlay_to_base(
            base_values=base,
            prices=prices,
            specs=[spec],
            global_cap=global_cap,
            financing_cost_annual=financing_cost_annual,
            execution_lag=1,
            name=label,
        )
        overlay_results[label] = (spec, result)
        values[label] = result.value_series

    if "SPY" in prices:
        spy = prices["SPY"].dropna()
        values[BENCHMARK_LABEL] = spy / spy.iloc[0] * 100.0
    values = values.dropna(how="all")
    if values.empty:
        raise ValueError("No strategy value series could be built.")

    diagnostics = _overlay_diagnostics(overlay_results)
    signal_history = _signal_history(overlay_results)
    grid = threshold_grid(
        base_values=base,
        prices=prices,
        overlay_specs=overlay_specs,
        global_cap=global_cap,
        entry_grid=entry_grid,
        exit_grid=exit_grid,
        leverage_grid=leverage_grid,
        financing_cost_annual=financing_cost_annual,
    )
    capital = capital_view(base, prices, overlay_results, global_cap)
    overlay_summary = build_overlay_summary(values, diagnostics, capital)
    leverage_summary = build_leverage_summary(grid, overlay_summary)
    yearly_overlay = yearly_overlay_metrics(values, diagnostics)

    bundle_dir = _bundle_dir(output_root, generated_at, strategy_id)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    _write_json(bundle_dir / "manifest.json", _manifest(
        strategy_id=strategy_id,
        allocation=allocation,
        values=values,
        start_date=start_date,
        end_date=end_date,
        overlay_specs=overlay_specs,
        global_cap=global_cap,
        financing_cost_annual=financing_cost_annual,
        apply_fees=apply_fees,
        data_source=data_source,
        generated_at=generated_at,
        entry_grid=entry_grid,
        exit_grid=exit_grid,
        leverage_grid=leverage_grid,
    ))
    _write_json(bundle_dir / "price_provenance.json", price_provenance)
    _daily_series(values).to_csv(bundle_dir / "daily_series.csv", index=False)
    monthly_returns(values).to_csv(bundle_dir / "monthly_returns.csv", index=False)
    summary_metrics(values, benchmark=BENCHMARK_LABEL).to_csv(
        bundle_dir / "summary_metrics.csv", index=False
    )
    calendar_year_metrics(values).to_csv(bundle_dir / "calendar_year_metrics.csv", index=False)
    rolling_metrics(values, benchmark=BENCHMARK_LABEL).to_csv(
        bundle_dir / "rolling_metrics.csv", index=False
    )
    drawdown_events(values, top_n=5).to_csv(bundle_dir / "drawdown_events.csv", index=False)
    stress_period_metrics(values, _available_stress_periods(values)).to_csv(
        bundle_dir / "stress_period_metrics.csv", index=False
    )
    signal_history.to_csv(bundle_dir / "signal_history.csv", index=False)
    diagnostics.to_csv(bundle_dir / "overlay_diagnostics.csv", index=False)
    grid.to_csv(bundle_dir / "threshold_grid.csv", index=False)
    capital.to_csv(bundle_dir / "capital_view.csv", index=False)
    overlay_summary.to_csv(bundle_dir / "overlay_summary.csv", index=False)
    leverage_summary.to_csv(bundle_dir / "leverage_summary.csv", index=False)
    yearly_overlay.to_csv(bundle_dir / "yearly_overlay_metrics.csv", index=False)
    return bundle_dir


def build_from_yfinance(strategy_id: str,
                        start_date: str,
                        end_date: str,
                        output_root: str | Path = DEFAULT_OUTPUT_ROOT,
                        overlay_tickers: list[str] | None = None,
                        global_cap: float = 0.20,
                        financing_cost_annual: float = 0.0,
                        apply_fees: bool = True) -> Path:
    """Fetch prices and write a leverage comparison bundle."""
    payload = load_strategy(strategy_id)
    allocation = payload["allocation"]
    specs = default_overlay_specs(allocation, overlay_tickers)
    prices = fetch_prices(required_tickers(allocation, specs), start_date, end_date)
    return build_report_bundle(
        prices=prices,
        strategy_id=strategy_id,
        allocation=allocation,
        output_root=output_root,
        start_date=start_date,
        end_date=end_date,
        overlay_specs=specs,
        global_cap=global_cap,
        financing_cost_annual=financing_cost_annual,
        apply_fees=apply_fees,
        data_source=config.DATA_SOURCE,
    )


def threshold_grid(base_values: pd.Series,
                   prices: pd.DataFrame,
                   overlay_specs: list[OverlaySpec],
                   global_cap: float,
                   entry_grid: tuple[float, ...],
                   exit_grid: tuple[float, ...],
                   leverage_grid: tuple[float, ...],
                   financing_cost_annual: float = 0.0) -> pd.DataFrame:
    """Evaluate entry/exit threshold grids for each enabled overlay ticker."""
    rows = []
    spy = prices["SPY"].dropna() if "SPY" in prices else None
    base_values_named = base_values.rename(BASE_LABEL)
    base_frame = pd.DataFrame({BASE_LABEL: base_values_named})
    base_metrics = summary_metrics(base_frame)
    base_row = base_metrics[base_metrics["Strategy"] == BASE_LABEL].iloc[0].to_dict()

    for base_spec in [spec for spec in overlay_specs if spec.enabled]:
        for entry in entry_grid:
            for exit_ in exit_grid:
                if exit_ <= entry:
                    continue
                for leverage in leverage_grid:
                    spec = OverlaySpec(
                        ticker=base_spec.ticker,
                        indicator=base_spec.indicator,
                        lookback=base_spec.lookback,
                        entry_threshold=entry,
                        exit_threshold=exit_,
                        overlay_weight=leverage,
                        enabled=True,
                    )
                    label = overlay_strategy_name(spec.ticker)
                    effective_cap = max(global_cap, leverage)
                    result = apply_overlay_to_base(
                        base_values=base_values,
                        prices=prices,
                        specs=[spec],
                        global_cap=effective_cap,
                        financing_cost_annual=financing_cost_annual,
                        execution_lag=1,
                        name=label,
                    )
                    rf_result = apply_overlay_to_base(
                        base_values=base_values,
                        prices=prices,
                        specs=[spec],
                        global_cap=effective_cap,
                        financing_cost_annual=config.RISK_FREE_RATE,
                        execution_lag=1,
                        name=label,
                    )
                    values = pd.DataFrame({label: result.value_series})
                    rf_values = pd.DataFrame({label: rf_result.value_series})
                    if spy is not None and not spy.empty:
                        values[BENCHMARK_LABEL] = spy / spy.iloc[0] * 100.0
                        rf_values[BENCHMARK_LABEL] = spy / spy.iloc[0] * 100.0
                    metrics = summary_metrics(values, benchmark=BENCHMARK_LABEL)
                    rf_metrics = summary_metrics(rf_values, benchmark=BENCHMARK_LABEL)
                    row = metrics[metrics["Strategy"] == label].iloc[0].to_dict()
                    rf_row = rf_metrics[rf_metrics["Strategy"] == label].iloc[0].to_dict()
                    exposure = result.daily_diagnostics["Overlay Exposure"]
                    turnover = result.positions.diff().abs().sum(axis=1).sum()
                    avg_exposure = float(exposure.mean() * 100)
                    incremental_cagr = float(row["CAGR (%)"] - base_row["CAGR (%)"])
                    incremental_calmar = float(row["Calmar"] - base_row["Calmar"])
                    incremental_max_dd = float(row["Max Drawdown (%)"] - base_row["Max Drawdown (%)"])
                    rows.append({
                        "Ticker": spec.ticker,
                        "Strategy": label,
                        "Lookback": spec.lookback,
                        "Entry Threshold": entry,
                        "Exit Threshold": exit_,
                        "Overlay Weight": spec.overlay_weight,
                        "Overlay Weight (%)": round(spec.overlay_weight * 100, 4),
                        "Applied Global Cap": effective_cap,
                        "Active Days (%)": round(float((exposure > 0).mean() * 100), 4),
                        "Average Overlay Weight (%)": round(avg_exposure, 4),
                        "Max Overlay Weight (%)": round(float(exposure.max() * 100), 4),
                        "Turnover": round(float(turnover), 6),
                        "CAGR (%)": row["CAGR (%)"],
                        "Sharpe": row["Sharpe"],
                        "Calmar": row["Calmar"],
                        "Max Drawdown (%)": row["Max Drawdown (%)"],
                        "Worst Month (%)": row["Worst Month (%)"],
                        "Total Return (%)": row["Total Return (%)"],
                        "RF Opportunity Cost CAGR (%)": rf_row["CAGR (%)"],
                        "Incremental CAGR (%)": round(incremental_cagr, 4),
                        "Incremental Calmar": round(incremental_calmar, 4),
                        "Incremental MaxDD (%)": round(incremental_max_dd, 4),
                        "Incremental CAGR per Avg Overlay": round(incremental_cagr / avg_exposure, 6)
                        if abs(avg_exposure) > 1e-12 else np.nan,
                    })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["Ticker", "Calmar", "Max Drawdown (%)", "CAGR (%)"],
            ascending=[True, False, False, False],
        )
    return out


def capital_view(base_values: pd.Series,
                 prices: pd.DataFrame,
                 overlay_results: dict[str, tuple[OverlaySpec, object]],
                 global_cap: float) -> pd.DataFrame:
    """Return account-view and opportunity-cost-adjusted capital diagnostics."""
    rows = [_capital_row("BASE", BASE_LABEL, base_values, pd.Series(0.0, index=base_values.index), 0.0)]
    for label, (spec, result) in overlay_results.items():
        rf_result = apply_overlay_to_base(
            base_values=base_values,
            prices=prices,
            specs=[spec],
            global_cap=global_cap,
            financing_cost_annual=config.RISK_FREE_RATE,
            execution_lag=1,
            name=label,
        )
        rows.append(_capital_row(
            spec.ticker,
            label,
            result.value_series,
            result.daily_diagnostics["Overlay Exposure"],
            0.0,
        ))
        rows.append(_capital_row(
            spec.ticker,
            f"{label} - RF Opportunity Cost",
            rf_result.value_series,
            rf_result.daily_diagnostics["Overlay Exposure"],
            config.RISK_FREE_RATE,
        ))
    return pd.DataFrame(rows)


def build_overlay_summary(values: pd.DataFrame,
                          diagnostics: pd.DataFrame,
                          capital: pd.DataFrame) -> pd.DataFrame:
    """Rank baseline and ETF overlays by headline full-window metrics."""
    summary = summary_metrics(values, benchmark=BENCHMARK_LABEL)
    full = summary[summary["Window"] == "Full History"].copy()
    rows = []
    for _, row in full.iterrows():
        strategy = row["Strategy"]
        if strategy == BENCHMARK_LABEL:
            continue
        diag = diagnostics[diagnostics["Overlay Strategy"] == strategy] if not diagnostics.empty else pd.DataFrame()
        cap_rf = capital[capital["Strategy"] == f"{strategy} - RF Opportunity Cost"]
        rows.append({
            "Ticker": _ticker_from_strategy(strategy),
            "Strategy": strategy,
            "Total Return (%)": row["Total Return (%)"],
            "CAGR (%)": row["CAGR (%)"],
            "Sharpe": row["Sharpe"],
            "Calmar": row["Calmar"],
            "Max Drawdown (%)": row["Max Drawdown (%)"],
            "Volatility (%)": row["Volatility (%)"],
            "Worst Day (%)": row["Worst Day (%)"],
            "Worst Month (%)": row["Worst Month (%)"],
            "Active Days (%)": round(float((diag["Overlay Exposure"] > 0).mean() * 100), 4) if not diag.empty else 0.0,
            "Average Overlay Exposure (%)": round(float(diag["Overlay Exposure"].mean() * 100), 4) if not diag.empty else 0.0,
            "Overlay Return Contribution (%)": round(float(diag["Overlay Return"].sum() * 100), 4) if not diag.empty else 0.0,
            "RF Opportunity Cost CAGR (%)": cap_rf["CAGR (%)"].iloc[0] if not cap_rf.empty else np.nan,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["Calmar", "CAGR (%)"], ascending=[False, False])
    return out


def build_leverage_summary(threshold_grid_df: pd.DataFrame,
                           overlay_summary: pd.DataFrame) -> pd.DataFrame:
    """Summarise default and best leverage-grid rows for each ETF."""
    if threshold_grid_df.empty:
        return pd.DataFrame()

    rows = []
    base_row = overlay_summary[overlay_summary["Ticker"] == "BASE"]
    base_calmar = float(base_row["Calmar"].iloc[0]) if not base_row.empty else np.nan
    base_cagr = float(base_row["CAGR (%)"].iloc[0]) if not base_row.empty else np.nan
    base_max_dd = float(base_row["Max Drawdown (%)"].iloc[0]) if not base_row.empty else np.nan

    for ticker, group in threshold_grid_df.groupby("Ticker"):
        selectors = {
            "Best Calmar": group.sort_values(["Calmar", "CAGR (%)"], ascending=[False, False]).head(1),
            "Best CAGR": group.sort_values(["CAGR (%)", "Calmar"], ascending=[False, False]).head(1),
            "Best MaxDD Preservation": group.sort_values(["Max Drawdown (%)", "Calmar"], ascending=[False, False]).head(1),
        }
        default = group[
            (group["Overlay Weight"].round(10) == 0.20)
            & (group["Entry Threshold"] == 30.0)
            & (group["Exit Threshold"] == 50.0)
        ].head(1)
        selectors["Default 20% 30/50"] = default

        for label, selected in selectors.items():
            if selected.empty:
                continue
            row = selected.iloc[0].to_dict()
            row["Selection"] = label
            row["Base CAGR (%)"] = base_cagr
            row["Base Calmar"] = base_calmar
            row["Base Max Drawdown (%)"] = base_max_dd
            rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        first_cols = ["Ticker", "Selection", "Entry Threshold", "Exit Threshold", "Overlay Weight", "Overlay Weight (%)"]
        out = out[first_cols + [col for col in out.columns if col not in first_cols]]
        out = out.sort_values(["Ticker", "Selection"])
    return out


def yearly_overlay_metrics(values: pd.DataFrame,
                           diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Return yearly return/risk/overlay-activity metrics by strategy."""
    rows = []
    for strategy in values.columns:
        if strategy == BENCHMARK_LABEL:
            continue
        s = values[strategy].dropna().astype(float)
        if len(s) < 2:
            continue
        for year, chunk in s.groupby(s.index.year):
            if len(chunk) < 2:
                continue
            returns = chunk.pct_change().dropna()
            dd = (chunk / chunk.cummax() - 1.0) * 100
            period_return = float((chunk.iloc[-1] / chunk.iloc[0] - 1.0) * 100)
            max_dd = float(dd.min())
            vol = float(returns.std() * math.sqrt(TRADING_DAYS_PER_YEAR) * 100) if len(returns) else np.nan
            sharpe = float((returns.mean() / returns.std()) * math.sqrt(TRADING_DAYS_PER_YEAR)) if len(returns) and returns.std() > 1e-12 else np.nan
            diag = diagnostics[
                (diagnostics["Overlay Strategy"] == strategy)
                & (pd.to_datetime(diagnostics["Date"]).dt.year == year)
            ] if not diagnostics.empty else pd.DataFrame()
            rows.append({
                "Year": int(year),
                "Ticker": _ticker_from_strategy(strategy),
                "Strategy": strategy,
                "Annual Return (%)": round(period_return, 4),
                "Annual Volatility (%)": round(vol, 4),
                "Annual Sharpe": round(sharpe, 4) if pd.notna(sharpe) else np.nan,
                "Annual Calmar": round(period_return / abs(max_dd), 4) if abs(max_dd) > 1e-12 else np.nan,
                "Max Drawdown (%)": round(max_dd, 4),
                "Worst Day (%)": round(float(returns.min() * 100), 4) if len(returns) else np.nan,
                "Active Days": int((diag["Overlay Exposure"] > 0).sum()) if not diag.empty else 0,
                "Overlay Return Contribution (%)": round(float(diag["Overlay Return"].sum() * 100), 4) if not diag.empty else 0.0,
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ETF leverage overlay comparison artifacts.")
    parser.add_argument("--strategy-id", default=config.DEFAULT_STRATEGY)
    parser.add_argument("--start-date", default=config.BACKTEST_START)
    parser.add_argument("--end-date", default=config.BACKTEST_END)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--overlay-tickers", default="", help="Comma-separated allocation tickers; default is all.")
    parser.add_argument("--global-cap", type=float, default=0.20)
    parser.add_argument("--financing-cost-annual", type=float, default=0.0)
    parser.add_argument("--no-fees", action="store_true", help="Do not apply DIY ETF expense-ratio drag.")
    args = parser.parse_args()

    overlay_tickers = [t.strip().upper() for t in args.overlay_tickers.split(",") if t.strip()] or None
    bundle = build_from_yfinance(
        strategy_id=args.strategy_id,
        start_date=args.start_date,
        end_date=args.end_date,
        output_root=args.output_root,
        overlay_tickers=overlay_tickers,
        global_cap=args.global_cap,
        financing_cost_annual=args.financing_cost_annual,
        apply_fees=not args.no_fees,
    )
    print(f"Leverage comparison bundle written to: {bundle}")


def _clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    out = prices.copy()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out = out.sort_index().dropna(how="all").ffill()
    return out


def _overlay_diagnostics(overlay_results: dict[str, tuple[OverlaySpec, object]]) -> pd.DataFrame:
    frames = []
    for label, (spec, result) in overlay_results.items():
        frame = result.daily_diagnostics.reset_index()
        frame.insert(1, "Ticker", spec.ticker)
        frame.insert(2, "Overlay Strategy", label)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _signal_history(overlay_results: dict[str, tuple[OverlaySpec, object]]) -> pd.DataFrame:
    frames = []
    for label, (spec, result) in overlay_results.items():
        frame = result.signal_history.copy()
        frame.insert(1, "Overlay Strategy", label)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _bundle_dir(output_root: str | Path, generated_at: datetime, strategy_id: str) -> Path:
    timestamp = generated_at.strftime("%Y-%m-%d_%H-%M-%S")
    return Path(output_root) / f"{timestamp}_{strategy_id}"


def _daily_series(values: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy in values.columns:
        s = values[strategy].dropna()
        if s.empty:
            continue
        daily_return = s.pct_change() * 100
        drawdowns = drawdown_series(s)
        for date_idx, value in s.items():
            rows.append({
                "Date": date_idx,
                "Strategy": strategy,
                "Value": round(float(value), 8),
                "Indexed Value": round(float(value / s.iloc[0] * 100), 8),
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


def _capital_row(ticker: str,
                 strategy: str,
                 values: pd.Series,
                 overlay_exposure: pd.Series,
                 financing_cost_annual: float) -> dict:
    s = values.dropna().astype(float)
    exposure = overlay_exposure.reindex(s.index).fillna(0.0)
    return {
        "Ticker": ticker,
        "Strategy": strategy,
        "Start Date": s.index[0].date().isoformat(),
        "End Date": s.index[-1].date().isoformat(),
        "Final Value": round(float(s.iloc[-1]), 4),
        "Total Return (%)": round(float((s.iloc[-1] / s.iloc[0] - 1) * 100), 4),
        "CAGR (%)": round(_cagr(s), 4),
        "Average Overlay Exposure (%)": round(float(exposure.mean() * 100), 4),
        "Max Overlay Exposure (%)": round(float(exposure.max() * 100), 4),
        "Average Gross Exposure (%)": round(float((1.0 + exposure).mean() * 100), 4),
        "Max Gross Exposure (%)": round(float((1.0 + exposure).max() * 100), 4),
        "Active Days (%)": round(float((exposure > 0).mean() * 100), 4),
        "Financing / Opportunity Cost (%)": round(financing_cost_annual * 100, 4),
    }


def _cagr(series: pd.Series) -> float:
    years = max((series.index[-1] - series.index[0]).days / 365.25, 1 / 365.25)
    return float(((series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1) * 100)


def _ticker_from_strategy(strategy: str) -> str:
    if strategy == BASE_LABEL:
        return "BASE"
    if strategy.startswith("My Strategy + ") and " RSI Overlay" in strategy:
        return strategy.replace("My Strategy + ", "").replace(" RSI Overlay", "")
    return ""


def _manifest(strategy_id: str,
              allocation: dict[str, float],
              values: pd.DataFrame,
              start_date: str,
              end_date: str,
              overlay_specs: list[OverlaySpec],
              global_cap: float,
              financing_cost_annual: float,
              apply_fees: bool,
              data_source: str,
              generated_at: datetime,
              entry_grid: tuple[float, ...],
              exit_grid: tuple[float, ...],
              leverage_grid: tuple[float, ...]) -> dict:
    return {
        "generated_at": generated_at.isoformat(timespec="seconds"),
        "strategy_id": strategy_id,
        "date_range": {
            "requested_start": start_date,
            "requested_end": end_date,
            "actual_start": values.index.min().date().isoformat(),
            "actual_end": values.index.max().date().isoformat(),
        },
        "data_source": data_source,
        "pricing_model": config.PRICING_MODEL,
        "apply_fees": apply_fees,
        "fees": {"DIY": DIY_FEE if apply_fees else 0.0},
        "allocation": allocation,
        "overlay_specs": [spec.__dict__ for spec in overlay_specs],
        "global_overlay_cap": global_cap,
        "execution_lag_days": 1,
        "financing_cost_annual": financing_cost_annual,
        "opportunity_cost_reference": config.RISK_FREE_RATE,
        "threshold_grid": {
            "entry_thresholds": list(entry_grid),
            "exit_thresholds": list(exit_grid),
        },
        "leverage_grid": list(leverage_grid),
        "strategies": list(values.columns),
        "benchmark_strategy": BENCHMARK_LABEL,
        "plot_excluded_strategies": [BENCHMARK_LABEL],
        "artifacts": [
            "manifest.json",
            "price_provenance.json",
            "daily_series.csv",
            "monthly_returns.csv",
            "summary_metrics.csv",
            "calendar_year_metrics.csv",
            "rolling_metrics.csv",
            "drawdown_events.csv",
            "stress_period_metrics.csv",
            "signal_history.csv",
            "overlay_diagnostics.csv",
            "threshold_grid.csv",
            "capital_view.csv",
            "overlay_summary.csv",
            "leverage_summary.csv",
            "yearly_overlay_metrics.csv",
        ],
    }


def _write_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
