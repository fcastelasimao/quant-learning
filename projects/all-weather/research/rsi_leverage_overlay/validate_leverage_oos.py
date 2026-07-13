"""
validate_leverage_oos.py
========================
Out-of-sample validation for RSI ETF leverage overlays.

The runner selects overlay rules using only pre-split data and evaluates the
selected rules on the corresponding OOS window. It is research-only and does
not change production portfolio logic.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine import config
from engine.analytics import (
    MONTH_END,
    apply_annual_fee,
    build_monthly_rebalanced_series,
    stress_period_metrics,
    summary_metrics,
)
from engine.data import fetch_prices, get_price_provenance
from engine.leverage import (
    TRADING_DAYS_PER_YEAR,
    OverlaySpec,
    apply_overlay_to_base,
    compute_rsi,
    generate_hysteresis_signal,
)
from research.rsi_leverage_overlay.build_leverage_comparison_report import (
    BASE_LABEL,
    DEFAULT_ENTRY_GRID,
    DEFAULT_EXIT_GRID,
    DEFAULT_LEVERAGE_GRID,
    DIY_FEE,
    STRESS_PERIODS,
    _clean_prices,
    _daily_series,
    _write_json,
    default_overlay_specs,
    load_strategy,
    overlay_strategy_name,
    required_tickers,
)


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "results" / "leverage_oos_validation"
OOS_SPLITS = ("2018-01-01", "2020-01-01", "2022-01-01")
GLD_EXTENDED_LEVERAGE_GRID = tuple(round(x / 100, 4) for x in range(15, 101, 5))
SELECTORS = (
    "default_30_50_20",
    "best_calmar",
    "best_maxdd_preservation",
    "best_cagr_with_maxdd_guard",
    "robust_calmar_region",
)


def build_oos_validation_bundle(
    prices: pd.DataFrame,
    strategy_id: str,
    allocation: dict[str, float],
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    start_date: str = "2006-01-01",
    end_date: str | None = None,
    splits: tuple[str, ...] = OOS_SPLITS,
    overlay_tickers: list[str] | None = None,
    entry_grid: tuple[float, ...] = DEFAULT_ENTRY_GRID,
    exit_grid: tuple[float, ...] = DEFAULT_EXIT_GRID,
    leverage_grid: tuple[float, ...] = DEFAULT_LEVERAGE_GRID,
    gld_leverage_grid: tuple[float, ...] = GLD_EXTENDED_LEVERAGE_GRID,
    apply_fees: bool = True,
    generated_at: datetime | None = None,
    data_source: str = "yfinance",
) -> Path:
    """Build an OOS validation bundle from an in-memory price frame."""
    generated_at = generated_at or datetime.now()
    end_date = end_date or date.today().strftime("%Y-%m-%d")
    prices = _clean_prices(prices)
    price_provenance = get_price_provenance(prices)
    specs = default_overlay_specs(allocation, overlay_tickers)

    all_is_grid: list[pd.DataFrame] = []
    selected_rules: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    daily_frames: list[pd.DataFrame] = []
    signal_frames: list[pd.DataFrame] = []
    diagnostic_frames: list[pd.DataFrame] = []
    stress_frames: list[pd.DataFrame] = []
    episode_rows: list[dict[str, object]] = []

    for split in splits:
        split_started = time.perf_counter()
        print(f"[leverage-oos] Split {split[:4]}: building IS/OOS windows...", flush=True)
        split_ts = pd.Timestamp(split)
        is_prices = prices.loc[prices.index < split_ts]
        oos_prices = prices.loc[prices.index >= split_ts]
        if is_prices.empty or oos_prices.empty:
            continue

        is_base = _base_series(is_prices, allocation, apply_fees)
        oos_base = _base_series(oos_prices, allocation, apply_fees)
        if is_base.empty or oos_base.empty:
            continue

        is_base_metrics = _strategy_metrics(is_base, BASE_LABEL)
        oos_base_metrics = _strategy_metrics(oos_base, BASE_LABEL)

        base_daily = _daily_series(pd.DataFrame({BASE_LABEL: oos_base}))
        base_daily.insert(0, "Selector", "base")
        base_daily.insert(0, "Ticker", "BASE")
        base_daily.insert(0, "Split", split[:4])
        base_daily.insert(1, "OOS Start", split)
        daily_frames.append(base_daily)

        split_grid = _split_threshold_grid(
            base_values=is_base,
            prices=is_prices,
            specs=specs,
            entry_grid=entry_grid,
            exit_grid=exit_grid,
            leverage_grid=leverage_grid,
            gld_leverage_grid=gld_leverage_grid,
        )
        if split_grid.empty:
            continue
        print(
            f"[leverage-oos] Split {split[:4]}: IS grid complete "
            f"({len(split_grid):,} rows) in {time.perf_counter() - split_started:.1f}s",
            flush=True,
        )
        split_grid.insert(0, "Split", split[:4])
        split_grid.insert(1, "OOS Start", split)
        split_grid.insert(2, "IS Start Date", is_base.index[0].date().isoformat())
        split_grid.insert(3, "IS End Date", is_base.index[-1].date().isoformat())
        all_is_grid.append(split_grid)

        for ticker, ticker_grid in split_grid.groupby("Ticker"):
            rules = select_rules(ticker_grid, is_base_metrics)
            if rules.empty:
                continue
            rules = rules.drop(columns=["Split", "OOS Start", "IS Start Date", "IS End Date"], errors="ignore")
            rules.insert(0, "Split", split[:4])
            rules.insert(1, "OOS Start", split)
            rules.insert(2, "IS Start Date", is_base.index[0].date().isoformat())
            rules.insert(3, "IS End Date", is_base.index[-1].date().isoformat())
            selected_rules.append(rules)

            for _, rule in rules.iterrows():
                evaluation = _evaluate_oos_rule(
                    oos_base=oos_base,
                    oos_prices=oos_prices,
                    rule=rule,
                    base_metrics=oos_base_metrics,
                    split=split,
                )
                summary_rows.append(evaluation["summary"])
                daily_frames.append(evaluation["daily"])
                signal_frames.append(evaluation["signals"])
                diagnostic_frames.append(evaluation["diagnostics"])
                stress_frames.append(evaluation["stress"])
                episode_rows.extend(evaluation["episodes"])
        print(
            f"[leverage-oos] Split {split[:4]}: OOS selected-rule evaluation complete "
            f"in {time.perf_counter() - split_started:.1f}s",
            flush=True,
        )

    is_grid_df = pd.concat(all_is_grid, ignore_index=True) if all_is_grid else pd.DataFrame()
    selected_df = pd.concat(selected_rules, ignore_index=True) if selected_rules else pd.DataFrame()
    oos_summary = pd.DataFrame(summary_rows)
    pass_fail = build_pass_fail_summary(oos_summary)

    bundle_dir = Path(output_root) / f"{generated_at.strftime('%Y-%m-%d_%H-%M-%S')}_{strategy_id}"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        bundle_dir / "manifest.json",
        _manifest(
            strategy_id=strategy_id,
            allocation=allocation,
            start_date=start_date,
            end_date=end_date,
            prices=prices,
            splits=splits,
            specs=specs,
            entry_grid=entry_grid,
            exit_grid=exit_grid,
            leverage_grid=leverage_grid,
            gld_leverage_grid=gld_leverage_grid,
            apply_fees=apply_fees,
            generated_at=generated_at,
            data_source=data_source,
        ),
    )
    _write_json(bundle_dir / "price_provenance.json", price_provenance)
    is_grid_df.to_csv(bundle_dir / "is_threshold_grid.csv", index=False)
    selected_df.to_csv(bundle_dir / "selected_rules.csv", index=False)
    oos_summary.to_csv(bundle_dir / "oos_summary.csv", index=False)
    _concat_or_empty(daily_frames).to_csv(bundle_dir / "oos_daily_series.csv", index=False)
    _concat_or_empty(signal_frames).to_csv(bundle_dir / "oos_signal_history.csv", index=False)
    _concat_or_empty(diagnostic_frames).to_csv(bundle_dir / "oos_overlay_diagnostics.csv", index=False)
    _concat_or_empty(stress_frames).to_csv(bundle_dir / "oos_stress_metrics.csv", index=False)
    pd.DataFrame(episode_rows).to_csv(bundle_dir / "oos_trade_episodes.csv", index=False)
    pass_fail.to_csv(bundle_dir / "pass_fail_summary.csv", index=False)
    return bundle_dir


def build_from_yfinance(
    strategy_id: str,
    start_date: str,
    end_date: str,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    overlay_tickers: list[str] | None = None,
    apply_fees: bool = True,
) -> Path:
    """Fetch prices and write the leverage OOS validation bundle."""
    payload = load_strategy(strategy_id)
    allocation = payload["allocation"]
    specs = default_overlay_specs(allocation, overlay_tickers)
    prices = fetch_prices(required_tickers(allocation, specs), start_date, end_date)
    return build_oos_validation_bundle(
        prices=prices,
        strategy_id=strategy_id,
        allocation=allocation,
        output_root=output_root,
        start_date=start_date,
        end_date=end_date,
        overlay_tickers=overlay_tickers,
        apply_fees=apply_fees,
        data_source=config.DATA_SOURCE,
    )


def select_rules(grid: pd.DataFrame, base_metrics: dict[str, float]) -> pd.DataFrame:
    """Return one selected grid row per fixed selector for a single ETF."""
    if grid.empty:
        return pd.DataFrame()

    selections: list[pd.Series] = []
    default = grid[
        np.isclose(grid["Entry Threshold"], 30.0, atol=1e-9)
        & np.isclose(grid["Exit Threshold"], 50.0, atol=1e-9)
        & np.isclose(grid["Overlay Weight"], 0.20, atol=1e-9)
    ]
    if not default.empty:
        selections.append(_mark_selection(default.iloc[0], "default_30_50_20"))

    best_calmar = grid.sort_values(["Calmar", "CAGR (%)"], ascending=[False, False]).iloc[0]
    selections.append(_mark_selection(best_calmar, "best_calmar"))

    best_maxdd = grid.sort_values(["Max Drawdown (%)", "Calmar"], ascending=[False, False]).iloc[0]
    selections.append(_mark_selection(best_maxdd, "best_maxdd_preservation"))

    maxdd_floor = float(base_metrics["Max Drawdown (%)"]) - 1.0
    guarded = grid[grid["Max Drawdown (%)"] >= maxdd_floor]
    if guarded.empty:
        best_guard = grid.sort_values(["Max Drawdown (%)", "CAGR (%)"], ascending=[False, False]).iloc[0]
        selections.append(_mark_selection(
            best_guard,
            "best_cagr_with_maxdd_guard",
            warning="no row satisfied the 1pp MaxDD guard",
            guard_passed=False,
        ))
    else:
        best_guard = guarded.sort_values(["CAGR (%)", "Calmar"], ascending=[False, False]).iloc[0]
        selections.append(_mark_selection(best_guard, "best_cagr_with_maxdd_guard", guard_passed=True))

    robust = _robust_calmar_selection(grid)
    selections.append(_mark_selection(
        robust["row"],
        "robust_calmar_region",
        robust_avg_calmar=robust["avg_calmar"],
        robust_neighborhood_size=robust["size"],
    ))

    out = pd.DataFrame([s.to_dict() for s in selections])
    first_cols = [
        "Selector", "Ticker", "Entry Threshold", "Exit Threshold", "Overlay Weight",
        "Overlay Weight (%)", "Selection Warning", "IS MaxDD Guard Passed",
        "Robust Avg Calmar", "Robust Neighborhood Size",
    ]
    return out[first_cols + [c for c in out.columns if c not in first_cols]]


def build_pass_fail_summary(oos_summary: pd.DataFrame) -> pd.DataFrame:
    """Summarise split-level pass/fail rows into ETF/selector gates."""
    if oos_summary.empty:
        return pd.DataFrame()

    rows = []
    for (ticker, selector), group in oos_summary.groupby(["Ticker", "Selector"]):
        pass_count = int(group["Pass Split"].sum())
        maxdd_breach = bool((group["OOS MaxDD Delta (%)"] < -3.0).any())
        low_trade_count = int(group["Low Trade Count Flag"].sum())
        rows.append({
            "Ticker": ticker,
            "Selector": selector,
            "Splits Tested": int(len(group)),
            "Splits Passed": pass_count,
            "Low Trade Count Splits": low_trade_count,
            "Worst OOS Calmar Delta": round(float(group["OOS Calmar Delta"].min()), 4),
            "Worst OOS MaxDD Delta (%)": round(float(group["OOS MaxDD Delta (%)"].min()), 4),
            "Average OOS CAGR Delta (%)": round(float(group["OOS CAGR Delta (%)"].mean()), 4),
            "Overall Pass": bool(pass_count >= 2 and not maxdd_breach),
            "MaxDD Breach >3pp": maxdd_breach,
        })
    return pd.DataFrame(rows).sort_values(
        ["Overall Pass", "Splits Passed", "Worst OOS MaxDD Delta (%)"],
        ascending=[False, False, False],
    )


def _split_threshold_grid(
    base_values: pd.Series,
    prices: pd.DataFrame,
    specs: list[OverlaySpec],
    entry_grid: tuple[float, ...],
    exit_grid: tuple[float, ...],
    leverage_grid: tuple[float, ...],
    gld_leverage_grid: tuple[float, ...],
) -> pd.DataFrame:
    frames = []
    for spec in specs:
        started = time.perf_counter()
        ticker_grid = gld_leverage_grid if spec.ticker == "GLD" else leverage_grid
        frames.append(_fast_single_ticker_threshold_grid(
            base_values=base_values,
            prices=prices,
            spec=spec,
            entry_grid=entry_grid,
            exit_grid=exit_grid,
            leverage_grid=ticker_grid,
        ))
        print(
            f"[leverage-oos]   {spec.ticker} IS grid "
            f"({len(frames[-1]):,} rows) in {time.perf_counter() - started:.1f}s",
            flush=True,
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _fast_single_ticker_threshold_grid(
    base_values: pd.Series,
    prices: pd.DataFrame,
    spec: OverlaySpec,
    entry_grid: tuple[float, ...],
    exit_grid: tuple[float, ...],
    leverage_grid: tuple[float, ...],
) -> pd.DataFrame:
    """
    Fast grid evaluator for one-ETF overlays.

    The generic report builder uses apply_overlay_to_base for every grid row,
    which is convenient but expensive for OOS validation. Here RSI is computed
    once per ticker, the binary signal once per entry/exit pair, and leverage
    variants are simple vector multiplications.
    """
    if spec.ticker not in prices.columns:
        raise KeyError(f"Overlay ticker '{spec.ticker}' not found in prices.")

    base = base_values.dropna().astype(float).sort_index()
    clean_prices = prices.sort_index().astype(float).reindex(base.index)
    asset_returns = clean_prices[spec.ticker].pct_change().fillna(0.0)
    base_returns = base.pct_change().fillna(0.0)
    base_stats = _grid_metrics(base)
    rsi = compute_rsi(clean_prices[spec.ticker], spec.lookback)
    rows = []

    for entry in entry_grid:
        for exit_ in exit_grid:
            if exit_ <= entry:
                continue
            raw_signal = generate_hysteresis_signal(rsi, entry, exit_)
            lagged_signal = raw_signal.shift(1).fillna(0.0).astype(float)
            active_pct = float((lagged_signal > 0).mean() * 100)
            signal_turnover = float(lagged_signal.diff().abs().sum())
            for leverage in leverage_grid:
                position = lagged_signal * leverage
                overlay_returns = position * asset_returns
                strategy_returns = base_returns + overlay_returns
                values = base.iloc[0] * (1.0 + strategy_returns).cumprod()
                values.iloc[0] = base.iloc[0]

                daily_rf = (1.0 + config.RISK_FREE_RATE) ** (1 / TRADING_DAYS_PER_YEAR) - 1.0
                rf_values = base.iloc[0] * (1.0 + strategy_returns - position * daily_rf).cumprod()
                rf_values.iloc[0] = base.iloc[0]

                metrics = _grid_metrics(values)
                rf_metrics = _grid_metrics(rf_values)
                avg_exposure = float(position.mean() * 100)
                incremental_cagr = float(metrics["CAGR (%)"] - base_stats["CAGR (%)"])
                incremental_calmar = float(metrics["Calmar"] - base_stats["Calmar"])
                incremental_max_dd = float(metrics["Max Drawdown (%)"] - base_stats["Max Drawdown (%)"])
                rows.append({
                    "Ticker": spec.ticker,
                    "Strategy": overlay_strategy_name(spec.ticker),
                    "Lookback": spec.lookback,
                    "Entry Threshold": entry,
                    "Exit Threshold": exit_,
                    "Overlay Weight": leverage,
                    "Overlay Weight (%)": round(leverage * 100, 4),
                    "Applied Global Cap": max(leverage_grid),
                    "Active Days (%)": round(active_pct, 4),
                    "Average Overlay Weight (%)": round(avg_exposure, 4),
                    "Max Overlay Weight (%)": round(float(position.max() * 100), 4),
                    "Turnover": round(signal_turnover * leverage, 6),
                    "CAGR (%)": metrics["CAGR (%)"],
                    "Sharpe": metrics["Sharpe"],
                    "Calmar": metrics["Calmar"],
                    "Max Drawdown (%)": metrics["Max Drawdown (%)"],
                    "Worst Month (%)": metrics["Worst Month (%)"],
                    "Total Return (%)": metrics["Total Return (%)"],
                    "RF Opportunity Cost CAGR (%)": rf_metrics["CAGR (%)"],
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


def _evaluate_oos_rule(
    oos_base: pd.Series,
    oos_prices: pd.DataFrame,
    rule: pd.Series,
    base_metrics: dict[str, float],
    split: str,
) -> dict[str, object]:
    ticker = str(rule["Ticker"])
    selector = str(rule["Selector"])
    spec = OverlaySpec(
        ticker=ticker,
        lookback=int(rule["Lookback"]),
        entry_threshold=float(rule["Entry Threshold"]),
        exit_threshold=float(rule["Exit Threshold"]),
        overlay_weight=float(rule["Overlay Weight"]),
    )
    label = f"{ticker} {selector}"
    cap = max(float(rule["Overlay Weight"]), 0.20)
    result = apply_overlay_to_base(
        base_values=oos_base,
        prices=oos_prices,
        specs=[spec],
        global_cap=cap,
        execution_lag=1,
        name=label,
    )
    rf_result = apply_overlay_to_base(
        base_values=oos_base,
        prices=oos_prices,
        specs=[spec],
        global_cap=cap,
        financing_cost_annual=config.RISK_FREE_RATE,
        execution_lag=1,
        name=label,
    )

    overlay_metrics = _strategy_metrics(result.value_series, label)
    rf_metrics = _strategy_metrics(rf_result.value_series, label)
    diagnostics = result.daily_diagnostics.copy()
    episodes = trade_episodes(diagnostics, result.value_series, split, ticker, selector)
    active_days = int((diagnostics["Overlay Exposure"] > 0).sum())
    avg_exposure = float(diagnostics["Overlay Exposure"].mean() * 100)
    overlay_contribution = float(diagnostics["Overlay Return"].sum() * 100)
    pass_flags = split_pass_flags(base_metrics, overlay_metrics, rf_metrics, len(episodes))

    summary = {
        "Split": split[:4],
        "OOS Start": split,
        "Ticker": ticker,
        "Selector": selector,
        "Entry Threshold": spec.entry_threshold,
        "Exit Threshold": spec.exit_threshold,
        "Overlay Weight": spec.overlay_weight,
        "Overlay Weight (%)": round(spec.overlay_weight * 100, 4),
        "IS CAGR (%)": rule["CAGR (%)"],
        "IS Calmar": rule["Calmar"],
        "IS Max Drawdown (%)": rule["Max Drawdown (%)"],
        "IS Active Days (%)": rule["Active Days (%)"],
        "OOS Base CAGR (%)": base_metrics["CAGR (%)"],
        "OOS Base Calmar": base_metrics["Calmar"],
        "OOS Base Max Drawdown (%)": base_metrics["Max Drawdown (%)"],
        "OOS Overlay CAGR (%)": overlay_metrics["CAGR (%)"],
        "OOS Overlay Calmar": overlay_metrics["Calmar"],
        "OOS Overlay Max Drawdown (%)": overlay_metrics["Max Drawdown (%)"],
        "OOS RF Opportunity Cost CAGR (%)": rf_metrics["CAGR (%)"],
        "OOS CAGR Delta (%)": round(float(overlay_metrics["CAGR (%)"] - base_metrics["CAGR (%)"]), 4),
        "OOS Calmar Delta": round(float(overlay_metrics["Calmar"] - base_metrics["Calmar"]), 4),
        "OOS MaxDD Delta (%)": round(float(overlay_metrics["Max Drawdown (%)"] - base_metrics["Max Drawdown (%)"]), 4),
        "OOS Active Days": active_days,
        "OOS Active Days (%)": round(float((diagnostics["Overlay Exposure"] > 0).mean() * 100), 4),
        "OOS Average Overlay Exposure (%)": round(avg_exposure, 4),
        "OOS Overlay Return Contribution (%)": round(overlay_contribution, 4),
        "OOS Trade Episodes": int(len(episodes)),
        **pass_flags,
    }

    values = pd.DataFrame({label: result.value_series})
    daily = _daily_series(values)
    daily.insert(0, "Selector", selector)
    daily.insert(0, "Ticker", ticker)
    daily.insert(0, "Split", split[:4])
    daily.insert(1, "OOS Start", split)

    signals = result.signal_history.copy()
    signals.insert(0, "Selector", selector)
    signals.insert(0, "Overlay Strategy", label)
    signals.insert(0, "Split", split[:4])
    signals.insert(1, "OOS Start", split)

    diagnostics_out = diagnostics.reset_index()
    diagnostics_out.insert(0, "Selector", selector)
    diagnostics_out.insert(0, "Ticker", ticker)
    diagnostics_out.insert(0, "Overlay Strategy", label)
    diagnostics_out.insert(0, "Split", split[:4])
    diagnostics_out.insert(1, "OOS Start", split)

    stress_values = pd.DataFrame({BASE_LABEL: oos_base, label: result.value_series})
    stress = stress_period_metrics(stress_values, _available_stress_periods(stress_values))
    stress.insert(0, "Selector", selector)
    stress.insert(0, "Ticker", ticker)
    stress.insert(0, "Split", split[:4])
    stress.insert(1, "OOS Start", split)

    return {
        "summary": summary,
        "daily": daily,
        "signals": signals,
        "diagnostics": diagnostics_out,
        "stress": stress,
        "episodes": episodes,
    }


def split_pass_flags(
    base_metrics: dict[str, float],
    overlay_metrics: dict[str, float],
    rf_metrics: dict[str, float],
    episode_count: int,
) -> dict[str, object]:
    """Return split-level pass criteria and caveat flags."""
    calmar_pass = bool(overlay_metrics["Calmar"] > base_metrics["Calmar"])
    cagr_pass = bool(overlay_metrics["CAGR (%)"] >= base_metrics["CAGR (%)"])
    maxdd_pass = bool(overlay_metrics["Max Drawdown (%)"] >= base_metrics["Max Drawdown (%)"] - 1.0)
    rf_pass = bool(rf_metrics["CAGR (%)"] >= base_metrics["CAGR (%)"])
    low_trade_count = bool(episode_count < 3)
    metric_pass = bool(calmar_pass and cagr_pass and maxdd_pass and rf_pass)
    notes = []
    if low_trade_count:
        notes.append("low-trade-count")
    if not metric_pass:
        notes.append("failed-metric-gate")
    return {
        "Calmar Pass": calmar_pass,
        "CAGR Pass": cagr_pass,
        "MaxDD Pass": maxdd_pass,
        "RF Cost Pass": rf_pass,
        "Low Trade Count Flag": low_trade_count,
        "Pass Split": metric_pass,
        "Pass Notes": ", ".join(notes),
    }


def trade_episodes(
    diagnostics: pd.DataFrame,
    value_series: pd.Series,
    split: str,
    ticker: str,
    selector: str,
) -> list[dict[str, object]]:
    """Return consecutive active overlay periods."""
    if diagnostics.empty:
        return []
    active = diagnostics["Overlay Exposure"] > 0
    if not active.any():
        return []

    rows = []
    start = None
    episode = 0
    index = diagnostics.index
    for pos, date_idx in enumerate(index):
        if active.loc[date_idx] and start is None:
            start = date_idx
        is_last = pos == len(index) - 1
        if start is not None and ((not active.loc[date_idx]) or is_last):
            end = date_idx if active.loc[date_idx] and is_last else index[pos - 1]
            chunk = diagnostics.loc[start:end]
            values = value_series.loc[start:end]
            episode += 1
            rows.append({
                "Split": split[:4],
                "OOS Start": split,
                "Ticker": ticker,
                "Selector": selector,
                "Episode": episode,
                "Entry Date": start.date().isoformat(),
                "Exit Date": end.date().isoformat(),
                "Trading Days": int(len(chunk)),
                "Average Overlay Exposure (%)": round(float(chunk["Overlay Exposure"].mean() * 100), 4),
                "Overlay Return Contribution (%)": round(float(chunk["Overlay Return"].sum() * 100), 4),
                "Strategy Return (%)": round(float((values.iloc[-1] / values.iloc[0] - 1) * 100), 4)
                if len(values) >= 2 else 0.0,
            })
            start = None
    return rows


def _robust_calmar_selection(grid: pd.DataFrame) -> dict[str, object]:
    best_score = None
    best_row = None
    best_size = 0
    for _, row in grid.iterrows():
        neighborhood = grid[
            (grid["Entry Threshold"].sub(row["Entry Threshold"]).abs() <= 2.0 + 1e-12)
            & (grid["Exit Threshold"].sub(row["Exit Threshold"]).abs() <= 2.0 + 1e-12)
            & (grid["Overlay Weight"].sub(row["Overlay Weight"]).abs() <= 0.05 + 1e-12)
        ]
        avg_calmar = float(neighborhood["Calmar"].mean())
        score = (avg_calmar, float(row["Calmar"]), float(row["CAGR (%)"]))
        if best_score is None or score > best_score:
            best_score = score
            best_row = row
            best_size = len(neighborhood)
    return {"row": best_row, "avg_calmar": round(best_score[0], 6), "size": int(best_size)}


def _mark_selection(row: pd.Series,
                    selector: str,
                    warning: str = "",
                    guard_passed: bool | None = None,
                    robust_avg_calmar: float | None = None,
                    robust_neighborhood_size: int | None = None) -> pd.Series:
    out = row.copy()
    out["Selector"] = selector
    out["Selection Warning"] = warning
    out["IS MaxDD Guard Passed"] = guard_passed if guard_passed is not None else pd.NA
    out["Robust Avg Calmar"] = robust_avg_calmar if robust_avg_calmar is not None else pd.NA
    out["Robust Neighborhood Size"] = robust_neighborhood_size if robust_neighborhood_size is not None else pd.NA
    return out


def _base_series(prices: pd.DataFrame,
                 allocation: dict[str, float],
                 apply_fees: bool) -> pd.Series:
    clean = prices[list(allocation)].dropna()
    base = build_monthly_rebalanced_series(clean, allocation, start_value=100.0)
    base = apply_annual_fee(base, DIY_FEE) if apply_fees else base
    return base.rename(BASE_LABEL)


def _strategy_metrics(series: pd.Series, label: str) -> dict[str, float]:
    metrics = summary_metrics(pd.DataFrame({label: series}))
    return metrics[metrics["Strategy"] == label].iloc[0].to_dict()


def _grid_metrics(series: pd.Series) -> dict[str, float]:
    """Small metric subset for high-volume threshold-grid evaluation."""
    s = series.dropna().astype(float)
    returns = s.pct_change().dropna()
    years = max((s.index[-1] - s.index[0]).days / 365.25, 1 / 365.25)
    total_return = float((s.iloc[-1] / s.iloc[0] - 1.0) * 100)
    cagr = float(((s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1.0) * 100)
    drawdown = (s / s.cummax() - 1.0) * 100
    max_dd = float(drawdown.min())
    rf_daily = (1.0 + config.RISK_FREE_RATE) ** (1 / TRADING_DAYS_PER_YEAR) - 1.0
    sharpe = (
        float(((returns.mean() - rf_daily) / returns.std()) * np.sqrt(TRADING_DAYS_PER_YEAR))
        if len(returns) and returns.std() > 1e-12 else np.nan
    )
    monthly = s.resample(MONTH_END).last().pct_change().dropna() * 100
    return {
        "Total Return (%)": round(total_return, 4),
        "CAGR (%)": round(cagr, 4),
        "Sharpe": round(sharpe, 4) if pd.notna(sharpe) else np.nan,
        "Calmar": round(cagr / abs(max_dd), 4) if abs(max_dd) > 1e-12 else np.nan,
        "Max Drawdown (%)": round(max_dd, 4),
        "Worst Month (%)": round(float(monthly.min()), 4) if len(monthly) else np.nan,
    }


def _available_stress_periods(values: pd.DataFrame) -> dict[str, tuple[str, str]]:
    start = values.index.min()
    end = values.index.max()
    periods = {}
    for name, (period_start, period_end) in STRESS_PERIODS.items():
        ps = pd.Timestamp(period_start)
        pe = pd.Timestamp(period_end)
        if pe >= start and ps <= end:
            periods[name] = (max(ps, start).date().isoformat(), min(pe, end).date().isoformat())
    periods["Full OOS window"] = (start.date().isoformat(), end.date().isoformat())
    return periods


def _concat_or_empty(frames: list[pd.DataFrame]) -> pd.DataFrame:
    clean = [frame for frame in frames if frame is not None and not frame.empty]
    return pd.concat(clean, ignore_index=True) if clean else pd.DataFrame()


def _manifest(
    strategy_id: str,
    allocation: dict[str, float],
    start_date: str,
    end_date: str,
    prices: pd.DataFrame,
    splits: tuple[str, ...],
    specs: list[OverlaySpec],
    entry_grid: tuple[float, ...],
    exit_grid: tuple[float, ...],
    leverage_grid: tuple[float, ...],
    gld_leverage_grid: tuple[float, ...],
    apply_fees: bool,
    generated_at: datetime,
    data_source: str,
) -> dict:
    return {
        "generated_at": generated_at.isoformat(timespec="seconds"),
        "strategy_id": strategy_id,
        "date_range": {
            "requested_start": start_date,
            "requested_end": end_date,
            "actual_start": prices.index.min().date().isoformat(),
            "actual_end": prices.index.max().date().isoformat(),
        },
        "data_source": data_source,
        "pricing_model": config.PRICING_MODEL,
        "apply_fees": apply_fees,
        "fees": {"DIY": DIY_FEE if apply_fees else 0.0},
        "allocation": allocation,
        "oos_splits": list(splits),
        "overlay_specs": [spec.__dict__ for spec in specs],
        "threshold_grid": {
            "entry_thresholds": list(entry_grid),
            "exit_thresholds": list(exit_grid),
            "standard_leverage_grid": list(leverage_grid),
            "gld_leverage_grid": list(gld_leverage_grid),
        },
        "selectors": list(SELECTORS),
        "pass_fail_rules": {
            "maxdd_split_guard_pp": 1.0,
            "overall_maxdd_breach_pp": 3.0,
            "overall_min_passed_splits": 2,
            "low_trade_count_threshold": 3,
        },
        "artifacts": [
            "manifest.json",
            "price_provenance.json",
            "is_threshold_grid.csv",
            "selected_rules.csv",
            "oos_summary.csv",
            "oos_daily_series.csv",
            "oos_signal_history.csv",
            "oos_overlay_diagnostics.csv",
            "oos_stress_metrics.csv",
            "oos_trade_episodes.csv",
            "pass_fail_summary.csv",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate RSI ETF leverage overlays out of sample.")
    parser.add_argument("--strategy-id", default=config.DEFAULT_STRATEGY)
    parser.add_argument("--start-date", default=config.BACKTEST_START)
    parser.add_argument("--end-date", default=config.BACKTEST_END)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--overlay-tickers", default="", help="Comma-separated allocation tickers; default is all.")
    parser.add_argument("--no-fees", action="store_true", help="Do not apply DIY ETF expense-ratio drag.")
    args = parser.parse_args()

    overlay_tickers = [t.strip().upper() for t in args.overlay_tickers.split(",") if t.strip()] or None
    bundle = build_from_yfinance(
        strategy_id=args.strategy_id,
        start_date=args.start_date,
        end_date=args.end_date,
        output_root=args.output_root,
        overlay_tickers=overlay_tickers,
        apply_fees=not args.no_fees,
    )
    print(f"Leverage OOS validation bundle written to: {bundle}")


if __name__ == "__main__":
    main()
