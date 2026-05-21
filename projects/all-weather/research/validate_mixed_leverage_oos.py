"""
validate_mixed_leverage_oos.py
==============================
Out-of-sample validation for capped multi-ETF RSI leverage overlays.

Mirrors validate_leverage_oos.py but operates on MixedOverlayCandidate
objects (capped SPY+GLD sleeves) rather than per-ticker single-ETF rules.

Two evaluation paths share the same bundle:
  - Fixed candidates: evaluate each named MixedOverlayCandidate from
    build_mixed_leverage_report.default_mixed_candidates() on each OOS split.
  - Grid + selectors: scan a coarse mixed grid on IS-only data, pick
    winners per selector, evaluate winners on OOS.

Research-only. Production portfolio logic is untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
from research.build_leverage_comparison_report import (
    BASE_LABEL,
    DIY_FEE,
    STRESS_PERIODS,
    _clean_prices,
    _daily_series,
    _write_json,
)
from research.build_mixed_leverage_report import (
    MixedOverlayCandidate,
    default_mixed_candidates,
)


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "mixed_leverage_oos_validation"
DEFAULT_SWEEP_OUTPUT_ROOT = PROJECT_ROOT / "results" / "mixed_leverage_sweep"
DEFAULT_FULL_GRID_OUTPUT_ROOT = PROJECT_ROOT / "results" / "mixed_leverage_full_grid_oos"
OOS_SPLITS = ("2018-01-01", "2020-01-01", "2022-01-01")
WALK_FORWARD_YEARS = tuple(range(2014, 2026))
CONTROL_SELECTOR = "default_30_50_20"
CONTROL_CANDIDATE_NAME = "SPY+GLD default 20% total cap"
CONFIG_COLS = [
    "SPY Entry",
    "SPY Exit",
    "SPY Weight",
    "GLD Entry",
    "GLD Exit",
    "GLD Weight",
    "Global Cap",
]

# Coarse mixed grid (kept intentionally small — total 12,288 mixed configs per split).
DEFAULT_MIXED_ENTRY_GRID = (22.0, 26.0, 30.0, 34.0)
DEFAULT_MIXED_EXIT_GRID = (42.0, 50.0, 58.0, 66.0)
DEFAULT_MIXED_LEVERAGE_GRID = (0.10, 0.15, 0.20, 0.25)
DEFAULT_MIXED_CAP_GRID = (0.20, 0.25, 0.30)

DISCIPLINED_SPY_ENTRY_GRID = tuple(float(x) for x in range(18, 38, 2))
DISCIPLINED_GLD_ENTRY_GRID = tuple(float(x) for x in range(18, 38, 2))
DISCIPLINED_SPY_EXIT_GRID = (40.0, 44.0, 48.0, 52.0, 56.0, 60.0, 64.0, 68.0)
DISCIPLINED_GLD_EXIT_GRID = DISCIPLINED_SPY_EXIT_GRID
DISCIPLINED_SPY_WEIGHT_GRID = (0.05, 0.10, 0.15, 0.20, 0.25)
DISCIPLINED_GLD_WEIGHT_GRID = (0.10, 0.15, 0.20, 0.25, 0.30)
DISCIPLINED_CAP_GRID = (0.15, 0.20, 0.25, 0.30, 0.35)

FULL_GRID_ENTRY_GRID = tuple(float(x) for x in range(18, 42, 2))
FULL_GRID_EXIT_GRID = tuple(float(x) for x in range(40, 70, 2))
FULL_GRID_SPY_WEIGHT_GRID = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
FULL_GRID_GLD_WEIGHT_GRID = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
FULL_GRID_CAP_GRID = (0.15, 0.20, 0.25, 0.30)

MIXED_TICKERS = ("SPY", "GLD")

SELECTORS = (
    CONTROL_SELECTOR,
    "best_calmar",
    "best_maxdd_preservation",
    "best_cagr_with_maxdd_guard",
    "robust_calmar_region",
    "simple_stable_region",
)


def load_strategy(strategy_id: str) -> dict:
    """Load one strategy payload, accepting canonical ids and registry aliases."""
    return config.load_strategy(strategy_id)


def is_control_candidate_name(name: object) -> bool:
    """Return True for rows kept only as controls, not promotion candidates."""
    text = str(name)
    return text in {CONTROL_SELECTOR, CONTROL_CANDIDATE_NAME} or "default 30/50" in text.lower()


def _limit_tuple(values: tuple[float, ...], max_value: float | None) -> tuple[float, ...]:
    """Return grid values not above max_value, preserving at least one value."""
    if max_value is None:
        return values
    filtered = tuple(v for v in values if v <= max_value + 1e-12)
    if filtered:
        return filtered
    return (min(values),)


def _limit_fixed_candidates(
    candidates: list[MixedOverlayCandidate],
    max_global_cap: float | None,
    max_sleeve_weight: float | None,
) -> list[MixedOverlayCandidate]:
    """Drop fixed candidates outside broker caps for constrained research runs."""
    kept = []
    for candidate in candidates:
        if max_global_cap is not None and candidate.global_cap > max_global_cap + 1e-12:
            continue
        if max_sleeve_weight is not None and any(
            spec.overlay_weight > max_sleeve_weight + 1e-12 for spec in candidate.specs
        ):
            continue
        kept.append(candidate)
    return kept


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build_mixed_oos_validation_bundle(
    prices: pd.DataFrame,
    strategy_id: str,
    allocation: dict[str, float],
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    start_date: str = "2006-01-01",
    end_date: str | None = None,
    splits: tuple[str, ...] = OOS_SPLITS,
    fixed_candidates: list[MixedOverlayCandidate] | None = None,
    entry_grid: tuple[float, ...] = DEFAULT_MIXED_ENTRY_GRID,
    exit_grid: tuple[float, ...] = DEFAULT_MIXED_EXIT_GRID,
    leverage_grid: tuple[float, ...] = DEFAULT_MIXED_LEVERAGE_GRID,
    cap_grid: tuple[float, ...] = DEFAULT_MIXED_CAP_GRID,
    apply_fees: bool = True,
    fixed_only: bool = False,
    max_global_cap: float | None = None,
    max_sleeve_weight: float | None = None,
    funding_spread: float = 0.0,
    generated_at: datetime | None = None,
    data_source: str = "yfinance",
) -> Path:
    """Build a mixed-leverage OOS validation bundle from an in-memory price frame."""
    generated_at = generated_at or datetime.now()
    end_date = end_date or date.today().strftime("%Y-%m-%d")
    fixed_candidates = fixed_candidates or default_mixed_candidates()
    cap_grid = _limit_tuple(cap_grid, max_global_cap)
    leverage_grid = _limit_tuple(leverage_grid, max_sleeve_weight)
    fixed_candidates = _limit_fixed_candidates(fixed_candidates, max_global_cap, max_sleeve_weight)
    prices = _clean_prices(prices)
    price_provenance = get_price_provenance(prices)

    fixed_summary_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    grid_frames: list[pd.DataFrame] = []
    daily_frames: list[pd.DataFrame] = []
    signal_frames: list[pd.DataFrame] = []
    diagnostic_frames: list[pd.DataFrame] = []
    stress_frames: list[pd.DataFrame] = []
    episode_rows: list[dict[str, object]] = []

    for split in splits:
        split_started = time.perf_counter()
        print(f"[mixed-oos] Split {split[:4]}: building IS/OOS windows...", flush=True)
        split_ts = pd.Timestamp(split)
        is_prices = prices.loc[prices.index < split_ts]
        oos_prices = prices.loc[prices.index >= split_ts]
        if is_prices.empty or oos_prices.empty:
            print(f"[mixed-oos] Split {split[:4]}: insufficient data, skipping.", flush=True)
            continue

        is_base = _base_series(is_prices, allocation, apply_fees)
        oos_base = _base_series(oos_prices, allocation, apply_fees)
        if is_base.empty or oos_base.empty:
            continue

        is_base_metrics = _strategy_metrics(is_base, BASE_LABEL)
        oos_base_metrics = _strategy_metrics(oos_base, BASE_LABEL)

        base_daily = _daily_series(pd.DataFrame({BASE_LABEL: oos_base}))
        base_daily.insert(0, "Selector", "base")
        base_daily.insert(0, "Overlay Strategy", BASE_LABEL)
        base_daily.insert(0, "Split", split[:4])
        base_daily.insert(1, "OOS Start", split)
        daily_frames.append(base_daily)

        # ---- Fixed-candidate path ----
        for candidate in fixed_candidates:
            fixed_eval = _evaluate_mixed_oos(
                is_base=is_base,
                is_prices=is_prices,
                oos_base=oos_base,
                oos_prices=oos_prices,
                specs=list(candidate.specs),
                global_cap=candidate.global_cap,
                label=candidate.name,
                selector="fixed_candidate",
                base_metrics=oos_base_metrics,
                split=split,
                funding_spread=funding_spread,
                extra={"Candidate Name": candidate.name, "Notes": candidate.notes},
            )
            fixed_summary_rows.append(fixed_eval["summary"])
            daily_frames.append(fixed_eval["daily"])
            signal_frames.append(fixed_eval["signals"])
            diagnostic_frames.append(fixed_eval["diagnostics"])
            stress_frames.append(fixed_eval["stress"])
            episode_rows.extend(fixed_eval["episodes"])
        print(
            f"[mixed-oos] Split {split[:4]}: fixed candidates done "
            f"in {time.perf_counter() - split_started:.1f}s",
            flush=True,
        )

        if fixed_only:
            continue

        # ---- Grid + selectors path ----
        grid_started = time.perf_counter()
        split_grid = _fast_mixed_grid(
            is_base=is_base,
            is_prices=is_prices,
            entry_grid=entry_grid,
            exit_grid=exit_grid,
            leverage_grid=leverage_grid,
            cap_grid=cap_grid,
        )
        print(
            f"[mixed-oos] Split {split[:4]}: mixed IS grid "
            f"({len(split_grid):,} rows) in {time.perf_counter() - grid_started:.1f}s",
            flush=True,
        )
        if split_grid.empty:
            continue
        split_grid.insert(0, "Split", split[:4])
        split_grid.insert(1, "OOS Start", split)
        split_grid.insert(2, "IS Start Date", is_base.index[0].date().isoformat())
        split_grid.insert(3, "IS End Date", is_base.index[-1].date().isoformat())
        grid_frames.append(split_grid)

        selections = select_mixed_rules(split_grid, is_base_metrics)
        for _, rule in selections.iterrows():
            selected_rows.append(rule.to_dict())
            specs = [
                OverlaySpec(
                    ticker="SPY",
                    entry_threshold=float(rule["SPY Entry"]),
                    exit_threshold=float(rule["SPY Exit"]),
                    overlay_weight=float(rule["SPY Weight"]),
                ),
                OverlaySpec(
                    ticker="GLD",
                    entry_threshold=float(rule["GLD Entry"]),
                    exit_threshold=float(rule["GLD Exit"]),
                    overlay_weight=float(rule["GLD Weight"]),
                ),
            ]
            label = f"mixed {rule['Selector']} (split {split[:4]})"
            grid_eval = _evaluate_mixed_oos(
                is_base=is_base,
                is_prices=is_prices,
                oos_base=oos_base,
                oos_prices=oos_prices,
                specs=specs,
                global_cap=float(rule["Global Cap"]),
                label=label,
                selector=str(rule["Selector"]),
                base_metrics=oos_base_metrics,
                split=split,
                funding_spread=funding_spread,
                extra={
                    "Candidate Name": label,
                    "Notes": str(rule.get("Selection Warning", "") or ""),
                    "Robust Avg Calmar": rule.get("Robust Avg Calmar", pd.NA),
                    "Robust Neighborhood Size": rule.get(
                        "Robust Neighborhood Size", pd.NA
                    ),
                },
            )
            summary_rows.append(grid_eval["summary"])
            daily_frames.append(grid_eval["daily"])
            signal_frames.append(grid_eval["signals"])
            diagnostic_frames.append(grid_eval["diagnostics"])
            stress_frames.append(grid_eval["stress"])
            episode_rows.extend(grid_eval["episodes"])
        print(
            f"[mixed-oos] Split {split[:4]}: selector OOS evaluation done "
            f"in {time.perf_counter() - split_started:.1f}s",
            flush=True,
        )

    is_grid_df = (
        pd.concat(grid_frames, ignore_index=True) if grid_frames else pd.DataFrame()
    )
    selected_df = pd.DataFrame(selected_rows)
    fixed_oos_df = pd.DataFrame(fixed_summary_rows)
    oos_summary_df = pd.DataFrame(summary_rows)
    pass_fail_df = build_pass_fail_summary(fixed_oos_df, oos_summary_df)
    fixed_walk_forward_df = fixed_candidate_walk_forward_summary(
        prices=prices,
        allocation=allocation,
        candidates=fixed_candidates,
        years=WALK_FORWARD_YEARS,
        apply_fees=apply_fees,
        funding_spread=funding_spread,
    )

    bundle_dir = (
        Path(output_root)
        / f"{generated_at.strftime('%Y-%m-%d_%H-%M-%S')}_{strategy_id}"
    )
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
            fixed_candidates=fixed_candidates,
            entry_grid=entry_grid,
            exit_grid=exit_grid,
            leverage_grid=leverage_grid,
            cap_grid=cap_grid,
            apply_fees=apply_fees,
            fixed_only=fixed_only,
            max_global_cap=max_global_cap,
            max_sleeve_weight=max_sleeve_weight,
            funding_spread=funding_spread,
            generated_at=generated_at,
            data_source=data_source,
        ),
    )
    _write_json(bundle_dir / "price_provenance.json", price_provenance)
    is_grid_df.to_csv(bundle_dir / "is_mixed_grid.csv", index=False)
    selected_df.to_csv(bundle_dir / "selected_rules.csv", index=False)
    fixed_oos_df.to_csv(bundle_dir / "fixed_candidates_oos.csv", index=False)
    fixed_walk_forward_df.to_csv(
        bundle_dir / "fixed_candidate_walk_forward_summary.csv", index=False
    )
    oos_summary_df.to_csv(bundle_dir / "oos_summary.csv", index=False)
    _concat_or_empty(daily_frames).to_csv(
        bundle_dir / "oos_daily_series.csv", index=False
    )
    _concat_or_empty(signal_frames).to_csv(
        bundle_dir / "oos_signal_history.csv", index=False
    )
    _concat_or_empty(diagnostic_frames).to_csv(
        bundle_dir / "oos_overlay_diagnostics.csv", index=False
    )
    _concat_or_empty(stress_frames).to_csv(
        bundle_dir / "oos_stress_metrics.csv", index=False
    )
    pd.DataFrame(episode_rows).to_csv(
        bundle_dir / "oos_trade_episodes.csv", index=False
    )
    pass_fail_df.to_csv(bundle_dir / "pass_fail_summary.csv", index=False)
    return bundle_dir


def build_from_yfinance(
    strategy_id: str,
    start_date: str,
    end_date: str,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    apply_fees: bool = True,
    splits: tuple[str, ...] = OOS_SPLITS,
    fixed_only: bool = False,
    max_global_cap: float | None = None,
    max_sleeve_weight: float | None = None,
    funding_spread: float = 0.0,
) -> Path:
    """Fetch prices and write the mixed-leverage OOS validation bundle."""
    payload = load_strategy(strategy_id)
    allocation = payload["allocation"]
    tickers = sorted(set(allocation) | set(MIXED_TICKERS))
    prices = fetch_prices(tickers, start_date, end_date)
    return build_mixed_oos_validation_bundle(
        prices=prices,
        strategy_id=strategy_id,
        allocation=allocation,
        output_root=output_root,
        start_date=start_date,
        end_date=end_date,
        splits=splits,
        apply_fees=apply_fees,
        fixed_only=fixed_only,
        max_global_cap=max_global_cap,
        max_sleeve_weight=max_sleeve_weight,
        funding_spread=funding_spread,
        data_source=config.DATA_SOURCE,
    )


def build_mixed_sweep_bundle(
    prices: pd.DataFrame,
    strategy_id: str,
    allocation: dict[str, float],
    output_root: str | Path = DEFAULT_SWEEP_OUTPUT_ROOT,
    start_date: str = "2006-01-01",
    end_date: str | None = None,
    splits: tuple[str, ...] = OOS_SPLITS,
    walk_forward_years: tuple[int, ...] = WALK_FORWARD_YEARS,
    spy_entry_grid: tuple[float, ...] = DISCIPLINED_SPY_ENTRY_GRID,
    spy_exit_grid: tuple[float, ...] = DISCIPLINED_SPY_EXIT_GRID,
    spy_weight_grid: tuple[float, ...] = DISCIPLINED_SPY_WEIGHT_GRID,
    gld_entry_grid: tuple[float, ...] = DISCIPLINED_GLD_ENTRY_GRID,
    gld_exit_grid: tuple[float, ...] = DISCIPLINED_GLD_EXIT_GRID,
    gld_weight_grid: tuple[float, ...] = DISCIPLINED_GLD_WEIGHT_GRID,
    cap_grid: tuple[float, ...] = DISCIPLINED_CAP_GRID,
    apply_fees: bool = True,
    max_global_cap: float | None = None,
    max_sleeve_weight: float | None = None,
    funding_spread: float = 0.0,
    generated_at: datetime | None = None,
    data_source: str = "yfinance",
) -> Path:
    """Build the disciplined SPY/GLD parameter-surface sweep bundle."""
    generated_at = generated_at or datetime.now()
    end_date = end_date or date.today().strftime("%Y-%m-%d")
    cap_grid = _limit_tuple(cap_grid, max_global_cap)
    spy_weight_grid = _limit_tuple(spy_weight_grid, max_sleeve_weight)
    gld_weight_grid = _limit_tuple(gld_weight_grid, max_sleeve_weight)
    prices = _clean_prices(prices)

    grid_frames: list[pd.DataFrame] = []
    selected_rows: list[dict[str, object]] = []
    oos_rows: list[dict[str, object]] = []

    for split in splits:
        started = time.perf_counter()
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

        split_grid = _fast_mixed_grid(
            is_base=is_base,
            is_prices=is_prices,
            entry_grid=DEFAULT_MIXED_ENTRY_GRID,
            exit_grid=DEFAULT_MIXED_EXIT_GRID,
            leverage_grid=DEFAULT_MIXED_LEVERAGE_GRID,
            cap_grid=cap_grid,
            spy_entry_grid=spy_entry_grid,
            spy_exit_grid=spy_exit_grid,
            spy_weight_grid=spy_weight_grid,
            gld_entry_grid=gld_entry_grid,
            gld_exit_grid=gld_exit_grid,
            gld_weight_grid=gld_weight_grid,
            sort_output=False,
        )
        if split_grid.empty:
            continue
        split_grid.insert(0, "Split", split[:4])
        split_grid.insert(1, "OOS Start", split)
        split_grid.insert(2, "IS Start Date", is_base.index[0].date().isoformat())
        split_grid.insert(3, "IS End Date", is_base.index[-1].date().isoformat())
        grid_frames.append(split_grid)

        selections = select_mixed_rules(split_grid, is_base_metrics)
        for _, rule in selections.iterrows():
            selected_rows.append(rule.to_dict())
            eval_out = _evaluate_rule_row(
                rule=rule,
                is_base=is_base,
                is_prices=is_prices,
                oos_base=oos_base,
                oos_prices=oos_prices,
                base_metrics=oos_base_metrics,
                split=split,
                label=f"sweep {rule['Selector']} (split {split[:4]})",
                funding_spread=funding_spread,
            )
            oos_rows.append(eval_out["summary"])
        print(
            f"[mixed-sweep] Split {split[:4]}: {len(split_grid):,} IS rows, "
            f"{len(selections)} selectors in {time.perf_counter() - started:.1f}s",
            flush=True,
        )

    is_grid_df = pd.concat(grid_frames, ignore_index=True) if grid_frames else pd.DataFrame()
    selected_df = pd.DataFrame(selected_rows)
    oos_summary = pd.DataFrame(oos_rows)
    walk_forward = _walk_forward_summary(
        prices=prices,
        allocation=allocation,
        years=walk_forward_years,
        apply_fees=apply_fees,
        spy_entry_grid=spy_entry_grid,
        spy_exit_grid=spy_exit_grid,
        spy_weight_grid=spy_weight_grid,
        gld_entry_grid=gld_entry_grid,
        gld_exit_grid=gld_exit_grid,
        gld_weight_grid=gld_weight_grid,
        cap_grid=cap_grid,
        funding_spread=funding_spread,
    )
    stability = parameter_stability_summary(selected_df, walk_forward)
    heatmaps = sweep_heatmap_tables(is_grid_df)
    leaderboard = sweep_grid_leaderboard(is_grid_df)
    pass_fail = disciplined_pass_fail_summary(oos_summary, walk_forward, stability)

    bundle_dir = (
        Path(output_root)
        / f"{generated_at.strftime('%Y-%m-%d_%H-%M-%S')}_{strategy_id}"
    )
    bundle_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        bundle_dir / "manifest.json",
        _sweep_manifest(
            strategy_id=strategy_id,
            allocation=allocation,
            start_date=start_date,
            end_date=end_date,
            prices=prices,
            splits=splits,
            walk_forward_years=walk_forward_years,
            spy_entry_grid=spy_entry_grid,
            spy_exit_grid=spy_exit_grid,
            spy_weight_grid=spy_weight_grid,
            gld_entry_grid=gld_entry_grid,
            gld_exit_grid=gld_exit_grid,
            gld_weight_grid=gld_weight_grid,
            cap_grid=cap_grid,
            apply_fees=apply_fees,
            max_global_cap=max_global_cap,
            max_sleeve_weight=max_sleeve_weight,
            funding_spread=funding_spread,
            generated_at=generated_at,
            data_source=data_source,
        ),
    )
    is_grid_df.to_parquet(bundle_dir / "is_sweep_grid.parquet", index=False, compression="snappy")
    leaderboard.to_csv(bundle_dir / "is_sweep_leaderboard.csv", index=False)
    selected_df.to_csv(bundle_dir / "selected_rules.csv", index=False)
    oos_summary.to_csv(bundle_dir / "oos_summary.csv", index=False)
    walk_forward.to_csv(bundle_dir / "walk_forward_summary.csv", index=False)
    stability.to_csv(bundle_dir / "parameter_stability.csv", index=False)
    heatmaps.to_csv(bundle_dir / "sweep_heatmap_tables.csv", index=False)
    pass_fail.to_csv(bundle_dir / "pass_fail_summary.csv", index=False)
    return bundle_dir


def sweep_grid_leaderboard(grid: pd.DataFrame, top_n_per_split: int = 250) -> pd.DataFrame:
    """Return a compact CSV-friendly leaderboard from the full IS sweep grid."""
    if grid.empty:
        return pd.DataFrame()

    sort_cols = ["Split", "Calmar", "Incremental Calmar", "Max Drawdown (%)", "CAGR (%)"]
    available_sort_cols = [col for col in sort_cols if col in grid.columns]
    ascending = [True, False, False, False, False][:len(available_sort_cols)]
    ranked = grid.sort_values(available_sort_cols, ascending=ascending).copy()
    ranked["IS Calmar Rank"] = ranked.groupby("Split").cumcount() + 1
    ranked = ranked[ranked["IS Calmar Rank"] <= top_n_per_split]
    ranked["SPY Rule"] = ranked.apply(lambda row: _row_rule_label(row, "SPY"), axis=1)
    ranked["GLD Rule"] = ranked.apply(lambda row: _row_rule_label(row, "GLD"), axis=1)
    ranked["Broker Profile"] = np.select(
        [
            (ranked["Global Cap"] <= 0.20)
            & (ranked["SPY Weight"] <= 0.20)
            & (ranked["GLD Weight"] <= 0.20),
            (ranked["Global Cap"] <= 0.30)
            & (ranked["SPY Weight"] <= 0.30)
            & (ranked["GLD Weight"] <= 0.30),
        ],
        ["Strict Pilot", "IBKR Safe"],
        default="Research Unrestricted",
    )
    first_cols = [
        "Split",
        "IS Calmar Rank",
        "Broker Profile",
        "SPY Rule",
        "GLD Rule",
        "Global Cap",
        "SPY Entry",
        "SPY Exit",
        "SPY Weight",
        "GLD Entry",
        "GLD Exit",
        "GLD Weight",
        "Calmar",
        "Incremental Calmar",
        "Max Drawdown (%)",
        "Incremental MaxDD (%)",
        "CAGR (%)",
        "Incremental CAGR (%)",
        "Active Days (%)",
        "Both Active Days (%)",
        "Average Overlay Exposure (%)",
        "Max Overlay Exposure (%)",
    ]
    ordered_cols = [col for col in first_cols if col in ranked.columns]
    return ranked[ordered_cols + [col for col in ranked.columns if col not in ordered_cols]]


def _row_rule_label(row: pd.Series, ticker: str) -> str:
    entry = row.get(f"{ticker} Entry")
    exit_ = row.get(f"{ticker} Exit")
    weight = row.get(f"{ticker} Weight")
    if pd.isna(entry) or pd.isna(exit_) or pd.isna(weight):
        return ""
    return f"{float(entry):g}/{float(exit_):g} @ {float(weight) * 100:g}%"


def build_sweep_from_yfinance(
    strategy_id: str,
    start_date: str,
    end_date: str,
    output_root: str | Path = DEFAULT_SWEEP_OUTPUT_ROOT,
    apply_fees: bool = True,
    splits: tuple[str, ...] = OOS_SPLITS,
    walk_forward_years: tuple[int, ...] = WALK_FORWARD_YEARS,
    max_global_cap: float | None = None,
    max_sleeve_weight: float | None = None,
    funding_spread: float = 0.0,
) -> Path:
    """Fetch prices and write the disciplined mixed-leverage sweep bundle."""
    payload = load_strategy(strategy_id)
    allocation = payload["allocation"]
    tickers = sorted(set(allocation) | set(MIXED_TICKERS))
    prices = fetch_prices(tickers, start_date, end_date)
    return build_mixed_sweep_bundle(
        prices=prices,
        strategy_id=strategy_id,
        allocation=allocation,
        output_root=output_root,
        start_date=start_date,
        end_date=end_date,
        splits=splits,
        walk_forward_years=walk_forward_years,
        apply_fees=apply_fees,
        max_global_cap=max_global_cap,
        max_sleeve_weight=max_sleeve_weight,
        funding_spread=funding_spread,
        data_source=config.DATA_SOURCE,
    )


def build_full_grid_from_yfinance(
    strategy_id: str,
    start_date: str,
    end_date: str,
    output_root: str | Path = DEFAULT_FULL_GRID_OUTPUT_ROOT,
    apply_fees: bool = True,
    splits: tuple[str, ...] = OOS_SPLITS,
    walk_forward_years: tuple[int, ...] = WALK_FORWARD_YEARS,
    max_global_cap: float | None = None,
    max_sleeve_weight: float | None = None,
    funding_spread: float = 0.0,
) -> Path:
    """Fetch prices and write full-grid structural OOS plus annual walk-forward results."""
    payload = load_strategy(strategy_id)
    allocation = payload["allocation"]
    tickers = sorted(set(allocation) | set(MIXED_TICKERS))
    prices = fetch_prices(tickers, start_date, end_date)
    return build_full_grid_oos_bundle(
        prices=prices,
        strategy_id=strategy_id,
        allocation=allocation,
        output_root=output_root,
        start_date=start_date,
        end_date=end_date,
        splits=splits,
        walk_forward_years=walk_forward_years,
        apply_fees=apply_fees,
        max_global_cap=max_global_cap,
        max_sleeve_weight=max_sleeve_weight,
        funding_spread=funding_spread,
        data_source=config.DATA_SOURCE,
    )


def build_full_grid_oos_bundle(
    prices: pd.DataFrame,
    strategy_id: str,
    allocation: dict[str, float],
    output_root: str | Path = DEFAULT_FULL_GRID_OUTPUT_ROOT,
    start_date: str = "2006-01-01",
    end_date: str | None = None,
    splits: tuple[str, ...] = OOS_SPLITS,
    walk_forward_years: tuple[int, ...] = WALK_FORWARD_YEARS,
    entry_grid: tuple[float, ...] = FULL_GRID_ENTRY_GRID,
    exit_grid: tuple[float, ...] = FULL_GRID_EXIT_GRID,
    spy_weight_grid: tuple[float, ...] = FULL_GRID_SPY_WEIGHT_GRID,
    gld_weight_grid: tuple[float, ...] = FULL_GRID_GLD_WEIGHT_GRID,
    cap_grid: tuple[float, ...] = FULL_GRID_CAP_GRID,
    apply_fees: bool = True,
    max_global_cap: float | None = None,
    max_sleeve_weight: float | None = None,
    funding_spread: float = 0.0,
    generated_at: datetime | None = None,
    data_source: str = "yfinance",
) -> Path:
    """Evaluate every full-grid config OOS and annual walk-forward."""
    generated_at = generated_at or datetime.now()
    end_date = end_date or date.today().strftime("%Y-%m-%d")
    cap_grid = _limit_tuple(cap_grid, max_global_cap)
    spy_weight_grid = _limit_tuple(spy_weight_grid, max_sleeve_weight)
    gld_weight_grid = _limit_tuple(gld_weight_grid, max_sleeve_weight)
    prices = _clean_prices(prices)

    bundle_dir = (
        Path(output_root)
        / f"{generated_at.strftime('%Y-%m-%d_%H-%M-%S')}_{strategy_id}"
    )
    bundle_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        bundle_dir / "manifest.json",
        _full_grid_manifest(
            strategy_id=strategy_id,
            allocation=allocation,
            start_date=start_date,
            end_date=end_date,
            prices=prices,
            splits=splits,
            walk_forward_years=walk_forward_years,
            entry_grid=entry_grid,
            exit_grid=exit_grid,
            spy_weight_grid=spy_weight_grid,
            gld_weight_grid=gld_weight_grid,
            cap_grid=cap_grid,
            apply_fees=apply_fees,
            max_global_cap=max_global_cap,
            max_sleeve_weight=max_sleeve_weight,
            funding_spread=funding_spread,
            generated_at=generated_at,
            data_source=data_source,
        ),
    )

    structural = _full_grid_structural_oos(
        prices=prices,
        allocation=allocation,
        splits=splits,
        apply_fees=apply_fees,
        entry_grid=entry_grid,
        exit_grid=exit_grid,
        spy_weight_grid=spy_weight_grid,
        gld_weight_grid=gld_weight_grid,
        cap_grid=cap_grid,
        funding_spread=funding_spread,
        output_path=bundle_dir / "structural_full_grid_oos.parquet",
    )
    annual = _full_grid_annual_walk_forward(
        prices=prices,
        allocation=allocation,
        years=walk_forward_years,
        apply_fees=apply_fees,
        entry_grid=entry_grid,
        exit_grid=exit_grid,
        spy_weight_grid=spy_weight_grid,
        gld_weight_grid=gld_weight_grid,
        cap_grid=cap_grid,
        funding_spread=funding_spread,
        output_path=bundle_dir / "annual_full_grid_walk_forward.parquet",
    )
    structural_summary = structural
    annual_summary = annual
    combined = _full_grid_combined_summary(structural_summary, annual_summary)
    structural_leaderboard = _full_grid_leaderboard(combined)
    annual_leaderboard = _full_grid_annual_leaderboard(annual_summary)
    compact_summary = _full_grid_compact_summary(structural_summary)

    annual_summary.to_parquet(
        bundle_dir / "annual_full_grid_summary.parquet", index=False, compression="snappy"
    )
    combined.to_csv(bundle_dir / "all_considered_strategies.csv", index=False)
    compact_summary.to_csv(bundle_dir / "structural_full_grid_summary.csv", index=False)
    structural_leaderboard.to_csv(bundle_dir / "structural_full_grid_leaderboard.csv", index=False)
    annual_leaderboard.to_csv(bundle_dir / "annual_full_grid_leaderboard.csv", index=False)
    return bundle_dir


def _full_grid_structural_oos(
    prices: pd.DataFrame,
    allocation: dict[str, float],
    splits: tuple[str, ...],
    apply_fees: bool,
    entry_grid: tuple[float, ...],
    exit_grid: tuple[float, ...],
    spy_weight_grid: tuple[float, ...],
    gld_weight_grid: tuple[float, ...],
    cap_grid: tuple[float, ...],
    funding_spread: float,
    output_path: Path | None = None,
) -> pd.DataFrame:
    output_path = Path(output_path) if output_path is not None else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)
    accumulator: dict[str, object] | None = None
    for split in splits:
        started = time.perf_counter()
        split_ts = pd.Timestamp(split)
        is_prices = prices.loc[prices.index < split_ts]
        oos_prices = prices.loc[prices.index >= split_ts]
        if is_prices.empty or oos_prices.empty:
            continue
        is_base = _base_series(is_prices, allocation, apply_fees)
        oos_base = _base_series(oos_prices, allocation, apply_fees)
        if is_base.empty or oos_base.empty:
            continue
        period = _full_grid_period(
            is_base=is_base,
            is_prices=is_prices,
            eval_base=oos_base,
            eval_prices=oos_prices,
            entry_grid=entry_grid,
            exit_grid=exit_grid,
            spy_weight_grid=spy_weight_grid,
            gld_weight_grid=gld_weight_grid,
            cap_grid=cap_grid,
            funding_spread=funding_spread,
            split=split,
            period_label="OOS",
            include_is=False,
        )
        if output_path is not None:
            period.to_parquet(
                output_path / f"split={split[:4]}.parquet",
                index=False,
                compression="snappy",
            )
        accumulator = _update_full_grid_structural_accumulator(accumulator, period)
        print(
            f"[full-grid-oos] Split {split[:4]}: {len(period):,} configs in "
            f"{time.perf_counter() - started:.1f}s",
            flush=True,
        )
    return _finalize_full_grid_structural_accumulator(accumulator)


def _full_grid_annual_walk_forward(
    prices: pd.DataFrame,
    allocation: dict[str, float],
    years: tuple[int, ...],
    apply_fees: bool,
    entry_grid: tuple[float, ...],
    exit_grid: tuple[float, ...],
    spy_weight_grid: tuple[float, ...],
    gld_weight_grid: tuple[float, ...],
    cap_grid: tuple[float, ...],
    funding_spread: float,
    output_path: Path | None = None,
) -> pd.DataFrame:
    output_path = Path(output_path) if output_path is not None else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)
    accumulator: dict[str, object] | None = None
    for year in years:
        started = time.perf_counter()
        train_end = pd.Timestamp(f"{year}-01-01")
        eval_end = pd.Timestamp(f"{year + 1}-01-01")
        train_prices = prices.loc[prices.index < train_end]
        eval_prices = prices.loc[(prices.index >= train_end) & (prices.index < eval_end)]
        if train_prices.empty or eval_prices.empty:
            continue
        is_base = _base_series(train_prices, allocation, apply_fees)
        eval_base = _base_series(eval_prices, allocation, apply_fees)
        if is_base.empty or eval_base.empty:
            continue
        period = _full_grid_period(
            is_base=is_base,
            is_prices=train_prices,
            eval_base=eval_base,
            eval_prices=eval_prices,
            entry_grid=entry_grid,
            exit_grid=exit_grid,
            spy_weight_grid=spy_weight_grid,
            gld_weight_grid=gld_weight_grid,
            cap_grid=cap_grid,
            funding_spread=funding_spread,
            split=f"{year}-01-01",
            period_label="Annual",
            include_is=False,
        )
        period["Year"] = int(year)
        period["Is Partial Year"] = bool(year == date.today().year)
        period["Calmar Improvement"] = period["OOS Calmar Delta"] > 0
        if output_path is not None:
            period.to_parquet(
                output_path / f"year={year}.parquet",
                index=False,
                compression="snappy",
            )
        accumulator = _update_full_grid_annual_accumulator(accumulator, period)
        print(
            f"[full-grid-wf] Year {year}: {len(period):,} configs in "
            f"{time.perf_counter() - started:.1f}s",
            flush=True,
        )
    return _finalize_full_grid_annual_accumulator(accumulator)


def _full_grid_period(
    is_base: pd.Series,
    is_prices: pd.DataFrame,
    eval_base: pd.Series,
    eval_prices: pd.DataFrame,
    entry_grid: tuple[float, ...],
    exit_grid: tuple[float, ...],
    spy_weight_grid: tuple[float, ...],
    gld_weight_grid: tuple[float, ...],
    cap_grid: tuple[float, ...],
    funding_spread: float,
    split: str,
    period_label: str,
    include_is: bool = True,
) -> pd.DataFrame:
    is_grid = pd.DataFrame()
    if include_is:
        started = time.perf_counter()
        is_grid = _fast_mixed_grid_vectorized(
            is_base=is_base,
            is_prices=is_prices,
            entry_grid=entry_grid,
            exit_grid=exit_grid,
            leverage_grid=DEFAULT_MIXED_LEVERAGE_GRID,
            cap_grid=cap_grid,
            spy_entry_grid=entry_grid,
            spy_exit_grid=exit_grid,
            spy_weight_grid=spy_weight_grid,
            gld_entry_grid=entry_grid,
            gld_exit_grid=exit_grid,
            gld_weight_grid=gld_weight_grid,
            sort_output=False,
            progress_label=f"{period_label} {split[:4]} IS",
        )
        print(
            f"[full-grid] {period_label} {split[:4]} IS grid: {len(is_grid):,} rows "
            f"in {time.perf_counter() - started:.1f}s",
            flush=True,
        )
    started = time.perf_counter()
    oos_grid = _fast_mixed_grid_vectorized(
        is_base=eval_base,
        is_prices=eval_prices,
        entry_grid=entry_grid,
        exit_grid=exit_grid,
        leverage_grid=DEFAULT_MIXED_LEVERAGE_GRID,
        cap_grid=cap_grid,
        spy_entry_grid=entry_grid,
        spy_exit_grid=exit_grid,
        spy_weight_grid=spy_weight_grid,
        gld_entry_grid=entry_grid,
        gld_exit_grid=exit_grid,
        gld_weight_grid=gld_weight_grid,
        sort_output=False,
        progress_label=f"{period_label} {split[:4]} eval",
    )
    print(
        f"[full-grid] {period_label} {split[:4]} eval grid: {len(oos_grid):,} rows "
        f"in {time.perf_counter() - started:.1f}s",
        flush=True,
    )
    if include_is:
        out = _combine_prefixed_grids(is_grid, oos_grid)
    else:
        out = _prefix_grid(oos_grid, "OOS")
    base_metrics = _strategy_metrics(eval_base, BASE_LABEL)
    out.insert(0, "Split", split[:4])
    out.insert(1, "OOS Start", split)
    out.insert(2, "Period Label", period_label)
    out["OOS Base CAGR (%)"] = base_metrics["CAGR (%)"]
    out["OOS Base Calmar"] = base_metrics["Calmar"]
    out["OOS Base Max Drawdown (%)"] = base_metrics["Max Drawdown (%)"]
    out["OOS CAGR Delta (%)"] = out["OOS Incremental CAGR (%)"]
    out["OOS Calmar Delta"] = out["OOS Incremental Calmar"]
    out["OOS MaxDD Delta (%)"] = out["OOS Incremental MaxDD (%)"]
    funding_cost = config.RISK_FREE_RATE + funding_spread
    out["Estimated OOS RF Cost CAGR (%)"] = (
        out["OOS CAGR (%)"] - funding_cost * out["OOS Average Overlay Exposure (%)"]
    )
    out["RF Cost Pass"] = out["Estimated OOS RF Cost CAGR (%)"] >= out["OOS Base CAGR (%)"]
    out["Calmar Pass"] = out["OOS Calmar Delta"] > 0
    out["CAGR Pass"] = out["OOS CAGR Delta (%)"] >= 0
    out["MaxDD Pass"] = out["OOS MaxDD Delta (%)"] >= -1.0
    out["Low Trade Count Flag"] = out["OOS Trade Episodes"] < 3
    out["Pass Split"] = (
        out["Calmar Pass"]
        & out["CAGR Pass"]
        & out["MaxDD Pass"]
        & out["RF Cost Pass"]
    )
    out["Broker Profile"] = _broker_profile_series(out)
    return out


def _prefix_grid(grid: pd.DataFrame, prefix: str) -> pd.DataFrame:
    rename = {col: f"{prefix} {col}" for col in grid.columns if col not in CONFIG_COLS}
    return grid.rename(columns=rename)


def _combine_prefixed_grids(is_grid: pd.DataFrame, oos_grid: pd.DataFrame) -> pd.DataFrame:
    if is_grid.empty or oos_grid.empty:
        return pd.DataFrame()
    left = is_grid.reset_index(drop=True)
    right = oos_grid.reset_index(drop=True)
    if len(left) == len(right) and left[CONFIG_COLS].equals(right[CONFIG_COLS]):
        return pd.concat(
            [
                left[CONFIG_COLS],
                left.drop(columns=CONFIG_COLS).add_prefix("IS "),
                right.drop(columns=CONFIG_COLS).add_prefix("OOS "),
            ],
            axis=1,
        )
    return _prefix_grid(left, "IS").merge(
        _prefix_grid(right, "OOS"), on=CONFIG_COLS, how="inner"
    )


def _full_grid_structural_summary(structural: pd.DataFrame) -> pd.DataFrame:
    if structural.empty:
        return pd.DataFrame()
    group = structural.groupby(CONFIG_COLS, dropna=False)
    out = group.agg(
        **{
            "Structural Splits Tested": ("Split", "nunique"),
            "Structural Splits Passed": ("Pass Split", "sum"),
            "Average OOS Calmar": ("OOS Calmar", "mean"),
            "Average OOS Calmar Delta": ("OOS Calmar Delta", "mean"),
            "Worst OOS Calmar Delta": ("OOS Calmar Delta", "min"),
            "Worst OOS MaxDD Delta (%)": ("OOS MaxDD Delta (%)", "min"),
            "Average OOS CAGR Delta (%)": ("OOS CAGR Delta (%)", "mean"),
            "RF Cost Pass Splits": ("RF Cost Pass", "sum"),
            "Average OOS Active Days": ("OOS Active Days", "mean"),
            "Average OOS SPY Active Days": ("OOS SPY Active Days", "mean"),
            "Average OOS GLD Active Days": ("OOS GLD Active Days", "mean"),
            "Average OOS Both Active Days": ("OOS Both Active Days", "mean"),
            "Average OOS Cap Binding Days": ("OOS Cap Binding Days", "mean"),
            "Min OOS Trade Episodes": ("OOS Trade Episodes", "min"),
            "Min OOS SPY Trade Episodes": ("OOS SPY Trade Episodes", "min"),
            "Min OOS GLD Trade Episodes": ("OOS GLD Trade Episodes", "min"),
            "Average OOS Active Days (%)": ("OOS Active Days (%)", "mean"),
            "Average OOS SPY Active Days (%)": ("OOS SPY Active Days (%)", "mean"),
            "Average OOS GLD Active Days (%)": ("OOS GLD Active Days (%)", "mean"),
            "Average OOS Both Active Days (%)": ("OOS Both Active Days (%)", "mean"),
            "Average OOS Cap Binding Days (%)": ("OOS Cap Binding Days (%)", "mean"),
            "Average OOS Exposure (%)": ("OOS Average Overlay Exposure (%)", "mean"),
            "Average OOS Max Overlay Exposure (%)": ("OOS Max Overlay Exposure (%)", "mean"),
        }
    ).reset_index()
    out["MaxDD Breach >3pp"] = out["Worst OOS MaxDD Delta (%)"] < -3.0
    out["Structural Pass"] = (
        (out["Structural Splits Passed"] == out["Structural Splits Tested"])
        & ~out["MaxDD Breach >3pp"]
    )
    out["SPY Rule"] = out.apply(lambda row: _row_rule_label(row, "SPY"), axis=1)
    out["GLD Rule"] = out.apply(lambda row: _row_rule_label(row, "GLD"), axis=1)
    out["Broker Profile"] = _broker_profile_series(out)
    return _round_float_cols(out)


def _full_grid_annual_summary(annual: pd.DataFrame) -> pd.DataFrame:
    if annual.empty:
        return pd.DataFrame()
    group = annual.groupby(CONFIG_COLS, dropna=False)
    out = group.agg(
        **{
            "Annual Years Tested": ("Year", "nunique"),
            "Annual Calmar Improvement Years": ("Calmar Improvement", "sum"),
            "Average Annual Calmar": ("OOS Calmar", "mean"),
            "Average Annual Calmar Delta": ("OOS Calmar Delta", "mean"),
            "Worst Annual Calmar Delta": ("OOS Calmar Delta", "min"),
            "Worst Annual MaxDD Delta (%)": ("OOS MaxDD Delta (%)", "min"),
            "Average Annual Exposure (%)": ("OOS Average Overlay Exposure (%)", "mean"),
            "Average Annual Active Days": ("OOS Active Days", "mean"),
            "Average Annual SPY Active Days": ("OOS SPY Active Days", "mean"),
            "Average Annual GLD Active Days": ("OOS GLD Active Days", "mean"),
            "Average Annual Both Active Days": ("OOS Both Active Days", "mean"),
            "Average Annual Cap Binding Days": ("OOS Cap Binding Days", "mean"),
            "Average Annual SPY Active Days (%)": ("OOS SPY Active Days (%)", "mean"),
            "Average Annual GLD Active Days (%)": ("OOS GLD Active Days (%)", "mean"),
            "Average Annual Both Active Days (%)": ("OOS Both Active Days (%)", "mean"),
            "Average Annual Cap Binding Days (%)": ("OOS Cap Binding Days (%)", "mean"),
            "Min Annual Trade Episodes": ("OOS Trade Episodes", "min"),
            "Min Annual SPY Trade Episodes": ("OOS SPY Trade Episodes", "min"),
            "Min Annual GLD Trade Episodes": ("OOS GLD Trade Episodes", "min"),
        }
    ).reset_index()
    out["SPY Rule"] = out.apply(lambda row: _row_rule_label(row, "SPY"), axis=1)
    out["GLD Rule"] = out.apply(lambda row: _row_rule_label(row, "GLD"), axis=1)
    out["Broker Profile"] = _broker_profile_series(out)
    return _round_float_cols(out)


def _update_full_grid_structural_accumulator(
    accumulator: dict[str, object] | None, period: pd.DataFrame
) -> dict[str, object] | None:
    if period.empty:
        return accumulator
    configs = period[CONFIG_COLS].reset_index(drop=True)
    if accumulator is None:
        accumulator = {
            "configs": configs,
            "tested": np.zeros(len(period), dtype=np.int16),
            "passed": np.zeros(len(period), dtype=np.int16),
            "calmar_sum": np.zeros(len(period), dtype=float),
            "calmar_delta_sum": np.zeros(len(period), dtype=float),
            "worst_calmar_delta": np.full(len(period), np.inf),
            "worst_maxdd_delta": np.full(len(period), np.inf),
            "cagr_delta_sum": np.zeros(len(period), dtype=float),
            "rf_pass": np.zeros(len(period), dtype=np.int16),
            "active_days_sum": np.zeros(len(period), dtype=float),
            "spy_active_days_sum": np.zeros(len(period), dtype=float),
            "gld_active_days_sum": np.zeros(len(period), dtype=float),
            "both_active_days_sum": np.zeros(len(period), dtype=float),
            "cap_binding_days_sum": np.zeros(len(period), dtype=float),
            "active_days_pct_sum": np.zeros(len(period), dtype=float),
            "spy_active_days_pct_sum": np.zeros(len(period), dtype=float),
            "gld_active_days_pct_sum": np.zeros(len(period), dtype=float),
            "both_active_days_pct_sum": np.zeros(len(period), dtype=float),
            "cap_binding_days_pct_sum": np.zeros(len(period), dtype=float),
            "min_trades": np.full(len(period), np.inf),
            "min_spy_trades": np.full(len(period), np.inf),
            "min_gld_trades": np.full(len(period), np.inf),
            "exposure_sum": np.zeros(len(period), dtype=float),
            "max_exposure_sum": np.zeros(len(period), dtype=float),
        }
    elif not configs.equals(accumulator["configs"]):
        raise ValueError("Full-grid structural accumulator received a different config order.")

    accumulator["tested"] += 1
    accumulator["passed"] += period["Pass Split"].to_numpy(dtype=np.int16)
    accumulator["calmar_sum"] += period["OOS Calmar"].to_numpy(dtype=float)
    accumulator["calmar_delta_sum"] += period["OOS Calmar Delta"].to_numpy(dtype=float)
    accumulator["worst_calmar_delta"] = np.minimum(
        accumulator["worst_calmar_delta"],
        period["OOS Calmar Delta"].to_numpy(dtype=float),
    )
    accumulator["worst_maxdd_delta"] = np.minimum(
        accumulator["worst_maxdd_delta"],
        period["OOS MaxDD Delta (%)"].to_numpy(dtype=float),
    )
    accumulator["cagr_delta_sum"] += period["OOS CAGR Delta (%)"].to_numpy(dtype=float)
    accumulator["rf_pass"] += period["RF Cost Pass"].to_numpy(dtype=np.int16)
    accumulator["active_days_sum"] += period["OOS Active Days"].to_numpy(dtype=float)
    accumulator["spy_active_days_sum"] += period["OOS SPY Active Days"].to_numpy(dtype=float)
    accumulator["gld_active_days_sum"] += period["OOS GLD Active Days"].to_numpy(dtype=float)
    accumulator["both_active_days_sum"] += period["OOS Both Active Days"].to_numpy(dtype=float)
    accumulator["cap_binding_days_sum"] += period["OOS Cap Binding Days"].to_numpy(dtype=float)
    accumulator["active_days_pct_sum"] += period["OOS Active Days (%)"].to_numpy(dtype=float)
    accumulator["spy_active_days_pct_sum"] += period["OOS SPY Active Days (%)"].to_numpy(dtype=float)
    accumulator["gld_active_days_pct_sum"] += period["OOS GLD Active Days (%)"].to_numpy(dtype=float)
    accumulator["both_active_days_pct_sum"] += period["OOS Both Active Days (%)"].to_numpy(dtype=float)
    accumulator["cap_binding_days_pct_sum"] += period["OOS Cap Binding Days (%)"].to_numpy(dtype=float)
    accumulator["min_trades"] = np.minimum(
        accumulator["min_trades"],
        period["OOS Trade Episodes"].to_numpy(dtype=float),
    )
    accumulator["min_spy_trades"] = np.minimum(
        accumulator["min_spy_trades"],
        period["OOS SPY Trade Episodes"].to_numpy(dtype=float),
    )
    accumulator["min_gld_trades"] = np.minimum(
        accumulator["min_gld_trades"],
        period["OOS GLD Trade Episodes"].to_numpy(dtype=float),
    )
    accumulator["exposure_sum"] += period["OOS Average Overlay Exposure (%)"].to_numpy(dtype=float)
    accumulator["max_exposure_sum"] += period["OOS Max Overlay Exposure (%)"].to_numpy(dtype=float)
    return accumulator


def _finalize_full_grid_structural_accumulator(
    accumulator: dict[str, object] | None,
) -> pd.DataFrame:
    if accumulator is None:
        return pd.DataFrame()
    tested = accumulator["tested"].astype(float)
    out = accumulator["configs"].copy()
    out["Structural Splits Tested"] = accumulator["tested"]
    out["Structural Splits Passed"] = accumulator["passed"]
    out["Average OOS Calmar"] = accumulator["calmar_sum"] / tested
    out["Average OOS Calmar Delta"] = accumulator["calmar_delta_sum"] / tested
    out["Worst OOS Calmar Delta"] = accumulator["worst_calmar_delta"]
    out["Worst OOS MaxDD Delta (%)"] = accumulator["worst_maxdd_delta"]
    out["Average OOS CAGR Delta (%)"] = accumulator["cagr_delta_sum"] / tested
    out["RF Cost Pass Splits"] = accumulator["rf_pass"]
    out["Average OOS Active Days"] = accumulator["active_days_sum"] / tested
    out["Average OOS SPY Active Days"] = accumulator["spy_active_days_sum"] / tested
    out["Average OOS GLD Active Days"] = accumulator["gld_active_days_sum"] / tested
    out["Average OOS Both Active Days"] = accumulator["both_active_days_sum"] / tested
    out["Average OOS Cap Binding Days"] = accumulator["cap_binding_days_sum"] / tested
    out["Min OOS Trade Episodes"] = accumulator["min_trades"].astype(int)
    out["Min OOS SPY Trade Episodes"] = accumulator["min_spy_trades"].astype(int)
    out["Min OOS GLD Trade Episodes"] = accumulator["min_gld_trades"].astype(int)
    out["Average OOS Active Days (%)"] = accumulator["active_days_pct_sum"] / tested
    out["Average OOS SPY Active Days (%)"] = accumulator["spy_active_days_pct_sum"] / tested
    out["Average OOS GLD Active Days (%)"] = accumulator["gld_active_days_pct_sum"] / tested
    out["Average OOS Both Active Days (%)"] = accumulator["both_active_days_pct_sum"] / tested
    out["Average OOS Cap Binding Days (%)"] = accumulator["cap_binding_days_pct_sum"] / tested
    out["Average OOS Exposure (%)"] = accumulator["exposure_sum"] / tested
    out["Average OOS Max Overlay Exposure (%)"] = accumulator["max_exposure_sum"] / tested
    out["MaxDD Breach >3pp"] = out["Worst OOS MaxDD Delta (%)"] < -3.0
    out["Structural Pass"] = (
        (out["Structural Splits Passed"] == out["Structural Splits Tested"])
        & ~out["MaxDD Breach >3pp"]
    )
    out["SPY Rule"] = out.apply(lambda row: _row_rule_label(row, "SPY"), axis=1)
    out["GLD Rule"] = out.apply(lambda row: _row_rule_label(row, "GLD"), axis=1)
    out["Broker Profile"] = _broker_profile_series(out)
    return _round_float_cols(out)


def _update_full_grid_annual_accumulator(
    accumulator: dict[str, object] | None, period: pd.DataFrame
) -> dict[str, object] | None:
    if period.empty:
        return accumulator
    configs = period[CONFIG_COLS].reset_index(drop=True)
    if accumulator is None:
        accumulator = {
            "configs": configs,
            "tested": np.zeros(len(period), dtype=np.int16),
            "improved": np.zeros(len(period), dtype=np.int16),
            "calmar_sum": np.zeros(len(period), dtype=float),
            "calmar_delta_sum": np.zeros(len(period), dtype=float),
            "worst_calmar_delta": np.full(len(period), np.inf),
            "worst_maxdd_delta": np.full(len(period), np.inf),
            "exposure_sum": np.zeros(len(period), dtype=float),
            "active_days_sum": np.zeros(len(period), dtype=float),
            "spy_active_days_sum": np.zeros(len(period), dtype=float),
            "gld_active_days_sum": np.zeros(len(period), dtype=float),
            "both_active_days_sum": np.zeros(len(period), dtype=float),
            "cap_binding_days_sum": np.zeros(len(period), dtype=float),
            "spy_active_days_pct_sum": np.zeros(len(period), dtype=float),
            "gld_active_days_pct_sum": np.zeros(len(period), dtype=float),
            "both_active_days_pct_sum": np.zeros(len(period), dtype=float),
            "cap_binding_days_pct_sum": np.zeros(len(period), dtype=float),
            "min_trades": np.full(len(period), np.inf),
            "min_spy_trades": np.full(len(period), np.inf),
            "min_gld_trades": np.full(len(period), np.inf),
        }
    elif not configs.equals(accumulator["configs"]):
        raise ValueError("Full-grid annual accumulator received a different config order.")

    accumulator["tested"] += 1
    accumulator["improved"] += period["Calmar Improvement"].to_numpy(dtype=np.int16)
    accumulator["calmar_sum"] += period["OOS Calmar"].to_numpy(dtype=float)
    accumulator["calmar_delta_sum"] += period["OOS Calmar Delta"].to_numpy(dtype=float)
    accumulator["worst_calmar_delta"] = np.minimum(
        accumulator["worst_calmar_delta"],
        period["OOS Calmar Delta"].to_numpy(dtype=float),
    )
    accumulator["worst_maxdd_delta"] = np.minimum(
        accumulator["worst_maxdd_delta"],
        period["OOS MaxDD Delta (%)"].to_numpy(dtype=float),
    )
    accumulator["exposure_sum"] += period["OOS Average Overlay Exposure (%)"].to_numpy(dtype=float)
    accumulator["active_days_sum"] += period["OOS Active Days"].to_numpy(dtype=float)
    accumulator["spy_active_days_sum"] += period["OOS SPY Active Days"].to_numpy(dtype=float)
    accumulator["gld_active_days_sum"] += period["OOS GLD Active Days"].to_numpy(dtype=float)
    accumulator["both_active_days_sum"] += period["OOS Both Active Days"].to_numpy(dtype=float)
    accumulator["cap_binding_days_sum"] += period["OOS Cap Binding Days"].to_numpy(dtype=float)
    accumulator["spy_active_days_pct_sum"] += period["OOS SPY Active Days (%)"].to_numpy(dtype=float)
    accumulator["gld_active_days_pct_sum"] += period["OOS GLD Active Days (%)"].to_numpy(dtype=float)
    accumulator["both_active_days_pct_sum"] += period["OOS Both Active Days (%)"].to_numpy(dtype=float)
    accumulator["cap_binding_days_pct_sum"] += period["OOS Cap Binding Days (%)"].to_numpy(dtype=float)
    accumulator["min_trades"] = np.minimum(
        accumulator["min_trades"],
        period["OOS Trade Episodes"].to_numpy(dtype=float),
    )
    accumulator["min_spy_trades"] = np.minimum(
        accumulator["min_spy_trades"],
        period["OOS SPY Trade Episodes"].to_numpy(dtype=float),
    )
    accumulator["min_gld_trades"] = np.minimum(
        accumulator["min_gld_trades"],
        period["OOS GLD Trade Episodes"].to_numpy(dtype=float),
    )
    return accumulator


def _finalize_full_grid_annual_accumulator(
    accumulator: dict[str, object] | None,
) -> pd.DataFrame:
    if accumulator is None:
        return pd.DataFrame()
    tested = accumulator["tested"].astype(float)
    out = accumulator["configs"].copy()
    out["Annual Years Tested"] = accumulator["tested"]
    out["Annual Calmar Improvement Years"] = accumulator["improved"]
    out["Average Annual Calmar"] = accumulator["calmar_sum"] / tested
    out["Average Annual Calmar Delta"] = accumulator["calmar_delta_sum"] / tested
    out["Worst Annual Calmar Delta"] = accumulator["worst_calmar_delta"]
    out["Worst Annual MaxDD Delta (%)"] = accumulator["worst_maxdd_delta"]
    out["Average Annual Exposure (%)"] = accumulator["exposure_sum"] / tested
    out["Average Annual Active Days"] = accumulator["active_days_sum"] / tested
    out["Average Annual SPY Active Days"] = accumulator["spy_active_days_sum"] / tested
    out["Average Annual GLD Active Days"] = accumulator["gld_active_days_sum"] / tested
    out["Average Annual Both Active Days"] = accumulator["both_active_days_sum"] / tested
    out["Average Annual Cap Binding Days"] = accumulator["cap_binding_days_sum"] / tested
    out["Average Annual SPY Active Days (%)"] = accumulator["spy_active_days_pct_sum"] / tested
    out["Average Annual GLD Active Days (%)"] = accumulator["gld_active_days_pct_sum"] / tested
    out["Average Annual Both Active Days (%)"] = accumulator["both_active_days_pct_sum"] / tested
    out["Average Annual Cap Binding Days (%)"] = accumulator["cap_binding_days_pct_sum"] / tested
    out["Min Annual Trade Episodes"] = accumulator["min_trades"].astype(int)
    out["Min Annual SPY Trade Episodes"] = accumulator["min_spy_trades"].astype(int)
    out["Min Annual GLD Trade Episodes"] = accumulator["min_gld_trades"].astype(int)
    out["SPY Rule"] = out.apply(lambda row: _row_rule_label(row, "SPY"), axis=1)
    out["GLD Rule"] = out.apply(lambda row: _row_rule_label(row, "GLD"), axis=1)
    out["Broker Profile"] = _broker_profile_series(out)
    return _round_float_cols(out)


def _full_grid_combined_summary(
    structural_summary: pd.DataFrame, annual_summary: pd.DataFrame
) -> pd.DataFrame:
    if structural_summary.empty:
        return pd.DataFrame()
    annual_cols = [
        "Annual Years Tested",
        "Annual Calmar Improvement Years",
        "Average Annual Calmar",
        "Average Annual Calmar Delta",
        "Worst Annual Calmar Delta",
        "Worst Annual MaxDD Delta (%)",
        "Average Annual Exposure (%)",
        "Average Annual Active Days",
        "Average Annual SPY Active Days",
        "Average Annual GLD Active Days",
        "Average Annual Both Active Days",
        "Average Annual Cap Binding Days",
        "Average Annual SPY Active Days (%)",
        "Average Annual GLD Active Days (%)",
        "Average Annual Both Active Days (%)",
        "Average Annual Cap Binding Days (%)",
        "Min Annual Trade Episodes",
        "Min Annual SPY Trade Episodes",
        "Min Annual GLD Trade Episodes",
    ]
    annual_merge_cols = CONFIG_COLS + [
        col for col in annual_cols if col in annual_summary.columns
    ]
    out = structural_summary.merge(
        annual_summary[annual_merge_cols]
        if not annual_summary.empty
        else pd.DataFrame(columns=CONFIG_COLS),
        on=CONFIG_COLS,
        how="left",
    )
    if "Annual Calmar Improvement Years" not in out:
        out["Annual Calmar Improvement Years"] = 0
    if "Annual Years Tested" not in out:
        out["Annual Years Tested"] = 0
    out["Overall Pass"] = (
        out["Structural Pass"]
        & (out["Annual Calmar Improvement Years"].fillna(0) >= 8)
    )
    return _round_float_cols(out)


def _full_grid_leaderboard(summary: pd.DataFrame, top_n: int = 1000) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    ranked = summary.sort_values(
        [
            "Overall Pass",
            "Average OOS Calmar",
            "Worst OOS Calmar Delta",
            "Worst OOS MaxDD Delta (%)",
            "Annual Calmar Improvement Years",
            "Average OOS Exposure (%)",
        ],
        ascending=[False, False, False, False, False, True],
    ).head(top_n).copy()
    ranked.insert(0, "Rank", range(1, len(ranked) + 1))
    first_cols = [
        "Rank",
        "Overall Pass",
        "Broker Profile",
        "SPY Rule",
        "GLD Rule",
        "Global Cap",
        "Average OOS Calmar",
        "Average OOS Calmar Delta",
        "Worst OOS Calmar Delta",
        "Worst OOS MaxDD Delta (%)",
        "Average OOS CAGR Delta (%)",
        "RF Cost Pass Splits",
        "Min OOS Trade Episodes",
        "Average OOS Exposure (%)",
        "Annual Calmar Improvement Years",
        "Annual Years Tested",
    ]
    ordered = [col for col in first_cols if col in ranked.columns]
    return ranked[ordered + [col for col in ranked.columns if col not in ordered]]


def _full_grid_annual_leaderboard(summary: pd.DataFrame, top_n: int = 1000) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    ranked = summary.sort_values(
        [
            "Annual Calmar Improvement Years",
            "Average Annual Calmar",
            "Worst Annual Calmar Delta",
            "Worst Annual MaxDD Delta (%)",
            "Average Annual Exposure (%)",
        ],
        ascending=[False, False, False, False, True],
    ).head(top_n).copy()
    ranked.insert(0, "Rank", range(1, len(ranked) + 1))
    return ranked


def _full_grid_compact_summary(structural: pd.DataFrame) -> pd.DataFrame:
    if structural.empty:
        return pd.DataFrame()
    group_cols = ["Broker Profile", "Global Cap"]
    out = structural.groupby(group_cols).agg(
        Rows=("Structural Splits Tested", "sum"),
        Configs=("SPY Entry", "count"),
        Avg_OOS_Calmar=("Average OOS Calmar", "mean"),
        Best_OOS_Calmar=("Average OOS Calmar", "max"),
        Avg_OOS_Calmar_Delta=("Average OOS Calmar Delta", "mean"),
        Pass_Rows=("Structural Splits Passed", "sum"),
    ).reset_index()
    return _round_float_cols(out)


def _broker_profile_series(data: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.select(
            [
                (data["Global Cap"] <= 0.20)
                & (data["SPY Weight"] <= 0.20)
                & (data["GLD Weight"] <= 0.20),
                (data["Global Cap"] <= 0.30)
                & (data["SPY Weight"] <= 0.30)
                & (data["GLD Weight"] <= 0.30),
            ],
            ["Strict Pilot", "IBKR Safe"],
            default="Research Unrestricted",
        ),
        index=data.index,
    )


def _round_float_cols(df: pd.DataFrame, ndigits: int = 4) -> pd.DataFrame:
    out = df.copy()
    float_cols = out.select_dtypes(include=["float"]).columns
    out[float_cols] = out[float_cols].round(ndigits)
    return out


# ---------------------------------------------------------------------------
# Fast mixed grid
# ---------------------------------------------------------------------------


def _fast_mixed_grid_vectorized(
    is_base: pd.Series,
    is_prices: pd.DataFrame,
    entry_grid: tuple[float, ...],
    exit_grid: tuple[float, ...],
    leverage_grid: tuple[float, ...],
    cap_grid: tuple[float, ...],
    spy_entry_grid: tuple[float, ...] | None = None,
    spy_exit_grid: tuple[float, ...] | None = None,
    spy_weight_grid: tuple[float, ...] | None = None,
    gld_entry_grid: tuple[float, ...] | None = None,
    gld_exit_grid: tuple[float, ...] | None = None,
    gld_weight_grid: tuple[float, ...] | None = None,
    lookback: int = 14,
    financing_cost_annual: float = 0.0,
    configs_per_chunk: int = 100_000,
    sort_output: bool = False,
    progress_label: str | None = None,
) -> pd.DataFrame:
    """Chunked NumPy full-grid scanner.

    This is used by the brute-force OOS job. It keeps the parameter grid intact
    but evaluates many configs per NumPy call, avoiding millions of Python-level
    daily-return loops.
    """
    for ticker in MIXED_TICKERS:
        if ticker not in is_prices.columns:
            raise KeyError(f"Mixed grid requires '{ticker}' in price frame.")

    entry_grids = {
        "SPY": spy_entry_grid or entry_grid,
        "GLD": gld_entry_grid or entry_grid,
    }
    exit_grids = {
        "SPY": spy_exit_grid or exit_grid,
        "GLD": gld_exit_grid or exit_grid,
    }
    weight_grids = {
        "SPY": spy_weight_grid or leverage_grid,
        "GLD": gld_weight_grid or leverage_grid,
    }

    base = is_base.dropna().astype(float).sort_index()
    if len(base) < 2:
        return pd.DataFrame()
    prices = is_prices.sort_index().astype(float).reindex(base.index)
    base_returns = base.pct_change().fillna(0.0).to_numpy(dtype=float)
    spy_returns = prices["SPY"].pct_change().fillna(0.0).to_numpy(dtype=float)
    gld_returns = prices["GLD"].pct_change().fillna(0.0).to_numpy(dtype=float)

    rules_by_ticker: dict[str, list[tuple[float, float]]] = {}
    signals_by_ticker: dict[str, np.ndarray] = {}
    for ticker in MIXED_TICKERS:
        rules = _valid_rule_pairs(entry_grids[ticker], exit_grids[ticker])
        rsi = compute_rsi(prices[ticker], lookback)
        signals = [
            generate_hysteresis_signal(rsi, entry, exit_)
            .shift(1)
            .fillna(0.0)
            .astype(float)
            .to_numpy(dtype=np.float32)
            for entry, exit_ in rules
        ]
        rules_by_ticker[ticker] = rules
        signals_by_ticker[ticker] = np.vstack(signals) if signals else np.empty((0, len(base)))

    configs = _full_grid_config_frame(
        spy_rules=rules_by_ticker["SPY"],
        gld_rules=rules_by_ticker["GLD"],
        spy_weight_grid=weight_grids["SPY"],
        gld_weight_grid=weight_grids["GLD"],
        cap_grid=cap_grid,
    )
    if configs.empty:
        return pd.DataFrame()

    month_end_positions = _month_end_positions(base.index)
    years = max((base.index[-1] - base.index[0]).days / 365.25, 1 / 365.25)
    base_stats = _grid_metrics_from_arrays(
        base.to_numpy(dtype=float),
        base_returns,
        month_end_positions,
        years,
    )
    daily_financing_rate = (
        (1.0 + financing_cost_annual) ** (1 / TRADING_DAYS_PER_YEAR) - 1.0
        if financing_cost_annual > 0
        else 0.0
    )

    chunk_size = max(1_000, int(configs_per_chunk))
    chunks = []
    chunk_total = int(np.ceil(len(configs) / chunk_size))
    for start in range(0, len(configs), chunk_size):
        chunk_started = time.perf_counter()
        chunk = configs.iloc[start : start + chunk_size].copy()
        metrics = _evaluate_mixed_config_chunk(
            chunk=chunk,
            spy_signals=signals_by_ticker["SPY"],
            gld_signals=signals_by_ticker["GLD"],
            base_returns=base_returns,
            spy_returns=spy_returns,
            gld_returns=gld_returns,
            base_start=float(base.iloc[0]),
            month_end_positions=month_end_positions,
            years=years,
            base_stats=base_stats,
            daily_financing_rate=daily_financing_rate,
        )
        chunks.append(pd.concat([chunk[CONFIG_COLS].reset_index(drop=True), metrics], axis=1))
        if progress_label:
            chunk_no = start // chunk_size + 1
            print(
                f"[full-grid-chunk] {progress_label}: {chunk_no:,}/{chunk_total:,} "
                f"({min(start + chunk_size, len(configs)):,}/{len(configs):,}) in "
                f"{time.perf_counter() - chunk_started:.1f}s",
                flush=True,
            )

    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def _valid_rule_pairs(
    entry_grid: tuple[float, ...], exit_grid: tuple[float, ...]
) -> list[tuple[float, float]]:
    return [
        (float(entry), float(exit_))
        for entry in entry_grid
        for exit_ in exit_grid
        if float(exit_) > float(entry)
    ]


def _full_grid_config_frame(
    spy_rules: list[tuple[float, float]],
    gld_rules: list[tuple[float, float]],
    spy_weight_grid: tuple[float, ...],
    gld_weight_grid: tuple[float, ...],
    cap_grid: tuple[float, ...],
) -> pd.DataFrame:
    if not spy_rules or not gld_rules:
        return pd.DataFrame()
    product = pd.MultiIndex.from_product(
        [
            range(len(spy_rules)),
            range(len(gld_rules)),
            tuple(float(x) for x in spy_weight_grid),
            tuple(float(x) for x in gld_weight_grid),
            tuple(float(x) for x in cap_grid),
        ],
        names=[
            "SPY Rule Index",
            "GLD Rule Index",
            "SPY Weight",
            "GLD Weight",
            "Global Cap",
        ],
    ).to_frame(index=False)
    spy_entries = np.array([entry for entry, _ in spy_rules], dtype=float)
    spy_exits = np.array([exit_ for _, exit_ in spy_rules], dtype=float)
    gld_entries = np.array([entry for entry, _ in gld_rules], dtype=float)
    gld_exits = np.array([exit_ for _, exit_ in gld_rules], dtype=float)
    spy_idx = product["SPY Rule Index"].to_numpy(dtype=np.intp)
    gld_idx = product["GLD Rule Index"].to_numpy(dtype=np.intp)
    product["SPY Entry"] = spy_entries[spy_idx]
    product["SPY Exit"] = spy_exits[spy_idx]
    product["GLD Entry"] = gld_entries[gld_idx]
    product["GLD Exit"] = gld_exits[gld_idx]
    return product[
        [
            "SPY Rule Index",
            "GLD Rule Index",
            "SPY Entry",
            "SPY Exit",
            "SPY Weight",
            "GLD Entry",
            "GLD Exit",
            "GLD Weight",
            "Global Cap",
        ]
    ]


def _evaluate_mixed_config_chunk(
    chunk: pd.DataFrame,
    spy_signals: np.ndarray,
    gld_signals: np.ndarray,
    base_returns: np.ndarray,
    spy_returns: np.ndarray,
    gld_returns: np.ndarray,
    base_start: float,
    month_end_positions: np.ndarray,
    years: float,
    base_stats: dict[str, float],
    daily_financing_rate: float,
) -> pd.DataFrame:
    spy_idx = chunk["SPY Rule Index"].to_numpy(dtype=np.intp)
    gld_idx = chunk["GLD Rule Index"].to_numpy(dtype=np.intp)
    spy_weight = chunk["SPY Weight"].to_numpy(dtype=float)
    gld_weight = chunk["GLD Weight"].to_numpy(dtype=float)
    cap = chunk["Global Cap"].to_numpy(dtype=float)
    n_configs = len(chunk)
    n_days = len(base_returns)

    current_values = np.full(n_configs, base_start, dtype=float)
    running_max = np.full(n_configs, base_start, dtype=float)
    max_dd = np.zeros(n_configs, dtype=float)
    return_sum = np.zeros(n_configs, dtype=float)
    return_sumsq = np.zeros(n_configs, dtype=float)
    return_count = 0
    active_days = np.zeros(n_configs, dtype=np.int32)
    spy_active_days = np.zeros(n_configs, dtype=np.int32)
    gld_active_days = np.zeros(n_configs, dtype=np.int32)
    both_active_days = np.zeros(n_configs, dtype=np.int32)
    cap_binding_days = np.zeros(n_configs, dtype=np.int32)
    exposure_sum = np.zeros(n_configs, dtype=float)
    max_exposure = np.zeros(n_configs, dtype=float)
    trade_episodes = np.zeros(n_configs, dtype=np.int32)
    spy_trade_episodes = np.zeros(n_configs, dtype=np.int32)
    gld_trade_episodes = np.zeros(n_configs, dtype=np.int32)
    was_active = np.zeros(n_configs, dtype=bool)
    was_spy_active = np.zeros(n_configs, dtype=bool)
    was_gld_active = np.zeros(n_configs, dtype=bool)
    previous_month_values: np.ndarray | None = None
    worst_month = np.full(n_configs, np.nan)
    month_end_set = set(int(pos) for pos in month_end_positions)

    for t in range(n_days):
        spy_desired = spy_signals[spy_idx, t].astype(float, copy=False) * spy_weight
        gld_desired = gld_signals[gld_idx, t].astype(float, copy=False) * gld_weight
        desired_sum = spy_desired + gld_desired
        scale = np.ones(n_configs, dtype=float)
        np.divide(cap, desired_sum, out=scale, where=desired_sum > cap)
        scale = np.minimum(scale, 1.0)
        spy_pos = spy_desired * scale
        gld_pos = gld_desired * scale
        overlay_exposure = spy_pos + gld_pos
        strategy_returns = (
            base_returns[t]
            + spy_pos * spy_returns[t]
            + gld_pos * gld_returns[t]
        )
        if daily_financing_rate > 0:
            strategy_returns = strategy_returns - overlay_exposure * daily_financing_rate

        if t > 0:
            current_values *= 1.0 + strategy_returns
            running_max = np.maximum(running_max, current_values)
            max_dd = np.minimum(max_dd, current_values / running_max - 1.0)
            return_sum += strategy_returns
            return_sumsq += strategy_returns * strategy_returns
            return_count += 1

        active = overlay_exposure > 0
        spy_active = spy_pos > 0
        gld_active = gld_pos > 0
        active_days += active
        spy_active_days += spy_active
        gld_active_days += gld_active
        both_active_days += spy_active & gld_active
        cap_binding_days += (desired_sum > cap) & active
        exposure_sum += overlay_exposure
        max_exposure = np.maximum(max_exposure, overlay_exposure)
        trade_episodes += active & ~was_active
        spy_trade_episodes += spy_active & ~was_spy_active
        gld_trade_episodes += gld_active & ~was_gld_active
        was_active = active
        was_spy_active = spy_active
        was_gld_active = gld_active

        if t in month_end_set:
            if previous_month_values is not None:
                monthly = (current_values / previous_month_values - 1.0) * 100
                worst_month = np.fmin(worst_month, monthly)
            previous_month_values = current_values.copy()

    total_return = (current_values / base_start - 1.0) * 100
    cagr = np.where(
        current_values > 0,
        ((current_values / base_start) ** (1 / years) - 1.0) * 100,
        np.nan,
    )
    daily_mean = return_sum / max(return_count, 1)
    variance = (
        (return_sumsq - (return_sum * return_sum / return_count)) / (return_count - 1)
        if return_count > 1
        else np.zeros(n_configs)
    )
    daily_std = np.sqrt(np.maximum(variance, 0.0))
    rf_daily = (1.0 + config.RISK_FREE_RATE) ** (1 / TRADING_DAYS_PER_YEAR) - 1.0
    sharpe = np.full(n_configs, np.nan)
    valid_std = daily_std > 1e-12
    sharpe[valid_std] = (
        (daily_mean[valid_std] - rf_daily)
        / daily_std[valid_std]
        * np.sqrt(TRADING_DAYS_PER_YEAR)
    )
    max_dd_pct = max_dd * 100
    calmar = np.full(n_configs, np.nan)
    valid_dd = np.abs(max_dd_pct) > 1e-12
    calmar[valid_dd] = cagr[valid_dd] / np.abs(max_dd_pct[valid_dd])

    metrics = {
        "Active Days": active_days.astype(int),
        "Active Days (%)": np.round(active_days / max(n_days, 1) * 100, 4),
        "SPY Active Days": spy_active_days.astype(int),
        "SPY Active Days (%)": np.round(spy_active_days / max(n_days, 1) * 100, 4),
        "GLD Active Days": gld_active_days.astype(int),
        "GLD Active Days (%)": np.round(gld_active_days / max(n_days, 1) * 100, 4),
        "Both Active Days": both_active_days.astype(int),
        "Both Active Days (%)": np.round(both_active_days / max(n_days, 1) * 100, 4),
        "Cap Binding Days": cap_binding_days.astype(int),
        "Cap Binding Days (%)": np.round(cap_binding_days / max(n_days, 1) * 100, 4),
        "Average Overlay Exposure (%)": np.round(exposure_sum / max(n_days, 1) * 100, 4),
        "Max Overlay Exposure (%)": np.round(max_exposure * 100, 4),
        "Trade Episodes": trade_episodes.astype(int),
        "SPY Trade Episodes": spy_trade_episodes.astype(int),
        "GLD Trade Episodes": gld_trade_episodes.astype(int),
        "CAGR (%)": np.round(cagr, 4),
        "Sharpe": np.round(sharpe, 4),
        "Calmar": np.round(calmar, 4),
        "Max Drawdown (%)": np.round(max_dd_pct, 4),
        "Worst Month (%)": np.round(worst_month, 4),
        "Total Return (%)": np.round(total_return, 4),
    }
    metrics["Incremental CAGR (%)"] = np.round(metrics["CAGR (%)"] - base_stats["CAGR (%)"], 4)
    metrics["Incremental Calmar"] = np.round(metrics["Calmar"] - base_stats["Calmar"], 4)
    metrics["Incremental MaxDD (%)"] = np.round(
        metrics["Max Drawdown (%)"] - base_stats["Max Drawdown (%)"],
        4,
    )
    return pd.DataFrame(metrics)


def _grid_metrics_matrix(
    values: np.ndarray,
    returns: np.ndarray,
    month_end_positions: np.ndarray,
    years: float,
) -> dict[str, np.ndarray]:
    start = values[:, 0]
    end = values[:, -1]
    valid_growth = (start > 0) & (end > 0)
    total_return = np.where(valid_growth, (end / start - 1.0) * 100, np.nan)
    cagr = np.where(valid_growth, ((end / start) ** (1 / years) - 1.0) * 100, np.nan)

    running_max = np.maximum.accumulate(values, axis=1)
    max_dd = np.min(values / running_max - 1.0, axis=1) * 100

    daily = returns[:, 1:]
    daily_std = np.std(daily, axis=1, ddof=1) if daily.shape[1] > 1 else np.zeros(len(values))
    rf_daily = (1.0 + config.RISK_FREE_RATE) ** (1 / TRADING_DAYS_PER_YEAR) - 1.0
    sharpe = np.full(len(values), np.nan)
    valid_std = daily_std > 1e-12
    sharpe[valid_std] = (
        (np.mean(daily[valid_std], axis=1) - rf_daily)
        / daily_std[valid_std]
        * np.sqrt(TRADING_DAYS_PER_YEAR)
    )

    if len(month_end_positions) >= 2:
        month_values = values[:, month_end_positions]
        monthly = (month_values[:, 1:] / month_values[:, :-1] - 1.0) * 100
        worst_month = np.min(monthly, axis=1)
    else:
        worst_month = np.full(len(values), np.nan)

    calmar = np.full(len(values), np.nan)
    valid_dd = np.abs(max_dd) > 1e-12
    calmar[valid_dd] = cagr[valid_dd] / np.abs(max_dd[valid_dd])
    return {
        "Active Days (%)": np.empty(len(values)),
        "Both Active Days (%)": np.empty(len(values)),
        "Average Overlay Exposure (%)": np.empty(len(values)),
        "Max Overlay Exposure (%)": np.empty(len(values)),
        "Trade Episodes": np.empty(len(values), dtype=int),
        "CAGR (%)": np.round(cagr, 4),
        "Sharpe": np.round(sharpe, 4),
        "Calmar": np.round(calmar, 4),
        "Max Drawdown (%)": np.round(max_dd, 4),
        "Worst Month (%)": np.round(worst_month, 4),
        "Total Return (%)": np.round(total_return, 4),
    }


def _fast_mixed_grid(
    is_base: pd.Series,
    is_prices: pd.DataFrame,
    entry_grid: tuple[float, ...],
    exit_grid: tuple[float, ...],
    leverage_grid: tuple[float, ...],
    cap_grid: tuple[float, ...],
    spy_entry_grid: tuple[float, ...] | None = None,
    spy_exit_grid: tuple[float, ...] | None = None,
    spy_weight_grid: tuple[float, ...] | None = None,
    gld_entry_grid: tuple[float, ...] | None = None,
    gld_exit_grid: tuple[float, ...] | None = None,
    gld_weight_grid: tuple[float, ...] | None = None,
    lookback: int = 14,
    financing_cost_annual: float = 0.0,
    chunk_size: int = 50_000,
    sort_output: bool = True,
) -> pd.DataFrame:
    """Vectorised SPY+GLD grid scanner on an IS window.

    Pre-computes RSI and per-(entry, exit) lagged signals once per ticker, then
    iterates over (spy_weight, gld_weight, cap) tuples using pure vector math.
    """
    for ticker in MIXED_TICKERS:
        if ticker not in is_prices.columns:
            raise KeyError(f"Mixed grid requires '{ticker}' in price frame.")

    entry_grids = {
        "SPY": spy_entry_grid or entry_grid,
        "GLD": gld_entry_grid or entry_grid,
    }
    exit_grids = {
        "SPY": spy_exit_grid or exit_grid,
        "GLD": gld_exit_grid or exit_grid,
    }
    weight_grids = {
        "SPY": spy_weight_grid or leverage_grid,
        "GLD": gld_weight_grid or leverage_grid,
    }

    base = is_base.dropna().astype(float).sort_index()
    if len(base) < 2:
        return pd.DataFrame()
    prices = is_prices.sort_index().astype(float).reindex(base.index)
    base_returns = base.pct_change().fillna(0.0).to_numpy(dtype=float)
    daily_financing_rate = (
        (1.0 + financing_cost_annual) ** (1 / TRADING_DAYS_PER_YEAR) - 1.0
        if financing_cost_annual > 0
        else 0.0
    )
    asset_returns = {
        ticker: prices[ticker].pct_change().fillna(0.0).to_numpy(dtype=float)
        for ticker in MIXED_TICKERS
    }
    rsi_by_ticker = {
        ticker: compute_rsi(prices[ticker], lookback) for ticker in MIXED_TICKERS
    }

    # Pre-compute lagged 0/1 signals for every (entry, exit) combo per ticker.
    signals_by_ticker: dict[str, dict[tuple[float, float], np.ndarray]] = {}
    for ticker in MIXED_TICKERS:
        rsi = rsi_by_ticker[ticker]
        signals_by_ticker[ticker] = {}
        for entry in entry_grids[ticker]:
            for exit_ in exit_grids[ticker]:
                if exit_ <= entry:
                    continue
                raw = generate_hysteresis_signal(rsi, entry, exit_)
                lagged = raw.shift(1).fillna(0.0).astype(float).to_numpy()
                signals_by_ticker[ticker][(entry, exit_)] = lagged

    month_end_positions = _month_end_positions(base.index)
    years = max((base.index[-1] - base.index[0]).days / 365.25, 1 / 365.25)
    base_stats = _grid_metrics_from_arrays(
        base.to_numpy(dtype=float),
        base_returns,
        month_end_positions,
        years,
    )
    base_start = float(base.iloc[0])
    rows = []
    chunks: list[pd.DataFrame] = []
    spy_returns = asset_returns["SPY"]
    gld_returns = asset_returns["GLD"]

    for (spy_entry, spy_exit), spy_signal in signals_by_ticker["SPY"].items():
        for (gld_entry, gld_exit), gld_signal in signals_by_ticker["GLD"].items():
            for spy_weight in weight_grids["SPY"]:
                spy_desired = spy_signal * spy_weight
                for gld_weight in weight_grids["GLD"]:
                    gld_desired = gld_signal * gld_weight
                    desired_sum = spy_desired + gld_desired
                    for cap in cap_grid:
                        if cap <= 0:
                            continue
                        scale = np.ones_like(desired_sum, dtype=float)
                        over_cap = desired_sum > cap
                        scale[over_cap] = cap / desired_sum[over_cap]
                        spy_pos = spy_desired * scale
                        gld_pos = gld_desired * scale
                        overlay_returns = (
                            spy_pos * spy_returns
                            + gld_pos * gld_returns
                        )
                        strategy_returns = base_returns + overlay_returns
                        values_arr = base_start * np.cumprod(1.0 + strategy_returns)
                        values_arr[0] = base_start

                        metrics = _grid_metrics_from_arrays(
                            values_arr,
                            strategy_returns,
                            month_end_positions,
                            years,
                        )
                        overlay_exposure = spy_pos + gld_pos
                        if daily_financing_rate > 0:
                            strategy_returns = strategy_returns - overlay_exposure * daily_financing_rate
                            values_arr = base_start * np.cumprod(1.0 + strategy_returns)
                            values_arr[0] = base_start
                            metrics = _grid_metrics_from_arrays(
                                values_arr,
                                strategy_returns,
                                month_end_positions,
                                years,
                            )
                        active_pct = float(np.mean(overlay_exposure > 0) * 100)
                        avg_exposure = float(np.mean(overlay_exposure) * 100)
                        max_exposure = float(np.max(overlay_exposure) * 100)
                        both_active_pct = float(
                            np.mean((spy_pos > 0) & (gld_pos > 0)) * 100
                        )
                        trade_episodes = int(
                            np.sum(
                                (overlay_exposure > 0)
                                & np.r_[True, overlay_exposure[:-1] <= 0]
                            )
                        )
                        rows.append(
                            {
                                "SPY Entry": spy_entry,
                                "SPY Exit": spy_exit,
                                "SPY Weight": spy_weight,
                                "GLD Entry": gld_entry,
                                "GLD Exit": gld_exit,
                                "GLD Weight": gld_weight,
                                "Global Cap": cap,
                                "Active Days (%)": round(active_pct, 4),
                                "Both Active Days (%)": round(both_active_pct, 4),
                                "Average Overlay Exposure (%)": round(avg_exposure, 4),
                                "Max Overlay Exposure (%)": round(max_exposure, 4),
                                "Trade Episodes": trade_episodes,
                                "CAGR (%)": metrics["CAGR (%)"],
                                "Sharpe": metrics["Sharpe"],
                                "Calmar": metrics["Calmar"],
                                "Max Drawdown (%)": metrics["Max Drawdown (%)"],
                                "Worst Month (%)": metrics["Worst Month (%)"],
                                "Total Return (%)": metrics["Total Return (%)"],
                                "Incremental CAGR (%)": round(
                                    metrics["CAGR (%)"] - base_stats["CAGR (%)"], 4
                                ),
                                "Incremental Calmar": round(
                                    metrics["Calmar"] - base_stats["Calmar"], 4
                                ),
                                "Incremental MaxDD (%)": round(
                                    metrics["Max Drawdown (%)"]
                                    - base_stats["Max Drawdown (%)"],
                                    4,
                                ),
                            }
                        )
                        if len(rows) >= chunk_size:
                            chunks.append(pd.DataFrame(rows))
                            rows = []

    if rows:
        chunks.append(pd.DataFrame(rows))
    out = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    if sort_output and not out.empty:
        out = out.sort_values(
            ["Calmar", "Max Drawdown (%)", "CAGR (%)"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Selectors
# ---------------------------------------------------------------------------


def select_mixed_rules(
    grid: pd.DataFrame, base_metrics: dict[str, float]
) -> pd.DataFrame:
    """Return one row per selector picked from the IS mixed grid."""
    if grid.empty:
        return pd.DataFrame()

    selections: list[pd.Series] = []

    default = grid[
        np.isclose(grid["SPY Entry"], 30.0)
        & np.isclose(grid["SPY Exit"], 50.0)
        & np.isclose(grid["SPY Weight"], 0.20)
        & np.isclose(grid["GLD Entry"], 30.0)
        & np.isclose(grid["GLD Exit"], 50.0)
        & np.isclose(grid["GLD Weight"], 0.20)
        & np.isclose(grid["Global Cap"], 0.20)
    ]
    if not default.empty:
        selections.append(_mark_selection(default.iloc[0], "default_30_50_20"))

    best_calmar = grid.sort_values(
        ["Calmar", "CAGR (%)"], ascending=[False, False]
    ).iloc[0]
    selections.append(_mark_selection(best_calmar, "best_calmar"))

    best_maxdd = grid.sort_values(
        ["Max Drawdown (%)", "Calmar"], ascending=[False, False]
    ).iloc[0]
    selections.append(_mark_selection(best_maxdd, "best_maxdd_preservation"))

    maxdd_floor = float(base_metrics["Max Drawdown (%)"]) - 1.0
    guarded = grid[grid["Max Drawdown (%)"] >= maxdd_floor]
    if guarded.empty:
        fallback = grid.sort_values(
            ["Max Drawdown (%)", "CAGR (%)"], ascending=[False, False]
        ).iloc[0]
        selections.append(
            _mark_selection(
                fallback,
                "best_cagr_with_maxdd_guard",
                warning="no row satisfied the 1pp MaxDD guard",
                guard_passed=False,
            )
        )
    else:
        best_guard = guarded.sort_values(
            ["CAGR (%)", "Calmar"], ascending=[False, False]
        ).iloc[0]
        selections.append(
            _mark_selection(best_guard, "best_cagr_with_maxdd_guard", guard_passed=True)
        )

    robust = _robust_calmar_selection(grid)
    selections.append(
        _mark_selection(
            robust["row"],
            "robust_calmar_region",
            robust_avg_calmar=robust["avg_calmar"],
            robust_neighborhood_size=robust["size"],
        )
    )
    simple = _simple_stable_selection(grid, robust)
    if not simple["row"].empty:
        selections.append(
            _mark_selection(
                simple["row"],
                "simple_stable_region",
                robust_avg_calmar=simple["threshold_calmar"],
                robust_neighborhood_size=simple["eligible_rows"],
            )
        )

    out = pd.DataFrame([s.to_dict() for s in selections])
    first_cols = [
        "Selector",
        "SPY Entry",
        "SPY Exit",
        "SPY Weight",
        "GLD Entry",
        "GLD Exit",
        "GLD Weight",
        "Global Cap",
        "Selection Warning",
        "IS MaxDD Guard Passed",
        "Robust Avg Calmar",
        "Robust Neighborhood Size",
    ]
    return out[first_cols + [c for c in out.columns if c not in first_cols]]


def _robust_calmar_selection(grid: pd.DataFrame) -> dict[str, object]:
    """Pick a config whose Calmar is stable across a small parameter neighborhood."""
    if grid.empty:
        return {"row": pd.Series(dtype=object), "avg_calmar": np.nan, "size": 0}

    arr = grid.reset_index(drop=True)
    param_cols = [
        "SPY Entry", "SPY Exit", "SPY Weight",
        "GLD Entry", "GLD Exit", "GLD Weight", "Global Cap",
    ]
    windows = [4.0, 8.0, 0.05, 4.0, 8.0, 0.05, 0.05]
    uniques = [
        np.sort(pd.to_numeric(arr[col], errors="coerce").dropna().unique())
        for col in param_cols
    ]
    radii = [_regular_grid_radius(values, window) for values, window in zip(uniques, windows)]
    if any(radius is None for radius in radii):
        return _robust_calmar_selection_slow(arr)

    shape = tuple(len(values) for values in uniques)
    coords = tuple(
        np.searchsorted(values, pd.to_numeric(arr[col], errors="coerce").to_numpy())
        for col, values in zip(param_cols, uniques)
    )
    calmar = pd.to_numeric(arr["Calmar"], errors="coerce").to_numpy(dtype=float)
    cagr = pd.to_numeric(arr["CAGR (%)"], errors="coerce").to_numpy(dtype=float)

    calmar_cube = np.full(shape, np.nan, dtype=float)
    calmar_cube[coords] = calmar
    valid_cube = np.isfinite(calmar_cube).astype(float)
    calmar_sum = np.nan_to_num(calmar_cube, nan=0.0)
    count_cube = valid_cube
    for axis, radius in enumerate(radii):
        calmar_sum = _window_sum_axis(calmar_sum, int(radius), axis)
        count_cube = _window_sum_axis(count_cube, int(radius), axis)

    with np.errstate(invalid="ignore", divide="ignore"):
        avg_cube = calmar_sum / count_cube
    avg_calmar = avg_cube[coords]
    neighborhood_size = count_cube[coords].astype(int)

    score_avg = np.nan_to_num(avg_calmar, nan=-np.inf)
    score_calmar = np.nan_to_num(calmar, nan=-np.inf)
    score_cagr = np.nan_to_num(cagr, nan=-np.inf)
    best_idx = int(np.lexsort((score_cagr, score_calmar, score_avg))[-1])
    best_row = arr.iloc[best_idx]
    best_avg = float(avg_calmar[best_idx])
    best_size = int(neighborhood_size[best_idx])
    return {
        "row": best_row,
        "avg_calmar": round(best_avg, 6) if np.isfinite(best_avg) else np.nan,
        "size": best_size,
    }


def _regular_grid_radius(values: np.ndarray, window: float) -> int | None:
    if len(values) <= 1:
        return 0
    steps = np.diff(values.astype(float))
    step = float(steps[0])
    if step <= 0 or not np.allclose(steps, step, rtol=1e-8, atol=1e-12):
        return None
    return int(np.floor((window + 1e-12) / step))


def _window_sum_axis(values: np.ndarray, radius: int, axis: int) -> np.ndarray:
    if radius <= 0:
        return values
    pad_width = [(0, 0)] * values.ndim
    pad_width[axis] = (radius, radius)
    padded = np.pad(values, pad_width, mode="constant", constant_values=0.0)
    cumsum = np.cumsum(padded, axis=axis)
    zero_shape = list(cumsum.shape)
    zero_shape[axis] = 1
    cumsum = np.concatenate([np.zeros(zero_shape, dtype=values.dtype), cumsum], axis=axis)
    length = values.shape[axis]
    window = radius * 2 + 1
    high = np.take(cumsum, np.arange(window, window + length), axis=axis)
    low = np.take(cumsum, np.arange(0, length), axis=axis)
    return high - low


def _robust_calmar_selection_slow(grid: pd.DataFrame) -> dict[str, object]:
    """Fallback for irregular grids; intended for tiny ad hoc test grids."""
    best_score = None
    best_row = grid.iloc[0]
    best_size = 0
    spy_entry = grid["SPY Entry"].to_numpy()
    spy_exit = grid["SPY Exit"].to_numpy()
    spy_weight = grid["SPY Weight"].to_numpy()
    gld_entry = grid["GLD Entry"].to_numpy()
    gld_exit = grid["GLD Exit"].to_numpy()
    gld_weight = grid["GLD Weight"].to_numpy()
    cap = grid["Global Cap"].to_numpy()
    calmar = grid["Calmar"].to_numpy()
    cagr = grid["CAGR (%)"].to_numpy()

    for i in range(len(grid)):
        mask = (
            (np.abs(spy_entry - spy_entry[i]) <= 4.0 + 1e-12)
            & (np.abs(spy_exit - spy_exit[i]) <= 8.0 + 1e-12)
            & (np.abs(spy_weight - spy_weight[i]) <= 0.05 + 1e-12)
            & (np.abs(gld_entry - gld_entry[i]) <= 4.0 + 1e-12)
            & (np.abs(gld_exit - gld_exit[i]) <= 8.0 + 1e-12)
            & (np.abs(gld_weight - gld_weight[i]) <= 0.05 + 1e-12)
            & (np.abs(cap - cap[i]) <= 0.05 + 1e-12)
        )
        avg_calmar = float(np.nanmean(calmar[mask])) if mask.any() else float("nan")
        score = (avg_calmar, float(calmar[i]), float(cagr[i]))
        if best_score is None or score > best_score:
            best_score = score
            best_row = grid.iloc[i]
            best_size = int(mask.sum())
    return {
        "row": best_row,
        "avg_calmar": round(float(best_score[0]), 6) if best_score is not None else np.nan,
        "size": best_size,
    }


def _simple_stable_selection(
    grid: pd.DataFrame,
    robust: dict[str, object],
) -> dict[str, object]:
    """Pick the lowest-exposure row within 95% of the robust-region Calmar."""
    robust_calmar = float(robust.get("avg_calmar", np.nan))
    robust_row = robust.get("row", pd.Series(dtype=object))
    if grid.empty:
        return {
            "row": pd.Series(dtype=object),
            "threshold_calmar": np.nan,
            "eligible_rows": 0,
        }
    if not np.isfinite(robust_calmar) and isinstance(robust_row, pd.Series) and "Calmar" in robust_row:
        robust_calmar = float(robust_row["Calmar"])
    if not np.isfinite(robust_calmar):
        robust_calmar = float(grid["Calmar"].max())
    if not np.isfinite(robust_calmar):
        selected = grid.sort_values(
            ["Average Overlay Exposure (%)", "Global Cap"],
            ascending=[True, True],
        ).iloc[0]
        return {
            "row": selected,
            "threshold_calmar": np.nan,
            "eligible_rows": int(len(grid)),
        }
    floor = robust_calmar * 0.95 if robust_calmar >= 0 else robust_calmar * 1.05
    eligible = grid[grid["Calmar"] >= floor].copy()
    if eligible.empty:
        return {
            "row": pd.Series(dtype=object),
            "threshold_calmar": round(floor, 6),
            "eligible_rows": 0,
        }
    selected = eligible.sort_values(
        ["Average Overlay Exposure (%)", "Global Cap", "Calmar", "CAGR (%)"],
        ascending=[True, True, False, False],
    ).iloc[0]
    return {
        "row": selected,
        "threshold_calmar": round(floor, 6),
        "eligible_rows": int(len(eligible)),
    }


def _mark_selection(
    row: pd.Series,
    selector: str,
    warning: str = "",
    guard_passed: bool | None = None,
    robust_avg_calmar: float | None = None,
    robust_neighborhood_size: int | None = None,
) -> pd.Series:
    out = row.copy()
    out["Selector"] = selector
    out["Selection Warning"] = warning
    out["IS MaxDD Guard Passed"] = (
        guard_passed if guard_passed is not None else pd.NA
    )
    out["Robust Avg Calmar"] = (
        robust_avg_calmar if robust_avg_calmar is not None else pd.NA
    )
    out["Robust Neighborhood Size"] = (
        robust_neighborhood_size if robust_neighborhood_size is not None else pd.NA
    )
    return out


def _evaluate_rule_row(
    rule: pd.Series,
    is_base: pd.Series,
    is_prices: pd.DataFrame,
    oos_base: pd.Series,
    oos_prices: pd.DataFrame,
    base_metrics: dict[str, float],
    split: str,
    label: str,
    funding_spread: float = 0.0,
) -> dict[str, object]:
    specs = [
        OverlaySpec(
            ticker="SPY",
            entry_threshold=float(rule["SPY Entry"]),
            exit_threshold=float(rule["SPY Exit"]),
            overlay_weight=float(rule["SPY Weight"]),
        ),
        OverlaySpec(
            ticker="GLD",
            entry_threshold=float(rule["GLD Entry"]),
            exit_threshold=float(rule["GLD Exit"]),
            overlay_weight=float(rule["GLD Weight"]),
        ),
    ]
    return _evaluate_mixed_oos(
        is_base=is_base,
        is_prices=is_prices,
        oos_base=oos_base,
        oos_prices=oos_prices,
        specs=specs,
        global_cap=float(rule["Global Cap"]),
        label=label,
        selector=str(rule["Selector"]),
        base_metrics=base_metrics,
        split=split,
        funding_spread=funding_spread,
        extra={
            "Candidate Name": label,
            "Robust Avg Calmar": rule.get("Robust Avg Calmar", pd.NA),
            "Robust Neighborhood Size": rule.get("Robust Neighborhood Size", pd.NA),
        },
    )


def _walk_forward_summary(
    prices: pd.DataFrame,
    allocation: dict[str, float],
    years: tuple[int, ...],
    apply_fees: bool,
    spy_entry_grid: tuple[float, ...],
    spy_exit_grid: tuple[float, ...],
    spy_weight_grid: tuple[float, ...],
    gld_entry_grid: tuple[float, ...],
    gld_exit_grid: tuple[float, ...],
    gld_weight_grid: tuple[float, ...],
    cap_grid: tuple[float, ...],
    funding_spread: float = 0.0,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for year in years:
        train_end = pd.Timestamp(f"{year}-01-01")
        eval_end = pd.Timestamp(f"{year + 1}-01-01")
        train_prices = prices.loc[prices.index < train_end]
        eval_prices = prices.loc[(prices.index >= train_end) & (prices.index < eval_end)]
        if train_prices.empty or eval_prices.empty:
            continue
        is_base = _base_series(train_prices, allocation, apply_fees)
        oos_base = _base_series(eval_prices, allocation, apply_fees)
        if is_base.empty or oos_base.empty:
            continue
        grid = _fast_mixed_grid(
            is_base=is_base,
            is_prices=train_prices,
            entry_grid=DEFAULT_MIXED_ENTRY_GRID,
            exit_grid=DEFAULT_MIXED_EXIT_GRID,
            leverage_grid=DEFAULT_MIXED_LEVERAGE_GRID,
            cap_grid=cap_grid,
            spy_entry_grid=spy_entry_grid,
            spy_exit_grid=spy_exit_grid,
            spy_weight_grid=spy_weight_grid,
            gld_entry_grid=gld_entry_grid,
            gld_exit_grid=gld_exit_grid,
            gld_weight_grid=gld_weight_grid,
            sort_output=False,
        )
        if grid.empty:
            continue
        selections = select_mixed_rules(grid, _strategy_metrics(is_base, BASE_LABEL))
        base_metrics = _strategy_metrics(oos_base, BASE_LABEL)
        for _, rule in selections.iterrows():
            eval_out = _evaluate_rule_row(
                rule=rule,
                is_base=is_base,
                is_prices=train_prices,
                oos_base=oos_base,
                oos_prices=eval_prices,
                base_metrics=base_metrics,
                split=f"{year}-01-01",
                label=f"walk-forward {rule['Selector']} ({year})",
                funding_spread=funding_spread,
            )
            summary = eval_out["summary"]
            summary["Year"] = int(year)
            summary["Is Partial Year"] = bool(year == 2026)
            summary["Calmar Improvement"] = bool(summary["OOS Calmar Delta"] > 0)
            summary["Config Key"] = _config_key(summary)
            rows.append(summary)
    return pd.DataFrame(rows)


def fixed_candidate_walk_forward_summary(
    prices: pd.DataFrame,
    allocation: dict[str, float],
    candidates: list[MixedOverlayCandidate],
    years: tuple[int, ...],
    apply_fees: bool,
    funding_spread: float = 0.0,
) -> pd.DataFrame:
    """Evaluate fixed mixed candidates one calendar year at a time."""
    rows: list[dict[str, object]] = []
    for year in years:
        eval_start = pd.Timestamp(f"{year}-01-01")
        eval_end = pd.Timestamp(f"{year + 1}-01-01")
        is_prices = prices.loc[prices.index < eval_start]
        oos_prices = prices.loc[(prices.index >= eval_start) & (prices.index < eval_end)]
        if is_prices.empty or oos_prices.empty:
            continue
        is_base = _base_series(is_prices, allocation, apply_fees)
        oos_base = _base_series(oos_prices, allocation, apply_fees)
        if is_base.empty or oos_base.empty:
            continue
        base_metrics = _strategy_metrics(oos_base, BASE_LABEL)
        for candidate in candidates:
            eval_out = _evaluate_mixed_oos(
                is_base=is_base,
                is_prices=is_prices,
                oos_base=oos_base,
                oos_prices=oos_prices,
                specs=list(candidate.specs),
                global_cap=candidate.global_cap,
                label=candidate.name,
                selector="fixed_candidate",
                base_metrics=base_metrics,
                split=f"{year}-01-01",
                funding_spread=funding_spread,
                extra={"Candidate Name": candidate.name, "Notes": candidate.notes},
            )
            summary = eval_out["summary"]
            summary["Year"] = int(year)
            summary["Is Partial Year"] = bool(year == date.today().year)
            summary["Calmar Improvement"] = bool(summary["OOS Calmar Delta"] > 0)
            summary["Control Only"] = is_control_candidate_name(candidate.name)
            summary["Benchmark"] = "base"
            rows.append(summary)
    return pd.DataFrame(rows)


def parameter_stability_summary(
    selected_rules: pd.DataFrame,
    walk_forward: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if selected_rules.empty:
        return pd.DataFrame()
    structural = selected_rules.copy()
    structural["Config Key"] = structural.apply(_config_key, axis=1)
    for selector, group in structural.groupby("Selector"):
        config_counts = group["Config Key"].value_counts()
        mode_count = int(config_counts.iloc[0]) if not config_counts.empty else 0
        mode_share = mode_count / len(group) if len(group) else 0.0
        wf = walk_forward[walk_forward["Selector"] == selector] if not walk_forward.empty else pd.DataFrame()
        annual_years = _full_year_rows(wf)
        annual_improvements = int(annual_years["Calmar Improvement"].sum()) if not annual_years.empty else 0
        robust_sizes = pd.to_numeric(group.get("Robust Neighborhood Size"), errors="coerce")
        avg_robust_size = float(robust_sizes.dropna().mean()) if robust_sizes.notna().any() else np.nan
        stable_pass = bool(mode_count >= 2 or (pd.notna(avg_robust_size) and avg_robust_size >= 2))
        rows.append({
            "Selector": selector,
            "Structural Selections": int(len(group)),
            "Unique Configs": int(group["Config Key"].nunique()),
            "Most Common Config Count": mode_count,
            "Most Common Config Share": round(float(mode_share), 4),
            "Average Robust Neighborhood Size": round(avg_robust_size, 4) if pd.notna(avg_robust_size) else np.nan,
            "Stable Neighborhood Pass": stable_pass,
            "Annual Years Tested": int(len(annual_years)),
            "Annual Calmar Improvement Years": annual_improvements,
            "Most Common Config": config_counts.index[0] if not config_counts.empty else "",
        })
    return pd.DataFrame(rows)


def sweep_heatmap_tables(grid: pd.DataFrame) -> pd.DataFrame:
    if grid.empty:
        return pd.DataFrame()
    specs = [
        ("SPY Threshold", "SPY Entry", "SPY Exit"),
        ("GLD Threshold", "GLD Entry", "GLD Exit"),
        ("Weights", "SPY Weight", "GLD Weight"),
        ("Cap by SPY Weight", "Global Cap", "SPY Weight"),
        ("Cap by GLD Weight", "Global Cap", "GLD Weight"),
    ]
    frames = []
    for dimension, x_col, y_col in specs:
        grouped = grid.groupby(["Split", x_col, y_col], as_index=False).agg(
            Avg_Calmar=("Calmar", "mean"),
            Max_Calmar=("Calmar", "max"),
            Avg_CAGR=("CAGR (%)", "mean"),
            Avg_MaxDD=("Max Drawdown (%)", "mean"),
            Rows=("Calmar", "size"),
        )
        grouped.insert(0, "Dimension", dimension)
        grouped = grouped.rename(columns={x_col: "X", y_col: "Y"})
        frames.append(grouped)
    out = pd.concat(frames, ignore_index=True)
    for col in ["Avg_Calmar", "Max_Calmar", "Avg_CAGR", "Avg_MaxDD"]:
        out[col] = out[col].round(6)
    return out


def disciplined_pass_fail_summary(
    oos_summary: pd.DataFrame,
    walk_forward: pd.DataFrame,
    stability: pd.DataFrame,
) -> pd.DataFrame:
    if oos_summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    stability_by_selector = stability.set_index("Selector") if not stability.empty else pd.DataFrame()
    for selector, group in oos_summary.groupby("Selector"):
        wf = walk_forward[walk_forward["Selector"] == selector] if not walk_forward.empty else pd.DataFrame()
        wf_full = _full_year_rows(wf)
        structural_pass_count = int(group["Pass Split"].sum())
        maxdd_breach = bool((group["OOS MaxDD Delta (%)"] < -3.0).any())
        min_episodes = int(group["OOS Trade Episodes"].min()) if "OOS Trade Episodes" in group else 0
        annual_pass = int(wf_full["Calmar Improvement"].sum()) if not wf_full.empty else 0
        stable_pass = False
        if not stability_by_selector.empty and selector in stability_by_selector.index:
            stable_pass = bool(stability_by_selector.loc[selector, "Stable Neighborhood Pass"])
        max_cap = float(pd.to_numeric(group["Global Cap"]).max())
        aggressive = bool(max_cap > 0.30)
        overall = bool(
            structural_pass_count == 3
            and not maxdd_breach
            and annual_pass >= 8
            and min_episodes >= 3
            and stable_pass
        )
        rows.append({
            "Selector": selector,
            "Benchmark": "base",
            "Control Only": is_control_candidate_name(selector),
            "Global Cap": round(max_cap, 4),
            "SPY Rule": _rule_label(group, "SPY"),
            "GLD Rule": _rule_label(group, "GLD"),
            "Structural Splits Tested": int(len(group)),
            "Structural Splits Passed": structural_pass_count,
            "Average OOS Calmar": round(float(group["OOS Overlay Calmar"].mean()), 4),
            "Average OOS Calmar Delta": round(float(group["OOS Calmar Delta"].mean()), 4),
            "Worst OOS Calmar Delta": round(float(group["OOS Calmar Delta"].min()), 4),
            "Worst OOS MaxDD Delta (%)": round(float(group["OOS MaxDD Delta (%)"].min()), 4),
            "Average OOS CAGR Delta (%)": round(float(group["OOS CAGR Delta (%)"].mean()), 4),
            "RF Cost Pass Splits": int(group["RF Cost Pass"].sum()) if "RF Cost Pass" in group else np.nan,
            "Min OOS Trade Episodes": min_episodes,
            "Average OOS Exposure (%)": round(float(group["OOS Average Overlay Exposure (%)"].mean()), 4),
            "Annual Years Tested": int(len(wf_full)),
            "Annual Calmar Improvement Years": annual_pass,
            "Stable Neighborhood Pass": stable_pass,
            "Aggressive Cap >30%": aggressive,
            "Promotion Tier": "aggressive research" if aggressive else "production candidate",
            "Overall Pass": overall,
        })
    return pd.DataFrame(rows).sort_values(
        ["Overall Pass", "Average OOS Calmar", "Worst OOS Calmar Delta", "Worst OOS MaxDD Delta (%)"],
        ascending=[False, False, False, False],
    )


def _config_key(row: pd.Series | dict[str, object]) -> str:
    return "|".join(
        f"{float(row[col]):.4g}"
        for col in [
            "SPY Entry", "SPY Exit", "SPY Weight",
            "GLD Entry", "GLD Exit", "GLD Weight", "Global Cap",
        ]
        if col in row and pd.notna(row[col])
    )


def _full_year_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "Is Partial Year" not in frame.columns:
        return frame
    return frame[~frame["Is Partial Year"].astype(bool)]


# ---------------------------------------------------------------------------
# OOS evaluation
# ---------------------------------------------------------------------------


def _evaluate_mixed_oos(
    is_base: pd.Series,
    is_prices: pd.DataFrame,
    oos_base: pd.Series,
    oos_prices: pd.DataFrame,
    specs: list[OverlaySpec],
    global_cap: float,
    label: str,
    selector: str,
    base_metrics: dict[str, float],
    split: str,
    funding_spread: float = 0.0,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    """Evaluate one capped mixed-spec config IS and OOS, returning bundle rows."""
    extra = extra or {}

    is_result = apply_overlay_to_base(
        base_values=is_base,
        prices=is_prices,
        specs=specs,
        global_cap=global_cap,
        execution_lag=1,
        name=label,
    )
    is_metrics = _strategy_metrics(is_result.value_series, label)

    oos_result = apply_overlay_to_base(
        base_values=oos_base,
        prices=oos_prices,
        specs=specs,
        global_cap=global_cap,
        execution_lag=1,
        name=label,
    )
    oos_rf_result = apply_overlay_to_base(
        base_values=oos_base,
        prices=oos_prices,
        specs=specs,
        global_cap=global_cap,
        financing_cost_annual=config.RISK_FREE_RATE + funding_spread,
        execution_lag=1,
        name=label,
    )
    oos_metrics = _strategy_metrics(oos_result.value_series, label)
    rf_metrics = _strategy_metrics(oos_rf_result.value_series, label)

    diagnostics = oos_result.daily_diagnostics.copy()
    positions = oos_result.positions
    episodes = _trade_episodes(
        diagnostics, oos_result.value_series, split, label, selector
    )
    avg_exposure = float(diagnostics["Overlay Exposure"].mean() * 100)
    overlay_contribution = float(diagnostics["Overlay Return"].sum() * 100)
    both_active = 0.0
    if {"SPY", "GLD"} <= set(positions.columns):
        both_active = float(
            ((positions["SPY"] > 0) & (positions["GLD"] > 0)).mean() * 100
        )
    avg_spy = float(positions.get("SPY", pd.Series(dtype=float)).mean() * 100) if "SPY" in positions.columns else 0.0
    avg_gld = float(positions.get("GLD", pd.Series(dtype=float)).mean() * 100) if "GLD" in positions.columns else 0.0

    pass_flags = _split_pass_flags(base_metrics, oos_metrics, rf_metrics, len(episodes))

    summary = {
        "Split": split[:4],
        "OOS Start": split,
        "IS Start Date": is_base.index[0].date().isoformat(),
        "IS End Date": is_base.index[-1].date().isoformat(),
        "Selector": selector,
        "Benchmark": "base",
        "Global Cap": global_cap,
        "SPY Entry": _spec_value(specs, "SPY", "entry_threshold"),
        "SPY Exit": _spec_value(specs, "SPY", "exit_threshold"),
        "SPY Weight": _spec_value(specs, "SPY", "overlay_weight"),
        "GLD Entry": _spec_value(specs, "GLD", "entry_threshold"),
        "GLD Exit": _spec_value(specs, "GLD", "exit_threshold"),
        "GLD Weight": _spec_value(specs, "GLD", "overlay_weight"),
        "IS CAGR (%)": is_metrics["CAGR (%)"],
        "IS Calmar": is_metrics["Calmar"],
        "IS Max Drawdown (%)": is_metrics["Max Drawdown (%)"],
        "OOS Base CAGR (%)": base_metrics["CAGR (%)"],
        "OOS Base Calmar": base_metrics["Calmar"],
        "OOS Base Max Drawdown (%)": base_metrics["Max Drawdown (%)"],
        "OOS Overlay CAGR (%)": oos_metrics["CAGR (%)"],
        "OOS Overlay Calmar": oos_metrics["Calmar"],
        "OOS Overlay Max Drawdown (%)": oos_metrics["Max Drawdown (%)"],
        "OOS RF Opportunity Cost CAGR (%)": rf_metrics["CAGR (%)"],
        "Funding Cost Annual (%)": round(float((config.RISK_FREE_RATE + funding_spread) * 100), 4),
        "Funding Spread (%)": round(float(funding_spread * 100), 4),
        "OOS CAGR Delta (%)": round(
            float(oos_metrics["CAGR (%)"] - base_metrics["CAGR (%)"]), 4
        ),
        "OOS Calmar Delta": round(
            float(oos_metrics["Calmar"] - base_metrics["Calmar"]), 4
        ),
        "OOS MaxDD Delta (%)": round(
            float(oos_metrics["Max Drawdown (%)"] - base_metrics["Max Drawdown (%)"]),
            4,
        ),
        "OOS Active Days": int((diagnostics["Overlay Exposure"] > 0).sum()),
        "OOS Active Days (%)": round(
            float((diagnostics["Overlay Exposure"] > 0).mean() * 100), 4
        ),
        "OOS Both Active Days (%)": round(both_active, 4),
        "OOS Average Overlay Exposure (%)": round(avg_exposure, 4),
        "OOS Average SPY Weight (%)": round(avg_spy, 4),
        "OOS Average GLD Weight (%)": round(avg_gld, 4),
        "OOS Overlay Return Contribution (%)": round(overlay_contribution, 4),
        "OOS Trade Episodes": int(len(episodes)),
        **pass_flags,
        **extra,
    }
    summary["Control Only"] = is_control_candidate_name(
        summary.get("Candidate Name", selector)
    )

    daily = _daily_series(pd.DataFrame({label: oos_result.value_series}))
    daily.insert(0, "Selector", selector)
    daily.insert(0, "Overlay Strategy", label)
    daily.insert(0, "Split", split[:4])
    daily.insert(1, "OOS Start", split)

    signals = oos_result.signal_history.copy()
    signals.insert(0, "Selector", selector)
    signals.insert(0, "Overlay Strategy", label)
    signals.insert(0, "Split", split[:4])
    signals.insert(1, "OOS Start", split)

    diagnostics_out = diagnostics.reset_index()
    diagnostics_out.insert(0, "Selector", selector)
    diagnostics_out.insert(0, "Overlay Strategy", label)
    diagnostics_out.insert(0, "Split", split[:4])
    diagnostics_out.insert(1, "OOS Start", split)
    diagnostics_out.insert(4, "Global Cap", global_cap)
    if "SPY" in positions.columns:
        diagnostics_out["SPY Position"] = positions["SPY"].reindex(diagnostics.index).fillna(0.0).values
    if "GLD" in positions.columns:
        diagnostics_out["GLD Position"] = positions["GLD"].reindex(diagnostics.index).fillna(0.0).values

    stress_values = pd.DataFrame({BASE_LABEL: oos_base, label: oos_result.value_series})
    stress = stress_period_metrics(stress_values, _available_stress_periods(stress_values))
    stress.insert(0, "Selector", selector)
    stress.insert(0, "Overlay Strategy", label)
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


def _trade_episodes(
    diagnostics: pd.DataFrame,
    value_series: pd.Series,
    split: str,
    label: str,
    selector: str,
) -> list[dict[str, object]]:
    """Return consecutive active overlay periods (any sleeve > 0)."""
    if diagnostics.empty:
        return []
    active = diagnostics["Overlay Exposure"] > 0
    if not active.any():
        return []

    rows: list[dict[str, object]] = []
    start = None
    episode = 0
    index = diagnostics.index
    for pos, date_idx in enumerate(index):
        is_last = pos == len(index) - 1
        if active.loc[date_idx] and start is None:
            start = date_idx
        if start is not None and ((not active.loc[date_idx]) or is_last):
            end = (
                date_idx
                if active.loc[date_idx] and is_last
                else index[pos - 1]
            )
            chunk = diagnostics.loc[start:end]
            values = value_series.loc[start:end]
            episode += 1
            rows.append(
                {
                    "Split": split[:4],
                    "OOS Start": split,
                    "Overlay Strategy": label,
                    "Selector": selector,
                    "Episode": episode,
                    "Entry Date": start.date().isoformat(),
                    "Exit Date": end.date().isoformat(),
                    "Trading Days": int(len(chunk)),
                    "Average Overlay Exposure (%)": round(
                        float(chunk["Overlay Exposure"].mean() * 100), 4
                    ),
                    "Overlay Return Contribution (%)": round(
                        float(chunk["Overlay Return"].sum() * 100), 4
                    ),
                    "Strategy Return (%)": round(
                        float((values.iloc[-1] / values.iloc[0] - 1) * 100), 4
                    )
                    if len(values) >= 2
                    else 0.0,
                }
            )
            start = None
    return rows


def _split_pass_flags(
    base_metrics: dict[str, float],
    overlay_metrics: dict[str, float],
    rf_metrics: dict[str, float],
    episode_count: int,
) -> dict[str, object]:
    calmar_pass = bool(overlay_metrics["Calmar"] > base_metrics["Calmar"])
    cagr_pass = bool(overlay_metrics["CAGR (%)"] >= base_metrics["CAGR (%)"])
    maxdd_pass = bool(
        overlay_metrics["Max Drawdown (%)"] >= base_metrics["Max Drawdown (%)"] - 1.0
    )
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


def build_pass_fail_summary(
    fixed_oos: pd.DataFrame, selector_oos: pd.DataFrame
) -> pd.DataFrame:
    """Aggregate per-split rows into per-candidate pass/fail summaries."""
    frames = []
    if not fixed_oos.empty:
        frames.append(_aggregate_pass_fail(fixed_oos, key="Candidate Name", source="fixed"))
    if not selector_oos.empty:
        frames.append(
            _aggregate_pass_fail(selector_oos, key="Selector", source="selector")
        )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(
        ["Overall Pass", "Average OOS Calmar", "Worst OOS Calmar Delta", "Worst OOS MaxDD Delta (%)"],
        ascending=[False, False, False, False],
    )


def _aggregate_pass_fail(df: pd.DataFrame, key: str, source: str) -> pd.DataFrame:
    rows = []
    for name, group in df.groupby(key):
        pass_count = int(group["Pass Split"].sum())
        maxdd_breach = bool((group["OOS MaxDD Delta (%)"] < -3.0).any())
        low_trade_count = int(group["Low Trade Count Flag"].sum())
        rows.append(
            {
                "Source": source,
                "Name": name,
                "Benchmark": "base",
                "Control Only": is_control_candidate_name(name),
                "Global Cap": round(float(pd.to_numeric(group["Global Cap"]).max()), 4),
                "SPY Rule": _rule_label(group, "SPY"),
                "GLD Rule": _rule_label(group, "GLD"),
                "Splits Tested": int(len(group)),
                "Splits Passed": pass_count,
                "Low Trade Count Splits": low_trade_count,
                "Average OOS Calmar": round(
                    float(group["OOS Overlay Calmar"].mean()), 4
                ),
                "Average OOS Calmar Delta": round(
                    float(group["OOS Calmar Delta"].mean()), 4
                ),
                "Worst OOS Calmar Delta": round(
                    float(group["OOS Calmar Delta"].min()), 4
                ),
                "Worst OOS MaxDD Delta (%)": round(
                    float(group["OOS MaxDD Delta (%)"].min()), 4
                ),
                "Average OOS CAGR Delta (%)": round(
                    float(group["OOS CAGR Delta (%)"].mean()), 4
                ),
                "RF Cost Pass Splits": int(group["RF Cost Pass"].sum()) if "RF Cost Pass" in group else np.nan,
                "Min OOS Trade Episodes": int(group["OOS Trade Episodes"].min()),
                "Average OOS Exposure (%)": round(
                    float(group["OOS Average Overlay Exposure (%)"].mean()), 4
                ),
                "Overall Pass": bool(pass_count >= 2 and not maxdd_breach),
                "MaxDD Breach >3pp": maxdd_breach,
            }
        )
    return pd.DataFrame(rows)


def _rule_label(group: pd.DataFrame, ticker: str) -> str:
    """Summarise the modal rule for a ticker in an aggregate row."""
    entry_col = f"{ticker} Entry"
    exit_col = f"{ticker} Exit"
    weight_col = f"{ticker} Weight"
    if not {entry_col, exit_col, weight_col} <= set(group.columns):
        return ""
    keys = (
        group[[entry_col, exit_col, weight_col]]
        .round(4)
        .astype(str)
        .agg("/".join, axis=1)
    )
    if keys.empty:
        return ""
    mode = keys.value_counts().index[0].split("/")
    return f"{float(mode[0]):g}/{float(mode[1]):g} @ {float(mode[2]) * 100:g}%"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_series(
    prices: pd.DataFrame, allocation: dict[str, float], apply_fees: bool
) -> pd.Series:
    clean = prices[list(allocation)].dropna()
    if clean.empty:
        return pd.Series(dtype=float).rename(BASE_LABEL)
    base = build_monthly_rebalanced_series(clean, allocation, start_value=100.0)
    base = apply_annual_fee(base, DIY_FEE) if apply_fees else base
    return base.rename(BASE_LABEL)


def _strategy_metrics(series: pd.Series, label: str) -> dict[str, float]:
    metrics = summary_metrics(pd.DataFrame({label: series}))
    return metrics[metrics["Strategy"] == label].iloc[0].to_dict()


def _grid_metrics(series: pd.Series) -> dict[str, float]:
    """Cheap metric subset used inside the IS grid scanner."""
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
        if len(returns) and returns.std() > 1e-12
        else np.nan
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


def _month_end_positions(index: pd.DatetimeIndex) -> np.ndarray:
    """Return integer positions of the final available row in each calendar month."""
    month_number = index.year * 12 + index.month
    return np.flatnonzero(np.r_[month_number[1:] != month_number[:-1], True])


def _grid_metrics_from_arrays(
    values: np.ndarray,
    returns: np.ndarray,
    month_end_positions: np.ndarray,
    years: float,
) -> dict[str, float]:
    """NumPy-only metric subset for the grid scanner hot path."""
    start = float(values[0])
    end = float(values[-1])
    total_return = (end / start - 1.0) * 100 if start > 0 else np.nan
    cagr = ((end / start) ** (1 / years) - 1.0) * 100 if start > 0 and end > 0 else np.nan

    running_max = np.maximum.accumulate(values)
    drawdown = values / running_max - 1.0
    max_dd = float(np.min(drawdown) * 100)

    daily_returns = returns[1:]
    daily_std = float(np.std(daily_returns, ddof=1)) if len(daily_returns) > 1 else 0.0
    rf_daily = (1.0 + config.RISK_FREE_RATE) ** (1 / TRADING_DAYS_PER_YEAR) - 1.0
    sharpe = (
        float(((np.mean(daily_returns) - rf_daily) / daily_std) * np.sqrt(TRADING_DAYS_PER_YEAR))
        if daily_std > 1e-12
        else np.nan
    )

    month_values = values[month_end_positions]
    if len(month_values) >= 2:
        monthly = (month_values[1:] / month_values[:-1] - 1.0) * 100
        worst_month = float(np.min(monthly))
    else:
        worst_month = np.nan

    return {
        "Total Return (%)": round(float(total_return), 4) if np.isfinite(total_return) else np.nan,
        "CAGR (%)": round(float(cagr), 4) if np.isfinite(cagr) else np.nan,
        "Sharpe": round(sharpe, 4) if np.isfinite(sharpe) else np.nan,
        "Calmar": round(float(cagr / abs(max_dd)), 4)
        if np.isfinite(cagr) and abs(max_dd) > 1e-12
        else np.nan,
        "Max Drawdown (%)": round(max_dd, 4),
        "Worst Month (%)": round(worst_month, 4) if np.isfinite(worst_month) else np.nan,
    }


def _spec_value(specs: list[OverlaySpec], ticker: str, attr: str) -> float:
    for spec in specs:
        if spec.ticker == ticker:
            return float(getattr(spec, attr))
    return float("nan")


def _available_stress_periods(values: pd.DataFrame) -> dict[str, tuple[str, str]]:
    start = values.index.min()
    end = values.index.max()
    periods = {}
    for name, (period_start, period_end) in STRESS_PERIODS.items():
        ps = pd.Timestamp(period_start)
        pe = pd.Timestamp(period_end)
        if pe >= start and ps <= end:
            periods[name] = (
                max(ps, start).date().isoformat(),
                min(pe, end).date().isoformat(),
            )
    periods["Full OOS window"] = (
        start.date().isoformat(),
        end.date().isoformat(),
    )
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
    fixed_candidates: list[MixedOverlayCandidate],
    entry_grid: tuple[float, ...],
    exit_grid: tuple[float, ...],
    leverage_grid: tuple[float, ...],
    cap_grid: tuple[float, ...],
    apply_fees: bool,
    fixed_only: bool,
    max_global_cap: float | None,
    max_sleeve_weight: float | None,
    funding_spread: float,
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
        "broker_constraints": {
            "max_global_cap": max_global_cap,
            "max_sleeve_weight": max_sleeve_weight,
            "funding_spread": funding_spread,
            "funding_cost_annual": config.RISK_FREE_RATE + funding_spread,
        },
        "allocation": allocation,
        "oos_splits": list(splits),
        "fixed_only": fixed_only,
        "fixed_candidates": [
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
            for candidate in fixed_candidates
        ],
        "mixed_grid": {
            "entry_thresholds": list(entry_grid),
            "exit_thresholds": list(exit_grid),
            "leverage_grid": list(leverage_grid),
            "cap_grid": list(cap_grid),
            "tickers": list(MIXED_TICKERS),
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
            "is_mixed_grid.csv",
            "selected_rules.csv",
            "fixed_candidates_oos.csv",
            "fixed_candidate_walk_forward_summary.csv",
            "oos_summary.csv",
            "oos_daily_series.csv",
            "oos_signal_history.csv",
            "oos_overlay_diagnostics.csv",
            "oos_stress_metrics.csv",
            "oos_trade_episodes.csv",
            "pass_fail_summary.csv",
        ],
    }


def _sweep_manifest(
    strategy_id: str,
    allocation: dict[str, float],
    start_date: str,
    end_date: str,
    prices: pd.DataFrame,
    splits: tuple[str, ...],
    walk_forward_years: tuple[int, ...],
    spy_entry_grid: tuple[float, ...],
    spy_exit_grid: tuple[float, ...],
    spy_weight_grid: tuple[float, ...],
    gld_entry_grid: tuple[float, ...],
    gld_exit_grid: tuple[float, ...],
    gld_weight_grid: tuple[float, ...],
    cap_grid: tuple[float, ...],
    apply_fees: bool,
    max_global_cap: float | None,
    max_sleeve_weight: float | None,
    funding_spread: float,
    generated_at: datetime,
    data_source: str,
) -> dict:
    return {
        "generated_at": generated_at.isoformat(timespec="seconds"),
        "strategy_id": strategy_id,
        "sweep_depth": "disciplined",
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
        "broker_constraints": {
            "max_global_cap": max_global_cap,
            "max_sleeve_weight": max_sleeve_weight,
            "funding_spread": funding_spread,
            "funding_cost_annual": config.RISK_FREE_RATE + funding_spread,
        },
        "allocation": allocation,
        "oos_splits": list(splits),
        "walk_forward_years": list(walk_forward_years),
        "rsi_lookback": 14,
        "mixed_grid": {
            "spy_entry_thresholds": list(spy_entry_grid),
            "spy_exit_thresholds": list(spy_exit_grid),
            "spy_weight_grid": list(spy_weight_grid),
            "gld_entry_thresholds": list(gld_entry_grid),
            "gld_exit_thresholds": list(gld_exit_grid),
            "gld_weight_grid": list(gld_weight_grid),
            "cap_grid": list(cap_grid),
            "tickers": list(MIXED_TICKERS),
        },
        "selectors": list(SELECTORS),
        "pass_fail_rules": {
            "structural_splits_required": 3,
            "annual_calmar_improvement_years_required": 8,
            "maxdd_breach_pp": 3.0,
            "min_trade_episodes_per_structural_split": 3,
            "simple_stable_region_threshold": "Calmar >= 95% of robust-region Calmar, then lowest average exposure",
            "aggressive_cap_threshold": 0.30,
        },
        "artifacts": [
            "manifest.json",
            "is_sweep_grid.parquet",
            "is_sweep_leaderboard.csv",
            "selected_rules.csv",
            "oos_summary.csv",
            "walk_forward_summary.csv",
            "parameter_stability.csv",
            "sweep_heatmap_tables.csv",
            "pass_fail_summary.csv",
        ],
    }


def _full_grid_manifest(
    strategy_id: str,
    allocation: dict[str, float],
    start_date: str,
    end_date: str,
    prices: pd.DataFrame,
    splits: tuple[str, ...],
    walk_forward_years: tuple[int, ...],
    entry_grid: tuple[float, ...],
    exit_grid: tuple[float, ...],
    spy_weight_grid: tuple[float, ...],
    gld_weight_grid: tuple[float, ...],
    cap_grid: tuple[float, ...],
    apply_fees: bool,
    max_global_cap: float | None,
    max_sleeve_weight: float | None,
    funding_spread: float,
    generated_at: datetime,
    data_source: str,
) -> dict:
    return {
        "generated_at": generated_at.isoformat(timespec="seconds"),
        "strategy_id": strategy_id,
        "mode": "full_grid_oos_walk_forward",
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
        "broker_constraints": {
            "max_global_cap": max_global_cap,
            "max_sleeve_weight": max_sleeve_weight,
            "funding_spread": funding_spread,
            "funding_cost_annual": config.RISK_FREE_RATE + funding_spread,
        },
        "allocation": allocation,
        "oos_splits": list(splits),
        "walk_forward_years": list(walk_forward_years),
        "rsi_lookback": 14,
        "mixed_grid": {
            "entry_thresholds": list(entry_grid),
            "exit_thresholds": list(exit_grid),
            "spy_weight_grid": list(spy_weight_grid),
            "gld_weight_grid": list(gld_weight_grid),
            "cap_grid": list(cap_grid),
            "tickers": list(MIXED_TICKERS),
            "invalid_rule": "entry >= exit is skipped",
        },
        "artifacts": [
            "manifest.json",
            "structural_full_grid_oos.parquet",
            "all_considered_strategies.csv",
            "structural_full_grid_leaderboard.csv",
            "structural_full_grid_summary.csv",
            "annual_full_grid_walk_forward.parquet",
            "annual_full_grid_summary.parquet",
            "annual_full_grid_leaderboard.csv",
        ],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_splits(value: str) -> tuple[str, ...]:
    if not value:
        return OOS_SPLITS
    parts = [chunk.strip() for chunk in value.split(",") if chunk.strip()]
    return tuple(parts) if parts else OOS_SPLITS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate capped multi-ETF RSI leverage overlays out of sample."
    )
    parser.add_argument("--strategy-id", default=config.DEFAULT_STRATEGY)
    parser.add_argument("--start-date", default=config.BACKTEST_START)
    parser.add_argument("--end-date", default=config.BACKTEST_END)
    parser.add_argument("--output-root", default="")
    parser.add_argument(
        "--sweep-depth",
        choices=("coarse", "disciplined"),
        default="coarse",
        help="coarse preserves the existing mixed OOS bundle; disciplined writes the expanded sweep bundle.",
    )
    parser.add_argument(
        "--splits",
        default="",
        help="Comma-separated OOS split dates; default 2018/2020/2022.",
    )
    parser.add_argument(
        "--fixed-only",
        action="store_true",
        help="Skip the mixed grid; evaluate only the named fixed candidates.",
    )
    parser.add_argument(
        "--full-grid-oos",
        action="store_true",
        help="Evaluate every full-grid config structurally OOS and annual walk-forward.",
    )
    parser.add_argument("--max-global-cap", type=float, default=None)
    parser.add_argument("--max-sleeve-weight", type=float, default=None)
    parser.add_argument(
        "--funding-spread",
        type=float,
        default=0.0,
        help="Additional annual funding spread over the project risk-free rate, decimal form.",
    )
    parser.add_argument("--data-source", choices=("yfinance", "fmp"), default=config.DATA_SOURCE)
    parser.add_argument("--fmp-price-column", default=config.FMP_PRICE_COLUMN)
    parser.add_argument("--no-fees", action="store_true", help="Skip DIY expense-ratio drag.")
    args = parser.parse_args()

    config.DATA_SOURCE = args.data_source
    config.FMP_PRICE_COLUMN = args.fmp_price_column
    splits = _parse_splits(args.splits)
    if args.full_grid_oos:
        bundle = build_full_grid_from_yfinance(
            strategy_id=args.strategy_id,
            start_date=args.start_date,
            end_date=args.end_date,
            output_root=args.output_root or DEFAULT_FULL_GRID_OUTPUT_ROOT,
            apply_fees=not args.no_fees,
            splits=splits,
            max_global_cap=args.max_global_cap,
            max_sleeve_weight=args.max_sleeve_weight,
            funding_spread=args.funding_spread,
        )
        print(f"Mixed leverage full-grid OOS bundle written to: {bundle}")
    elif args.sweep_depth == "disciplined":
        bundle = build_sweep_from_yfinance(
            strategy_id=args.strategy_id,
            start_date=args.start_date,
            end_date=args.end_date,
            output_root=args.output_root or DEFAULT_SWEEP_OUTPUT_ROOT,
            apply_fees=not args.no_fees,
            splits=splits,
            max_global_cap=args.max_global_cap,
            max_sleeve_weight=args.max_sleeve_weight,
            funding_spread=args.funding_spread,
        )
        print(f"Mixed leverage disciplined sweep bundle written to: {bundle}")
    else:
        bundle = build_from_yfinance(
            strategy_id=args.strategy_id,
            start_date=args.start_date,
            end_date=args.end_date,
            output_root=args.output_root or DEFAULT_OUTPUT_ROOT,
            apply_fees=not args.no_fees,
            splits=splits,
            fixed_only=args.fixed_only,
            max_global_cap=args.max_global_cap,
            max_sleeve_weight=args.max_sleeve_weight,
            funding_spread=args.funding_spread,
        )
        print(f"Mixed leverage OOS validation bundle written to: {bundle}")


if __name__ == "__main__":
    main()
