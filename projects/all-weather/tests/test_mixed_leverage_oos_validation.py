from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from research.validate_mixed_leverage_oos import (
    FULL_GRID_GLD_WEIGHT_GRID,
    FULL_GRID_SPY_WEIGHT_GRID,
    build_full_grid_oos_bundle,
    build_mixed_oos_validation_bundle,
    build_mixed_sweep_bundle,
    load_strategy,
    select_mixed_rules,
    _fast_mixed_grid,
    _full_grid_combined_summary,
    _robust_calmar_selection,
    _robust_calmar_selection_slow,
)
from research.build_mixed_leverage_report import default_mixed_candidates


EXPECTED_SWEEP_ARTIFACTS = {
    "manifest.json",
    "is_sweep_grid.parquet",
    "is_sweep_leaderboard.csv",
    "selected_rules.csv",
    "oos_summary.csv",
    "walk_forward_summary.csv",
    "parameter_stability.csv",
    "sweep_heatmap_tables.csv",
    "pass_fail_summary.csv",
}
EXPECTED_MIXED_OOS_ARTIFACTS = {
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
}
EXPECTED_FULL_GRID_ARTIFACTS = {
    "manifest.json",
    "structural_full_grid_oos.parquet",
    "all_considered_strategies.csv",
    "structural_full_grid_leaderboard.csv",
    "structural_full_grid_summary.csv",
    "annual_full_grid_walk_forward.parquet",
    "annual_full_grid_summary.parquet",
    "annual_full_grid_leaderboard.csv",
}


def test_full_grid_weight_defaults_are_symmetric_5_to_30_percent():
    expected = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)

    assert FULL_GRID_SPY_WEIGHT_GRID == expected
    assert FULL_GRID_GLD_WEIGHT_GRID == expected


def test_mixed_leverage_strategy_loader_accepts_baseline_alias():
    payload = load_strategy("6_asset_rp_baseline")

    assert payload["allocation"]["SPY"] == 0.134
    assert payload["allocation"]["GLD"] == 0.142


def test_full_grid_combined_summary_keeps_failed_rows_as_diagnostics():
    structural = pd.DataFrame([
        {
            "SPY Entry": 30.0, "SPY Exit": 50.0, "SPY Weight": 0.10,
            "GLD Entry": 30.0, "GLD Exit": 50.0, "GLD Weight": 0.10,
            "Global Cap": 0.20, "Structural Pass": True,
        },
        {
            "SPY Entry": 32.0, "SPY Exit": 50.0, "SPY Weight": 0.30,
            "GLD Entry": 32.0, "GLD Exit": 50.0, "GLD Weight": 0.30,
            "Global Cap": 0.30, "Structural Pass": False,
        },
    ])
    annual = structural.drop(columns=["Structural Pass"]).copy()
    annual["Annual Years Tested"] = [12, 12]
    annual["Annual Calmar Improvement Years"] = [9, 2]
    annual["Average Annual Calmar"] = [1.2, 0.8]
    annual["Average Annual Calmar Delta"] = [0.2, -0.1]
    annual["Worst Annual Calmar Delta"] = [-0.1, -0.4]
    annual["Worst Annual MaxDD Delta (%)"] = [-0.5, -2.0]

    combined = _full_grid_combined_summary(structural, annual)

    assert len(combined) == 2
    assert set(combined["Overall Pass"]) == {True, False}


def test_disciplined_sweep_bundle_writes_expected_artifacts_and_walk_forward(tmp_path, monkeypatch):
    # The reproducible env includes pyarrow, but this keeps the unit test
    # runnable in stripped-down local envs while still verifying the artifact contract.
    monkeypatch.setattr(
        pd.DataFrame,
        "to_parquet",
        lambda self, path, index=False, compression=None: self.to_pickle(path),
    )
    monkeypatch.setattr(pd, "read_parquet", lambda path: pd.read_pickle(path))

    split = "2018-01-01"
    bundle = build_mixed_sweep_bundle(
        prices=_synthetic_prices(),
        strategy_id="synthetic_mixed_sweep",
        allocation=_allocation(),
        output_root=tmp_path,
        start_date="2012-01-01",
        end_date="2021-12-31",
        splits=(split,),
        walk_forward_years=(2017, 2018),
        spy_entry_grid=(30.0,),
        spy_exit_grid=(50.0,),
        spy_weight_grid=(0.20,),
        gld_entry_grid=(30.0,),
        gld_exit_grid=(50.0,),
        gld_weight_grid=(0.20,),
        cap_grid=(0.20,),
        generated_at=datetime(2026, 1, 2, 3, 4, 5),
    )

    assert {path.name for path in bundle.iterdir()} == EXPECTED_SWEEP_ARTIFACTS

    grid = pd.read_parquet(bundle / "is_sweep_grid.parquet")
    leaderboard = pd.read_csv(bundle / "is_sweep_leaderboard.csv")
    selected = pd.read_csv(bundle / "selected_rules.csv")
    oos = pd.read_csv(bundle / "oos_summary.csv")
    walk = pd.read_csv(bundle / "walk_forward_summary.csv")
    stability = pd.read_csv(bundle / "parameter_stability.csv")
    heatmaps = pd.read_csv(bundle / "sweep_heatmap_tables.csv")
    pass_fail = pd.read_csv(bundle / "pass_fail_summary.csv")

    assert not grid.empty
    assert {"Split", "IS End Date", "SPY Entry", "GLD Entry", "Global Cap", "Calmar"} <= set(grid.columns)
    assert {"IS Calmar Rank", "Broker Profile", "SPY Rule", "GLD Rule", "Calmar"} <= set(leaderboard.columns)
    assert not leaderboard.empty
    assert {"default_30_50_20", "simple_stable_region"} <= set(selected["Selector"])
    assert (pd.to_datetime(selected["IS End Date"]) < pd.Timestamp(split)).all()
    assert {"OOS RF Opportunity Cost CAGR (%)", "OOS Trade Episodes", "Pass Split"} <= set(oos.columns)
    assert set(walk["Year"]) == {2017, 2018}
    assert {"Stable Neighborhood Pass", "Annual Calmar Improvement Years"} <= set(stability.columns)
    assert {"Dimension", "Split", "X", "Y", "Avg_Calmar", "Max_Calmar"} <= set(heatmaps.columns)
    assert {"Overall Pass", "Promotion Tier"} <= set(pass_fail.columns)


def test_mixed_oos_bundle_diagnostics_respect_global_cap(tmp_path):
    bundle = build_mixed_oos_validation_bundle(
        prices=_synthetic_prices(),
        strategy_id="synthetic_mixed_oos",
        allocation=_allocation(),
        output_root=tmp_path,
        start_date="2012-01-01",
        end_date="2021-12-31",
        generated_at=datetime(2026, 1, 2, 3, 4, 5),
        splits=("2018-01-01",),
        entry_grid=(30.0,),
        exit_grid=(50.0,),
        leverage_grid=(0.20,),
        cap_grid=(0.20,),
    )

    assert {path.name for path in bundle.iterdir()} == EXPECTED_MIXED_OOS_ARTIFACTS

    diagnostics = pd.read_csv(bundle / "oos_overlay_diagnostics.csv")
    fixed = pd.read_csv(bundle / "fixed_candidates_oos.csv")
    pass_fail = pd.read_csv(bundle / "pass_fail_summary.csv")
    fixed_walk = pd.read_csv(bundle / "fixed_candidate_walk_forward_summary.csv")

    heatmap_candidate = fixed[fixed["Candidate Name"] == "SPY34/42 + GLD32/64 30% cap"]
    assert not heatmap_candidate.empty
    assert heatmap_candidate["SPY Entry"].eq(34.0).all()
    assert heatmap_candidate["SPY Exit"].eq(42.0).all()
    assert heatmap_candidate["GLD Entry"].eq(32.0).all()
    assert heatmap_candidate["GLD Exit"].eq(64.0).all()
    assert pass_fail["Benchmark"].eq("base").all()
    assert {"Average OOS Calmar", "Average OOS Calmar Delta", "RF Cost Pass Splits", "SPY Rule", "GLD Rule"} <= set(pass_fail.columns)
    assert "SPY+GLD default 20% total cap" in set(
        pass_fail.loc[pass_fail["Control Only"].astype(bool), "Name"]
    )
    assert not fixed_walk.empty
    assert fixed_walk["Benchmark"].eq("base").all()

    with_positions = diagnostics.dropna(subset=["SPY Position", "GLD Position"], how="all").copy()
    with_positions[["SPY Position", "GLD Position"]] = with_positions[["SPY Position", "GLD Position"]].fillna(0.0)

    assert not with_positions.empty
    exposure = with_positions["SPY Position"] + with_positions["GLD Position"]
    assert (exposure <= with_positions["Global Cap"] + 1e-12).all()


def test_fast_mixed_grid_respects_cap_for_each_row():
    prices = _synthetic_prices()
    base = pd.Series(100.0, index=prices.index, name="base")

    grid = _fast_mixed_grid(
        is_base=base,
        is_prices=prices,
        entry_grid=(30.0,),
        exit_grid=(50.0,),
        leverage_grid=(0.20, 0.30),
        cap_grid=(0.15, 0.25),
    )

    assert not grid.empty
    assert (grid["Max Overlay Exposure (%)"] <= grid["Global Cap"] * 100 + 1e-9).all()


def test_full_grid_oos_bundle_writes_every_valid_config(tmp_path, monkeypatch):
    def _read_pickled_parquet(path):
        path = Path(path)
        if path.is_dir():
            return pd.concat(
                [pd.read_pickle(child) for child in sorted(path.glob("*.parquet"))],
                ignore_index=True,
            )
        return pd.read_pickle(path)

    monkeypatch.setattr(
        pd.DataFrame,
        "to_parquet",
        lambda self, path, index=False, compression=None: self.to_pickle(path),
    )
    monkeypatch.setattr(pd, "read_parquet", _read_pickled_parquet)

    bundle = build_full_grid_oos_bundle(
        prices=_synthetic_prices(),
        strategy_id="synthetic_full_grid",
        allocation=_allocation(),
        output_root=tmp_path,
        start_date="2012-01-01",
        end_date="2021-12-31",
        splits=("2018-01-01",),
        walk_forward_years=(2018,),
        entry_grid=(38.0, 40.0),
        exit_grid=(40.0, 42.0),
        spy_weight_grid=(0.20,),
        gld_weight_grid=(0.20,),
        cap_grid=(0.20,),
        max_global_cap=0.30,
        max_sleeve_weight=0.30,
        generated_at=datetime(2026, 1, 2, 3, 4, 5),
    )

    assert {path.name for path in bundle.iterdir()} == EXPECTED_FULL_GRID_ARTIFACTS

    structural = pd.read_parquet(bundle / "structural_full_grid_oos.parquet")
    annual = pd.read_parquet(bundle / "annual_full_grid_walk_forward.parquet")
    annual_summary = pd.read_parquet(bundle / "annual_full_grid_summary.parquet")
    all_considered = pd.read_csv(bundle / "all_considered_strategies.csv")
    structural_leaderboard = pd.read_csv(bundle / "structural_full_grid_leaderboard.csv")
    annual_leaderboard = pd.read_csv(bundle / "annual_full_grid_leaderboard.csv")

    assert len(structural) == 9
    assert len(annual) == 9
    assert len(all_considered) == 9
    assert not annual_summary.empty
    assert not structural_leaderboard.empty
    assert not annual_leaderboard.empty
    assert {42.0} <= set(structural["SPY Exit"])
    assert {38.0, 40.0} <= set(structural["SPY Entry"])
    assert not (
        structural["SPY Entry"].eq(40.0)
        & structural["SPY Exit"].eq(40.0)
    ).any()
    assert not (
        structural["GLD Entry"].eq(40.0)
        & structural["GLD Exit"].eq(40.0)
    ).any()
    assert structural["Global Cap"].max() <= 0.30
    assert structural["SPY Weight"].max() <= 0.30
    assert structural["GLD Weight"].max() <= 0.30
    assert {
        "OOS Active Days",
        "OOS SPY Active Days",
        "OOS GLD Active Days",
        "OOS Both Active Days",
        "OOS Cap Binding Days",
        "OOS SPY Trade Episodes",
        "OOS GLD Trade Episodes",
        "OOS Calmar Delta",
        "OOS MaxDD Delta (%)",
        "RF Cost Pass",
        "Broker Profile",
    } <= set(structural.columns)
    assert {
        "Average OOS Calmar",
        "Worst OOS Calmar Delta",
        "Worst OOS MaxDD Delta (%)",
        "Average OOS SPY Active Days",
        "Average OOS GLD Active Days",
        "Average OOS Cap Binding Days",
        "Min OOS SPY Trade Episodes",
        "Min OOS GLD Trade Episodes",
        "Annual Calmar Improvement Years",
    } <= set(structural_leaderboard.columns)
    assert {
        "Overall Pass",
        "Structural Pass",
        "Average OOS Calmar",
        "Average OOS SPY Active Days",
        "Average OOS GLD Active Days",
        "Average Annual SPY Active Days",
        "Average Annual GLD Active Days",
    } <= set(all_considered.columns)
    assert all_considered["Overall Pass"].isin([True, False]).all()


def test_select_mixed_rules_is_deterministic_and_adds_simple_stable_region():
    grid = pd.DataFrame([
        _grid_row(30, 50, 0.20, 30, 50, 0.20, 0.20, 0.34, 8.0, -22.0, 3.5),
        _grid_row(34, 42, 0.25, 22, 50, 0.25, 0.30, 0.50, 8.2, -20.0, 4.0),
        _grid_row(34, 42, 0.20, 22, 50, 0.20, 0.25, 0.49, 8.0, -19.5, 2.5),
        _grid_row(22, 42, 0.10, 22, 42, 0.10, 0.20, 0.48, 7.8, -19.0, 1.0),
    ])

    selected = select_mixed_rules(grid, {"Max Drawdown (%)": -20.0})

    assert {
        "default_30_50_20",
        "best_calmar",
        "best_maxdd_preservation",
        "best_cagr_with_maxdd_guard",
        "robust_calmar_region",
        "simple_stable_region",
    } <= set(selected["Selector"])
    simple = selected[selected["Selector"] == "simple_stable_region"].iloc[0]
    assert simple["Average Overlay Exposure (%)"] == 1.0


def test_default_mixed_candidates_include_calmar_audit_configs():
    candidates = {candidate.name: candidate for candidate in default_mixed_candidates()}

    assert "SPY34/42 + GLD32/64 30% cap" in candidates
    candidate = candidates["SPY34/42 + GLD32/64 30% cap"]
    assert candidate.global_cap == 0.30
    specs = {spec.ticker: spec for spec in candidate.specs}
    assert specs["SPY"].entry_threshold == 34.0
    assert specs["SPY"].exit_threshold == 42.0
    assert specs["GLD"].entry_threshold == 32.0
    assert specs["GLD"].exit_threshold == 64.0
    assert "CONTROL ONLY" in candidates["SPY+GLD default 20% total cap"].notes


def test_robust_selector_fast_path_matches_slow_path_on_regular_grid():
    rows = []
    for spy_entry in (30.0, 32.0):
        for spy_exit in (48.0, 52.0):
            for spy_weight in (0.15, 0.20):
                for gld_entry in (28.0, 30.0):
                    for gld_exit in (48.0, 52.0):
                        for gld_weight in (0.15, 0.20):
                            for cap in (0.20, 0.25):
                                score = (
                                    spy_entry / 100
                                    + spy_exit / 200
                                    - spy_weight
                                    + gld_entry / 150
                                    + gld_exit / 250
                                    - gld_weight
                                    - cap / 10
                                )
                                rows.append(
                                    _grid_row(
                                        spy_entry, spy_exit, spy_weight,
                                        gld_entry, gld_exit, gld_weight,
                                        cap, score, score * 10, -20.0, cap * 20,
                                    )
                                )
    grid = pd.DataFrame(rows)

    fast = _robust_calmar_selection(grid)
    slow = _robust_calmar_selection_slow(grid)

    assert fast["size"] == slow["size"]
    assert fast["avg_calmar"] == slow["avg_calmar"]
    assert fast["row"]["Calmar"] == slow["row"]["Calmar"]


def _grid_row(
    spy_entry,
    spy_exit,
    spy_weight,
    gld_entry,
    gld_exit,
    gld_weight,
    cap,
    calmar,
    cagr,
    maxdd,
    avg_exposure,
) -> dict[str, float]:
    return {
        "SPY Entry": float(spy_entry),
        "SPY Exit": float(spy_exit),
        "SPY Weight": float(spy_weight),
        "GLD Entry": float(gld_entry),
        "GLD Exit": float(gld_exit),
        "GLD Weight": float(gld_weight),
        "Global Cap": float(cap),
        "Active Days (%)": avg_exposure * 2,
        "Both Active Days (%)": 0.0,
        "Average Overlay Exposure (%)": avg_exposure,
        "Max Overlay Exposure (%)": cap * 100,
        "CAGR (%)": float(cagr),
        "Sharpe": 0.5,
        "Calmar": float(calmar),
        "Max Drawdown (%)": float(maxdd),
        "Worst Month (%)": -5.0,
        "Total Return (%)": 50.0,
        "Incremental CAGR (%)": 1.0,
        "Incremental Calmar": 0.1,
        "Incremental MaxDD (%)": 0.0,
    }


def _allocation() -> dict[str, float]:
    return {
        "SPY": 0.134,
        "QQQ": 0.103,
        "TLT": 0.175,
        "TIP": 0.348,
        "GLD": 0.142,
        "GSG": 0.098,
    }


def _synthetic_prices() -> pd.DataFrame:
    dates = pd.bdate_range("2012-01-01", "2021-12-31")
    n = len(dates)
    phase = np.arange(n)
    cycle = np.sin(phase / 20) / 260
    shock = np.zeros(n)
    shock[650:690] = -0.014
    shock[690:750] = 0.008
    spy_returns = 0.00022 + cycle + shock
    gold_returns = 0.00010 - cycle * 0.5 - shock * 0.25 + np.cos(phase / 23) / 1600

    return pd.DataFrame({
        "SPY": 100 * np.cumprod(1 + spy_returns),
        "QQQ": 100 * np.cumprod(1 + spy_returns * 1.12 + 0.00003),
        "TLT": 100 * np.cumprod(1 + 0.00008 - cycle * 0.35),
        "TIP": 100 * np.cumprod(1 + 0.00006 - cycle * 0.12),
        "GLD": 100 * np.cumprod(1 + gold_returns),
        "GSG": 100 * np.cumprod(1 + 0.00004 + np.sin(phase / 21) / 1400),
    }, index=dates)
