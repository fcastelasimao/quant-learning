from datetime import datetime

import numpy as np
import pandas as pd

from research.validate_mixed_leverage_oos import (
    build_mixed_oos_validation_bundle,
    build_mixed_sweep_bundle,
    select_mixed_rules,
    _fast_mixed_grid,
    _robust_calmar_selection,
    _robust_calmar_selection_slow,
)


EXPECTED_SWEEP_ARTIFACTS = {
    "manifest.json",
    "is_sweep_grid.parquet",
    "selected_rules.csv",
    "oos_summary.csv",
    "walk_forward_summary.csv",
    "parameter_stability.csv",
    "sweep_heatmap_tables.csv",
    "pass_fail_summary.csv",
}


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
    selected = pd.read_csv(bundle / "selected_rules.csv")
    oos = pd.read_csv(bundle / "oos_summary.csv")
    walk = pd.read_csv(bundle / "walk_forward_summary.csv")
    stability = pd.read_csv(bundle / "parameter_stability.csv")
    heatmaps = pd.read_csv(bundle / "sweep_heatmap_tables.csv")
    pass_fail = pd.read_csv(bundle / "pass_fail_summary.csv")

    assert not grid.empty
    assert {"Split", "IS End Date", "SPY Entry", "GLD Entry", "Global Cap", "Calmar"} <= set(grid.columns)
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

    diagnostics = pd.read_csv(bundle / "oos_overlay_diagnostics.csv")
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
