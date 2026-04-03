from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

import config
from backtest import build_stats, format_stats_table, run_backtest
from data_loader import load_data
from plotting import (
    plot_benchmark_comparison,
    plot_validation_summary,
    plot_walkforward_summary,
)
from research_io import archive_selected_outputs, save_run_metadata, timestamp_label
from strategy import (
    AGGRESSIVE_PARAMETERS,
    DEFENSIVE_PARAMETERS,
    build_strategy_callback,
    warmup_bars,
    with_overrides,
)


@dataclass(frozen=True)
class ValidationScenario:
    name: str
    params: object
    rebalance_frequency: int
    transaction_cost_pct: float
    rebalance_threshold: float


VALIDATION_SCENARIOS = [
    ValidationScenario(
        name="Wave_Rider_Aggressive",
        params=AGGRESSIVE_PARAMETERS,
        rebalance_frequency=config.REBALANCE_FREQUENCY,
        transaction_cost_pct=config.TRANSACTION_COST_PCT,
        rebalance_threshold=config.NO_TRADE_BAND,
    ),
    ValidationScenario(
        name="Wave_Rider_Defensive",
        params=DEFENSIVE_PARAMETERS,
        rebalance_frequency=config.REBALANCE_FREQUENCY,
        transaction_cost_pct=config.TRANSACTION_COST_PCT,
        rebalance_threshold=config.NO_TRADE_BAND,
    ),
    ValidationScenario(
        name="Legacy_Cash_No_Band",
        params=with_overrides(
            AGGRESSIVE_PARAMETERS,
            parking_asset=None,
        ),
        rebalance_frequency=config.REBALANCE_FREQUENCY,
        transaction_cost_pct=config.TRANSACTION_COST_PCT,
        rebalance_threshold=0.00,
    ),
    ValidationScenario(
        name="Cash_2pct_Band",
        params=with_overrides(
            AGGRESSIVE_PARAMETERS,
            parking_asset=None,
        ),
        rebalance_frequency=config.REBALANCE_FREQUENCY,
        transaction_cost_pct=config.TRANSACTION_COST_PCT,
        rebalance_threshold=0.02,
    ),
    ValidationScenario(
        name="Parking_IEF_No_Band",
        params=with_overrides(
            AGGRESSIVE_PARAMETERS,
            parking_asset="INT_BOND",
        ),
        rebalance_frequency=config.REBALANCE_FREQUENCY,
        transaction_cost_pct=config.TRANSACTION_COST_PCT,
        rebalance_threshold=0.00,
    ),
    ValidationScenario(
        name="Parking_SHY_2pct",
        params=with_overrides(
            AGGRESSIVE_PARAMETERS,
            parking_asset="CASH",
        ),
        rebalance_frequency=config.REBALANCE_FREQUENCY,
        transaction_cost_pct=config.TRANSACTION_COST_PCT,
        rebalance_threshold=0.02,
    ),
    ValidationScenario(
        name="Parking_TIP_2pct",
        params=with_overrides(
            AGGRESSIVE_PARAMETERS,
            parking_asset="INFL_LINKED",
        ),
        rebalance_frequency=config.REBALANCE_FREQUENCY,
        transaction_cost_pct=config.TRANSACTION_COST_PCT,
        rebalance_threshold=0.02,
    ),
    ValidationScenario(
        name="Target_Vol_12pct",
        params=with_overrides(
            AGGRESSIVE_PARAMETERS,
            target_vol=0.12,
        ),
        rebalance_frequency=config.REBALANCE_FREQUENCY,
        transaction_cost_pct=config.TRANSACTION_COST_PCT,
        rebalance_threshold=config.NO_TRADE_BAND,
    ),
    ValidationScenario(
        name="Target_Vol_14pct",
        params=with_overrides(
            AGGRESSIVE_PARAMETERS,
            target_vol=0.14,
        ),
        rebalance_frequency=config.REBALANCE_FREQUENCY,
        transaction_cost_pct=config.TRANSACTION_COST_PCT,
        rebalance_threshold=config.NO_TRADE_BAND,
    ),
    ValidationScenario(
        name="Max_Weight_35pct",
        params=with_overrides(
            AGGRESSIVE_PARAMETERS,
            max_weight=0.35,
        ),
        rebalance_frequency=config.REBALANCE_FREQUENCY,
        transaction_cost_pct=config.TRANSACTION_COST_PCT,
        rebalance_threshold=config.NO_TRADE_BAND,
    ),
    ValidationScenario(
        name="Five_Holdings",
        params=with_overrides(
            AGGRESSIVE_PARAMETERS,
            max_assets=5,
        ),
        rebalance_frequency=config.REBALANCE_FREQUENCY,
        transaction_cost_pct=config.TRANSACTION_COST_PCT,
        rebalance_threshold=config.NO_TRADE_BAND,
    ),
    ValidationScenario(
        name="Target_Vol_12_Max35",
        params=with_overrides(
            AGGRESSIVE_PARAMETERS,
            target_vol=0.12,
            max_weight=0.35,
        ),
        rebalance_frequency=config.REBALANCE_FREQUENCY,
        transaction_cost_pct=config.TRANSACTION_COST_PCT,
        rebalance_threshold=config.NO_TRADE_BAND,
    ),
    ValidationScenario(
        name="No_Trade_Band_5pct",
        params=with_overrides(
            AGGRESSIVE_PARAMETERS,
        ),
        rebalance_frequency=config.REBALANCE_FREQUENCY,
        transaction_cost_pct=config.TRANSACTION_COST_PCT,
        rebalance_threshold=0.05,
    ),
    ValidationScenario(
        name="Cost_5bps",
        params=AGGRESSIVE_PARAMETERS,
        rebalance_frequency=config.REBALANCE_FREQUENCY,
        transaction_cost_pct=0.0005,
        rebalance_threshold=config.NO_TRADE_BAND,
    ),
    ValidationScenario(
        name="Cost_7_5bps",
        params=AGGRESSIVE_PARAMETERS,
        rebalance_frequency=config.REBALANCE_FREQUENCY,
        transaction_cost_pct=0.00075,
        rebalance_threshold=config.NO_TRADE_BAND,
    ),
    ValidationScenario(
        name="Cost_10bps",
        params=AGGRESSIVE_PARAMETERS,
        rebalance_frequency=config.REBALANCE_FREQUENCY,
        transaction_cost_pct=0.0010,
        rebalance_threshold=config.NO_TRADE_BAND,
    ),
    ValidationScenario(
        name="Cost_15bps",
        params=AGGRESSIVE_PARAMETERS,
        rebalance_frequency=config.REBALANCE_FREQUENCY,
        transaction_cost_pct=0.0015,
        rebalance_threshold=config.NO_TRADE_BAND,
    ),
    ValidationScenario(
        name="Cost_20bps",
        params=AGGRESSIVE_PARAMETERS,
        rebalance_frequency=config.REBALANCE_FREQUENCY,
        transaction_cost_pct=0.0020,
        rebalance_threshold=config.NO_TRADE_BAND,
    ),
]


def _stats_frame(result_stats: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame([asdict(stat) for stat in result_stats.values()])


def _slice_stats(
    name: str,
    history: pd.DataFrame,
    start_date: str,
    end_date: str | None,
) -> dict[str, object] | None:
    sliced = history.loc[start_date:end_date].copy()
    if len(sliced) < 2:
        return None

    first_value_col = f"{name} Value"
    initial_capital = float(sliced[first_value_col].iloc[0])
    stat = build_stats(name, sliced, initial_capital)
    row = asdict(stat)
    row["window_start"] = start_date
    row["window_end"] = end_date or "latest"
    return row


def run_walkforward_validation(history: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    walkforward_rows: list[dict[str, object]] = []
    for label, start_date, end_date in config.WALKFORWARD_WINDOWS:
        row = _slice_stats("Wave_Rider_Aggressive", history, start_date, end_date)
        if row is None:
            continue
        row["window"] = label
        walkforward_rows.append(row)

    walkforward_df = pd.DataFrame(walkforward_rows)
    if not walkforward_df.empty:
        walkforward_df.to_csv(output_dir / "walkforward_summary.csv", index=False)
        plot_walkforward_summary(walkforward_df, output_dir / "walkforward_summary.png")
    return walkforward_df


def run_validation(refresh_data: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prices = load_data(
        config.ASSET_TO_TICKER,
        start=config.START_DATE,
        cache_file=config.CACHE_FILE,
        refresh=refresh_data,
    )

    validation_rows: list[dict[str, object]] = []
    benchmark_rows: list[dict[str, object]] = []
    baseline_history: pd.DataFrame | None = None

    validation_dir = config.RESULTS_DIR / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)

    for scenario in VALIDATION_SCENARIOS:
        result = run_backtest(
            prices,
            build_strategy_callback(scenario.params),
            strategy_name=scenario.name,
            initial_capital=config.INITIAL_CAPITAL,
            rebalance_frequency=scenario.rebalance_frequency,
            warmup_bars=warmup_bars(scenario.params),
            transaction_cost_pct=scenario.transaction_cost_pct,
            rebalance_threshold=scenario.rebalance_threshold,
            benchmark_weights=config.BENCHMARKS if scenario.name == "Wave_Rider_Aggressive" else None,
        )

        variant_dir = validation_dir / scenario.name.lower()
        variant_dir.mkdir(parents=True, exist_ok=True)
        result.history.to_csv(variant_dir / "history.csv")
        if not result.signal_log.empty:
            result.signal_log.to_csv(variant_dir / "signal_log.csv")

        stats_df = _stats_frame(result.stats)
        stats_df.to_csv(variant_dir / "stats.csv", index=False)
        (variant_dir / "stats_table.txt").write_text(format_stats_table(result.stats) + "\n")

        row = asdict(result.stats[scenario.name])
        row["rebalance_frequency"] = scenario.rebalance_frequency
        row["transaction_cost_pct"] = scenario.transaction_cost_pct
        row["rebalance_threshold"] = scenario.rebalance_threshold
        validation_rows.append(row)
        if scenario.name == "Wave_Rider_Aggressive":
            baseline_history = result.history.copy()
            for benchmark_name, stat in result.stats.items():
                if benchmark_name == scenario.name:
                    continue
                benchmark_rows.append(asdict(stat))

    validation_summary = pd.DataFrame(validation_rows).sort_values(
        by=["calmar", "cagr_pct"],
        ascending=False,
    )
    benchmark_summary = pd.DataFrame(benchmark_rows).sort_values(
        by="calmar",
        ascending=False,
    )

    validation_summary.to_csv(validation_dir / "validation_summary.csv", index=False)
    benchmark_summary.to_csv(validation_dir / "benchmark_summary.csv", index=False)
    (validation_dir / "validation_summary.txt").write_text(
        format_stats_table({row["name"]: type("Obj", (), row)() for row in validation_rows}) + "\n"
    )

    plot_validation_summary(validation_summary, validation_dir / "validation_comparison.png")
    plot_benchmark_comparison(benchmark_summary, validation_dir / "benchmark_comparison.png")

    walkforward_summary = pd.DataFrame()
    if baseline_history is not None:
        walkforward_summary = run_walkforward_validation(baseline_history, validation_dir)

    run_dir = config.RUNS_DIR / f"{timestamp_label()}_validation"
    archive_selected_outputs(
        validation_dir,
        run_dir,
        [
            "validation_summary.csv",
            "benchmark_summary.csv",
            "validation_summary.txt",
            "validation_comparison.png",
            "benchmark_comparison.png",
            "walkforward_summary.csv",
            "walkforward_summary.png",
        ],
    )
    save_run_metadata(
        run_dir / "run_metadata.json",
        {
            "run_type": "validation",
            "universe_preset": config.UNIVERSE_PRESET,
            "start_date": config.START_DATE,
            "transaction_cost_pct": config.TRANSACTION_COST_PCT,
            "aggressive_parameters": vars(AGGRESSIVE_PARAMETERS),
            "defensive_parameters": vars(DEFENSIVE_PARAMETERS),
            "variants": [
                {
                    "name": scenario.name,
                    "params": vars(scenario.params),
                    "rebalance_frequency": scenario.rebalance_frequency,
                    "transaction_cost_pct": scenario.transaction_cost_pct,
                    "rebalance_threshold": scenario.rebalance_threshold,
                }
                for scenario in VALIDATION_SCENARIOS
            ],
        },
    )

    return validation_summary, benchmark_summary, walkforward_summary


if __name__ == "__main__":
    validation_summary, benchmark_summary, walkforward_summary = run_validation()
    print("Validation variants:")
    print(validation_summary.to_string(index=False))
    print("\nBaseline benchmark comparison:")
    if benchmark_summary.empty:
        print("No benchmark results produced.")
    else:
        print(benchmark_summary.to_string(index=False))
    print("\nWalk-forward summary:")
    if walkforward_summary.empty:
        print("No walk-forward windows produced.")
    else:
        print(walkforward_summary.to_string(index=False))
