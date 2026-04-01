from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import config
from plotting import _prepare_matplotlib
from research_io import archive_selected_outputs, save_run_metadata, timestamp_label


WINDOW_START = "2025-03-06"
ALL_WEATHER_FILE_GLOB = "*_allw_comparison_from2025-03-06_to*.xlsx"
ALL_WEATHER_HISTORY_GLOB = "*_oos_evaluate_6assets_M_*_rpavg_monthly_2022oos/backtest_history.csv"


def _compute_window_stats(
    history: pd.DataFrame,
    strategy_name: str,
    start_date: str,
    end_date: str | None = None,
) -> dict[str, object]:
    sliced = history.loc[pd.Timestamp(start_date) : pd.Timestamp(end_date) if end_date else None].copy()
    values = sliced[f"{strategy_name} Value"].dropna()
    if values.empty:
        raise ValueError(f"No values found for {strategy_name} from {start_date}.")

    peak = values.cummax()
    max_drawdown_pct = float(((values - peak) / peak).min() * 100)
    total_return_pct = float(((values.iloc[-1] / values.iloc[0]) - 1.0) * 100)
    years = len(values) / config.TRADING_DAYS_PER_YEAR
    cagr_pct = 0.0
    if years > 0:
        cagr_pct = float(((values.iloc[-1] / values.iloc[0]) ** (1.0 / years) - 1.0) * 100)
    calmar = cagr_pct / abs(max_drawdown_pct) if abs(max_drawdown_pct) > 1e-12 else 0.0

    return {
        "name": strategy_name,
        "window_start": values.index[0].date().isoformat(),
        "window_end": values.index[-1].date().isoformat(),
        "trading_days": len(values),
        "total_return_pct": total_return_pct,
        "cagr_pct": cagr_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "calmar": calmar,
        "source": "wave_rider_backtest",
    }


def _load_all_weather_rows(results_path: Path) -> pd.DataFrame:
    comparison_files = sorted(results_path.glob(ALL_WEATHER_FILE_GLOB))
    if not comparison_files:
        raise FileNotFoundError("No matching all-weather comparison workbook found.")

    latest_file = comparison_files[-1]
    raw = pd.read_excel(latest_file, sheet_name="ALLW Comparison", header=1)
    raw = raw.rename(
        columns={
            raw.columns[0]: "strategy",
            raw.columns[1]: "allocations",
            raw.columns[2]: "total_return_pct",
            raw.columns[3]: "cagr_pct",
            raw.columns[4]: "max_drawdown_pct",
            raw.columns[5]: "calmar",
        }
    )
    raw["strategy"] = raw["strategy"].astype(str)

    keep = {
        "6asset_rpavg_backtest (gross)": "All Weather DIY",
        "→ fee-adj 0.12% p.a.": "All Weather DIY Fee-Adj",
        "ALLW  (Bridgewater, gross)": "ALLW Gross",
        "SPY  (benchmark)": "SPY",
        "60/40  (SPY 60% / TLT 40%)": "All Weather 60/40",
    }

    rows: list[dict[str, object]] = []
    fee_adj_pending = False
    for _, row in raw.iterrows():
        label = row["strategy"]
        if label == "6asset_rpavg_backtest (gross)":
            fee_adj_pending = True
        elif label == "→ fee-adj 0.12% p.a." and not fee_adj_pending:
            continue
        else:
            fee_adj_pending = False

        if label not in keep:
            continue
        if label == "→ fee-adj 0.12% p.a.":
            output_name = "All Weather DIY Fee-Adj"
        else:
            output_name = keep[label]

        rows.append(
            {
                "name": output_name,
                "window_start": WINDOW_START,
                "window_end": "2026-03-30",
                "trading_days": 268,
                "total_return_pct": float(row["total_return_pct"]),
                "cagr_pct": float(row["cagr_pct"]),
                "max_drawdown_pct": float(row["max_drawdown_pct"]),
                "calmar": float(row["calmar"]),
                "source": f"all_weather_workbook:{latest_file.name}",
            }
        )

    if not rows:
        raise ValueError("Failed to extract all-weather rows from workbook.")
    return pd.DataFrame(rows)


def _plot_metrics(comparison: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    names = comparison["name"]

    axes[0].bar(names, comparison["cagr_pct"], color="#2f6db3")
    axes[0].set_title("CAGR %")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(names, comparison["max_drawdown_pct"], color="#b33f3f")
    axes[1].set_title("Max Drawdown %")
    axes[1].tick_params(axis="x", rotation=30)

    axes[2].bar(names, comparison["calmar"], color="#4a8f52")
    axes[2].set_title("Calmar")
    axes[2].tick_params(axis="x", rotation=30)

    fig.suptitle("Same-Window Comparison: 2025-03-06 to Latest")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _load_all_weather_growth(results_path: Path, start_date: str, end_date: str) -> pd.Series | None:
    history_files = sorted(results_path.glob(ALL_WEATHER_HISTORY_GLOB))
    if not history_files:
        return None

    latest_file = history_files[-1]
    history = pd.read_csv(latest_file, parse_dates=["Date"]).set_index("Date").sort_index()
    if "All Weather Value" not in history.columns:
        return None

    sliced = history.loc[pd.Timestamp(start_date) : pd.Timestamp(end_date), "All Weather Value"].dropna()
    if sliced.empty:
        return None
    return (sliced / sliced.iloc[0]).rename("All Weather DIY Monthly")


def _plot_growth_comparison(
    wave_history: pd.DataFrame,
    start_date: str,
    end_date: str,
    output_path: Path,
) -> None:
    plt = _prepare_matplotlib()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sliced = wave_history.loc[pd.Timestamp(start_date) : pd.Timestamp(end_date)].copy()
    if sliced.empty:
        return

    fig, ax = plt.subplots(figsize=(13, 6.5))

    for name in ["Wave Rider Aggressive", "Wave Rider Defensive"]:
        column = f"{name} Value"
        if column not in sliced.columns:
            continue
        series = sliced[column].dropna()
        if series.empty:
            continue
        ax.plot(series.index, series / series.iloc[0], linewidth=2.2, label=name)

    all_weather_growth = _load_all_weather_growth(
        Path("All_weather_portfolio/results"),
        start_date,
        end_date,
    )
    if all_weather_growth is not None:
        ax.plot(
            all_weather_growth.index,
            all_weather_growth,
            linewidth=2.0,
            linestyle="--",
            label=all_weather_growth.name,
        )

    ax.set_title("Same-Window Growth Comparison")
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("Date")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_same_window_comparison() -> pd.DataFrame:
    wave_history = pd.read_csv(config.RESULTS_DIR / "backtest_history.csv", index_col=0, parse_dates=True)
    all_weather_results = _load_all_weather_rows(Path("All_weather_portfolio/results"))
    comparison_end = str(all_weather_results["window_end"].iloc[0])
    comparison_rows = [
        _compute_window_stats(wave_history, "Wave Rider Aggressive", WINDOW_START, comparison_end),
        _compute_window_stats(wave_history, "Wave Rider Defensive", WINDOW_START, comparison_end),
    ]
    comparison = pd.DataFrame(comparison_rows)
    comparison = pd.concat([comparison, all_weather_results], ignore_index=True)
    comparison = comparison.sort_values(by=["calmar", "cagr_pct"], ascending=False)

    output_csv = config.RESULTS_DIR / "same_window_comparison.csv"
    output_txt = config.RESULTS_DIR / "same_window_comparison.txt"
    output_png = config.RESULTS_DIR / "same_window_comparison.png"
    output_growth_png = config.RESULTS_DIR / "same_window_growth_comparison.png"

    comparison.to_csv(output_csv, index=False)
    output_txt.write_text(comparison.to_string(index=False) + "\n")
    _plot_metrics(comparison, output_png)
    _plot_growth_comparison(wave_history, WINDOW_START, comparison_end, output_growth_png)

    run_dir = config.RUNS_DIR / f"{timestamp_label()}_same_window_compare"
    archive_selected_outputs(
        config.RESULTS_DIR,
        run_dir,
        [
            "same_window_comparison.csv",
            "same_window_comparison.txt",
            "same_window_comparison.png",
            "same_window_growth_comparison.png",
        ],
    )
    save_run_metadata(
        run_dir / "run_metadata.json",
        {
            "run_type": "same_window_compare",
            "window_start": WINDOW_START,
            "wave_rider_source": str(config.RESULTS_DIR / "backtest_history.csv"),
            "all_weather_file_glob": ALL_WEATHER_FILE_GLOB,
        },
    )
    return comparison


if __name__ == "__main__":
    result = run_same_window_comparison()
    print(result.to_string(index=False))
