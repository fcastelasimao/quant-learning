from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


def _prepare_matplotlib():
    mpl_dir = Path(__file__).resolve().parent / ".mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _value_columns(history: pd.DataFrame) -> list[str]:
    return [column for column in history.columns if column.endswith(" Value")]


def _drawdown_series(value_series: pd.Series) -> pd.Series:
    peak = value_series.cummax()
    return ((value_series - peak) / peak) * 100


def plot_backtest_overview(history: pd.DataFrame, output_path: Path) -> None:
    plt = _prepare_matplotlib()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    value_columns = _value_columns(history)
    if not value_columns:
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    for column in value_columns:
        label = column.replace(" Value", "")
        series = history[column].dropna()
        if series.empty:
            continue
        normalized = series / series.iloc[0]
        axes[0].plot(series.index, normalized, label=label, linewidth=2)
        axes[1].plot(series.index, _drawdown_series(series), label=label, linewidth=1.8)

    axes[0].set_title("Equity Curves (Normalized)")
    axes[0].set_ylabel("Growth of $1")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].set_title("Drawdowns")
    axes[1].set_ylabel("Drawdown %")
    axes[1].set_xlabel("Date")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_strategy_state(
    history: pd.DataFrame,
    signal_log: pd.DataFrame,
    strategy_name: str,
    output_path: Path,
) -> None:
    if history.empty:
        return

    plt = _prepare_matplotlib()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

    cash_col = f"{strategy_name} CashWeight"
    exposure_col = f"{strategy_name} Exposure"
    drawdown_col = f"{strategy_name} Drawdown"

    if cash_col in history.columns and exposure_col in history.columns:
        axes[0].plot(history.index, history[exposure_col] * 100, label="Exposure %", linewidth=2)
        axes[0].plot(history.index, history[cash_col] * 100, label="Cash %", linewidth=2)
        axes[0].set_title("Portfolio State")
        axes[0].set_ylabel("Percent")
        axes[0].legend(loc="best")
        axes[0].grid(alpha=0.25)

    if drawdown_col in history.columns:
        axes[1].plot(history.index, history[drawdown_col] * 100, color="tab:red", linewidth=2)
        axes[1].set_title("Strategy Drawdown")
        axes[1].set_ylabel("Drawdown %")
        axes[1].grid(alpha=0.25)

    if not signal_log.empty:
        aligned = signal_log.reindex(history.index).ffill()
        if "breadth" in aligned.columns:
            axes[2].plot(aligned.index, aligned["breadth"], label="Breadth", linewidth=2)
        if "gross_scale" in aligned.columns:
            axes[2].plot(aligned.index, aligned["gross_scale"], label="Defense Scale", linewidth=2)
        axes[2].set_title("Signal State")
        axes[2].set_ylabel("Level")
        axes[2].set_xlabel("Date")
        axes[2].legend(loc="best")
        axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_validation_summary(validation_summary: pd.DataFrame, output_path: Path) -> None:
    if validation_summary.empty:
        return

    plt = _prepare_matplotlib()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = validation_summary.copy()
    df = df.sort_values("calmar", ascending=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 8), sharey=True)

    axes[0].barh(df["name"], df["calmar"], color="tab:blue")
    axes[0].set_title("Calmar")
    axes[0].grid(axis="x", alpha=0.25)

    axes[1].barh(df["name"], df["cagr_pct"], color="tab:green")
    axes[1].set_title("CAGR %")
    axes[1].grid(axis="x", alpha=0.25)

    axes[2].barh(df["name"], df["max_drawdown_pct"], color="tab:red")
    axes[2].set_title("Max Drawdown %")
    axes[2].grid(axis="x", alpha=0.25)

    fig.suptitle("Validation Variant Comparison")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_benchmark_comparison(benchmark_summary: pd.DataFrame, output_path: Path) -> None:
    if benchmark_summary.empty:
        return

    plt = _prepare_matplotlib()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = benchmark_summary.copy().sort_values("calmar", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    axes[0].barh(df["name"], df["calmar"], color="tab:purple")
    axes[0].set_title("Benchmark Calmar")
    axes[0].grid(axis="x", alpha=0.25)

    axes[1].barh(df["name"], df["cagr_pct"], color="tab:orange")
    axes[1].set_title("Benchmark CAGR %")
    axes[1].grid(axis="x", alpha=0.25)

    fig.suptitle("Baseline Benchmark Comparison")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_walkforward_summary(walkforward_summary: pd.DataFrame, output_path: Path) -> None:
    if walkforward_summary.empty:
        return

    plt = _prepare_matplotlib()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = walkforward_summary.copy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)
    axes[0].barh(df["window"], df["calmar"], color="tab:blue")
    axes[0].set_title("Walk-Forward Calmar")
    axes[0].grid(axis="x", alpha=0.25)

    axes[1].barh(df["window"], df["cagr_pct"], color="tab:green")
    axes[1].set_title("Walk-Forward CAGR %")
    axes[1].grid(axis="x", alpha=0.25)

    axes[2].barh(df["window"], df["max_drawdown_pct"], color="tab:red")
    axes[2].set_title("Walk-Forward Max Drawdown %")
    axes[2].grid(axis="x", alpha=0.25)

    fig.suptitle("Wave Rider Walk-Forward Windows")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
