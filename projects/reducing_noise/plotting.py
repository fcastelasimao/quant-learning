from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from filters import fft_split_series, rolling_fft_components


def _prepare_matplotlib():
    mpl_dir = Path(__file__).resolve().parent / ".mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _write_html(fig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)


def plot_denoised_series_html(
    prices: pd.Series,
    primary_smoothed: pd.Series,
    primary_method: str,
    output_path: Path,
) -> None:
    import plotly.graph_objects as go

    log_price = np.log(prices.astype(float).dropna())
    hindsight, _ = fft_split_series(log_price, retained_energy=0.95)
    plot_index = log_price.index[-1000:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_index, y=log_price.reindex(plot_index), name="SPY log price", mode="lines"))
    fig.add_trace(
        go.Scatter(
            x=plot_index,
            y=hindsight.reindex(plot_index),
            name="Full-sample FFT hindsight only",
            mode="lines",
            line={"dash": "dash"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=plot_index,
            y=primary_smoothed.reindex(plot_index),
            name=f"Primary causal FFT: {primary_method}",
            mode="lines",
        )
    )
    fig.update_layout(
        title="Denoised SPY Log Price",
        xaxis_title="Date",
        yaxis_title="Log price",
        hovermode="x unified",
        template="plotly_white",
    )
    _write_html(fig, output_path)


def plot_denoised_series(
    prices: pd.Series,
    primary_smoothed: pd.Series,
    primary_method: str,
    output_path: Path,
) -> None:
    plt = _prepare_matplotlib()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log_price = np.log(prices.astype(float).dropna())
    hindsight, _ = fft_split_series(log_price, retained_energy=0.95)
    plot_index = log_price.index[-1000:]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(plot_index, log_price.reindex(plot_index), label="SPY log price", linewidth=1.5, alpha=0.75)
    ax.plot(
        plot_index,
        hindsight.reindex(plot_index),
        label="Full-sample FFT hindsight only",
        linewidth=2,
        linestyle="--",
    )
    ax.plot(plot_index, primary_smoothed.reindex(plot_index), label=f"Primary causal FFT: {primary_method}", linewidth=2)
    ax.set_title("Denoised SPY Log Price")
    ax.set_ylabel("Log price")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_fft_component_split(prices: pd.Series, output_path: Path) -> None:
    plt = _prepare_matplotlib()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log_price = np.log(prices.astype(float).dropna())
    main, residual = rolling_fft_components(log_price, window=128, retained_energy=0.95)
    residual_slope = residual.diff()
    plot_index = log_price.index[-1000:]

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    axes[0].plot(plot_index, log_price.reindex(plot_index), label="SPY log price", linewidth=1.5)
    axes[0].plot(plot_index, main.reindex(plot_index), label="Causal FFT retained component", linewidth=2)
    axes[0].set_title("Retained FFT Component")
    axes[0].set_ylabel("Log price")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(plot_index, residual.reindex(plot_index), color="tab:orange", linewidth=1.5)
    axes[1].axhline(0, color="black", linewidth=0.8, alpha=0.5)
    axes[1].set_title("Residual Component Previously Called Noise")
    axes[1].set_ylabel("Residual")
    axes[1].grid(alpha=0.25)

    axes[2].plot(plot_index, residual_slope.reindex(plot_index), color="tab:green", linewidth=1.2)
    axes[2].axhline(0, color="black", linewidth=0.8, alpha=0.5)
    axes[2].set_title("Residual Slope")
    axes[2].set_ylabel("Delta residual")
    axes[2].set_xlabel("Date")
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_fft_component_split_html(prices: pd.Series, output_path: Path) -> None:
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    log_price = np.log(prices.astype(float).dropna())
    main, residual = rolling_fft_components(log_price, window=128, retained_energy=0.95)
    residual_slope = residual.diff()
    plot_index = log_price.index[-1000:]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=[
            "Retained FFT Component",
            "Residual Component Previously Called Noise",
            "Residual Slope",
        ],
        vertical_spacing=0.08,
    )
    fig.add_trace(go.Scatter(x=plot_index, y=log_price.reindex(plot_index), name="SPY log price", mode="lines"), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=plot_index, y=main.reindex(plot_index), name="Causal FFT retained component", mode="lines"),
        row=1,
        col=1,
    )
    fig.add_trace(go.Scatter(x=plot_index, y=residual.reindex(plot_index), name="Residual", mode="lines"), row=2, col=1)
    fig.add_hline(y=0, line_width=1, line_color="black", opacity=0.5, row=2, col=1)
    fig.add_trace(
        go.Scatter(x=plot_index, y=residual_slope.reindex(plot_index), name="Residual slope", mode="lines"),
        row=3,
        col=1,
    )
    fig.add_hline(y=0, line_width=1, line_color="black", opacity=0.5, row=3, col=1)
    fig.update_layout(
        title="FFT Component Split",
        hovermode="x unified",
        template="plotly_white",
        height=900,
    )
    fig.update_yaxes(title_text="Log price", row=1, col=1)
    fig.update_yaxes(title_text="Residual", row=2, col=1)
    fig.update_yaxes(title_text="Delta residual", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    _write_html(fig, output_path)


def _selected_equity_columns(summary: pd.DataFrame, equity_curves: pd.DataFrame, primary_method: str) -> list[str]:
    columns = ["buy_and_hold", primary_method]

    for family in ["fft", "fft_noise"]:
        rows = summary[summary["family"] == family]
        if not rows.empty:
            columns.append(str(rows.sort_values(["sharpe", "correlation"], ascending=False).iloc[0]["name"]))

    columns.append("ma_60")
    return list(dict.fromkeys(column for column in columns if column in equity_curves.columns))


def plot_equity_curves(
    equity_curves: pd.DataFrame,
    summary: pd.DataFrame,
    prices: pd.Series,
    primary_smoothed: pd.Series,
    primary_method: str,
    output_path: Path,
) -> None:
    plt = _prepare_matplotlib()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    columns = _selected_equity_columns(summary, equity_curves, primary_method)
    log_price = np.log(prices.astype(float).dropna())

    fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True)
    axes[0].plot(log_price.index, log_price, label="SPY log price", linewidth=1.4, alpha=0.75)
    axes[0].plot(primary_smoothed.index, primary_smoothed, label=f"Primary FFT: {primary_method}", linewidth=1.8)
    axes[0].set_title("SPY Log Price vs Primary FFT Component")
    axes[0].set_ylabel("Log price")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    for column in columns:
        series = equity_curves[column].dropna()
        if series.empty:
            continue
        normalized = series / series.iloc[0]
        axes[1].plot(normalized.index, normalized, label=column, linewidth=2)
        drawdown = normalized / normalized.cummax() - 1.0
        axes[2].plot(drawdown.index, drawdown * 100, label=column, linewidth=1.8)

    axes[1].set_title("Equity Curves")
    axes[1].set_ylabel("Growth of $1")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.25, which="both")
    axes[1].legend(loc="best")

    axes[2].set_title("Drawdowns")
    axes[2].set_ylabel("Drawdown %")
    axes[2].set_xlabel("Date")
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_equity_curves_html(
    equity_curves: pd.DataFrame,
    summary: pd.DataFrame,
    prices: pd.Series,
    primary_smoothed: pd.Series,
    primary_method: str,
    output_path: Path,
) -> None:
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    columns = _selected_equity_columns(summary, equity_curves, primary_method)
    log_price = np.log(prices.astype(float).dropna())

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=[
            "SPY Log Price vs Primary FFT Component",
            "Equity Curves",
            "Drawdowns",
        ],
        vertical_spacing=0.08,
    )
    fig.add_trace(go.Scatter(x=log_price.index, y=log_price, name="SPY log price", mode="lines"), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=primary_smoothed.index, y=primary_smoothed, name=f"Primary FFT: {primary_method}", mode="lines"),
        row=1,
        col=1,
    )

    for column in columns:
        series = equity_curves[column].dropna()
        if series.empty:
            continue
        normalized = series / series.iloc[0]
        drawdown = normalized / normalized.cummax() - 1.0
        fig.add_trace(go.Scatter(x=normalized.index, y=normalized, name=column, mode="lines"), row=2, col=1)
        fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown * 100, name=f"{column} drawdown", mode="lines"), row=3, col=1)

    fig.update_layout(
        title="SPY FFT Research: Price, Equity, Drawdowns",
        hovermode="x unified",
        template="plotly_white",
        height=1000,
    )
    fig.update_yaxes(title_text="Log price", row=1, col=1)
    fig.update_yaxes(title_text="Growth of $1", type="log", row=2, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    _write_html(fig, output_path)


def plot_fft_support_resistance(
    prices: pd.Series,
    fft_component: pd.Series,
    residual: pd.Series,
    events: pd.DataFrame,
    output_path: Path,
) -> None:
    plt = _prepare_matplotlib()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log_price = np.log(prices.astype(float).dropna())
    fig, axes = plt.subplots(2, 1, figsize=(15, 11), sharex=True)

    axes[0].plot(log_price.index, log_price, label="SPY log price", linewidth=1.3, alpha=0.7)
    axes[0].plot(fft_component.index, fft_component, label="Causal FFT retained component", linewidth=1.8)

    if not events.empty:
        event_dates = pd.to_datetime(events["date"])
        event_log_price = log_price.reindex(event_dates).to_numpy()
        event_dates_array = event_dates.to_numpy()
        floor_bounce = (events["outcome"] == "floor_bounce").to_numpy()
        ceiling_rejection = (events["outcome"] == "ceiling_rejection").to_numpy()
        breaks = events["outcome"].isin(["upward_break", "downward_break"]).to_numpy()

        axes[0].scatter(
            event_dates_array[floor_bounce],
            event_log_price[floor_bounce],
            marker="^",
            s=35,
            color="tab:green",
            label="Floor bounce",
            zorder=3,
        )
        axes[0].scatter(
            event_dates_array[ceiling_rejection],
            event_log_price[ceiling_rejection],
            marker="v",
            s=35,
            color="tab:red",
            label="Ceiling rejection",
            zorder=3,
        )
        axes[0].scatter(
            event_dates_array[breaks],
            event_log_price[breaks],
            marker="x",
            s=35,
            color="black",
            label="Break",
            zorder=3,
        )

    axes[0].set_title("FFT Support/Resistance Touches")
    axes[0].set_ylabel("Log price")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(residual.index, residual, label="Residual: log price - FFT component", linewidth=1.2)
    axes[1].axhline(0, color="black", linewidth=0.8, alpha=0.6)
    axes[1].set_title("Distance From FFT Component")
    axes[1].set_ylabel("Residual")
    axes[1].set_xlabel("Date")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_fft_support_resistance_html(
    prices: pd.Series,
    fft_component: pd.Series,
    residual: pd.Series,
    events: pd.DataFrame,
    output_path: Path,
) -> None:
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    log_price = np.log(prices.astype(float).dropna())
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=["FFT Support/Resistance Touches", "Distance From FFT Component"],
        vertical_spacing=0.08,
    )
    fig.add_trace(go.Scatter(x=log_price.index, y=log_price, name="SPY log price", mode="lines"), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=fft_component.index, y=fft_component, name="Causal FFT retained component", mode="lines"),
        row=1,
        col=1,
    )

    if not events.empty:
        event_dates = pd.to_datetime(events["date"])
        event_log_price = log_price.reindex(event_dates).to_numpy()
        event_dates_array = event_dates.to_numpy()
        marker_specs = [
            ("floor_bounce", "Floor bounce", "triangle-up", "green"),
            ("ceiling_rejection", "Ceiling rejection", "triangle-down", "red"),
        ]
        for outcome, label, symbol, color in marker_specs:
            mask = (events["outcome"] == outcome).to_numpy()
            fig.add_trace(
                go.Scatter(
                    x=event_dates_array[mask],
                    y=event_log_price[mask],
                    name=label,
                    mode="markers",
                    marker={"symbol": symbol, "size": 9, "color": color},
                    customdata=events.loc[mask, ["side", "outcome", "future_return_20d"]].to_numpy(),
                    hovertemplate=(
                        "Date=%{x}<br>Log price=%{y:.4f}<br>"
                        "Side=%{customdata[0]}<br>Outcome=%{customdata[1]}<br>"
                        "20d return=%{customdata[2]:.2%}<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )
        break_mask = events["outcome"].isin(["upward_break", "downward_break"]).to_numpy()
        fig.add_trace(
            go.Scatter(
                x=event_dates_array[break_mask],
                y=event_log_price[break_mask],
                name="Break",
                mode="markers",
                marker={"symbol": "x", "size": 9, "color": "black"},
                customdata=events.loc[break_mask, ["side", "outcome", "future_return_20d"]].to_numpy(),
                hovertemplate=(
                    "Date=%{x}<br>Log price=%{y:.4f}<br>"
                    "Side=%{customdata[0]}<br>Outcome=%{customdata[1]}<br>"
                    "20d return=%{customdata[2]:.2%}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(x=residual.index, y=residual, name="Residual: log price - FFT component", mode="lines"),
        row=2,
        col=1,
    )
    fig.add_hline(y=0, line_width=1, line_color="black", opacity=0.6, row=2, col=1)
    fig.update_layout(
        title="FFT Support/Resistance Investigation",
        hovermode="x unified",
        template="plotly_white",
        height=850,
    )
    fig.update_yaxes(title_text="Log price", row=1, col=1)
    fig.update_yaxes(title_text="Residual", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    _write_html(fig, output_path)


def plot_fft_accuracy_over_time(accuracy: pd.DataFrame, output_path: Path) -> None:
    if accuracy.empty:
        return

    plt = _prepare_matplotlib()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dates = pd.to_datetime(accuracy["date"])
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates, accuracy["cumulative_hit_rate"] * 100, label="Cumulative hit rate", linewidth=2)
    ax.plot(dates, accuracy["rolling_hit_rate"] * 100, label="Rolling event hit rate", linewidth=2)
    ax.axhline(50, color="black", linewidth=0.9, alpha=0.6, linestyle="--")
    ax.set_title("FFT Floor/Ceiling Prediction Accuracy Over Time")
    ax.set_ylabel("Correct predictions %")
    ax.set_xlabel("Date")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_fft_accuracy_over_time_html(accuracy: pd.DataFrame, output_path: Path) -> None:
    if accuracy.empty:
        return

    import plotly.graph_objects as go

    dates = pd.to_datetime(accuracy["date"])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=accuracy["cumulative_hit_rate"] * 100,
            name="Cumulative hit rate",
            mode="lines",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=accuracy["rolling_hit_rate"] * 100,
            name="Rolling event hit rate",
            mode="lines",
        )
    )
    fig.add_hline(y=50, line_dash="dash", line_color="black", opacity=0.6)
    fig.update_layout(
        title="FFT Floor/Ceiling Prediction Accuracy Over Time",
        xaxis_title="Date",
        yaxis_title="Correct predictions %",
        yaxis={"range": [0, 100]},
        hovermode="x unified",
        template="plotly_white",
    )
    _write_html(fig, output_path)
