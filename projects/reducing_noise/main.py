from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from experiments import (
    best_method_by_family,
    best_research_method,
    compute_smoothed_log_price,
    default_method_specs,
    get_method_spec,
    run_parameter_sweep,
)
from filters import rolling_fft_components
from plotting import (
    plot_denoised_series,
    plot_denoised_series_html,
    plot_equity_curves,
    plot_equity_curves_html,
    plot_fft_accuracy_over_time,
    plot_fft_accuracy_over_time_html,
    plot_fft_component_split,
    plot_fft_component_split_html,
    plot_fft_support_resistance,
    plot_fft_support_resistance_html,
)
from support_resistance import TouchConfig, accuracy_over_time, find_touch_events, summarize_touch_events


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"


def load_prices(ticker: str, start: str, *, refresh: bool = False) -> pd.Series:
    cache_file = DATA_DIR / f"{ticker.lower()}_prices.csv"
    if cache_file.exists() and not refresh:
        cached = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        prices = cached["price"].sort_index()
        requested_start = pd.Timestamp(start)
        if not prices.empty and prices.index.min().normalize() <= requested_start.normalize():
            return prices.loc[prices.index >= requested_start]

    raw = yf.download(ticker, start=start, progress=False, auto_adjust=False)
    if raw.empty:
        raise RuntimeError(f"No data downloaded for {ticker}")

    if isinstance(raw.columns, pd.MultiIndex):
        field = "Adj Close" if "Adj Close" in raw.columns.get_level_values(0) else "Close"
        prices = raw[field].squeeze()
    else:
        field = "Adj Close" if "Adj Close" in raw.columns else "Close"
        prices = raw[field].squeeze()

    prices = prices.dropna().sort_index()
    prices.name = "price"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    prices.to_csv(cache_file)
    return prices


def _format_pct(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value * 100:.2f}%"


def write_research_notes(
    summary: pd.DataFrame,
    best_row: pd.Series,
    touch_summary: pd.DataFrame,
    accuracy: pd.DataFrame,
    output_path: Path,
    ticker: str,
    start: str,
) -> None:
    benchmark = summary[summary["name"] == "buy_and_hold"].iloc[0]
    best_fft = best_method_by_family(summary, "fft") or "n/a"
    best_fft_noise = best_method_by_family(summary, "fft_noise") or "n/a"
    overall_touch = touch_summary[touch_summary["side"] == "all"]
    touch_line = "No FFT touch events were found with the current thresholds."
    if not overall_touch.empty:
        row = overall_touch.iloc[0]
        touch_line = (
            f"Touch events: {int(row['events'])}; "
            f"20d floor/ceiling hit rate: {_format_pct(row['hit_rate_20d'])}."
        )
    accuracy_line = "No rolling accuracy series was produced."
    if not accuracy.empty:
        accuracy_line = (
            f"Latest cumulative hit rate: {_format_pct(accuracy['cumulative_hit_rate'].iloc[-1])}; "
            f"latest rolling hit rate: {_format_pct(accuracy['rolling_hit_rate'].iloc[-1])}."
        )
    notes = f"""# Reducing Noise Research Notes

Data: `{ticker}` adjusted close from `{start}`.

Best causal method by strategy Sharpe: `{best_row['name']}`.
Best retained FFT method: `{best_fft}`.
Best FFT residual method: `{best_fft_noise}`.

{touch_line}
{accuracy_line}

| Metric | Best Method | Buy & Hold |
| --- | ---: | ---: |
| Sharpe | {best_row['sharpe']:.3f} | {benchmark['sharpe']:.3f} |
| CAGR | {_format_pct(best_row['cagr'])} | {_format_pct(benchmark['cagr'])} |
| Max drawdown | {_format_pct(best_row['max_drawdown'])} | {_format_pct(benchmark['max_drawdown'])} |
| Final equity | {best_row['final_equity']:.3f} | {benchmark['final_equity']:.3f} |

## Interpretation

- Performance used rolling causal filters only.
- Signals are shifted by one day before returns are applied.
- The full-sample FFT plot is included only as a visual warning about hindsight smoothing.
- FFT residual methods treat the discarded frequencies as a separate component instead of deleting them.
- The FFT support/resistance markers are a hypothesis; use the event and accuracy CSVs to test whether they persist through time.
- A better-looking smoothed curve is not enough; the useful question is whether the slope predicts next-day returns after lag and turnover.

## Next Ideas

- Add walk-forward train/test parameter selection instead of ranking on the whole history.
- Test on multiple ETFs to see whether any edge is SPY-specific.
- Add wavelet denoising or Kalman smoothing if the simple methods show enough promise.
"""
    output_path.write_text(notes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SPY denoising research experiment.")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--start", default="1993-01-29")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--cost-bps", type=float, default=1.0)
    parser.add_argument("--primary-method", default="fft_e95_w128")
    parser.add_argument("--touch-threshold", type=float, default=0.25)
    parser.add_argument("--prior-distance-threshold", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    prices = load_prices(args.ticker, args.start, refresh=args.refresh)
    specs = default_method_specs()
    primary_spec = get_method_spec(args.primary_method, specs)
    if primary_spec.family != "fft":
        raise ValueError("--primary-method must be a retained FFT method, for example fft_e95_w128")

    summary, equity_curves = run_parameter_sweep(prices, specs, cost_bps=args.cost_bps)
    best_row = best_research_method(summary)

    summary.sort_values(["family", "sharpe"], ascending=[True, False]).to_csv(RESULTS_DIR / "parameter_sweep.csv", index=False)
    summary.to_csv(RESULTS_DIR / "backtest_summary.csv", index=False)
    equity_curves.to_csv(RESULTS_DIR / "equity_curves.csv")

    log_price = np.log(prices)
    primary_smoothed = compute_smoothed_log_price(log_price, primary_spec)
    fft_component, residual = rolling_fft_components(
        log_price,
        window=int(primary_spec.params["window"]),
        retained_energy=primary_spec.params.get("retained_energy"),
        top_k=primary_spec.params.get("top_k"),
    )

    touch_config = TouchConfig(
        touch_threshold=args.touch_threshold,
        prior_distance_threshold=args.prior_distance_threshold,
    )
    touch_events = find_touch_events(prices, fft_component, residual, config=touch_config)
    touch_summary = summarize_touch_events(prices, touch_events, horizons=touch_config.horizons)
    touch_accuracy = accuracy_over_time(
        touch_events,
        horizon=touch_config.accuracy_horizon,
        rolling_event_window=touch_config.rolling_event_window,
    )
    touch_events.to_csv(RESULTS_DIR / "fft_touch_events.csv", index=False)
    touch_summary.to_csv(RESULTS_DIR / "fft_support_resistance_summary.csv", index=False)
    touch_accuracy.to_csv(RESULTS_DIR / "fft_accuracy_over_time.csv", index=False)

    plot_denoised_series(prices, primary_smoothed, args.primary_method, RESULTS_DIR / "denoised_series.png")
    plot_denoised_series_html(prices, primary_smoothed, args.primary_method, RESULTS_DIR / "denoised_series.html")
    plot_fft_component_split(prices, RESULTS_DIR / "fft_component_split.png")
    plot_fft_component_split_html(prices, RESULTS_DIR / "fft_component_split.html")
    plot_equity_curves(
        equity_curves,
        summary,
        prices,
        primary_smoothed,
        args.primary_method,
        RESULTS_DIR / "equity_curves.png",
    )
    plot_equity_curves_html(
        equity_curves,
        summary,
        prices,
        primary_smoothed,
        args.primary_method,
        RESULTS_DIR / "equity_curves.html",
    )
    plot_fft_support_resistance(prices, fft_component, residual, touch_events, RESULTS_DIR / "fft_support_resistance.png")
    plot_fft_support_resistance_html(
        prices,
        fft_component,
        residual,
        touch_events,
        RESULTS_DIR / "fft_support_resistance.html",
    )
    plot_fft_accuracy_over_time(touch_accuracy, RESULTS_DIR / "fft_accuracy_over_time.png")
    plot_fft_accuracy_over_time_html(touch_accuracy, RESULTS_DIR / "fft_accuracy_over_time.html")
    write_research_notes(
        summary,
        best_row,
        touch_summary,
        touch_accuracy,
        RESULTS_DIR / "research_notes.md",
        args.ticker,
        args.start,
    )

    print(f"Wrote results to {RESULTS_DIR}")
    print(f"Best method: {best_row['name']} Sharpe={best_row['sharpe']:.3f}")


if __name__ == "__main__":
    main()
