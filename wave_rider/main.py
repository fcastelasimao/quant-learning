from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import config
from backtest import BacktestResult, format_stats_table, run_backtest
from data_loader import load_data
from portfolio import Portfolio
from plotting import plot_backtest_overview, plot_strategy_state
from research_io import archive_selected_outputs, save_run_metadata, timestamp_label
from strategy import (
    AGGRESSIVE_PARAMETERS,
    DEFENSIVE_PARAMETERS,
    build_strategy_callback,
    warmup_bars,
)


def _load_all_weather_proxy_weights() -> dict[str, float]:
    strategies_path = Path("All_weather_portfolio/strategies.json")
    if not strategies_path.exists():
        return {
            "US_LC": 0.134,
            "US_TECH": 0.103,
            "LONG_BOND": 0.175,
            "INFL_LINKED": 0.348,
            "GOLD": 0.142,
            "COMMODITIES": 0.098,
        }

    data = json.loads(strategies_path.read_text())
    allocation = data["strategies"]["6asset_tip_gsg_rpavg"]["allocation"]
    ticker_to_asset = {
        "SPY": "US_LC",
        "QQQ": "US_TECH",
        "TLT": "LONG_BOND",
        "TIP": "INFL_LINKED",
        "GLD": "GOLD",
        "GSG": "COMMODITIES",
    }
    return {
        ticker_to_asset[ticker]: float(weight)
        for ticker, weight in allocation.items()
        if ticker in ticker_to_asset
    }


def _build_all_weather_proxy(prices: pd.DataFrame, history_index: pd.Index) -> pd.DataFrame:
    weights = _load_all_weather_proxy_weights()
    required_assets = [asset for asset in weights if asset in prices.columns]
    if len(required_assets) < 5:
        return pd.DataFrame(index=history_index)

    portfolio = Portfolio(config.INITIAL_CAPITAL)
    target_weights = {asset: weights[asset] for asset in required_assets}
    portfolio.rebalance(target_weights, transaction_cost_pct=config.TRANSACTION_COST_PCT)

    returns = prices[required_assets].sort_index().pct_change(fill_method=None).dropna(how="all")
    month_period = returns.index.to_period("M")

    records: list[dict[str, object]] = []
    for i, (date, returns_row) in enumerate(returns.iterrows()):
        portfolio.apply_returns(returns_row)
        records.append({"date": date, "All Weather Proxy Value": portfolio.value})

        is_month_end = i == len(returns) - 1 or month_period[i + 1] != month_period[i]
        if is_month_end:
            portfolio.rebalance(target_weights, transaction_cost_pct=config.TRANSACTION_COST_PCT)

    proxy_history = pd.DataFrame(records).set_index("date")
    return proxy_history.reindex(history_index).ffill()


def _combine_results(
    primary: BacktestResult,
    secondary: BacktestResult,
    overlay_history: pd.DataFrame | None = None,
) -> BacktestResult:
    history = primary.history.join(secondary.history, how="left")
    if overlay_history is not None and not overlay_history.empty:
        history = history.join(overlay_history, how="left")

    signal_log = primary.signal_log.add_prefix("aggressive_")
    if not secondary.signal_log.empty:
        signal_log = signal_log.join(secondary.signal_log.add_prefix("defensive_"), how="outer")

    stats = dict(primary.stats)
    stats.update(secondary.stats)
    return BacktestResult(history=history, signal_log=signal_log, stats=stats)


def save_results(
    result: BacktestResult,
    aggressive_result: BacktestResult,
    defensive_result: BacktestResult,
    results_dir: Path,
    run_label: str,
) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)

    result.history.to_csv(results_dir / "backtest_history.csv")
    if not result.signal_log.empty:
        result.signal_log.to_csv(results_dir / "signal_log.csv")
    if not aggressive_result.signal_log.empty:
        aggressive_result.signal_log.to_csv(results_dir / "signal_log_aggressive.csv")
    if not defensive_result.signal_log.empty:
        defensive_result.signal_log.to_csv(results_dir / "signal_log_defensive.csv")

    stats_df = pd.DataFrame([vars(stat) for stat in result.stats.values()])
    stats_df.to_csv(results_dir / "stats.csv", index=False)
    (results_dir / "stats_table.txt").write_text(format_stats_table(result.stats) + "\n")

    plot_backtest_overview(result.history, results_dir / "backtest_overview.png")
    plot_strategy_state(
        result.history,
        aggressive_result.signal_log,
        strategy_name="Wave Rider Aggressive",
        output_path=results_dir / "strategy_state_aggressive.png",
    )
    plot_strategy_state(
        result.history,
        defensive_result.signal_log,
        strategy_name="Wave Rider Defensive",
        output_path=results_dir / "strategy_state_defensive.png",
    )

    run_dir = config.RUNS_DIR / f"{timestamp_label()}_{run_label}"
    archive_selected_outputs(
        results_dir,
        run_dir,
        [
            "backtest_history.csv",
            "signal_log.csv",
            "signal_log_aggressive.csv",
            "signal_log_defensive.csv",
            "stats.csv",
            "stats_table.txt",
            "backtest_overview.png",
            "strategy_state_aggressive.png",
            "strategy_state_defensive.png",
        ],
    )
    save_run_metadata(
        run_dir / "run_metadata.json",
        {
            "run_type": run_label,
            "universe_preset": config.UNIVERSE_PRESET,
            "start_date": config.START_DATE,
            "transaction_cost_pct": config.TRANSACTION_COST_PCT,
            "aggressive_parameters": vars(AGGRESSIVE_PARAMETERS),
            "defensive_parameters": vars(DEFENSIVE_PARAMETERS),
            "all_weather_overlay_source": "full-history monthly all-weather proxy built from wave_rider prices",
        },
    )
    return run_dir


def run(refresh_data: bool = False) -> BacktestResult:
    prices = load_data(
        config.ASSET_TO_TICKER,
        start=config.START_DATE,
        cache_file=config.CACHE_FILE,
        refresh=refresh_data,
    )

    aggressive_result = run_backtest(
        prices,
        build_strategy_callback(AGGRESSIVE_PARAMETERS),
        strategy_name="Wave Rider Aggressive",
        initial_capital=config.INITIAL_CAPITAL,
        rebalance_frequency=config.REBALANCE_FREQUENCY,
        warmup_bars=warmup_bars(AGGRESSIVE_PARAMETERS),
        transaction_cost_pct=config.TRANSACTION_COST_PCT,
        rebalance_threshold=config.NO_TRADE_BAND,
        benchmark_weights=config.BENCHMARKS,
    )

    defensive_result = run_backtest(
        prices,
        build_strategy_callback(DEFENSIVE_PARAMETERS),
        strategy_name="Wave Rider Defensive",
        initial_capital=config.INITIAL_CAPITAL,
        rebalance_frequency=config.REBALANCE_FREQUENCY,
        warmup_bars=warmup_bars(DEFENSIVE_PARAMETERS),
        transaction_cost_pct=config.TRANSACTION_COST_PCT,
        rebalance_threshold=config.NO_TRADE_BAND,
        benchmark_weights=None,
    )

    all_weather_proxy = _build_all_weather_proxy(prices, aggressive_result.history.index)
    result = _combine_results(aggressive_result, defensive_result, all_weather_proxy)
    run_dir = save_results(
        result,
        aggressive_result,
        defensive_result,
        config.RESULTS_DIR,
        run_label="profiles",
    )

    print(f"Universe preset: {config.UNIVERSE_PRESET}")
    print(format_stats_table(result.stats))
    if not aggressive_result.signal_log.empty:
        print("\nLatest aggressive signal snapshot:")
        print(aggressive_result.signal_log.tail(1).T)
    if not defensive_result.signal_log.empty:
        print("\nLatest defensive signal snapshot:")
        print(defensive_result.signal_log.tail(1).T)
    print(f"\nSaved latest results to: {config.RESULTS_DIR}")
    print(f"Archived run to: {run_dir}")

    return result


if __name__ == "__main__":
    run()
