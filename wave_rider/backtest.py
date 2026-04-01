from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

import config
from portfolio import Portfolio


StrategyCallback = Callable[[pd.DataFrame], tuple[dict[str, float], dict[str, object]]]


@dataclass
class StrategyStats:
    name: str
    final_value: float
    total_return_pct: float
    cagr_pct: float
    max_drawdown_pct: float
    sharpe: float
    sortino: float
    calmar: float
    ulcer_index: float
    avg_turnover_pct: float
    total_cost_pct: float
    avg_cash_weight_pct: float
    percent_days_invested: float


@dataclass
class BacktestResult:
    history: pd.DataFrame
    signal_log: pd.DataFrame
    stats: dict[str, StrategyStats]


def compute_cagr(value_series: pd.Series, periods_per_year: int = config.TRADING_DAYS_PER_YEAR) -> float:
    clean = value_series.dropna()
    if len(clean) < 2 or clean.iloc[0] <= 0:
        return 0.0
    years = (len(clean) - 1) / periods_per_year
    if years <= 0:
        return 0.0
    return ((clean.iloc[-1] / clean.iloc[0]) ** (1 / years) - 1) * 100


def compute_max_drawdown(value_series: pd.Series) -> float:
    clean = value_series.dropna()
    if clean.empty:
        return 0.0
    peak = clean.cummax()
    return ((clean - peak) / peak).min() * 100


def compute_sharpe(
    return_series: pd.Series,
    rf_annual: float = config.RISK_FREE_RATE,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> float:
    clean = return_series.dropna()
    if clean.empty or clean.std() < 1e-10:
        return 0.0
    rf_period = (1 + rf_annual) ** (1 / periods_per_year) - 1
    return float(((clean.mean() - rf_period) / clean.std()) * np.sqrt(periods_per_year))


def compute_sortino(
    return_series: pd.Series,
    rf_annual: float = config.RISK_FREE_RATE,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> float:
    clean = return_series.dropna()
    if clean.empty:
        return 0.0
    rf_period = (1 + rf_annual) ** (1 / periods_per_year) - 1
    downside = clean[clean < rf_period]
    if downside.empty or downside.std() < 1e-10:
        return 0.0
    return float(((clean.mean() - rf_period) / downside.std()) * np.sqrt(periods_per_year))


def compute_calmar(cagr_pct: float, max_drawdown_pct: float) -> float:
    if max_drawdown_pct == 0.0:
        return 0.0
    return cagr_pct / abs(max_drawdown_pct)


def compute_ulcer_index(value_series: pd.Series) -> float:
    clean = value_series.dropna()
    if clean.empty:
        return 0.0
    peak = clean.cummax()
    drawdown_pct = ((clean - peak) / peak) * 100
    return float(np.sqrt((drawdown_pct ** 2).mean()))


def build_stats(
    name: str,
    history: pd.DataFrame,
    initial_capital: float,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> StrategyStats:
    values = history[f"{name} Value"]
    returns = history[f"{name} Return"]
    turnover = history[f"{name} TurnoverPct"]
    costs = history[f"{name} TransactionCost"]
    cash_weight = history[f"{name} CashWeight"]
    exposure = history[f"{name} Exposure"]

    starting_value = float(initial_capital)
    ending_value = float(values.iloc[-1])
    total_return_pct = 0.0
    if starting_value > 0:
        total_return_pct = ((ending_value / starting_value) - 1) * 100

    cagr_series = pd.concat([pd.Series([starting_value]), values.reset_index(drop=True)], ignore_index=True)
    cagr_pct = compute_cagr(cagr_series, periods_per_year)
    max_drawdown_pct = float(compute_max_drawdown(values))
    sharpe = compute_sharpe(returns, periods_per_year=periods_per_year)
    sortino = compute_sortino(returns, periods_per_year=periods_per_year)
    calmar = compute_calmar(cagr_pct, max_drawdown_pct)
    ulcer_index = compute_ulcer_index(values)

    return StrategyStats(
        name=name,
        final_value=ending_value,
        total_return_pct=total_return_pct,
        cagr_pct=cagr_pct,
        max_drawdown_pct=max_drawdown_pct,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        ulcer_index=ulcer_index,
        avg_turnover_pct=float(turnover.mean() * 100),
        total_cost_pct=float(costs.sum() / starting_value * 100) if starting_value > 0 else 0.0,
        avg_cash_weight_pct=float(cash_weight.mean() * 100),
        percent_days_invested=float((exposure > 0).mean() * 100),
    )


def _run_static_benchmark(
    name: str,
    returns: pd.DataFrame,
    initial_capital: float,
    weights: dict[str, float],
    transaction_cost_pct: float,
) -> pd.DataFrame:
    portfolio = Portfolio(initial_capital)
    portfolio.rebalance(weights, transaction_cost_pct=transaction_cost_pct)

    records = []
    previous_value = initial_capital
    for date, returns_row in returns.iterrows():
        portfolio.apply_returns(returns_row)
        current_value = portfolio.value
        records.append(
            {
                "date": date,
                f"{name} Value": current_value,
                f"{name} Return": (current_value / previous_value) - 1 if previous_value > 0 else 0.0,
                f"{name} CashWeight": portfolio.cash / current_value if current_value > 0 else 0.0,
                f"{name} Exposure": portfolio.exposure(),
                f"{name} TurnoverPct": 0.0,
                f"{name} TransactionCost": 0.0,
            }
        )
        previous_value = current_value

    benchmark_history = pd.DataFrame(records).set_index("date")
    if not benchmark_history.empty:
        benchmark_history.iloc[0, benchmark_history.columns.get_loc(f"{name} TransactionCost")] = portfolio.total_costs
        benchmark_history.iloc[0, benchmark_history.columns.get_loc(f"{name} TurnoverPct")] = (
            portfolio.total_turnover / initial_capital if initial_capital > 0 else 0.0
        )
    return benchmark_history


def run_backtest(
    prices: pd.DataFrame,
    strategy_callback: StrategyCallback,
    *,
    strategy_name: str = "Wave Rider",
    initial_capital: float,
    rebalance_frequency: int,
    warmup_bars: int,
    transaction_cost_pct: float = 0.0,
    rebalance_threshold: float = 0.0,
    benchmark_weights: dict[str, dict[str, float]] | None = None,
) -> BacktestResult:
    returns = prices.sort_index().pct_change(fill_method=None).dropna(how="all")
    if len(returns) <= warmup_bars:
        raise ValueError("Not enough data after warmup to run the backtest.")

    portfolio = Portfolio(initial_capital)
    strategy_records: list[dict[str, object]] = []
    signal_records: list[dict[str, object]] = []

    previous_value = initial_capital
    for i in range(warmup_bars, len(returns)):
        date = returns.index[i]
        returns_row = returns.iloc[i]
        turnover_pct = 0.0
        transaction_cost = 0.0

        if (i - warmup_bars) % rebalance_frequency == 0:
            lookback_returns = returns.iloc[:i]
            target_weights, diagnostics = strategy_callback(lookback_returns)
            rebalance = portfolio.rebalance(
                target_weights,
                transaction_cost_pct=transaction_cost_pct,
                rebalance_threshold=rebalance_threshold,
            )
            current_value = portfolio.value
            turnover_pct = rebalance.turnover_value / current_value if current_value > 0 else 0.0
            transaction_cost = rebalance.transaction_cost

            signal_record = {"date": date}
            for key, value in diagnostics.items():
                signal_record[key] = value
            signal_record["rebalance_executed"] = rebalance.executed
            signal_record["max_weight_change"] = rebalance.max_weight_change
            signal_record["rebalance_threshold"] = rebalance_threshold
            for asset, weight in sorted(target_weights.items()):
                signal_record[f"target_{asset.lower()}"] = weight
            signal_records.append(signal_record)

        portfolio.apply_returns(returns_row)
        current_value = portfolio.value
        strategy_records.append(
            {
                "date": date,
                f"{strategy_name} Value": current_value,
                f"{strategy_name} Return": (current_value / previous_value) - 1 if previous_value > 0 else 0.0,
                f"{strategy_name} CashWeight": portfolio.cash / current_value if current_value > 0 else 0.0,
                f"{strategy_name} Exposure": portfolio.exposure(),
                f"{strategy_name} Drawdown": portfolio.drawdown(),
                f"{strategy_name} TurnoverPct": turnover_pct,
                f"{strategy_name} TransactionCost": transaction_cost,
            }
        )
        previous_value = current_value

    history = pd.DataFrame(strategy_records).set_index("date")
    stats = {strategy_name: build_stats(strategy_name, history, initial_capital)}

    if benchmark_weights:
        benchmark_returns = returns.iloc[warmup_bars:]
        for name, weights in benchmark_weights.items():
            benchmark_history = _run_static_benchmark(
                name=name,
                returns=benchmark_returns,
                initial_capital=initial_capital,
                weights=weights,
                transaction_cost_pct=transaction_cost_pct,
            )
            history = history.join(benchmark_history, how="left")
            stats[name] = build_stats(name, history, initial_capital)

    signal_log = pd.DataFrame(signal_records).set_index("date") if signal_records else pd.DataFrame()
    return BacktestResult(history=history, signal_log=signal_log, stats=stats)


def format_stats_table(stats: dict[str, StrategyStats]) -> str:
    columns = [
        "Name",
        "Final",
        "Total Return %",
        "CAGR %",
        "Max DD %",
        "Sharpe",
        "Sortino",
        "Calmar",
        "Ulcer",
        "Avg Turnover %",
        "Total Cost %",
        "Avg Cash %",
        "Days Invested %",
    ]
    rows = [" | ".join(columns)]
    rows.append(" | ".join(["---"] * len(columns)))

    for stat in stats.values():
        rows.append(
            " | ".join(
                [
                    stat.name,
                    f"{stat.final_value:.2f}",
                    f"{stat.total_return_pct:.2f}",
                    f"{stat.cagr_pct:.2f}",
                    f"{stat.max_drawdown_pct:.2f}",
                    f"{stat.sharpe:.2f}",
                    f"{stat.sortino:.2f}",
                    f"{stat.calmar:.2f}",
                    f"{stat.ulcer_index:.2f}",
                    f"{stat.avg_turnover_pct:.2f}",
                    f"{stat.total_cost_pct:.2f}",
                    f"{stat.avg_cash_weight_pct:.2f}",
                    f"{stat.percent_days_invested:.2f}",
                ]
            )
        )
    return "\n".join(rows)
