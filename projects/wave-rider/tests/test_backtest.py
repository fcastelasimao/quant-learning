from __future__ import annotations

import pandas as pd

from backtest import run_backtest
from portfolio import Portfolio


def make_prices() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=6)
    return pd.DataFrame(
        {
            "A": [100.0, 110.0, 121.0, 133.1, 146.41, 161.051],
            "B": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        },
        index=dates,
    )


def always_long_a(_: pd.DataFrame) -> tuple[dict[str, float], dict[str, object]]:
    return {"A": 1.0}, {"signal": "always_long_a"}


def test_portfolio_compounds_between_rebalances():
    portfolio = Portfolio(100.0)
    portfolio.rebalance({"A": 1.0})

    portfolio.apply_returns(pd.Series({"A": 0.10}))
    portfolio.apply_returns(pd.Series({"A": 0.10}))

    assert round(portfolio.value, 2) == 121.00


def test_rebalance_keeps_residual_cash():
    portfolio = Portfolio(100.0)
    portfolio.rebalance({"A": 0.40})

    assert round(portfolio.positions["A"], 2) == 40.00
    assert round(portfolio.cash, 2) == 60.00


def test_transaction_costs_reduce_investable_value():
    portfolio = Portfolio(100.0)
    rebalance = portfolio.rebalance({"A": 1.0}, transaction_cost_pct=0.01)

    assert round(rebalance.turnover_value, 2) == 100.00
    assert round(rebalance.transaction_cost, 2) == 1.00
    assert round(portfolio.value, 2) == 99.00


def test_backtest_returns_history_stats_and_signal_log():
    prices = make_prices()

    result = run_backtest(
        prices,
        always_long_a,
        initial_capital=100.0,
        rebalance_frequency=1,
        warmup_bars=1,
        benchmark_weights={"Static_A": {"A": 1.0}},
    )

    assert not result.history.empty
    assert "Wave Rider Value" in result.history.columns
    assert "Static_A Value" in result.history.columns
    assert not result.signal_log.empty
    assert round(result.stats["Wave Rider"].final_value, 2) == 146.41
    assert round(result.stats["Static_A"].final_value, 2) == 146.41


def test_rebalance_threshold_skips_small_weight_changes():
    portfolio = Portfolio(100.0)
    portfolio.rebalance({"A": 1.0})
    portfolio.apply_returns(pd.Series({"A": 0.01}))

    rebalance = portfolio.rebalance({"A": 1.0}, rebalance_threshold=0.02)

    assert rebalance.executed is False
    assert round(rebalance.max_weight_change, 4) == 0.0
