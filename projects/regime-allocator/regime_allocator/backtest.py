import warnings

import numpy as np
import pandas as pd

from .regimes import fit_hmm, get_regime_probabilities, predict_states, label_regimes
from .features import standardize_expanding
from .allocation import (
    regime_allocation,
    apply_max_position,
    apply_turnover_buffer,
    apply_drawdown_guard,
)


def run_walkforward(daily_prices, daily_features, config):
    """
    Walk-forward backtest. HMM trains on daily features, allocation monthly.

    At each month-end:
      1. Standardize daily features up to today (expanding window z-score)
      2. Train HMM on standardized daily features (multiple restarts)
      3. Get filtered regime probabilities
      4. Compute momentum-based allocation, tilted by regime signal
      5. Apply position limits, turnover buffer, drawdown guard
      6. Earn next month's return, net of transaction costs
    """
    monthly_prices = daily_prices.resample("M").last().dropna()
    monthly_returns = monthly_prices.pct_change().dropna()
    tickers = monthly_prices.columns.tolist()

    rebalance_dates = monthly_returns.index
    daily_dates = daily_features.index

    records = []
    portfolio_value = 1.0
    peak_value = 1.0
    current_weights = None

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Model is not converging")

        for i in range(len(rebalance_dates) - 1):
            date = rebalance_dates[i]
            next_date = rebalance_dates[i + 1]

            daily_mask = daily_dates <= date
            if daily_mask.sum() < config.min_train_days:
                continue

            train_daily = daily_features.loc[daily_mask]
            std_daily = standardize_expanding(train_daily)
            if len(std_daily) < config.min_train_days:
                continue

            model = fit_hmm(std_daily.values, config.n_regimes)
            regime_probs = get_regime_probabilities(model, std_daily.values)

            labels = label_regimes(model)
            risk_on_states = [s for s, l in labels.items() if l in ("risk_on", "growth")]
            if not risk_on_states:
                risk_on_idx = np.argmax(model.means_[:, 0])
            else:
                risk_on_idx = risk_on_states[0]

            monthly_mask = monthly_prices.index <= date
            prices_up_to_now = monthly_prices.loc[monthly_mask]

            target = regime_allocation(
                prices_up_to_now, tickers, regime_probs, risk_on_idx, lookback=12
            )
            target = apply_max_position(target, config.max_position)
            target = apply_turnover_buffer(current_weights, target, config.turnover_buffer)

            drawdown = (peak_value - portfolio_value) / peak_value if peak_value > 0 else 0
            target, dd_triggered = apply_drawdown_guard(
                target, drawdown, config.drawdown_cutback
            )

            if current_weights is not None:
                turnover = np.sum(np.abs(target - current_weights))
                cost = turnover * config.cost_bps / 10_000
            else:
                turnover = 0.0
                cost = 0.0

            current_weights = target.copy()

            asset_rets = monthly_returns.loc[next_date].values
            port_ret = np.dot(current_weights, asset_rets) - cost
            portfolio_value *= 1 + port_ret
            peak_value = max(peak_value, portfolio_value)

            records.append(
                {
                    "date": next_date,
                    "portfolio_value": portfolio_value,
                    "return": port_ret,
                    "turnover": turnover,
                    "cost": cost,
                    "dd_triggered": dd_triggered,
                    "drawdown": (peak_value - portfolio_value) / peak_value,
                    "risk_on_prob": regime_probs[risk_on_idx],
                    **{f"w_{t}": w for t, w in zip(tickers, current_weights)},
                    **{f"regime_p{s}": p for s, p in enumerate(regime_probs)},
                }
            )

    return pd.DataFrame(records).set_index("date")


def benchmark_buyhold(monthly_prices, ticker="SPY"):
    vals = monthly_prices[ticker] / monthly_prices[ticker].iloc[0]
    rets = monthly_prices[ticker].pct_change().dropna()
    return vals, rets


def benchmark_sixty_forty(monthly_prices, eq="SPY", bond="TLT"):
    rets = monthly_prices[[eq, bond]].pct_change().dropna()
    port_rets = 0.6 * rets[eq] + 0.4 * rets[bond]
    vals = (1 + port_rets).cumprod()
    return vals, port_rets


def benchmark_equal_weight(monthly_prices):
    rets = monthly_prices.pct_change().dropna()
    port_rets = rets.mean(axis=1)
    vals = (1 + port_rets).cumprod()
    return vals, port_rets


def benchmark_pure_momentum(monthly_prices, lookback=12, cost_bps=10):
    """
    Pure 12-month momentum: allocate proportionally to positive momentum
    across all assets, rebalanced monthly. No regime overlay.
    Isolates momentum alpha from regime alpha.
    """
    from .allocation import momentum_weights

    tickers = monthly_prices.columns.tolist()
    rets = monthly_prices.pct_change().dropna()
    records = []
    current_weights = None

    for i in range(lookback + 1, len(rets)):
        prices_so_far = monthly_prices.iloc[: i + 1]
        target = momentum_weights(prices_so_far, tickers, lookback)

        if current_weights is not None:
            turnover = np.sum(np.abs(target - current_weights))
            cost = turnover * cost_bps / 10_000
        else:
            turnover = 0.0
            cost = 0.0

        current_weights = target.copy()
        port_ret = np.dot(current_weights, rets.iloc[i].values) - cost
        records.append({"date": rets.index[i], "return": port_ret})

    df = pd.DataFrame(records).set_index("date")
    vals = (1 + df["return"]).cumprod()
    return vals, df["return"]
