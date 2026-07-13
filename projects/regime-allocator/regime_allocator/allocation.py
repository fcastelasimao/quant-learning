import numpy as np


EQUITY_TICKERS = {"SPY", "EFA", "EEM", "VNQ"}
DEFENSIVE_TICKERS = {"TLT", "IEF", "TIP", "GLD"}
COMMODITY_TICKERS = {"DBC"}


def momentum_weights(monthly_prices, tickers, lookback=12, eligible=None):
    """
    Allocate proportionally to trailing momentum among eligible assets.
    Assets with negative momentum get zero weight.
    If all negative, equal-weight the eligible set.
    """
    if eligible is None:
        eligible = set(tickers)

    if len(monthly_prices) < lookback + 1:
        w = np.array([1.0 / len(tickers) if t in eligible else 0 for t in tickers])
        return w / w.sum() if w.sum() > 0 else np.ones(len(tickers)) / len(tickers)

    mom = (monthly_prices.iloc[-1] / monthly_prices.iloc[-lookback - 1] - 1)

    weights = np.zeros(len(tickers))
    for i, t in enumerate(tickers):
        if t in eligible and mom[t] > 0:
            weights[i] = mom[t]

    if weights.sum() == 0:
        for i, t in enumerate(tickers):
            if t in eligible:
                weights[i] = 1.0

    weights /= weights.sum()
    return weights


def regime_allocation(monthly_prices, tickers, regime_probs, risk_on_idx, lookback=12):
    """
    Blend risk-on and risk-off allocations using regime probabilities.

    Risk-on:  momentum-weighted across ALL assets.
    Risk-off: momentum-weighted across DEFENSIVE assets only.
    """
    all_set = set(tickers)
    defensive_set = (DEFENSIVE_TICKERS | COMMODITY_TICKERS) & all_set
    if not defensive_set:
        defensive_set = all_set

    w_risk_on = momentum_weights(monthly_prices, tickers, lookback, eligible=all_set)
    w_risk_off = momentum_weights(monthly_prices, tickers, lookback, eligible=defensive_set)

    risk_on_prob = regime_probs[risk_on_idx]
    blended = risk_on_prob * w_risk_on + (1 - risk_on_prob) * w_risk_off
    blended /= blended.sum()
    return blended


def apply_max_position(weights, max_pos):
    """Iteratively cap positions and redistribute excess. Cash-implicit if all at cap."""
    for _ in range(10):
        capped = weights > max_pos
        if not capped.any():
            break
        excess = weights[capped].sum() - capped.sum() * max_pos
        weights[capped] = max_pos
        uncapped = ~capped & (weights > 0)
        if uncapped.any():
            scale = (weights[uncapped].sum() + excess) / weights[uncapped].sum()
            weights[uncapped] *= scale
        else:
            break
    return weights


def apply_turnover_buffer(current_weights, target_weights, buffer):
    """Only rebalance positions that deviate from target by more than buffer."""
    if current_weights is None:
        return target_weights
    diff = target_weights - current_weights
    mask = np.abs(diff) > buffer
    if not mask.any():
        return current_weights / current_weights.sum()
    result = current_weights.copy()
    result[mask] = target_weights[mask]
    total = result.sum()
    if total > 0:
        result /= total
    return result


def apply_drawdown_guard(weights, drawdown, threshold):
    """Cut risk in half if drawdown exceeds threshold."""
    if drawdown > threshold:
        return weights * 0.5, True
    return weights, False
