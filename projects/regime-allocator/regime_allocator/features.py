import numpy as np
import pandas as pd


def compute_hmm_features(prices, config):
    """
    Extract 3 regime-detection features from daily prices.
    Fewer features = fewer HMM parameters = more robust with limited data.

    Features:
      1. eq_return    — SPY 21-day cumulative return (equity trend)
      2. eq_vol       — SPY 21-day realized vol, annualized (risk level)
      3. eq_bond_corr — SPY vs TLT 63-day rolling correlation (diversification regime)
    """
    rets = prices.pct_change()

    eq_ret = rets["SPY"].rolling(config.vol_lookback).sum()
    eq_vol = rets["SPY"].rolling(config.vol_lookback).std() * np.sqrt(252)
    eq_bond_corr = rets["SPY"].rolling(config.corr_lookback).corr(rets["TLT"])

    features = pd.DataFrame(
        {
            "eq_return": eq_ret,
            "eq_vol": eq_vol,
            "eq_bond_corr": eq_bond_corr,
        },
        index=prices.index,
    )
    return features.dropna()


def standardize_expanding(features):
    """
    Expanding-window z-score standardization.
    At each point, uses only mean/std from data up to (including) that point.
    No lookahead.
    """
    expanding_mean = features.expanding(min_periods=20).mean()
    expanding_std = features.expanding(min_periods=20).std()
    standardized = (features - expanding_mean) / expanding_std.replace(0, 1)
    return standardized.dropna()


def compute_allocation_features(monthly_returns, states, n_regimes):
    """
    For each regime, compute annualized mean return vector and covariance
    from historical monthly returns, with Ledoit-Wolf-style shrinkage.
    """
    n_assets = monthly_returns.shape[1]
    regime_stats = []

    for s in range(n_regimes):
        mask = states == s
        if mask.sum() < 6:
            regime_stats.append(
                (np.ones(n_assets) * 0.04, np.eye(n_assets) * 0.04)
            )
            continue

        regime_rets = monthly_returns[mask]
        raw_mu = regime_rets.mean().values * 12
        raw_cov = regime_rets.cov().values * 12

        # Shrink mean toward cross-sectional average (James-Stein-like)
        grand_mean = raw_mu.mean()
        shrinkage_mu = 0.3
        mu = (1 - shrinkage_mu) * raw_mu + shrinkage_mu * grand_mean

        # Shrink covariance toward diagonal (Ledoit-Wolf lite)
        diag_target = np.diag(np.diag(raw_cov))
        shrinkage_cov = max(0.2, min(0.8, 6.0 / mask.sum()))
        cov = (1 - shrinkage_cov) * raw_cov + shrinkage_cov * diag_target
        cov += np.eye(n_assets) * 1e-6

        regime_stats.append((mu, cov))

    return regime_stats
