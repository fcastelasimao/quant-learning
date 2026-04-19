"""
Transaction cost model.

Implements a simplified Almgren-Chriss model for estimating round-trip costs.
Net-of-cost IC is the primary metric in qframe — this module is non-optional.

=============================================================================
COST COMPONENTS — WHAT IS AND IS NOT MODELLED
=============================================================================

MODELLED (Phase 1):
  1. Half-spread (bid-ask)         : spread_bps / 2
  2. Market impact (Almgren-Chriss): gamma * (trade_size / ADV)^eta
  3. Short-borrow cost             : short_borrow_bps_annual / 252 per long-short
                                     portfolio day (approximated as 50% of the
                                     portfolio is short)
  4. Funding / leverage cost       : funding_cost_bps_annual / 252 per day for
                                     leveraged notional (applied to full AUM)

NOT MODELLED — must be considered before live deployment:
  ┌──────────────────────────────────┬──────────────────────────────────────────┐
  │ Gap                              │ Impact & recommendation                  │
  ├──────────────────────────────────┼──────────────────────────────────────────┤
  │ Position-size-dependent impact   │ PARTIALLY MODELLED: pass adv_df to      │
  │                                  │ net_ic() or WalkForwardValidator to      │
  │                                  │ enable per-stock ADV-weighted impact     │
  │                                  │ via compute_per_stock_impact_bps().      │
  │                                  │ Build adv_df with compute_per_stock_adv  │
  │                                  │ (prices, volume, window=20). Without     │
  │                                  │ adv_df, falls back to uniform fraction.  │
  ├──────────────────────────────────┼──────────────────────────────────────────┤
  │ Signal decay during execution    │ The harness uses EOD → EOD forward       │
  │                                  │ returns (1-day horizon), correctly        │
  │                                  │ assuming signals execute at next day's   │
  │                                  │ open. For intraday execution this is OK. │
  │                                  │ For MOC (market-on-close) execution the  │
  │                                  │ 1-day return is correct already.         │
  │                                  │ Risk: for a daily factor the alpha has   │
  │                                  │ typically decayed ~30% by the next open. │
  ├──────────────────────────────────┼──────────────────────────────────────────┤
  │ Short-sale restrictions (SSR)    │ On down-days, shorting at the bid is     │
  │                                  │ restricted. Not modelled. Effect is      │
  │                                  │ small for daily-rebalancing strategies.  │
  ├──────────────────────────────────┼──────────────────────────────────────────┤
  │ Hard-to-borrow names             │ Some stocks have borrow rates of         │
  │                                  │ 5–20 %/year. The fixed                   │
  │                                  │ short_borrow_bps_annual applies a        │
  │                                  │ uniform rate. Flag any short position    │
  │                                  │ > 2 % weight for borrow-rate check.      │
  ├──────────────────────────────────┼──────────────────────────────────────────┤
  │ Exchange fees & SEC fees         │ ~0.23 bps per notional sold (SEC) +      │
  │                                  │ ~0.30 bps exchange fee. Small; add 1 bps │
  │                                  │ to spread_bps to conservatively cover it.│
  ├──────────────────────────────────┼──────────────────────────────────────────┤
  │ Tax drag                         │ US short-term capital gains ~37 %.       │
  │                                  │ Model returns are pre-tax. Report        │
  │                                  │ post-tax separately at investor level.   │
  ├──────────────────────────────────┼──────────────────────────────────────────┤
  │ Slippage model calibration       │ gamma=30, eta=0.6 are Almgren et al.     │
  │                                  │ (2005) defaults. Verify against broker   │
  │                                  │ execution reports before going live.     │
  │                                  │ Reduce to gamma=15 for highly liquid     │
  │                                  │ large-caps (S&P 100 top 50 names).       │
  └──────────────────────────────────┴──────────────────────────────────────────┘

LIVE DEPLOYMENT CHECKLIST:
  □ Replace adv_fraction with per-stock ADV-weighted position sizing
  □ Obtain borrow rates for short positions from prime broker
  □ Confirm that execution is via VWAP/MOC (affects which forward-return
    the harness should use)
  □ Run shadow portfolio for 3 months before full live deployment
  □ Monitor realised slippage vs model slippage weekly

All costs are expressed in basis points (bps) of notional traded.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------

@dataclass
class CostParams:
    """
    Parameters for the full transaction cost model.

    Attributes:
        spread_bps:              Round-trip bid-ask spread in basis points.
                                 Default 10 bps ≈ liquid S&P 500 large-cap.
                                 Add ~1 bps to cover SEC + exchange fees.
        adv_fraction:            Fraction of ADV traded per rebalance (assumed
                                 uniform across all stocks and dates).
                                 Default 0.10 (10% of daily volume).
                                 ⚠ See NOT MODELLED above — use per-stock ADV
                                   sizing in live deployment.
        gamma:                   Market impact coefficient in basis points.
                                 Default 30 bps — calibrated so that trading
                                 10% ADV costs ~7.5 bps one-way market impact
                                 (total one-way ≈ 12.5 bps).
        eta:                     Market impact exponent. Default 0.6 (concave
                                 impact, per Almgren et al. 2005).
        short_borrow_bps_annual: Annual cost to borrow shares for short selling.
                                 Default 50 bps/year (0.05 %) = typical S&P 500
                                 easy-to-borrow rate from prime brokers.
                                 Translates to ~0.19 bps/day.
                                 ⚠ Hard-to-borrow names can be 500–2000 bps/year.
        funding_cost_bps_annual: Annual cost of leverage / margin financing.
                                 Default 0 bps (unlevered strategy).
                                 Set to Fed Funds + 50 bps for leveraged books
                                 (e.g. 550 bps in 2024 conditions).
    """
    spread_bps:              float = 10.0
    adv_fraction:            float = 0.10
    gamma:                   float = 30.0   # bps — market impact scale
    eta:                     float = 0.6
    short_borrow_bps_annual: float = 50.0   # bps/year for short positions
    funding_cost_bps_annual: float = 0.0    # bps/year for leverage

    def __post_init__(self) -> None:
        if self.spread_bps < 0:
            raise ValueError(f"spread_bps must be ≥ 0, got {self.spread_bps}")
        if self.adv_fraction <= 0:
            raise ValueError(f"adv_fraction must be > 0, got {self.adv_fraction}")
        if not (0 < self.eta <= 1):
            raise ValueError(f"eta must be in (0, 1], got {self.eta}")
        if self.gamma < 0:
            raise ValueError(f"gamma must be ≥ 0, got {self.gamma}")
        if self.short_borrow_bps_annual < 0:
            raise ValueError(f"short_borrow_bps_annual must be ≥ 0, got {self.short_borrow_bps_annual}")
        if self.funding_cost_bps_annual < 0:
            raise ValueError(f"funding_cost_bps_annual must be ≥ 0, got {self.funding_cost_bps_annual}")


DEFAULT_COST_PARAMS = CostParams()

# Aggressive (pessimistic) parameters for worst-case sensitivity check
AGGRESSIVE_COST_PARAMS = CostParams(
    spread_bps=20.0,               # wider spread (mid-cap / stressed conditions)
    adv_fraction=0.10,
    gamma=50.0,                    # higher market impact
    eta=0.6,
    short_borrow_bps_annual=150.0, # harder-to-borrow universe
    funding_cost_bps_annual=550.0, # Fed Funds 5% + 50bps prime-broker spread
)


# ---------------------------------------------------------------------------
# Single-trade cost
# ---------------------------------------------------------------------------

def estimate_cost_bps(
    adv_fraction: float | None = None,
    params: CostParams = DEFAULT_COST_PARAMS,
) -> float:
    """
    Estimate the one-way transaction cost in basis points for a single trade.

    cost = spread/2  +  gamma * (adv_fraction)^eta

    Args:
        adv_fraction: fraction of ADV being traded. If None, uses params.adv_fraction.
        params:       CostParams instance.

    Returns:
        One-way cost in basis points (excludes borrow and funding costs,
        which are time-based rather than per-trade).
    """
    if adv_fraction is None:
        adv_fraction = params.adv_fraction

    half_spread = params.spread_bps / 2.0
    # gamma is already in bps; impact scales concavely with trade size / ADV
    impact_bps = params.gamma * (adv_fraction ** params.eta)
    return half_spread + impact_bps


def round_trip_cost_bps(
    adv_fraction: float | None = None,
    params: CostParams = DEFAULT_COST_PARAMS,
) -> float:
    """
    Estimate the round-trip (buy + sell) transaction cost in basis points.

    Returns:
        Round-trip cost in basis points (spread + impact only; see net_ic for
        daily borrow and funding drag).
    """
    return 2.0 * estimate_cost_bps(adv_fraction=adv_fraction, params=params)


# ---------------------------------------------------------------------------
# Portfolio-level turnover
# ---------------------------------------------------------------------------

def compute_turnover(
    weights: pd.DataFrame,
) -> pd.Series:
    """
    Compute one-way portfolio turnover at each rebalance date.

    Turnover(t) = 0.5 * sum_i |w_i(t) - w_i(t-1)|

    This is the standard one-way turnover definition used in the factor
    literature (half of the sum of absolute weight changes).

    Args:
        weights: pd.DataFrame (dates x tickers) of portfolio weights.
                 NaN treated as 0 (stock not held).

    Returns:
        pd.Series of one-way turnover, indexed by date.
        First date is NaN (no previous weights to compare against).
    """
    w = weights.fillna(0.0)
    # diff() NaNs the first row; sum(skipna=True) would silently give 0 — force NaN instead
    delta = w.diff().abs().sum(axis=1) / 2.0
    delta.iloc[0] = np.nan
    delta.name = "turnover"
    return delta


# ---------------------------------------------------------------------------
# Short-side fraction of portfolio
# ---------------------------------------------------------------------------

def compute_short_fraction(weights: pd.DataFrame) -> pd.Series:
    """
    Compute the fraction of the portfolio that is short at each date.

    For a dollar-neutral long/short portfolio, this is approximately 0.5
    (equal long and short). For long-only factors it is 0.

    Args:
        weights: (dates x tickers) portfolio weights (signed).

    Returns:
        pd.Series of short fraction (0 to 1), indexed by date.
    """
    w = weights.fillna(0.0)
    short_notional = w.clip(upper=0).abs().sum(axis=1)
    gross_notional = w.abs().sum(axis=1)
    frac = short_notional / gross_notional.replace(0, np.nan)
    frac.name = "short_fraction"
    return frac


# ---------------------------------------------------------------------------
# Per-stock ADV helpers  (B7 upgrade)
# ---------------------------------------------------------------------------

def compute_per_stock_adv(
    prices: pd.DataFrame,
    volume: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    """
    Compute per-stock dollar average daily volume (ADV) using a rolling window.

    ADV_i(t) = rolling_mean( price_i × volume_i, window )

    Args:
        prices:  (dates × tickers) adjusted close prices.
        volume:  (dates × tickers) share volume (same shape and index as prices).
        window:  Rolling window in trading days (default 20 ≈ 1 month).

    Returns:
        pd.DataFrame (dates × tickers) of dollar ADV, aligned to prices.
        NaN where fewer than ``window // 2`` days of data exist.
    """
    dollar_vol = prices * volume
    return dollar_vol.rolling(window, min_periods=window // 2).mean()


def compute_per_stock_impact_bps(
    weight_delta: pd.DataFrame,
    adv_df: pd.DataFrame,
    portfolio_nav: float,
    params: CostParams = DEFAULT_COST_PARAMS,
) -> pd.Series:
    """
    Compute the per-stock market impact in basis points, summed to a portfolio
    level cost series.

    For each stock i on date t:
        adv_frac_i(t)  = |Δw_i(t)| × portfolio_nav / adv_i(t)
        impact_i(t)    = gamma × adv_frac_i(t)^eta    [bps one-way]

    Portfolio impact(t) = Σ_i |Δw_i(t)| × impact_i(t)   (weight-averaged)

    This replaces the uniform ``adv_fraction`` approximation in ``estimate_cost_bps``
    with a per-stock estimate that correctly penalises small-cap / illiquid names
    more than large-caps.

    Args:
        weight_delta:   (dates × tickers) weight changes (= w_t − w_{t−1}).
                        Use weights.diff() from the portfolio weights DataFrame.
        adv_df:         (dates × tickers) per-stock dollar ADV (from compute_per_stock_adv).
        portfolio_nav:  Portfolio NAV in dollars.
        params:         CostParams instance.

    Returns:
        pd.Series of portfolio-level one-way impact in bps, indexed by date.
        NaN on dates with zero turnover or missing ADV.
    """
    # Align shapes
    shared_idx = weight_delta.index.intersection(adv_df.index)
    shared_col = weight_delta.columns.intersection(adv_df.columns)
    dw  = weight_delta.loc[shared_idx, shared_col].abs()
    adv = adv_df.loc[shared_idx, shared_col].replace(0, np.nan)

    # Per-stock ADV fraction: notional traded / ADV
    adv_frac = (dw * portfolio_nav) / adv

    # Per-stock impact (bps)
    per_stock_impact = params.gamma * (adv_frac ** params.eta)  # bps one-way

    # Weight-average impact across the portfolio on each date
    total_dw = dw.sum(axis=1).replace(0, np.nan)
    impact_series = (per_stock_impact * dw).sum(axis=1) / total_dw
    impact_series.name = "impact_bps_per_stock"
    return impact_series


# ---------------------------------------------------------------------------
# Net-of-cost IC
# ---------------------------------------------------------------------------

def net_ic(
    gross_ic: pd.Series,
    turnover: pd.Series,
    params: CostParams = DEFAULT_COST_PARAMS,
    horizon: int = 1,
    weights: pd.DataFrame | None = None,
    adv_df: pd.DataFrame | None = None,
    portfolio_nav: float = 1e6,
) -> pd.Series:
    """
    Compute net-of-cost IC by subtracting all cost drags from gross IC.

    Cost components applied:
    1. Trading costs (spread + market impact): scaled by turnover and horizon
    2. Short-borrow cost: applied daily to the short fraction of the portfolio
    3. Funding cost: applied daily to the full AUM (if leverage is used)

    Args:
        gross_ic:       pd.Series of daily IC values.
        turnover:       pd.Series of daily one-way turnover (same index).
        params:         CostParams instance.
        horizon:        factor holding horizon in days (used to scale trading cost drag).
        weights:        (dates x tickers) portfolio weights. If provided, used to
                        compute the short fraction for borrow cost. If None, assumes
                        50% short (dollar-neutral portfolio).
        adv_df:         (dates x tickers) average daily volume in dollar terms. When
                        provided, computes per-stock ADV fraction instead of the fixed
                        scalar `params.adv_fraction`. Phase 3 Gate 0 upgrade.
                        ADV fraction = |weight × portfolio_nav| / adv_stock.
        portfolio_nav:  Portfolio net asset value in dollars, used with adv_df to
                        compute per-stock position sizes. Default 1M.

    Returns:
        pd.Series of net-of-cost IC, same index as gross_ic.

    Notes:
        The trading cost drag formula is an approximation:
            drag ≈ round_trip_cost_bps * turnover / horizon
        This works for IC series because IC ≈ (signal × return), and cost
        reduces the achievable return proportionally to turnover.

        The borrow and funding drags are expressed in IC units by dividing by
        the expected annualised IC. This is a rough conversion — for a more
        rigorous treatment, model the P&L directly.

        ⚠ For live trading: obtain realised slippage data from your broker
        and compare with estimate_cost_bps() to calibrate the model.
    """
    # --- 1. Trading cost drag ---
    # When adv_df is provided, use true per-stock impact (compute_per_stock_impact_bps)
    # instead of a fixed scalar adv_fraction.  This correctly penalises illiquid names.
    shared = gross_ic.index.intersection(turnover.index)
    g = gross_ic.loc[shared]
    t = turnover.loc[shared].fillna(0.0)

    if adv_df is not None and weights is not None:
        try:
            # Use the per-stock impact helper (includes spread separately below)
            w_delta = weights.fillna(0.0).diff()
            half_spread_frac = params.spread_bps / 2.0 / 10_000.0
            impact_series = compute_per_stock_impact_bps(
                w_delta, adv_df, portfolio_nav, params
            )
            # round-trip: 2× one-way impact + spread
            rt_cost_series = (
                2.0 * impact_series / 10_000.0
                + 2.0 * half_spread_frac
            ).reindex(shared)
            trading_drag = rt_cost_series.fillna(
                round_trip_cost_bps(params=params) / 10_000.0
            ) * t / max(horizon, 1)
        except Exception:
            rt_cost = round_trip_cost_bps(params=params) / 10_000.0
            trading_drag = rt_cost * t / max(horizon, 1)
    else:
        rt_cost = round_trip_cost_bps(params=params) / 10_000.0  # scalar fallback
        trading_drag = rt_cost * t / max(horizon, 1)

    # --- 2. Short-borrow drag ---
    # Default: 50% of portfolio is short (dollar-neutral assumption)
    borrow_daily_frac = params.short_borrow_bps_annual / 10_000.0 / 252.0
    if weights is not None:
        short_frac = compute_short_fraction(weights).reindex(shared).fillna(0.5)
    else:
        short_frac = pd.Series(0.5, index=shared)
    borrow_drag = borrow_daily_frac * short_frac

    # --- 3. Funding cost drag ---
    funding_daily_frac = params.funding_cost_bps_annual / 10_000.0 / 252.0
    funding_drag = pd.Series(funding_daily_frac, index=shared)

    net = g - trading_drag - borrow_drag - funding_drag
    net.name = "net_IC"
    return net


# ---------------------------------------------------------------------------
# Cost breakdown summary (useful for live deployment review)
# ---------------------------------------------------------------------------

def cost_summary(
    turnover_mean: float,
    horizon: int = 1,
    short_fraction: float = 0.5,
    params: CostParams = DEFAULT_COST_PARAMS,
) -> dict[str, float]:
    """
    Return a human-readable cost breakdown in bps/day and bps/year.

    Useful for pre-trade analysis and live-deployment cost budgeting.

    Args:
        turnover_mean:  Average daily one-way turnover (e.g. 0.05 = 5%).
        horizon:        Signal holding horizon in days.
        short_fraction: Fraction of portfolio that is short (default 0.5).
        params:         CostParams instance.

    Returns:
        dict with annual bps figures for each cost component.

    Example:
        >>> cost_summary(0.05, horizon=1)
        {'trading_bps_per_day': 1.25, 'borrow_bps_per_day': 0.10, ...}
    """
    rt_cost = round_trip_cost_bps(params=params)
    trading_daily = rt_cost * turnover_mean / max(horizon, 1)
    borrow_daily  = params.short_borrow_bps_annual / 252.0 * short_fraction
    funding_daily = params.funding_cost_bps_annual / 252.0

    return {
        "one_way_cost_bps":      estimate_cost_bps(params=params),
        "round_trip_cost_bps":   rt_cost,
        "trading_drag_bps_day":  trading_daily,
        "trading_drag_bps_year": trading_daily * 252,
        "borrow_drag_bps_day":   borrow_daily,
        "borrow_drag_bps_year":  borrow_daily * 252,
        "funding_drag_bps_day":  funding_daily,
        "funding_drag_bps_year": funding_daily * 252,
        "total_drag_bps_day":    trading_daily + borrow_daily + funding_daily,
        "total_drag_bps_year":   (trading_daily + borrow_daily + funding_daily) * 252,
    }
