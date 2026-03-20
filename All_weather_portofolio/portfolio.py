"""
portfolio.py
============
Manages your real portfolio -- the shares you actually own today.

Deliberately separate from backtest.py:
  - backtest.py  is a pure simulation (hypothetical, historical)
  - portfolio.py deals with your real holdings (actual shares, rebalancing now)

Contains:
  - load_holdings            read current holdings from JSON
  - save_holdings            persist holdings to JSON
  - initialise_holdings      create fresh holdings from an allocation
  - current_weights          compute current % weights from live prices
  - rebalancing_instructions compare current weights to targets, produce BUY/SELL
  - apply_rebalance          reset holdings to target weights

Notes on shares vs dollars
--------------------------
Holdings are stored as share counts, not dollar amounts. This is deliberate:
the dollar value of your holdings changes every day as prices move, but the
number of shares you own stays fixed until you trade. Storing shares is the
stable ground truth -- to get the current dollar value multiply shares * price.
If we stored dollar amounts we would need to re-fetch prices just to read the file.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import pandas as pd

import config


# ===========================================================================
# LOAD / SAVE
# ===========================================================================

def load_holdings() -> Optional[dict]:
    """
    Load current holdings from JSON file.
    Returns None if the file does not exist (first run).
    """
    if os.path.exists(config.HOLDINGS_FILE):
        with open(config.HOLDINGS_FILE) as f:
            return json.load(f)
    return None


def save_holdings(holdings: dict):
    """Persist holdings to JSON file, overwriting previous state."""
    with open(config.HOLDINGS_FILE, "w") as f:
        json.dump(holdings, f, indent=2)
    print(f"Holdings saved to {config.HOLDINGS_FILE}")


def initialise_holdings(prices_row: pd.Series,
                        allocation: dict,
                        portfolio_value: float) -> dict:
    """
    Create a fresh set of holdings by splitting portfolio_value
    according to allocation at the current prices.

    Returns
    -------
    dict like {"VTI": {"shares": 12.34, "last_price": 210.50}, ...}
    """
    holdings = {}
    for ticker, weight in allocation.items():
        dollar_amount    = portfolio_value * weight
        price            = float(prices_row[ticker])
        holdings[ticker] = {
            "shares":     round(dollar_amount / price, 6),
            "last_price": round(price, 4),
        }
    return holdings


# ===========================================================================
# WEIGHTS
# ===========================================================================

def current_weights(holdings: dict,
                    prices_row: pd.Series) -> tuple[dict, float]:
    """
    Compute current portfolio weights and total value given live prices.

    Returns
    -------
    weights : dict of {ticker: fraction}  (fractions sum to 1.0)
    total   : total portfolio value in dollars
    """
    values = {t: h["shares"] * float(prices_row[t]) for t, h in holdings.items()}
    total  = sum(values.values())
    return {t: v / total for t, v in values.items()}, total


# ===========================================================================
# REBALANCING
# ===========================================================================

def rebalancing_instructions(holdings: dict,
                             prices_row: pd.Series,
                             allocation: dict,
                             threshold: float) -> tuple[pd.DataFrame, float]:
    """
    Compare current weights to target allocation and produce buy/sell instructions.

    Only flags assets where drift exceeds `threshold` (e.g. 0.05 = 5%).
    Assets within threshold are marked HOLD.

    Parameters
    ----------
    holdings   : current holdings dict from load_holdings()
    prices_row : latest prices Series, one value per ticker
    allocation : target allocation dict {ticker: weight}
    threshold  : minimum drift fraction to trigger a BUY or SELL

    Returns
    -------
    instructions : pd.DataFrame with columns Ticker, Current Weight,
                   Target Weight, Drift (%), Action, $ Amount, etc.
    total_value  : current total portfolio value in dollars

    Notes on weights.get vs weights[ticker]
    ----------------------------------------
    weights.get(ticker, 0.0) returns 0.0 if the ticker is missing rather than
    raising a KeyError. This handles the edge case where a ticker exists in
    the allocation but not in holdings (e.g. a manually edited JSON file).
    The allocation-change detection in main prevents this in normal use, but
    defensive code is good practice.
    """
    weights, total_value = current_weights(holdings, prices_row)

    rows = []
    for ticker, target in allocation.items():
        current = weights.get(ticker, 0.0)
        drift   = current - target
        dollar  = drift * total_value       # positive = overweight -> sell
        action  = "HOLD"
        if abs(drift) > threshold:
            action = "SELL" if drift > 0 else "BUY"

        rows.append({
            "Ticker":         ticker,
            "Current Weight": round(current * 100, 2),
            "Target Weight":  round(target  * 100, 2),
            "Drift (%)":      round(drift   * 100, 2),
            "Action":         action,
            "$ Amount":       round(abs(dollar), 2),
            "Current Price":  round(float(prices_row[ticker]), 2),
            "Current Shares": round(holdings.get(ticker, {}).get("shares", 0.0), 4),
        })

    return pd.DataFrame(rows), total_value


def apply_rebalance(holdings: dict,
                    prices_row: pd.Series,
                    allocation: dict,
                    portfolio_value: float) -> dict:
    """
    Reset each holding to its target dollar weight at the current prices.

    Calculates target_dollar = portfolio_value * weight, then converts to
    shares by dividing by the current price. Shares are stored (not dollars)
    because shares stay fixed between trades while dollar values fluctuate.
    """
    for ticker, target in allocation.items():
        price            = float(prices_row[ticker])
        holdings[ticker] = {
            "shares":     round((portfolio_value * target) / price, 6),
            "last_price": round(price, 4),
        }
    return holdings
