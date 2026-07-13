"""SQQQ focus rule: skip trades in the RSI × ATR danger cell.

Evidence base: item 08 (`08_focus_rule_recheck`). The rule flags SQQQ trades
in a narrow RSI × ATR cell that produced severe losses with high precision
in both IS and OOS. Block-bootstrap CI is entirely positive.

Rule: SQQQ trades where
    RSI_entry ∈ [56.4, 59.85]  AND  atr_pct ∈ [0.39, 0.47]

Action: skip (size_multiplier = 0.0).

Item 08 OOS stats: 7–10 trades flagged, ~86–100% precision (severe loss rate),
block-bootstrap CI [+0.06, +0.12] pp net pnl impact.
"""
from __future__ import annotations

import pandas as pd

# Rule thresholds (from item 04 cell discovery + item 08 re-check)
RSI_LO: float = 56.4
RSI_HI: float = 59.85
ATR_LO: float = 0.39
ATR_HI: float = 0.47

SQQQ_FOCUS_RULE: dict = {
    "symbol": "SQQQ",
    "conditions": {
        "RSI_entry": (RSI_LO, RSI_HI),
        "atr_pct":   (ATR_LO, ATR_HI),
    },
    "action": "skip",
    "size_multiplier": 0.0,
    "evidence_item": "08_focus_rule_recheck",
    "oos_n_flagged": 7,          # item 08 regime-labeled count
    "oos_precision": 0.86,       # severe-loss rate in flagged trades
}


def sqqq_focus_rule_mask(df: pd.DataFrame) -> pd.Series:
    """Return boolean Series: True = apply rule (skip this trade).

    Trades not in the SQQQ RSI × ATR cell return False.
    NaN in RSI_entry or atr_pct → False (safe default: do not skip).

    Parameters
    ----------
    df : DataFrame with columns ``RSI_entry`` and ``atr_pct``.
         Must be SQQQ trades only (caller's responsibility).
    """
    rsi = pd.to_numeric(df["RSI_entry"], errors="coerce")
    atr = pd.to_numeric(df["atr_pct"], errors="coerce")
    return (
        (rsi >= RSI_LO) & (rsi <= RSI_HI) &
        (atr >= ATR_LO) & (atr <= ATR_HI)
    ).fillna(False)
