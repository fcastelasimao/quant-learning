"""TQQQ regime rule: skip trades in the sideways_lowvol × ATR × MA100 danger zone.

Evidence base: item 11 (`11_regime_conditional_rules`). The meta-rule
"TQQQ severe-loss rule restricted to regime == sideways_lowvol" produced
10 flagged trades in R-OOS with 100% precision and +10.6 pp net pnl impact.
Embargo (2026) held.

Rule: TQQQ trades where ALL of:
    regime_entry == "sideways_lowvol"
    atr_pct ∈ (0.4236, 0.476]      (> lower bound, ≤ upper bound)
    MA100_D1 ≤ 0.0002044

Action: skip (size_multiplier = 0.0).

Item 11 R-OOS stats: 10 trades flagged, 100% precision (severe loss rate),
net pnl impact +10.6 pp. Embargo flag survived.
"""
from __future__ import annotations

import pandas as pd

# Rule thresholds (from item 04 depth-4 tree leaf + item 11 regime conditioning)
ATR_LO: float = 0.4236
ATR_HI: float = 0.476
MA100_D1_HI: float = 0.0002044
REGIME: str = "sideways_lowvol"

TQQQ_REGIME_RULE: dict = {
    "symbol": "TQQQ",
    "conditions": {
        "regime_entry": REGIME,
        "atr_pct":    (ATR_LO, ATR_HI),     # > ATR_LO AND <= ATR_HI
        "MA100_D1":   (None, MA100_D1_HI),  # <= MA100_D1_HI
    },
    "action": "skip",
    "size_multiplier": 0.0,
    "evidence_item": "11_regime_conditional_rules",
    "oos_n_flagged": 10,
    "oos_precision": 1.00,
    "oos_net_pnl_impact_pp": 10.6,
}


def tqqq_regime_rule_mask(df: pd.DataFrame) -> pd.Series:
    """Return boolean Series: True = apply rule (skip this trade).

    Trades outside the sideways_lowvol × ATR × MA100 cell return False.
    NaN in any condition column → False (safe default: do not skip).

    Parameters
    ----------
    df : DataFrame with columns ``regime_entry``, ``atr_pct``, ``MA100_D1``.
         Must be TQQQ trades only (caller's responsibility).
    """
    atr = pd.to_numeric(df["atr_pct"], errors="coerce")
    ma = pd.to_numeric(df["MA100_D1"], errors="coerce")
    regime_match = (df["regime_entry"] == REGIME).fillna(False)
    return (
        regime_match &
        (atr > ATR_LO) & (atr <= ATR_HI) &
        (ma <= MA100_D1_HI)
    ).fillna(False)
