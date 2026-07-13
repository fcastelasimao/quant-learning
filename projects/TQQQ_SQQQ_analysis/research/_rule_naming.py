"""Shared rule-naming utility.

Per the project's naming convention:
  PRIMARY NAME = {symbol}_{target_short}_{feat_abbr_list}_{4-char hash}
  where:
    - symbol is "TQQQ"/"SQQQ"
    - target_short is "loser" / "severe" / "swin" etc.
    - feat_abbr_list joins the abbreviations of the features used (max 3)
    - hash is md5 of the canonical condition list (4 hex chars) for stable identity

A separate "description" string carries the full human-readable path conditions.
"""
from __future__ import annotations

import hashlib

ABBR = {
    "atr_pct": "atr",
    "atr_pct_roll_pctile_252": "atrp",
    "RSI_entry": "rsi",
    "RSI_entry_roll_pctile_252": "rsip",
    "BBP_entry": "bbp",
    "BBP_entry_roll_pctile_252": "bbpp",
    "dist_to_MA20": "d20",
    "dist_to_MA50": "d50",
    "dist_to_MA100": "d100",
    "MA20_D1": "s20a",
    "MA20_D3": "s20b",
    "MA20_D5": "s20c",
    "MA50_D1": "s50a",
    "MA50_D3": "s50b",
    "MA50_D5": "s50c",
    "MA100_D1": "s100",
    "log_volume_ratio": "vol",
    "volume_ratio": "vol",
    "volume_ratio_roll_pctile_252": "volp",
    "volume_cur": "vcur",
    "bars_since_last_stop": "bsls",
    "hour_of_entry": "hr",
    "is_bullish_c1c2": "c1c2",
    "regime_bull": "rgB",
    "regime_chop_highvol": "rgC",
    "regime_sideways_lowvol": "rgS",
    # context-enrichment features (added in item 06)
    "QQQ_RSI_14": "qrsi",
    "QQQ_dist_MA20": "qd20",
    "QQQ_dist_MA50": "qd50",
    "QQQ_dist_MA200": "qd200",
    "QQQ_realized_vol_20d": "qrv20",
    "QQQ_dist_high_20d": "qhi20",
    "QQQ_50d_return": "qr50",
    "SPY_RSI_14": "srsi",
    "SPY_dist_MA50": "sd50",
    "VIX_level": "vix",
    "VIX_5d_change": "vix5d",
    "VIX_pctile_252d": "vixp",
    "VIX_term_structure": "vixts",
    "HYG_LQD_ratio": "cred",
    "HYG_5d_change": "hyg5d",
    "yield_curve_slope": "ycs",
    "TNX_5d_change": "tnx5d",
}

TARGET_SHORT = {
    "is_loser": "loser",
    "is_severe_loss": "severe",
    "is_severe_win": "swin",
    "pnl_pct": "pnl",
}


def abbr_feature(name: str) -> str:
    return ABBR.get(name, name[:6])


def short_target(target: str) -> str:
    return TARGET_SHORT.get(target, target[:6])


def canonical_conditions(conditions: list[tuple[str, str, float]]) -> str:
    """Stable string representation of the rule path for hashing."""
    items = sorted(((f, op, round(float(t), 8)) for f, op, t in conditions))
    return "|".join(f"{f}{op}{t}" for f, op, t in items)


def rule_hash(conditions: list[tuple[str, str, float]]) -> str:
    return hashlib.md5(canonical_conditions(conditions).encode()).hexdigest()[:4]


def rule_name(symbol: str, target: str, conditions: list[tuple[str, str, float]],
              max_feats_in_name: int = 3) -> str:
    feats = []
    seen = set()
    for f, _, _ in conditions:
        if f in seen:
            continue
        seen.add(f)
        feats.append(abbr_feature(f))
        if len(feats) >= max_feats_in_name:
            break
    feat_part = "-".join(feats)
    h = rule_hash(conditions)
    return f"{symbol}_{short_target(target)}_{feat_part}_{h}"


def rule_description(conditions: list[tuple[str, str, float]]) -> str:
    """Plain-English rule description, e.g.:
        'atr_pct > 0.476 AND RSI_entry <= 60.0'
    """
    return " AND ".join(f"{f} {op} {t:.4g}" for f, op, t in conditions)
