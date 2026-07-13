"""Shared feature contract for the walk-forward `p_severe` scorer."""
from __future__ import annotations

SUPPORTED_SYMBOLS = ("TQQQ", "SQQQ")

TARGET_COLUMN = "is_severe_loss"
TARGET_DESCRIPTION = "pnl_pct <= -1.0"
CONTEXT_POLICY = "strict_prior_daily_context_date_lt_entry_date"

CURATED_NUMERIC = [
    "atr_pct",
    "RSI_entry",
    "BBP_entry",
    "dist_to_MA20",
    "dist_to_MA50",
    "dist_to_MA100",
    "MA20_D5",
    "MA50_D5",
    "MA100_D1",
    "log_volume_ratio",
    "bars_since_last_stop",
    "hour_of_entry",
]

DAILY_CONTEXT = [
    "QQQ_RSI_14",
    "QQQ_dist_MA20",
    "QQQ_dist_MA50",
    "QQQ_dist_MA200",
    "QQQ_realized_vol_20d",
    "QQQ_dist_high_20d",
    "QQQ_50d_return",
    "QQQ_50d_return_pctile_252",
    "QQQ_drawdown_5d",
    "QQQ_drawdown_60d",
    "QQQ_gap_overnight",
    "SPY_RSI_14",
    "SPY_dist_MA50",
    "VIX_level",
    "VIX_5d_change",
    "VIX_pctile_252d",
    "VIX_term_structure",
    "HYG_LQD_ratio",
    "HYG_5d_change",
    "yield_curve_slope",
    "TNX_5d_change",
]

REGIME_DUMMIES = ["regime_chop_highvol", "regime_sideways_lowvol"]

MODEL_FEATURES = CURATED_NUMERIC + REGIME_DUMMIES + DAILY_CONTEXT

STRATEGY_INTERNAL_COLUMNS = [
    "RSI_entry",
    "BBP_entry",
    "bars_since_last_stop",
    "regime_entry",
]

RAW_DAILY_SYMBOLS = ("QQQ", "SPY", "^VIX", "^VIX3M", "HYG", "LQD", "^TNX", "^IRX")
