"""Tests for the shared rule-naming utility used by items 04, 06, 08, 11, 17.

The contract:
  - rule_hash is deterministic from canonical (sorted) conditions
  - rule_hash is invariant to condition ordering in input
  - rule_name is stable (same hash) across reorderings
  - rule_description renders human-readable text
"""
from __future__ import annotations

import pytest

import _rule_naming as rn


def test_hash_is_deterministic():
    conds = [("atr_pct", "<=", 0.476), ("RSI_entry", ">", 60.0)]
    h1 = rn.rule_hash(conds)
    h2 = rn.rule_hash(conds)
    assert h1 == h2
    assert len(h1) == 4
    assert all(c in "0123456789abcdef" for c in h1)


def test_hash_is_order_invariant():
    """Condition order should not affect the hash — that's why canonical_conditions sorts."""
    a = [("atr_pct", "<=", 0.476), ("RSI_entry", ">", 60.0)]
    b = [("RSI_entry", ">", 60.0), ("atr_pct", "<=", 0.476)]
    assert rn.rule_hash(a) == rn.rule_hash(b)


def test_hash_distinguishes_different_thresholds():
    a = [("atr_pct", "<=", 0.476)]
    b = [("atr_pct", "<=", 0.477)]
    assert rn.rule_hash(a) != rn.rule_hash(b)


def test_hash_distinguishes_different_operators():
    a = [("atr_pct", "<=", 0.476)]
    b = [("atr_pct", ">", 0.476)]
    assert rn.rule_hash(a) != rn.rule_hash(b)


def test_rule_name_includes_symbol_target_hash():
    conds = [("atr_pct", "<=", 0.476), ("RSI_entry", ">", 60.0)]
    name = rn.rule_name("SQQQ", "is_severe_loss", conds)
    assert name.startswith("SQQQ_severe_")
    assert name.endswith("_" + rn.rule_hash(conds))


def test_rule_name_is_order_invariant():
    a = [("atr_pct", "<=", 0.476), ("RSI_entry", ">", 60.0)]
    b = [("RSI_entry", ">", 60.0), ("atr_pct", "<=", 0.476)]
    # The HASH is order-invariant. The feature-abbreviation segment depends
    # on input order — that's a known cosmetic quirk; the hash guarantees identity.
    assert rn.rule_name("SQQQ", "is_severe_loss", a).split("_")[-1] == \
           rn.rule_name("SQQQ", "is_severe_loss", b).split("_")[-1]


def test_rule_description_renders_human_readable():
    conds = [("atr_pct", "<=", 0.476), ("RSI_entry", ">", 60.0)]
    desc = rn.rule_description(conds)
    assert "atr_pct <= 0.476" in desc
    assert "RSI_entry > 60" in desc
    assert " AND " in desc


def test_abbreviation_table_covers_curated_features():
    """Every feature in item 04's curated_12 + regime dummies should have an abbreviation."""
    curated = [
        "atr_pct", "RSI_entry", "BBP_entry",
        "dist_to_MA20", "dist_to_MA50", "dist_to_MA100",
        "MA20_D5", "MA50_D5", "MA100_D1",
        "log_volume_ratio", "bars_since_last_stop", "hour_of_entry",
        "regime_chop_highvol", "regime_sideways_lowvol",
    ]
    for feat in curated:
        assert feat in rn.ABBR, f"Missing abbreviation for {feat}"


def test_target_short_table_covers_used_targets():
    for target in ("is_loser", "is_severe_loss", "is_severe_win", "pnl_pct"):
        assert target in rn.TARGET_SHORT, f"Missing short for target {target}"


def test_unknown_feature_falls_back_gracefully():
    """A feature not in ABBR should produce a truncated stem, not crash."""
    conds = [("some_brand_new_feature", "<=", 1.0)]
    name = rn.rule_name("TQQQ", "is_loser", conds)
    # Doesn't raise; produces a string
    assert isinstance(name, str)
    assert len(name) > 0
