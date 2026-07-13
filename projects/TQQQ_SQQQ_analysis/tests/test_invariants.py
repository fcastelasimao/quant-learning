"""Invariants that must hold across the dataset and the research pipeline.

These catch the silent kinds of bugs:
  - pnl_pct double-scaling
  - regime filter inconsistency
  - walk-forward boundary leakage
  - row-count drift between items
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# --------------------------------------------------------------------------- #
# Source-data invariants
# --------------------------------------------------------------------------- #

def test_canonical_row_counts(canon_tqqq, canon_sqqq):
    """Canonical CSVs have stable row counts. If these change, every downstream
    number quoted in findings docs is suspect."""
    assert len(canon_tqqq) == 2343, f"TQQQ canonical changed: {len(canon_tqqq)} rows"
    assert len(canon_sqqq) == 1930, f"SQQQ canonical changed: {len(canon_sqqq)} rows"


def test_regime_labeled_subset_counts(regime_tqqq, regime_sqqq):
    """SESSION_HANDOFF and SYNTHESIS quote these numbers explicitly."""
    assert len(regime_tqqq) == 1782, f"TQQQ regime-labeled changed: {len(regime_tqqq)}"
    assert len(regime_sqqq) == 1511, f"SQQQ regime-labeled changed: {len(regime_sqqq)}"


def test_pnl_pct_is_in_percentage_points(canon_tqqq, canon_sqqq):
    """Median |pnl_pct| should be single-digit. Double-scaling produces medians ~100."""
    for sym, df in (("TQQQ", canon_tqqq), ("SQQQ", canon_sqqq)):
        med = float(df["pnl_pct"].median())
        assert abs(med) < 5, f"{sym} median pnl_pct = {med:.2f} — looks double-scaled"
        # Sanity range
        assert df["pnl_pct"].abs().quantile(0.99) < 50, f"{sym} pnl_pct p99 > 50% — implausible"


def test_is_loser_consistency_with_pnl(canon_tqqq, canon_sqqq):
    """is_loser must equal pnl_pct < 0."""
    for sym, df in (("TQQQ", canon_tqqq), ("SQQQ", canon_sqqq)):
        derived = (df["pnl_pct"] < 0)
        stored = df["is_loser"].astype(bool)
        mismatch = (derived != stored).sum()
        assert mismatch == 0, f"{sym} is_loser has {mismatch} mismatches with pnl_pct < 0"


def test_is_severe_loss_consistency_with_pnl(canon_tqqq, canon_sqqq):
    """is_severe_loss must equal pnl_pct <= -1."""
    for sym, df in (("TQQQ", canon_tqqq), ("SQQQ", canon_sqqq)):
        derived = (df["pnl_pct"] <= -1.0)
        stored = df["is_severe_loss"].astype(bool)
        mismatch = (derived != stored).sum()
        assert mismatch == 0, f"{sym} is_severe_loss has {mismatch} mismatches with pnl_pct <= -1"


def test_regime_values_are_known(regime_tqqq, regime_sqqq):
    """After filtering NaN, only the three expected regimes should remain."""
    expected = {"bull", "chop_highvol", "sideways_lowvol"}
    for sym, df in (("TQQQ", regime_tqqq), ("SQQQ", regime_sqqq)):
        actual = set(df["regime_entry"].unique())
        assert actual.issubset(expected), f"{sym} unexpected regimes: {actual - expected}"
        assert actual == expected, f"{sym} missing regimes: {expected - actual}"


def test_regime_labeled_earliest_year_is_2015(regime_tqqq, regime_sqqq):
    """Item 01 finding: regime labels only start in 2015. If this drifts,
    the IS/OOS window definitions in items 04-17 need re-checking."""
    for sym, df in (("TQQQ", regime_tqqq), ("SQQQ", regime_sqqq)):
        earliest = df["entry_time"].dt.year.min()
        assert earliest == 2015, f"{sym} earliest regime-labeled year = {earliest}, expected 2015"


def test_high_water_mark_is_alias_of_entry_price(canon_tqqq, canon_sqqq):
    """Known bug in source data (FEATURE_DICTIONARY.md, item 01 caveats).
    If this test breaks, the data team fixed it and we should re-enable
    `high_water_mark_entry` as a meaningful feature."""
    for sym, df in (("TQQQ", canon_tqqq), ("SQQQ", canon_sqqq)):
        if "high_water_mark_entry" not in df.columns:
            pytest.skip(f"{sym} has no high_water_mark_entry column")
        equal = (df["high_water_mark_entry"] == df["avg_order_price"]).all()
        assert equal, (
            f"{sym} high_water_mark_entry NO LONGER equals avg_order_price — "
            "if the data team fixed this, update FEATURE_DICTIONARY.md and SESSION_HANDOFF.md."
        )


# --------------------------------------------------------------------------- #
# Cross-item consistency
# --------------------------------------------------------------------------- #

def test_item_01_sample_sizes_match_canonical(research_dir, regime_tqqq, regime_sqqq):
    """Item 01's sample_sizes.csv totals must equal the regime-labeled fixture."""
    path = research_dir / "01_data_diagnostics" / "sample_sizes.csv"
    if not path.exists():
        pytest.skip("item 01 sample_sizes.csv not present")
    ss = pd.read_csv(path)
    for sym, df_ref in (("TQQQ", regime_tqqq), ("SQQQ", regime_sqqq)):
        total_row = ss[(ss["symbol"] == sym) & (ss["regime_entry"] == "_TOTAL_regime_labeled")]
        assert len(total_row) == 1
        assert int(total_row.iloc[0]["n"]) == len(df_ref), (
            f"{sym} item-01 total {int(total_row.iloc[0]['n'])} != {len(df_ref)}"
        )


# --------------------------------------------------------------------------- #
# Walk-forward boundary checks (read code, not data)
# --------------------------------------------------------------------------- #

def _scan_for_wf_boundary_bugs(scripts: list[Path]) -> list[str]:
    """A walk-forward train mask should use `< y_test` or `<= y_train_end`.
    A `<= y` boundary on the test-year variable would leak future data.

    Returns a list of suspicious lines (file:lineno).
    """
    findings = []
    for script in scripts:
        if not script.exists():
            continue
        text = script.read_text()
        for i, line in enumerate(text.splitlines(), start=1):
            # Look for "train = df[df['year'] <= y_test]" style — bad
            # vs "train = df[df['year'] < y]" — good
            # Heuristic: any "year"] <= " in a train/fit context
            if re.search(r"train\s*=.*year.*<=", line) and "y_end" not in line and "year_end" not in line:
                findings.append(f"{script.name}:{i}: {line.strip()}")
    return findings


def test_no_walkforward_leakage_in_build_scripts(research_dir):
    """Scan each build_*.py for a `train = df[df.year <= y]` pattern where `y`
    is the test year. False positives are possible — `y_end` is the train-end
    year and `<= y_end` is correct."""
    scripts = list(research_dir.glob("*/build_*.py"))
    findings = _scan_for_wf_boundary_bugs(scripts)
    assert not findings, "Possible WF leakage:\n  " + "\n  ".join(findings)


# --------------------------------------------------------------------------- #
# Sanity on item 05 / 13 equity-metric outputs
# --------------------------------------------------------------------------- #

def test_item_05_metrics_compare_present(research_dir):
    path = research_dir / "05_capital_normalization" / "metrics_compare.csv"
    if not path.exists():
        pytest.skip("item 05 metrics_compare.csv not present")
    df = pd.read_csv(path)
    # The Sharpe row should show old vs new with the diff under ~0.1 — item 05's
    # finding that Sharpe is scale-invariant.
    sharpe = df[df["metric"] == "sharpe"]
    assert len(sharpe) >= 2  # both symbols
    for _, row in sharpe.iterrows():
        old = float(row["old_pipeline_inflated"])
        new = float(row["new_constant_notional_full"])
        assert abs(old - new) < 0.15, (
            f"Item 05 Sharpe drift {row['symbol']}: old={old:.3f} new={new:.3f}. "
            "Item 05 finding was Sharpe is scale-invariant — this drifted."
        )


def test_item_13_compounded_cagr_lower_than_constant_notional(research_dir):
    """Item 13 finding: within-CSV compounded CAGR is *lower* than the constant-
    notional annualized arithmetic return, because compounding spreads the same
    total return over time."""
    path = research_dir / "13_within_csv_compounding" / "compounding_compare.csv"
    if not path.exists():
        pytest.skip("item 13 compounding_compare.csv not present")
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        const_ann = float(row["constant_notional_annualized"])
        geo_cagr = float(row["geo_mean_cagr"])
        assert geo_cagr < const_ann, (
            f"{row['symbol']}: within-CSV CAGR {geo_cagr:.3f} >= constant_notional "
            f"annualized {const_ann:.3f}. Item 13's central finding broke."
        )
