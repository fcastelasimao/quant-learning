"""
test_production_validation_tax.py (D.17)
========================================
Tests the tax-addendum policy parser and (when the central FMP store is
present) the artifact-writing path of research/production_validation.py.

The full bundle build (build_production_validation) needs network/yfinance, so
it is not exercised here; we test the decoupled addendum directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.production_validation import _resolve_policy


def test_resolve_policy_monthly():
    assert _resolve_policy("monthly").label == "monthly_unconditional"
    assert _resolve_policy("monthly_unconditional").mode == "monthly_unconditional"


def test_resolve_policy_drift_absolute():
    p = _resolve_policy("drift_absolute:0.05")
    assert p.label == "drift_absolute(0.05)"
    assert p.absolute_threshold == pytest.approx(0.05)


def test_resolve_policy_drift_relative():
    p = _resolve_policy("drift_relative:0.2")
    assert p.label == "drift_relative(0.2)"
    assert p.relative_threshold == pytest.approx(0.2)


def test_resolve_policy_rejects_garbage():
    with pytest.raises(ValueError):
        _resolve_policy("nonsense")
    with pytest.raises(ValueError):
        _resolve_policy("drift_sideways:0.1")


# --- addendum artifact path (skipped if central FMP store unavailable) ------

def _fmp_store_available() -> bool:
    try:
        from engine.data import _repo_data_dir
        return (Path(_repo_data_dir()) / "DB_SPY_historical_data.db").exists()
    except Exception:
        return False


@pytest.mark.skipif(not _fmp_store_available(), reason="central FMP store not present")
def test_build_tax_addendum_writes_artifacts(tmp_path):
    from engine import config
    config.DATA_SOURCE = "fmp"
    config.FMP_PRICE_COLUMN = "adj_close"
    from research.production_validation import build_tax_addendum

    build_tax_addendum(
        tmp_path, config.DEFAULT_STRATEGY, "2018-01-01", "2022-12-31",
        regime_name="us", lot_selector="fifo",
        policy_spec="drift_absolute:0.05", transaction_cost_pct=0.001,
    )
    for name in ("rebalance_events.csv", "tax_summary.csv",
                 "tax_monthly_series.csv", "tax_addendum_manifest.json"):
        assert (tmp_path / name).exists(), name

    manifest = json.loads((tmp_path / "tax_addendum_manifest.json").read_text())
    assert manifest["tax_regime"]["name"] == "us"
    assert manifest["rebalance_policy"]["label"] == "drift_absolute(0.05)"
    assert manifest["lot_selector"]["name"] == "fifo"
    assert manifest["results"]["final_after_tax_value"] > 0
