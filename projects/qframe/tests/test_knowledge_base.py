"""
Unit tests for knowledge_base/db.py — uses a temporary in-memory SQLite database.
"""
import pytest

from qframe.knowledge_base.db import KnowledgeBase


@pytest.fixture
def kb(tmp_path):
    db = KnowledgeBase(db_path=tmp_path / "test.db")
    db.init_schema()
    return db


class TestHypotheses:
    def test_add_and_retrieve(self, kb):
        hyp_id = kb.add_hypothesis(
            description="Test momentum factor",
            rationale="Underreaction",
            mechanism_score=4,
        )
        assert isinstance(hyp_id, int)
        hyp = kb.get_hypothesis(hyp_id)
        assert hyp["description"] == "Test momentum factor"
        assert hyp["mechanism_score"] == 4
        assert hyp["status"] == "backlog"

    def test_update_status(self, kb):
        hyp_id = kb.add_hypothesis(description="Test factor")
        kb.update_hypothesis_status(hyp_id, "active")
        hyp = kb.get_hypothesis(hyp_id)
        assert hyp["status"] == "active"

    def test_nonexistent_hypothesis_returns_none(self, kb):
        result = kb.get_hypothesis(99999)
        assert result is None


class TestImplementations:
    def test_add_implementation(self, kb):
        hyp_id = kb.add_hypothesis(description="Test factor")
        impl_id = kb.add_implementation(
            hypothesis_id=hyp_id,
            code="factor = prices.pct_change(21)",
            git_hash="abc123",
        )
        assert isinstance(impl_id, int)


class TestBacktestResults:
    def test_log_and_retrieve(self, kb):
        hyp_id = kb.add_hypothesis(description="Momentum")
        impl_id = kb.add_implementation(hypothesis_id=hyp_id, code="...")

        metrics = {
            "ic": 0.045,
            "icir": 0.72,
            "net_ic": 0.031,
            "sharpe": 0.95,
            "turnover": 0.4,
            "decay_halflife": 14.2,
            "ic_horizon_1": 0.045,
            "ic_horizon_5": 0.038,
            "ic_horizon_21": 0.021,
            "ic_horizon_63": 0.008,
            "regime": "all",
            "universe": "sp500",
            "oos_start": "2018-01-01",
            "oos_end": "2024-12-31",
            "cost_bps": 10.0,
            "gate_level": 0,
        }
        result_id = kb.log_result(impl_id, metrics)
        assert isinstance(result_id, int)

        results = kb.get_results(implementation_id=impl_id)
        assert len(results) == 1
        assert abs(results[0]["ic"] - 0.045) < 1e-9

    def test_unknown_metric_keys_ignored(self, kb):
        hyp_id = kb.add_hypothesis(description="Test")
        impl_id = kb.add_implementation(hypothesis_id=hyp_id, code="...")
        # Should not raise even with unknown keys
        result_id = kb.log_result(impl_id, {"ic": 0.01, "unknown_field": "ignored"})
        assert isinstance(result_id, int)

    def test_get_results_empty(self, kb):
        results = kb.get_results(implementation_id=99999)
        assert results == []


class TestFactorCorrelations:
    def test_log_and_upsert(self, kb):
        kb.log_factor_correlation("momentum", "value", 0.12, "2010-2024", "sp500")
        # Upsert same key with different value
        kb.log_factor_correlation("momentum", "value", 0.15, "2010-2024", "sp500")
        # Should not raise — idempotent upsert
