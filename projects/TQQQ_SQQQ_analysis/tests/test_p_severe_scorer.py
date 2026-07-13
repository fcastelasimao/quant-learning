"""Regression tests for the deployable walk-forward p_severe scorer."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from TQQQ_SQQQ_analysis.deployable_strategies.continuous_sizing.p_severe_scorer.constants import MODEL_FEATURES
from TQQQ_SQQQ_analysis.deployable_strategies.continuous_sizing.p_severe_scorer.features import FeatureContractError, compute_required_features
from TQQQ_SQQQ_analysis.deployable_strategies.continuous_sizing.p_severe_scorer.score import (
    score_trades,
    score_trades_both,
    score_trades_by_symbol,
    score_trades_with_models_both,
    score_trades_with_models_by_symbol,
)
from TQQQ_SQQQ_analysis.deployable_strategies.continuous_sizing.p_severe_scorer.training import train_models_from_history

_ENRICHED = Path(__file__).resolve().parent.parent / "research" / "06_context_enrichment" / "enriched_trades_TQQQ.csv"
if not _ENRICHED.exists():
    pytest.skip("enriched trade data not present", allow_module_level=True)


@pytest.fixture(scope="session")
def scorer_dir(proj_root):
    return proj_root / "deployable_strategies" / "continuous_sizing" / "p_severe_scorer"


def _item17_module():
    research = Path(__file__).resolve().parent.parent / "research" / "17_sizing_with_enriched_features"
    sys.path.insert(0, str(research))
    import build_17_sizing_with_enriched_features as item17  # noqa: PLC0415

    return item17


def test_p_severe_scorer_matches_item_17_walkforward_probabilities(proj_root, scorer_dir):
    item17 = _item17_module()
    artifact_dir = scorer_dir / "artifacts"

    for sym in ("TQQQ", "SQQQ"):
        df = pd.read_csv(
            proj_root / "research" / "06_context_enrichment" / f"enriched_trades_{sym}.csv",
            parse_dates=["entry_time", "exit_time"],
        )
        df = compute_required_features(df)
        df["year"] = df["entry_time"].dt.year
        features = item17.CURATED_NUMERIC + item17.REGIME_DUMMIES + item17.DAILY_CONTEXT
        expected = item17.fit_predict_walkforward(df.copy(), features, "is_severe_loss")
        scored = score_trades(df, sym, artifact_dir=artifact_dir, on_missing_model="nan", validate_features=False)
        mask = expected.notna()
        assert mask.sum() > 100
        assert np.allclose(scored.loc[mask, "p_severe"], expected.loc[mask], atol=1e-12)
        assert np.allclose(scored.loc[mask, "size_multiplier"], 1.0 - expected.loc[mask], atol=1e-12)


def test_p_severe_scorer_is_lookahead_safe(proj_root, scorer_dir):
    df = pd.read_csv(
        proj_root / "research" / "06_context_enrichment" / "enriched_trades_TQQQ.csv",
        parse_dates=["entry_time", "exit_time"],
    )
    scored = score_trades(
        df,
        "TQQQ",
        artifact_dir=scorer_dir / "artifacts",
        on_missing_model="nan",
        validate_features=False,
    )
    scored = scored.dropna(subset=["p_severe"]).copy()
    assert (scored["model_train_end_year"] < scored["entry_time"].dt.year).all()


def test_item_06_context_is_strictly_prior(proj_root):
    for sym in ("TQQQ", "SQQQ"):
        df = pd.read_csv(
            proj_root / "research" / "06_context_enrichment" / f"enriched_trades_{sym}.csv",
            parse_dates=["entry_time", "date"],
        )
        assert (df["date"] < df["entry_time"].dt.normalize()).all()


def test_artifacts_record_strict_prior_metadata(scorer_dir):
    path = scorer_dir / "artifacts" / "TQQQ" / "model_2026.json"
    data = json.loads(path.read_text())
    assert data["context_policy"] == "strict_prior_daily_context_date_lt_entry_date"
    assert len(data["dataset_hash"]) == 64
    assert data["target_description"] == "pnl_pct <= -1.0"


def test_p_severe_future_year_uses_latest_prior_model(proj_root, scorer_dir):
    df = pd.read_csv(
        proj_root / "research" / "06_context_enrichment" / "enriched_trades_TQQQ.csv",
        parse_dates=["entry_time", "exit_time"],
    )
    df = compute_required_features(df)
    row = df.dropna(subset=MODEL_FEATURES).tail(1).copy()
    row["entry_time"] = pd.Timestamp("2027-01-15 10:00")
    scored = score_trades(
        row,
        "TQQQ",
        artifact_dir=scorer_dir / "artifacts",
        on_missing_model="raise",
    )
    assert int(scored.iloc[0]["model_year_used"]) == 2026
    assert int(scored.iloc[0]["model_train_end_year"]) == 2025
    assert scored.iloc[0]["model_train_end_year"] < scored.iloc[0]["entry_time"].year


def test_score_trades_both_includes_tqqq_and_sqqq_outputs(proj_root, scorer_dir):
    df = pd.read_csv(
        proj_root / "research" / "06_context_enrichment" / "enriched_trades_TQQQ.csv",
        parse_dates=["entry_time", "exit_time"],
    )
    df = compute_required_features(df)
    sample = df.dropna(subset=MODEL_FEATURES).tail(5).copy()
    artifact_dir = scorer_dir / "artifacts"

    both = score_trades_both(sample, artifact_dir=artifact_dir)
    tqqq = score_trades(sample, "TQQQ", artifact_dir=artifact_dir)
    sqqq = score_trades(sample, "SQQQ", artifact_dir=artifact_dir)

    for col in (
        "p_severe_TQQQ",
        "size_multiplier_TQQQ",
        "model_year_used_TQQQ",
        "model_train_end_year_TQQQ",
        "p_severe_SQQQ",
        "size_multiplier_SQQQ",
        "model_year_used_SQQQ",
        "model_train_end_year_SQQQ",
    ):
        assert col in both.columns
    assert np.allclose(both["p_severe_TQQQ"], tqqq["p_severe"], atol=1e-12)
    assert np.allclose(both["size_multiplier_TQQQ"], 1.0 - both["p_severe_TQQQ"], atol=1e-12)
    assert np.allclose(both["p_severe_SQQQ"], sqqq["p_severe"], atol=1e-12)
    assert np.allclose(both["size_multiplier_SQQQ"], 1.0 - both["p_severe_SQQQ"], atol=1e-12)


def test_score_trades_by_symbol_uses_each_rows_own_symbol(scorer_dir):
    fixture = pd.read_csv(
        scorer_dir / "examples" / "candidate_trades_example.csv",
        parse_dates=["entry_time"],
    )
    artifact_dir = scorer_dir / "artifacts"
    auto = score_trades_by_symbol(fixture, artifact_dir=artifact_dir)
    tqqq = score_trades(fixture[fixture["symbol"] == "TQQQ"], "TQQQ", artifact_dir=artifact_dir)
    sqqq = score_trades(fixture[fixture["symbol"] == "SQQQ"], "SQQQ", artifact_dir=artifact_dir)

    assert auto["p_severe"].notna().all()
    assert auto["size_multiplier"].notna().all()
    assert auto.loc[auto["symbol"] == "TQQQ", "p_severe"].iloc[0] == pytest.approx(tqqq["p_severe"].iloc[0])
    assert auto.loc[auto["symbol"] == "SQQQ", "p_severe"].iloc[0] == pytest.approx(sqqq["p_severe"].iloc[0])


def test_incremental_training_scores_next_year_only(proj_root, scorer_dir):
    tqqq = pd.read_csv(
        proj_root / "research" / "06_context_enrichment" / "enriched_trades_TQQQ.csv",
        parse_dates=["entry_time", "exit_time"],
    )
    sqqq = pd.read_csv(
        proj_root / "research" / "06_context_enrichment" / "enriched_trades_SQQQ.csv",
        parse_dates=["entry_time", "exit_time"],
    )
    history = pd.concat([tqqq, sqqq], ignore_index=True)
    history = compute_required_features(history)
    history_through_2025 = history[history["entry_time"].dt.year <= 2025].copy()

    models = train_models_from_history(history_through_2025, predict_year=2026)
    assert models["TQQQ"].train_end_year == 2025
    assert models["SQQQ"].train_end_year == 2025

    candidates_2026 = history[
        (history["symbol"] == "TQQQ")
        & (history["entry_time"].dt.year == 2026)
    ].dropna(subset=MODEL_FEATURES).head(3).copy()

    in_memory = score_trades_with_models_both(candidates_2026, models)
    artifact = score_trades_both(
        candidates_2026,
        artifact_dir=scorer_dir / "artifacts",
    )
    assert np.allclose(in_memory["p_severe_TQQQ"], artifact["p_severe_TQQQ"], atol=1e-12)
    assert np.allclose(in_memory["p_severe_SQQQ"], artifact["p_severe_SQQQ"], atol=1e-12)

    training_year_rows = history[
        (history["symbol"] == "TQQQ")
        & (history["entry_time"].dt.year == 2025)
    ].dropna(subset=MODEL_FEATURES).head(1).copy()
    with pytest.raises(ValueError, match="cannot score entry years"):
        score_trades_with_models_both(training_year_rows, models)


def test_incremental_training_by_symbol_scores_candidate_rows(proj_root, scorer_dir):
    tqqq = pd.read_csv(
        proj_root / "research" / "06_context_enrichment" / "enriched_trades_TQQQ.csv",
        parse_dates=["entry_time", "exit_time"],
    )
    sqqq = pd.read_csv(
        proj_root / "research" / "06_context_enrichment" / "enriched_trades_SQQQ.csv",
        parse_dates=["entry_time", "exit_time"],
    )
    history = pd.concat([tqqq, sqqq], ignore_index=True)
    history = compute_required_features(history)
    models = train_models_from_history(history[history["entry_time"].dt.year <= 2025].copy(), predict_year=2026)

    fixture = pd.read_csv(
        scorer_dir / "examples" / "candidate_trades_example.csv",
        parse_dates=["entry_time"],
    )
    scored = score_trades_with_models_by_symbol(fixture, models)
    assert scored["p_severe"].between(0, 1).all()
    assert np.allclose(scored["size_multiplier"], 1.0 - scored["p_severe"], atol=1e-12)


def test_calibration_outputs_present(proj_root):
    for name in ("calibration_deciles_enriched_1pct.csv", "calibration_yearly_enriched_1pct.csv"):
        path = proj_root / "research" / "17_sizing_with_enriched_features" / name
        df = pd.read_csv(path)
        assert {"symbol", "n", "p_mean", "severe_rate", "abs_calibration_error"}.issubset(df.columns)
        assert len(df) > 0


def test_yearly_checkpoint_example_runs(scorer_dir, proj_root):
    script = scorer_dir / "examples" / "yearly_checkpoint_example.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=proj_root.parent,
        check=True,
        text=True,
        capture_output=True,
    )
    assert "p_severe" in result.stdout
    assert "size_multiplier" in result.stdout


def test_compute_required_features_derives_non_strategy_columns():
    raw = pd.DataFrame({
        "entry_time": ["2024-01-02 11:15:00"],
        "avg_order_price": [100.0],
        "atr": [1.25],
        "volume_ratio": [2.0],
        "MA20": [95.0],
        "MA50": [90.0],
        "MA100": [80.0],
        "regime_entry": ["sideways_lowvol"],
    })
    out = compute_required_features(raw)
    assert out.loc[0, "atr_pct"] == pytest.approx(1.25)
    assert out.loc[0, "log_volume_ratio"] == pytest.approx(np.log(2.0))
    assert out.loc[0, "hour_of_entry"] == 11
    assert out.loc[0, "dist_to_MA20"] == pytest.approx(100.0 / 95.0 - 1.0)
    assert out.loc[0, "regime_sideways_lowvol"] == 1
    assert out.loc[0, "regime_chop_highvol"] == 0


def test_score_trades_reports_missing_strategy_internal_fields(scorer_dir):
    df = pd.DataFrame({
        "entry_time": [pd.Timestamp("2024-01-02 10:00")],
        "avg_order_price": [100.0],
        "atr": [1.0],
        "volume_ratio": [1.2],
        "MA20": [99.0],
        "MA50": [98.0],
        "MA100": [97.0],
    })
    with pytest.raises(FeatureContractError, match="Strategy-internal fields"):
        score_trades(df, "TQQQ", artifact_dir=scorer_dir / "artifacts")
