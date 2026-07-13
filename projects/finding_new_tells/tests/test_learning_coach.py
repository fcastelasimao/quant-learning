"""Tests for the prediction-first learning coach."""
from __future__ import annotations

import pandas as pd

from learning_coach import (
    Prediction,
    build_case_packet,
    eligible_case_dates,
    score_prediction,
)
from signal_diagnostics import (
    TEST_START,
    research_split_panel,
    tradable_open_forward_returns,
)


def test_eligible_case_dates_have_history_and_forward_outcome(synthetic_panel):
    lookback = 63
    horizon = 5

    dates = eligible_case_dates(
        synthetic_panel,
        split="val",
        horizon=horizon,
        lookback=lookback,
    )

    panel_val = research_split_panel(synthetic_panel, "val")
    tqqq_fwd = tradable_open_forward_returns(
        panel_val, symbol="TQQQ", horizons=(horizon,)
    )[horizon]
    assert len(dates) > 0
    for date in dates:
        pos = panel_val.index.get_loc(date)
        assert pos >= lookback - 1
        assert pd.notna(tqqq_fwd.loc[date])


def test_default_research_cases_exclude_frozen_test_window(synthetic_panel):
    dates = eligible_case_dates(
        synthetic_panel,
        split="train+val",
        horizon=20,
        lookback=126,
    )

    assert len(dates) > 0
    assert dates.max() < TEST_START


def test_case_packet_forward_return_matches_diagnostics_timing(synthetic_panel):
    horizon = 10
    lookback = 63
    date = eligible_case_dates(
        synthetic_panel,
        split="val",
        horizon=horizon,
        lookback=lookback,
    )[20]

    case = build_case_packet(
        synthetic_panel,
        split="val",
        horizon=horizon,
        lookback=lookback,
        case_date=date,
        families=["mean_rev"],
    )

    panel_val = research_split_panel(synthetic_panel, "val")
    expected = tradable_open_forward_returns(
        panel_val, symbol="TQQQ", horizons=(horizon,)
    )[horizon].loc[date]
    assert case.outcome["target_kind"] == "tradable_open"
    assert case.outcome["tqqq_forward_return"] == expected


def test_score_prediction_is_deterministic_for_direction_and_confidence():
    correct = score_prediction(
        Prediction(
            direction="bullish",
            confidence=5,
            likely_regime="weak bull",
            mechanism="trend continuation",
            expected_family="trend",
            prove_wrong="breakdown",
        ),
        actual_return=0.02,
    )
    partial = score_prediction(
        {"direction": "neutral", "confidence": 3},
        actual_return=0.02,
    )
    wrong = score_prediction(
        {"direction": "bearish", "confidence": 4},
        actual_return=0.02,
    )

    assert correct["actual_direction"] == "bullish"
    assert correct["direction_score"] == 1.0
    assert partial["direction_score"] == 0.5
    assert wrong["direction_score"] == 0.0
    assert "High-confidence miss" in wrong["confidence_note"]
