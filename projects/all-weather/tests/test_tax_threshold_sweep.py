"""
test_tax_threshold_sweep.py (D.18)
==================================
Tests the pre-registered kill criterion in research/tax_threshold_sweep.py.
These exercise the decision boundary on hand-built summary frames — no network,
no simulation — so the verdict logic is pinned independently of the data.
"""

from __future__ import annotations

import pandas as pd

from research.tax_drift_trigger.tax_threshold_sweep import (
    KILL_CRITERION_PCT,
    OOS_WINDOWS,
    evaluate_kill_criterion,
)


def _summary(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _make(selector, regime, policy, calmars: dict[str, float]) -> list[dict]:
    return [
        {"selector": selector, "regime": regime, "policy": policy,
         "window": win, "calmar": calmars[win], "cagr": 0.0, "mdd": -0.1,
         "final_value": 1.0, "cumulative_tax": 0.0, "rebalances": 1}
        for win in OOS_WINDOWS
    ]


def test_no_improvement_stays_research_only():
    rows = []
    rows += _make("fifo", "us", "monthly_unconditional", {"2018": 0.30, "2020": 0.30, "2022": 0.30})
    rows += _make("fifo", "us", "drift_absolute_5pp", {"2018": 0.30, "2020": 0.30, "2022": 0.30})
    verdict = evaluate_kill_criterion(_summary(rows))
    assert verdict["decision"] == "research_only"
    assert verdict["passing_policies"] == []


def test_improvement_on_all_windows_proposes_new_policy():
    big = 0.30 * (1 + KILL_CRITERION_PCT + 0.01)
    rows = []
    rows += _make("fifo", "us", "monthly_unconditional", {"2018": 0.30, "2020": 0.30, "2022": 0.30})
    rows += _make("fifo", "us", "drift_absolute_5pp", {"2018": big, "2020": big, "2022": big})
    verdict = evaluate_kill_criterion(_summary(rows))
    assert verdict["decision"] == "propose_new_production_policy"
    assert "fifo/drift_absolute_5pp" in verdict["passing_policies"]
    assert verdict["detail"]["fifo/drift_absolute_5pp"]["passed"] is True


def test_two_of_three_windows_is_enough():
    win_c = 0.30 * (1 + KILL_CRITERION_PCT + 0.01)
    rows = []
    rows += _make("fifo", "us", "monthly_unconditional", {"2018": 0.30, "2020": 0.30, "2022": 0.30})
    # wins 2018 & 2020, loses 2022
    rows += _make("fifo", "us", "drift_relative_20pct", {"2018": win_c, "2020": win_c, "2022": 0.29})
    verdict = evaluate_kill_criterion(_summary(rows))
    assert "fifo/drift_relative_20pct" in verdict["passing_policies"]
    assert verdict["detail"]["fifo/drift_relative_20pct"]["windows_won"] == ["2018", "2020"]


def test_one_window_is_not_enough():
    win_c = 0.30 * (1 + KILL_CRITERION_PCT + 0.01)
    rows = []
    rows += _make("fifo", "us", "monthly_unconditional", {"2018": 0.30, "2020": 0.30, "2022": 0.30})
    rows += _make("fifo", "us", "drift_relative_20pct", {"2018": win_c, "2020": 0.29, "2022": 0.29})
    verdict = evaluate_kill_criterion(_summary(rows))
    assert verdict["decision"] == "research_only"
    assert "fifo/drift_relative_20pct" not in verdict["passing_policies"]


def test_just_below_threshold_does_not_win():
    just_under = 0.30 * (1 + KILL_CRITERION_PCT - 0.005)  # below 5%
    rows = []
    rows += _make("fifo", "us", "monthly_unconditional", {"2018": 0.30, "2020": 0.30, "2022": 0.30})
    rows += _make("fifo", "us", "drift_absolute_5pp",
                  {"2018": just_under, "2020": just_under, "2022": just_under})
    verdict = evaluate_kill_criterion(_summary(rows))
    assert verdict["decision"] == "research_only"


def test_none_regime_is_ignored_by_criterion():
    """The kill criterion only looks at the US regime."""
    big = 0.30 * (1 + KILL_CRITERION_PCT + 0.01)
    rows = []
    rows += _make("fifo", "none", "monthly_unconditional", {"2018": 0.30, "2020": 0.30, "2022": 0.30})
    rows += _make("fifo", "none", "drift_absolute_5pp", {"2018": big, "2020": big, "2022": big})
    verdict = evaluate_kill_criterion(_summary(rows))
    # no US rows at all -> nothing to evaluate
    assert verdict["decision"] == "research_only"


def test_selectors_evaluated_independently():
    big = 0.30 * (1 + KILL_CRITERION_PCT + 0.01)
    rows = []
    for sel in ("fifo", "tax_optimal"):
        rows += _make(sel, "us", "monthly_unconditional", {"2018": 0.30, "2020": 0.30, "2022": 0.30})
    rows += _make("fifo", "us", "drift_absolute_5pp", {"2018": big, "2020": big, "2022": big})
    rows += _make("tax_optimal", "us", "drift_absolute_5pp", {"2018": 0.30, "2020": 0.30, "2022": 0.30})
    verdict = evaluate_kill_criterion(_summary(rows))
    assert "fifo/drift_absolute_5pp" in verdict["passing_policies"]
    assert "tax_optimal/drift_absolute_5pp" not in verdict["passing_policies"]
