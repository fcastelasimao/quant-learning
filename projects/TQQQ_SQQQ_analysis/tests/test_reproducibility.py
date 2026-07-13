"""Reproducibility tests: headline numbers quoted in findings docs must match
the CSVs they were drawn from.

These catch the case where a refactor changes a number but the prose doesn't.
Test only the load-bearing headline values, not every cell.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def _isclose(a: float, b: float, abs_tol: float = 1e-3) -> bool:
    return abs(a - b) < abs_tol


# --------------------------------------------------------------------------- #
# Item 04 — model headline AUCs
# --------------------------------------------------------------------------- #

def test_item_04_headline_tqqq(research_dir):
    p = research_dir / "04_loss_region_models" / "headline_model_perf_TQQQ.csv"
    if not p.exists():
        pytest.skip("item 04 headline missing")
    df = pd.read_csv(p)
    row = df.iloc[0]
    # SYNTHESIS / findings quote 0.670 / 0.563 / 0.593 for tree_severe / tree_loser / l1_loser
    assert _isclose(row["tree_severe_oos_auc"], 0.670, abs_tol=0.01)
    assert _isclose(row["tree_loser_oos_auc"], 0.563, abs_tol=0.01)
    assert _isclose(row["logit_loser_oos_auc"], 0.593, abs_tol=0.01)


def test_item_04_headline_sqqq(research_dir):
    p = research_dir / "04_loss_region_models" / "headline_model_perf_SQQQ.csv"
    if not p.exists():
        pytest.skip("item 04 headline missing")
    df = pd.read_csv(p)
    row = df.iloc[0]
    assert _isclose(row["tree_severe_oos_auc"], 0.616, abs_tol=0.01)


def test_item_04_gbm_r2_is_negative_median(research_dir):
    """SYNTHESIS claim (corrected after first run of this test): GBM on pnl_pct
    has *predominantly* negative OOS R² across WF years. TQQQ 2023 was a fluke
    at +0.013; every other window across both symbols is solidly negative.
    Test asserts median is negative and at least 7 of 9 windows are negative.
    """
    for sym in ("TQQQ", "SQQQ"):
        p = research_dir / "04_loss_region_models" / f"wf_model_eval_{sym}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        n_neg = int((df["gbm_r2"] < 0).sum())
        assert df["gbm_r2"].median() < 0, f"{sym} median GBM R² >= 0"
        assert n_neg >= 7, (
            f"{sym} only {n_neg} of {len(df)} WF years have negative GBM R² — "
            "the 'GBM is unusable' SYNTHESIS finding may have broken"
        )


# --------------------------------------------------------------------------- #
# Item 05 / 13 — equity metrics
# --------------------------------------------------------------------------- #

def test_item_05_tqqq_constant_notional_sharpe(research_dir):
    p = research_dir / "05_capital_normalization" / "metrics_constant_notional.csv"
    if not p.exists():
        pytest.skip("item 05 metrics missing")
    df = pd.read_csv(p)
    full = df[(df["symbol"] == "TQQQ") & (df["scope"] == "full_history")]
    assert len(full) == 1
    sharpe = float(full.iloc[0]["sharpe_daily"])
    # SYNTHESIS / findings quote 4.00 (TQQQ constant-notional Sharpe)
    assert _isclose(sharpe, 4.00, abs_tol=0.05)


def test_item_13_within_csv_compounded_cagr_tqqq(research_dir):
    p = research_dir / "13_within_csv_compounding" / "compounding_compare.csv"
    if not p.exists():
        pytest.skip("item 13 missing")
    df = pd.read_csv(p)
    tqqq = df[df["symbol"] == "TQQQ"].iloc[0]
    # SYNTHESIS quotes 59% CAGR
    assert _isclose(float(tqqq["geo_mean_cagr"]), 0.59, abs_tol=0.03)


# --------------------------------------------------------------------------- #
# Item 06 — context enrichment AUC jumps
# --------------------------------------------------------------------------- #

def test_item_06_l1_logit_tqqq_severe_jumps(research_dir):
    p = research_dir / "06_context_enrichment" / "headline_auc_compare.csv"
    if not p.exists():
        pytest.skip("item 06 headline missing")
    df = pd.read_csv(p)
    tqqq = df[df["symbol"] == "TQQQ"]
    base = tqqq[tqqq["feature_set"] == "curated_12_only"].iloc[0]
    enr = tqqq[tqqq["feature_set"] == "curated_plus_context"].iloc[0]
    # Strict-prior context headline: context still helps L1 severe modestly,
    # but the old same-day-context 0.716 result was not causal.
    assert _isclose(float(base["auc_l1_severe"]), 0.593, abs_tol=0.02)
    assert _isclose(float(enr["auc_l1_severe"]), 0.611, abs_tol=0.02)


# --------------------------------------------------------------------------- #
# Item 07 — severity threshold sweep
# --------------------------------------------------------------------------- #

def test_item_07_predictability_rises_with_severity(research_dir):
    """SYNTHESIS table: TQQQ WF-AUC at thresholds {-1, -2}: {0.65, 0.91}."""
    p = research_dir / "07_severity_threshold_sweep" / "severity_sweep.csv"
    if not p.exists():
        pytest.skip("item 07 sweep missing")
    df = pd.read_csv(p)
    tqqq_loss = df[(df["symbol"] == "TQQQ") & (df["side"] == "loss") & (df["model"] == "tree")]
    by_thr = tqqq_loss.set_index("threshold")["wf_auc_median"]
    assert by_thr[-1.0] < by_thr[-2.0] < by_thr[-3.0], (
        "TQQQ predictability should rise monotonically as threshold becomes more extreme"
    )


# --------------------------------------------------------------------------- #
# Item 08 — focus rule bootstrap
# --------------------------------------------------------------------------- #

def test_item_08_focus_rule_block_bootstrap_positive(research_dir):
    """SYNTHESIS: block-bootstrap CI is entirely positive."""
    p = research_dir / "08_focus_rule_recheck" / "focus_rule_bootstrap.csv"
    if not p.exists():
        pytest.skip("item 08 bootstrap missing")
    df = pd.read_csv(p)
    # Take the regime_labeled row (item 08's primary scope)
    row = df[df["scope"] == "regime_labeled"].iloc[0]
    assert float(row["block_boot_ci_lo"]) > 0, (
        "Block-bootstrap CI lower bound should be > 0 (SYNTHESIS headline)"
    )


# --------------------------------------------------------------------------- #
# Item 11 — TQQQ severe-in-sideways meta-rule
# --------------------------------------------------------------------------- #

def test_item_11_tqqq_severe_sideways_positive(research_dir):
    p = research_dir / "11_regime_conditional_rules" / "regime_conditional_meta_rules.csv"
    if not p.exists():
        pytest.skip("item 11 meta-rules missing")
    df = pd.read_csv(p)
    headline = df[(df["symbol"] == "TQQQ") & (df["regime"] == "sideways_lowvol")
                  & (df["parent_rule_name"].str.contains("severe"))]
    assert len(headline) >= 1, "TQQQ severe-rule x sideways_lowvol not found"
    # SYNTHESIS: 10 flagged, 100% precision, +10.6 pp
    best = headline.sort_values("research_oos_net_pnl_impact", ascending=False).iloc[0]
    assert best["research_oos_n_flagged"] >= 8
    assert float(best["research_oos_precision"]) >= 0.9
    assert float(best["research_oos_net_pnl_impact"]) >= 8.0


# --------------------------------------------------------------------------- #
# Item 12 — original sizing simulation
# --------------------------------------------------------------------------- #

def test_item_12_linear_skip_cuts_drawdown(research_dir):
    p = research_dir / "12_continuous_sizing_simulation" / "sizing_simulation_summary.csv"
    if not p.exists():
        pytest.skip("item 12 summary missing")
    df = pd.read_csv(p)
    df = df[df["scope"].isna() | (df["scope"] != "yearly")] if "scope" in df.columns else df
    # Filter to the 1pct target for comparison with original baseline
    if "target" in df.columns:
        df = df[df["target"] == "1pct"]
    for sym in ("TQQQ", "SQQQ"):
        base = df[(df["symbol"] == sym) & (df["sizing"] == "baseline_full")].iloc[0]
        linear = df[(df["symbol"] == sym) & (df["sizing"] == "linear_skip")].iloc[0]
        assert float(linear["max_drawdown"]) > float(base["max_drawdown"]), (
            f"{sym} linear sizing should cut MaxDD vs baseline (negative numbers; less negative = better)"
        )


# --------------------------------------------------------------------------- #
# Item 17 — enriched sizing Pareto improvement
# --------------------------------------------------------------------------- #

def test_item_17_enriched_sizing_pareto_improvement(research_dir):
    p = research_dir / "17_sizing_with_enriched_features" / "sizing_enriched_summary.csv"
    if not p.exists():
        pytest.skip("item 17 summary missing")
    df = pd.read_csv(p)
    tqqq_base = df[(df["symbol"] == "TQQQ") & (df["sizing"] == "baseline_full")].iloc[0]
    tqqq_enr = df[(df["symbol"] == "TQQQ") &
                  (df["sizing"].str.contains("linear_skip_enriched_1pct"))].iloc[0]
    assert float(tqqq_enr["sharpe_daily"]) > float(tqqq_base["sharpe_daily"])
    assert float(tqqq_enr["max_drawdown"]) > float(tqqq_base["max_drawdown"])

    sqqq_base = df[(df["symbol"] == "SQQQ") & (df["sizing"] == "baseline_full")].iloc[0]
    sqqq_enr = df[(df["symbol"] == "SQQQ") &
                  (df["sizing"].str.contains("linear_skip_enriched_1pct"))].iloc[0]
    # Strict-prior validation: SQQQ drawdown improves, but Sharpe no longer
    # exceeds baseline. This prevents the old non-causal Pareto claim from
    # silently returning.
    assert float(sqqq_enr["max_drawdown"]) > float(sqqq_base["max_drawdown"])
    assert float(sqqq_enr["sharpe_daily"]) <= float(sqqq_base["sharpe_daily"])


# --------------------------------------------------------------------------- #
# Item 18 — combined strategy Sharpe is reasonable
# --------------------------------------------------------------------------- #

def test_item_18_combined_sharpe_reasonable(research_dir):
    """Item 18 combined scenario Sharpe is in a plausible range [2.0, 6.0]."""
    p = research_dir / "18_combined_strategy" / "combined_strategy_summary.csv"
    if not p.exists():
        pytest.skip("item 18 summary missing")
    df = pd.read_csv(p)
    for sym in ("TQQQ", "SQQQ"):
        combined = df[(df["symbol"] == sym) & (df["scenario"] == "combined")]
        assert len(combined) == 1, f"{sym} combined row not found"
        sharpe = float(combined.iloc[0]["sharpe_daily"])
        assert 2.0 <= sharpe <= 6.0, (
            f"{sym} combined Sharpe = {sharpe:.3f}, expected in [2.0, 6.0]"
        )


def test_item_18_combined_beats_rules_only(research_dir):
    """Combined scenario Sharpe > rules_only Sharpe for both symbols."""
    p = research_dir / "18_combined_strategy" / "combined_strategy_summary.csv"
    if not p.exists():
        pytest.skip("item 18 summary missing")
    df = pd.read_csv(p)
    for sym in ("TQQQ", "SQQQ"):
        combined = df[(df["symbol"] == sym) & (df["scenario"] == "combined")].iloc[0]
        rules = df[(df["symbol"] == sym) & (df["scenario"] == "rules_only")].iloc[0]
        assert float(combined["sharpe_daily"]) >= float(rules["sharpe_daily"]), (
            f"{sym}: combined Sharpe should be >= rules_only Sharpe"
        )


# --------------------------------------------------------------------------- #
# Item 19 — SQQQ target grid completeness
# --------------------------------------------------------------------------- #

def test_item_19_grid_has_all_targets(research_dir):
    """Item 19 grid contains all 3 targets × at least 4 sizing functions."""
    p = research_dir / "19_sqqq_target_exploration" / "sqqq_target_sizing_grid.csv"
    if not p.exists():
        pytest.skip("item 19 grid missing")
    df = pd.read_csv(p)
    non_baseline = df[df["target"] != "baseline"]
    targets_found = set(non_baseline["target"].unique())
    assert {"1pct", "1p5pct", "2pct"}.issubset(targets_found), (
        f"Missing targets in item 19 grid: {{'1pct','1p5pct','2pct'}} vs {targets_found}"
    )
    assert len(non_baseline) >= 12, (
        f"Expected >= 12 non-baseline rows (3 targets × 4+ sizing), got {len(non_baseline)}"
    )


def test_item_19_sqqq_2pct_sqrt_beats_baseline_sharpe(research_dir):
    """SQQQ -2% sqrt_skip Sharpe > baseline Sharpe (item 19 headline)."""
    p = research_dir / "19_sqqq_target_exploration" / "sqqq_target_sizing_grid.csv"
    if not p.exists():
        pytest.skip("item 19 grid missing")
    df = pd.read_csv(p)
    base = df[df["target"] == "baseline"]
    sq = df[(df["target"] == "2pct") & (df["sizing"] == "sqrt_skip")]
    if base.empty or sq.empty:
        pytest.skip("baseline or 2pct/sqrt_skip row missing")
    assert float(sq.iloc[0]["sharpe_daily"]) > float(base.iloc[0]["sharpe_daily"]), (
        "SQQQ -2% sqrt_skip should beat baseline Sharpe (item 19 finding)"
    )


# --------------------------------------------------------------------------- #
# Item 20 — own-symbol AUC > cross-symbol AUC
# --------------------------------------------------------------------------- #

def test_item_20_own_symbol_beats_cross(research_dir):
    """Own-symbol AUC > cross-symbol AUC for both scored symbols (item 20)."""
    p = research_dir / "20_cross_symbol_signal" / "cross_symbol_auc.csv"
    if not p.exists():
        pytest.skip("item 20 AUC summary missing")
    df = pd.read_csv(p)
    for score_sym in ("TQQQ", "SQQQ"):
        sub = df[df["score_symbol"] == score_sym]
        own = sub[sub["kind"] == "own_symbol"]["agg_oos_auc"]
        cross = sub[sub["kind"] == "cross_symbol"]["agg_oos_auc"]
        if own.empty or cross.empty:
            continue
        assert float(own.iloc[0]) > float(cross.iloc[0]), (
            f"{score_sym}: own-symbol AUC {own.iloc[0]:.3f} should exceed "
            f"cross-symbol AUC {cross.iloc[0]:.3f} (item 20 finding)"
        )
