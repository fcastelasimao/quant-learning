"""
Unit tests for factor_harness/multiple_testing.py

Covers:
- compute_t_stat / compute_p_value
- bhy_correction with known p-values
- correct_ic_pvalues (slow_icir_63 fallback, HLZ threshold floor)
"""
import math

import numpy as np
import pandas as pd
import pytest

from qframe.factor_harness.multiple_testing import (
    bhy_correction,
    bonferroni_correction,
    compute_p_value,
    compute_slow_t_stat,
    compute_t_stat,
    correct_ic_pvalues,
)

_OOS_DAYS = 1762  # default used throughout


# ---------------------------------------------------------------------------
# compute_t_stat
# ---------------------------------------------------------------------------

class TestComputeTStat:
    def test_positive_ic_positive_t(self):
        t = compute_t_stat(ic=0.02, sharpe=0.4, n_oos_days=_OOS_DAYS)
        assert t > 0

    def test_zero_ic_returns_zero(self):
        t = compute_t_stat(ic=0.0, sharpe=0.4, n_oos_days=_OOS_DAYS)
        assert t == 0.0

    def test_proportional_to_sqrt_n(self):
        """Doubling OOS days should multiply t by sqrt(2)."""
        t1 = compute_t_stat(ic=0.02, sharpe=0.4, n_oos_days=1000)
        t2 = compute_t_stat(ic=0.02, sharpe=0.4, n_oos_days=4000)
        assert abs(t2 / t1 - 2.0) < 1e-9

    def test_non_finite_returns_zero(self):
        assert compute_t_stat(ic=float("nan"), sharpe=0.4) == 0.0
        assert compute_t_stat(ic=0.02, sharpe=float("inf")) == 0.0


# ---------------------------------------------------------------------------
# compute_p_value
# ---------------------------------------------------------------------------

class TestComputePValue:
    def test_high_t_stat_low_p(self):
        p = compute_p_value(t_stat=5.0, n_oos_days=_OOS_DAYS)
        assert p < 0.001

    def test_zero_t_stat_p_half(self):
        # t=0 → one-tailed p ≈ 0.5
        p = compute_p_value(t_stat=0.0, n_oos_days=_OOS_DAYS)
        assert abs(p - 0.5) < 0.01

    def test_negative_t_stat_p_above_half(self):
        p = compute_p_value(t_stat=-2.0, n_oos_days=_OOS_DAYS)
        assert p > 0.5

    def test_non_finite_returns_one(self):
        assert compute_p_value(float("nan")) == 1.0


# ---------------------------------------------------------------------------
# bhy_correction
# ---------------------------------------------------------------------------

class TestBHYCorrection:
    def test_empty_array(self):
        result = bhy_correction(np.array([]), alpha=0.05)
        assert len(result) == 0

    def test_all_high_p_not_significant(self):
        """p-values all ≥ 0.5 should never survive BHY at alpha=0.05."""
        p = np.array([0.5, 0.6, 0.7, 0.8])
        sig = bhy_correction(p, alpha=0.05)
        assert not sig.any()

    def test_all_tiny_p_all_significant(self):
        """Very small p-values should all be significant."""
        p = np.array([1e-10, 2e-10, 3e-10])
        sig = bhy_correction(p, alpha=0.05)
        assert sig.all()

    def test_monotone_threshold_logic(self):
        """
        Known-outcome test: with m=4 and harmonic c(4)=1+1/2+1/3+1/4,
        the BHY threshold for rank k is k / (m * c(m)) * alpha.
        Sorted p-values: [0.001, 0.01, 0.03, 0.20] at alpha=0.05, m=4.
        c(4) = 1+0.5+0.333+0.25 = 2.0833
        thresholds: k/(4*2.0833)*0.05 = [0.006, 0.012, 0.018, 0.024]
        p[0]=0.001 ≤ 0.006  → significant
        p[1]=0.01  ≤ 0.012  → significant
        p[2]=0.03  > 0.018  → not significant (and p[3] also not)
        Last significant at sorted index 1 → first 2 are significant.
        """
        c4 = sum(1 / i for i in range(1, 5))  # ≈ 2.0833
        p = np.array([0.001, 0.01, 0.03, 0.20])
        sig = bhy_correction(p, alpha=0.05)
        assert sig[0], "p=0.001 should be significant"
        assert sig[1], "p=0.01 should be significant"
        assert not sig[2], "p=0.03 should not be significant"
        assert not sig[3], "p=0.20 should not be significant"

    def test_result_length_matches_input(self):
        p = np.array([0.01, 0.05, 0.10])
        sig = bhy_correction(p)
        assert len(sig) == 3

    def test_single_element_significant(self):
        sig = bhy_correction(np.array([0.001]), alpha=0.05)
        assert sig[0]

    def test_single_element_not_significant(self):
        sig = bhy_correction(np.array([0.99]), alpha=0.05)
        assert not sig[0]


# ---------------------------------------------------------------------------
# bonferroni_correction
# ---------------------------------------------------------------------------

class TestBonferroniCorrection:
    def test_basic(self):
        p = np.array([0.001, 0.02, 0.10])
        # alpha/m = 0.05/3 = 0.0167
        sig = bonferroni_correction(p, alpha=0.05)
        assert sig[0]
        assert not sig[1]
        assert not sig[2]


# ---------------------------------------------------------------------------
# correct_ic_pvalues — slow_icir_63 fallback and HLZ threshold
# ---------------------------------------------------------------------------

class TestCorrectICPvalues:
    def _make_results(self, **overrides):
        """Build a minimal single-factor result dict."""
        base = {
            "id": 1,
            "factor_name": "test_factor",
            "ic": 0.025,
            "icir": 0.35,
            "sharpe": 0.35,
            "slow_icir_63": None,
            "passed_gate": 1,
        }
        base.update(overrides)
        return [base]

    def test_returns_dataframe(self):
        results = self._make_results()
        df = correct_ic_pvalues(results, n_oos_days=_OOS_DAYS)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_empty_results(self):
        df = correct_ic_pvalues([], n_oos_days=_OOS_DAYS)
        assert df.empty

    def test_negative_ic_excluded(self):
        """Factors with IC ≤ 0 should not appear in the output (one-sided test)."""
        results = self._make_results(ic=-0.01)
        df = correct_ic_pvalues(results, n_oos_days=_OOS_DAYS)
        assert df.empty

    def test_slow_icir_63_zero_uses_fast_formula(self):
        """
        slow_icir_63 = 0.0 must NOT cause the slow formula to be used.
        Python `if 0.0 and ...` would be falsy — the old bug.
        The fix uses `if _s63 is not None and pd.notna(_s63) and _s63 > 0`.
        With slow_icir_63=0.0 the fast formula should be used, producing
        t_stat = (sharpe / sqrt(252)) * sqrt(n_oos_days).
        """
        results_slow = self._make_results(slow_icir_63=0.0, sharpe=0.4)
        results_none = self._make_results(slow_icir_63=None, sharpe=0.4)

        df_slow = correct_ic_pvalues(results_slow, n_oos_days=_OOS_DAYS)
        df_none = correct_ic_pvalues(results_none, n_oos_days=_OOS_DAYS)

        # Both should use the fast formula → same t_stat
        assert abs(df_slow["t_stat"].iloc[0] - df_none["t_stat"].iloc[0]) < 1e-9, (
            f"slow_icir_63=0.0 incorrectly used slow formula: "
            f"{df_slow['t_stat'].iloc[0]:.4f} vs {df_none['t_stat'].iloc[0]:.4f}"
        )

    def test_slow_icir_63_positive_uses_slow_formula(self):
        """
        When slow_icir_63 > 0, the slow t-stat formula should be used.
        t_slow = slow_icir_63 * sqrt(floor(n_oos / 63))
        """
        slow63 = 1.2
        results = self._make_results(slow_icir_63=slow63, sharpe=0.4)
        df = correct_ic_pvalues(results, n_oos_days=_OOS_DAYS)
        n_windows = _OOS_DAYS // 63
        expected_t = slow63 * math.sqrt(n_windows)
        assert abs(df["t_stat"].iloc[0] - expected_t) < 1e-9, (
            f"Expected slow t-stat {expected_t:.4f}, got {df['t_stat'].iloc[0]:.4f}"
        )

    def test_hlz_threshold_floor_for_m1(self):
        """
        With a single factor (m=1), HLZ threshold must be ≥ 3.0.
        Before the fix: sqrt(2*ln(1)) = 0.0 — any t-stat looked significant.
        """
        results = self._make_results(ic=0.001, sharpe=0.05)
        df = correct_ic_pvalues(results, n_oos_days=_OOS_DAYS)
        assert df["hlz_t_threshold"].iloc[0] >= 3.0, (
            f"HLZ threshold for m=1 should be ≥ 3.0, "
            f"got {df['hlz_t_threshold'].iloc[0]}"
        )

    def test_hlz_threshold_grows_with_m(self):
        """
        For m ≥ 2, HLZ threshold should be max(3.0, sqrt(2*ln(m))).
        At m=100: sqrt(2*ln(100)) ≈ 3.03 — just above the floor.
        At m=1000: sqrt(2*ln(1000)) ≈ 3.72.
        """
        many_results = [
            {
                "id": i,
                "factor_name": f"f{i}",
                "ic": 0.02 + i * 0.001,
                "icir": 0.3,
                "sharpe": 0.3,
                "slow_icir_63": None,
                "passed_gate": 1,
            }
            for i in range(20)
        ]
        df = correct_ic_pvalues(many_results, n_oos_days=_OOS_DAYS)
        m = len(df)
        expected_hlz = max(3.0, math.sqrt(2 * math.log(m)))
        assert abs(df["hlz_t_threshold"].iloc[0] - round(expected_hlz, 3)) < 1e-3

    def test_required_output_columns(self):
        results = self._make_results()
        df = correct_ic_pvalues(results, n_oos_days=_OOS_DAYS)
        required = {"factor_name", "ic", "t_stat", "p_raw",
                    "bhy_significant", "hlz_sig", "bonferroni_sig"}
        assert required.issubset(df.columns), (
            f"Missing columns: {required - set(df.columns)}"
        )

    def test_sorted_by_ic_descending(self):
        """Output DataFrame should be sorted by IC descending."""
        many = [
            {"id": i, "factor_name": f"f{i}", "ic": float(i) * 0.01 + 0.001,
             "icir": 0.3, "sharpe": 0.3, "slow_icir_63": None, "passed_gate": 1}
            for i in range(5)
        ]
        df = correct_ic_pvalues(many, n_oos_days=_OOS_DAYS)
        assert list(df["ic"]) == sorted(df["ic"], reverse=True)
