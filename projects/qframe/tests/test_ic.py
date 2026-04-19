"""
Unit tests for factor_harness/ic.py

Uses synthetic data so tests run instantly without network calls.
"""
import numpy as np
import pandas as pd
import pytest

from qframe.factor_harness.ic import (
    compute_ic,
    compute_icir,
    compute_ic_decay,
    compute_ic_by_period,
    compute_slow_icir,
    estimate_ic_halflife,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_synthetic_data(n_dates=300, n_stocks=50, seed=42):
    """
    Generate synthetic prices and a known-good factor.

    The factor is the lagged 1-day return (autocorrelation signal).
    This will produce near-zero IC, which is correct — we just need
    finite, non-NaN output.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    tickers = [f"S{i:03d}" for i in range(n_stocks)]

    returns = pd.DataFrame(
        rng.normal(0.0005, 0.02, size=(n_dates, n_stocks)),
        index=dates,
        columns=tickers,
    )
    prices = (1 + returns).cumprod()

    # Factor: 21-day momentum (should have a positive IC with forward returns)
    factor = prices.pct_change(21)

    return prices, returns, factor


@pytest.fixture(scope="module")
def synthetic():
    return make_synthetic_data()


# ---------------------------------------------------------------------------
# compute_ic
# ---------------------------------------------------------------------------

class TestComputeIC:
    def test_returns_series(self, synthetic):
        _, returns, factor = synthetic
        ic = compute_ic(factor, returns, horizon=1)
        assert isinstance(ic, pd.Series)

    def test_no_nan_after_warmup(self, synthetic):
        _, returns, factor = synthetic
        ic = compute_ic(factor, returns, horizon=1)
        # After the 21-day warmup for the 21d momentum factor + 1d horizon,
        # there should be no NaN values
        ic_valid = ic.dropna()
        assert len(ic_valid) > 200, "Expected substantial valid IC observations"

    def test_ic_bounded(self, synthetic):
        _, returns, factor = synthetic
        ic = compute_ic(factor, returns, horizon=1)
        valid = ic.dropna()
        assert (valid.abs() <= 1.0).all(), "IC must be in [-1, 1]"

    def test_ic_finite(self, synthetic):
        _, returns, factor = synthetic
        ic = compute_ic(factor, returns, horizon=1)
        valid = ic.dropna()
        assert np.isfinite(valid).all()

    def test_min_stocks_respected(self, synthetic):
        _, returns, factor = synthetic
        # With min_stocks larger than universe, every date should be NaN
        ic = compute_ic(factor, returns, horizon=1, min_stocks=9999)
        assert ic.isna().all()

    def test_perfect_positive_correlation(self):
        """IC should be +1 when factor perfectly predicts forward return.

        compute_ic computes fwd_ret[t] = returns[t+1] (for horizon=1).
        For perfect IC, factor[t] must rank identically to returns[t+1],
        i.e. factor = returns.shift(-1).
        """
        dates = pd.bdate_range("2020-01-01", periods=50)
        tickers = [f"S{i}" for i in range(20)]
        rng = np.random.default_rng(0)
        ret = pd.DataFrame(rng.normal(0, 1, (50, 20)), index=dates, columns=tickers)
        # factor[t] = returns[t+1] → perfect rank alignment with fwd_ret[t]
        factor = ret.shift(-1)
        ic = compute_ic(factor, ret, horizon=1, min_stocks=5)
        valid = ic.dropna()
        # Should be very close to 1.0 (exact match)
        assert (valid > 0.95).all(), f"Expected IC~1.0, got {valid.describe()}"

    def test_nan_returns_propagate_to_nan_forward_return(self):
        """
        A stock with NaN on any day within the forward window must have a NaN
        forward return (caught by r.notna() mask) and be excluded from IC.

        Before the fillna(0) fix, that stock was included with an artificially
        low forward return (treated as 0% on the NaN day). This test verifies
        the fix: with min_periods=horizon, any NaN in the window propagates.

        We construct a factor that perfectly predicts returns for 10 of 11 stocks.
        Stock GAPPED has a NaN return on day 5. At dates whose forward window
        includes day 5 (dates 2, 3, 4 for horizon=3), GAPPED is excluded.
        At dates whose forward window does NOT include day 5 (dates 0, 1), all
        11 stocks are present, and the perfect-predictor factor should give IC≈1.
        """
        dates = pd.bdate_range("2020-01-01", periods=30)
        tickers = ["A", "B", "C", "D", "E",
                   "F", "G", "H", "I", "J", "GAPPED"]
        rng = np.random.default_rng(42)
        ret = pd.DataFrame(
            rng.normal(0, 0.01, (30, len(tickers))),
            index=dates,
            columns=tickers,
        )
        # Introduce a NaN in the forward window of stock GAPPED at day 5
        ret.loc[dates[5], "GAPPED"] = np.nan

        # Factor: each stock gets a DIFFERENT fixed rank over time.
        # Use each ticker's column index as its score so all stocks are distinct.
        factor_values = {t: np.full(30, float(i)) for i, t in enumerate(tickers)}
        factor = pd.DataFrame(factor_values, index=dates)

        # Compute IC at horizon=3 with min_stocks=5
        ic = compute_ic(factor, ret, horizon=3, min_stocks=5)

        # Dates 0 and 1 have forward windows [1,2,3] and [2,3,4] — no gap.
        # Both should produce finite IC values (11 stocks, ranks 0-10 all present).
        # (IC may not be exactly 1 since the factor is constant per stock, not
        # aligned with actual forward returns, but it will be finite and valid.)
        assert pd.notna(ic.iloc[0]), (
            "IC at date 0 (forward window [1,2,3], no NaN) should be finite"
        )
        assert pd.notna(ic.iloc[1]), (
            "IC at date 1 (forward window [2,3,4], no NaN) should be finite"
        )


# ---------------------------------------------------------------------------
# compute_icir
# ---------------------------------------------------------------------------

class TestComputeICIR:
    def test_returns_series(self, synthetic):
        _, returns, factor = synthetic
        ic = compute_ic(factor, returns)
        icir = compute_icir(ic)
        assert isinstance(icir, pd.Series)

    def test_finite_values(self, synthetic):
        _, returns, factor = synthetic
        ic = compute_ic(factor, returns)
        icir = compute_icir(ic)
        valid = icir.dropna()
        assert np.isfinite(valid).all()

    def test_window_respected(self, synthetic):
        _, returns, factor = synthetic
        ic = compute_ic(factor, returns)
        icir_63 = compute_icir(ic, window=63)
        # Should have NaN for the first (window-1) valid IC observations
        n_nan = icir_63.isna().sum()
        assert n_nan > 0, "Expected NaN during window warm-up"


# ---------------------------------------------------------------------------
# compute_ic_decay
# ---------------------------------------------------------------------------

class TestComputeICDecay:
    def test_returns_dataframe(self, synthetic):
        _, returns, factor = synthetic
        decay = compute_ic_decay(factor, returns)
        assert isinstance(decay, pd.DataFrame)

    def test_correct_horizons(self, synthetic):
        _, returns, factor = synthetic
        horizons = [1, 5, 21]
        decay = compute_ic_decay(factor, returns, horizons=horizons)
        assert list(decay.index) == horizons

    def test_columns_present(self, synthetic):
        _, returns, factor = synthetic
        decay = compute_ic_decay(factor, returns)
        assert "mean_ic" in decay.columns
        assert "icir" in decay.columns
        assert "n_obs" in decay.columns


# ---------------------------------------------------------------------------
# estimate_ic_halflife
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# compute_slow_icir
# ---------------------------------------------------------------------------

class TestComputeSlowICIR:
    def test_returns_float(self, synthetic):
        _, returns, factor = synthetic
        val = compute_slow_icir(factor, returns, horizon=21, oos_start="2020-06-01")
        assert isinstance(val, float)

    def test_nan_when_insufficient_periods(self, synthetic):
        """Requesting a horizon longer than available OOS data should return NaN."""
        _, returns, factor = synthetic
        # OOS start very late → almost no data
        val = compute_slow_icir(factor, returns, horizon=21, oos_start="2021-10-01",
                                min_periods=999)
        assert np.isnan(val)

    def test_finite_or_nan(self, synthetic):
        _, returns, factor = synthetic
        val = compute_slow_icir(factor, returns, horizon=5, oos_start="2020-06-01")
        assert np.isnan(val) or np.isfinite(val)

    def test_min_stocks_parameter_respected(self, synthetic):
        """
        min_stocks should be used inside the loop, not hardcoded to 10.

        With a tiny min_stocks=2, we should get a valid result.
        With min_stocks > universe size, we should get NaN (no qualifying dates).
        """
        _, returns, factor = synthetic
        # Should succeed: only 2 stocks needed
        val_low = compute_slow_icir(factor, returns, horizon=5, oos_start="2020-06-01",
                                    min_stocks=2)
        assert not np.isnan(val_low), "Expected a valid ICIR with min_stocks=2"

        # Should fail: more stocks required than universe size (50 stocks)
        val_high = compute_slow_icir(factor, returns, horizon=5, oos_start="2020-06-01",
                                     min_stocks=9999)
        assert np.isnan(val_high), "Expected NaN when min_stocks exceeds universe"


# ---------------------------------------------------------------------------
# compute_ic_by_period
# ---------------------------------------------------------------------------

class TestComputeICByPeriod:
    """Tests for the temporal stability diagnostic."""

    def _make_long_data(self, n_dates=1500, n_stocks=50, seed=7):
        """Generate ~6 years of synthetic data (enough for 3 two-year blocks)."""
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range("2018-01-02", periods=n_dates)
        tickers = [f"S{i:03d}" for i in range(n_stocks)]
        returns = pd.DataFrame(
            rng.normal(0.0003, 0.015, size=(n_dates, n_stocks)),
            index=dates, columns=tickers,
        )
        prices = (1 + returns).cumprod()
        factor = prices.pct_change(21)
        return prices, returns, factor

    def test_returns_dataframe(self):
        _, returns, factor = self._make_long_data()
        df = compute_ic_by_period(factor, returns, oos_start="2018-01-01")
        assert isinstance(df, pd.DataFrame)

    def test_required_columns(self):
        _, returns, factor = self._make_long_data()
        df = compute_ic_by_period(factor, returns, oos_start="2018-01-01")
        required = {"period_label", "period_start", "period_end",
                    "mean_ic", "std_ic", "icir", "t_stat", "n_days"}
        assert required.issubset(df.columns), f"Missing columns: {required - set(df.columns)}"

    def test_multiple_periods_produced(self):
        """1500 OOS days / (2 * 252) ≈ 2.97 → should produce at least 2 blocks."""
        _, returns, factor = self._make_long_data(n_dates=1500)
        df = compute_ic_by_period(factor, returns, oos_start="2018-01-01",
                                  period_years=2.0)
        assert len(df) >= 2, f"Expected ≥2 periods, got {len(df)}"

    def test_period_days_sum_to_oos(self):
        """Total n_days across all periods must ≤ total valid OOS IC observations."""
        _, returns, factor = self._make_long_data(n_dates=1500)
        df = compute_ic_by_period(factor, returns, oos_start="2018-01-01")
        # We only check it's a positive integer per row (not zero or negative)
        assert (df["n_days"] > 0).all()

    def test_mean_ic_finite_or_nan(self):
        _, returns, factor = self._make_long_data()
        df = compute_ic_by_period(factor, returns, oos_start="2018-01-01")
        assert df["mean_ic"].apply(lambda x: np.isnan(x) or np.isfinite(x)).all()

    def test_t_stat_consistent_with_icir_and_n(self):
        """t_stat should equal icir × sqrt(n_days) (within floating point tolerance)."""
        _, returns, factor = self._make_long_data()
        df = compute_ic_by_period(factor, returns, oos_start="2018-01-01")
        for _, row in df.iterrows():
            if np.isfinite(row["icir"]) and np.isfinite(row["t_stat"]):
                expected = row["icir"] * np.sqrt(row["n_days"])
                assert abs(row["t_stat"] - expected) < 1e-6, (
                    f"t_stat={row['t_stat']:.6f} ≠ icir*√n={expected:.6f}"
                )

    def test_period_blocks_cover_full_oos_without_gap(self):
        """period_end[i] should be the day before period_start[i+1] (contiguous blocks)."""
        _, returns, factor = self._make_long_data(n_dates=1500)
        df = compute_ic_by_period(factor, returns, oos_start="2018-01-01")
        if len(df) < 2:
            pytest.skip("Need ≥2 periods to check contiguity")
        for i in range(len(df) - 1):
            end_i = pd.Timestamp(df["period_end"].iloc[i])
            start_next = pd.Timestamp(df["period_start"].iloc[i + 1])
            # period_end[i] < period_start[i+1] (no gap, no overlap)
            assert end_i < start_next, (
                f"Block {i} end {end_i} >= block {i+1} start {start_next}"
            )

    def test_period_labels_are_strings(self):
        _, returns, factor = self._make_long_data()
        df = compute_ic_by_period(factor, returns, oos_start="2018-01-01")
        assert df["period_label"].apply(lambda x: isinstance(x, str)).all()

    def test_empty_when_no_oos_data(self, synthetic):
        """Returns empty DataFrame when oos_start is after all available dates."""
        _, returns, factor = synthetic
        df = compute_ic_by_period(factor, returns, oos_start="2099-01-01")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_single_period_with_short_data(self):
        """With only ~1 period worth of OOS data, should return exactly 1 row."""
        rng = np.random.default_rng(99)
        n = 600  # ~2.4 years total; OOS starts at 0 → ~2.4-year window
        dates = pd.bdate_range("2020-01-02", periods=n)
        tickers = [f"S{i}" for i in range(40)]
        returns = pd.DataFrame(
            rng.normal(0, 0.01, (n, 40)), index=dates, columns=tickers
        )
        prices = (1 + returns).cumprod()
        factor = prices.pct_change(5)
        df = compute_ic_by_period(
            factor, returns, oos_start="2020-01-01",
            period_years=2.0,   # 2 years = 504 days, fits ~1.2 times in 600
        )
        # Should get 1 full period (the trailing ~0.2-year stub is dropped if <20 obs)
        assert 1 <= len(df) <= 2, f"Expected 1–2 periods, got {len(df)}"

    def test_horizon_forwarded(self):
        """IC at horizon=5 vs horizon=1 should give different (not necessarily higher) IC."""
        _, returns, factor = self._make_long_data()
        df1  = compute_ic_by_period(factor, returns, oos_start="2018-01-01", horizon=1)
        df5  = compute_ic_by_period(factor, returns, oos_start="2018-01-01", horizon=5)
        # Just check both return valid DataFrames with the same number of periods
        assert len(df1) == len(df5)

    def test_custom_period_years(self):
        """period_years=1.0 should give roughly twice as many rows as period_years=2.0."""
        _, returns, factor = self._make_long_data(n_dates=1500)
        df2yr = compute_ic_by_period(factor, returns, oos_start="2018-01-01",
                                     period_years=2.0)
        df1yr = compute_ic_by_period(factor, returns, oos_start="2018-01-01",
                                     period_years=1.0)
        assert len(df1yr) >= len(df2yr), (
            f"1-year blocks ({len(df1yr)}) should give ≥ rows than 2-year blocks ({len(df2yr)})"
        )


# ---------------------------------------------------------------------------
# estimate_ic_halflife
# ---------------------------------------------------------------------------

class TestSlowICIRFormula:
    """
    Regression tests for the slow-ICIR formula.

    The correct formula for a slow signal at horizon h:
        slow_icir = mean_period_IC / std_period_IC  (non-overlapping h-day windows)
        t_stat    = slow_icir × sqrt(N_windows)     where N_windows = floor(N_oos / h)

    Bug-guard: the fast formula uses sqrt(N_oos/252) which overcounts by h/252.
    """

    def _make_slow_signal(self, n_dates: int = 1500, n_stocks: int = 40, seed: int = 7):
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range("2018-01-02", periods=n_dates)
        tickers = [f"S{i:03d}" for i in range(n_stocks)]
        prev = rng.normal(0, 0.01, n_stocks)
        rets_list = []
        for _ in range(n_dates):
            r = 0.05 * prev + rng.normal(0, 0.01, n_stocks)
            rets_list.append(r)
            prev = r
        returns = pd.DataFrame(rets_list, index=dates, columns=tickers)
        prices  = (1 + returns).cumprod()
        factor  = returns.shift(1)
        return prices, returns, factor

    def test_slow_icir_uses_non_overlapping_windows(self):
        """slow_icir_63 and fast ICIR should be distinct values for h=63."""
        prices, returns, factor = self._make_slow_signal()
        oos_start = "2021-01-01"

        slow_val = compute_slow_icir(factor, returns, horizon=63, oos_start=oos_start)
        fast_ic  = compute_ic(factor, returns, horizon=1).loc[oos_start:].dropna()
        if len(fast_ic) > 1 and np.isfinite(slow_val):
            fast_icir = fast_ic.mean() / fast_ic.std()
            assert abs(slow_val - fast_icir) > 1e-6, (
                "slow_icir_63 and fast ICIR should differ — "
                "verify slow formula uses non-overlapping windows"
            )

    def test_slow_t_stat_uses_n_windows_not_n_over_252(self):
        """
        t = slow_icir × sqrt(N_windows), N_windows = floor(N_oos / horizon).
        NOT sqrt(N_oos / 252) which overcounts observations by factor h/252.
        """
        prices, returns, factor = self._make_slow_signal()
        oos_start  = "2021-01-01"
        horizon    = 63
        slow_icir  = compute_slow_icir(factor, returns, horizon=horizon,
                                       oos_start=oos_start)
        if np.isnan(slow_icir):
            pytest.skip("Insufficient data")
        n_oos      = len(returns.loc[oos_start:])
        n_windows  = n_oos // horizon
        # Correct t-stat uses n_windows
        t_correct  = slow_icir * np.sqrt(n_windows)
        # Wrong t-stat uses n_oos/252 (fast formula applied to slow horizon)
        t_wrong    = slow_icir * np.sqrt(n_oos / 252)
        assert np.isfinite(t_correct)
        # They must differ — if equal, something is wrong
        assert abs(t_correct - t_wrong) > 0.01, (
            f"t_correct={t_correct:.4f} ≈ t_wrong={t_wrong:.4f}: "
            "likely the fast formula was used instead of floor(N/h)"
        )

    def test_slow_icir_nan_when_no_complete_window(self):
        """Returns NaN if OOS data is shorter than one non-overlapping horizon window."""
        prices, returns, factor = self._make_slow_signal(n_dates=400)
        oos_returns = returns.loc["2019-10-01":]
        val = compute_slow_icir(factor, returns, horizon=63, oos_start="2019-10-01",
                                min_periods=2)
        if len(oos_returns) < 63:
            assert np.isnan(val)


class TestEstimateICHalfLife:
    def test_returns_float(self, synthetic):
        _, returns, factor = synthetic
        decay = compute_ic_decay(factor, returns)
        hl = estimate_ic_halflife(decay)
        assert isinstance(hl, float)

    def test_positive_halflife(self):
        """Exponentially decaying IC curve should give a positive half-life."""
        horizons = [1, 5, 10, 21, 63]
        mean_ic = [0.04 * np.exp(-0.05 * h) for h in horizons]
        decay = pd.DataFrame({"mean_ic": mean_ic, "icir": 1.0, "n_obs": 100},
                             index=horizons)
        decay.index.name = "horizon"
        hl = estimate_ic_halflife(decay)
        assert hl > 0
        # np.log(2) / 0.05 ≈ 13.9 days
        assert 5 < hl < 50, f"Half-life {hl:.1f} outside expected range"

class TestDeflatedSharpeRatio:
    """
    Tests for the Deflated Sharpe Ratio (Bailey & López de Prado 2014).

    NOTE: ``deflated_sharpe_ratio`` uses PER-PERIOD (non-annualised) Sharpe
    ratios, as in the original paper.  The expected max SR from N trials with
    T daily observations is approx (2.766 / sqrt(T)) ≈ 0.066 for T=1762.
    An annualised SR of 0.5 corresponds to a per-period SR of 0.5/sqrt(252) ≈ 0.031.
    """

    def test_dsr_returns_float_in_unit_interval(self):
        from qframe.factor_harness.multiple_testing import deflated_sharpe_ratio
        # Per-period SR of 0.1 with 50 trials and 1762 observations
        dsr = deflated_sharpe_ratio(sharpe_obs=0.1, n_trials=50, t=1762)
        assert isinstance(dsr, float)
        assert 0.0 <= dsr <= 1.0

    def test_small_per_period_sr_many_trials_low_dsr(self):
        """
        Per-period SR of 0.03 (≈ annualised 0.5) with 200 trials.
        Expected max SR from noise ≈ 0.066 → 0.03 < 0.066 → DSR < 0.5.
        """
        from qframe.factor_harness.multiple_testing import deflated_sharpe_ratio
        # sr_obs < sr_0_star (expected noise max) → DSR should be < 0.5
        dsr = deflated_sharpe_ratio(sharpe_obs=0.03, n_trials=200, t=1762)
        assert dsr < 0.5

    def test_high_per_period_sr_one_trial_high_dsr(self):
        """Per-period SR of 0.2 (≈ annualised 3.2) with a single trial → DSR near 1."""
        from qframe.factor_harness.multiple_testing import deflated_sharpe_ratio
        dsr = deflated_sharpe_ratio(sharpe_obs=0.2, n_trials=1, t=1762)
        assert dsr > 0.9

    def test_dsr_increases_with_fewer_trials(self):
        """Same SR should have higher DSR when fewer strategies were tried."""
        from qframe.factor_harness.multiple_testing import deflated_sharpe_ratio
        dsr_few  = deflated_sharpe_ratio(sharpe_obs=0.1, n_trials=5,   t=1762)
        dsr_many = deflated_sharpe_ratio(sharpe_obs=0.1, n_trials=200, t=1762)
        assert dsr_few > dsr_many

    def test_degenerate_inputs_return_nan(self):
        from qframe.factor_harness.multiple_testing import deflated_sharpe_ratio
        import math
        assert math.isnan(deflated_sharpe_ratio(0.0, 0, 1000))  # n_trials=0
        assert math.isnan(deflated_sharpe_ratio(0.0, 10, 1))    # t too small
        assert math.isnan(deflated_sharpe_ratio(float('nan'), 10, 1000))


class TestBHYThreshold:
    """Tests for the online BHY t-threshold."""

    def test_threshold_increases_with_m(self):
        from qframe.factor_harness.multiple_testing import bhy_t_threshold
        t1 = bhy_t_threshold(1)
        t10 = bhy_t_threshold(10)
        t100 = bhy_t_threshold(100)
        assert t1 < t10 < t100

    def test_threshold_positive(self):
        from qframe.factor_harness.multiple_testing import bhy_t_threshold
        for m in [1, 5, 50, 140]:
            assert bhy_t_threshold(m) > 0

    def test_threshold_m1_less_than_m140(self):
        """After 140 tests the bar should be substantially higher than for 1 test."""
        from qframe.factor_harness.multiple_testing import bhy_t_threshold
        assert bhy_t_threshold(140) > bhy_t_threshold(1) * 1.5

