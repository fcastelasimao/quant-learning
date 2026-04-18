"""
RegimeICAnalyzer — integrate HSMM, Hurst, and velocity into regime-conditional
IC analysis.

This is the core Phase 2 deliverable.  It answers the question:

    Does conditioning factor exposure on the current market regime
    increase net-IC compared to unconditional exposure?

Workflow
--------
1. ``fit(market_returns, is_end)``
        Fit canonical HSMM on IS data [start : is_end].
        Run rolling walk-forward HSMM on full series to get daily posteriors.
        Compute rolling Hurst exponent (DFA).
        Compute KL-divergence velocity + EWM-smoothed velocity.

2. ``regime_ic_decomposition(factor_df, prices, oos_start, horizon)``
        For each regime state s, compute cross-sectional IC over all OOS dates
        where that state has the highest posterior probability (hard assignment)
        and also via soft-weighted average.
        Returns a DataFrame: state × {IC, ICIR, t_stat, n_days, pct_time}.

3. ``unconditional_vs_conditional(factor_df, prices, oos_start, horizon)``
        One-table summary: unconditional IC vs. best-regime IC vs. worst-regime
        IC.  The go/no-go verdict for regime conditioning.

4. ``regime_weights(factor_df, oos_start, ic_by_state)``
        Returns a time-series of daily scaling multipliers:
            w_t = Σ_s  π_t(s) · μ_s   (expected IC given current posteriors)
        where μ_s is estimated from historical OOS regime-conditional IC.
        Optional velocity filter and Hurst filter applied on top.

No look-ahead bias:
- The HSMM posteriors for date t come from a model fitted on [t-window : t].
- Hurst at t uses returns ending at t.
- IC computation uses factor scores at t and forward returns from t+1.
- Regime labels used for conditioning are from the *same* date as the factor
  score, never from future dates.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from qframe.regime.hsmm import RegimeHSMM
from qframe.regime.hurst import HurstEstimator
from qframe.regime.velocity import kl_divergence_velocity, smoothed_velocity


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class RegimeDecomposition:
    """
    Per-state IC breakdown for a single factor.

    Attributes:
        by_state:       DataFrame indexed by state with columns:
                        ic, icir, t_stat, n_days, pct_time, mean_ret_ann.
        unconditional:  IC over the full OOS period (no regime filter).
        best_state:     State index with highest IC (hard-label decomposition).
        worst_state:    State index with lowest IC.
        lift:           best_state IC / unconditional IC.  >1 means conditioning
                        helps; 1.5 is the Phase 2 go/no-go threshold.
        soft_ic_series: Daily expected IC series (Σ_s π_t(s) · μ_s).
    """
    by_state: pd.DataFrame
    unconditional: float
    best_state: int
    worst_state: int
    lift: float
    soft_ic_series: pd.Series


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class RegimeICAnalyzer:
    """
    Regime-conditional IC analyzer.

    Args:
        n_states:        Number of HSMM states (default 3).  Test both 3 and 5.
        hurst_window:    Rolling Hurst lookback in days (default 252).
        velocity_window: KL-velocity lookback window in days (default 21).
        velocity_ewm_hl: Half-life for smoothing velocity (default 21).
        hsmm_window:     Rolling HSMM training window in days (default 504 ≈ 2yr).
        hsmm_step:       Walk-forward step size in days (default 63 ≈ 1Q).
        min_state_days:  Minimum OOS days per state for IC estimation.
                         States with fewer days are excluded from decomposition.
        random_state:    Seed for HSMM reproducibility.
    """

    def __init__(
        self,
        n_states: int = 3,
        hurst_window: int = 252,
        velocity_window: int = 21,
        velocity_ewm_hl: int = 21,
        hsmm_window: int = 504,
        hsmm_step: int = 63,
        min_state_days: int = 30,
        random_state: int = 42,
    ) -> None:
        self.n_states = n_states
        self.hurst_window = hurst_window
        self.velocity_window = velocity_window
        self.velocity_ewm_hl = velocity_ewm_hl
        self.hsmm_window = hsmm_window
        self.hsmm_step = hsmm_step
        self.min_state_days = min_state_days
        self.random_state = random_state

        # Set by fit()
        self._is_model: Optional[RegimeHSMM] = None
        self.proba_df: Optional[pd.DataFrame] = None      # (dates, n_states)
        self.hurst_series: Optional[pd.Series] = None     # rolling Hurst
        self.velocity_raw: Optional[pd.Series] = None     # raw KL velocity
        self.velocity_smooth: Optional[pd.Series] = None  # EWM-smoothed velocity
        self._market_returns: Optional[pd.Series] = None

    # ------------------------------------------------------------------
    # Step 1: fit
    # ------------------------------------------------------------------

    def fit(
        self,
        market_returns: pd.Series,
        is_end: str,
    ) -> "RegimeICAnalyzer":
        """
        Fit canonical HSMM on IS period, then run walk-forward on full series.

        Also computes rolling Hurst and KL velocity over the full series.

        Args:
            market_returns: pd.Series of daily market returns (equal-weight or
                            value-weight S&P 500 proxy), DatetimeIndex.
            is_end:         Last date of in-sample period (exclusive end of OOS).
                            E.g. "2017-12-31".

        Returns:
            self.
        """
        self._market_returns = market_returns.dropna().sort_index()
        is_returns = self._market_returns.loc[:is_end]

        # 1. Fit canonical model on IS data to establish state ordering
        self._is_model = RegimeHSMM(
            n_states=self.n_states,
            n_iter=200,
            random_state=self.random_state,
        )
        self._is_model.fit(is_returns)

        # 2. Walk-forward rolling HSMM on full series (no look-ahead)
        roller = RegimeHSMM(
            n_states=self.n_states,
            n_iter=200,
            random_state=self.random_state,
        )
        self.proba_df = roller.fit_rolling(
            self._market_returns,
            window=self.hsmm_window,
            step=self.hsmm_step,
        )

        # 3. Rolling Hurst (DFA)
        estimator = HurstEstimator(min_periods=max(100, self.hurst_window // 2))
        self.hurst_series = estimator.fit_rolling(
            self._market_returns,
            window=self.hurst_window,
        )

        # 4. KL-divergence velocity
        self.velocity_raw = kl_divergence_velocity(
            self.proba_df,
            window=self.velocity_window,
        )
        self.velocity_smooth = smoothed_velocity(
            self.velocity_raw,
            halflife=self.velocity_ewm_hl,
        )

        return self

    # ------------------------------------------------------------------
    # Step 2: regime IC decomposition
    # ------------------------------------------------------------------

    def regime_ic_decomposition(
        self,
        factor_df: pd.DataFrame,
        prices: pd.DataFrame,
        oos_start: str,
        horizon: int = 1,
        min_stocks: int = 20,
    ) -> RegimeDecomposition:
        """
        Decompose factor IC by regime state over the OOS period.

        Two approaches:
        - Hard assignment: date t is assigned to state argmax(π_t).
        - Soft weighting: each date contributes to all states weighted by π_t(s).
          (used for the soft_ic_series only; hard assignment for by_state table)

        Args:
            factor_df:  (dates × tickers) factor scores.
            prices:     (dates × tickers) adjusted close prices.
            oos_start:  Start of OOS evaluation period.
            horizon:    Forward return horizon in days (default 1).
            min_stocks: Minimum stocks per date for a valid IC observation.

        Returns:
            RegimeDecomposition with per-state IC breakdown.
        """
        self._check_fitted()

        # Build returns and forward returns
        returns_df = prices.pct_change()
        fwd_ret = self._build_fwd_returns(returns_df, horizon)

        # Align all series to OOS period
        oos_dates = factor_df.index[factor_df.index >= oos_start]
        oos_dates = oos_dates.intersection(fwd_ret.index).intersection(
            self.proba_df.index
        )

        factor_oos = factor_df.loc[oos_dates]
        fwd_oos = fwd_ret.loc[oos_dates]
        proba_oos = self.proba_df.loc[oos_dates]

        # Hard-assignment regime labels
        hard_labels = proba_oos.values.argmax(axis=1)

        # Compute per-date IC (needed for both unconditional and per-state)
        ic_per_date = self._compute_daily_ic(factor_oos, fwd_oos, min_stocks)

        # Unconditional IC
        valid_ic = ic_per_date.dropna()
        unconditional_ic = float(valid_ic.mean()) if len(valid_ic) > 0 else np.nan

        # Per-state IC decomposition (hard assignment)
        rows = []
        for s in range(self.n_states):
            mask = (hard_labels == s)
            dates_s = oos_dates[mask]
            ic_s = ic_per_date.loc[dates_s].dropna()
            n_days = len(ic_s)

            if n_days < self.min_state_days:
                rows.append({
                    "state": s,
                    "ic": np.nan,
                    "icir": np.nan,
                    "t_stat": np.nan,
                    "n_days": n_days,
                    "pct_time": mask.sum() / len(oos_dates),
                    "mean_ret_ann": np.nan,
                    "note": f"insufficient days ({n_days} < {self.min_state_days})",
                })
                continue

            ic_mean = float(ic_s.mean())
            ic_std = float(ic_s.std(ddof=1))
            icir = ic_mean / ic_std if ic_std > 0 else np.nan
            t_stat = ic_mean / (ic_std / np.sqrt(n_days)) if ic_std > 0 else np.nan

            # Mean annualised market return in this state
            if self._market_returns is not None:
                mr = self._market_returns.loc[dates_s].dropna()
                mean_ret_ann = float(mr.mean() * 252) if len(mr) > 0 else np.nan
            else:
                mean_ret_ann = np.nan

            rows.append({
                "state": s,
                "ic": ic_mean,
                "icir": icir,
                "t_stat": t_stat,
                "n_days": n_days,
                "pct_time": mask.sum() / len(oos_dates),
                "mean_ret_ann": mean_ret_ann,
                "note": "",
            })

        by_state = pd.DataFrame(rows).set_index("state")

        # Best / worst state (by IC, ignoring NaN)
        ic_col = by_state["ic"].dropna()
        best_state = int(ic_col.idxmax()) if len(ic_col) > 0 else 0
        worst_state = int(ic_col.idxmin()) if len(ic_col) > 0 else 0
        best_ic = float(ic_col.max()) if len(ic_col) > 0 else np.nan
        lift = best_ic / unconditional_ic if (
            unconditional_ic and np.isfinite(unconditional_ic) and unconditional_ic > 0
        ) else np.nan

        # Soft IC series: daily expected IC = Σ_s π_t(s) · μ_s
        # μ_s estimated from this decomposition
        mu = by_state["ic"].fillna(0.0).values  # shape (n_states,)
        soft_ic_arr = proba_oos.values @ mu      # (n_oos_dates,)
        soft_ic_series = pd.Series(
            soft_ic_arr, index=oos_dates, name=f"soft_ic_h{horizon}"
        )

        return RegimeDecomposition(
            by_state=by_state,
            unconditional=unconditional_ic,
            best_state=best_state,
            worst_state=worst_state,
            lift=lift,
            soft_ic_series=soft_ic_series,
        )

    # ------------------------------------------------------------------
    # Step 3: summary table
    # ------------------------------------------------------------------

    def unconditional_vs_conditional(
        self,
        factor_df: pd.DataFrame,
        prices: pd.DataFrame,
        oos_start: str,
        horizon: int = 1,
    ) -> pd.DataFrame:
        """
        One-table go/no-go summary for regime conditioning.

        Returns:
            pd.DataFrame with rows: unconditional, best_regime, worst_regime.
            Columns: ic, n_days, pct_time, lift.
        """
        decomp = self.regime_ic_decomposition(
            factor_df, prices, oos_start, horizon=horizon
        )
        by_state = decomp.by_state

        best = by_state.loc[decomp.best_state]
        worst = by_state.loc[decomp.worst_state]
        n_oos = by_state["n_days"].sum()

        rows = [
            {
                "label": "unconditional",
                "ic": decomp.unconditional,
                "n_days": n_oos,
                "pct_time": 1.0,
                "lift": 1.0,
            },
            {
                "label": f"best_regime (state {decomp.best_state})",
                "ic": best["ic"],
                "n_days": best["n_days"],
                "pct_time": best["pct_time"],
                "lift": decomp.lift,
            },
            {
                "label": f"worst_regime (state {decomp.worst_state})",
                "ic": worst["ic"],
                "n_days": worst["n_days"],
                "pct_time": worst["pct_time"],
                "lift": (
                    worst["ic"] / decomp.unconditional
                    if decomp.unconditional and np.isfinite(decomp.unconditional) and decomp.unconditional > 0
                    else np.nan
                ),
            },
        ]
        return pd.DataFrame(rows).set_index("label")

    # ------------------------------------------------------------------
    # Step 4: regime-conditional portfolio weights
    # ------------------------------------------------------------------

    def regime_weights(
        self,
        factor_df: pd.DataFrame,
        oos_start: str,
        ic_by_state: Optional[np.ndarray] = None,
        velocity_threshold_pct: float = 90.0,
        velocity_scale_down: float = 0.5,
        hurst_threshold_up: float = 0.55,
        hurst_threshold_down: float = 0.45,
        hurst_boost: float = 0.10,
    ) -> pd.Series:
        """
        Daily scaling multiplier for factor exposure based on regime.

        The multiplier combines three signals:
        1. Regime-expected IC:  w_base = Σ_s π_t(s) · μ_s / μ_global
           (1.0 when regime IC matches unconditional IC)
        2. Velocity filter:     if smooth_velocity > `velocity_threshold_pct`-th
                                percentile, scale down by `velocity_scale_down`
        3. Hurst boost:         if H > `hurst_threshold_up`, add `hurst_boost`
                                if H < `hurst_threshold_down`, subtract it

        Args:
            factor_df:              Factor scores (used only for date alignment).
            oos_start:              Start of OOS evaluation period.
            ic_by_state:            Array of length n_states with μ_s values.
                                    If None, all states get equal weight (1.0).
            velocity_threshold_pct: Percentile of smoothed velocity above which
                                    exposure is scaled down (default 90).
            velocity_scale_down:    Multiplier applied when velocity is high.
            hurst_threshold_up:     H above this → Calmar/momentum boost.
            hurst_threshold_down:   H below this → momentum penalty.
            hurst_boost:            Magnitude of Hurst adjustment.

        Returns:
            pd.Series of daily multipliers indexed by OOS dates.
            1.0 = full unconditional exposure; >1 = increased; <1 = reduced.
        """
        self._check_fitted()

        oos_dates = factor_df.index[factor_df.index >= oos_start]
        oos_dates = oos_dates.intersection(self.proba_df.index)

        proba_oos = self.proba_df.loc[oos_dates]

        # 1. Regime-expected IC base weight
        if ic_by_state is None:
            mu = np.ones(self.n_states)
        else:
            mu = np.asarray(ic_by_state, dtype=float)

        mu_global = mu.mean() if mu.mean() != 0 else 1.0
        raw_weights = proba_oos.values @ mu / mu_global  # normalised to ~1.0
        weight_series = pd.Series(raw_weights, index=oos_dates, name="regime_weight")

        # 2. Velocity filter
        vel_oos = self.velocity_smooth.reindex(oos_dates)
        vel_threshold = vel_oos.quantile(velocity_threshold_pct / 100.0)
        high_vel = vel_oos > vel_threshold
        weight_series = weight_series.where(~high_vel, weight_series * velocity_scale_down)

        # 3. Hurst filter
        hurst_oos = self.hurst_series.reindex(oos_dates)
        weight_series = weight_series.where(
            hurst_oos.isna() | (hurst_oos <= hurst_threshold_up),
            weight_series + hurst_boost,
        )
        weight_series = weight_series.where(
            hurst_oos.isna() | (hurst_oos >= hurst_threshold_down),
            weight_series - hurst_boost,
        )

        # Clip to [0, 2] — never short the signal or over-lever beyond 2×
        weight_series = weight_series.clip(lower=0.0, upper=2.0)
        return weight_series

    # ------------------------------------------------------------------
    # Step 5: per-regime IC-weighted blend weights (Phase 2.5b)
    # ------------------------------------------------------------------

    def regime_blend_weights(
        self,
        ic_by_state_dict: "dict[str, np.ndarray]",
        oos_start: str,
        shrinkage: float = 0.0,
    ) -> pd.DataFrame:
        """
        Compute time-varying IC-proportional blend weights for multiple factors.

        At each OOS date *t*, given the HSMM posterior π_t(s) over states s:

            numerator_i(t)  = Σ_s  π_t(s) · IC_i(s)
            w_i(t)          = numerator_i(t) / Σ_j numerator_j(t)

        where IC_i(s) is factor i's IC in regime state s (from
        ``regime_ic_decomposition(...).by_state['ic']``).

        When the posterior is uniform (regime ambiguous), ``w_i(t)`` falls back
        to IC-proportional weights automatically.  No parameters are fitted —
        all IC_i(s) values come directly from Phase 2 decompositions.

        Optionally shrink toward equal weights to guard against estimation
        noise in regimes with few observations:
            w_final = (1 - shrinkage) * w_posterior + shrinkage * (1/N)

        Args:
            ic_by_state_dict: ``{factor_name: np.ndarray of length n_states}``
                              The per-state IC for each factor.  Obtain via
                              ``decomp.by_state['ic'].fillna(0).values``.
            oos_start:        Start of OOS evaluation period.
            shrinkage:        Fraction in [0, 1] to blend toward equal weights.
                              0.0 = pure posterior-weighted (default).
                              0.5 = halfway between posterior and equal weights.

        Returns:
            pd.DataFrame of shape (OOS dates, n_factors), columns = factor names,
            values = blend weights summing to 1.0 per row.

        Raises:
            RuntimeError: if fit() has not been called.
            ValueError:   if ic_by_state arrays have wrong length.
        """
        self._check_fitted()

        factor_names = list(ic_by_state_dict.keys())
        n_factors = len(factor_names)
        if n_factors == 0:
            raise ValueError("ic_by_state_dict must contain at least one factor.")

        # Validate all IC arrays
        for name, arr in ic_by_state_dict.items():
            arr = np.asarray(arr, dtype=float)
            if len(arr) != self.n_states:
                raise ValueError(
                    f"ic_by_state for '{name}' has length {len(arr)}, "
                    f"expected {self.n_states} (= n_states)."
                )
            ic_by_state_dict[name] = arr

        # Align OOS dates
        proba_oos = self.proba_df.loc[self.proba_df.index >= oos_start].dropna(how="all")

        # Stack IC arrays: shape (n_factors, n_states)
        ic_matrix = np.stack(
            [ic_by_state_dict[name] for name in factor_names], axis=0
        )  # (n_factors, n_states)

        # posterior-weighted numerators: (n_dates, n_factors)
        # numerator_i(t) = Σ_s π_t(s) · IC_i(s)  = proba_oos @ ic_matrix.T
        proba_arr = proba_oos.values  # (n_dates, n_states)
        numerators = proba_arr @ ic_matrix.T  # (n_dates, n_factors)

        # Clip negatives to zero before normalising
        # (a factor with negative expected IC in the current regime gets 0 weight)
        numerators = np.clip(numerators, 0.0, None)

        # Row-normalise → blend weights summing to 1
        row_sums = numerators.sum(axis=1, keepdims=True)
        # Guard against rows where all factors have zero or negative expected IC
        safe_sums = np.where(row_sums > 0, row_sums, 1.0)
        weights = numerators / safe_sums

        # Fallback: rows where all factors had ≤ 0 numerator → equal weights
        all_zero_mask = (row_sums <= 0).flatten()
        if all_zero_mask.any():
            weights[all_zero_mask] = 1.0 / n_factors

        # Optional shrinkage toward equal weights
        if shrinkage > 0.0:
            equal_w = np.full_like(weights, 1.0 / n_factors)
            weights = (1.0 - shrinkage) * weights + shrinkage * equal_w

        return pd.DataFrame(weights, index=proba_oos.index, columns=factor_names)

    # ------------------------------------------------------------------
    # Accessors for downstream visualization
    # ------------------------------------------------------------------

    def hard_labels(self, oos_start: Optional[str] = None) -> pd.Series:
        """Return hard regime labels (argmax of posteriors)."""
        self._check_fitted()
        labels = self.proba_df.values.argmax(axis=1)
        s = pd.Series(labels, index=self.proba_df.index, name="regime", dtype=float)
        # Preserve NaN rows (warm-up period has NaN posteriors)
        nan_mask = self.proba_df.isna().all(axis=1)
        s[nan_mask] = np.nan
        if oos_start:
            s = s.loc[oos_start:]
        return s

    def regime_stats_oos(self, oos_start: str) -> pd.DataFrame:
        """Per-state regime statistics (pct_time, mean_ann return, std_ann, sharpe)."""
        self._check_fitted()
        if self._market_returns is None:
            raise RuntimeError("Call fit() first.")
        labels = self.hard_labels(oos_start=oos_start).dropna()
        model = RegimeHSMM(n_states=self.n_states)
        return model.regime_stats(
            self._market_returns.loc[labels.index],
            labels.astype(int),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_fwd_returns(
        returns_df: pd.DataFrame,
        horizon: int,
    ) -> pd.DataFrame:
        """Build (dates × tickers) compounded forward return DataFrame."""
        log_ret = np.log1p(returns_df)
        fwd_log = log_ret.rolling(horizon, min_periods=horizon).sum().shift(-horizon)
        return np.expm1(fwd_log)

    @staticmethod
    def _compute_daily_ic(
        factor_df: pd.DataFrame,
        fwd_ret: pd.DataFrame,
        min_stocks: int,
    ) -> pd.Series:
        """Compute cross-sectional Spearman IC at each date."""
        shared_tickers = factor_df.columns.intersection(fwd_ret.columns)
        f = factor_df[shared_tickers]
        r = fwd_ret[shared_tickers]

        ic_vals: dict = {}
        for date in f.index:
            if date not in r.index:
                ic_vals[date] = np.nan
                continue
            fi = f.loc[date]
            ri = r.loc[date]
            mask = fi.notna() & ri.notna()
            if mask.sum() < min_stocks:
                ic_vals[date] = np.nan
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                corr, _ = spearmanr(fi[mask], ri[mask])
            ic_vals[date] = float(corr)

        return pd.Series(ic_vals, name="ic")

    def _check_fitted(self) -> None:
        if self.proba_df is None:
            raise RuntimeError(
                "RegimeICAnalyzer is not fitted. Call .fit() first."
            )
