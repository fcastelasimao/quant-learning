"""
Hidden Semi-Markov Model (HSMM) for market regime detection.

This module wraps ``hmmlearn.hmm.GaussianHMM`` to provide an n-state regime
classifier for daily equity returns.  The default 3-state parameterisation
corresponds to:

    State 0 (most bearish): negative or near-zero mean, elevated volatility
    State 1 (neutral):      near-zero mean, moderate volatility
    State 2 (most bullish): positive mean, lower volatility

No labels are hard-coded — the model learns them from data.  The canonical
ordering (most bearish → most bullish by mean return) is established once on
the in-sample period and used to reorder subsequent rolling-fit posteriors,
solving the label-switching problem.

Input features per day: [r_t, r_t²]
    - r_t   captures the return level (sign and magnitude)
    - r_t²  proxies for daily realised variance (standard in GARCH/HMM literature)
      and is more informative than |r_t| because it emphasises large moves
      and is uncorrelated with r_t in sign.

Walk-forward design
-------------------
``fit_rolling`` implements a strict walk-forward protocol:
    1. Fit on window [t - window : t]
    2. Produce posteriors (predict_proba) for NEXT `step` days [t : t + step]
    3. Reorder columns to match canonical ordering from IS fit
    4. Advance t by `step`
No look-ahead: training data never includes the prediction window.
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SQRT252 = np.sqrt(252)


# ---------------------------------------------------------------------------
# RegimeHSMM
# ---------------------------------------------------------------------------

class RegimeHSMM:
    """
    n-state Gaussian HMM regime detector for daily return series.

    Features: [r_t, r_t²] — return level and realised variance proxy.

    Args:
        n_states:     Number of hidden states (default 3; 3 is recommended for
                      stability, 5 for finer granularity).
        n_iter:       EM algorithm max iterations (default 200).
        random_state: Random seed for reproducibility.
    """

    def __init__(
        self,
        n_states: int = 3,
        n_iter: int = 200,
        random_state: int = 42,
    ) -> None:
        self.n_states = n_states
        self.n_iter = n_iter
        self.random_state = random_state
        self._model: Optional[GaussianHMM] = None
        # Canonical column order: index i maps to the i-th state ordered by
        # ascending mean return.  Set by fit() on IS data.
        self._canonical_order: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, returns: pd.Series) -> "RegimeHSMM":
        """
        Fit the HMM on the full return history and establish canonical ordering.

        Canonical ordering: states sorted ascending by their mean return
        (state 0 = most bearish, state n_states-1 = most bullish).
        This ordering is stored and applied to all subsequent predict_proba
        calls, eliminating the label-switching problem for IS analysis.

        Args:
            returns: pd.Series of daily returns (NaN values are dropped).

        Returns:
            self (for method chaining).
        """
        X = self._build_features(returns.dropna())
        model = self._make_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X)
        self._model = model
        # Establish canonical state ordering by mean return of each state's
        # Gaussian component.  means_ shape: (n_states, n_features).
        # Feature 0 is r_t, so means_[:, 0] gives each state's expected return.
        self._canonical_order = np.argsort(model.means_[:, 0])
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, returns: pd.Series) -> pd.Series:
        """
        Assign hard regime labels via Viterbi, reordered to canonical ordering.

        Args:
            returns: pd.Series of daily returns.

        Returns:
            pd.Series of integer labels in [0, n_states-1], ascending by mean
            return (0 = most bearish), indexed by date.
        """
        self._check_fitted()
        clean = returns.dropna()
        X = self._build_features(clean)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw_labels = self._model.predict(X)
        # Remap raw labels to canonical ordering
        remap = np.argsort(self._canonical_order)  # raw_label → canonical_label
        labels = remap[raw_labels]
        return pd.Series(labels, index=clean.index, name="regime", dtype=int)

    def predict_proba(self, returns: pd.Series) -> pd.DataFrame:
        """
        Return posterior state probabilities, columns reordered canonically.

        Args:
            returns: pd.Series of daily returns.

        Returns:
            pd.DataFrame of shape (n_dates, n_states), rows sum to 1.
            Column j = probability of being in the j-th state in canonical
            ordering (0 = most bearish, n_states-1 = most bullish).
        """
        self._check_fitted()
        clean = returns.dropna()
        X = self._build_features(clean)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            posteriors = self._model.predict_proba(X)
        # Reorder columns to canonical ordering
        reordered = posteriors[:, self._canonical_order]
        return pd.DataFrame(
            reordered,
            index=clean.index,
            columns=list(range(self.n_states)),
        )

    # ------------------------------------------------------------------
    # Rolling walk-forward
    # ------------------------------------------------------------------

    def fit_rolling(
        self,
        returns: pd.Series,
        window: int = 504,
        step: int = 63,
        canonical_model: Optional["RegimeHSMM"] = None,
    ) -> pd.DataFrame:
        """
        Walk-forward rolling regime estimation returning posterior probabilities.

        Uses ``predict_proba`` (not hard Viterbi labels) throughout to avoid
        the label-switching problem.  Column ordering at each window is aligned
        to the canonical ordering via Hungarian assignment.

        Protocol (no look-ahead):
            For each anchor t (advancing by `step`):
                1. Fit model on returns[t - window : t]
                2. Produce posteriors for returns[t : t + step]
                3. Reorder columns to match canonical mean-return ordering

        Args:
            returns:          pd.Series of daily returns, DatetimeIndex.
            window:           Training window in days (default 504 ≈ 2 years).
            step:             Prediction horizon per fit, also the step (default
                              63 ≈ 1 quarter).
            canonical_model:  A pre-fitted RegimeHSMM whose canonical ordering
                              defines the reference.  If None, canonical
                              ordering is derived from each window's own
                              mean-return ordering (simple and robust for
                              research purposes).

        Returns:
            pd.DataFrame of shape (n_dates, n_states), posterior probabilities.
            Dates in the warm-up period (first `window` days) are NaN.
        """
        clean = returns.dropna()
        n = len(clean)
        proba_arr = np.full((n, self.n_states), np.nan, dtype=float)

        t = window
        while t < n:
            train = clean.iloc[t - window : t]
            pred_end = min(t + step, n)
            pred_slice = clean.iloc[t : pred_end]

            if len(pred_slice) == 0:
                break

            model = self._make_model()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(self._build_features(train))
                    raw_post = model.predict_proba(self._build_features(pred_slice))

                # Order columns by ascending mean return of THIS window's fit
                order = np.argsort(model.means_[:, 0])
                reordered = raw_post[:, order]
                proba_arr[t : pred_end] = reordered

            except Exception:
                # Fit failure (e.g. degenerate window) — leave as NaN
                pass

            t += step

        return pd.DataFrame(
            proba_arr,
            index=clean.index,
            columns=list(range(self.n_states)),
        )

    # ------------------------------------------------------------------
    # Regime statistics
    # ------------------------------------------------------------------

    def regime_stats(
        self,
        returns: pd.Series,
        regimes: pd.Series,
    ) -> pd.DataFrame:
        """
        Compute per-regime summary statistics.

        Args:
            returns: pd.Series of daily returns.
            regimes: pd.Series of integer regime labels (same index).

        Returns:
            pd.DataFrame indexed by regime_id with columns:
                count, pct_time, mean_ann, std_ann, sharpe.
        """
        aligned = pd.DataFrame({"ret": returns, "regime": regimes}).dropna()
        total = len(aligned)
        rows: list[dict] = []

        for state in sorted(aligned["regime"].unique()):
            mask = aligned["regime"] == state
            r = aligned.loc[mask, "ret"]
            mean_daily = r.mean()
            std_daily = r.std(ddof=1)
            mean_ann = mean_daily * 252
            std_ann = std_daily * _SQRT252
            sharpe = mean_ann / std_ann if std_ann > 0 else np.nan
            rows.append({
                "regime": int(state),
                "count": int(mask.sum()),
                "pct_time": mask.sum() / total,
                "mean_ann": mean_ann,
                "std_ann": std_ann,
                "sharpe": sharpe,
            })

        return pd.DataFrame(rows).set_index("regime")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_model(self) -> GaussianHMM:
        return GaussianHMM(
            n_components=self.n_states,
            covariance_type="diag",
            n_iter=self.n_iter,
            random_state=self.random_state,
        )

    @staticmethod
    def _build_features(returns: pd.Series) -> np.ndarray:
        """
        Build the 2-feature observation matrix [r_t, r_t²].

        r_t²  (squared return) proxies for daily realised variance and
        emphasises large moves more than |r_t|, while being uncorrelated
        with the sign of r_t.

        Args:
            returns: pd.Series of returns, NaN already removed.

        Returns:
            np.ndarray of shape (n_obs, 2).
        """
        r = returns.values.astype(float)
        return np.column_stack([r, r ** 2])

    def _check_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError(
                "RegimeHSMM is not fitted. Call .fit() before predicting."
            )
