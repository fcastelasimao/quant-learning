"""
HMM Regime Detection for Wave Rider defense mode.

Provides a 3-state Student-t Hidden Markov Model that classifies market
conditions into bull, sideways, and bear regimes using causal (forward-only)
filtering.  No future data is used at any point — safe for backtesting.

Extracted and generalised from HMM/HMMSPYfirsttry.py (v2).

Components
----------
StudentTHMM            : HMM with Student-t emission distributions
BOCPD                  : Bayesian Online Changepoint Detection
forward_filter         : Causal P(state_t | y_1:t)
apply_min_duration     : Suppress noisy short-lived regime flips
build_feature_matrix   : 5-feature observation vector from price + VIX
RegimeDetector         : High-level detector that fits, labels, and scores
regime_defense_scale   : Maps regime posteriors to a gross-exposure scalar
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import digamma
from scipy.stats import t as student_t
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

try:
    from hmmlearn.hmm import BaseHMM
except ImportError:
    raise ImportError(
        "hmmlearn is required for regime detection. "
        "Install it with: pip install hmmlearn"
    )


# ═══════════════════════════════════════════════════════════════════════════
# 1.  STUDENT-t HMM
# ═══════════════════════════════════════════════════════════════════════════

class StudentTHMM(BaseHMM):
    """
    HMM with diagonal Student-t emission distributions.

    Each state k has parameters:
        means_[k]  — location vector  (n_features,)
        scales_[k] — scale vector     (n_features,)
        dof_       — degrees of freedom (scalar, shared)

    Setting fixed_dof=True uses dof=init_dof (default 4) and skips
    estimation — faster and more stable on short samples.
    """

    def __init__(
        self,
        n_components: int = 3,
        n_iter: int = 200,
        tol: float = 1e-4,
        random_state: int | None = None,
        fixed_dof: bool = True,
        init_dof: float = 4.0,
    ):
        super().__init__(
            n_components=n_components,
            n_iter=n_iter,
            tol=tol,
            random_state=random_state,
            params="stmc",
            init_params="stmc",
        )
        self.fixed_dof = fixed_dof
        self.dof_ = float(init_dof)

    def _get_n_fit_scalars_per_param(self) -> dict[str, int]:
        n, d = self.n_components, self.n_features
        return {"s": n - 1, "t": n * (n - 1), "m": n * d, "c": n * d}

    def _check(self) -> None:
        super()._check()
        self.means_ = np.asarray(self.means_)
        self.scales_ = np.asarray(self.scales_)

    def _init(self, X: np.ndarray, lengths: np.ndarray | None = None) -> None:
        super()._init(X, lengths)
        n, d = self.n_components, X.shape[1]
        rng = np.random.RandomState(self.random_state)
        self.means_ = X[rng.choice(len(X), n, replace=False)]
        self.scales_ = np.ones((n, d))
        self.n_features = d

    def _compute_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        T = len(X)
        log_prob = np.zeros((T, self.n_components))
        v = self.dof_
        for k in range(self.n_components):
            log_prob[:, k] = np.sum(
                student_t.logpdf(X, df=v, loc=self.means_[k], scale=self.scales_[k]),
                axis=1,
            )
        return log_prob

    def _generate_sample_from_state(
        self, state: int, random_state: int | None = None
    ) -> np.ndarray:
        rng = np.random.RandomState(random_state)
        return student_t.rvs(
            df=self.dof_,
            loc=self.means_[state],
            scale=self.scales_[state],
            random_state=rng,
        )

    def _initialize_sufficient_statistics(self) -> dict:
        stats = super()._initialize_sufficient_statistics()
        stats["post"] = np.zeros(self.n_components)
        stats["sum_w"] = np.zeros((self.n_components, self.n_features))
        stats["sum_wx"] = np.zeros((self.n_components, self.n_features))
        stats["sum_wd"] = np.zeros((self.n_components, self.n_features))
        return stats

    def _accumulate_sufficient_statistics(
        self, stats, X, framelogprob, posteriors, fwdlattice, bwdlattice
    ) -> None:
        super()._accumulate_sufficient_statistics(
            stats, X, framelogprob, posteriors, fwdlattice, bwdlattice
        )
        v = self.dof_
        for k in range(self.n_components):
            mu = self.means_[k]
            sigma = self.scales_[k]
            gamma = posteriors[:, k]
            delta2 = ((X - mu) / sigma) ** 2
            u = (v + 1) / (v + delta2)
            stats["post"][k] += gamma.sum()
            stats["sum_w"][k] += (gamma[:, None] * u).sum(axis=0)
            stats["sum_wx"][k] += (gamma[:, None] * u * X).sum(axis=0)
            stats["sum_wd"][k] += (gamma[:, None] * u * delta2).sum(axis=0)

    def _do_mstep(self, stats) -> None:
        super()._do_mstep(stats)
        for k in range(self.n_components):
            w = stats["sum_w"][k]
            wx = stats["sum_wx"][k]
            wd = stats["sum_wd"][k]
            N = stats["post"][k]
            if N > 1e-6:
                self.means_[k] = wx / w
                self.scales_[k] = np.sqrt(wd / N)
                self.scales_[k] = np.maximum(self.scales_[k], 1e-6)
        if not self.fixed_dof:
            self.dof_ = self._update_dof(stats)

    def _update_dof(self, stats) -> float:
        N = stats["post"].sum()
        if N < 1e-6:
            return self.dof_

        def neg_obj(v):
            return -(
                -np.log(2)
                + digamma((v + 1) / 2)
                - digamma(v / 2)
                + np.log(v / 2)
                - (v + 1)
                / 2
                * (
                    np.log(1 + stats["sum_wd"].sum() / (v * N))
                    - stats["sum_wd"].sum() / (v * N)
                )
            )

        result = minimize_scalar(neg_obj, bounds=(2.01, 100.0), method="bounded")
        return result.x if result.success else self.dof_

    @property
    def covars_(self) -> np.ndarray:
        return self.scales_ ** 2

    @covars_.setter
    def covars_(self, value: np.ndarray) -> None:
        self.scales_ = np.sqrt(np.abs(value))


# ═══════════════════════════════════════════════════════════════════════════
# 2.  BOCPD — Bayesian Online Changepoint Detection
# ═══════════════════════════════════════════════════════════════════════════

class BOCPD:
    """
    Bayesian Online Changepoint Detection (Adams & MacKay, 2007).

    Uses a Normal-Gamma conjugate prior for Gaussian observations.
    Outputs P(changepoint at t) for each time step causally.
    """

    def __init__(
        self,
        hazard_rate: float = 1 / 20,
        alpha0: float = 1.0,
        beta0: float = 0.1,
        mu0: float = 0.0,
        kappa0: float = 1.0,
    ):
        self.h = hazard_rate
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.mu0 = mu0
        self.kappa0 = kappa0

    def run(self, data: np.ndarray) -> np.ndarray:
        """Process a 1-D series and return P(changepoint at t)."""
        T = len(data)
        run_length_probs = np.array([1.0])
        alpha = np.array([self.alpha0])
        beta = np.array([self.beta0])
        mu = np.array([self.mu0])
        kappa = np.array([self.kappa0])
        changepoint_probs = np.zeros(T)

        for t, x in enumerate(data):
            pred_mean = mu
            pred_var = beta * (kappa + 1) / (alpha * kappa)
            pred_df = 2 * alpha
            log_pred = student_t.logpdf(
                x, df=pred_df, loc=pred_mean, scale=np.sqrt(pred_var)
            )
            pred = np.exp(log_pred - log_pred.max())
            joint = run_length_probs * pred
            growth = joint * (1 - self.h)
            cp_mass = (joint * self.h).sum()
            new_run_length_probs = np.append(cp_mass, growth)
            norm = new_run_length_probs.sum()
            if norm > 1e-300:
                new_run_length_probs /= norm

            kappa_new = np.append(self.kappa0, kappa + 1)
            mu_new = np.append(self.mu0, (kappa * mu + x) / (kappa + 1))
            alpha_new = np.append(self.alpha0, alpha + 0.5)
            beta_new = np.append(
                self.beta0, beta + (kappa * (x - mu) ** 2) / (2 * (kappa + 1))
            )

            run_length_probs = new_run_length_probs
            alpha, beta, mu, kappa = alpha_new, beta_new, mu_new, kappa_new
            changepoint_probs[t] = new_run_length_probs[0]

        return changepoint_probs


# ═══════════════════════════════════════════════════════════════════════════
# 3.  FORWARD FILTER (causal — no future look-ahead)
# ═══════════════════════════════════════════════════════════════════════════

def forward_filter(model: StudentTHMM, X_scaled: np.ndarray) -> np.ndarray:
    """
    Causal forward filtering: P(s_t | y_1:t) at each time step.

    Uses only past and present observations — no future look-ahead.
    Returns shape (T, n_states).
    """
    T = len(X_scaled)
    n = model.n_components
    A = model.transmat_
    pi = model.startprob_
    log_B = model._compute_log_likelihood(X_scaled)

    filtered = np.zeros((T, n))
    alpha = pi * np.exp(log_B[0] - log_B[0].max())
    alpha /= alpha.sum() + 1e-300
    filtered[0] = alpha

    for t in range(1, T):
        alpha_pred = A.T @ alpha
        log_b = log_B[t]
        b = np.exp(log_b - log_b.max())
        alpha = alpha_pred * b
        alpha /= alpha.sum() + 1e-300
        filtered[t] = alpha

    return filtered


# ═══════════════════════════════════════════════════════════════════════════
# 4.  MINIMUM DURATION FILTER
# ═══════════════════════════════════════════════════════════════════════════

def apply_min_duration(states: pd.Series, min_days: int = 5) -> pd.Series:
    """Suppress regime runs shorter than min_days by merging into neighbours."""
    cleaned = states.copy()
    valid = cleaned.dropna()
    if len(valid) == 0:
        return cleaned

    values = valid.values
    indices = valid.index

    # Build run-length encoding
    runs: list[dict] = []
    i = 0
    while i < len(values):
        j = i
        while j < len(values) and values[j] == values[i]:
            j += 1
        runs.append({"start": i, "end": j - 1, "label": values[i], "length": j - i})
        i = j

    changed = True
    while changed:
        changed = False
        new_runs: list[dict] = []
        for idx, run in enumerate(runs):
            if run["length"] < min_days:
                changed = True
                prev_label = runs[idx - 1]["label"] if idx > 0 else None
                next_label = runs[idx + 1]["label"] if idx < len(runs) - 1 else None
                if prev_label is None:
                    replace = next_label
                elif next_label is None:
                    replace = prev_label
                else:
                    prev_len = runs[idx - 1]["length"]
                    next_len = runs[idx + 1]["length"]
                    replace = prev_label if prev_len >= next_len else next_label
                if new_runs and new_runs[-1]["label"] == replace:
                    new_runs[-1]["end"] = run["end"]
                    new_runs[-1]["length"] += run["length"]
                else:
                    new_runs.append({
                        "start": run["start"],
                        "end": run["end"],
                        "label": replace,
                        "length": run["length"],
                    })
            else:
                if new_runs and new_runs[-1]["label"] == run["label"]:
                    new_runs[-1]["end"] = run["end"]
                    new_runs[-1]["length"] += run["length"]
                else:
                    new_runs.append(run.copy())
        runs = new_runs

    new_values = np.empty(len(values), dtype=object)
    for run in runs:
        new_values[run["start"] : run["end"] + 1] = run["label"]
    cleaned.loc[indices] = new_values
    return cleaned


# ═══════════════════════════════════════════════════════════════════════════
# 5.  FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════

def _rolling_regression_channel(
    prices: np.ndarray, window: int = 60, n_std: float = 2.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rolling OLS linear regression channel. Returns slope, upper, lower, position."""
    T = len(prices)
    slope = np.full(T, np.nan)
    upper_band = np.full(T, np.nan)
    lower_band = np.full(T, np.nan)
    ch_position = np.full(T, np.nan)

    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_demeaned = x - x_mean
    x_var = (x_demeaned**2).sum()

    for i in range(window - 1, T):
        y = prices[i - window + 1 : i + 1]
        y_mean = y.mean()
        s = (x_demeaned * (y - y_mean)).sum() / x_var
        b = y_mean - s * x_mean
        fitted = s * x + b
        resid = y - fitted
        sigma = resid.std()

        slope[i] = s
        upper_band[i] = fitted[-1] + n_std * sigma
        lower_band[i] = fitted[-1] - n_std * sigma
        width = upper_band[i] - lower_band[i]
        ch_position[i] = (prices[i] - lower_band[i]) / width if width > 1e-8 else 0.5

    return slope, upper_band, lower_band, ch_position


def build_feature_matrix(
    prices: pd.Series,
    vix: pd.Series | None = None,
    window: int = 60,
) -> pd.DataFrame:
    """
    Build the 5-feature observation vector for the HMM.

    Parameters
    ----------
    prices : pd.Series — daily close prices (e.g. SPY)
    vix    : pd.Series — VIX close prices (optional; uses realised vol proxy if None)
    window : int — rolling regression window

    Features
    --------
    channel_slope     : slope / rolling_vol  (dimensionless)
    channel_position  : 0 = at support, 1 = at resistance
    channel_width_pct : band width as % of price
    ewma_vol          : EWMA conditional volatility (span=30)
    vol_index_log     : log(VIX) or log(realised vol * 100) if VIX unavailable
    """
    price_vals = prices.values.squeeze()
    log_returns = np.log(prices / prices.shift(1)).dropna()
    returns_arr = log_returns.values.squeeze()

    # Align to returns index (one shorter than prices)
    price_aligned = prices.reindex(log_returns.index).values.squeeze()

    raw_slope, upper, lower, position = _rolling_regression_channel(price_aligned, window)

    roll_vol = (
        pd.Series(returns_arr, index=log_returns.index)
        .rolling(window, min_periods=window)
        .std()
        .values
    )
    norm_slope = raw_slope / (roll_vol + 1e-10)

    ewma_vol = (
        pd.Series(returns_arr, index=log_returns.index)
        .ewm(span=30, min_periods=30)
        .std()
        .values
    )

    if vix is not None:
        vix_aligned = vix.reindex(log_returns.index, method="ffill").values.squeeze()
        vol_index_log = np.log(np.maximum(vix_aligned, 1e-6))
    else:
        # Fallback: use 20-day realised vol as a VIX proxy
        realised = (
            pd.Series(returns_arr, index=log_returns.index)
            .rolling(20, min_periods=20)
            .std()
            * np.sqrt(252)
            * 100
        )
        vol_index_log = np.log(np.maximum(realised.values, 1e-6))

    width_pct = (upper - lower) / (price_aligned + 1e-10)

    return pd.DataFrame(
        {
            "channel_slope": norm_slope,
            "channel_position": position,
            "channel_width_pct": width_pct,
            "ewma_vol": ewma_vol,
            "vol_index_log": vol_index_log,
        },
        index=log_returns.index,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 6.  REGIME DETECTOR (high-level wrapper)
# ═══════════════════════════════════════════════════════════════════════════

class RegimeDetector:
    """
    3-state Student-t HMM for bull / sideways / bear detection.

    Uses forward filtering (causal) and minimum duration post-processing.
    Labels are assigned using a composite slope - vol_index score so that
    states are consistently named regardless of HMM random init.
    """

    LABEL_BULL = "bull"
    LABEL_SIDEWAYS = "sideways"
    LABEL_BEAR = "bear"

    def __init__(
        self,
        n_states: int = 3,
        channel_window: int = 60,
        n_restarts: int = 30,
        min_duration: int = 5,
        fixed_dof: bool = True,
        init_dof: float = 4.0,
    ):
        self.n_states = n_states
        self.channel_window = channel_window
        self.n_restarts = n_restarts
        self.min_duration = min_duration
        self.fixed_dof = fixed_dof
        self.init_dof = init_dof

        self.model: StudentTHMM | None = None
        self.scaler = StandardScaler()
        self.label_map_: dict[int, str] = {}
        self.filtered_probs_: np.ndarray | None = None
        self.valid_idx_: pd.DatetimeIndex | None = None

    def fit(
        self,
        prices: pd.Series,
        vix: pd.Series | None = None,
    ) -> RegimeDetector:
        """Fit the HMM on training data. Only call on in-sample data."""
        features = build_feature_matrix(prices, vix, self.channel_window)
        X = features.dropna().values
        self.valid_idx_ = features.dropna().index

        if len(X) < self.n_states * 10:
            raise ValueError(
                f"Not enough valid observations ({len(X)}) to fit "
                f"{self.n_states}-state HMM. Need at least {self.n_states * 10}."
            )

        X_scaled = self.scaler.fit_transform(X)

        best_score, best_model = -np.inf, None
        for seed in range(self.n_restarts):
            m = StudentTHMM(
                n_components=self.n_states,
                n_iter=200,
                tol=1e-4,
                random_state=seed,
                fixed_dof=self.fixed_dof,
                init_dof=self.init_dof,
            )
            try:
                m.fit(X_scaled)
                s = m.score(X_scaled)
                if s > best_score:
                    best_score, best_model = s, m
            except Exception:
                continue

        if best_model is None:
            raise RuntimeError("All HMM restarts failed — check your data.")
        self.model = best_model

        # Assign labels using composite slope - vol score
        slope_col = 0  # channel_slope
        vol_col = 4  # vol_index_log
        bull_score = self.model.means_[:, slope_col] - self.model.means_[:, vol_col]
        ranked = np.argsort(bull_score)
        self.label_map_ = {
            ranked[-1]: self.LABEL_BULL,
            ranked[0]: self.LABEL_BEAR,
            ranked[1]: self.LABEL_SIDEWAYS,
        }
        for s in range(self.n_states):
            if s not in self.label_map_:
                self.label_map_[s] = self.LABEL_SIDEWAYS

        logger.info(
            "HMM fitted: best score=%.1f, label_map=%s",
            best_score,
            self.label_map_,
        )
        return self

    def predict(
        self,
        prices: pd.Series,
        vix: pd.Series | None = None,
    ) -> pd.DataFrame:
        """
        Causal forward-filtered regime probabilities on (possibly new) data.

        Uses the scaler fitted during fit() — no data leakage.

        Returns DataFrame with columns [bull, sideways, bear] indexed by date.
        """
        if self.model is None:
            raise RuntimeError("RegimeDetector has not been fitted yet.")

        features = build_feature_matrix(prices, vix, self.channel_window)
        valid = features.dropna()
        if valid.empty:
            return pd.DataFrame(columns=[self.LABEL_BULL, self.LABEL_SIDEWAYS, self.LABEL_BEAR])

        X_scaled = self.scaler.transform(valid.values)
        filtered = forward_filter(self.model, X_scaled)
        self.filtered_probs_ = filtered
        self.valid_idx_ = valid.index

        cols = [self.label_map_[i] for i in range(self.n_states)]
        return pd.DataFrame(filtered, index=valid.index, columns=cols)

    def current_regime_label(
        self,
        prices: pd.Series,
        vix: pd.Series | None = None,
    ) -> str:
        """Return the most likely regime label for the latest observation."""
        proba = self.predict(prices, vix)
        if proba.empty:
            return self.LABEL_SIDEWAYS
        latest = proba.iloc[-1]
        return str(latest.idxmax())


# ═══════════════════════════════════════════════════════════════════════════
# 7.  DEFENSE SCALE FROM REGIME
# ═══════════════════════════════════════════════════════════════════════════

# Default mapping: regime label -> gross exposure multiplier
DEFAULT_REGIME_SCALES: dict[str, float] = {
    RegimeDetector.LABEL_BULL: 1.0,
    RegimeDetector.LABEL_SIDEWAYS: 0.75,
    RegimeDetector.LABEL_BEAR: 0.25,
}


def regime_defense_scale(
    regime_probs: pd.Series,
    regime_scales: dict[str, float] | None = None,
) -> float:
    """
    Map regime posterior probabilities to a gross exposure scalar.

    Parameters
    ----------
    regime_probs  : Series with index [bull, sideways, bear] and values
                    summing to ~1.0 (the latest row from RegimeDetector.predict)
    regime_scales : mapping from regime label to exposure multiplier

    Returns
    -------
    float in [0.25, 1.0] — weighted average of regime scales
    """
    scales = regime_scales or DEFAULT_REGIME_SCALES
    weighted = sum(
        regime_probs.get(label, 0.0) * scale
        for label, scale in scales.items()
    )
    return float(np.clip(weighted, 0.25, 1.0))
