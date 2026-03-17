"""
HMM Channel Regime Detection  —  v2
=====================================
Improvements over v1:
  1. Volatility-normalised slope
  2. Online forward filtering (causal inference, no future look-ahead)
  3. Walk-forward validation with expanding window
  4. Minimum duration post-processing
  5. Student-t emission distributions (fat tails)
  6. BOCPD breakout detector running in parallel

Dependencies:
    pip install yfinance hmmlearn scikit-learn matplotlib scipy

Usage:
    python hmm_regime_detection_v2.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from scipy.stats import t as student_t, gaussian_kde
from scipy.special import digamma, gammaln
from scipy.optimize import minimize_scalar
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import BaseHMM
import warnings
warnings.filterwarnings('ignore')


# ════════════════════════════════════════════════════════════════════════════
# 1.  DATA
# ════════════════════════════════════════════════════════════════════════════

def get_clean_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download adjusted OHLC and compute log returns."""
    raw = yf.download(ticker, start=start, end=end,
                      auto_adjust=True, progress=False)
    raw = raw.dropna()
    raw['log_return'] = np.log(
        raw['Close'].squeeze() / raw['Close'].squeeze().shift(1)
    )
    return raw.dropna()


# ════════════════════════════════════════════════════════════════════════════
# 2.  FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════════════════

def rolling_regression_channel(prices: np.ndarray, window: int = 60,
                                n_std: float = 2.0):
    """
    Rolling OLS linear regression channel.

    Returns slope, upper_band, lower_band, channel_position — shape (T,).
    First (window-1) values are NaN.
    """
    T           = len(prices)
    slope       = np.full(T, np.nan)
    upper_band  = np.full(T, np.nan)
    lower_band  = np.full(T, np.nan)
    ch_position = np.full(T, np.nan)

    x          = np.arange(window, dtype=float)
    x_mean     = x.mean()
    x_demeaned = x - x_mean
    x_var      = (x_demeaned ** 2).sum()

    for i in range(window - 1, T):
        y      = prices[i - window + 1: i + 1]
        y_mean = y.mean()
        s      = (x_demeaned * (y - y_mean)).sum() / x_var
        b      = y_mean - s * x_mean
        fitted = s * x + b
        resid  = y - fitted
        sigma  = resid.std()

        slope[i]       = s
        upper_band[i]  = fitted[-1] + n_std * sigma
        lower_band[i]  = fitted[-1] - n_std * sigma
        width          = upper_band[i] - lower_band[i]
        ch_position[i] = (
            (prices[i] - lower_band[i]) / width if width > 1e-8 else 0.5
        )

    return slope, upper_band, lower_band, ch_position


def build_feature_matrix(spy: pd.DataFrame, vix: pd.DataFrame,
                          window: int = 60) -> pd.DataFrame:
    """
    Build the 5-feature observation vector for the HMM.

    Features
    --------
    channel_slope     : IMPROVEMENT 1 — slope / rolling_vol  (dimensionless,
                        comparable across assets and volatility regimes)
    channel_position  : 0 = at support, 1 = at resistance
    channel_width_pct : band width as % of price
    ewma_vol          : EWMA conditional volatility (span=30)
    vix_log           : log(VIX)
    """
    prices  = spy['Close'].values.squeeze()
    returns = spy['log_return'].values.squeeze()

    raw_slope, upper, lower, position = rolling_regression_channel(
        prices, window
    )

    # ── Improvement 1: vol-normalised slope ──────────────────────────────
    # rolling std of returns over same window — comparable across vol regimes
    roll_vol = (
        pd.Series(returns)
        .rolling(window, min_periods=window)
        .std()
        .values
    )
    # divide by vol, not price; epsilon avoids division by zero
    norm_slope = raw_slope / (roll_vol + 1e-10)

    ewma_vol = (
        pd.Series(returns)
        .ewm(span=30, min_periods=30)
        .std()
        .values
    )

    vix_aligned = (
        vix['Close']
        .reindex(spy.index, method='ffill')
        .values.squeeze()
    )
    vix_log   = np.log(vix_aligned)
    width_pct = (upper - lower) / prices

    return pd.DataFrame({
        'channel_slope'    : norm_slope,
        'channel_position' : position,
        'channel_width_pct': width_pct,
        'ewma_vol'         : ewma_vol,
        'vix_log'          : vix_log,
    }, index=spy.index)


# ════════════════════════════════════════════════════════════════════════════
# 3.  STUDENT-t HMM  (Improvement 5)
# ════════════════════════════════════════════════════════════════════════════

class StudentTHMM(BaseHMM):
    """
    Gaussian HMM with Student-t emission distributions.

    Each state k has parameters:
        means_[k]  — location vector  (n_features,)
        scales_[k] — scale vector     (n_features,)   (not variance)
        dof_       — degrees of freedom (scalar, shared across states/features)

    The degrees of freedom ν is estimated from data in the M-step via a
    fixed-point update. Setting fixed_dof=True uses dof=4 (typical for
    daily financial returns) and skips ν estimation — faster and more
    stable on short samples.

    The diagonal independence assumption is retained (same as covariance_type
    ='diag' in GaussianHMM) — features are conditionally independent per state.
    """

    def __init__(self, n_components=3, n_iter=200, tol=1e-4,
                 random_state=None, fixed_dof=True, init_dof=4.0):
        super().__init__(
            n_components=n_components,
            n_iter=n_iter,
            tol=tol,
            random_state=random_state,
            params='stmc',   # startprob, transmat, means, covars (repurposed as scales²)
            init_params='stmc',
        )
        self.fixed_dof = fixed_dof
        self.dof_      = float(init_dof)

    # ── hmmlearn interface ────────────────────────────────────────────────

    def _get_n_fit_scalars_per_param(self):
        """Number of free scalars per parameter set (for AIC/BIC)."""
        n, d = self.n_components, self.n_features
        return {
            's': n - 1,
            't': n * (n - 1),
            'm': n * d,
            'c': n * d,
        }

    def _check(self):
        super()._check()
        self.means_  = np.asarray(self.means_)
        self.scales_ = np.asarray(self.scales_)

    def _init(self, X, lengths=None):
        super()._init(X, lengths)
        n, d       = self.n_components, X.shape[1]
        rng        = np.random.RandomState(self.random_state)
        self.means_  = X[rng.choice(len(X), n, replace=False)]
        self.scales_ = np.ones((n, d))
        self.n_features = d

    def _compute_log_likelihood(self, X):
        """
        Log-likelihood of each observation under each state's Student-t.

        Returns array of shape (T, n_states).
        Each feature is treated as independent (diagonal assumption).
        """
        T = len(X)
        log_prob = np.zeros((T, self.n_components))
        ν = self.dof_

        for k in range(self.n_components):
            μ = self.means_[k]    # (d,)
            σ = self.scales_[k]   # (d,)  scale, not variance
            # sum log-pdf over features (independence assumption)
            log_prob[:, k] = np.sum(
                student_t.logpdf(X, df=ν, loc=μ, scale=σ), axis=1
            )
        return log_prob

    def _generate_sample_from_state(self, state, random_state=None):
        rng = np.random.RandomState(random_state)
        return student_t.rvs(
            df=self.dof_,
            loc=self.means_[state],
            scale=self.scales_[state],
            random_state=rng,
        )

    def _initialize_sufficient_statistics(self):
        stats = super()._initialize_sufficient_statistics()
        stats['post']   = np.zeros(self.n_components)
        stats['sum_w']  = np.zeros((self.n_components, self.n_features))
        stats['sum_wx'] = np.zeros((self.n_components, self.n_features))
        stats['sum_wd'] = np.zeros((self.n_components, self.n_features))
        return stats

    def _accumulate_sufficient_statistics(self, stats, X, framelogprob,
                                          posteriors, fwdlattice, bwdlattice):
        super()._accumulate_sufficient_statistics(
            stats, X, framelogprob, posteriors, fwdlattice, bwdlattice
        )
        ν = self.dof_
        for k in range(self.n_components):
            μ = self.means_[k]
            σ = self.scales_[k]
            γ = posteriors[:, k]          # (T,) soft assignments

            # Mahalanobis-like delta² per feature
            δ2 = ((X - μ) / σ) ** 2      # (T, d)

            # E-step weights for Student-t:  u_{tk} = (ν + d) / (ν + δ²_tk)
            # Here we use per-feature version (diagonal assumption)
            u = (ν + 1) / (ν + δ2)       # (T, d)

            stats['post'][k]      += γ.sum()
            stats['sum_w'][k]     += (γ[:, None] * u).sum(axis=0)
            stats['sum_wx'][k]    += (γ[:, None] * u * X).sum(axis=0)
            stats['sum_wd'][k]    += (γ[:, None] * u * δ2).sum(axis=0)

    def _do_mstep(self, stats):
        super()._do_mstep(stats)

        # M-step for means and scales
        for k in range(self.n_components):
            w  = stats['sum_w'][k]    # (d,)
            wx = stats['sum_wx'][k]   # (d,)
            wd = stats['sum_wd'][k]   # (d,)
            N  = stats['post'][k]

            if N > 1e-6:
                self.means_[k]  = wx / w
                self.scales_[k] = np.sqrt(wd / N)
                self.scales_[k] = np.maximum(self.scales_[k], 1e-6)

        # M-step for ν (fixed-point update, only if not fixed)
        if not self.fixed_dof:
            self.dof_ = self._update_dof(stats)

    def _update_dof(self, stats):
        """
        Fixed-point update for degrees of freedom ν.
        Solves:  -ψ(ν/2) + log(ν/2) + 1 + (1/N)Σ_k Σ_t γ_tk(log u_tk - u_tk) = 0
        Uses scalar optimisation over ν ∈ (2, 100).
        """
        N   = stats['post'].sum()
        if N < 1e-6:
            return self.dof_

        # approximate objective using accumulated stats
        def neg_obj(ν):
            return -(
                -np.log(2) + digamma((ν + 1) / 2) - digamma(ν / 2)
                + np.log(ν / 2) - (ν + 1) / 2 * (
                    np.log(1 + stats['sum_wd'].sum() / (ν * N)) -
                    stats['sum_wd'].sum() / (ν * N)
                )
            )

        result = minimize_scalar(neg_obj, bounds=(2.01, 100.0), method='bounded')
        return result.x if result.success else self.dof_

    # Expose covars_ as scales_² so hmmlearn's parent class is satisfied
    @property
    def covars_(self):
        return self.scales_ ** 2

    @covars_.setter
    def covars_(self, value):
        self.scales_ = np.sqrt(np.abs(value))


# ════════════════════════════════════════════════════════════════════════════
# 4.  BOCPD — Bayesian Online Changepoint Detection  (Improvement 6)
# ════════════════════════════════════════════════════════════════════════════

class BOCPD:
    """
    Bayesian Online Changepoint Detection (Adams & MacKay, 2007).

    Uses a Normal-Gamma conjugate prior for Gaussian observations.
    Outputs P(changepoint at t) for each time step causally.

    Parameters
    ----------
    hazard_rate : expected fraction of days with a changepoint.
                  1/20 means expecting a change every ~20 days on average.
                  Higher values → more sensitive, more alerts.
    alpha0, beta0, mu0, kappa0 : Normal-Gamma hyperparameters.
                  beta0=0.1 gives a tighter prior that adapts faster to
                  new regimes than the default beta0=1.0.
    """

    def __init__(self, hazard_rate: float = 1 / 20,
                 alpha0: float = 1.0, beta0: float = 0.1,
                 mu0: float = 0.0, kappa0: float = 1.0):
        self.h       = hazard_rate
        self.alpha0  = alpha0
        self.beta0   = beta0
        self.mu0     = mu0
        self.kappa0  = kappa0

    def run(self, data: np.ndarray) -> np.ndarray:
        """
        Process a 1-D series and return P(changepoint at t) for each t.

        Parameters
        ----------
        data : 1-D array of observations (e.g. log returns)

        Returns
        -------
        changepoint_probs : array of shape (T,)
        """
        T = len(data)

        # Sufficient statistics for each possible run length
        # We maintain a distribution over run lengths R_t ∈ {0, 1, ..., t}
        # R_t = 0 means a changepoint just occurred at t

        # Initialise: at t=0, run length is 0 with prob 1
        run_length_probs = np.array([1.0])

        # Hyperparameters for each run length hypothesis
        alpha = np.array([self.alpha0])
        beta  = np.array([self.beta0])
        mu    = np.array([self.mu0])
        kappa = np.array([self.kappa0])

        changepoint_probs = np.zeros(T)

        for t, x in enumerate(data):
            # ── Predictive probability under each run-length hypothesis ──
            # Student-t predictive distribution with 2*alpha degrees of freedom
            pred_mean = mu
            pred_var  = beta * (kappa + 1) / (alpha * kappa)
            pred_df   = 2 * alpha

            log_pred = student_t.logpdf(x, df=pred_df,
                                         loc=pred_mean, scale=np.sqrt(pred_var))
            pred = np.exp(log_pred - log_pred.max())   # numerically stable

            # ── Joint probability: run length × predictive ────────────────
            joint = run_length_probs * pred

            # ── Growth probabilities (no changepoint) ────────────────────
            growth = joint * (1 - self.h)

            # ── Changepoint probability (run resets to 0) ─────────────────
            cp_mass = (joint * self.h).sum()

            # ── New run-length distribution ───────────────────────────────
            new_run_length_probs = np.append(cp_mass, growth)

            # Normalise
            norm = new_run_length_probs.sum()
            if norm > 1e-300:
                new_run_length_probs /= norm

            # ── Update sufficient statistics ──────────────────────────────
            # For run-length r: update using the last r observations
            # Conjugate updates for Normal-Gamma:
            kappa_new  = np.append(self.kappa0,  kappa  + 1)
            mu_new     = np.append(self.mu0,     (kappa * mu + x) / (kappa + 1))
            alpha_new  = np.append(self.alpha0,  alpha  + 0.5)
            beta_new   = np.append(
                self.beta0,
                beta + (kappa * (x - mu) ** 2) / (2 * (kappa + 1))
            )

            run_length_probs = new_run_length_probs
            alpha, beta, mu, kappa = alpha_new, beta_new, mu_new, kappa_new

            # P(changepoint at t) = mass on run length 0
            changepoint_probs[t] = new_run_length_probs[0]

        return changepoint_probs


# ════════════════════════════════════════════════════════════════════════════
# 5.  FORWARD FILTER  (Improvement 2)
# ════════════════════════════════════════════════════════════════════════════

def forward_filter(model: StudentTHMM, X_scaled: np.ndarray) -> np.ndarray:
    """
    Causal forward filtering: P(s_t | y_1:t) at each time step.

    Uses only past and present observations — no future look-ahead.
    This is what you would have computed in real time.

    Compare with model.predict_proba() which uses the full Baum-Welch
    smoother (past AND future) — cleaner but not causal.

    Parameters
    ----------
    model    : fitted StudentTHMM
    X_scaled : standardised feature matrix, shape (T, n_features)

    Returns
    -------
    filtered_probs : shape (T, n_states)  — P(s_t | y_1:t)
    """
    T        = len(X_scaled)
    n        = model.n_components
    A        = model.transmat_           # (n, n)
    π        = model.startprob_          # (n,)

    # Emission log-likelihoods from Student-t model
    log_B    = model._compute_log_likelihood(X_scaled)   # (T, n)

    filtered = np.zeros((T, n))
    α        = π * np.exp(log_B[0] - log_B[0].max())
    α       /= α.sum() + 1e-300
    filtered[0] = α

    for t in range(1, T):
        # Predict: α_pred = A^T · α_{t-1}
        α_pred = A.T @ α                              # (n,)

        # Update: multiply by emission likelihood
        log_b  = log_B[t]
        b      = np.exp(log_b - log_b.max())          # (n,)  numerically stable
        α      = α_pred * b
        norm   = α.sum()
        α     /= norm + 1e-300
        filtered[t] = α

    return filtered


# ════════════════════════════════════════════════════════════════════════════
# 6.  MINIMUM DURATION FILTER  (Improvement 4)
# ════════════════════════════════════════════════════════════════════════════

def apply_min_duration(states: pd.Series, min_days: int = 5) -> pd.Series:
    """
    Suppress regime runs shorter than min_days by merging them into the
    longer adjacent regime.

    Algorithm
    ---------
    1. Identify all runs (consecutive identical states).
    2. For each run shorter than min_days, replace it with whichever
       neighbour has the longer run. If both neighbours are equal, use that.
       If the short run is at the boundary, use the single available neighbour.

    Parameters
    ----------
    states   : pd.Series of regime labels (may contain NaN at start)
    min_days : minimum run length to keep (trading days)

    Returns
    -------
    cleaned  : pd.Series with short runs merged
    """
    cleaned = states.copy()
    valid   = cleaned.dropna()

    if len(valid) == 0:
        return cleaned

    # Build run-length encoding
    values  = valid.values
    indices = valid.index

    # Identify run boundaries
    runs = []
    i    = 0
    while i < len(values):
        j = i
        while j < len(values) and values[j] == values[i]:
            j += 1
        runs.append({'start': i, 'end': j - 1, 'label': values[i],
                     'length': j - i})
        i = j

    # Iteratively merge short runs (repeat until stable)
    changed = True
    while changed:
        changed = False
        new_runs = []
        skip     = False

        for idx, run in enumerate(runs):
            if skip:
                skip = False
                continue

            if run['length'] < min_days:
                changed = True
                # Determine replacement label from neighbours
                prev_label = runs[idx - 1]['label'] if idx > 0 else None
                next_label = runs[idx + 1]['label'] if idx < len(runs) - 1 else None

                if prev_label is None:
                    replace = next_label
                elif next_label is None:
                    replace = prev_label
                else:
                    prev_len = runs[idx - 1]['length']
                    next_len = runs[idx + 1]['length']
                    replace  = prev_label if prev_len >= next_len else next_label

                # Merge: extend previous run or create new run at start
                if new_runs and new_runs[-1]['label'] == replace:
                    new_runs[-1]['end']    = run['end']
                    new_runs[-1]['length'] += run['length']
                else:
                    new_runs.append({'start': run['start'], 'end': run['end'],
                                     'label': replace,
                                     'length': run['length']})
            else:
                if new_runs and new_runs[-1]['label'] == run['label']:
                    new_runs[-1]['end']    = run['end']
                    new_runs[-1]['length'] += run['length']
                else:
                    new_runs.append(run.copy())

        runs = new_runs

    # Reconstruct series from runs
    new_values = np.empty(len(values), dtype=object)
    for run in runs:
        new_values[run['start']: run['end'] + 1] = run['label']

    cleaned.loc[indices] = new_values
    return cleaned


# ════════════════════════════════════════════════════════════════════════════
# 7.  CHANNEL REGIME DETECTOR
# ════════════════════════════════════════════════════════════════════════════

class ChannelRegimeDetector:
    """
    3-state Student-t HMM for bull / bear / sideways channel detection.

    Key differences from v1
    -----------------------
    - Student-t emissions (fat tails)
    - Forward filtering instead of Viterbi (causal)
    - Minimum duration post-processing on hard labels
    - BOCPD runs in parallel and is stored as breakout_probs_
    - Scaler is an explicit attribute so walk-forward can refit it

    Parameters
    ----------
    n_states    : number of hidden states
    window      : rolling regression window (trading days)
    n_restarts  : random restarts for Baum-Welch
    min_dur     : minimum regime duration (trading days)
    fixed_dof   : if True, use dof=4 for Student-t (faster, more stable)
    hazard_rate : BOCPD expected changepoint frequency
    """

    def __init__(self, n_states: int = 3, window: int = 60,
                 n_restarts: int = 30, min_dur: int = 5,
                 fixed_dof: bool = True, hazard_rate: float = 1 / 20):
        self.n_states    = n_states
        self.window      = window
        self.n_restarts  = n_restarts
        self.min_dur     = min_dur
        self.fixed_dof   = fixed_dof
        self.hazard_rate = hazard_rate

        self.model           = None
        self.scaler          = StandardScaler()
        self.features_       = None
        self.states_         = None        # hard labels after min-duration filter
        self.filtered_probs_ = None        # (T, n_states) causal posteriors
        self.breakout_probs_ = None        # (T,) BOCPD changepoint probability
        self.label_map_      = {}
        self.valid_idx_      = None

    def fit(self, spy: pd.DataFrame, vix: pd.DataFrame) -> 'ChannelRegimeDetector':

        # ── Build features ────────────────────────────────────────────────
        features        = build_feature_matrix(spy, vix, self.window)
        self.features_  = features
        X               = features.dropna().values
        self.valid_idx_ = features.dropna().index
        X_scaled        = self.scaler.fit_transform(X)

        # ── Fit Student-t HMM with multiple restarts ──────────────────────
        best_score, best_model = -np.inf, None
        for seed in range(self.n_restarts):
            m = StudentTHMM(
                n_components = self.n_states,
                n_iter       = 200,
                tol          = 1e-4,
                random_state = seed,
                fixed_dof    = self.fixed_dof,
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

        # ── Improvement 2: Forward filtering (causal posteriors) ──────────
        self.filtered_probs_ = forward_filter(self.model, X_scaled)

        # ── Assign regime labels using composite slope − VIX score ──────────
        # Sorting by slope alone fails when two states both have negative slopes
        # (one mild bear, one strong bear). Subtracting the VIX mean from the
        # slope mean gives a single "bull-ness" score that separates all three
        # states cleanly:
        #   bull     → high slope, low VIX  → large positive score
        #   bear     → low slope,  high VIX → large negative score
        #   sideways → near-zero slope, moderate VIX → score near zero
        slope_col  = 0   # channel_slope is feature 0
        vix_col    = 4   # vix_log is feature 4
        bull_score = (self.model.means_[:, slope_col]
                      - self.model.means_[:, vix_col])
        ranked = np.argsort(bull_score)   # ascending: bear → sideways → bull
        self.label_map_ = {
            ranked[-1]: 'bull',
            ranked[0] : 'bear',
            ranked[1] : 'sideways',
        }
        for s in range(self.n_states):
            if s not in self.label_map_:
                self.label_map_[s] = 'sideways'

        # Hard labels from argmax of filtered posteriors
        raw_hard = np.argmax(self.filtered_probs_, axis=1)
        raw_states = pd.Series(index=spy.index, dtype=object)
        raw_states.loc[self.valid_idx_] = [
            self.label_map_[s] for s in raw_hard
        ]

        # ── Improvement 4: Minimum duration filter ────────────────────────
        self.states_ = apply_min_duration(raw_states, self.min_dur)

        # ── Improvement 6: BOCPD ──────────────────────────────────────────
        # BOCPD is run on standardised returns so the Normal-Gamma prior
        # operates at unit scale. Raw log returns (~0.01) make the prior
        # adapt far too slowly and probabilities never exceed the threshold.
        returns_valid = spy['log_return'].reindex(self.valid_idx_).values.squeeze()
        ret_std       = returns_valid.std()
        ret_scaled    = returns_valid / (ret_std + 1e-10)   # unit-scale
        bocpd         = BOCPD(hazard_rate=self.hazard_rate)
        self.breakout_probs_ = pd.Series(
            bocpd.run(ret_scaled), index=self.valid_idx_
        ).reindex(spy.index)

        return self

    def predict_proba(self) -> pd.DataFrame:
        """
        Causal filtered posteriors P(regime | y_1:t).
        Column order: bull, sideways, bear.
        """
        cols = [self.label_map_[i] for i in range(self.n_states)]
        return pd.DataFrame(
            self.filtered_probs_,
            index   = self.valid_idx_,
            columns = cols,
        )

    def score_on(self, spy: pd.DataFrame, vix: pd.DataFrame) -> float:
        """
        Out-of-sample log-likelihood per observation.
        Uses the scaler fitted on training data — no data leakage.
        """
        features = build_feature_matrix(spy, vix, self.window).dropna()
        X_scaled = self.scaler.transform(features.values)
        return self.model.score(X_scaled)


# ════════════════════════════════════════════════════════════════════════════
# 8.  WALK-FORWARD VALIDATION  (Improvement 3)
# ════════════════════════════════════════════════════════════════════════════

def walk_forward_validation(spy: pd.DataFrame, vix: pd.DataFrame,
                             initial_train_years: int = 2,
                             step_months: int = 6,
                             window: int = 60,
                             n_restarts: int = 15) -> pd.DataFrame:
    """
    Expanding-window walk-forward validation.

    Fits a fresh detector on [start, train_end], evaluates out-of-sample
    log-likelihood on the next step_months of data.

    The scaler is always fit on training data only — no leakage.

    Parameters
    ----------
    spy, vix             : full data series
    initial_train_years  : size of first training window
    step_months          : how far to advance the window each fold
    window               : regression window passed to detector
    n_restarts           : restarts per fold (reduce for speed)

    Returns
    -------
    results : DataFrame with columns [train_end, test_start, test_end,
                                       test_loglik, n_test_days]
    """
    dates      = spy.index
    start_date = dates[0]
    end_date   = dates[-1]

    train_end = start_date + pd.DateOffset(years=initial_train_years)
    records   = []
    fold      = 0

    print(f"\n── Walk-forward validation ──────────────────────────────────")
    print(f"  Initial train: {start_date.date()} → {train_end.date()}")
    print(f"  Step: {step_months} months\n")

    while True:
        test_start = train_end
        test_end   = train_end + pd.DateOffset(months=step_months)

        if test_end > end_date:
            break

        # Slice data — strictly no future leakage
        spy_train = spy[spy.index <= train_end]
        vix_train = vix[vix.index <= train_end]
        spy_test  = spy[(spy.index > train_end) & (spy.index <= test_end)]
        vix_test  = vix[(vix.index > train_end) & (vix.index <= test_end)]

        if len(spy_train) < window * 2 or len(spy_test) < 10:
            train_end = test_end
            continue

        try:
            detector = ChannelRegimeDetector(
                window=window, n_restarts=n_restarts
            )
            detector.fit(spy_train, vix_train)
            test_ll      = detector.score_on(spy_test, vix_test)
            n_test       = len(spy_test)
            # Normalise by number of test observations so folds are comparable
            test_ll_norm = test_ll / n_test if n_test > 0 else np.nan

            records.append({
                'fold'          : fold,
                'train_end'     : train_end.date(),
                'test_start'    : test_start.date(),
                'test_end'      : test_end.date(),
                'test_loglik'   : round(test_ll, 4),
                'loglik_per_obs': round(test_ll_norm, 4),
                'n_test_days'   : n_test,
            })

            print(f"  Fold {fold:>2d} | train→{train_end.date()} "
                  f"| test {test_start.date()}→{test_end.date()} "
                  f"| logL = {test_ll:+.1f} "
                  f"| logL/obs = {test_ll_norm:+.4f} "
                  f"| n = {n_test}")

        except Exception as e:
            print(f"  Fold {fold:>2d} failed: {e}")

        train_end = test_end
        fold     += 1

    results = pd.DataFrame(records)
    if len(results):
        print(f"\n  Mean logL/obs across folds : {results['loglik_per_obs'].mean():+.4f}")
        print(f"  Std  logL/obs              :  {results['loglik_per_obs'].std():.4f}")
        print(f"  Best fold  : {results.loc[results['loglik_per_obs'].idxmax(), 'test_start']}"
              f"  ({results['loglik_per_obs'].max():+.4f})")
        print(f"  Worst fold : {results.loc[results['loglik_per_obs'].idxmin(), 'test_start']}"
              f"  ({results['loglik_per_obs'].min():+.4f})")
        _plot_walk_forward(results)

    return results


def _plot_walk_forward(results: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color('#21262d')
    ax.spines['left'].set_color('#21262d')
    ax.tick_params(colors='#8b949e', labelsize=8)

    folds = results['fold'].values
    lls   = results['loglik_per_obs'].values
    ax.bar(folds, lls,
           color=['#22c55e' if v > lls.mean() else '#ef4444' for v in lls],
           alpha=0.7, width=0.6)
    ax.axhline(lls.mean(), color='#f0f6fc', linewidth=1.0,
               linestyle='--', label=f'Mean: {lls.mean():+.4f}')
    ax.set_xlabel('Fold', color='#8b949e')
    ax.set_ylabel('logL / observation (comparable across folds)', color='#8b949e')
    ax.set_title('Walk-forward validation — out-of-sample log-likelihood per observation',
                 color='#f0f6fc', fontsize=11)
    ax.legend(fontsize=9, facecolor='#161b22', labelcolor='#c9d1d9')
    plt.tight_layout()
    plt.savefig('walk_forward_validation.png', dpi=150,
                bbox_inches='tight', facecolor='#0d1117')
    plt.show()
    print("Saved → walk_forward_validation.png")


# ════════════════════════════════════════════════════════════════════════════
# 9.  PLOTTING
# ════════════════════════════════════════════════════════════════════════════

COLORS  = {'bull': '#22c55e', 'bear': '#ef4444', 'sideways': '#94a3b8'}
DARK_BG = '#0d1117'


def _style_ax(ax):
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors='#8b949e', labelsize=8)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color('#21262d')
    ax.spines['left'].set_color('#21262d')


def _draw_candlesticks(ax, ohlc: pd.DataFrame,
                       up_color: str   = '#22c55e',
                       down_color: str = '#ef4444',
                       doji_color: str = '#94a3b8',
                       body_width: float = 0.6,
                       wick_width: float = 0.8) -> None:
    """
    Draw OHLC candlesticks on ax using matplotlib bar/vline primitives.

    Parameters
    ----------
    ax         : matplotlib Axes to draw on
    ohlc       : DataFrame with columns Open, High, Low, Close and a
                 DatetimeIndex. Columns are squeezed to 1-D automatically
                 to handle yfinance MultiIndex output.
    up_color   : candle colour when Close >= Open
    down_color : candle colour when Close <  Open
    doji_color : candle colour when close ~= open (doji candle)
    body_width : fraction of the inter-bar spacing used for the candle body
    wick_width : wick line width in points
    """
    opens  = ohlc['Open'].squeeze().values
    highs  = ohlc['High'].squeeze().values
    lows   = ohlc['Low'].squeeze().values
    closes = ohlc['Close'].squeeze().values
    dates  = ohlc.index

    # Convert dates to matplotlib float for positioning
    x = mdates.date2num(dates)

    # Estimate bar width from median gap between bars
    if len(x) > 1:
        gaps      = np.diff(x)
        med_gap   = np.median(gaps[gaps > 0])
        half_body = med_gap * body_width / 2
    else:
        half_body = 0.3

    for i in range(len(dates)):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]

        # Classify candle
        pct_move = abs(c - o) / (abs(c) + 1e-10)
        if pct_move < 0.0001:
            color = doji_color
        elif c >= o:
            color = up_color
        else:
            color = down_color

        xi = x[i]

        # Wick (high-low line)
        ax.plot([xi, xi], [l, h],
                color=color, linewidth=wick_width,
                solid_capstyle='round', zorder=3)

        # Body (open-close rectangle)
        body_bottom = min(o, c)
        body_height = max(abs(c - o), (h - l) * 0.005)
        ax.bar(xi, body_height,
               bottom=body_bottom,
               width=half_body * 2,
               color=color,
               edgecolor=color,
               linewidth=0,
               zorder=4)

    ax.xaxis_date()


def plot_channel_regimes(spy: pd.DataFrame, vix: pd.DataFrame,
                          detector: ChannelRegimeDetector,
                          title: str = 'SPY',
                          figsize=(18, 14)) -> None:
    """
    Four-panel chart:
      1. Price + channel bands coloured by regime
      2. Causal posterior probabilities
      3. BOCPD changepoint probability
      4. VIX
    """
    prices = spy['Close'].squeeze()
    states = detector.states_
    proba  = detector.predict_proba()

    _, upper, lower, _ = rolling_regression_channel(
        prices.values, detector.window
    )
    upper = pd.Series(upper, index=spy.index)
    lower = pd.Series(lower, index=spy.index)
    mid   = (upper + lower) / 2

    vix_s = vix['Close'].reindex(spy.index, method='ffill').squeeze()

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True,
                              gridspec_kw={'height_ratios': [3.5, 1.2, 1, 1]})
    fig.patch.set_facecolor(DARK_BG)
    for ax in axes:
        _style_ax(ax)

    # ── Panel 1: Candlestick chart + channel ─────────────────────────────
    ax = axes[0]

    price_low  = spy['Low'].squeeze().min()
    price_high = spy['High'].squeeze().max()

    # Regime background shading and channel band fill — drawn first (behind candles)
    for regime, color in COLORS.items():
        mask = (states == regime).values
        ax.fill_between(prices.index,
                        price_low  * 0.97, price_high * 1.03,
                        where=mask, color=color, alpha=0.08, step='post')
        ax.fill_between(prices.index, lower, upper,
                        where=mask, color=color, alpha=0.15,
                        step='post', linewidth=0)

    # Channel boundary lines
    ax.plot(prices.index, upper, color='#58a6ff', linewidth=0.7,
            linestyle='--', alpha=0.6, label='Upper (2σ)')
    ax.plot(prices.index, lower, color='#f85149', linewidth=0.7,
            linestyle='--', alpha=0.6, label='Lower (2σ)')
    ax.plot(prices.index, mid,   color='#8b949e', linewidth=0.5,
            linestyle=':', alpha=0.5, label='Midline')

    # Candlesticks drawn on top of everything
    _draw_candlesticks(ax, spy)

    # Legend
    channel_patches = [
        mpatches.Patch(color=COLORS[r], alpha=0.6, label=f'{r} channel')
        for r in ['bull', 'sideways', 'bear']
    ]
    candle_patches = [
        mpatches.Patch(color='#22c55e', alpha=0.9, label='Up candle'),
        mpatches.Patch(color='#ef4444', alpha=0.9, label='Down candle'),
    ]
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + candle_patches + channel_patches,
              loc='upper left', fontsize=8,
              facecolor='#161b22', labelcolor='#c9d1d9',
              framealpha=0.9, edgecolor='#30363d')

    ax.set_ylabel('Price ($)', color='#8b949e', fontsize=10)
    ax.set_title(
        f'{title} · Channel Regime Detection v2  '
        f'(Student-t HMM · Forward Filter · BOCPD)',
        color='#f0f6fc', fontsize=11, pad=10
    )
    ax.set_ylim(price_low * 0.97, price_high * 1.03)

    # ── Panel 2: Causal posterior probabilities ───────────────────────────
    ax = axes[1]
    for regime, color in COLORS.items():
        if regime in proba.columns:
            ax.plot(proba.index, proba[regime], color=color,
                    linewidth=0.9, alpha=0.85, label=f'P({regime})')
    ax.axhline(0.5, color='#334155', linewidth=0.5, linestyle=':')
    ax.set_ylim(0, 1)
    ax.set_ylabel('P(regime|past)', color='#8b949e', fontsize=9)
    ax.legend(loc='upper left', fontsize=7, facecolor='#161b22',
              labelcolor='#c9d1d9', framealpha=0.9)

    # ── Panel 3: BOCPD changepoint probability ────────────────────────────
    ax = axes[2]
    bp = detector.breakout_probs_.dropna()
    ax.fill_between(bp.index, 0, bp.values, color='#f59e0b', alpha=0.6)
    ax.plot(bp.index, bp.values, color='#f59e0b', linewidth=0.8)
    ax.axhline(0.3, color='#f85149', linewidth=0.7, linestyle='--',
               alpha=0.7, label='Alert threshold')
    ax.set_ylim(0, min(bp.max() * 1.2, 1.0))
    ax.set_ylabel('P(breakout)', color='#8b949e', fontsize=9)
    ax.legend(loc='upper right', fontsize=7, facecolor='#161b22',
              labelcolor='#c9d1d9', framealpha=0.9)

    # ── Panel 4: VIX ──────────────────────────────────────────────────────
    ax = axes[3]
    for regime, color in COLORS.items():
        mask = (states == regime).values
        ax.fill_between(vix_s.index, 0, vix_s,
                        where=mask, color=color, alpha=0.2, step='post')
    ax.plot(vix_s.index, vix_s, color='#e3b341', linewidth=0.9)
    ax.fill_between(vix_s.index, 0, vix_s, color='#e3b341', alpha=0.12)
    ax.axhline(30, color='#f85149', linewidth=0.6, linestyle='--',
               alpha=0.7, label='VIX=30')
    ax.set_ylabel('VIX', color='#8b949e', fontsize=9)
    ax.set_ylim(0, vix_s.max() * 1.1)
    ax.legend(loc='upper right', fontsize=7, facecolor='#161b22',
              labelcolor='#c9d1d9', framealpha=0.9)

    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.08)
    fname = f'{title}_regimes_v2.png'
    plt.savefig(fname, dpi=180, bbox_inches='tight', facecolor=DARK_BG)
    plt.show()
    print(f"Saved → {fname}")


# ════════════════════════════════════════════════════════════════════════════
# 10.  DIAGNOSTICS
# ════════════════════════════════════════════════════════════════════════════

def diagnose_feature_distributions(detector: ChannelRegimeDetector) -> None:
    features = detector.features_.copy()
    features['regime'] = detector.states_
    features = features.dropna()

    feat_cols   = ['channel_slope', 'channel_position', 'channel_width_pct',
                   'ewma_vol', 'vix_log']
    feat_labels = ['Slope / vol', 'Channel position', 'Width %',
                   'EWMA vol', 'log(VIX)']

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.patch.set_facecolor(DARK_BG)

    for ax, col, label in zip(axes, feat_cols, feat_labels):
        _style_ax(ax)
        for regime in ['bull', 'sideways', 'bear']:
            vals = features.loc[features['regime'] == regime, col].dropna()
            if len(vals) < 5:
                continue
            ax.hist(vals, bins=40, color=COLORS[regime],
                    alpha=0.35, density=True, label=regime)
            kde = gaussian_kde(vals)
            x   = np.linspace(vals.min(), vals.max(), 200)
            ax.plot(x, kde(x), color=COLORS[regime], linewidth=1.8)
        ax.set_title(label, color='#f0f6fc', fontsize=9)
        ax.set_xlabel('Value', color='#8b949e', fontsize=8)

    axes[0].legend(fontsize=8, facecolor='#161b22',
                   labelcolor='#c9d1d9', framealpha=0.9)
    fig.suptitle('Feature distributions by regime (v2)',
                 color='#f0f6fc', fontsize=12)
    plt.tight_layout()
    plt.savefig('diag_feature_distributions_v2.png', dpi=150,
                bbox_inches='tight', facecolor=DARK_BG)
    plt.show()


def diagnose_regime_durations(detector: ChannelRegimeDetector) -> pd.DataFrame:
    states = detector.states_.dropna()
    runs   = []
    current_regime = states.iloc[0]
    current_start  = states.index[0]

    for date, regime in states.items():
        if regime != current_regime:
            runs.append({'regime': current_regime, 'start': current_start,
                         'end': date,
                         'duration': (date - current_start).days})
            current_regime = regime
            current_start  = date
    runs_df = pd.DataFrame(runs)

    print("\n── Regime duration statistics (after min-duration filter) ──────")
    for regime in ['bull', 'sideways', 'bear']:
        r = runs_df[runs_df['regime'] == regime]['duration']
        if len(r) == 0:
            continue
        print(f"\n  {regime}:")
        print(f"    episodes : {len(r)}")
        print(f"    mean     : {r.mean():.0f} days")
        print(f"    median   : {r.median():.0f} days")
        print(f"    min/max  : {r.min():.0f} / {r.max():.0f} days")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.patch.set_facecolor(DARK_BG)
    for ax, regime in zip(axes, ['bull', 'sideways', 'bear']):
        _style_ax(ax)
        r = runs_df[runs_df['regime'] == regime]['duration']
        if len(r) == 0:
            continue
        ax.hist(r, bins=20, color=COLORS[regime], alpha=0.7, edgecolor='none')
        ax.axvline(r.median(), color='white', linewidth=1.0, linestyle='--',
                   label=f'Median: {r.median():.0f}d')
        ax.set_title(f'{regime} durations', color='#f0f6fc', fontsize=10)
        ax.set_xlabel('Calendar days', color='#8b949e', fontsize=8)
        ax.legend(fontsize=8, facecolor='#161b22', labelcolor='#c9d1d9')

    plt.tight_layout()
    plt.savefig('diag_regime_durations_v2.png', dpi=150,
                bbox_inches='tight', facecolor=DARK_BG)
    plt.show()
    return runs_df


def diagnose_transitions(detector: ChannelRegimeDetector) -> None:
    regimes = [detector.label_map_[i] for i in range(detector.n_states)]
    trans   = detector.model.transmat_

    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    im = ax.imshow(trans, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, label='Probability')
    ax.set_xticks(range(len(regimes)))
    ax.set_yticks(range(len(regimes)))
    ax.set_xticklabels([f'→ {r}' for r in regimes],
                       color='#8b949e', fontsize=9)
    ax.set_yticklabels([f'{r} →' for r in regimes],
                       color='#8b949e', fontsize=9)
    for i in range(len(regimes)):
        for j in range(len(regimes)):
            ax.text(j, i, f'{trans[i, j]:.3f}', ha='center', va='center',
                    fontsize=11,
                    color='black' if trans[i, j] > 0.4 else '#c9d1d9')
    ax.set_title('Transition matrix (Student-t HMM)',
                 color='#f0f6fc', fontsize=11)
    plt.tight_layout()
    plt.savefig('diag_transition_matrix_v2.png', dpi=150,
                bbox_inches='tight', facecolor=DARK_BG)
    plt.show()


def diagnose_bocpd(detector: ChannelRegimeDetector,
                   spy: pd.DataFrame, threshold: float = 0.3) -> None:
    """
    Show BOCPD alerts against price, and count how many alerts precede
    a genuine regime transition (within 10 trading days).
    """
    bp     = detector.breakout_probs_.dropna()
    states = detector.states_

    # Identify alert dates
    alerts = bp[bp > threshold].index

    # Identify regime transition dates
    s_valid   = states.dropna()
    trans_dates = s_valid.index[
        (s_valid != s_valid.shift()).fillna(False)
    ]

    # Score: fraction of alerts within 10 days of a transition
    hits = 0
    for alert in alerts:
        diffs = np.abs((trans_dates - alert).days)
        if len(diffs) > 0 and diffs.min() <= 10:
            hits += 1

    precision = hits / len(alerts) if len(alerts) > 0 else 0.0
    print(f"\n── BOCPD diagnostic (threshold={threshold}) ─────────────────")
    print(f"  Total alerts       : {len(alerts)}")
    print(f"  Regime transitions : {len(trans_dates)}")
    print(f"  Precision (alert within 10d of transition): {precision:.1%}")

    fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True,
                              gridspec_kw={'height_ratios': [3, 1.2]})
    fig.patch.set_facecolor(DARK_BG)
    for ax in axes:
        _style_ax(ax)

    ax = axes[0]
    _draw_candlesticks(ax, spy)
    for alert in alerts:
        ax.axvline(alert, color='#f59e0b', linewidth=0.6, alpha=0.5)
    for td in trans_dates:
        ax.axvline(td, color='#e879f9', linewidth=0.8, linestyle='--', alpha=0.6)
    ax.set_ylabel('Price ($)', color='#8b949e')
    ax.set_title('BOCPD alerts (amber) vs regime transitions (purple)',
                 color='#f0f6fc', fontsize=11)

    ax = axes[1]
    ax.fill_between(bp.index, 0, bp.values, color='#f59e0b', alpha=0.6)
    ax.plot(bp.index, bp.values, color='#f59e0b', linewidth=0.8)
    ax.axhline(threshold, color='#f85149', linewidth=0.7,
               linestyle='--', alpha=0.8)
    ax.set_ylabel('P(breakout)', color='#8b949e')
    ax.set_ylim(0, min(bp.max() * 1.2, 1.0))

    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.06)
    plt.savefig('diag_bocpd.png', dpi=150, bbox_inches='tight',
                facecolor=DARK_BG)
    plt.show()
    print("Saved → diag_bocpd.png")


# ════════════════════════════════════════════════════════════════════════════
# 11.  MAIN
# ════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    # ── Configuration ──────────────────────────────────────────────────────
    TICKER              = 'SPY'
    START               = '2020-01-01'
    END                 = '2025-01-01'
    WINDOW              = 60
    MIN_DURATION        = 5       # trading days
    N_RESTARTS          = 30
    FIXED_DOF           = True    # set False to estimate ν from data (slower)
    HAZARD_RATE         = 1 / 20  # BOCPD: expect changepoint every ~20 days
    RUN_WALK_FORWARD    = True
    RUN_DIAGNOSTICS     = True

    # ── Download data ──────────────────────────────────────────────────────
    print("Downloading data…")
    spy = get_clean_data(TICKER, START, END)
    vix = yf.download('^VIX', start=START, end=END,
                      auto_adjust=True, progress=False)

    # ── Fit detector ───────────────────────────────────────────────────────
    print("Fitting Student-t HMM with forward filtering…")
    detector = ChannelRegimeDetector(
        n_states    = 3,
        window      = WINDOW,
        n_restarts  = N_RESTARTS,
        min_dur     = MIN_DURATION,
        fixed_dof   = FIXED_DOF,
        hazard_rate = HAZARD_RATE,
    )
    detector.fit(spy, vix)

    # ── Summary ────────────────────────────────────────────────────────────
    states = detector.states_.dropna()
    print("\n── Regime summary (% of days) ───────────────────────────────")
    for regime in ['bull', 'sideways', 'bear']:
        pct = (states == regime).mean() * 100
        print(f"  {regime:>10s}: {pct:.1f}%")

    print("\n── Transition matrix ────────────────────────────────────────")
    trans = pd.DataFrame(
        detector.model.transmat_,
        index  =[detector.label_map_[i] for i in range(3)],
        columns=[detector.label_map_[i] for i in range(3)],
    )
    print(trans.round(3))

    print("\n── Student-t emission parameters ────────────────────────────")
    feat_names = ['slope/vol', 'position', 'width_pct', 'ewma_vol', 'vix_log']
    for i in range(3):
        regime = detector.label_map_[i]
        print(f"\n  {regime}  (dof={detector.model.dof_:.1f})")
        for name, mean, scale in zip(feat_names,
                                      detector.model.means_[i],
                                      detector.model.scales_[i]):
            print(f"    {name:>12s}:  μ={mean:+.3f}  σ={scale:.3f}")

    # ── Main chart ─────────────────────────────────────────────────────────
    plot_channel_regimes(spy, vix, detector, title=TICKER)

    # ── Diagnostics ────────────────────────────────────────────────────────
    if RUN_DIAGNOSTICS:
        print("\nRunning diagnostics…")
        diagnose_feature_distributions(detector)
        diagnose_regime_durations(detector)
        diagnose_transitions(detector)
        diagnose_bocpd(detector, spy, threshold=0.04)

    # ── Walk-forward validation ────────────────────────────────────────────
    if RUN_WALK_FORWARD:
        wf_results = walk_forward_validation(
            spy, vix,
            initial_train_years = 2,
            step_months         = 6,
            window              = WINDOW,
            n_restarts          = 15,   # fewer restarts for speed
        )
        print("\nWalk-forward results:")
        print(wf_results[['fold','train_end','test_start','test_end',
                           'loglik_per_obs','n_test_days']].to_string(index=False))