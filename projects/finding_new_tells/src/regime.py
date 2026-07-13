"""5-state Hidden Semi-Markov Model for market regime detection.

Reference: Zakamulin (2023) "Not all bull and bear markets are alike:
insights from a five-state hidden semi-Markov model."

Implementation uses Viterbi EM (hard EM) for fitting and the forward
algorithm for filtered state probabilities (causal — no future data).
States are relabeled post-hoc by conditional mean log-return so that
state 0 = strongest bull, state 4 = strongest bear.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm, poisson
from sklearn.cluster import KMeans


# ---------------------------------------------------------------------------
# HSMM core — forward pass + Viterbi
# ---------------------------------------------------------------------------

def _log_gaussian_diag(X: np.ndarray, means: np.ndarray, log_vars: np.ndarray) -> np.ndarray:
    """Log-likelihood under diagonal Gaussian for each state.

    X        : (T, F)
    means    : (N, F)
    log_vars : (N, F)   log of diagonal variance per state
    Returns  : (T, N)
    """
    T, F = X.shape
    N = means.shape[0]
    out = np.zeros((T, N))
    for j in range(N):
        diff = X - means[j]
        out[:, j] = -0.5 * (np.sum(diff ** 2 / np.exp(log_vars[j]), axis=1)
                             + np.sum(log_vars[j])
                             + F * np.log(2 * np.pi))
    return out


def _log_poisson_dur(lam: np.ndarray, max_dur: int) -> np.ndarray:
    """Log P(duration = d | state j) for d = 1 .. max_dur.

    Duration is modeled as 1 + Poisson(lambda), so minimum duration = 1.
    Returns (N, max_dur), rows normalized to sum to 1 over the truncated range.
    """
    d_arr = np.arange(max_dur)  # 0 .. max_dur-1  →  Poisson realizations for duration-1
    log_p = np.zeros((len(lam), max_dur))
    for j, lj in enumerate(lam):
        log_p[j] = poisson.logpmf(d_arr, max(lj, 1e-6))
    # renormalize truncated distribution
    log_norm = np.log(np.exp(log_p).sum(axis=1, keepdims=True).clip(1e-300))
    return log_p - log_norm


def _forward(log_emit: np.ndarray, log_pi: np.ndarray,
             log_A: np.ndarray, log_p_dur: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """HSMM forward pass (causal / filtered).

    log_emit  : (T, N)  log emission probs
    log_pi    : (N,)    log initial state distribution
    log_A     : (N, N)  log transition matrix (diag = -inf)
    log_p_dur : (N, D)  log duration distribution

    Returns (log_as, log_a), each (T, N):
      log_as[t, j] = log P(o_{0..t-1}, new sojourn in j starts at t)
      log_a[t, j]  = log P(o_{0..t},   state j is active at t)
    """
    T, N = log_emit.shape
    D = log_p_dur.shape[1]
    log_emit_cs = np.cumsum(log_emit, axis=0)  # (T, N)

    log_as = np.full((T, N), -np.inf)
    log_a = np.full((T, N), -np.inf)
    log_as[0] = log_pi

    for t in range(T):
        max_d = min(t + 1, D)
        d_arr = np.arange(1, max_d + 1)          # (max_d,)
        t_prev = t - d_arr                        # (max_d,)

        # Block log-emission from (t - d + 1) to t
        prev_cs = np.where(
            t_prev[:, None] >= 0,
            log_emit_cs[t_prev.clip(0)],          # (max_d, N)
            0.0,
        )
        blocks = log_emit_cs[t] - prev_cs          # (max_d, N)

        sojourn_starts = (t - d_arr + 1)           # always >= 0 because max_d <= t+1
        # terms[k, j] = log_as[sojourn_starts[k], j] + log_p_dur[j, d-1] + block[k, j]
        terms = (log_as[sojourn_starts]            # (max_d, N)
                 + log_p_dur[:, d_arr - 1].T       # (max_d, N)
                 + blocks)                          # (max_d, N)

        # logsumexp over d dimension
        m = terms.max(axis=0, keepdims=True)
        log_a[t] = np.log(np.exp(terms - m).sum(axis=0) + 1e-300) + m[0]

        if t < T - 1:
            # log_as[t+1, j] = logsumexp_i (log_a[t, i] + log_A[i, j])
            mat = log_a[t, :, None] + log_A        # (N, N)
            m2 = mat.max(axis=0, keepdims=True)
            log_as[t + 1] = np.log(np.exp(mat - m2).sum(axis=0) + 1e-300) + m2[0]

    return log_as, log_a


def _viterbi(log_emit: np.ndarray, log_pi: np.ndarray,
             log_A: np.ndarray, log_p_dur: np.ndarray) -> np.ndarray:
    """HSMM Viterbi decoding (MAP state sequence).

    Same structure as forward but uses max instead of logsumexp.
    Returns (T,) array of state indices.
    """
    T, N = log_emit.shape
    D = log_p_dur.shape[1]
    log_emit_cs = np.cumsum(log_emit, axis=0)

    log_delta_s = np.full((T, N), -np.inf)   # best score entering state at t
    log_delta = np.full((T, N), -np.inf)     # best score occupying state at t
    psi_state = np.zeros((T, N), dtype=int)   # backpointer: previous state
    psi_start = np.zeros((T, N), dtype=int)   # backpointer: sojourn start

    log_delta_s[0] = log_pi

    for t in range(T):
        max_d = min(t + 1, D)
        d_arr = np.arange(1, max_d + 1)
        t_prev = t - d_arr

        prev_cs = np.where(
            t_prev[:, None] >= 0,
            log_emit_cs[t_prev.clip(0)],
            0.0,
        )
        blocks = log_emit_cs[t] - prev_cs

        sojourn_starts = t - d_arr + 1
        terms = (log_delta_s[sojourn_starts]
                 + log_p_dur[:, d_arr - 1].T
                 + blocks)  # (max_d, N)

        best_d_idx = terms.argmax(axis=0)           # (N,)
        log_delta[t] = terms[best_d_idx, np.arange(N)]
        psi_start[t] = sojourn_starts[best_d_idx]

        if t < T - 1:
            mat = log_delta[t, :, None] + log_A     # (N, N)
            best_prev = mat.argmax(axis=0)           # (N,)
            log_delta_s[t + 1] = mat[best_prev, np.arange(N)]
            psi_state[t + 1] = best_prev

    # Backtrack
    states = np.empty(T, dtype=int)
    states[T - 1] = log_delta[T - 1].argmax()
    t = T - 1
    while t > 0:
        j = states[t]
        start = psi_start[t, j]
        prev_j = psi_state[start, j] if start > 0 else -1
        for s in range(start, t + 1):
            states[s] = j
        if start > 0:
            states[start - 1] = prev_j
        t = start - 1

    return states


# ---------------------------------------------------------------------------
# HSMM5 class
# ---------------------------------------------------------------------------

@dataclass
class HSMM5:
    """5-state HSMM with diagonal Gaussian emissions and Poisson durations.

    States are always relabeled after fitting so that:
      0 = strongest bull (highest mean return)
      4 = strongest bear  (lowest mean return)
    """
    n_states: int = 5
    max_dur: int = 200        # cap on sojourn duration in days
    n_iter: int = 80          # EM iterations
    tol: float = 1e-4         # log-likelihood convergence threshold
    random_state: int = 42

    # Fitted parameters (set after .fit())
    pi_: np.ndarray = field(default=None, repr=False)
    A_: np.ndarray = field(default=None, repr=False)
    means_: np.ndarray = field(default=None, repr=False)
    log_vars_: np.ndarray = field(default=None, repr=False)
    dur_lambda_: np.ndarray = field(default=None, repr=False)
    state_labels_: list[str] = field(default=None, repr=False)

    # ---------------------------------------------------------------------------

    def _init_params(self, X: np.ndarray) -> None:
        rng = np.random.default_rng(self.random_state)
        N, F = self.n_states, X.shape[1]
        km = KMeans(n_clusters=N, random_state=self.random_state, n_init=5).fit(X)
        self.means_ = km.cluster_centers_.copy()
        self.log_vars_ = np.log(np.var(X, axis=0).clip(1e-6) * np.ones((N, F)))
        # Uniform transition (no self-loops)
        A = np.ones((N, N)) - np.eye(N)
        self.A_ = A / A.sum(axis=1, keepdims=True)
        self.pi_ = np.ones(N) / N
        # Initial expected durations: vary by state (20..80 days)
        self.dur_lambda_ = np.linspace(20, 60, N).astype(float)

    def _compute_log_emit(self, X: np.ndarray) -> np.ndarray:
        return _log_gaussian_diag(X, self.means_, self.log_vars_)

    def _relabel_states(self, X: np.ndarray) -> None:
        """Relabel states by descending conditional mean of first feature (log-return)."""
        states = self.predict(X)
        mean_ret = np.array([X[states == j, 0].mean() if (states == j).any() else 0.0
                             for j in range(self.n_states)])
        order = np.argsort(mean_ret)[::-1]  # highest return → state 0
        self.means_ = self.means_[order]
        self.log_vars_ = self.log_vars_[order]
        self.dur_lambda_ = self.dur_lambda_[order]
        self.pi_ = self.pi_[order]
        # Permute A
        self.A_ = self.A_[order][:, order]
        labels = ["strong_bull", "weak_bull", "sideways", "weak_bear", "strong_bear"]
        self.state_labels_ = labels[:self.n_states]

    # ---------------------------------------------------------------------------
    # Viterbi EM (hard EM)
    # ---------------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "HSMM5":
        """Fit HSMM via Viterbi EM. X: (T, F) array."""
        X = np.asarray(X, dtype=float)
        self._init_params(X)
        N, F = self.n_states, X.shape[1]
        ll_prev = -np.inf

        for _ in range(self.n_iter):
            # E-step: Viterbi decode
            states = self._viterbi_internal(X)

            # M-step from hard assignments
            # --- pi ---
            first = states[0]
            pi = np.zeros(N)
            pi[first] = 1.0
            self.pi_ = pi

            # --- means and variances ---
            for j in range(N):
                mask = states == j
                if mask.sum() < 2:
                    continue
                self.means_[j] = X[mask].mean(axis=0)
                self.log_vars_[j] = np.log(X[mask].var(axis=0).clip(1e-6))

            # --- transitions + duration from segments ---
            segments = self._extract_segments(states)
            trans_counts = np.zeros((N, N))
            dur_sums = np.zeros(N)
            dur_counts = np.zeros(N)
            for k, (j, start, end) in enumerate(segments):
                dur_sums[j] += end - start
                dur_counts[j] += 1
                if k + 1 < len(segments):
                    next_j = segments[k + 1][0]
                    trans_counts[j, next_j] += 1

            for j in range(N):
                row = trans_counts[j].copy()
                row[j] = 0.0
                if row.sum() > 0:
                    self.A_[j] = row / row.sum()
            for j in range(N):
                if dur_counts[j] > 0:
                    # Poisson rate = mean(duration) - 1
                    self.dur_lambda_[j] = max(dur_sums[j] / dur_counts[j] - 1, 0.5)

            # Convergence check
            log_emit = self._compute_log_emit(X)
            log_p_dur = _log_poisson_dur(self.dur_lambda_, self.max_dur)
            log_pi = np.log(self.pi_.clip(1e-300))
            log_A_mat = np.where(np.eye(N, dtype=bool), -np.inf,
                                 np.log(self.A_.clip(1e-300)))
            _, log_a = _forward(log_emit, log_pi, log_A_mat, log_p_dur)
            m = log_a[-1].max()
            ll = np.log(np.exp(log_a[-1] - m).sum()) + m
            if abs(ll - ll_prev) < self.tol:
                break
            ll_prev = ll

        self._relabel_states(X)
        return self

    def _viterbi_internal(self, X: np.ndarray) -> np.ndarray:
        N = self.n_states
        log_emit = self._compute_log_emit(X)
        log_p_dur = _log_poisson_dur(self.dur_lambda_, self.max_dur)
        log_pi = np.log(self.pi_.clip(1e-300))
        log_A = np.where(np.eye(N, dtype=bool), -np.inf,
                         np.log(self.A_.clip(1e-300)))
        return _viterbi(log_emit, log_pi, log_A, log_p_dur)

    @staticmethod
    def _extract_segments(states: np.ndarray) -> list[tuple[int, int, int]]:
        """Return list of (state, start_idx, end_idx_exclusive)."""
        segments = []
        if len(states) == 0:
            return segments
        cur = states[0]
        start = 0
        for t in range(1, len(states)):
            if states[t] != cur:
                segments.append((int(cur), start, t))
                cur = states[t]
                start = t
        segments.append((int(cur), start, len(states)))
        return segments

    # ---------------------------------------------------------------------------
    # Public inference
    # ---------------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Viterbi MAP state sequence. Returns (T,) int array."""
        return self._viterbi_internal(np.asarray(X, dtype=float))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Filtered (causal) state probabilities from forward pass. Returns (T, N)."""
        X = np.asarray(X, dtype=float)
        N = self.n_states
        log_emit = self._compute_log_emit(X)
        log_p_dur = _log_poisson_dur(self.dur_lambda_, self.max_dur)
        log_pi = np.log(self.pi_.clip(1e-300))
        log_A = np.where(np.eye(N, dtype=bool), -np.inf,
                         np.log(self.A_.clip(1e-300)))
        _, log_a = _forward(log_emit, log_pi, log_A, log_p_dur)
        # Normalize rows to get P(state | data up to t)
        m = log_a.max(axis=1, keepdims=True)
        proba = np.exp(log_a - m)
        proba /= proba.sum(axis=1, keepdims=True).clip(1e-300)
        return proba


# ---------------------------------------------------------------------------
# Walk-forward regime series for research use
# ---------------------------------------------------------------------------

def _normalize_refit_freq(refit_freq: str | pd.DateOffset) -> str | pd.DateOffset:
    """Use pandas offsets for year-end aliases that vary across pandas versions."""
    if not isinstance(refit_freq, str):
        return refit_freq

    freq = refit_freq.upper()
    if freq.endswith("YE"):
        multiple = freq[:-2]
        n = int(multiple) if multiple else 1
        return pd.offsets.YearEnd(n)
    return refit_freq


def build_regime_features(panel: pd.DataFrame) -> np.ndarray:
    """Construct the 2-feature input: (QQQ log-return, QQQ 20d realized vol)."""
    c = panel.get("QQQ_close")
    if c is None:
        raise KeyError("Panel must contain QQQ_close for regime features.")
    log_ret = np.log(c / c.shift(1)).fillna(0).values
    rv20 = c.pct_change(fill_method=None).pow(2).rolling(20).mean().apply(
        lambda x: np.sqrt(252 * x) if not np.isnan(x) else np.nan
    ).ffill().fillna(0).values
    return np.column_stack([log_ret, rv20])


def walk_forward_states(
    panel: pd.DataFrame,
    *,
    min_train_years: int = 5,
    refit_freq: str | pd.DateOffset = "1YE",
    max_dur: int = 200,
    n_iter: int = 80,
) -> tuple[pd.Series, pd.DataFrame]:
    """Fit HSMM with annual expanding-window walk-forward.

    Protocol (standard out-of-sample CV):
      - Fit model on X[:train_end]
      - Apply model to X[train_end:next_train_end]   ← out-of-sample
      - predict_proba (causal forward pass) ensures state at t
        depends only on observations ≤ t.
      - Rows before the first refit remain -1 / NaN (warmup).

    Returns:
      state_series  : pd.Series of int state (0=strong_bull … 4=strong_bear)
      proba_df      : pd.DataFrame (T, 5) of filtered state probabilities
    """
    X = build_regime_features(panel)
    idx = panel.index
    min_rows = min_train_years * 252

    state_arr = np.full(len(idx), -1, dtype=int)
    proba_arr = np.full((len(idx), 5), np.nan)

    # Collect refit boundaries (row indices where training data ends)
    refit_ends: list[int] = []
    for rd in pd.date_range(
        idx[min(min_rows - 1, len(idx) - 1)],
        idx[-1],
        freq=_normalize_refit_freq(refit_freq),
    ):
        te = int(idx.searchsorted(rd, side="right"))
        if te >= min_rows and (not refit_ends or te > refit_ends[-1]):
            refit_ends.append(te)

    if not refit_ends:
        return (
            pd.Series(state_arr, index=idx, name="regime_state"),
            pd.DataFrame(proba_arr, index=idx, columns=[f"p_state_{i}" for i in range(5)]),
        )

    model = HSMM5(max_dur=max_dur, n_iter=n_iter)

    for i, train_end in enumerate(refit_ends):
        # Fit on data strictly before this refit boundary
        try:
            model.fit(X[:train_end])
        except Exception as e:
            warnings.warn(f"HSMM fit failed at train_end={train_end}: {e}", stacklevel=2)
            continue

        # Apply to the OUT-OF-SAMPLE segment: [train_end, next_train_end)
        seg_start = train_end
        seg_end = refit_ends[i + 1] if i + 1 < len(refit_ends) else len(idx)
        if seg_start >= seg_end:
            continue

        try:
            # Forward pass is causal: log_a[t] depends only on X[0..t]
            probas = model.predict_proba(X[seg_start:seg_end])
            state_arr[seg_start:seg_end] = probas.argmax(axis=1)
            proba_arr[seg_start:seg_end] = probas
        except Exception:
            pass

    state_series = pd.Series(state_arr, index=idx, name="regime_state")
    proba_df = pd.DataFrame(
        proba_arr, index=idx,
        columns=[f"p_state_{i}" for i in range(5)],
    )
    return state_series, proba_df
