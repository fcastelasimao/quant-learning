"""Vote ensemble + regime-conditional softmax strategy.

Decision at time t's close; filled at t+1's open.
Probabilities: p_buy ∝ exp(+s/τ), p_sell ∝ exp(-s/τ), p_hold ∝ exp(α(regime)/τ).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from metrics import REGISTRY, vote_all


# Default temperature (softmax sharpness)
_DEFAULT_TAU: float = 1.0

# Default hold-bias per regime state (0=strong_bull … 4=strong_bear)
# Bear states bias toward holding cash (high α → high p_hold)
_DEFAULT_ALPHA: np.ndarray = np.array([0.0, 0.2, 0.5, 0.8, 1.2])


def _softmax3(s: float, alpha: float, tau: float) -> tuple[float, float, float]:
    """Compute (p_buy, p_hold, p_sell) from score s and hold-bias alpha."""
    log_buy  =  s / tau
    log_hold = alpha / tau
    log_sell = -s / tau
    m = max(log_buy, log_hold, log_sell)
    buy  = np.exp(log_buy  - m)
    hold = np.exp(log_hold - m)
    sell = np.exp(log_sell - m)
    total = buy + hold + sell
    return buy / total, hold / total, sell / total


def decide(
    panel_until_t: pd.DataFrame,
    regime_state_t: int,
    regime_probs_t: np.ndarray,
    *,
    tau: float = _DEFAULT_TAU,
    alpha: np.ndarray | None = None,
    metric_names: list[str] | None = None,
) -> dict:
    """Compute (p_buy, p_hold, p_sell) and per-metric votes at time t.

    Parameters
    ----------
    panel_until_t : pd.DataFrame
        Wide panel up to and including time t (no future data).
    regime_state_t : int
        Current regime state index (0 = strong_bull, 4 = strong_bear).
    regime_probs_t : np.ndarray
        Shape (5,) filtered state probabilities at time t.
    tau : float
        Softmax temperature (higher → more uniform).
    alpha : np.ndarray, optional
        Per-regime hold-bias, shape (n_states,). Defaults to _DEFAULT_ALPHA.
    metric_names : list[str], optional
        Subset of REGISTRY keys to use. Defaults to all voting metrics.

    Returns
    -------
    dict with keys: p_buy, p_hold, p_sell, score, votes, regime_state
    """
    if alpha is None:
        alpha = _DEFAULT_ALPHA

    if metric_names is None:
        metric_names = [n for n, m in REGISTRY.items() if m.status == "voting"]

    # Get last-row votes from each metric
    votes_df = vote_all(panel_until_t, names=metric_names)
    last_votes = votes_df.iloc[-1]

    votes: dict[str, int] = {}
    valid_votes: list[int] = []
    for name in metric_names:
        if name in last_votes.index:
            v = last_votes[name]
            if np.isnan(v):
                votes[name] = 0
            else:
                votes[name] = int(v)
                valid_votes.append(int(v))
        else:
            votes[name] = 0

    score = float(np.mean(valid_votes)) if valid_votes else 0.0

    regime_state_t = 2 if regime_state_t < 0 else int(np.clip(regime_state_t, 0, len(alpha) - 1))
    alpha_t = float(alpha[regime_state_t])

    p_buy, p_hold, p_sell = _softmax3(score, alpha_t, tau)

    return {
        "p_buy":        p_buy,
        "p_hold":       p_hold,
        "p_sell":       p_sell,
        "score":        score,
        "votes":        votes,
        "regime_state": regime_state_t,
    }


def decide_series(
    panel: pd.DataFrame,
    state_series: pd.Series,
    proba_df: pd.DataFrame,
    *,
    tau: float = _DEFAULT_TAU,
    alpha: np.ndarray | None = None,
    metric_names: list[str] | None = None,
) -> pd.DataFrame:
    """Vectorized version of decide() across all time steps.

    Returns DataFrame with columns: p_buy, p_hold, p_sell, score, action.
    Efficient: computes votes once for the full panel, then applies softmax row-wise.
    action ∈ {'buy', 'hold', 'sell'}.
    """
    if alpha is None:
        alpha = _DEFAULT_ALPHA

    if metric_names is None:
        metric_names = [n for n, m in REGISTRY.items() if m.status == "voting"]

    votes_df = vote_all(panel, names=metric_names)
    score_series = votes_df.mean(axis=1).fillna(0.0)

    rows = []
    for t in panel.index:
        s = float(score_series.loc[t]) if t in score_series.index else 0.0
        r = int(state_series.loc[t]) if t in state_series.index else 2
        r = 2 if r < 0 else int(np.clip(r, 0, len(alpha) - 1))
        pb, ph, ps = _softmax3(s, float(alpha[r]), tau)
        rows.append({"p_buy": pb, "p_hold": ph, "p_sell": ps, "score": s})

    result = pd.DataFrame(rows, index=panel.index)
    result["action"] = result[["p_buy", "p_hold", "p_sell"]].idxmax(axis=1).str.replace(
        "p_", "", regex=False
    )
    return result
