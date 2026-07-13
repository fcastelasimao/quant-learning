"""Shared walk-forward fitting, equity metrics, and sizing functions.

Used by research items 12, 17, 18, 19, 20 to avoid code duplication.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


SEED = 42


def fit_predict_walkforward_logit(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    seed: int = SEED,
    C: float = 0.1,
    min_train: int = 100,
    min_positives: int = 10,
) -> pd.Series:
    """Walk-forward L1-logit: train on years < Y, predict on year Y.

    Returns a Series of predicted probabilities aligned to df.index.
    """
    p = pd.Series(np.nan, index=df.index, name=f"p_{target_col}")
    sub = df.dropna(subset=feature_cols + [target_col])
    for y in sorted(sub["year"].unique()):
        train = sub[sub["year"] < y]
        test_idx = sub.index[sub["year"] == y]
        if len(train) < min_train or len(test_idx) == 0:
            continue
        if train[target_col].sum() < min_positives:
            continue
        sc = StandardScaler().fit(train[feature_cols].values)
        m = LogisticRegression(
            penalty="l1", solver="liblinear", C=C,
            random_state=seed, max_iter=2000,
        )
        m.fit(
            sc.transform(train[feature_cols].values),
            train[target_col].astype(int).values,
        )
        p.loc[test_idx] = m.predict_proba(
            sc.transform(sub.loc[test_idx, feature_cols].values)
        )[:, 1]
    return p


def equity_metrics(df: pd.DataFrame, sized_r_col: str) -> dict:
    """Constant-notional Sharpe, MaxDD, Calmar, CAGR from a sized-return column."""
    g = df.dropna(subset=[sized_r_col]).sort_values("exit_time").copy()
    g["exit_date"] = g["exit_time"].dt.normalize()
    g = g.dropna(subset=["exit_date"])
    if g.empty:
        return {}
    start = g["entry_time"].min().normalize()
    end = g["exit_date"].max()
    bdays = pd.bdate_range(start, end)
    daily = g.groupby("exit_date")[sized_r_col].sum().reindex(bdays, fill_value=0.0)
    total_return = float(daily.sum())
    years = len(daily) / 252.0
    mean_d = float(daily.mean())
    std_d = float(daily.std(ddof=1))
    sharpe = mean_d / std_d * np.sqrt(252) if std_d > 0 else np.nan
    eq = 1.0 + daily.cumsum()
    dd = eq - eq.cummax()
    max_dd = float(dd.min())
    return {
        "total_return": total_return,
        "annualized_return": float(total_return / years) if years > 0 else np.nan,
        "sharpe_daily": sharpe,
        "max_drawdown": max_dd,
        "calmar": float((total_return / years) / abs(max_dd)) if years > 0 and max_dd < 0 else np.nan,
        "mean_daily_return": mean_d,
        "std_daily_return": std_d,
    }


def sizing_functions(p: np.ndarray | pd.Series) -> dict[str, np.ndarray | pd.Series]:
    """Standard sizing functions: baseline, linear, sqrt, step."""
    p_clip = np.clip(p, 0.0, 1.0)
    return {
        "baseline_full": np.ones_like(p_clip) if isinstance(p_clip, np.ndarray) else pd.Series(1.0, index=p_clip.index),
        "linear_skip": 1.0 - p_clip,
        "sqrt_skip": np.sqrt(np.clip(1.0 - p_clip, 0.0, 1.0)),
        "step_skip_at_50": np.where(p_clip < 0.5, 1.0, 0.0) if isinstance(p_clip, np.ndarray) else (p_clip < 0.5).astype(float),
    }


def extended_sizing_functions(p: np.ndarray | pd.Series) -> dict[str, np.ndarray | pd.Series]:
    """All sizing functions including aggressive and moderate variants."""
    base = sizing_functions(p)
    p_clip = np.clip(p, 0.0, 1.0)
    base["aggressive_2x"] = np.clip(1.0 - 2.0 * p_clip, 0.0, 1.0)
    base["moderate_1p5x"] = np.clip(1.0 - 1.5 * p_clip, 0.0, 1.0)
    return base
