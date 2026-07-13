"""Train in-memory walk-forward models from completed historical trades.

Use this module inside a year-by-year backtest loop:

1. collect completed trades through year B
2. call `train_models_from_history(...)`
3. score candidate trades in year B+1
4. append completed B+1 trades
5. retrain for B+2

Training requires labels (`pnl_pct` or `is_severe_loss`). Scoring candidate
trades does not.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .artifacts import AnnualModel
from .constants import CONTEXT_POLICY, MODEL_FEATURES, SUPPORTED_SYMBOLS, TARGET_COLUMN, TARGET_DESCRIPTION
from .features import compute_required_features, missing_model_features

SEED = 42
MIN_TRAIN_ROWS = 100
MIN_POSITIVES = 10


def _fit_model(X: np.ndarray, y: np.ndarray) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=0.1,
        random_state=SEED,
        max_iter=2000,
    )
    clf.fit(scaler.transform(X), y)
    return scaler, clf


def _prepare_training_frame(completed_trades: pd.DataFrame) -> pd.DataFrame:
    df = compute_required_features(completed_trades)
    if TARGET_COLUMN not in df.columns:
        if "pnl_pct" not in df.columns:
            raise ValueError("Completed training history must include either is_severe_loss or pnl_pct")
        df[TARGET_COLUMN] = pd.to_numeric(df["pnl_pct"], errors="coerce") <= -1.0

    missing = missing_model_features(df)
    if missing:
        raise ValueError(f"Completed training history is missing required model feature columns: {missing}")

    df["entry_time"] = pd.to_datetime(df["entry_time"])
    return df.dropna(subset=MODEL_FEATURES + [TARGET_COLUMN]).copy()


def train_model_from_history(
    completed_trades: pd.DataFrame,
    symbol: str,
    *,
    predict_year: int | None = None,
    min_train_rows: int = MIN_TRAIN_ROWS,
    min_positives: int = MIN_POSITIVES,
) -> AnnualModel:
    """Train one symbol's L1-logit model from completed historical trades.

    `completed_trades` must contain only history available before the year being
    scored. If `predict_year` is omitted, it defaults to one year after the
    latest completed trade year.
    """
    symbol = symbol.upper()
    if symbol not in SUPPORTED_SYMBOLS:
        raise ValueError(f"Unsupported symbol {symbol!r}; expected one of {SUPPORTED_SYMBOLS}")

    df = _prepare_training_frame(completed_trades)
    if "symbol" in df.columns:
        df = df[df["symbol"].astype(str).str.upper() == symbol].copy()
    if df.empty:
        raise ValueError(f"No completed training rows available for {symbol}")

    train_years = df["entry_time"].dt.year
    train_start_year = int(train_years.min())
    train_end_year = int(train_years.max())
    predict_year = int(predict_year if predict_year is not None else train_end_year + 1)
    if predict_year <= train_end_year:
        raise ValueError(
            f"predict_year={predict_year} must be after train_end_year={train_end_year} "
            "to avoid look-ahead bias"
        )
    if len(df) < min_train_rows:
        raise ValueError(f"{symbol} needs at least {min_train_rows} training rows; got {len(df)}")
    n_positive = int(df[TARGET_COLUMN].astype(int).sum())
    if n_positive < min_positives:
        raise ValueError(f"{symbol} needs at least {min_positives} severe-loss rows; got {n_positive}")

    scaler, clf = _fit_model(df[MODEL_FEATURES].values, df[TARGET_COLUMN].astype(int).values)

    return AnnualModel(
        symbol=symbol,
        predict_year=predict_year,
        train_start_year=train_start_year,
        train_end_year=train_end_year,
        feature_order=MODEL_FEATURES,
        scaler_mean=scaler.mean_.astype(float),
        scaler_scale=scaler.scale_.astype(float),
        intercept=float(clf.intercept_[0]),
        coefficients=clf.coef_[0].astype(float),
        target=TARGET_COLUMN,
        target_description=TARGET_DESCRIPTION,
        context_policy=CONTEXT_POLICY,
        dataset_hash="in_memory_training_history",
        source_path=None,
    )


def train_models_from_history(
    completed_trades: pd.DataFrame,
    symbols: Iterable[str] = SUPPORTED_SYMBOLS,
    *,
    predict_year: int | None = None,
    min_train_rows: int = MIN_TRAIN_ROWS,
    min_positives: int = MIN_POSITIVES,
) -> dict[str, AnnualModel]:
    """Train one in-memory model per requested symbol from completed history."""
    return {
        symbol.upper(): train_model_from_history(
            completed_trades,
            symbol,
            predict_year=predict_year,
            min_train_rows=min_train_rows,
            min_positives=min_positives,
        )
        for symbol in symbols
    }
