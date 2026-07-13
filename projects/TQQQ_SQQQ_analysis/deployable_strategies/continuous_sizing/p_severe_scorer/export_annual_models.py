"""Export annual walk-forward L1-logit artifacts for `p_severe`.

This reproduces item 17's model training loop:

- for prediction year Y, train on all labeled rows with entry year < Y
- target is `is_severe_loss`, equivalent to `pnl_pct <= -1.0`
- features are curated strategy features + regime dummies + daily context
- model is `StandardScaler` followed by L1 LogisticRegression(C=0.1)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .artifacts import DEFAULT_ARTIFACT_DIR
from .constants import CONTEXT_POLICY, MODEL_FEATURES, SUPPORTED_SYMBOLS, TARGET_COLUMN, TARGET_DESCRIPTION
from .features import compute_required_features, missing_model_features
from .training import MIN_POSITIVES, MIN_TRAIN_ROWS, _fit_model


def training_dataset_hash(train: pd.DataFrame) -> str:
    """Stable fingerprint of the rows/features used to fit one annual model."""
    cols = ["entry_time", TARGET_COLUMN] + MODEL_FEATURES
    work = train[cols].copy()
    work["entry_time"] = pd.to_datetime(work["entry_time"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    csv_bytes = work.sort_values(["entry_time"] + MODEL_FEATURES).to_csv(index=False, float_format="%.12g").encode()
    return hashlib.sha256(csv_bytes).hexdigest()


def _artifact_payload(
    *,
    symbol: str,
    predict_year: int,
    train: pd.DataFrame,
    scaler: StandardScaler,
    model: LogisticRegression,
    source: str | None,
) -> dict:
    return {
        "schema_version": 1,
        "symbol": symbol,
        "predict_year": int(predict_year),
        "train_start_year": int(train["entry_time"].dt.year.min()),
        "train_end_year": int(predict_year - 1),
        "n_train": int(len(train)),
        "n_positive": int(train[TARGET_COLUMN].astype(int).sum()),
        "target": TARGET_COLUMN,
        "target_description": TARGET_DESCRIPTION,
        "context_policy": CONTEXT_POLICY,
        "dataset_hash": training_dataset_hash(train),
        "feature_order": MODEL_FEATURES,
        "scaler_mean": scaler.mean_.astype(float).tolist(),
        "scaler_scale": scaler.scale_.astype(float).tolist(),
        "intercept": float(model.intercept_[0]),
        "coefficients": model.coef_[0].astype(float).tolist(),
        "training": {
            "model": "sklearn.linear_model.LogisticRegression",
            "penalty": "l1",
            "solver": "liblinear",
            "C": 0.1,
            "random_state": SEED,
            "max_iter": 2000,
            "min_train_rows": MIN_TRAIN_ROWS,
            "min_positives": MIN_POSITIVES,
            "source": source,
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
        },
    }


def export_annual_models(
    enriched_trades: pd.DataFrame,
    symbol: str,
    out_dir: str | Path | None = None,
    source: str | None = None,
) -> list[Path]:
    """Train and export annual walk-forward artifacts for one symbol.

    Returns the JSON paths written. Existing `model_*.json` files for the symbol
    are replaced to avoid stale annual weights.
    """
    symbol = symbol.upper()
    if symbol not in SUPPORTED_SYMBOLS:
        raise ValueError(f"Unsupported symbol {symbol!r}; expected one of {SUPPORTED_SYMBOLS}")

    df = compute_required_features(enriched_trades)
    if TARGET_COLUMN not in df.columns:
        if "pnl_pct" not in df.columns:
            raise ValueError("Training data must include either is_severe_loss or pnl_pct")
        df[TARGET_COLUMN] = pd.to_numeric(df["pnl_pct"], errors="coerce") <= -1.0
    missing = missing_model_features(df)
    if missing:
        raise ValueError(f"Training data is missing required model feature columns: {missing}")
    df["year"] = df["entry_time"].dt.year
    df = df.dropna(subset=MODEL_FEATURES + [TARGET_COLUMN]).copy()

    root = Path(out_dir) if out_dir is not None else DEFAULT_ARTIFACT_DIR
    symbol_dir = root / symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)
    for old in symbol_dir.glob("model_*.json"):
        old.unlink()

    written: list[Path] = []
    for year in sorted(df["year"].unique()):
        train = df[df["year"] < year].copy()
        if len(train) < MIN_TRAIN_ROWS:
            continue
        if int(train[TARGET_COLUMN].astype(int).sum()) < MIN_POSITIVES:
            continue

        scaler, model = _fit_model(train[MODEL_FEATURES].values, train[TARGET_COLUMN].astype(int).values)
        payload = _artifact_payload(
            symbol=symbol,
            predict_year=int(year),
            train=train,
            scaler=scaler,
            model=model,
            source=source,
        )
        path = symbol_dir / f"model_{int(year)}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        written.append(path)

    if not written:
        raise RuntimeError(f"No artifacts were exported for {symbol}; check training rows and positives")
    return written


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Enriched labeled trades CSV for one symbol")
    parser.add_argument("--symbol", required=True, choices=SUPPORTED_SYMBOLS)
    parser.add_argument("--out-dir", default=str(DEFAULT_ARTIFACT_DIR), help="Artifact root directory")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    df = pd.read_csv(args.input, parse_dates=["entry_time", "exit_time"])
    paths = export_annual_models(df, args.symbol, args.out_dir, source=args.input)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
