"""Score candidate trades with exported annual `p_severe` model artifacts."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import DEFAULT_ARTIFACT_DIR, AnnualModel, load_models, select_model_for_year
from .constants import MODEL_FEATURES, SUPPORTED_SYMBOLS
from .features import compute_required_features, validate_model_ready


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def _score_with_model(df: pd.DataFrame, model: AnnualModel) -> np.ndarray:
    x = df[model.feature_order].astype(float).values
    z = ((x - model.scaler_mean) / model.scaler_scale) @ model.coefficients + model.intercept
    return np.clip(_sigmoid(z), 0.0, 1.0)


def score_trades(
    trades: pd.DataFrame,
    symbol: str,
    artifact_dir: str | Path | None = None,
    *,
    on_missing_model: str = "nan",
    validate_features: bool = True,
) -> pd.DataFrame:
    """Compute `p_severe` and `size_multiplier = 1 - p_severe`.

    `on_missing_model` controls rows whose entry year has no prior-year model:
    use `"nan"` to leave probability columns null, or `"raise"` to fail.
    Future years are scored with the newest artifact whose `train_end_year` is
    strictly before the trade year, preserving the walk-forward no-look-ahead
    rule.
    """
    symbol = symbol.upper()
    if symbol not in SUPPORTED_SYMBOLS:
        raise ValueError(f"Unsupported symbol {symbol!r}; expected one of {SUPPORTED_SYMBOLS}")
    if on_missing_model not in {"nan", "raise"}:
        raise ValueError("on_missing_model must be 'nan' or 'raise'")

    df = compute_required_features(trades)
    if validate_features:
        validate_model_ready(df)

    models = load_models(symbol, artifact_dir)
    out = df.copy()
    out["entry_time"] = pd.to_datetime(out["entry_time"])
    out["p_severe"] = np.nan
    out["size_multiplier"] = np.nan
    out["model_year_used"] = np.nan
    out["model_train_end_year"] = np.nan

    for year, idx in out.groupby(out["entry_time"].dt.year).groups.items():
        model = select_model_for_year(models, int(year))
        if model is None:
            if on_missing_model == "raise":
                raise ValueError(f"No prior-year model artifact is available for {symbol} entry year {year}")
            continue
        p = _score_with_model(out.loc[idx, MODEL_FEATURES], model)
        out.loc[idx, "p_severe"] = p
        out.loc[idx, "size_multiplier"] = 1.0 - p
        out.loc[idx, "model_year_used"] = int(model.predict_year)
        out.loc[idx, "model_train_end_year"] = int(model.train_end_year)

    return out


def score_trades_with_model(
    trades: pd.DataFrame,
    model: AnnualModel,
    *,
    validate_features: bool = True,
) -> pd.DataFrame:
    """Score candidate trades with one in-memory model.

    This is the API to use inside a year-by-year backtest loop after calling
    `train_model_from_history(...)`. It rejects rows whose `entry_time` is not
    after the model's `train_end_year`.
    """
    df = compute_required_features(trades)
    if validate_features:
        validate_model_ready(df)

    out = df.copy()
    out["entry_time"] = pd.to_datetime(out["entry_time"])
    bad_years = sorted(out.loc[out["entry_time"].dt.year <= model.train_end_year, "entry_time"].dt.year.unique())
    if bad_years:
        raise ValueError(
            f"Model for {model.symbol} was trained through {model.train_end_year}; "
            f"cannot score entry years {bad_years} without look-ahead risk"
        )

    p = _score_with_model(out[MODEL_FEATURES], model)
    out["p_severe"] = p
    out["size_multiplier"] = 1.0 - p
    out["model_year_used"] = int(model.predict_year)
    out["model_train_end_year"] = int(model.train_end_year)
    return out


def score_trades_with_models_both(
    trades: pd.DataFrame,
    models: dict[str, AnnualModel],
    *,
    validate_features: bool = True,
) -> pd.DataFrame:
    """Score candidate trades with in-memory TQQQ and SQQQ models."""
    missing = [symbol for symbol in SUPPORTED_SYMBOLS if symbol not in models]
    if missing:
        raise ValueError(f"models is missing required symbols: {missing}")

    out: pd.DataFrame | None = None
    for symbol in SUPPORTED_SYMBOLS:
        scored = score_trades_with_model(trades, models[symbol], validate_features=validate_features)
        suffix_cols = {
            "p_severe": f"p_severe_{symbol}",
            "size_multiplier": f"size_multiplier_{symbol}",
            "model_year_used": f"model_year_used_{symbol}",
            "model_train_end_year": f"model_train_end_year_{symbol}",
        }
        scored = scored.rename(columns=suffix_cols)
        keep = list(suffix_cols.values())
        if out is None:
            out = scored
        else:
            out = out.join(scored[keep])
    assert out is not None
    return out


def score_trades_by_symbol(
    trades: pd.DataFrame,
    artifact_dir: str | Path | None = None,
    *,
    on_missing_model: str = "nan",
    validate_features: bool = True,
) -> pd.DataFrame:
    """Score each row using the model matching its own `symbol` column."""
    if "symbol" not in trades.columns:
        raise ValueError("score_trades_by_symbol requires a 'symbol' column")
    out = compute_required_features(trades)
    if validate_features:
        validate_model_ready(out)
    out["p_severe"] = np.nan
    out["size_multiplier"] = np.nan
    out["model_year_used"] = np.nan
    out["model_train_end_year"] = np.nan

    for symbol in SUPPORTED_SYMBOLS:
        idx = out.index[out["symbol"].astype(str).str.upper() == symbol]
        if len(idx) == 0:
            continue
        scored = score_trades(
            out.loc[idx].copy(),
            symbol,
            artifact_dir=artifact_dir,
            on_missing_model=on_missing_model,
            validate_features=False,
        )
        for col in ("p_severe", "size_multiplier", "model_year_used", "model_train_end_year"):
            out.loc[idx, col] = scored[col].values

    unsupported = sorted(set(out["symbol"].astype(str).str.upper()) - set(SUPPORTED_SYMBOLS))
    if unsupported:
        raise ValueError(f"Unsupported symbol values in trades: {unsupported}")
    return out


def score_trades_with_models_by_symbol(
    trades: pd.DataFrame,
    models: dict[str, AnnualModel],
    *,
    validate_features: bool = True,
) -> pd.DataFrame:
    """Score each row with the in-memory model matching its own `symbol`."""
    if "symbol" not in trades.columns:
        raise ValueError("score_trades_with_models_by_symbol requires a 'symbol' column")
    missing = [symbol for symbol in SUPPORTED_SYMBOLS if symbol not in models]
    if missing:
        raise ValueError(f"models is missing required symbols: {missing}")

    out = compute_required_features(trades)
    if validate_features:
        validate_model_ready(out)
    out["p_severe"] = np.nan
    out["size_multiplier"] = np.nan
    out["model_year_used"] = np.nan
    out["model_train_end_year"] = np.nan

    for symbol in SUPPORTED_SYMBOLS:
        idx = out.index[out["symbol"].astype(str).str.upper() == symbol]
        if len(idx) == 0:
            continue
        scored = score_trades_with_model(out.loc[idx].copy(), models[symbol], validate_features=False)
        for col in ("p_severe", "size_multiplier", "model_year_used", "model_train_end_year"):
            out.loc[idx, col] = scored[col].values

    unsupported = sorted(set(out["symbol"].astype(str).str.upper()) - set(SUPPORTED_SYMBOLS))
    if unsupported:
        raise ValueError(f"Unsupported symbol values in trades: {unsupported}")
    return out


def score_trades_both(
    trades: pd.DataFrame,
    artifact_dir: str | Path | None = None,
    *,
    on_missing_model: str = "nan",
    validate_features: bool = True,
) -> pd.DataFrame:
    """Compute TQQQ and SQQQ `p_severe` columns for the same candidate rows.

    The output contains:

    - `p_severe_TQQQ`, `size_multiplier_TQQQ`
    - `p_severe_SQQQ`, `size_multiplier_SQQQ`
    - `model_year_used_TQQQ`, `model_train_end_year_TQQQ`
    - `model_year_used_SQQQ`, `model_train_end_year_SQQQ`

    This is a diagnostic cross-symbol mode. For production backtests, prefer
    `score_trades_by_symbol`, because TQQQ and SQQQ rows contain different
    strategy-entry features.
    """
    out: pd.DataFrame | None = None
    for symbol in SUPPORTED_SYMBOLS:
        scored = score_trades(
            trades,
            symbol,
            artifact_dir=artifact_dir,
            on_missing_model=on_missing_model,
            validate_features=validate_features,
        )
        suffix_cols = {
            "p_severe": f"p_severe_{symbol}",
            "size_multiplier": f"size_multiplier_{symbol}",
            "model_year_used": f"model_year_used_{symbol}",
            "model_train_end_year": f"model_train_end_year_{symbol}",
        }
        scored = scored.rename(columns=suffix_cols)
        keep = list(suffix_cols.values())
        if out is None:
            out = scored
        else:
            out = out.join(scored[keep])
    assert out is not None
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", required=True, choices=SUPPORTED_SYMBOLS + ("BOTH", "AUTO"))
    parser.add_argument("--input", required=True, help="Candidate trades CSV")
    parser.add_argument("--output", required=True, help="Output CSV with p_severe and size_multiplier")
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument(
        "--on-missing-model",
        choices=("nan", "raise"),
        default="nan",
        help="How to handle years before the first annual artifact",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    df = pd.read_csv(args.input, parse_dates=["entry_time", "exit_time"])
    if args.symbol == "AUTO":
        scored = score_trades_by_symbol(
            df,
            artifact_dir=args.artifact_dir,
            on_missing_model=args.on_missing_model,
        )
    elif args.symbol == "BOTH":
        scored = score_trades_both(
            df,
            artifact_dir=args.artifact_dir,
            on_missing_model=args.on_missing_model,
        )
    else:
        scored = score_trades(
            df,
            args.symbol,
            artifact_dir=args.artifact_dir,
            on_missing_model=args.on_missing_model,
        )
    scored.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
