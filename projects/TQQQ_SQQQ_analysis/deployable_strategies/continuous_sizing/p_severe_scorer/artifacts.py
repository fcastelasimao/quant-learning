"""Load and select exported annual model artifacts.

Artifacts are plain JSON files so scoring does not need sklearn. Each file
stores the exact feature order, scaler parameters, and L1-logit coefficients
for one symbol and one prediction year.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .constants import MODEL_FEATURES, SUPPORTED_SYMBOLS


DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"


@dataclass(frozen=True)
class AnnualModel:
    symbol: str
    predict_year: int
    train_start_year: int
    train_end_year: int
    feature_order: list[str]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    intercept: float
    coefficients: np.ndarray
    target: str
    target_description: str
    context_policy: str = ""
    dataset_hash: str = ""
    source_path: Path | None = None

    @classmethod
    def from_json(cls, path: Path) -> "AnnualModel":
        data = json.loads(path.read_text())
        required = {
            "symbol",
            "predict_year",
            "train_start_year",
            "train_end_year",
            "feature_order",
            "scaler_mean",
            "scaler_scale",
            "intercept",
            "coefficients",
            "target",
            "target_description",
            "context_policy",
            "dataset_hash",
        }
        missing = sorted(required - set(data))
        if missing:
            raise ValueError(f"{path} is missing artifact fields: {missing}")
        feature_order = list(data["feature_order"])
        if feature_order != MODEL_FEATURES:
            raise ValueError(
                f"{path} feature_order does not match current MODEL_FEATURES. "
                "Re-export the annual artifacts before scoring."
            )
        return cls(
            symbol=str(data["symbol"]),
            predict_year=int(data["predict_year"]),
            train_start_year=int(data["train_start_year"]),
            train_end_year=int(data["train_end_year"]),
            feature_order=feature_order,
            scaler_mean=np.asarray(data["scaler_mean"], dtype=float),
            scaler_scale=np.asarray(data["scaler_scale"], dtype=float),
            intercept=float(data["intercept"]),
            coefficients=np.asarray(data["coefficients"], dtype=float),
            target=str(data["target"]),
            target_description=str(data["target_description"]),
            context_policy=str(data["context_policy"]),
            dataset_hash=str(data["dataset_hash"]),
            source_path=path,
        )

    def validate_shapes(self) -> None:
        n = len(self.feature_order)
        if self.scaler_mean.shape != (n,):
            raise ValueError(f"{self.source_path} scaler_mean length != {n}")
        if self.scaler_scale.shape != (n,):
            raise ValueError(f"{self.source_path} scaler_scale length != {n}")
        if self.coefficients.shape != (n,):
            raise ValueError(f"{self.source_path} coefficients length != {n}")
        if np.any(self.scaler_scale == 0):
            raise ValueError(f"{self.source_path} contains a zero scaler scale")


def _symbol_dir(artifact_dir: Path, symbol: str) -> Path:
    symbol = symbol.upper()
    if symbol not in SUPPORTED_SYMBOLS:
        raise ValueError(f"Unsupported symbol {symbol!r}; expected one of {SUPPORTED_SYMBOLS}")
    return artifact_dir / symbol


def load_models(symbol: str, artifact_dir: str | Path | None = None) -> list[AnnualModel]:
    """Load all annual artifacts for `symbol`, sorted by train end year."""
    root = Path(artifact_dir) if artifact_dir is not None else DEFAULT_ARTIFACT_DIR
    paths = sorted(_symbol_dir(root, symbol).glob("model_*.json"))
    if not paths:
        raise FileNotFoundError(f"No annual model artifacts found under {_symbol_dir(root, symbol)}")
    models = [AnnualModel.from_json(path) for path in paths]
    for model in models:
        model.validate_shapes()
        if model.symbol != symbol.upper():
            raise ValueError(f"{model.source_path} has symbol {model.symbol}, expected {symbol.upper()}")
    return sorted(models, key=lambda m: (m.train_end_year, m.predict_year))


def select_model_for_year(models: Iterable[AnnualModel], entry_year: int) -> AnnualModel | None:
    """Select the newest model trained strictly before `entry_year`.

    This is the look-ahead guard. For a 2026 trade the selected model must have
    `train_end_year <= 2025`; if an exact 2026 prediction artifact is absent, the
    latest older artifact is used instead. If no prior model exists, `None` is
    returned.
    """
    candidates = [m for m in models if m.train_end_year < int(entry_year)]
    if not candidates:
        return None
    return max(candidates, key=lambda m: (m.train_end_year, m.predict_year))
