"""Walk-forward severe-loss probability scorer for TQQQ/SQQQ trades.

The public entry points are:

- :func:`score_trades` for applying exported annual L1-logit weights.
- :func:`compute_required_features` for deriving model columns that can be
  derived outside the original strategy code.
- :func:`export_annual_models` for rebuilding the annual walk-forward artifacts
  from labeled enriched trade history.
"""
def compute_required_features(*args, **kwargs):
    """Lazy wrapper for feature derivation helpers."""
    from .features import compute_required_features as _compute_required_features

    return _compute_required_features(*args, **kwargs)


def score_trades(*args, **kwargs):
    """Lazy wrapper for applying exported annual artifacts."""
    from .score import score_trades as _score_trades

    return _score_trades(*args, **kwargs)


def score_trades_both(*args, **kwargs):
    """Lazy wrapper for applying both TQQQ and SQQQ artifacts."""
    from .score import score_trades_both as _score_trades_both

    return _score_trades_both(*args, **kwargs)


def score_trades_by_symbol(*args, **kwargs):
    """Lazy wrapper for scoring each row with its own symbol's artifact."""
    from .score import score_trades_by_symbol as _score_trades_by_symbol

    return _score_trades_by_symbol(*args, **kwargs)


def score_trades_with_model(*args, **kwargs):
    """Lazy wrapper for applying one in-memory annual model."""
    from .score import score_trades_with_model as _score_trades_with_model

    return _score_trades_with_model(*args, **kwargs)


def score_trades_with_models_both(*args, **kwargs):
    """Lazy wrapper for applying in-memory TQQQ and SQQQ annual models."""
    from .score import score_trades_with_models_both as _score_trades_with_models_both

    return _score_trades_with_models_both(*args, **kwargs)


def score_trades_with_models_by_symbol(*args, **kwargs):
    """Lazy wrapper for scoring each row with its own in-memory model."""
    from .score import score_trades_with_models_by_symbol as _score_trades_with_models_by_symbol

    return _score_trades_with_models_by_symbol(*args, **kwargs)


def train_model_from_history(*args, **kwargs):
    """Lazy wrapper for training one model from completed historical trades."""
    from .training import train_model_from_history as _train_model_from_history

    return _train_model_from_history(*args, **kwargs)


def train_models_from_history(*args, **kwargs):
    """Lazy wrapper for training TQQQ/SQQQ models from completed history."""
    from .training import train_models_from_history as _train_models_from_history

    return _train_models_from_history(*args, **kwargs)


def export_annual_models(*args, **kwargs):
    """Lazy wrapper so importing the scorer does not require sklearn."""
    from .export_annual_models import export_annual_models as _export_annual_models

    return _export_annual_models(*args, **kwargs)

__all__ = [
    "compute_required_features",
    "export_annual_models",
    "score_trades",
    "score_trades_both",
    "score_trades_by_symbol",
    "score_trades_with_model",
    "score_trades_with_models_both",
    "score_trades_with_models_by_symbol",
    "train_model_from_history",
    "train_models_from_history",
]
