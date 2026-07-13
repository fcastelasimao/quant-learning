"""Sizing functions mapping p_severe → position scalar in [0, 1].

All functions are vectorised: accept float, np.ndarray, or pd.Series and
return the same type. Values are clipped to [0, 1].

Usage
-----
from deployable_strategies.continuous_sizing.sizing import linear_skip, SIZING_FUNCTIONS

size = linear_skip(p_severe)   # 1 - p_severe
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _clip(p):
    """Clip p to [0, 1] preserving type."""
    if isinstance(p, pd.Series):
        return p.clip(0.0, 1.0)
    return float(np.clip(float(p), 0.0, 1.0)) if np.isscalar(p) else np.clip(p, 0.0, 1.0)


def baseline_full(p) -> float | np.ndarray | pd.Series:
    """Always trade at full size (no sizing). Size = 1."""
    if isinstance(p, pd.Series):
        return pd.Series(1.0, index=p.index)
    return np.ones_like(np.asarray(p, dtype=float)) if not np.isscalar(p) else 1.0


def linear_skip(p) -> float | np.ndarray | pd.Series:
    """Gracefully scale down. size = 1 - p_severe."""
    return 1.0 - _clip(p)


def sqrt_skip(p) -> float | np.ndarray | pd.Series:
    """Mild de-risking. size = sqrt(1 - p_severe)."""
    pc = _clip(p)
    if isinstance(pc, pd.Series):
        return (1.0 - pc).clip(0.0, 1.0).pow(0.5)
    return np.sqrt(np.clip(1.0 - np.asarray(pc), 0.0, 1.0)) if not np.isscalar(pc) else float(np.sqrt(max(0.0, 1.0 - float(pc))))


def step_skip(p, threshold: float = 0.5) -> float | np.ndarray | pd.Series:
    """Binary cutoff: 0 if p_severe >= threshold, else 1."""
    pc = _clip(p)
    if isinstance(pc, pd.Series):
        return (pc < threshold).astype(float)
    return float(float(pc) < threshold) if np.isscalar(p) else (np.asarray(pc) < threshold).astype(float)


def aggressive_skip(p, multiplier: float = 2.0) -> float | np.ndarray | pd.Series:
    """Amplified de-risking. size = max(0, 1 - multiplier * p_severe)."""
    pc = _clip(p)
    if isinstance(pc, pd.Series):
        return (1.0 - multiplier * pc).clip(0.0, 1.0)
    raw = 1.0 - multiplier * np.asarray(pc)
    return float(np.clip(raw, 0.0, 1.0)) if np.isscalar(p) else np.clip(raw, 0.0, 1.0)


def moderate_skip(p, multiplier: float = 1.5) -> float | np.ndarray | pd.Series:
    """Moderate de-risking. size = max(0, 1 - multiplier * p_severe)."""
    return aggressive_skip(p, multiplier=multiplier)


# Dict for iteration in simulations
SIZING_FUNCTIONS: dict = {
    "baseline_full": baseline_full,
    "linear_skip": linear_skip,
    "sqrt_skip": sqrt_skip,
    "step_skip_at_50": lambda p: step_skip(p, threshold=0.5),
    "aggressive_2x": lambda p: aggressive_skip(p, multiplier=2.0),
    "moderate_1p5x": lambda p: moderate_skip(p, multiplier=1.5),
}
