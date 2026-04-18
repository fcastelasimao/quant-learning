"""
Safe factor code executor.

Takes a string of LLM-generated Python code, extracts the `factor` function,
and runs it against a prices DataFrame with a timeout guard.

Phase 1: uses exec() in a restricted namespace. Sufficient for a research loop
where we own the full stack and review every result. Proper subprocess isolation
can be added later if the pipeline is ever exposed to untrusted code.
"""
from __future__ import annotations

import re
import threading
from typing import Callable

import numpy as np
import pandas as pd
import scipy.stats as _scipy_stats


# ---------------------------------------------------------------------------
# Code parsing
# ---------------------------------------------------------------------------

def extract_function(code: str) -> str:
    """
    Strip markdown fences (```python ... ```) if the LLM wrapped the code.
    Returns clean Python source.
    """
    code = re.sub(r"^```(?:python)?\s*", "", code.strip(), flags=re.MULTILINE)
    code = re.sub(r"```\s*$", "", code.strip(), flags=re.MULTILINE)
    return code.strip()


def make_factor_fn(code: str) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """
    Compile LLM-generated code and return the `factor` function.

    The generated code must define a function named `factor` with signature:
        def factor(prices: pd.DataFrame) -> pd.DataFrame

    Only `pandas` (as `pd`) and `numpy` (as `np`) are available in the
    execution namespace.

    Args:
        code: Python source string.

    Returns:
        Callable that takes a prices DataFrame and returns a factor DataFrame.

    Raises:
        ValueError: if `factor` is not defined or not callable.
        SyntaxError: if the code has syntax errors.
    """
    clean = extract_function(code)
    namespace: dict = {"pd": pd, "np": np, "stats": _scipy_stats}
    exec(compile(clean, "<llm_factor>", "exec"), namespace)  # noqa: S102

    fn = namespace.get("factor")
    if fn is None:
        raise ValueError(
            "Generated code must define a function named `factor`. "
            f"Found names: {[k for k in namespace if not k.startswith('_')]}"
        )
    if not callable(fn):
        raise ValueError("`factor` must be a callable function, not a value.")
    return fn


# ---------------------------------------------------------------------------
# Timeout execution
# ---------------------------------------------------------------------------

def run_factor_with_timeout(
    factor_fn: Callable[[pd.DataFrame], pd.DataFrame],
    prices: pd.DataFrame,
    timeout: int = 120,
) -> pd.DataFrame:
    """
    Run a factor function against prices with a wall-clock timeout.

    Args:
        factor_fn: callable from make_factor_fn.
        prices:    (dates x tickers) adjusted close prices.
        timeout:   seconds before raising TimeoutError. Default 120.

    Returns:
        pd.DataFrame of factor values, same shape as prices.

    Raises:
        TimeoutError: if the function does not complete within `timeout` seconds.
        Exception:    any exception raised inside the factor function.
    """
    result: list = [None]
    error: list = [None]

    def _target():
        try:
            result[0] = factor_fn(prices)
        except Exception as exc:
            error[0] = exc

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        raise TimeoutError(
            f"Factor computation did not complete within {timeout}s. "
            "Likely an infinite loop or very expensive operation in the generated code."
        )
    if error[0] is not None:
        raise error[0]
    if result[0] is None:
        raise RuntimeError(
            "Factor computation returned None — factor() must return a pd.DataFrame."
        )
    return result[0]


# ---------------------------------------------------------------------------
# Validation of factor output
# ---------------------------------------------------------------------------

def validate_factor_output(
    factor_df: pd.DataFrame,
    prices: pd.DataFrame,
    name: str = "factor",
) -> None:
    """
    Basic sanity checks on factor output before running the harness.

    Raises:
        ValueError with a descriptive message on any failed check.
    """
    if not isinstance(factor_df, pd.DataFrame):
        raise ValueError(f"{name}: must return a pd.DataFrame, got {type(factor_df)}")

    if factor_df.shape != prices.shape:
        raise ValueError(
            f"{name}: shape mismatch — factor is {factor_df.shape}, "
            f"prices is {prices.shape}"
        )

    nan_frac = factor_df.isna().mean().mean()
    if nan_frac > 0.50:
        raise ValueError(
            f"{name}: {nan_frac:.0%} of values are NaN — "
            "factor is mostly missing data (threshold: ≤50% NaN required)."
        )

    if not np.isfinite(factor_df.values[~np.isnan(factor_df.values)]).all():
        raise ValueError(f"{name}: factor contains infinite values.")

    # Constant-value check: a factor with zero cross-sectional variance on
    # every date produces all-NaN IC — it passes shape/NaN checks but has no
    # predictive content. Reject it here so the harness never runs on it.
    cross_std = factor_df.std(axis=1).dropna()
    if cross_std.empty or (cross_std < 1e-10).all():
        raise ValueError(
            f"{name}: factor has zero cross-sectional variance on all dates — "
            "every stock receives the same score."
        )
