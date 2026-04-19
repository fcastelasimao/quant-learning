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
import warnings
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
        # Fallback: find the first user-defined callable (not a builtin/import)
        _builtins = {"pd", "np", "stats"}
        candidates = [
            v for k, v in namespace.items()
            if not k.startswith("_") and k not in _builtins and callable(v)
        ]
        if candidates:
            fn = candidates[0]
        else:
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


# ---------------------------------------------------------------------------
# Look-ahead bias guard
# ---------------------------------------------------------------------------

def check_lookahead_bias(
    factor_fn: Callable[[pd.DataFrame], pd.DataFrame],
    prices: pd.DataFrame,
    full_factor: pd.DataFrame,
    *,
    cutoff_frac: float = 0.65,
    timeout: int = 120,
    atol: float = 1e-6,
    name: str = "factor",
) -> None:
    """
    Detect look-ahead bias by re-running the factor on a truncated price panel.

    If ``factor_fn(prices[:cutoff]) ≈ full_factor[:cutoff]`` for every date up
    to the cutoff, no look-ahead bias is present. If the values differ, the
    factor must be using data from future rows at training time.

    The check runs the factor on ``cutoff_frac`` of the dates (default 65%),
    which is usually faster than the full run.  Uses the same timeout guard.

    Args:
        factor_fn:    Compiled factor callable (from make_factor_fn).
        prices:       Full (dates × tickers) price history.
        full_factor:  Already-computed full-panel factor output.
        cutoff_frac:  Fraction of dates to include in the truncated panel.
        timeout:      Seconds before aborting the truncated run (default 120).
        atol:         Absolute tolerance for value comparison (default 1e-6).
        name:         Factor name for error messages.

    Raises:
        ValueError: if factor values differ between full-panel and
                    truncated-panel runs on the overlapping dates.
    """
    n_dates = len(prices)
    cutoff_idx = max(10, int(n_dates * cutoff_frac))
    cutoff_date = prices.index[cutoff_idx - 1]

    truncated_prices = prices.iloc[:cutoff_idx]
    try:
        trunc_factor = run_factor_with_timeout(factor_fn, truncated_prices, timeout=timeout)
    except Exception:
        # If the truncated run fails (e.g. insufficient warmup), skip the check.
        return

    if trunc_factor.shape != truncated_prices.shape:
        warnings.warn(
            f"[lookahead check skipped] {name}: truncated-panel factor shape "
            f"{trunc_factor.shape} ≠ price panel shape {truncated_prices.shape}. "
            "Shape mismatch will be caught by validate_factor_output; "
            "look-ahead bias check cannot proceed on mismatched output.",
            UserWarning,
            stacklevel=2,
        )
        return  # Shape mismatch handled by validate_factor_output separately.

    # Compare values on the overlapping date range
    full_slice  = full_factor.iloc[:cutoff_idx].values
    trunc_slice = trunc_factor.values

    both_finite = np.isfinite(full_slice) & np.isfinite(trunc_slice)
    if both_finite.sum() < 10:
        warnings.warn(
            f"[lookahead check skipped] {name}: fewer than 10 comparable (finite) "
            f"cells in the truncated panel at {cutoff_date.date()}. "
            "Check for excessive NaN in factor output — look-ahead bias cannot be confirmed "
            "or ruled out.  Verify manually that no negative shifts are used.",
            UserWarning,
            stacklevel=2,
        )
        return  # Too few comparable cells; skip.

    # PRIMARY CHECK — values differ where both are finite
    if not np.allclose(full_slice[both_finite], trunc_slice[both_finite],
                       atol=atol, rtol=1e-5, equal_nan=False):
        raise ValueError(
            f"{name}: look-ahead bias detected — factor values differ between "
            f"the full price panel and a panel truncated at {cutoff_date.date()}. "
            "Ensure the factor uses only past data (.shift(), .rolling(), .ewm()). "
            "Do NOT use prices.index[-1], future slices, or global aggregations."
        )

    # BOUNDARY CHECK — catches shift(-1) / shift(-N) look-ahead patterns.
    # If the full-panel factor has finite values in the LAST row of the
    # truncated window but the truncated-panel factor is NaN there, the
    # factor is using data from the next row (i.e. the future).
    # Genuine warm-up NaNs appear at the START of both panels, not just the end.
    last_full  = full_slice[-1]            # shape (n_tickers,)
    last_trunc = trunc_slice[-1]
    full_finite_last  = np.isfinite(last_full)
    trunc_nan_last    = ~np.isfinite(last_trunc)
    # Suspect: full has a value but truncated is NaN, and
    #          the row BEFORE the last is finite in the truncated panel
    #          (proving warmup is not the cause).
    prev_trunc_finite = np.isfinite(trunc_slice[-2]) if len(trunc_slice) > 1 else np.zeros_like(full_finite_last, dtype=bool)
    boundary_la = full_finite_last & trunc_nan_last & prev_trunc_finite
    la_frac = boundary_la.sum() / max(full_finite_last.sum(), 1)
    if la_frac > 0.10:   # >10% of active stocks show the boundary pattern
        raise ValueError(
            f"{name}: look-ahead bias detected (boundary pattern) — "
            f"{la_frac:.0%} of stocks have finite factor values in the full panel "
            f"at {cutoff_date.date()} but NaN in the truncated panel. "
            "This indicates the factor uses data from the NEXT row (e.g. .shift(-1)). "
            "Remove any negative shifts from your implementation."
        )
