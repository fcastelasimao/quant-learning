from __future__ import annotations

import numpy as np
import pandas as pd


def _as_float_series(series: pd.Series) -> pd.Series:
    clean = pd.Series(series, copy=True).astype(float)
    if clean.empty:
        raise ValueError("series must not be empty")
    return clean


def _largest_energy_mask(energy: np.ndarray, retained_energy: float | None, top_k: int | None) -> np.ndarray:
    if top_k is None and retained_energy is None:
        raise ValueError("Provide either retained_energy or top_k")
    if retained_energy is not None and not 0 < retained_energy <= 1:
        raise ValueError("retained_energy must be in (0, 1]")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be positive")

    order = np.argsort(energy)[::-1]
    if top_k is not None:
        keep = order[: min(top_k, len(order))]
    else:
        total = float(energy.sum())
        if total <= 0:
            keep = order[:1]
        else:
            cumulative = np.cumsum(energy[order]) / total
            count = int(np.searchsorted(cumulative, retained_energy, side="left") + 1)
            keep = order[:count]

    mask = np.zeros_like(energy, dtype=bool)
    mask[keep] = True
    return mask


def fft_denoise_series(
    series: pd.Series,
    *,
    retained_energy: float | None = 0.95,
    top_k: int | None = None,
) -> pd.Series:
    """Denoise one complete window with FFT coefficient truncation."""
    main, _ = fft_split_series(series, retained_energy=retained_energy, top_k=top_k)
    return main


def fft_split_series(
    series: pd.Series,
    *,
    retained_energy: float | None = 0.95,
    top_k: int | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Split a window into retained FFT component and discarded residual."""
    values = _as_float_series(series)
    mean = float(values.mean())
    centered = values.to_numpy() - mean

    coeffs = np.fft.rfft(centered)
    energy = np.abs(coeffs) ** 2
    mask = _largest_energy_mask(energy, retained_energy, top_k)
    main_coeffs = np.where(mask, coeffs, 0.0)
    residual_coeffs = np.where(mask, 0.0, coeffs)

    main = np.fft.irfft(main_coeffs, n=len(values)) + mean
    residual = np.fft.irfft(residual_coeffs, n=len(values))

    return (
        pd.Series(main, index=values.index, name=values.name),
        pd.Series(residual, index=values.index, name=values.name),
    )


def rolling_fft_signal(
    series: pd.Series,
    *,
    window: int,
    retained_energy: float | None = 0.95,
    top_k: int | None = None,
) -> pd.Series:
    """Return the last denoised value from each rolling window.

    The value at timestamp t only uses observations up to and including t.
    """
    if window < 4:
        raise ValueError("window must be at least 4")

    values = _as_float_series(series)
    output = pd.Series(np.nan, index=values.index, name="rolling_fft")
    for end in range(window - 1, len(values)):
        window_series = values.iloc[end - window + 1 : end + 1]
        output.iloc[end] = fft_denoise_series(
            window_series,
            retained_energy=retained_energy,
            top_k=top_k,
        ).iloc[-1]
    return output


def rolling_fft_components(
    series: pd.Series,
    *,
    window: int,
    retained_energy: float | None = 0.95,
    top_k: int | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Return causal retained and residual FFT components.

    The residual is not deleted; it is preserved as a separate testable signal.
    """
    if window < 4:
        raise ValueError("window must be at least 4")

    values = _as_float_series(series)
    main_output = pd.Series(np.nan, index=values.index, name="rolling_fft_main")
    residual_output = pd.Series(np.nan, index=values.index, name="rolling_fft_residual")
    for end in range(window - 1, len(values)):
        window_series = values.iloc[end - window + 1 : end + 1]
        main, residual = fft_split_series(
            window_series,
            retained_energy=retained_energy,
            top_k=top_k,
        )
        main_output.iloc[end] = main.iloc[-1]
        residual_output.iloc[end] = residual.iloc[-1]
    return main_output, residual_output


def _diagonal_averaging(matrix: np.ndarray) -> np.ndarray:
    rows, cols = matrix.shape
    reconstructed = np.zeros(rows + cols - 1)
    counts = np.zeros(rows + cols - 1)

    for row in range(rows):
        for col in range(cols):
            reconstructed[row + col] += matrix[row, col]
            counts[row + col] += 1

    return reconstructed / counts


def ssa_denoise_series(
    series: pd.Series,
    *,
    window_length: int,
    retained_energy: float | None = 0.95,
    n_components: int | None = None,
) -> pd.Series:
    """Denoise one complete window with Singular Spectrum Analysis."""
    values = _as_float_series(series)
    n = len(values)
    if not 2 <= window_length <= n:
        raise ValueError("window_length must be between 2 and len(series)")

    trajectory = np.column_stack([values.to_numpy()[i : i + window_length] for i in range(n - window_length + 1)])
    u, singular_values, vt = np.linalg.svd(trajectory, full_matrices=False)
    energy = singular_values**2
    mask = _largest_energy_mask(energy, retained_energy, n_components)

    reconstructed_matrix = np.zeros_like(trajectory)
    for component in np.where(mask)[0]:
        reconstructed_matrix += singular_values[component] * np.outer(u[:, component], vt[component, :])

    reconstructed = _diagonal_averaging(reconstructed_matrix)
    return pd.Series(reconstructed, index=values.index, name=values.name)


def rolling_ssa_signal(
    series: pd.Series,
    *,
    window: int,
    window_length: int,
    retained_energy: float | None = 0.95,
    n_components: int | None = None,
) -> pd.Series:
    """Return the last SSA-denoised value from each rolling window."""
    if window < 4:
        raise ValueError("window must be at least 4")
    if not 2 <= window_length <= window:
        raise ValueError("window_length must be between 2 and window")

    values = _as_float_series(series)
    output = pd.Series(np.nan, index=values.index, name="rolling_ssa")
    for end in range(window - 1, len(values)):
        window_series = values.iloc[end - window + 1 : end + 1]
        output.iloc[end] = ssa_denoise_series(
            window_series,
            window_length=window_length,
            retained_energy=retained_energy,
            n_components=n_components,
        ).iloc[-1]
    return output
