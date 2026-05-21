from __future__ import annotations

import numpy as np
import pandas as pd

from filters import fft_denoise_series, fft_split_series, rolling_fft_components, rolling_fft_signal


def test_fft_reconstruction_preserves_length_and_index_for_odd_even_windows() -> None:
    for n in [31, 32]:
        index = pd.date_range("2020-01-01", periods=n)
        series = pd.Series(np.sin(np.arange(n) / 3), index=index)
        denoised = fft_denoise_series(series, retained_energy=0.95)

        assert len(denoised) == n
        assert denoised.index.equals(series.index)


def test_rolling_fft_uses_no_future_data() -> None:
    index = pd.date_range("2020-01-01", periods=80)
    base = pd.Series(np.sin(np.arange(80) / 5), index=index)
    changed_future = base.copy()
    changed_future.iloc[50:] = changed_future.iloc[50:] + 100

    original = rolling_fft_signal(base, window=20, retained_energy=0.95)
    modified = rolling_fft_signal(changed_future, window=20, retained_energy=0.95)

    pd.testing.assert_series_equal(original.iloc[:50], modified.iloc[:50])


def test_fft_split_recombines_main_and_residual() -> None:
    index = pd.date_range("2020-01-01", periods=64)
    series = pd.Series(np.sin(np.arange(64) / 3) + 0.1 * np.sin(np.arange(64) * 2), index=index)

    main, residual = fft_split_series(series, retained_energy=0.90)

    np.testing.assert_allclose((main + residual).to_numpy(), series.to_numpy(), atol=1e-10)


def test_rolling_fft_components_recombine_at_valid_timestamps() -> None:
    index = pd.date_range("2020-01-01", periods=80)
    series = pd.Series(np.sin(np.arange(80) / 5) + np.arange(80) / 100, index=index)

    main, residual = rolling_fft_components(series, window=20, retained_energy=0.95)
    valid = main.dropna().index

    np.testing.assert_allclose((main.loc[valid] + residual.loc[valid]).to_numpy(), series.loc[valid].to_numpy())
