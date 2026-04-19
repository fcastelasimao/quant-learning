"""
Regression tests for src/qframe/data/crypto.py.
"""
import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Cache min_history_days regression test (§1.1 / §1.4 of 2026-04-19 plan)
# ---------------------------------------------------------------------------

class TestLoadBinanceCloseCache:
    """Cache hit path must apply min_history_days filter, not just the non-cached path."""

    def _make_cache_df(self, tmp_path, short_sym="SHORT", long_sym="LONG"):
        """Write a parquet cache with one short-history and one long-history symbol."""
        dates = pd.bdate_range("2020-01-01", periods=300)
        data = {
            long_sym: np.random.default_rng(0).lognormal(0, 0.01, 300).cumprod(),
            short_sym: np.concatenate([
                np.full(50, np.nan),          # 250 NaN → only 50 real days
                np.random.default_rng(1).lognormal(0, 0.01, 250).cumprod(),
            ]),
        }
        df = pd.DataFrame(data, index=dates)
        cache_path = tmp_path / "crypto_cache.parquet"
        df.to_parquet(cache_path)
        return str(cache_path), long_sym, short_sym

    def test_cache_hit_respects_min_history_days(self, tmp_path, monkeypatch):
        """
        Regression: on a cache hit, symbols with < min_history_days non-NaN rows
        must be dropped, matching the behaviour of the non-cached path.
        """
        from qframe.data.crypto import load_binance_close

        cache_path, long_sym, short_sym = self._make_cache_df(tmp_path)

        # Monkeypatch ccxt so the live fetch is never called
        import types
        fake_ccxt = types.ModuleType("ccxt")
        monkeypatch.setitem(__import__("sys").modules, "ccxt", fake_ccxt)

        result = load_binance_close(
            symbols=[long_sym, short_sym],
            start="2020-01-01",
            end="2021-12-31",
            min_history_days=252,
            cache_path=cache_path,
            force_refresh=False,
        )

        assert short_sym not in result.columns, (
            f"Symbol '{short_sym}' with < 252 days of data should be dropped on cache hit"
        )
        assert long_sym in result.columns, (
            f"Symbol '{long_sym}' with sufficient history should be present"
        )
