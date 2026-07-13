"""
tests/test_env.py
=================
Tests for the shared api_keys.env loader.
"""

from __future__ import annotations

import os

from live.env import load_api_keys_env, read_api_keys_env


def test_read_api_keys_env_parses_exports_quotes_and_comments(tmp_path):
    path = tmp_path / "api_keys.env"
    path.write_text(
        "\n".join(
            [
                "# comment",
                "FMP_API_KEY=fmp-key",
                "export ALPACA_API_KEY='alpaca-key'",
                'ALPACA_SECRET_KEY="alpaca-secret"  # local note',
            ]
        )
    )

    values = read_api_keys_env(path)

    assert values["FMP_API_KEY"] == "fmp-key"
    assert values["ALPACA_API_KEY"] == "alpaca-key"
    assert values["ALPACA_SECRET_KEY"] == "alpaca-secret"


def test_load_api_keys_env_populates_broker_aliases(monkeypatch, tmp_path):
    path = tmp_path / "api_keys.env"
    path.write_text(
        "\n".join(
            [
                "ALPACA_API_KEY=alpaca-key",
                "ALPACA_SECRET_KEY=alpaca-secret",
                "APCA_API_KEY_ID=",
                "TASTYTRADE_PROVIDER_SECRET=tt-secret",
                "TASTYTRADE_REFRESH_TOKEN=tt-token",
                "BROKER_TASTYTRADE_DEFAULT_PROVIDER_SECRET=",
                "SLACK_WEBHOOK_URL=https://example.test/hook",
            ]
        )
    )
    for key in [
        "ALPACA_API_KEY",
        "ALPACA_SECRET_KEY",
        "APCA_API_KEY_ID",
        "APCA_API_SECRET_KEY",
        "BROKER_ALPACA_DEFAULT_KEY",
        "BROKER_ALPACA_DEFAULT_SECRET",
        "BROKER_TASTYTRADE_DEFAULT_PROVIDER_SECRET",
        "BROKER_TASTYTRADE_DEFAULT_REFRESH_TOKEN",
        "ALLW_SLACK_WEBHOOK_URL",
        "ALLW_API_KEYS_ENV_LOADED",
    ]:
        monkeypatch.delenv(key, raising=False)

    loaded_path = load_api_keys_env(path=path)

    assert loaded_path == path
    assert os.getenv("APCA_API_KEY_ID") == "alpaca-key"
    assert os.getenv("APCA_API_SECRET_KEY") == "alpaca-secret"
    assert os.getenv("BROKER_ALPACA_DEFAULT_KEY") == "alpaca-key"
    assert os.getenv("BROKER_TASTYTRADE_DEFAULT_PROVIDER_SECRET") == "tt-secret"
    assert os.getenv("BROKER_TASTYTRADE_DEFAULT_REFRESH_TOKEN") == "tt-token"
    assert os.getenv("ALLW_SLACK_WEBHOOK_URL") == "https://example.test/hook"
