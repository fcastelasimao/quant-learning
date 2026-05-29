"""
live/env.py
===========
Shared environment loading for live trading.

The canonical secrets file lives at the QuantFinance workspace root:

    /Users/franciscosimao/Documents/QuantFinance/api_keys.env

Keep this parser dependency-free so health checks and broker adapters can load
credentials before any optional packages are imported.
"""

from __future__ import annotations

import os
from pathlib import Path


ENV_PATH_OVERRIDE = "ALLW_API_KEYS_ENV"

_ALIASES: dict[str, tuple[str, ...]] = {
    # Existing shared file names -> names expected by Alpaca/alpaca-py.
    "ALPACA_API_KEY": ("APCA_API_KEY_ID", "BROKER_ALPACA_DEFAULT_KEY"),
    "ALPACA_SECRET_KEY": ("APCA_API_SECRET_KEY", "BROKER_ALPACA_DEFAULT_SECRET"),
    # Short default-account Tastytrade names -> broker adapter names.
    "TASTYTRADE_PROVIDER_SECRET": ("BROKER_TASTYTRADE_DEFAULT_PROVIDER_SECRET",),
    "TASTYTRADE_REFRESH_TOKEN": ("BROKER_TASTYTRADE_DEFAULT_REFRESH_TOKEN",),
    "TASTYTRADE_ACCOUNT_NUMBER": ("BROKER_TASTYTRADE_DEFAULT_ACCOUNT_NUMBER",),
    # Optional notification aliases.
    "SLACK_WEBHOOK_URL": ("ALLW_SLACK_WEBHOOK_URL",),
    "SLACK_CHANNEL": ("ALLW_SLACK_CHANNEL",),
    "SMTP_HOST": ("ALLW_SMTP_HOST",),
    "SMTP_PORT": ("ALLW_SMTP_PORT",),
    "SMTP_USER": ("ALLW_SMTP_USER",),
    "SMTP_PASSWORD": ("ALLW_SMTP_PASSWORD",),
    "SMTP_FROM": ("ALLW_SMTP_FROM",),
    "NOTIFY_EMAIL": ("ALLW_NOTIFY_EMAIL",),
}


def default_api_keys_env_path() -> Path:
    """Return the shared QuantFinance api_keys.env path if it can be found."""
    override = os.getenv(ENV_PATH_OVERRIDE)
    if override:
        return Path(override).expanduser()

    probes = [Path(__file__).resolve(), Path.cwd().resolve()]
    for probe in probes:
        for parent in (probe.parent, *probe.parents):
            candidate = parent / "api_keys.env"
            if candidate.exists():
                return candidate

    return Path.home() / "Documents" / "QuantFinance" / "api_keys.env"


def _strip_inline_comment(value: str) -> str:
    quote: str | None = None
    for idx, char in enumerate(value):
        if char in {"'", '"'} and (idx == 0 or value[idx - 1] != "\\"):
            quote = None if quote == char else char
        if char == "#" and quote is None and idx > 0 and value[idx - 1].isspace():
            return value[:idx].rstrip()
    return value.strip()


def _parse_env_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export "):].strip()
    key, value = stripped.split("=", 1)
    key = key.strip()
    if not key or not (key[0].isalpha() or key[0] == "_"):
        return None
    value = _strip_inline_comment(value.strip())
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return key, value


def read_api_keys_env(path: str | Path | None = None) -> dict[str, str]:
    """Parse api_keys.env without mutating os.environ."""
    env_path = Path(path).expanduser() if path else default_api_keys_env_path()
    if not env_path.exists():
        return {}

    loaded: dict[str, str] = {}
    for line in env_path.read_text().splitlines():
        item = _parse_env_line(line)
        if item is None:
            continue
        key, value = item
        loaded[key] = value
    return loaded


def load_api_keys_env(*, override: bool = False, path: str | Path | None = None) -> Path | None:
    """Load the shared key file into os.environ and populate known aliases.

    Existing process environment values win by default. Pass override=True only
    for a deliberate test or one-off operator override.
    """
    env_path = Path(path).expanduser() if path else default_api_keys_env_path()
    values = read_api_keys_env(env_path)
    if not values:
        return env_path if env_path.exists() else None

    for key, value in values.items():
        if value == "" and not override:
            continue
        if override or key not in os.environ:
            os.environ[key] = value

    for source, aliases in _ALIASES.items():
        value = values.get(source) or os.environ.get(source)
        if not value:
            continue
        for alias in aliases:
            if override or not os.environ.get(alias):
                os.environ[alias] = value

    os.environ["ALLW_API_KEYS_ENV_LOADED"] = str(env_path)
    return env_path
