"""
Provider-agnostic LLM helper for the pipeline agents.

Provider is controlled by LLM_PROVIDER in .env (default: groq).
On quota exhaustion the router falls back through the ordered chain:

  groq → cerebras → deepseek → together → mistral → gemini → openrouter

Providers with no API key configured are skipped transparently.
Non-quota errors (auth failures, bad requests) propagate immediately — they
should be fixed, not silently bypassed.

Switch primary provider via .env:
  LLM_PROVIDER=groq        GROQ_MODEL=llama-3.3-70b-versatile
  LLM_PROVIDER=cerebras    CEREBRAS_MODEL=llama-3.3-70b
  LLM_PROVIDER=gemini      GEMINI_MODEL=gemini-2.0-flash
  (etc.)
"""
from __future__ import annotations

import os
import re
import time

from qframe.config import (
    GEMINI_API_KEY, GEMINI_MODEL,
    GROQ_API_KEY,
    CEREBRAS_API_KEY, CEREBRAS_MODEL,
    TOGETHER_API_KEY, TOGETHER_MODEL,
    DEEPSEEK_API_KEY, DEEPSEEK_MODEL,
    MISTRAL_API_KEY, MISTRAL_MODEL,
    OPENROUTER_API_KEY, OPENROUTER_MODEL,
)

_PROVIDER     = os.environ.get("LLM_PROVIDER", "groq")
_GROQ_MODEL   = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
_MAX_RETRIES  = 5
_DEFAULT_WAIT = 15
_FAILOVER_ON_ACCESS_DENIED = os.environ.get("LLM_FAILOVER_ON_ACCESS_DENIED", "0").strip() in (
    "1", "true", "TRUE", "yes", "YES", "on", "ON"
)


# ---------------------------------------------------------------------------
# Custom exception — signals "quota exhausted, try the next provider"
# ---------------------------------------------------------------------------

class QuotaExhaustedError(Exception):
    """
    Raised when a provider's daily/monthly quota is exhausted, or when its
    API key is not configured. generate() catches this and tries the next
    provider in the fallback chain.
    """
    def __init__(self, provider: str, detail: str = ""):
        self.provider = provider
        super().__init__(f"[{provider}] quota exhausted: {detail}")


# ---------------------------------------------------------------------------
# Retry delay parser
# ---------------------------------------------------------------------------

def _parse_retry_delay(msg: str) -> float:
    """Parse wait time from rate-limit error message. Handles 'Xm Ys' and 'Xs' formats."""
    # Match Xm Ys format: e.g. "22m50.303s"
    m = re.search(r"(\d+)m(\d+(?:\.\d+)?)s", msg)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2)) + 2
    # Match plain Xs format: e.g. "45.5s"
    m = re.search(r"(\d+(?:\.\d+)?)\s*s", msg, re.IGNORECASE)
    return float(m.group(1)) + 2 if m else _DEFAULT_WAIT


def _is_access_denied_error(status_code: int | None, message: str) -> bool:
    """
    Detect provider responses that indicate *policy/network access* denial rather
    than quota exhaustion. These are often transient/environmental (VPN, proxy,
    corporate egress policy, geo/IP restrictions) and may be resolvable by
    failing over to another provider.

    This is intentionally conservative; default behaviour remains to propagate.
    """
    msg = (message or "").lower()
    if status_code == 403:
        return True
    # Provider SDKs sometimes wrap this as 401/400 with "access denied" wording.
    return any(
        kw in msg
        for kw in (
            "access denied",
            "network settings",
            "blocked",
            "forbidden",
            "policy",
            "not authorized",
            "not authorised",
        )
    )


def _maybe_failover_on_access_denied(provider_name: str, status_code: int | None, message: str) -> None:
    if _FAILOVER_ON_ACCESS_DENIED and _is_access_denied_error(status_code, message):
        raise QuotaExhaustedError(provider_name, f"access denied (status={status_code})")  # noqa: TRY003


# ---------------------------------------------------------------------------
# Shared OpenAI-compatible backend (Cerebras, Together, DeepSeek, Mistral,
# OpenRouter all expose the same chat completions interface)
# ---------------------------------------------------------------------------

def _openai_compat_generate(
    prompt: str,
    *,
    base_url: str,
    api_key: str,
    model: str,
    provider_name: str,
    extra_headers: dict | None = None,
) -> str:
    """
    Call any OpenAI-compatible chat completions endpoint.

    The `openai` import is lazy (inside this function) so the module loads
    cleanly even when `openai` is not installed — other providers still work.

    Args:
        prompt:         User message text.
        base_url:       API base URL, e.g. "https://api.cerebras.ai/v1"
        api_key:        Provider API key.
        model:          Model identifier string.
        provider_name:  Used in log messages and QuotaExhaustedError.
        extra_headers:  Optional additional HTTP headers (e.g. OpenRouter needs
                        HTTP-Referer and X-Title).
    """
    from openai import OpenAI, RateLimitError, APIStatusError  # lazy import

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers=extra_headers or {},
    )
    last_exc: Exception | None = None

    for attempt in range(_MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1024,
            )
            return resp.choices[0].message.content

        except RateLimitError as exc:
            # IMPORTANT: RateLimitError is a subclass of APIStatusError in
            # openai v1.x — it MUST be caught before APIStatusError.
            err_str = str(exc).lower()
            # Daily/monthly quota — don't retry, fall through to next provider
            if any(kw in err_str for kw in ("quota", "limit exceeded", "daily")):
                raise QuotaExhaustedError(provider_name, str(exc)) from exc
            # Per-minute rate limit — wait and retry
            last_exc = exc
            wait = _parse_retry_delay(str(exc))
            print(f"      {provider_name} rate limit — waiting {wait:.0f}s "
                  f"(attempt {attempt + 1}/{_MAX_RETRIES})")
            time.sleep(wait)

        except APIStatusError as exc:
            if exc.status_code >= 500:
                last_exc = exc
                wait = _DEFAULT_WAIT * (2 ** attempt)
                print(f"      {provider_name} {exc.status_code} — retrying in {wait}s "
                      f"(attempt {attempt + 1}/{_MAX_RETRIES})")
                time.sleep(wait)
            else:
                _maybe_failover_on_access_denied(provider_name, exc.status_code, str(exc))
                raise

    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Groq
# ---------------------------------------------------------------------------

def _groq_generate(prompt: str, model: str = _GROQ_MODEL) -> str:
    from groq import Groq, RateLimitError, APIStatusError

    if not GROQ_API_KEY:
        raise EnvironmentError("GROQ_API_KEY is not set in .env")

    client = Groq(api_key=GROQ_API_KEY)
    last_exc: Exception | None = None

    for attempt in range(_MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1024,
            )
            return resp.choices[0].message.content

        except RateLimitError as exc:
            last_exc = exc
            err_str = str(exc)
            # Daily TPD limit — fall through to next provider immediately
            if "tokens per day" in err_str:
                raise QuotaExhaustedError("groq", err_str) from exc
            wait = _parse_retry_delay(err_str)
            print(f"      Groq rate limit — waiting {wait:.0f}s "
                  f"(attempt {attempt + 1}/{_MAX_RETRIES})")
            time.sleep(wait)

        except APIStatusError as exc:
            if exc.status_code >= 500:
                last_exc = exc
                wait = _DEFAULT_WAIT * (2 ** attempt)
                print(f"      Groq {exc.status_code} — retrying in {wait}s "
                      f"(attempt {attempt + 1}/{_MAX_RETRIES})")
                time.sleep(wait)
            else:
                _maybe_failover_on_access_denied("groq", exc.status_code, str(exc))
                raise

    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

def _gemini_generate(prompt: str, model: str = GEMINI_MODEL) -> str:
    from google import genai
    from google.genai import errors as genai_errors

    if not GEMINI_API_KEY:
        raise QuotaExhaustedError("gemini", "API key not configured")

    client = genai.Client(api_key=GEMINI_API_KEY)
    last_exc: Exception | None = None

    for attempt in range(_MAX_RETRIES):
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            return response.text

        except genai_errors.ServerError as exc:
            last_exc = exc
            wait = _DEFAULT_WAIT * (2 ** attempt)
            print(f"      Gemini {exc.code} server error — retrying in {wait}s "
                  f"(attempt {attempt + 1}/{_MAX_RETRIES})")
            time.sleep(wait)

        except genai_errors.ClientError as exc:
            if exc.code == 429:
                err_str = str(exc).lower()
                # Daily quota exhausted — don't retry, fall through to next provider
                if any(kw in err_str for kw in ("quota", "exhausted", "daily")):
                    raise QuotaExhaustedError("gemini", str(exc)) from exc
                last_exc = exc
                wait = _parse_retry_delay(str(exc))
                print(f"      Gemini 429 rate limit — waiting {wait:.0f}s "
                      f"(attempt {attempt + 1}/{_MAX_RETRIES})")
                time.sleep(wait)
            else:
                raise

    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# OpenAI-compatible providers
# ---------------------------------------------------------------------------

def _cerebras_generate(prompt: str) -> str:
    if not CEREBRAS_API_KEY:
        raise QuotaExhaustedError("cerebras", "API key not configured")
    return _openai_compat_generate(
        prompt,
        base_url="https://api.cerebras.ai/v1",
        api_key=CEREBRAS_API_KEY,
        model=CEREBRAS_MODEL,
        provider_name="cerebras",
    )


def _together_generate(prompt: str) -> str:
    if not TOGETHER_API_KEY:
        raise QuotaExhaustedError("together", "API key not configured")
    return _openai_compat_generate(
        prompt,
        base_url="https://api.together.xyz/v1",
        api_key=TOGETHER_API_KEY,
        model=TOGETHER_MODEL,
        provider_name="together",
    )


def _deepseek_generate(prompt: str) -> str:
    if not DEEPSEEK_API_KEY:
        raise QuotaExhaustedError("deepseek", "API key not configured")
    return _openai_compat_generate(
        prompt,
        base_url="https://api.deepseek.com/v1",
        api_key=DEEPSEEK_API_KEY,
        model=DEEPSEEK_MODEL,
        provider_name="deepseek",
    )


def _mistral_generate(prompt: str) -> str:
    if not MISTRAL_API_KEY:
        raise QuotaExhaustedError("mistral", "API key not configured")
    return _openai_compat_generate(
        prompt,
        base_url="https://api.mistral.ai/v1",
        api_key=MISTRAL_API_KEY,
        model=MISTRAL_MODEL,
        provider_name="mistral",
    )


def _openrouter_generate(prompt: str) -> str:
    if not OPENROUTER_API_KEY:
        raise QuotaExhaustedError("openrouter", "API key not configured")
    return _openai_compat_generate(
        prompt,
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        model=OPENROUTER_MODEL,
        provider_name="openrouter",
        # OpenRouter docs recommend these headers to avoid model-access restrictions
        extra_headers={
            "HTTP-Referer": "https://github.com/qframe",
            "X-Title": "qframe",
        },
    )


# ---------------------------------------------------------------------------
# Fallback chain — defined AFTER all provider functions
# ---------------------------------------------------------------------------

_PROVIDER_CHAIN: list[tuple[str, object]] = [
    ("groq",        _groq_generate),
    ("cerebras",    _cerebras_generate),
    ("deepseek",    _deepseek_generate),
    ("together",    _together_generate),
    ("mistral",     _mistral_generate),
    ("gemini",      _gemini_generate),
    ("openrouter",  _openrouter_generate),
]

_PROVIDER_MAP: dict[str, object] = {name: fn for name, fn in _PROVIDER_CHAIN}


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def generate(prompt: str) -> str:
    """
    Generate a completion using the configured LLM provider.

    Provider is set by LLM_PROVIDER env var (default: groq).
    On QuotaExhaustedError the router falls back through the ordered chain,
    skipping providers with no API key configured.

    Raises:
        RuntimeError: if every configured provider is exhausted.
        (other exceptions): non-quota errors propagate immediately.
    """
    import sys
    # Look up functions from the live module object so that test patches
    # (patch.object(module, "_groq_generate", ...)) are respected at call time.
    # All providers follow the naming convention _{name}_generate.
    _self = sys.modules[__name__]

    chain_names = [name for name, _ in _PROVIDER_CHAIN]
    if _PROVIDER not in _PROVIDER_MAP:
        raise ValueError(
            f"Unknown LLM_PROVIDER={_PROVIDER!r}. "
            f"Valid values: {chain_names}"
        )

    # Rotate so the configured primary is first, rest follow in order
    start = chain_names.index(_PROVIDER)
    ordered = _PROVIDER_CHAIN[start:] + _PROVIDER_CHAIN[:start]

    last_error: Exception | None = None
    for provider_name, _ in ordered:
        # Dynamic lookup — picks up any patches applied to the module attribute
        provider_fn = getattr(_self, f"_{provider_name}_generate")
        try:
            if provider_name != _PROVIDER:
                print(f"      LLM: falling back to {provider_name}")
            return provider_fn(prompt)
        except QuotaExhaustedError as exc:
            print(f"      {exc}")
            last_error = exc
            continue  # try next provider in chain

    raise RuntimeError(
        "All configured LLM providers exhausted. "
        "Check quotas or add more API keys to .env"
    ) from last_error
