"""
Unit tests for the multi-provider LLM router (_llm.py).

All HTTP calls are mocked — no real API requests are made.
Tests cover:
  - _parse_retry_delay
  - QuotaExhaustedError
  - generate() fallback chain behaviour
  - _openai_compat_generate (requires openai to be installed)
"""
from __future__ import annotations

import importlib
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# _parse_retry_delay
# ---------------------------------------------------------------------------

class TestParseRetryDelay:
    def _fn(self):
        from qframe.pipeline.agents._llm import _parse_retry_delay
        return _parse_retry_delay

    def test_minutes_and_seconds(self):
        fn = self._fn()
        # "2m30.303s" → 2*60 + 30.303 + 2 = 152.303
        result = fn("retry in 2m30.303s")
        assert abs(result - 152.303) < 0.01

    def test_plain_seconds(self):
        fn = self._fn()
        # "45s" → 45 + 2 = 47
        assert fn("wait 45s") == pytest.approx(47.0)

    def test_fractional_seconds(self):
        fn = self._fn()
        # "1.5s" → 1.5 + 2 = 3.5
        assert fn("please retry after 1.5s") == pytest.approx(3.5)

    def test_unrecognised_returns_default(self):
        fn = self._fn()
        # No time pattern → _DEFAULT_WAIT = 15
        assert fn("unknown rate limit error") == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# QuotaExhaustedError
# ---------------------------------------------------------------------------

class TestQuotaExhaustedError:
    def test_str_contains_provider(self):
        from qframe.pipeline.agents._llm import QuotaExhaustedError
        exc = QuotaExhaustedError("groq", "tokens per day exceeded")
        assert "groq" in str(exc)
        assert "tokens per day exceeded" in str(exc)

    def test_provider_attribute(self):
        from qframe.pipeline.agents._llm import QuotaExhaustedError
        exc = QuotaExhaustedError("cerebras", "daily limit")
        assert exc.provider == "cerebras"

    def test_is_exception_subclass(self):
        from qframe.pipeline.agents._llm import QuotaExhaustedError
        assert issubclass(QuotaExhaustedError, Exception)

    def test_empty_detail(self):
        from qframe.pipeline.agents._llm import QuotaExhaustedError
        exc = QuotaExhaustedError("gemini")
        assert exc.provider == "gemini"
        assert "gemini" in str(exc)


# ---------------------------------------------------------------------------
# generate() fallback chain
# ---------------------------------------------------------------------------

class TestGenerateFallback:
    """
    Tests for the fallback chain logic in generate().
    All provider functions are patched — no real HTTP calls.
    """

    def test_primary_success_no_fallback(self):
        """When primary succeeds, no fallback is triggered."""
        from qframe.pipeline.agents import _llm

        with patch.object(_llm, "_groq_generate", return_value="groq response"):
            result = _llm.generate("test prompt")
        assert result == "groq response"

    def test_falls_back_when_primary_quota_exhausted(self):
        """Groq quota exhausted → should fall back to Cerebras."""
        from qframe.pipeline.agents import _llm
        from qframe.pipeline.agents._llm import QuotaExhaustedError

        with patch.object(_llm, "_groq_generate",
                          side_effect=QuotaExhaustedError("groq", "daily limit")), \
             patch.object(_llm, "_cerebras_generate",
                          return_value="cerebras response"):
            result = _llm.generate("test prompt")
        assert result == "cerebras response"

    def test_falls_back_through_multiple_providers(self):
        """Groq + Cerebras exhausted → should reach DeepSeek."""
        from qframe.pipeline.agents import _llm
        from qframe.pipeline.agents._llm import QuotaExhaustedError

        quota = QuotaExhaustedError

        with patch.object(_llm, "_groq_generate",
                          side_effect=quota("groq", "daily")), \
             patch.object(_llm, "_cerebras_generate",
                          side_effect=quota("cerebras", "daily")), \
             patch.object(_llm, "_deepseek_generate",
                          return_value="deepseek response"):
            result = _llm.generate("test prompt")
        assert result == "deepseek response"

    def test_raises_when_all_providers_exhausted(self):
        """All 7 providers exhausted → RuntimeError with informative message."""
        from qframe.pipeline.agents import _llm
        from qframe.pipeline.agents._llm import QuotaExhaustedError

        quota = QuotaExhaustedError

        with patch.object(_llm, "_groq_generate",
                          side_effect=quota("groq", "daily")), \
             patch.object(_llm, "_cerebras_generate",
                          side_effect=quota("cerebras", "daily")), \
             patch.object(_llm, "_deepseek_generate",
                          side_effect=quota("deepseek", "daily")), \
             patch.object(_llm, "_together_generate",
                          side_effect=quota("together", "daily")), \
             patch.object(_llm, "_mistral_generate",
                          side_effect=quota("mistral", "daily")), \
             patch.object(_llm, "_gemini_generate",
                          side_effect=quota("gemini", "daily")), \
             patch.object(_llm, "_openrouter_generate",
                          side_effect=quota("openrouter", "daily")):
            with pytest.raises(RuntimeError, match="All configured LLM providers exhausted"):
                _llm.generate("test prompt")

    def test_unconfigured_provider_is_skipped(self):
        """
        A provider with no API key raises QuotaExhaustedError("x", "key not configured").
        The chain must skip it and try the next configured provider.
        """
        from qframe.pipeline.agents import _llm
        from qframe.pipeline.agents._llm import QuotaExhaustedError

        with patch.object(_llm, "_groq_generate",
                          side_effect=QuotaExhaustedError("groq", "daily")), \
             patch.object(_llm, "_cerebras_generate",
                          side_effect=QuotaExhaustedError("cerebras", "API key not configured")), \
             patch.object(_llm, "_deepseek_generate",
                          return_value="deepseek response"):
            result = _llm.generate("test prompt")
        assert result == "deepseek response"

    def test_non_quota_error_propagates_immediately(self):
        """
        A non-quota error (e.g. invalid API key 401, bad request 400) must NOT
        trigger a fallback — it should propagate immediately so the user can fix it.
        """
        from qframe.pipeline.agents import _llm

        with patch.object(_llm, "_groq_generate",
                          side_effect=ValueError("invalid auth")):
            with pytest.raises(ValueError, match="invalid auth"):
                _llm.generate("test prompt")

    def test_invalid_provider_name_raises_value_error(self):
        """Setting LLM_PROVIDER to an unknown value should raise ValueError."""
        from qframe.pipeline.agents import _llm

        original = _llm._PROVIDER
        try:
            _llm._PROVIDER = "unknown_provider"
            with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
                _llm.generate("test prompt")
        finally:
            _llm._PROVIDER = original  # restore


# ---------------------------------------------------------------------------
# Access denied failover (optional)
# ---------------------------------------------------------------------------

class TestAccessDeniedFailover:
    def test_default_strict_mode_does_not_failover(self, monkeypatch):
        """
        By default, access-denied errors must NOT be converted into QuotaExhaustedError.
        This preserves the project's "fix configuration/access, don't silently bypass"
        philosophy unless the user explicitly opts in.
        """
        monkeypatch.delenv("LLM_FAILOVER_ON_ACCESS_DENIED", raising=False)
        from qframe.pipeline.agents import _llm as llm
        importlib.reload(llm)

        # Should not raise in strict mode
        llm._maybe_failover_on_access_denied("groq", 403, "Access denied. Please check your network settings.")

    def test_opt_in_mode_converts_403_to_quota_exhausted(self, monkeypatch):
        monkeypatch.setenv("LLM_FAILOVER_ON_ACCESS_DENIED", "1")
        from qframe.pipeline.agents import _llm as llm
        importlib.reload(llm)

        with pytest.raises(llm.QuotaExhaustedError) as exc_info:
            llm._maybe_failover_on_access_denied(
                "groq", 403, "Access denied. Please check your network settings."
            )
        assert exc_info.value.provider == "groq"

    def test_is_access_denied_error_heuristics(self):
        from qframe.pipeline.agents import _llm as llm
        assert llm._is_access_denied_error(403, "anything")
        assert llm._is_access_denied_error(None, "Access denied. Please check your network settings.")
        assert not llm._is_access_denied_error(429, "daily quota exceeded")


# ---------------------------------------------------------------------------
# _openai_compat_generate — requires openai to be installed
# ---------------------------------------------------------------------------

openai = pytest.importorskip("openai", reason="openai package not installed")


class TestOpenAICompatGenerate:
    """
    Tests for the shared OpenAI-compatible backend.
    Requires `openai` to be installed (skipped otherwise).
    Uses patch("openai.OpenAI") to intercept all HTTP calls.
    """

    def _make_mock_client(self, response_text: str = "hello") -> MagicMock:
        """Build a mock openai.OpenAI client that returns response_text."""
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = response_text
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp
        return mock_client

    def test_successful_response(self):
        """Happy path: client returns a response, function returns the content string."""
        from qframe.pipeline.agents._llm import _openai_compat_generate

        mock_client = self._make_mock_client("hello world")
        with patch("openai.OpenAI", return_value=mock_client):
            result = _openai_compat_generate(
                "test prompt",
                base_url="https://api.cerebras.ai/v1",
                api_key="fake-key",
                model="llama-3.3-70b",
                provider_name="cerebras",
            )
        assert result == "hello world"

    def test_extra_headers_passed_to_client(self):
        """extra_headers must be forwarded as default_headers to OpenAI()."""
        from qframe.pipeline.agents._llm import _openai_compat_generate

        mock_client = self._make_mock_client("ok")
        headers = {"HTTP-Referer": "https://test", "X-Title": "test"}
        with patch("openai.OpenAI", return_value=mock_client) as MockOpenAI:
            _openai_compat_generate(
                "prompt",
                base_url="https://openrouter.ai/api/v1",
                api_key="fake",
                model="model",
                provider_name="openrouter",
                extra_headers=headers,
            )
        MockOpenAI.assert_called_once_with(
            api_key="fake",
            base_url="https://openrouter.ai/api/v1",
            default_headers=headers,
        )

    def test_quota_rate_limit_raises_quota_exhausted(self):
        """RateLimitError with 'daily' in message → QuotaExhaustedError, no retry."""
        from qframe.pipeline.agents._llm import _openai_compat_generate, QuotaExhaustedError
        from openai import RateLimitError

        # openai.RateLimitError requires response + body args
        fake_response = MagicMock()
        fake_response.status_code = 429
        fake_response.headers = {}
        fake_exc = RateLimitError(
            "daily quota exceeded for this model",
            response=fake_response,
            body={"error": {"message": "daily quota exceeded"}},
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = fake_exc

        with patch("openai.OpenAI", return_value=mock_client):
            with pytest.raises(QuotaExhaustedError) as exc_info:
                _openai_compat_generate(
                    "prompt",
                    base_url="https://api.cerebras.ai/v1",
                    api_key="fake",
                    model="llama-3.3-70b",
                    provider_name="cerebras",
                )
        assert exc_info.value.provider == "cerebras"

    def test_per_minute_rate_limit_retries(self):
        """RateLimitError without daily/quota keywords → retries with sleep."""
        from qframe.pipeline.agents._llm import _openai_compat_generate
        from openai import RateLimitError

        fake_response = MagicMock()
        fake_response.status_code = 429
        fake_response.headers = {}
        per_minute_exc = RateLimitError(
            "rate limit reached, retry after 5s",
            response=fake_response,
            body={"error": {"message": "rate limit: retry after 5s"}},
        )

        # Fail twice with per-minute error, then succeed
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "success after retry"
        mock_client.chat.completions.create.side_effect = [
            per_minute_exc,
            per_minute_exc,
            mock_resp,
        ]

        with patch("openai.OpenAI", return_value=mock_client), \
             patch("time.sleep"):  # skip actual waiting
            result = _openai_compat_generate(
                "prompt",
                base_url="https://api.cerebras.ai/v1",
                api_key="fake",
                model="llama-3.3-70b",
                provider_name="cerebras",
            )
        assert result == "success after retry"
        assert mock_client.chat.completions.create.call_count == 3

    def test_5xx_server_error_retries(self):
        """APIStatusError with status >= 500 → retries with exponential backoff."""
        from qframe.pipeline.agents._llm import _openai_compat_generate
        from openai import APIStatusError

        fake_response = MagicMock()
        fake_response.status_code = 503
        fake_response.headers = {}
        server_err = APIStatusError(
            "Service temporarily unavailable",
            response=fake_response,
            body={"error": "overloaded"},
        )

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "recovered"
        mock_client.chat.completions.create.side_effect = [server_err, mock_resp]

        with patch("openai.OpenAI", return_value=mock_client), \
             patch("time.sleep"):
            result = _openai_compat_generate(
                "prompt",
                base_url="https://api.mistral.ai/v1",
                api_key="fake",
                model="mistral-large-latest",
                provider_name="mistral",
            )
        assert result == "recovered"
        assert mock_client.chat.completions.create.call_count == 2
