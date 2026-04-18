"""
Shared Gemini client helper with retry logic for 503 (server overload)
and 429 (rate limit / quota).
"""
from __future__ import annotations

import re
import time

from google import genai
from google.genai import errors as genai_errors

from qframe.config import GEMINI_API_KEY, GEMINI_MODEL as _GEMINI_MODEL
_MAX_RETRIES = 5
_DEFAULT_WAIT = 15  # seconds between retries if no retry-after hint


def make_client() -> genai.Client:
    if not GEMINI_API_KEY:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Add it to projects/qframe/.env"
        )
    return genai.Client(api_key=GEMINI_API_KEY)


def _parse_retry_delay(exc: Exception) -> float:
    """Extract suggested retry delay from the error message (e.g. 'retry in 54s')."""
    match = re.search(r"retry[^\d]*(\d+(?:\.\d+)?)\s*s", str(exc), re.IGNORECASE)
    if match:
        return float(match.group(1)) + 2  # add 2s buffer
    return _DEFAULT_WAIT


def generate_with_retry(
    client: genai.Client,
    prompt: str,
    model: str = _GEMINI_MODEL,
) -> str:
    """
    Call Gemini with exponential-backoff retry on:
      - 503 ServerError  (high demand / transient)
      - 429 ClientError  (rate limit — wait the suggested retry-after time)
    """
    last_exc = None
    for attempt in range(_MAX_RETRIES):
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            return response.text

        except genai_errors.ServerError as exc:
            last_exc = exc
            wait = _DEFAULT_WAIT * (2 ** attempt)
            print(f"      Gemini {exc.code} server error — retrying in {wait}s (attempt {attempt+1}/{_MAX_RETRIES})")
            time.sleep(wait)

        except genai_errors.ClientError as exc:
            if exc.code == 429:
                last_exc = exc
                wait = _parse_retry_delay(exc)
                print(f"      Gemini 429 rate limit — waiting {wait:.0f}s (attempt {attempt+1}/{_MAX_RETRIES})")
                time.sleep(wait)
            else:
                raise  # 4xx errors other than 429 are not retryable

        except Exception:
            raise

    raise last_exc
