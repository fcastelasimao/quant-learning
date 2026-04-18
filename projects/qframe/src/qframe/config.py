"""
Central config — loads API keys from .env and exposes them as constants.

All modules that need API keys import from here. Never read os.environ directly
in research code.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Walk up from this file to find the .env at the repo root
_repo_root = Path(__file__).parent.parent.parent
load_dotenv(_repo_root / ".env")


def _get(key: str, required: bool = False) -> str | None:
    val = os.environ.get(key)
    if required and not val:
        raise EnvironmentError(
            f"Missing required API key: {key}\n"
            f"Add it to {_repo_root / '.env'}"
        )
    return val or None


def _get_any(keys: list[str]) -> str | None:
    for k in keys:
        v = _get(k)
        if v:
            return v
    return None


# Gemini key aliasing:
# - Code historically uses GEMINI_API_KEY
# - Some docs/tools refer to GOOGLE_API_KEY (AI Studio)
GEMINI_API_KEY    = _get_any(["GEMINI_API_KEY", "GOOGLE_API_KEY"])
GEMINI_MODEL      = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
GROQ_API_KEY      = _get("GROQ_API_KEY")
ANTHROPIC_API_KEY = _get("ANTHROPIC_API_KEY")

# OpenAI-compatible providers (optional — all participate in the fallback chain)
CEREBRAS_API_KEY   = _get("CEREBRAS_API_KEY")
CEREBRAS_MODEL     = os.environ.get("CEREBRAS_MODEL", "qwen-3-235b-a22b-instruct-2507")
TOGETHER_API_KEY   = _get("TOGETHER_API_KEY")
TOGETHER_MODEL     = os.environ.get("TOGETHER_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo")
DEEPSEEK_API_KEY   = _get("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL     = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
MISTRAL_API_KEY    = _get("MISTRAL_API_KEY")
MISTRAL_MODEL      = os.environ.get("MISTRAL_MODEL", "mistral-large-latest")
OPENROUTER_API_KEY = _get("OPENROUTER_API_KEY")
OPENROUTER_MODEL   = os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")
