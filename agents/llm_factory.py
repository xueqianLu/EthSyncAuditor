"""LLM factory for provider-switchable agent execution (Claude/Gemini)."""

from __future__ import annotations

import os
from typing import Literal

LlmProvider = Literal["claude", "gemini"]

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    # Optional dependency: environment can still be supplied by shell/export.
    pass


def _normalize_provider(provider: str | None) -> LlmProvider:
    raw = (provider or os.getenv("ETHAUDITOR_LLM_PROVIDER", "claude")).strip().lower()
    if raw in {"claude", "anthropic"}:
        return "claude"
    if raw in {"gemini", "google"}:
        return "gemini"
    raise ValueError(
        f"Unsupported llm provider: {raw}. Expected one of: claude, gemini"
    )


def create_claude_llm(model: str = "claude-3-5-sonnet-latest", temperature: float = 0.0):
    """Create a Claude chat model.

    Requires ANTHROPIC_API_KEY in environment.
    """

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is missing. Please set it in environment (or .env)."
        )

    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=model, temperature=temperature)


def create_gemini_llm(model: str = "gemini-1.5-pro", temperature: float = 0.0):
    """Create a Gemini chat model.

    Requires GOOGLE_API_KEY in environment.
    """

    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY is missing. Please set it in environment (or .env)."
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=model, temperature=temperature)


def create_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
):
    """Create an LLM client by provider.

    Provider priority:
    1) explicit function argument
    2) ETHAUDITOR_LLM_PROVIDER env var
    3) default "claude"
    """

    chosen = _normalize_provider(provider)

    if chosen == "claude":
        return create_claude_llm(
            model=model or os.getenv("ETHAUDITOR_CLAUDE_MODEL", "claude-3-5-sonnet-latest"),
            temperature=temperature,
        )

    return create_gemini_llm(
        model=model or os.getenv("ETHAUDITOR_GEMINI_MODEL", "gemini-1.5-pro"),
        temperature=temperature,
    )
