"""EthAuditor — shared utility functions."""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


def safe_serialize(obj: Any) -> Any:
    """Recursively convert an object to a JSON-serializable form."""
    if isinstance(obj, dict):
        return {k: safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_serialize(item) for item in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return {k: safe_serialize(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
    return str(obj)


def invoke_with_retry(
    chain: Any,
    prompt: Any,
    *,
    max_retries: int = 3,
    base_delay: float = 5.0,
    max_delay: float = 60.0,
    label: str = "LLM",
) -> Any:
    """Invoke a LangChain chain/runnable with exponential-backoff retries.

    Retries on any exception (network errors, rate limits, server disconnects).

    Parameters
    ----------
    chain:
        A LangChain runnable (e.g. ``llm.with_structured_output(...)``).
    prompt:
        The prompt to pass to ``chain.invoke()``.
    max_retries:
        Maximum number of retry attempts after the first failure.
    base_delay:
        Initial delay in seconds before the first retry.
    max_delay:
        Upper cap for the delay between retries.
    label:
        Human-readable label for log messages.

    Returns
    -------
    The result of ``chain.invoke(prompt)``.

    Raises
    ------
    Exception
        Re-raises the last exception if all retries are exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(1 + max_retries):
        try:
            return chain.invoke(prompt)
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                delay = min(base_delay * (2 ** attempt), max_delay)
                logger.warning(
                    "[%s] attempt %d/%d failed (%s: %s) — retrying in %.1fs",
                    label, attempt + 1, 1 + max_retries,
                    type(exc).__name__, exc, delay,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "[%s] all %d attempts failed — giving up",
                    label, 1 + max_retries,
                )
    raise last_exc  # type: ignore[misc]

