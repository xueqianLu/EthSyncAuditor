"""EthAuditor — shared utility functions."""

from __future__ import annotations

from typing import Any


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
