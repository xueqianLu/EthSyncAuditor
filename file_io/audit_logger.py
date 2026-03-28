"""EthAuditor — Audit logger via LangChain callbacks.

Records every LLM invocation's prompt, chain-of-thought and raw response
as structured JSON files under ``./output/audit_logs/``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from config import AUDIT_LOG_PATH

logger = logging.getLogger(__name__)


class AuditLogCallback:
    """LangChain-compatible callback handler for audit logging.

    Files are written as:
        audit_phase<P>_iter<N>_<agent_type>_<timestamp>.json
    """

    def __init__(self, phase: int = 0, iteration: int = 0, agent_type: str = "unknown"):
        self.phase = phase
        self.iteration = iteration
        self.agent_type = agent_type
        self._paths: list[str] = []

    # ── Callback hooks ──────────────────────────────────────────────────

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called before an LLM call."""
        self._save_event("llm_start", {
            "serialized": _safe_serialize(serialized),
            "prompts": prompts,
        })

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called after an LLM call."""
        self._save_event("llm_end", {
            "response": _safe_serialize(response),
        })

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called on LLM error."""
        self._save_event("llm_error", {
            "error": str(error),
        })

    # ── Internal ────────────────────────────────────────────────────────

    def _save_event(self, event_type: str, payload: dict[str, Any]) -> None:
        AUDIT_LOG_PATH.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
        filename = (
            f"audit_phase{self.phase}_iter{self.iteration}"
            f"_{self.agent_type}_{ts}.json"
        )
        path = AUDIT_LOG_PATH / filename

        record = {
            "event_type": event_type,
            "phase": self.phase,
            "iteration": self.iteration,
            "agent_type": self.agent_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False, default=str)

        self._paths.append(str(path))
        logger.debug("[audit_logger] %s → %s", event_type, path)

    @property
    def paths(self) -> list[str]:
        return list(self._paths)


def _safe_serialize(obj: Any) -> Any:
    """Make an object JSON-serializable."""
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(item) for item in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return {k: _safe_serialize(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
    return str(obj)
