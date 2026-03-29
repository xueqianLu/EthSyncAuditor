"""LangChain callback-based audit logger for EthAuditor."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable

from langchain_core.callbacks import BaseCallbackHandler

ROOT_DIR = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT_DIR / "output" / "audit_logs"


def _now_ts() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class AuditCallbackHandler(BaseCallbackHandler):
    """Capture llm start/end events and persist structured audit records."""

    def __init__(self, context_getter: Callable[[], dict[str, Any]]):
        super().__init__()
        self._context_getter = context_getter
        self._pending: dict[str, dict[str, Any]] = {}
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id", f"no-run-id-{_now_ts()}"))
        self._pending[run_id] = {
            "serialized": serialized,
            "prompts": prompts,
            "started_at": datetime.now(tz=timezone.utc).isoformat(),
        }

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id", "unknown-run-id"))
        started = self._pending.pop(run_id, {})

        ctx = self._context_getter() if self._context_getter else {}
        phase = ctx.get("phase", "unknown")
        iteration = ctx.get("iteration", "unknown")
        agent_type = ctx.get("agent_type", "unknown")

        payload = {
            "phase": phase,
            "iteration": iteration,
            "agent_type": agent_type,
            "run_id": run_id,
            "started": started,
            "ended_at": datetime.now(tz=timezone.utc).isoformat(),
            "llm_response": getattr(response, "model_dump", lambda: str(response))(),
            # Note: private model reasoning is not exposed via callback APIs.
            "chain_of_thought": "not_available_via_callback",
        }

        filename = f"audit_phase{phase}_iter{iteration}_{agent_type}_{_now_ts()}.json"
        out = AUDIT_DIR / filename
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def make_audit_callback(state_getter: Callable[[], dict[str, Any]]) -> AuditCallbackHandler:
    """Create callback handler that reads phase/iteration/agent from current state."""

    return AuditCallbackHandler(state_getter)
