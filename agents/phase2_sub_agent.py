"""EthAuditor — Phase 2 Sub-Agent.

ReAct agent that extracts the complete LSG (7 workflows) for a single client,
using call-graph directed hybrid search (Mode B).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jinja2 import Template

from config import LANGUAGE_GRAMMARS, WORKFLOW_IDS
from state import LSGFile

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "phase2_sub.j2"


def _load_prompt_template() -> Template:
    return Template(_PROMPT_PATH.read_text(encoding="utf-8"))


def build_phase2_sub_agent(client_name: str, llm=None):
    """Build a Phase 2 Sub-Agent for *client_name*.

    If *llm* is None, returns a mock implementation.
    """
    lang_key, _ = LANGUAGE_GRAMMARS[client_name]

    def _run(state: dict[str, Any]) -> dict[str, Any]:
        """Extract complete LSG from client source code."""
        guards = state.get("guards", [])
        actions = state.get("actions", [])
        a_class_feedback = state.get("a_class_feedback", [])

        template = _load_prompt_template()
        _prompt = template.render(
            client_name=client_name,
            language=lang_key,
            guards=guards,
            actions=actions,
            a_class_feedback=a_class_feedback,
        )

        if llm is not None:
            try:
                chain = llm.with_structured_output(LSGFile)
                lsg: LSGFile = chain.invoke(_prompt)
                return {"client_lsgs": {client_name: lsg.model_dump()}}
            except Exception:
                logger.error("LLM call failed for %s", client_name, exc_info=True)

        # Mock fallback
        logger.info("[phase2_sub_agent] client=%s — using mock response", client_name)
        workflows = []
        for wf_id in WORKFLOW_IDS:
            workflows.append({
                "id": wf_id,
                "name": wf_id.replace("_", " ").title(),
                "description": f"Mock {wf_id} workflow for {client_name}",
                "mode": "mock",
                "initial_state": f"{wf_id}.init",
                "states": [
                    {
                        "id": f"{wf_id}.init",
                        "label": "Init",
                        "category": "init",
                        "transitions": [{
                            "guard": "TRUE",
                            "actions": [],
                            "next_state": f"{wf_id}.done",
                            "evidence": None,
                        }],
                    },
                    {
                        "id": f"{wf_id}.done",
                        "label": "Done",
                        "category": "terminal",
                        "transitions": [],
                    },
                ],
            })

        lsg_dict = {
            "version": 1,
            "client": client_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "guards": list(guards),
            "actions": list(actions),
            "workflows": workflows,
        }
        return {"client_lsgs": {client_name: lsg_dict}}

    return _run
