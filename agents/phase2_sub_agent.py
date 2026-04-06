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
from utils import invoke_with_retry, serialize_lsg_compact, summarize_vocab_for_prompt

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "phase2_sub.j2"


def _load_prompt_template() -> Template:
    return Template(_PROMPT_PATH.read_text(encoding="utf-8"))


def _collect_referenced_names(workflows: list[dict]) -> tuple[set[str], set[str]]:
    """Scan workflows and return (guard_names, action_names) actually referenced."""
    guard_names: set[str] = set()
    action_names: set[str] = set()
    for wf in workflows:
        for st in wf.get("states", []):
            for tr in st.get("transitions", []):
                g = tr.get("guard", "")
                if g and g != "TRUE":
                    guard_names.add(g)
                for a in tr.get("actions", []):
                    if a:
                        action_names.add(a)
    return guard_names, action_names


def _backfill_vocab(
    lsg_dict: dict,
    global_guards: list[dict],
    global_actions: list[dict],
) -> None:
    """Populate ``guards``/``actions`` in *lsg_dict* from the global vocabulary.

    If the LLM already returned non-empty guards/actions, keep them.
    Otherwise, filter the global vocabulary down to only those names that
    appear in the client's workflow transitions — producing a meaningful
    per-client vocabulary rather than duplicating the entire global set.
    """
    if lsg_dict.get("guards") and lsg_dict.get("actions"):
        return  # LLM already populated them

    ref_guards, ref_actions = _collect_referenced_names(
        lsg_dict.get("workflows", []),
    )

    if not lsg_dict.get("guards") and global_guards:
        lsg_dict["guards"] = [
            g for g in global_guards
            if g.get("name") in ref_guards
        ]
        logger.debug(
            "Backfilled %d/%d guards for %s",
            len(lsg_dict["guards"]), len(ref_guards),
            lsg_dict.get("client", "?"),
        )

    if not lsg_dict.get("actions") and global_actions:
        lsg_dict["actions"] = [
            a for a in global_actions
            if a.get("name") in ref_actions
        ]
        logger.debug(
            "Backfilled %d/%d actions for %s",
            len(lsg_dict["actions"]), len(ref_actions),
            lsg_dict.get("client", "?"),
        )


def build_phase2_sub_agent(client_name: str, llm=None, callbacks=None):
    """Build a Phase 2 Sub-Agent for *client_name*.

    If *llm* is None, returns a mock implementation.
    """
    lang_key, _ = LANGUAGE_GRAMMARS[client_name]

    def _run(state: dict[str, Any]) -> dict[str, Any]:
        """Extract complete LSG from client source code."""
        guards = state.get("guards", [])
        actions = state.get("actions", [])
        iteration = state.get("phase2_iteration", 1)

        # ── Filter A-class feedback to this client only ─────────────────
        # A-class feedback contains vocabulary alignment directives from
        # the Main Agent.  Only pass directives relevant to *this* client
        # so the agent knows which guard/action names to rename.
        all_feedback = state.get("a_class_feedback", [])
        a_class_feedback = [
            fb for fb in all_feedback
            if client_name in fb.get("involved_clients", [])
        ]

        # ── Compact vocabulary summary (avoid token overflow) ───────────
        vocab = summarize_vocab_for_prompt(guards, actions, max_full_entries=80)

        # ── Previous-iteration LSG for incremental refinement ───────────
        previous_lsg_yaml: str | None = None
        prev_lsg = state.get("client_lsgs", {}).get(client_name)
        if prev_lsg and iteration > 1:
            previous_lsg_yaml = serialize_lsg_compact(
                prev_lsg, strip_evidence=True,
            )
            logger.info(
                "[phase2_sub_agent] client=%s — feeding back previous LSG "
                "(%d lines) for incremental refinement",
                client_name, previous_lsg_yaml.count("\n"),
            )

        # ── Sparsity hints (which workflows need expansion) ─────────────
        sparsity_hints = [
            h for h in state.get("sparsity_hints", [])
            if h.get("client") == client_name
        ]

        template = _load_prompt_template()
        _prompt = template.render(
            client_name=client_name,
            language=lang_key,
            vocab=vocab,
            a_class_feedback=a_class_feedback,
            previous_lsg_yaml=previous_lsg_yaml,
            iteration=iteration,
            sparsity_hints=sparsity_hints,
        )

        if llm is not None:
            try:
                chain = llm.with_structured_output(LSGFile)
                lsg: LSGFile = invoke_with_retry(
                    chain, _prompt, label=f"phase2_sub/{client_name}",
                    callbacks=callbacks,
                )
                lsg_dict = lsg.model_dump()
                # LLM often returns empty guards/actions — backfill from
                # global vocabulary filtered to names referenced in workflows.
                _backfill_vocab(lsg_dict, guards, actions)
                return {"client_lsgs": {client_name: lsg_dict}}
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
