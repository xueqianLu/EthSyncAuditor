"""EthAuditor — Phase 2 Sub-Agent.

Extracts a **single workflow** LSG for one client using call-graph directed
hybrid search (Mode B).  The workflow to extract is specified by
``state["current_workflow"]``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Template

from config import LANGUAGE_GRAMMARS
from state import LSGFile
from utils import invoke_with_retry, summarize_vocab_for_prompt

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "phase2_sub.j2"


def _load_prompt_template() -> Template:
    return Template(_PROMPT_PATH.read_text(encoding="utf-8"))


def _extract_workflow(lsg: dict, wf_id: str) -> dict | None:
    """Extract a single workflow dict from a full LSG dict."""
    for wf in lsg.get("workflows", []):
        if wf.get("id") == wf_id:
            return wf
    return None


def _replace_workflow(lsg: dict, wf_id: str, new_wf: dict) -> dict:
    """Return a copy of *lsg* with the workflow *wf_id* replaced by *new_wf*."""
    updated = dict(lsg)
    new_workflows = []
    replaced = False
    for wf in lsg.get("workflows", []):
        if wf.get("id") == wf_id:
            new_workflows.append(new_wf)
            replaced = True
        else:
            new_workflows.append(wf)
    if not replaced:
        new_workflows.append(new_wf)
    updated["workflows"] = new_workflows
    return updated


def _serialize_workflow_yaml(wf: dict) -> str:
    """Serialize a single workflow dict to compact YAML."""
    # Strip evidence to save tokens
    wf_copy = dict(wf)
    new_states = []
    for st in wf_copy.get("states", []):
        st_copy = dict(st)
        new_trans = []
        for tr in st_copy.get("transitions", []):
            new_trans.append({k: v for k, v in tr.items() if k != "evidence"})
        st_copy["transitions"] = new_trans
        new_states.append(st_copy)
    wf_copy["states"] = new_states
    return yaml.dump(wf_copy, default_flow_style=False, allow_unicode=True, sort_keys=False)


def build_phase2_sub_agent(client_name: str, llm=None, callbacks=None):
    """Build a Phase 2 Sub-Agent for *client_name*.

    If *llm* is None, returns a mock implementation.
    """
    lang_key, _ = LANGUAGE_GRAMMARS[client_name]

    def _run(state: dict[str, Any]) -> dict[str, Any]:
        """Extract a single workflow from client source code."""
        guards = state.get("guards", [])
        actions = state.get("actions", [])
        iteration = state.get("phase2_iteration", 1)
        current_wf = state.get("current_workflow", "")

        # ── Get existing full LSG for this client ───────────────────────
        existing_lsg = state.get("client_lsgs", {}).get(client_name, {})

        # ── Filter A-class feedback to this client + workflow ───────────
        all_feedback = state.get("a_class_feedback", [])
        a_class_feedback = [
            fb for fb in all_feedback
            if client_name in fb.get("involved_clients", [])
            and fb.get("workflow_id") == current_wf
        ]

        # ── Compact vocabulary summary ──────────────────────────────────
        vocab = summarize_vocab_for_prompt(guards, actions, max_full_entries=80)

        # ── Previous iteration's workflow for incremental refinement ────
        previous_wf_yaml: str | None = None
        prev_wf = _extract_workflow(existing_lsg, current_wf)
        if prev_wf and iteration > 1:
            previous_wf_yaml = _serialize_workflow_yaml(prev_wf)
            logger.info(
                "[phase2_sub_agent] client=%s wf=%s — feeding back previous "
                "workflow (%d lines)",
                client_name, current_wf, previous_wf_yaml.count("\n"),
            )
        elif prev_wf and iteration == 1:
            # First iteration: show merged baseline as reference
            previous_wf_yaml = _serialize_workflow_yaml(prev_wf)
            logger.info(
                "[phase2_sub_agent] client=%s wf=%s — using merged baseline "
                "(%d lines)",
                client_name, current_wf, previous_wf_yaml.count("\n"),
            )

        # ── Sparsity hints for this workflow ────────────────────────────
        sparsity_hints = [
            h for h in state.get("sparsity_hints", [])
            if h.get("client") == client_name
            and h.get("workflow_id") == current_wf
        ]

        template = _load_prompt_template()
        _prompt = template.render(
            client_name=client_name,
            language=lang_key,
            vocab=vocab,
            workflow_id=current_wf,
            a_class_feedback=a_class_feedback,
            previous_wf_yaml=previous_wf_yaml,
            iteration=iteration,
            sparsity_hints=sparsity_hints,
        )

        if llm is not None:
            try:
                chain = llm.with_structured_output(LSGFile)
                lsg: LSGFile = invoke_with_retry(
                    chain, _prompt, label=f"phase2_sub/{client_name}/{current_wf}",
                    callbacks=callbacks,
                )
                lsg_dict = lsg.model_dump()
                # Extract the target workflow from LLM output
                new_wf = _extract_workflow(lsg_dict, current_wf)
                if new_wf is None and lsg_dict.get("workflows"):
                    # LLM might return it as the only workflow
                    new_wf = lsg_dict["workflows"][0]
                    new_wf["id"] = current_wf  # ensure correct id

                if new_wf:
                    updated_lsg = _replace_workflow(existing_lsg, current_wf, new_wf)
                    updated_lsg["generated_at"] = datetime.now(timezone.utc).isoformat()
                    return {"client_lsgs": {client_name: updated_lsg}}

                logger.warning(
                    "[phase2_sub_agent] LLM returned no workflow for %s/%s",
                    client_name, current_wf,
                )
            except Exception:
                logger.error(
                    "LLM call failed for %s/%s", client_name, current_wf,
                    exc_info=True,
                )

        # ── Mock fallback ───────────────────────────────────────────────
        logger.info(
            "[phase2_sub_agent] client=%s wf=%s — using mock response",
            client_name, current_wf,
        )
        mock_wf = {
            "id": current_wf,
            "name": current_wf.replace("_", " ").title(),
            "description": f"Mock {current_wf} workflow for {client_name}",
            "mode": "mock",
            "initial_state": f"{current_wf}.init",
            "states": [
                {
                    "id": f"{current_wf}.init",
                    "label": "Init",
                    "category": "init",
                    "transitions": [{
                        "guard": "TRUE",
                        "actions": [],
                        "next_state": f"{current_wf}.done",
                        "evidence": None,
                    }],
                },
                {
                    "id": f"{current_wf}.done",
                    "label": "Done",
                    "category": "terminal",
                    "transitions": [],
                },
            ],
        }

        if existing_lsg:
            updated_lsg = _replace_workflow(existing_lsg, current_wf, mock_wf)
        else:
            updated_lsg = {
                "version": 1,
                "client": client_name,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "guards": list(guards),
                "actions": list(actions),
                "workflows": [mock_wf],
            }
        return {"client_lsgs": {client_name: updated_lsg}}

    return _run
