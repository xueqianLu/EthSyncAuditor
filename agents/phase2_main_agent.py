"""EthAuditor — Phase 2 Main Agent.

Horizontal comparison of client LSGs. Classifies differences as A-class
(implementation) or B-class (logic), computes logic_diff_rate.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from jinja2 import Template

from config import CLIENT_NAMES, WORKFLOW_IDS
from state import DiffItem, DiffReport

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "phase2_main.j2"


def _load_prompt_template() -> Template:
    return Template(_PROMPT_PATH.read_text(encoding="utf-8"))


def build_phase2_main_agent(llm=None):
    """Build the Phase 2 Main Agent.

    If *llm* is None, returns a deterministic comparison implementation.
    """

    def _run(state: dict[str, Any]) -> dict[str, Any]:
        """Compare client LSGs, classify diffs, compute logic_diff_rate."""
        client_lsgs = state.get("client_lsgs", {})

        if llm is not None:
            template = _load_prompt_template()
            _prompt = template.render(client_lsgs=client_lsgs)
            try:
                chain = llm.with_structured_output(DiffReport)
                report: DiffReport = chain.invoke(_prompt)
                return {
                    "diff_report": report.model_dump(),
                    "logic_diff_rate": report.logic_diff_rate,
                    "a_class_feedback": [d.model_dump() for d in report.a_class_diffs],
                }
            except Exception:
                logger.error("LLM call failed for phase2_main", exc_info=True)

        # Deterministic comparison fallback
        logger.info("[phase2_main_agent] deterministic comparison of %d clients", len(client_lsgs))

        a_diffs: list[dict] = []
        b_diffs: list[dict] = []
        total_items = 0

        # Build triple index: (workflow_id, state_id, guard) → {client: transition}
        triple_index: dict[tuple[str, str, str], dict[str, Any]] = {}

        for client, lsg in client_lsgs.items():
            for wf in lsg.get("workflows", []):
                wf_id = wf["id"]
                for st in wf.get("states", []):
                    state_id = st["id"]
                    for tr in st.get("transitions", []):
                        guard = tr["guard"]
                        key = (wf_id, state_id, guard)
                        if key not in triple_index:
                            triple_index[key] = {}
                        triple_index[key][client] = tr

        # Compare
        for (wf_id, state_id, guard), clients_map in triple_index.items():
            total_items += 1
            present = set(clients_map.keys())
            all_clients = set(client_lsgs.keys())

            if present != all_clients:
                missing = all_clients - present
                b_diffs.append({
                    "workflow_id": wf_id,
                    "state_id": state_id,
                    "transition_guard": guard,
                    "diff_type": "B",
                    "description": f"Transition missing in: {', '.join(sorted(missing))}",
                    "involved_clients": sorted(missing),
                    "evidence": {},
                })

        logic_diff_rate = len(b_diffs) / max(total_items, 1)

        return {
            "diff_report": {
                "a_class_diffs": a_diffs,
                "b_class_diffs": b_diffs,
                "logic_diff_rate": logic_diff_rate,
            },
            "logic_diff_rate": logic_diff_rate,
            "a_class_feedback": a_diffs,
        }

    return _run
