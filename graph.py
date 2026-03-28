"""EthAuditor — LangGraph graph definition with mock nodes.

Implements the full topology:
  preprocess → Phase 1 (fan-out sub-agents → main → router) → Phase 2 (fan-out → main → router) → END

All agent nodes are mocks that return placeholder data.
Conditional edges implement convergence checks and MAX_ITER guards.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.types import Send

import config
from config import (
    CLIENT_NAMES,
    WORKFLOW_IDS,
)
from state import GlobalState

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────
# Helper: initial state factory
# ────────────────────────────────────────────────────────────────────────


def make_initial_state() -> dict[str, Any]:
    """Return a fresh initial GlobalState dictionary."""
    return {
        "current_phase": 0,
        "phase1_iteration": 0,
        "phase2_iteration": 0,
        "guards": [],
        "actions": [],
        "vocab_version": 0,
        "diff_rate": 1.0,
        "client_lsgs": {},
        "diff_report": {},
        "logic_diff_rate": 1.0,
        "converged_phase1": False,
        "converged_phase2": False,
        "force_stopped": False,
        "preprocess_done": False,
        "preprocess_status": {},
        "audit_log_paths": [],
        "discovery_reports": [],
        "a_class_feedback": [],
    }


# ────────────────────────────────────────────────────────────────────────
# Node functions
# ────────────────────────────────────────────────────────────────────────


def preprocess_node(state: GlobalState) -> dict[str, Any]:
    """Offline preprocessing node (mock).

    Checks whether preprocessing products exist; if so, skips.
    """
    logger.info("[preprocess_node] phase=0 — checking preprocessing status")
    if state.get("preprocess_done"):
        logger.info("[preprocess_node] Preprocessing already done — skipping.")
        return {}

    statuses: dict[str, dict] = {}
    for client in CLIENT_NAMES:
        logger.info("[preprocess_node] Processing client=%s (mock)", client)
        statuses[client] = {
            "symbols_ready": True,
            "callgraph_ready": True,
            "vector_index_ready": True,
            "bm25_index_ready": True,
        }

    return {
        "preprocess_done": True,
        "preprocess_status": statuses,
        "current_phase": 1,
        "phase1_iteration": 1,
    }


# ── Phase 1 nodes ──────────────────────────────────────────────────────


def phase1_sub_agent_node(state: GlobalState) -> dict[str, Any]:
    """Phase 1 Sub-Agent node (mock).

    Simulates vocabulary discovery for a single client.
    The client name is passed via state['_client_name'] by Send().
    """
    client_name: str = state.get("_client_name", "unknown")  # type: ignore[arg-type]
    iteration = state.get("phase1_iteration", 1)
    logger.info(
        "[phase1_sub_agent] client=%s phase=1 iter=%d (mock)",
        client_name,
        iteration,
    )

    report = {
        "client_name": client_name,
        "new_guards": [],
        "new_actions": [],
    }

    if iteration == 1:
        report["new_guards"] = [
            {
                "name": f"MockGuard_{client_name}",
                "category": "mock",
                "description": f"Mock guard discovered in {client_name}",
            }
        ]
        report["new_actions"] = [
            {
                "name": f"MockAction_{client_name}",
                "category": "mock",
                "description": f"Mock action discovered in {client_name}",
            }
        ]

    return {"discovery_reports": [report]}


def phase1_main_agent_node(state: GlobalState) -> dict[str, Any]:
    """Phase 1 Main Agent node (mock).

    Merges discovery reports, computes diff_rate, bumps vocab_version.
    """
    iteration = state.get("phase1_iteration", 1)
    reports = state.get("discovery_reports", [])
    existing_guards = list(state.get("guards", []))
    existing_actions = list(state.get("actions", []))
    logger.info(
        "[phase1_main_agent] phase=1 iter=%d reports=%d (mock)",
        iteration,
        len(reports),
    )

    new_guards: list[dict] = []
    new_actions: list[dict] = []
    existing_guard_names = {g["name"] for g in existing_guards}
    existing_action_names = {a["name"] for a in existing_actions}

    for report in reports:
        for g in report.get("new_guards", []):
            if g["name"] not in existing_guard_names:
                new_guards.append(g)
                existing_guard_names.add(g["name"])
        for a in report.get("new_actions", []):
            if a["name"] not in existing_action_names:
                new_actions.append(a)
                existing_action_names.add(a["name"])

    total_vocab = len(existing_guards) + len(existing_actions) + len(new_guards) + len(new_actions)
    diff_rate = (len(new_guards) + len(new_actions)) / max(total_vocab, 1)

    logger.info(
        "[phase1_main_agent] new_guards=%d new_actions=%d diff_rate=%.4f",
        len(new_guards),
        len(new_actions),
        diff_rate,
    )

    return {
        "guards": new_guards,
        "actions": new_actions,
        "vocab_version": state.get("vocab_version", 0) + 1,
        "diff_rate": diff_rate,
        "discovery_reports": [],  # reset for next iteration
    }


# ── Phase 2 nodes ──────────────────────────────────────────────────────


def phase2_sub_agent_node(state: GlobalState) -> dict[str, Any]:
    """Phase 2 Sub-Agent node (mock).

    Simulates LSG extraction for a single client.
    """
    client_name: str = state.get("_client_name", "unknown")  # type: ignore[arg-type]
    iteration = state.get("phase2_iteration", 1)
    logger.info(
        "[phase2_sub_agent] client=%s phase=2 iter=%d (mock)",
        client_name,
        iteration,
    )

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
                    "transitions": [
                        {
                            "guard": "TRUE",
                            "actions": [],
                            "next_state": f"{wf_id}.done",
                            "evidence": None,
                        }
                    ],
                },
                {
                    "id": f"{wf_id}.done",
                    "label": "Done",
                    "category": "terminal",
                    "transitions": [],
                },
            ],
        })

    lsg = {
        "version": 1,
        "client": client_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "guards": list(state.get("guards", [])),
        "actions": list(state.get("actions", [])),
        "workflows": workflows,
    }

    return {"client_lsgs": {client_name: lsg}}


def phase2_main_agent_node(state: GlobalState) -> dict[str, Any]:
    """Phase 2 Main Agent node (mock).

    Compares client LSGs, classifies diffs, computes logic_diff_rate.
    """
    iteration = state.get("phase2_iteration", 1)
    client_lsgs = state.get("client_lsgs", {})
    logger.info(
        "[phase2_main_agent] phase=2 iter=%d clients=%d (mock)",
        iteration,
        len(client_lsgs),
    )

    diff_report: dict[str, Any] = {
        "a_class_diffs": [],
        "b_class_diffs": [],
        "logic_diff_rate": 0.0,
    }

    return {
        "diff_report": diff_report,
        "logic_diff_rate": 0.0,
        "a_class_feedback": [],
    }


# ────────────────────────────────────────────────────────────────────────
# Router / conditional-edge functions
# ────────────────────────────────────────────────────────────────────────


def route_after_preprocess(state: GlobalState) -> str:
    """After preprocessing, move to Phase 1 fan-out."""
    return "phase1_fanout"


def phase1_fanout(state: GlobalState) -> list[Send]:
    """Fan-out to per-client Phase 1 Sub-Agent nodes."""
    return [
        Send(
            "phase1_sub_agent",
            {**state, "_client_name": client},
        )
        for client in CLIENT_NAMES
    ]


def route_after_phase1_main(state: GlobalState) -> str:
    """Convergence / max-iter check for Phase 1."""
    iteration = state.get("phase1_iteration", 1)
    diff_rate = state.get("diff_rate", 1.0)

    if diff_rate < config.CONVERGENCE_THRESHOLD:
        logger.info(
            "[router_phase1] CONVERGED at iter=%d diff_rate=%.4f",
            iteration,
            diff_rate,
        )
        return "phase1_converged"

    if iteration >= config.MAX_ITER_PHASE1:
        logger.warning(
            "[router_phase1] MAX_ITER reached (%d) — force stopping Phase 1",
            config.MAX_ITER_PHASE1,
        )
        return "phase1_force_stop"

    return "phase1_next_iter"


def phase1_next_iter_node(state: GlobalState) -> dict[str, Any]:
    """Bump Phase 1 iteration counter."""
    return {"phase1_iteration": state.get("phase1_iteration", 1) + 1}


def phase1_converged_node(state: GlobalState) -> dict[str, Any]:
    """Mark Phase 1 as converged, transition to Phase 2."""
    return {
        "converged_phase1": True,
        "current_phase": 2,
        "phase2_iteration": 1,
    }


def phase1_force_stop_node(state: GlobalState) -> dict[str, Any]:
    """Force-stop Phase 1 after MAX_ITER."""
    return {
        "converged_phase1": False,
        "force_stopped": True,
        "current_phase": 2,
        "phase2_iteration": 1,
    }


def phase2_fanout(state: GlobalState) -> list[Send]:
    """Fan-out to per-client Phase 2 Sub-Agent nodes."""
    return [
        Send(
            "phase2_sub_agent",
            {**state, "_client_name": client},
        )
        for client in CLIENT_NAMES
    ]


def route_after_phase2_main(state: GlobalState) -> str:
    """Convergence / max-iter check for Phase 2."""
    iteration = state.get("phase2_iteration", 1)
    logic_diff_rate = state.get("logic_diff_rate", 1.0)

    if logic_diff_rate < config.CONVERGENCE_THRESHOLD:
        logger.info(
            "[router_phase2] CONVERGED at iter=%d logic_diff_rate=%.4f",
            iteration,
            logic_diff_rate,
        )
        return "phase2_converged"

    if iteration >= config.MAX_ITER_PHASE2:
        logger.warning(
            "[router_phase2] MAX_ITER reached (%d) — force stopping Phase 2",
            config.MAX_ITER_PHASE2,
        )
        return "phase2_force_stop"

    return "phase2_next_iter"


def phase2_next_iter_node(state: GlobalState) -> dict[str, Any]:
    """Bump Phase 2 iteration counter."""
    return {"phase2_iteration": state.get("phase2_iteration", 1) + 1}


def phase2_converged_node(state: GlobalState) -> dict[str, Any]:
    """Mark Phase 2 as converged."""
    return {
        "converged_phase2": True,
    }


def phase2_force_stop_node(state: GlobalState) -> dict[str, Any]:
    """Force-stop Phase 2 after MAX_ITER."""
    return {
        "converged_phase2": False,
        "force_stopped": True,
    }


# ────────────────────────────────────────────────────────────────────────
# Graph construction
# ────────────────────────────────────────────────────────────────────────


def build_graph() -> StateGraph:
    """Build and return the compiled LangGraph StateGraph."""

    graph = StateGraph(GlobalState)

    # ── Add nodes ───────────────────────────────────────────────────────
    graph.add_node("preprocess", preprocess_node)
    graph.add_node("phase1_sub_agent", phase1_sub_agent_node)
    graph.add_node("phase1_main_agent", phase1_main_agent_node)
    graph.add_node("phase1_next_iter", phase1_next_iter_node)
    graph.add_node("phase1_converged", phase1_converged_node)
    graph.add_node("phase1_force_stop", phase1_force_stop_node)
    graph.add_node("phase2_sub_agent", phase2_sub_agent_node)
    graph.add_node("phase2_main_agent", phase2_main_agent_node)
    graph.add_node("phase2_next_iter", phase2_next_iter_node)
    graph.add_node("phase2_converged", phase2_converged_node)
    graph.add_node("phase2_force_stop", phase2_force_stop_node)

    # ── Entry point ─────────────────────────────────────────────────────
    graph.set_entry_point("preprocess")

    # ── Preprocess → Phase 1 fan-out ────────────────────────────────────
    graph.add_conditional_edges(
        "preprocess",
        route_after_preprocess,
        {"phase1_fanout": "phase1_fanout"},
    )
    # Fan-out node (virtual) — dispatches to sub-agents via Send
    graph.add_node("phase1_fanout", lambda _state: {})
    graph.add_conditional_edges(
        "phase1_fanout",
        phase1_fanout,
        ["phase1_sub_agent"],
    )

    # ── Phase 1 sub → main ─────────────────────────────────────────────
    graph.add_edge("phase1_sub_agent", "phase1_main_agent")

    # ── Phase 1 main → router ──────────────────────────────────────────
    graph.add_conditional_edges(
        "phase1_main_agent",
        route_after_phase1_main,
        {
            "phase1_converged": "phase1_converged",
            "phase1_force_stop": "phase1_force_stop",
            "phase1_next_iter": "phase1_next_iter",
        },
    )

    # ── Phase 1 iteration loop ─────────────────────────────────────────
    graph.add_conditional_edges(
        "phase1_next_iter",
        lambda _s: "phase1_fanout",
        {"phase1_fanout": "phase1_fanout"},
    )

    # ── Phase 1 → Phase 2 transitions ─────────────────────────────────
    graph.add_node("phase2_fanout", lambda _state: {})
    graph.add_conditional_edges(
        "phase1_converged",
        lambda _s: "phase2_fanout",
        {"phase2_fanout": "phase2_fanout"},
    )
    graph.add_conditional_edges(
        "phase1_force_stop",
        lambda _s: "phase2_fanout",
        {"phase2_fanout": "phase2_fanout"},
    )

    # ── Phase 2 fan-out ────────────────────────────────────────────────
    graph.add_conditional_edges(
        "phase2_fanout",
        phase2_fanout,
        ["phase2_sub_agent"],
    )

    # ── Phase 2 sub → main ─────────────────────────────────────────────
    graph.add_edge("phase2_sub_agent", "phase2_main_agent")

    # ── Phase 2 main → router ──────────────────────────────────────────
    graph.add_conditional_edges(
        "phase2_main_agent",
        route_after_phase2_main,
        {
            "phase2_converged": "phase2_converged",
            "phase2_force_stop": "phase2_force_stop",
            "phase2_next_iter": "phase2_next_iter",
        },
    )

    # ── Phase 2 iteration loop ─────────────────────────────────────────
    graph.add_conditional_edges(
        "phase2_next_iter",
        lambda _s: "phase2_fanout",
        {"phase2_fanout": "phase2_fanout"},
    )

    # ── Terminal nodes ─────────────────────────────────────────────────
    graph.add_edge("phase2_converged", END)
    graph.add_edge("phase2_force_stop", END)

    return graph


def compile_graph():
    """Compile and return the runnable graph."""
    graph = build_graph()
    return graph.compile()
