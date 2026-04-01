"""EthAuditor — LangGraph graph definition.

Implements the full topology:
  preprocess → Phase 1 (fan-out sub-agents → main → router) → Phase 2 (fan-out → main → router) → END

Agent nodes delegate to the factory functions in ``agents/`` when available,
falling back to deterministic mock logic when no LLM is configured.
Conditional edges implement convergence checks and MAX_ITER guards.

Use :func:`configure_graph` to set the LLM and mock/live mode before
calling :func:`compile_graph`.
"""

from __future__ import annotations

import logging
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
# Graph-level configuration (set before compile_graph)
# ────────────────────────────────────────────────────────────────────────

_graph_config: dict[str, Any] = {
    "llm": None,
    "mock": True,
    "callbacks": None,
}


def configure_graph(
    *,
    llm: Any = None,
    mock: bool = True,
    callbacks: list[Any] | None = None,
) -> None:
    """Configure graph-wide settings before compilation.

    Parameters
    ----------
    llm:
        A LangChain-compatible LLM instance.  When provided (and *mock* is
        ``False``), agent nodes will invoke the LLM for real inference.
    mock:
        If ``True`` (default), all agent nodes use deterministic mock
        implementations — no LLM calls are made.
    callbacks:
        Optional list of LangChain callback handlers (e.g.
        :class:`AuditLogCallback`) to attach to every LLM invocation.
    """
    _graph_config["llm"] = llm
    _graph_config["mock"] = mock
    _graph_config["callbacks"] = callbacks


def get_graph_config() -> dict[str, Any]:
    """Return a *copy* of the current graph configuration."""
    return dict(_graph_config)


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
        "sparsity_hints": [],
    }


# ────────────────────────────────────────────────────────────────────────
# Node functions
# ────────────────────────────────────────────────────────────────────────


def preprocess_node(state: GlobalState) -> dict[str, Any]:
    """Offline preprocessing node.

    In **live** mode, delegates to
    :func:`tools.preprocessor.run_all_preprocessing` to run AST parsing,
    call-graph construction, vector-index and BM25-index builds.

    In **mock** mode (default), marks all clients as ready immediately.
    """
    logger.info("[preprocess_node] phase=0 — checking preprocessing status")
    if state.get("preprocess_done"):
        logger.info("[preprocess_node] Preprocessing already done — skipping.")
        return {}

    if not _graph_config["mock"]:
        from tools.preprocessor import run_all_preprocessing

        logger.info("[preprocess_node] Running real preprocessing pipeline …")
        statuses = run_all_preprocessing(force_rebuild=False)
    else:
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


def _get_llm() -> Any:
    """Return the configured LLM, or ``None`` if running in mock mode."""
    if _graph_config["mock"]:
        return None
    return _graph_config["llm"]


def _get_callbacks() -> list[Any] | None:
    """Return the configured callback handlers, or ``None``."""
    return _graph_config.get("callbacks") or None


def phase1_sub_agent_node(state: GlobalState) -> dict[str, Any]:
    """Phase 1 Sub-Agent node.

    Delegates to the agent factory from ``agents.phase1_sub_agent``.
    The client name is passed via state['_client_name'] by Send().
    """
    from agents.phase1_sub_agent import build_phase1_sub_agent

    client_name: str = state.get("_client_name", "unknown")  # type: ignore[arg-type]
    iteration = state.get("phase1_iteration", 1)
    logger.info(
        "[phase1_sub_agent] client=%s phase=1 iter=%d",
        client_name,
        iteration,
    )

    agent_fn = build_phase1_sub_agent(client_name, llm=_get_llm(), callbacks=_get_callbacks())
    return agent_fn(state)


def phase1_main_agent_node(state: GlobalState) -> dict[str, Any]:
    """Phase 1 Main Agent node.

    Delegates to the agent factory from ``agents.phase1_main_agent``.
    Merges discovery reports, computes diff_rate, bumps vocab_version.
    Saves a checkpoint after every iteration.
    """
    from agents.phase1_main_agent import build_phase1_main_agent
    from file_io.checkpoint import save_checkpoint

    iteration = state.get("phase1_iteration", 1)
    reports = state.get("discovery_reports", [])
    logger.info(
        "[phase1_main_agent] phase=1 iter=%d reports=%d",
        iteration,
        len(reports),
    )

    agent_fn = build_phase1_main_agent(llm=_get_llm(), callbacks=_get_callbacks())
    result = agent_fn(state)

    # Per-iteration checkpoint
    merged_state = {**state, **result}
    try:
        save_checkpoint(merged_state, phase=1, iteration=iteration)
    except Exception:
        logger.warning("[phase1_main_agent] checkpoint save failed", exc_info=True)

    return result


# ── Phase 2 nodes ──────────────────────────────────────────────────────


def phase2_sub_agent_node(state: GlobalState) -> dict[str, Any]:
    """Phase 2 Sub-Agent node.

    Delegates to the agent factory from ``agents.phase2_sub_agent``.
    Writes intermediate LSG YAML after each iteration.
    """
    from agents.phase2_sub_agent import build_phase2_sub_agent
    from file_io.writer import write_client_lsg

    client_name: str = state.get("_client_name", "unknown")  # type: ignore[arg-type]
    iteration = state.get("phase2_iteration", 1)
    logger.info(
        "[phase2_sub_agent] client=%s phase=2 iter=%d",
        client_name,
        iteration,
    )

    agent_fn = build_phase2_sub_agent(client_name, llm=_get_llm(), callbacks=_get_callbacks())
    result = agent_fn(state)

    # Write intermediate LSG YAML for this client+iteration
    lsg = result.get("client_lsgs", {}).get(client_name)
    if lsg is not None:
        try:
            lsg_with_iter = {**lsg, "_iteration": iteration}
            write_client_lsg(client_name, lsg_with_iter, final=False)
        except Exception:
            logger.warning("[phase2_sub_agent] intermediate LSG write failed", exc_info=True)

    return result


def phase2_main_agent_node(state: GlobalState) -> dict[str, Any]:
    """Phase 2 Main Agent node.

    Delegates to the agent factory from ``agents.phase2_main_agent``.
    Compares client LSGs, classifies diffs, computes logic_diff_rate.
    Saves a checkpoint after every iteration.
    """
    from agents.phase2_main_agent import build_phase2_main_agent
    from file_io.checkpoint import save_checkpoint

    iteration = state.get("phase2_iteration", 1)
    client_lsgs = state.get("client_lsgs", {})
    logger.info(
        "[phase2_main_agent] phase=2 iter=%d clients=%d",
        iteration,
        len(client_lsgs),
    )

    agent_fn = build_phase2_main_agent(llm=_get_llm(), callbacks=_get_callbacks())
    result = agent_fn(state)

    # Per-iteration checkpoint
    merged_state = {**state, **result}
    try:
        save_checkpoint(merged_state, phase=2, iteration=iteration)
    except Exception:
        logger.warning("[phase2_main_agent] checkpoint save failed", exc_info=True)

    return result


# ────────────────────────────────────────────────────────────────────────
# Router / conditional-edge functions
# ────────────────────────────────────────────────────────────────────────


def route_after_preprocess(state: GlobalState) -> str:
    """After preprocessing, move to Phase 1 fan-out — or skip to Phase 2
    if Phase 1 is already done (e.g. when resuming from a Phase 2 checkpoint).
    """
    if state.get("current_phase", 0) >= 2 or state.get("converged_phase1"):
        logger.info("[route_after_preprocess] Phase 1 already done — skipping to Phase 2")
        return "phase2_fanout"
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
        {
            "phase1_fanout": "phase1_fanout",
            "phase2_fanout": "phase2_fanout",
        },
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
