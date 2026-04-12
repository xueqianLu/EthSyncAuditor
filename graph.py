"""EthAuditor — LangGraph graph definition.

New topology (Phase 1 skipped — merged vocab loaded from disk):
  preprocess → load_vocab → workflow_scheduler
    → phase2_fanout (Send ×5 for current workflow)
    → phase2_sub_agent ×5 → phase2_main_agent → router
        → phase2_wf_converged / phase2_wf_force_stop
            → phase3_verify_fanout (Send × deviating clients)
            → phase3_verify_sub × N → phase3_verify_main
            → phase3_wf_verified → workflow_scheduler
        → phase2_next_iter → phase2_fanout
        → phase2_enter_b_class_focus → phase2_fanout
    → (all 7 workflows done) → final_aggregate → END

Phase 2 now iterates **one workflow at a time**.  The workflow_scheduler
picks the next unfinished workflow, resets per-workflow tracking state,
and fans out to all 5 client sub-agents.  Each sub-agent extracts only
the current workflow; the main agent compares only that workflow across
clients.  Once converged (or force-stopped), Phase 3 verification runs
for that workflow's B-class diffs before the scheduler moves to the next.

Use :func:`configure_graph` to set the LLM and mock/live mode before
calling :func:`compile_graph`.
"""

from __future__ import annotations

import logging
from typing import Any

import yaml
from langgraph.graph import END, StateGraph
from langgraph.types import Send

import config
from config import (
    CLIENT_NAMES,
    MERGED_LSG_DIR,
    MERGED_VOCAB_PATH,
    WORKFLOW_IDS,
)
from state import GlobalState

logger = logging.getLogger(__name__)

# Module-level convergence reason — set by router, read by convergence nodes.
_last_p2_convergence_reason: str = ""


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
    """Configure graph-wide settings before compilation."""
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
        "diff_rate": 0.0,
        "client_lsgs": {},
        "diff_report": {},
        "logic_diff_rate": 1.0,
        "converged_phase1": True,   # Phase 1 is skipped
        "converged_phase2": False,
        "force_stopped": False,
        "convergence_reason": "",
        "a_class_count": -1,
        "prev_a_class_count": -1,
        "iteration_history": [],
        "preprocess_done": False,
        "preprocess_status": {},
        "audit_log_paths": [],
        "discovery_reports": [],
        "a_class_feedback": [],
        "sparsity_hints": [],
        # B-class discovery phase
        "b_class_focus": False,
        "b_class_focus_iteration": 0,
        "prev_b_class_count": -1,
        # Per-workflow iteration
        "current_workflow": "",
        "completed_workflows": [],
        "workflow_diff_reports": {},
        "wf_iteration_history": [],
        # Phase 3: B-class verification
        "verified_b_diffs": [],
        "rejected_b_diffs": [],
        "reclassified_to_a": [],
        "verification_evidence": {},
    }


# ────────────────────────────────────────────────────────────────────────
# LLM / callback helpers
# ────────────────────────────────────────────────────────────────────────


def _get_llm() -> Any:
    """Return the configured LLM, or ``None`` if running in mock mode."""
    if _graph_config["mock"]:
        return None
    return _graph_config["llm"]


def _get_callbacks() -> list[Any] | None:
    return _graph_config.get("callbacks") or None


def _make_callbacks(phase: int, iteration: int, agent_type: str) -> list[Any] | None:
    """Create callbacks for a specific agent invocation."""
    if _graph_config["mock"]:
        return None

    from file_io.audit_logger import AuditLogCallback

    base = list(_graph_config.get("callbacks") or [])
    base = [cb for cb in base if not isinstance(cb, AuditLogCallback)]
    base.append(AuditLogCallback(phase=phase, iteration=iteration, agent_type=agent_type))
    return base or None


# ────────────────────────────────────────────────────────────────────────
# Node functions
# ────────────────────────────────────────────────────────────────────────


def preprocess_node(state: GlobalState) -> dict[str, Any]:
    """Offline preprocessing node."""
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
    }


# ── Load merged vocabulary & baseline LSGs ─────────────────────────────


def load_vocab_node(state: GlobalState) -> dict[str, Any]:
    """Load merged Phase 1 vocabulary and baseline client LSGs from disk.

    Reads:
      - docs/Global_LSG_Spec_Enriched.yaml  → guards, actions
      - docs/LSG_<client>_merged.yaml        → client_lsgs (×5)
    """
    logger.info("[load_vocab_node] Loading merged vocabulary from %s", MERGED_VOCAB_PATH)

    guards: list[dict] = []
    actions: list[dict] = []
    client_lsgs: dict[str, dict] = {}

    if MERGED_VOCAB_PATH.exists():
        with open(MERGED_VOCAB_PATH, encoding="utf-8") as f:
            spec = yaml.safe_load(f) or {}
        guards = spec.get("guards", [])
        actions = spec.get("actions", [])
        logger.info(
            "[load_vocab_node] Loaded %d guards, %d actions",
            len(guards), len(actions),
        )
    else:
        logger.warning("[load_vocab_node] Merged vocab not found: %s", MERGED_VOCAB_PATH)

    for client in CLIENT_NAMES:
        lsg_path = MERGED_LSG_DIR / f"LSG_{client}_merged.yaml"
        if lsg_path.exists():
            with open(lsg_path, encoding="utf-8") as f:
                client_lsgs[client] = yaml.safe_load(f) or {}
            n_wf = len(client_lsgs[client].get("workflows", []))
            logger.info("[load_vocab_node] Loaded %s LSG (%d workflows)", client, n_wf)
        else:
            logger.warning("[load_vocab_node] Merged LSG not found: %s", lsg_path)

    return {
        "guards": guards,
        "actions": actions,
        "vocab_version": 1,
        "converged_phase1": True,
        "current_phase": 2,
        "client_lsgs": client_lsgs,
    }


# ── Workflow scheduler ─────────────────────────────────────────────────


def workflow_scheduler_node(state: GlobalState) -> dict[str, Any]:
    """Pick the next unfinished workflow and reset per-workflow state."""
    completed = state.get("completed_workflows", [])

    for wf_id in WORKFLOW_IDS:
        if wf_id not in completed:
            logger.info(
                "[workflow_scheduler] Next workflow: %s  (completed: %s)",
                wf_id, completed,
            )
            return {
                "current_workflow": wf_id,
                "phase2_iteration": 1,
                "a_class_count": -1,
                "prev_a_class_count": -1,
                "wf_iteration_history": [],
                "b_class_focus": False,
                "b_class_focus_iteration": 0,
                "prev_b_class_count": -1,
                "diff_report": {},
                "a_class_feedback": [],
                "sparsity_hints": [],
            }

    # All workflows done
    logger.info("[workflow_scheduler] All %d workflows completed!", len(WORKFLOW_IDS))
    return {"current_workflow": ""}


def route_after_workflow_scheduler(state: GlobalState) -> str:
    """Route to phase2_fanout if there's a workflow to process, else to
    final_aggregate."""
    if state.get("current_workflow"):
        return "phase2_fanout"
    return "final_aggregate"


# ── Phase 2 nodes (per-workflow) ───────────────────────────────────────


def phase2_sub_agent_node(state: GlobalState) -> dict[str, Any]:
    """Phase 2 Sub-Agent node — extracts ONE workflow for one client."""
    from agents.phase2_sub_agent import build_phase2_sub_agent
    from file_io.writer import write_client_lsg

    client_name: str = state.get("_client_name", "unknown")  # type: ignore[arg-type]
    iteration = state.get("phase2_iteration", 1)
    current_wf = state.get("current_workflow", "unknown")
    logger.info(
        "[phase2_sub_agent] client=%s workflow=%s iter=%d",
        client_name, current_wf, iteration,
    )

    cbs = _make_callbacks(
        phase=2, iteration=iteration,
        agent_type=f"phase2_sub_{client_name}_{current_wf}",
    )
    agent_fn = build_phase2_sub_agent(client_name, llm=_get_llm(), callbacks=cbs)
    result = agent_fn(state)

    # Write intermediate LSG YAML for this client+iteration
    lsg = result.get("client_lsgs", {}).get(client_name)
    if lsg is not None:
        try:
            lsg_with_iter = {**lsg, "_iteration": iteration, "_workflow": current_wf}
            write_client_lsg(client_name, lsg_with_iter, final=False)
        except Exception:
            logger.warning("[phase2_sub_agent] intermediate LSG write failed", exc_info=True)

    return result


def phase2_main_agent_node(state: GlobalState) -> dict[str, Any]:
    """Phase 2 Main Agent node — compares ONE workflow across all clients."""
    from agents.phase2_main_agent import build_phase2_main_agent
    from file_io.checkpoint import save_checkpoint

    iteration = state.get("phase2_iteration", 1)
    current_wf = state.get("current_workflow", "unknown")
    client_lsgs = state.get("client_lsgs", {})
    logger.info(
        "[phase2_main_agent] workflow=%s iter=%d clients=%d",
        current_wf, iteration, len(client_lsgs),
    )

    agent_fn = build_phase2_main_agent(
        llm=_get_llm(),
        callbacks=_make_callbacks(
            phase=2, iteration=iteration,
            agent_type=f"phase2_main_{current_wf}",
        ),
    )
    result = agent_fn(state)

    # ── Record iteration metrics ────────────────────────────────────────
    diff_report = result.get("diff_report", {})
    a_count = result.get("a_class_count", len(diff_report.get("a_class_diffs", [])))
    b_count = len(diff_report.get("b_class_diffs", []))

    # Append to per-workflow iteration history (replace reducer)
    existing_wf_hist = list(state.get("wf_iteration_history", []))
    existing_wf_hist.append({
        "workflow": current_wf,
        "iteration": iteration,
        "a_class_count": a_count,
        "b_class_count": b_count,
        "logic_diff_rate": result.get("logic_diff_rate", 0.0),
    })
    result["wf_iteration_history"] = existing_wf_hist

    # Also append to global iteration_history (merge_lists reducer)
    result["iteration_history"] = [{
        "workflow": current_wf,
        "iteration": iteration,
        "a_class_count": a_count,
        "b_class_count": b_count,
        "logic_diff_rate": result.get("logic_diff_rate", 0.0),
    }]

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
    """Per-workflow convergence / max-iter check for Phase 2.

    Uses ``wf_iteration_history`` (per-workflow, resets each workflow)
    for convergence decisions.
    """
    global _last_p2_convergence_reason

    iteration = state.get("phase2_iteration", 1)
    current_wf = state.get("current_workflow", "?")
    a_class_count = state.get("a_class_count", -1)
    prev_a_class_count = state.get("prev_a_class_count", -1)
    history = state.get("wf_iteration_history", [])
    b_class_focus = state.get("b_class_focus", False)
    b_class_focus_iteration = state.get("b_class_focus_iteration", 0)

    # ════════════════════════════════════════════════════════════════════
    # Stage 2: B-class focus — check B-class stability
    # ════════════════════════════════════════════════════════════════════
    if b_class_focus:
        diff_report = state.get("diff_report", {})
        b_count = len(diff_report.get("b_class_diffs", []))
        prev_b_count = state.get("prev_b_class_count", -1)

        if prev_b_count >= 0:
            recent_b = [
                h.get("b_class_count", 0)
                for h in history[-config.B_CLASS_STABLE_WINDOW:]
            ] if len(history) >= config.B_CLASS_STABLE_WINDOW else []

            if (recent_b
                    and max(recent_b) - min(recent_b) <= config.B_CLASS_CHANGE_THRESHOLD):
                _last_p2_convergence_reason = (
                    f"[{current_wf}] B-class converged at iter {iteration} "
                    f"(b_class iter {b_class_focus_iteration}): stable at "
                    f"{b_count} (values={recent_b})."
                )
                logger.info("[router_phase2] %s B-CLASS CONVERGED: %s", current_wf, recent_b)
                return "phase2_wf_converged"

        if b_class_focus_iteration >= config.MAX_ITER_B_CLASS:
            _last_p2_convergence_reason = (
                f"[{current_wf}] MAX_ITER_B_CLASS ({config.MAX_ITER_B_CLASS}) "
                f"reached at iter {iteration}. B-class={b_count}."
            )
            logger.info("[router_phase2] %s B-CLASS MAX_ITER", current_wf)
            return "phase2_wf_converged"

        return "phase2_next_iter"

    # ════════════════════════════════════════════════════════════════════
    # Stage 1: Vocabulary alignment — check A-class convergence
    # ════════════════════════════════════════════════════════════════════

    if a_class_count == 0:
        _last_p2_convergence_reason = (
            f"[{current_wf}] Zero A-class diffs at iter {iteration}. "
            f"Entering B-class discovery."
        )
        logger.info("[router_phase2] %s A-class CONVERGED — entering B-class", current_wf)
        return "phase2_enter_b_class_focus"

    if prev_a_class_count >= 0 and a_class_count >= 0:
        delta = abs(a_class_count - prev_a_class_count)
        delta_rate = delta / max(prev_a_class_count, 1)
        if delta_rate < config.P2_A_CLASS_CONVERGENCE_THRESHOLD:
            _last_p2_convergence_reason = (
                f"[{current_wf}] A-class stabilized at iter {iteration}: "
                f"prev={prev_a_class_count}, cur={a_class_count}, "
                f"delta_rate={delta_rate:.4f}. Entering B-class."
            )
            logger.info("[router_phase2] %s A-class STABILIZED", current_wf)
            return "phase2_enter_b_class_focus"

    if len(history) >= config.OSCILLATION_WINDOW:
        recent = history[-config.OSCILLATION_WINDOW:]
        recent_a = [h.get("a_class_count", 0) for h in recent]
        if max(recent_a) - min(recent_a) <= config.OSCILLATION_BAND:
            _last_p2_convergence_reason = (
                f"[{current_wf}] A-class oscillation at iter {iteration}: "
                f"values={recent_a}. Entering B-class."
            )
            logger.info("[router_phase2] %s A-class OSCILLATING", current_wf)
            return "phase2_enter_b_class_focus"

    if iteration >= config.MAX_ITER_PHASE2:
        _last_p2_convergence_reason = (
            f"[{current_wf}] MAX_ITER_PHASE2 ({config.MAX_ITER_PHASE2}) "
            f"reached. A-class={a_class_count}."
        )
        logger.warning("[router_phase2] %s MAX_ITER — force stopping", current_wf)
        return "phase2_wf_force_stop"

    _last_p2_convergence_reason = ""
    return "phase2_next_iter"


def phase2_next_iter_node(state: GlobalState) -> dict[str, Any]:
    """Bump Phase 2 iteration counter."""
    diff_report = state.get("diff_report", {})
    b_count = len(diff_report.get("b_class_diffs", []))
    result: dict[str, Any] = {
        "phase2_iteration": state.get("phase2_iteration", 1) + 1,
        "prev_a_class_count": state.get("a_class_count", -1),
        "prev_b_class_count": b_count,
    }
    if state.get("b_class_focus"):
        result["b_class_focus_iteration"] = state.get("b_class_focus_iteration", 0) + 1
    return result


def phase2_enter_b_class_focus_node(state: GlobalState) -> dict[str, Any]:
    """Transition to B-class discovery mode for the current workflow."""
    diff_report = state.get("diff_report", {})
    b_count = len(diff_report.get("b_class_diffs", []))
    current_wf = state.get("current_workflow", "?")
    logger.info(
        "[phase2_enter_b_class_focus] workflow=%s B-class=%d",
        current_wf, b_count,
    )
    return {
        "b_class_focus": True,
        "b_class_focus_iteration": 1,
        "prev_b_class_count": b_count,
        "phase2_iteration": state.get("phase2_iteration", 1) + 1,
        "prev_a_class_count": state.get("a_class_count", -1),
    }


def phase2_wf_converged_node(state: GlobalState) -> dict[str, Any]:
    """Mark the current workflow as converged and save its diff report."""
    wf = state.get("current_workflow", "")
    diff_report = state.get("diff_report", {})
    logger.info("[phase2_wf_converged] workflow=%s — done", wf)
    return {
        "completed_workflows": [wf],
        "workflow_diff_reports": {wf: diff_report},
        "convergence_reason": _last_p2_convergence_reason,
    }


def phase2_wf_force_stop_node(state: GlobalState) -> dict[str, Any]:
    """Force-stop the current workflow and save its diff report."""
    wf = state.get("current_workflow", "")
    diff_report = state.get("diff_report", {})
    logger.warning("[phase2_wf_force_stop] workflow=%s — force stopped", wf)
    return {
        "completed_workflows": [wf],
        "workflow_diff_reports": {wf: diff_report},
        "convergence_reason": _last_p2_convergence_reason,
    }


# ── Phase 3: per-workflow B-class verification nodes ───────────────────


def phase3_verify_fanout(state: GlobalState) -> list[Send]:
    """Fan-out to per-client Phase 3 Verification Sub-Agent nodes.

    Sends only the clients involved in any B-class diff for the current
    workflow — i.e. deviating clients plus one majority client for each.
    """
    diff_report = state.get("diff_report", {})
    b_diffs = diff_report.get("b_class_diffs", [])

    if not b_diffs:
        logger.info("[phase3_verify_fanout] no B-class diffs — skipping verification")
        return [Send("phase3_verify_main", {**state, "_verify_b_diffs": []})]

    # Collect all clients involved in B-class diffs
    involved_clients: set[str] = set()
    for d in b_diffs:
        for c in d.get("deviating_clients", []):
            involved_clients.add(c)
        for c in d.get("involved_clients", []):
            involved_clients.add(c)

    # Ensure at least all CLIENT_NAMES are included (majority + deviating)
    if not involved_clients:
        involved_clients = set(CLIENT_NAMES)

    logger.info(
        "[phase3_verify_fanout] workflow=%s — verifying %d B-class diffs "
        "across %d clients",
        state.get("current_workflow", "?"), len(b_diffs),
        len(involved_clients),
    )

    return [
        Send(
            "phase3_verify_sub",
            {**state, "_client_name": client, "_verify_b_diffs": b_diffs},
        )
        for client in sorted(involved_clients)
    ]


def phase3_verify_sub_node(state: GlobalState) -> dict[str, Any]:
    """Phase 3 Verification Sub-Agent — searches one client's code."""
    from agents.phase3_verify_agent import build_phase3_verify_sub_agent

    client_name: str = state.get("_client_name", "unknown")  # type: ignore[arg-type]
    current_wf = state.get("current_workflow", "unknown")
    logger.info(
        "[phase3_verify_sub] client=%s workflow=%s",
        client_name, current_wf,
    )

    cbs = _make_callbacks(
        phase=3, iteration=0,
        agent_type=f"phase3_verify_sub_{client_name}_{current_wf}",
    )
    agent_fn = build_phase3_verify_sub_agent(
        client_name, llm=_get_llm(), callbacks=cbs,
    )
    return agent_fn(state)


def phase3_verify_main_node(state: GlobalState) -> dict[str, Any]:
    """Phase 3 Verification Main Agent — judges all B-class diffs."""
    from agents.phase3_verify_agent import build_phase3_verify_main_agent

    current_wf = state.get("current_workflow", "unknown")
    b_diffs = state.get("_verify_b_diffs", [])

    # If no diffs came through (empty fanout), extract from diff_report
    if not b_diffs:
        diff_report = state.get("diff_report", {})
        b_diffs = diff_report.get("b_class_diffs", [])

    logger.info(
        "[phase3_verify_main] workflow=%s — %d diffs to judge",
        current_wf, len(b_diffs),
    )

    cbs = _make_callbacks(
        phase=3, iteration=0,
        agent_type=f"phase3_verify_main_{current_wf}",
    )
    agent_fn = build_phase3_verify_main_agent(
        llm=_get_llm(), callbacks=cbs,
    )
    return agent_fn({**state, "_verify_b_diffs": b_diffs})


def phase3_wf_verified_node(state: GlobalState) -> dict[str, Any]:
    """Post-verification: update the workflow's diff report.

    Replaces the raw B-class diffs with only the verified ones and
    merges any reclassified diffs into A-class.
    """
    wf = state.get("current_workflow", "")
    wf_reports = state.get("workflow_diff_reports", {})
    report = dict(wf_reports.get(wf, {}))

    # Replace B-class diffs with only verified (CONFIRMED + DOWNGRADED)
    verified = state.get("verified_b_diffs", [])
    wf_verified = [d for d in verified if d.get("workflow_id") == wf]
    if wf_verified:
        report["b_class_diffs"] = wf_verified

    # Add reclassified diffs to A-class
    reclassified = state.get("reclassified_to_a", [])
    wf_reclassified = [d for d in reclassified if d.get("workflow_id") == wf]
    if wf_reclassified:
        existing_a = list(report.get("a_class_diffs", []))
        existing_a.extend(wf_reclassified)
        report["a_class_diffs"] = existing_a

    # Recompute logic_diff_rate
    n_a = len(report.get("a_class_diffs", []))
    n_b = len(report.get("b_class_diffs", []))
    report["logic_diff_rate"] = n_b / max(n_a + n_b, 1)

    logger.info(
        "[phase3_wf_verified] workflow=%s — %d verified B-class, "
        "%d reclassified to A-class",
        wf, len(wf_verified), len(wf_reclassified),
    )

    return {
        "workflow_diff_reports": {wf: report},
    }


def final_aggregate_node(state: GlobalState) -> dict[str, Any]:
    """Merge all per-workflow diff reports into a single final report."""
    wf_reports = state.get("workflow_diff_reports", {})

    all_a: list[dict] = []
    all_b: list[dict] = []
    total = 0

    for wf_id in WORKFLOW_IDS:
        report = wf_reports.get(wf_id, {})
        all_a.extend(report.get("a_class_diffs", []))
        all_b.extend(report.get("b_class_diffs", []))
        total += report.get("total_transitions", 0)

    logic_diff_rate = len(all_b) / max(total, 1)

    # Verification summary
    rejected = state.get("rejected_b_diffs", [])
    reclassified = state.get("reclassified_to_a", [])
    n_confirmed = sum(1 for d in all_b if d.get("verification_status") == "CONFIRMED")
    n_downgraded = sum(1 for d in all_b if d.get("verification_status") == "DOWNGRADED")

    logger.info(
        "[final_aggregate] %d workflows: A=%d B=%d (confirmed=%d downgraded=%d "
        "rejected=%d reclassified=%d) total=%d rate=%.4f",
        len(wf_reports), len(all_a), len(all_b),
        n_confirmed, n_downgraded, len(rejected), len(reclassified),
        total, logic_diff_rate,
    )

    return {
        "diff_report": {
            "a_class_diffs": all_a,
            "b_class_diffs": all_b,
            "logic_diff_rate": logic_diff_rate,
            "total_transitions": total,
        },
        "logic_diff_rate": logic_diff_rate,
        "converged_phase2": True,
    }


def _route_to_verify_or_scheduler(state: GlobalState) -> str:
    """Route to Phase 3 verification or directly to workflow_scheduler.

    Skips verification when ``config.VERIFY_ENABLED`` is ``False``.
    """
    if not config.VERIFY_ENABLED:
        return "workflow_scheduler"
    return "phase3_verify_fanout"


# ────────────────────────────────────────────────────────────────────────
# Graph construction
# ────────────────────────────────────────────────────────────────────────


def build_graph() -> StateGraph:
    """Build the per-workflow Phase 2 LangGraph StateGraph.

    Topology::

        START → preprocess → load_vocab → workflow_scheduler
            ─[has wf]→ phase2_fanout → phase2_sub_agent ×5 → phase2_main_agent
                ─[converged]→ phase2_wf_converged ──→ phase3_verify_fanout
                ─[force_stop]→ phase2_wf_force_stop → phase3_verify_fanout
                ─[next_iter]→ phase2_next_iter ─────→ phase2_fanout
                ─[b_class]→ phase2_enter_b_class ──→ phase2_fanout
            phase3_verify_fanout → phase3_verify_sub ×N → phase3_verify_main
                → phase3_wf_verified → workflow_scheduler
            ─[all done]→ final_aggregate → END
    """
    graph = StateGraph(GlobalState)

    # ── Add nodes ───────────────────────────────────────────────────────
    graph.add_node("preprocess", preprocess_node)
    graph.add_node("load_vocab", load_vocab_node)
    graph.add_node("workflow_scheduler", workflow_scheduler_node)
    graph.add_node("phase2_fanout", lambda _state: {})
    graph.add_node("phase2_sub_agent", phase2_sub_agent_node)
    graph.add_node("phase2_main_agent", phase2_main_agent_node)
    graph.add_node("phase2_next_iter", phase2_next_iter_node)
    graph.add_node("phase2_enter_b_class_focus", phase2_enter_b_class_focus_node)
    graph.add_node("phase2_wf_converged", phase2_wf_converged_node)
    graph.add_node("phase2_wf_force_stop", phase2_wf_force_stop_node)
    # Phase 3 verification nodes
    graph.add_node("phase3_verify_fanout", lambda _state: {})
    graph.add_node("phase3_verify_sub", phase3_verify_sub_node)
    graph.add_node("phase3_verify_main", phase3_verify_main_node)
    graph.add_node("phase3_wf_verified", phase3_wf_verified_node)
    graph.add_node("final_aggregate", final_aggregate_node)

    # ── Entry point ─────────────────────────────────────────────────────
    graph.set_entry_point("preprocess")

    # ── preprocess → load_vocab ─────────────────────────────────────────
    graph.add_edge("preprocess", "load_vocab")

    # ── load_vocab → workflow_scheduler ─────────────────────────────────
    graph.add_edge("load_vocab", "workflow_scheduler")

    # ── workflow_scheduler → fanout or final_aggregate ──────────────────
    graph.add_conditional_edges(
        "workflow_scheduler",
        route_after_workflow_scheduler,
        {
            "phase2_fanout": "phase2_fanout",
            "final_aggregate": "final_aggregate",
        },
    )

    # ── phase2_fanout → Send ×5 sub-agents ──────────────────────────────
    graph.add_conditional_edges(
        "phase2_fanout",
        phase2_fanout,
        ["phase2_sub_agent"],
    )

    # ── phase2_sub → phase2_main ────────────────────────────────────────
    graph.add_edge("phase2_sub_agent", "phase2_main_agent")

    # ── phase2_main → router ────────────────────────────────────────────
    graph.add_conditional_edges(
        "phase2_main_agent",
        route_after_phase2_main,
        {
            "phase2_wf_converged": "phase2_wf_converged",
            "phase2_wf_force_stop": "phase2_wf_force_stop",
            "phase2_next_iter": "phase2_next_iter",
            "phase2_enter_b_class_focus": "phase2_enter_b_class_focus",
        },
    )

    # ── phase2_next_iter → phase2_fanout (same workflow again) ──────────
    graph.add_conditional_edges(
        "phase2_next_iter",
        lambda _s: "phase2_fanout",
        {"phase2_fanout": "phase2_fanout"},
    )

    # ── phase2_enter_b_class_focus → phase2_fanout ──────────────────────
    graph.add_conditional_edges(
        "phase2_enter_b_class_focus",
        lambda _s: "phase2_fanout",
        {"phase2_fanout": "phase2_fanout"},
    )

    # ── wf_converged / wf_force_stop → Phase 3 verification ────────────
    graph.add_conditional_edges(
        "phase2_wf_converged",
        _route_to_verify_or_scheduler,
        {
            "phase3_verify_fanout": "phase3_verify_fanout",
            "workflow_scheduler": "workflow_scheduler",
        },
    )
    graph.add_conditional_edges(
        "phase2_wf_force_stop",
        _route_to_verify_or_scheduler,
        {
            "phase3_verify_fanout": "phase3_verify_fanout",
            "workflow_scheduler": "workflow_scheduler",
        },
    )

    # ── Phase 3 verification edges ──────────────────────────────────────
    graph.add_conditional_edges(
        "phase3_verify_fanout",
        phase3_verify_fanout,
        ["phase3_verify_sub"],
    )
    graph.add_edge("phase3_verify_sub", "phase3_verify_main")
    graph.add_edge("phase3_verify_main", "phase3_wf_verified")
    graph.add_edge("phase3_wf_verified", "workflow_scheduler")

    # ── Terminal ────────────────────────────────────────────────────────
    graph.add_edge("final_aggregate", END)

    return graph


def compile_graph():
    """Compile and return the runnable graph."""
    graph = build_graph()
    return graph.compile()
