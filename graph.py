"""LangGraph skeleton for EthAuditor Step 2.

This module intentionally uses mock agent nodes to validate topology,
state flow, fan-out scheduling, and convergence routing.
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from config import (
    CLIENT_NAMES,
    CONVERGENCE_THRESHOLD,
    MAX_ITER_PHASE1,
    MAX_ITER_PHASE2,
    PREPROCESS_PATH,
    WORKFLOW_IDS,
)
from agents.phase1_main_agent import run_phase1_main_agent
from agents.phase1_sub_agent import run_phase1_sub_agent
from agents.phase2_main_agent import run_phase2_main_agent
from agents.phase2_sub_agent import run_phase2_sub_agent
from agents.schemas import LSGFileModel, VocabDiscoveryReport
from eth_io import (
    save_checkpoint,
    write_diff_report,
    write_enriched_spec,
    write_final_lsgs,
    write_iteration_lsg,
)
from state import GlobalState, LSGFile, Phase1SubReport, Phase2SubReport, make_initial_state
from tools.preprocessor import run_preprocessing


def _is_mock_mode() -> bool:
    return os.getenv("ETHAUDITOR_RUN_MODE", "real").strip().lower() == "mock"


def _log(node: str, state: GlobalState) -> None:
    mode = "mock" if _is_mock_mode() else "real"
    phase = state.get("phase", "?")
    p1_iter = state.get("iteration_phase1", "?")
    p2_iter = state.get("iteration_phase2", "?")
    client = state.get("active_client", "")
    print(
        f"[{mode}:{node}] phase={phase} p1_iter={p1_iter} "
        f"p2_iter={p2_iter} client={client}"
    )


def _now_rfc3339() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _has_preprocess_artifacts(client_name: str) -> dict[str, bool]:
    base = Path(PREPROCESS_PATH)
    return {
        "symbols_json": (base / f"{client_name}_symbols.json").exists(),
        "callgraph_json": (base / f"{client_name}_callgraph.json").exists(),
        "bm25_pkl": (base / f"{client_name}_bm25.pkl").exists(),
        "chroma_dir": (base / f"{client_name}_chroma").exists(),
    }


def preprocess_node(state: GlobalState) -> dict[str, Any]:
    _log("preprocess_node", state)

    if not _is_mock_mode():
        per_client: dict[str, dict[str, bool]] = {}
        warnings: list[str] = []

        for client in CLIENT_NAMES:
            result = run_preprocessing(client_name=client, force_rebuild=False)
            skipped = bool(result.get("skipped", False))
            if skipped:
                warnings.append(f"{client}: preprocess cache hit")
            else:
                warnings.append(f"{client}: preprocess rebuilt")

            per_client[client] = {
                "symbols_json": bool(Path(str(result["symbols"])).exists()),
                "callgraph_json": bool(Path(str(result["callgraph"])).exists()),
                "bm25_pkl": bool(Path(str(result["bm25"])).exists()),
                "chroma_dir": bool(Path(str(result["chroma"])).exists()),
            }

        return {
            "phase": 1,
            "preprocess": {
                "done": True,
                "skipped": all("cache hit" in w for w in warnings),
                "path": PREPROCESS_PATH,
                "per_client": per_client,
            },
            "warnings": warnings,
        }

    per_client = {client: _has_preprocess_artifacts(client) for client in CLIENT_NAMES}
    all_ready = all(all(v.values()) for v in per_client.values())

    warnings = []
    if all_ready:
        warnings.append("preprocess artifacts found, skip mock build")
    else:
        warnings.append(
            "preprocess artifacts missing for some clients, using mock preprocess completion"
        )

    return {
        "phase": 1,
        "preprocess": {
            "done": True,
            "skipped": all_ready,
            "path": PREPROCESS_PATH,
            "per_client": per_client,
        },
        "warnings": warnings,
    }


def phase1_dispatch_node(state: GlobalState) -> dict[str, Any]:
    _log("phase1_dispatch_node", state)
    return {"phase": 1}


def phase1_fanout(state: GlobalState) -> list[Send]:
    _log("phase1_fanout", state)
    return [Send("phase1_subagent", {"active_client": client}) for client in CLIENT_NAMES]


def phase1_subagent_node(state: GlobalState) -> dict[str, Any]:
    _log("phase1_subagent_node", state)

    client_name = state.get("active_client", "unknown")

    if not _is_mock_mode():
        report = run_phase1_sub_agent(
            client_name=client_name,
            current_vocab={
                "guards": state.get("guards_vocab", []),
                "actions": state.get("actions_vocab", []),
            },
            iteration=state.get("iteration_phase1", 0) + 1,
        )

        real_report: Phase1SubReport = {
            "client_name": client_name,  # type: ignore[typeddict-item]
            "guards": [
                {
                    "name": g.name,
                    "category": g.category,
                    "description": g.description,
                }
                for g in report.discovered_guards
            ],
            "actions": [
                {
                    "name": a.name,
                    "category": a.category,
                    "description": a.description,
                }
                for a in report.discovered_actions
            ],
        }
        return {"phase1_sub_reports": [real_report]}

    report: Phase1SubReport = {
        "client_name": client_name,  # type: ignore[typeddict-item]
        "guards": [
            {
                "name": f"MockGuard_{client_name}",
                "category": "mock",
                "description": f"Mock discovered guard from {client_name}",
            }
        ],
        "actions": [
            {
                "name": f"MockAction_{client_name}",
                "category": "mock",
                "description": f"Mock discovered action from {client_name}",
            }
        ],
    }
    return {"phase1_sub_reports": [report]}


def phase1_collect_node(state: GlobalState) -> dict[str, Any]:
    _log("phase1_collect_node", state)
    return {}


def phase1_main_node(state: GlobalState) -> dict[str, Any]:
    _log("phase1_main_node", state)

    next_iter = state["iteration_phase1"] + 1

    latest_reports = state["phase1_sub_reports"][-len(CLIENT_NAMES) :]

    if not _is_mock_mode():
        reports: list[VocabDiscoveryReport] = []
        for report in latest_reports:
            reports.append(
                VocabDiscoveryReport.model_validate(
                    {
                        "client_name": report["client_name"],
                        "discovered_guards": report.get("guards", []),
                        "discovered_actions": report.get("actions", []),
                        "notes": "",
                    }
                )
            )

        enriched = run_phase1_main_agent(
            sub_reports=reports,
            current_vocab={
                "guards": state.get("guards_vocab", []),
                "actions": state.get("actions_vocab", []),
            },
            vocab_version=state.get("vocab_version", 0),
            iteration=next_iter,
        )

        new_guards = [
            {"name": g.name, "category": g.category, "description": g.description}
            for g in enriched.guards
        ]
        new_actions = [
            {"name": a.name, "category": a.category, "description": a.description}
            for a in enriched.actions
        ]

        old_set = {
            (v.get("name", ""), v.get("category", ""))
            for v in state.get("guards_vocab", []) + state.get("actions_vocab", [])
        }
        new_set = {
            (v.get("name", ""), v.get("category", ""))
            for v in new_guards + new_actions
        }
        delta = len(new_set - old_set)
        diff_rate = delta / max(len(new_set), 1)

        checkpoint_path = save_checkpoint(state, phase=1, iteration=next_iter)

        return {
            "iteration_phase1": next_iter,
            "vocab_version": enriched.vocab_version,
            "guards_vocab": new_guards,
            "actions_vocab": new_actions,
            "diff_rate": diff_rate,
            "checkpoint_paths": state["checkpoint_paths"] + [str(checkpoint_path)],
        }

    merged_guards: list[dict[str, str]] = []
    merged_actions: list[dict[str, str]] = []

    for report in latest_reports:
        merged_guards.extend(report["guards"])
        merged_actions.extend(report["actions"])

    # Mock convergence curve: 0.2, 0.1, 0.066..., 0.05, 0.04 ...
    diff_rate = 0.2 / next_iter

    print(
        f"[mock:phase1_main_node] iter={next_iter} reports={len(latest_reports)} diff_rate={diff_rate:.4f}"
    )

    checkpoint_path = save_checkpoint(state, phase=1, iteration=next_iter)

    return {
        "iteration_phase1": next_iter,
        "vocab_version": state["vocab_version"] + 1,
        "guards_vocab": merged_guards,
        "actions_vocab": merged_actions,
        "diff_rate": diff_rate,
        "checkpoint_paths": state["checkpoint_paths"] + [str(checkpoint_path)],
    }


def router_phase1(state: GlobalState) -> str:
    if state["diff_rate"] < CONVERGENCE_THRESHOLD:
        print("[router_phase1] converged -> phase1_finalize")
        return "phase1_finalize"

    if state["iteration_phase1"] >= MAX_ITER_PHASE1:
        print("[router_phase1] max iteration reached -> phase1_finalize (force stop)")
        return "phase1_finalize_force"

    print("[router_phase1] continue -> phase1_dispatch")
    return "phase1_continue"


def phase1_finalize_node(state: GlobalState) -> dict[str, Any]:
    _log("phase1_finalize_node", state)

    converged = state["diff_rate"] < CONVERGENCE_THRESHOLD
    force_stopped = state["iteration_phase1"] >= MAX_ITER_PHASE1 and not converged

    enriched_path = write_enriched_spec(
        {
            "guards": state["guards_vocab"],
            "actions": state["actions_vocab"],
        }
    )

    return {
        "converged_phase1": converged,
        "force_stopped": state["force_stopped"] or force_stopped,
        "phase": 2,
        "checkpoint_paths": state["checkpoint_paths"] + [str(enriched_path)],
    }


def phase2_dispatch_node(state: GlobalState) -> dict[str, Any]:
    _log("phase2_dispatch_node", state)
    return {"phase": 2}


def phase2_fanout(state: GlobalState) -> list[Send]:
    _log("phase2_fanout", state)
    return [Send("phase2_subagent", {"active_client": client}) for client in CLIENT_NAMES]


def _build_mock_lsg(client_name: str) -> LSGFile:
    workflow_id = WORKFLOW_IDS[0]
    return {
        "version": 1,
        "client": client_name,
        "generated_at": _now_rfc3339(),
        "guards": [
            {
                "name": "ModeIsInitialSync",
                "category": "mode",
                "description": "Mock reused guard",
            }
        ],
        "actions": [
            {
                "name": "SendRangeRequest",
                "category": "network",
                "description": "Mock reused action",
            }
        ],
        "workflows": [
            {
                "id": workflow_id,  # type: ignore[typeddict-item]
                "name": "Initial Sync (Mock)",
                "description": "Mock workflow for topology validation",
                "mode": "sync",
                "initial_state": "initial.start",
                "states": [
                    {
                        "id": "initial.start",
                        "label": "Start",
                        "category": "init",
                        "transitions": [
                            {
                                "guard": "TRUE",
                                "actions": ["SendRangeRequest"],
                                "next_state": "initial.done",
                                "evidence": {
                                    "file": f"code/{client_name}/mock/file",
                                    "function": "mock_function",
                                    "lines": (1, 10),
                                },
                            }
                        ],
                    },
                    {
                        "id": "initial.done",
                        "label": "Done",
                        "category": "terminal",
                        "transitions": [],
                    },
                ],
            }
        ],
    }


def phase2_subagent_node(state: GlobalState) -> dict[str, Any]:
    _log("phase2_subagent_node", state)

    client_name = state.get("active_client", "unknown")

    if not _is_mock_mode():
        lsg = run_phase2_sub_agent(
            client_name=client_name,
            enriched_spec={
                "guards": state.get("guards_vocab", []),
                "actions": state.get("actions_vocab", []),
            },
            iteration=state.get("iteration_phase2", 0) + 1,
        )

        report: Phase2SubReport = {
            "client_name": client_name,  # type: ignore[typeddict-item]
            "lsg": lsg.model_dump(),
        }
        return {"phase2_sub_reports": [report]}

    report: Phase2SubReport = {
        "client_name": client_name,  # type: ignore[typeddict-item]
        "lsg": _build_mock_lsg(client_name),
    }
    return {"phase2_sub_reports": [report]}


def phase2_collect_node(state: GlobalState) -> dict[str, Any]:
    _log("phase2_collect_node", state)
    return {}


def phase2_main_node(state: GlobalState) -> dict[str, Any]:
    _log("phase2_main_node", state)

    next_iter = state["iteration_phase2"] + 1
    latest_reports = state["phase2_sub_reports"][-len(CLIENT_NAMES) :]

    lsg_map = {r["client_name"]: r["lsg"] for r in latest_reports}

    if not _is_mock_mode():
        lsg_models = [LSGFileModel.model_validate(v) for v in lsg_map.values()]
        diff_report = run_phase2_main_agent(
            client_lsgs=lsg_models,
            iteration=next_iter,
        )

        logic_diff_rate = diff_report.logic_diff_rate

        iteration_paths: list[str] = []
        for client, lsg in lsg_map.items():
            iter_path = write_iteration_lsg(client_name=client, iteration=next_iter, lsg=lsg)
            iteration_paths.append(str(iter_path))

        checkpoint_path = save_checkpoint(state, phase=2, iteration=next_iter)

        return {
            "iteration_phase2": next_iter,
            "lsg_current_iter": lsg_map,
            "logic_diff_rate": logic_diff_rate,
            "diff_items_a": [item.model_dump() for item in diff_report.class_a],
            "diff_items_b": [item.model_dump() for item in diff_report.class_b],
            "checkpoint_paths": state["checkpoint_paths"]
            + [str(checkpoint_path)]
            + iteration_paths,
        }

    # Mock logic diff curve: 0.2, 0.1, 0.066..., 0.05, 0.04 ...
    logic_diff_rate = 0.2 / next_iter

    print(
        f"[mock:phase2_main_node] iter={next_iter} reports={len(latest_reports)} "
        f"logic_diff_rate={logic_diff_rate:.4f}"
    )

    iteration_paths: list[str] = []
    for client, lsg in lsg_map.items():
        iter_path = write_iteration_lsg(client_name=client, iteration=next_iter, lsg=lsg)
        iteration_paths.append(str(iter_path))

    checkpoint_path = save_checkpoint(state, phase=2, iteration=next_iter)

    return {
        "iteration_phase2": next_iter,
        "lsg_current_iter": lsg_map,
        "logic_diff_rate": logic_diff_rate,
        "diff_items_a": [
            {
                "diff_id": f"A-{next_iter}",
                "diff_class": "A",
                "workflow_id": "initial_sync",
                "state_id": "initial.start",
                "transition_guard": "TRUE",
                "involved_clients": CLIENT_NAMES,
                "summary": "mock implementation-level difference",
            }
        ],
        "diff_items_b": [],
        "checkpoint_paths": state["checkpoint_paths"] + [str(checkpoint_path)] + iteration_paths,
    }


def router_phase2(state: GlobalState) -> str:
    if state["logic_diff_rate"] < CONVERGENCE_THRESHOLD:
        print("[router_phase2] converged -> phase2_finalize")
        return "phase2_finalize"

    if state["iteration_phase2"] >= MAX_ITER_PHASE2:
        print("[router_phase2] max iteration reached -> phase2_finalize (force stop)")
        return "phase2_finalize_force"

    print("[router_phase2] continue -> phase2_dispatch")
    return "phase2_continue"


def phase2_finalize_node(state: GlobalState) -> dict[str, Any]:
    _log("phase2_finalize_node", state)

    converged = state["logic_diff_rate"] < CONVERGENCE_THRESHOLD
    force_stopped = state["iteration_phase2"] >= MAX_ITER_PHASE2 and not converged

    final_paths = write_final_lsgs(state["lsg_current_iter"])
    report_path = write_diff_report(
        diff_items_b=state["diff_items_b"],
        summary={
            "compared_items": len(state["diff_items_a"]) + len(state["diff_items_b"]),
            "a_diff_count": len(state["diff_items_a"]),
            "b_diff_count": len(state["diff_items_b"]),
            "logic_diff_rate": state["logic_diff_rate"],
        },
    )

    return {
        "converged_phase2": converged,
        "force_stopped": state["force_stopped"] or force_stopped,
        "lsg_final": state["lsg_current_iter"],
        "checkpoint_paths": state["checkpoint_paths"]
        + [str(report_path)]
        + [str(p) for p in final_paths],
    }


def build_graph():
    builder = StateGraph(GlobalState)

    builder.add_node("preprocess", preprocess_node)

    builder.add_node("phase1_dispatch", phase1_dispatch_node)
    builder.add_node("phase1_subagent", phase1_subagent_node)
    builder.add_node("phase1_collect", phase1_collect_node)
    builder.add_node("phase1_main", phase1_main_node)
    builder.add_node("phase1_finalize", phase1_finalize_node)

    builder.add_node("phase2_dispatch", phase2_dispatch_node)
    builder.add_node("phase2_subagent", phase2_subagent_node)
    builder.add_node("phase2_collect", phase2_collect_node)
    builder.add_node("phase2_main", phase2_main_node)
    builder.add_node("phase2_finalize", phase2_finalize_node)

    builder.add_edge(START, "preprocess")
    builder.add_edge("preprocess", "phase1_dispatch")

    builder.add_conditional_edges("phase1_dispatch", phase1_fanout, ["phase1_subagent"])
    builder.add_edge("phase1_subagent", "phase1_collect")
    builder.add_edge("phase1_collect", "phase1_main")
    builder.add_conditional_edges(
        "phase1_main",
        router_phase1,
        {
            "phase1_continue": "phase1_dispatch",
            "phase1_finalize": "phase1_finalize",
            "phase1_finalize_force": "phase1_finalize",
        },
    )

    builder.add_edge("phase1_finalize", "phase2_dispatch")

    builder.add_conditional_edges("phase2_dispatch", phase2_fanout, ["phase2_subagent"])
    builder.add_edge("phase2_subagent", "phase2_collect")
    builder.add_edge("phase2_collect", "phase2_main")
    builder.add_conditional_edges(
        "phase2_main",
        router_phase2,
        {
            "phase2_continue": "phase2_dispatch",
            "phase2_finalize": "phase2_finalize",
            "phase2_finalize_force": "phase2_finalize",
        },
    )

    builder.add_edge("phase2_finalize", END)
    return builder.compile()


def run_graph() -> GlobalState:
    graph = build_graph()
    final_state = graph.invoke(make_initial_state())
    print(
        "[run_graph] done",
        {
            "mode": "mock" if _is_mock_mode() else "real",
            "phase1_iter": final_state["iteration_phase1"],
            "phase2_iter": final_state["iteration_phase2"],
            "diff_rate": final_state["diff_rate"],
            "logic_diff_rate": final_state["logic_diff_rate"],
            "force_stopped": final_state["force_stopped"],
        },
    )
    return final_state


if __name__ == "__main__":
    run_graph()
