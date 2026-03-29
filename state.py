"""State definitions for EthAuditor LangGraph workflow."""

from __future__ import annotations

import operator
from typing import Annotated, Literal, NotRequired, TypedDict

ClientName = Literal["prysm", "lighthouse", "grandine", "teku", "lodestar"]
WorkflowId = Literal[
    "initial_sync",
    "regular_sync",
    "checkpoint_sync",
    "attestation_generate",
    "block_generate",
    "aggregate",
    "execute_layer_relation",
]


class Evidence(TypedDict):
    file: str
    function: str
    lines: tuple[int, int]


class VocabEntry(TypedDict):
    name: str
    category: str
    description: str


class Transition(TypedDict):
    guard: str
    actions: list[str]
    next_state: str
    evidence: NotRequired[Evidence]


class LSGStateNode(TypedDict):
    id: str
    label: str
    category: str
    transitions: list[Transition]


class LSGWorkflow(TypedDict):
    id: WorkflowId
    name: str
    description: str
    mode: str
    initial_state: str
    states: list[LSGStateNode]


class LSGFile(TypedDict):
    version: int
    client: str
    generated_at: str
    guards: list[VocabEntry]
    actions: list[VocabEntry]
    workflows: list[LSGWorkflow]


class PreprocessArtifactStatus(TypedDict):
    symbols_json: bool
    callgraph_json: bool
    bm25_pkl: bool
    chroma_dir: bool


class PreprocessStatus(TypedDict):
    done: bool
    skipped: bool
    path: str
    per_client: dict[ClientName, PreprocessArtifactStatus]


class Phase1SubReport(TypedDict):
    client_name: ClientName
    guards: list[VocabEntry]
    actions: list[VocabEntry]


class Phase2SubReport(TypedDict):
    client_name: ClientName
    lsg: LSGFile


class DiffItem(TypedDict):
    diff_id: str
    diff_class: Literal["A", "B"]
    workflow_id: str
    state_id: str
    transition_guard: str
    involved_clients: list[str]
    summary: str


class GlobalState(TypedDict):
    active_client: str

    # Runtime control
    phase: int
    iteration_phase1: int
    iteration_phase2: int
    converged_phase1: bool
    converged_phase2: bool
    force_stopped: bool

    # Preprocess gate
    preprocess: PreprocessStatus

    # Unified vocabulary
    vocab_version: int
    guards_vocab: list[VocabEntry]
    actions_vocab: list[VocabEntry]

    # Current-iteration and final LSG artifacts
    lsg_current_iter: dict[ClientName, LSGFile]
    lsg_final: dict[ClientName, LSGFile]

    # Metrics
    diff_rate: float
    logic_diff_rate: float

    # Diff outputs
    diff_items_a: list[DiffItem]
    diff_items_b: list[DiffItem]

    # Fan-out/fan-in buffers (reducers)
    phase1_sub_reports: Annotated[list[Phase1SubReport], operator.add]
    phase2_sub_reports: Annotated[list[Phase2SubReport], operator.add]

    # Audit/checkpoint traces
    checkpoint_paths: list[str]
    audit_log_paths: list[str]
    warnings: list[str]


def _empty_preprocess_artifacts() -> PreprocessArtifactStatus:
    return {
        "symbols_json": False,
        "callgraph_json": False,
        "bm25_pkl": False,
        "chroma_dir": False,
    }


def make_initial_state() -> GlobalState:
    return {
        "active_client": "",
        "phase": 0,
        "iteration_phase1": 0,
        "iteration_phase2": 0,
        "converged_phase1": False,
        "converged_phase2": False,
        "force_stopped": False,
        "preprocess": {
            "done": False,
            "skipped": False,
            "path": "./output/preprocess",
            "per_client": {
                "prysm": _empty_preprocess_artifacts(),
                "lighthouse": _empty_preprocess_artifacts(),
                "grandine": _empty_preprocess_artifacts(),
                "teku": _empty_preprocess_artifacts(),
                "lodestar": _empty_preprocess_artifacts(),
            },
        },
        "vocab_version": 0,
        "guards_vocab": [],
        "actions_vocab": [],
        "lsg_current_iter": {},
        "lsg_final": {},
        "diff_rate": 1.0,
        "logic_diff_rate": 1.0,
        "diff_items_a": [],
        "diff_items_b": [],
        "phase1_sub_reports": [],
        "phase2_sub_reports": [],
        "checkpoint_paths": [],
        "audit_log_paths": [],
        "warnings": [],
    }
