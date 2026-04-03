"""EthAuditor — global state definitions.

Every field is annotated with which graph nodes *write* and *read* it so
that the data‑flow contract is explicit.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, Field

# ────────────────────────────────────────────────────────────────────────
# Sub‑structures
# ────────────────────────────────────────────────────────────────────────


class VocabEntry(BaseModel):
    """A single Guard or Action vocabulary entry."""

    name: str
    category: str
    description: str
    evidence_file: str | None = None
    evidence_function: str | None = None
    evidence_lines: list[int] | None = None


class Evidence(BaseModel):
    """Source‑code evidence attached to a transition."""

    file: str
    function: str
    lines: list[int] = Field(default_factory=list)  # [start, end], 1‑based


class Transition(BaseModel):
    """A single state‑machine transition."""

    guard: str  # GuardName or "TRUE"
    actions: list[str] = Field(default_factory=list)
    next_state: str
    evidence: Evidence | None = None


class LSGState(BaseModel):
    """A single state node inside a workflow."""

    id: str
    label: str
    category: str
    transitions: list[Transition] = Field(default_factory=list)


class LSGWorkflow(BaseModel):
    """One of the 7 core workflows in a client's LSG."""

    id: str
    name: str
    description: str = ""
    mode: str = ""
    initial_state: str = ""
    states: list[LSGState] = Field(default_factory=list)


class LSGFile(BaseModel):
    """Complete LSG output for one client (maps 1‑to‑1 to the YAML file)."""

    version: int = 1
    client: str = ""
    generated_at: str = ""
    guards: list[VocabEntry] = Field(default_factory=list)
    actions: list[VocabEntry] = Field(default_factory=list)
    workflows: list[LSGWorkflow] = Field(default_factory=list)


class DiffItem(BaseModel):
    """A single diff entry between two or more clients."""

    workflow_id: str
    state_id: str
    transition_guard: str
    diff_type: str  # "A" or "B"
    description: str
    severity: str = ""  # "CRITICAL" / "MAJOR" / "MINOR" (B-class only)
    involved_clients: list[str] = Field(default_factory=list)
    evidence: dict[str, Evidence | None] = Field(default_factory=dict)


class DiffReport(BaseModel):
    """Aggregated diff report produced by Phase 2 Main Agent."""

    a_class_diffs: list[DiffItem] = Field(default_factory=list)
    b_class_diffs: list[DiffItem] = Field(default_factory=list)
    logic_diff_rate: float = 1.0
    total_transitions: int = 0  # total comparison items for similarity computation


class VocabDiscoveryReport(BaseModel):
    """Output of a Phase 1 Sub‑Agent: newly discovered vocab candidates."""

    client_name: str
    new_guards: list[VocabEntry] = Field(default_factory=list)
    new_actions: list[VocabEntry] = Field(default_factory=list)


class EnrichedSpec(BaseModel):
    """The enriched global specification (output of Phase 1)."""

    version: int = 1
    guards: list[VocabEntry] = Field(default_factory=list)
    actions: list[VocabEntry] = Field(default_factory=list)


class PreprocessStatus(BaseModel):
    """Tracks which preprocessing artifacts are ready for a client."""

    symbols_ready: bool = False
    callgraph_ready: bool = False
    vector_index_ready: bool = False
    bm25_index_ready: bool = False

    @property
    def all_ready(self) -> bool:
        return (
            self.symbols_ready
            and self.callgraph_ready
            and self.vector_index_ready
            and self.bm25_index_ready
        )


# ────────────────────────────────────────────────────────────────────────
# Reducer helpers — used by LangGraph `Annotated` fields
# ────────────────────────────────────────────────────────────────────────


def _replace(existing: Any, new: Any) -> Any:  # noqa: ANN401
    """Simple last‑write‑wins reducer."""
    return new


def _merge_lists(existing: list, new: list) -> list:
    """Append‑only list reducer (e.g. for audit log paths)."""
    if existing is None:
        existing = []
    if new is None:
        new = []
    return existing + new


def _merge_vocab(existing: list, new: list) -> list:
    """Deduplicated list merge for vocabulary entries (guards / actions).

    Each entry is a dict with a ``"name"`` key.  When *new* contains an
    entry whose name already exists in *existing*, the newer version
    **replaces** the old one.  This prevents the unbounded growth that
    ``_merge_lists`` causes when Phase 1 Main returns incremental entries
    across many iterations.
    """
    if existing is None:
        existing = []
    if new is None:
        new = []
    seen: dict[str, int] = {}          # name → index in result
    result: list = []
    for entry in existing:
        name = entry.get("name", "") if isinstance(entry, dict) else str(entry)
        if name and name not in seen:
            seen[name] = len(result)
            result.append(entry)
        elif not name:
            result.append(entry)       # keep entries without a name key
    for entry in new:
        name = entry.get("name", "") if isinstance(entry, dict) else str(entry)
        if name and name in seen:
            result[seen[name]] = entry  # replace with newer version
        else:
            if name:
                seen[name] = len(result)
            result.append(entry)
    return result


def _collect_then_clear(existing: list | None, new: list | None) -> list:
    """Fan-out / fan-in reducer.

    * ``new`` is a non-empty list → **append** to *existing*.
    * ``new`` is ``[]`` (empty list) → **clear** and return ``[]``.
    * ``new`` is ``None`` → keep *existing* unchanged.

    This enables the pattern where parallel sub-agents accumulate results
    into a shared list, and the main agent sends ``[]`` to reset it before
    the next iteration.
    """
    if new is None:
        return existing if existing is not None else []
    if existing is None:
        existing = []
    if len(new) == 0:
        return []
    return existing + new


def _merge_dicts(existing: dict, new: dict) -> dict:
    """Shallow‑merge dict reducer (e.g. client LSG maps)."""
    if existing is None:
        existing = {}
    if new is None:
        new = {}
    merged = {**existing, **new}
    return merged


# ────────────────────────────────────────────────────────────────────────
# GlobalState — the single source of truth flowing through the graph
# ────────────────────────────────────────────────────────────────────────


class GlobalState(TypedDict, total=False):
    """
    LangGraph‑compatible state dictionary.

    Write/Read contract (W = writer node, R = reader node):

    ┌──────────────────────────┬───────────────────────────┬───────────────────────────┐
    │ Field                    │ Writers                   │ Readers                   │
    ├──────────────────────────┼───────────────────────────┼───────────────────────────┤
    │ current_phase            │ router_*                  │ all nodes                 │
    │ phase1_iteration         │ router_phase1             │ phase1 nodes, router      │
    │ phase2_iteration         │ router_phase2             │ phase2 nodes, router      │
    │ guards                   │ phase1_main               │ phase1/2 sub, writer      │
    │ actions                  │ phase1_main               │ phase1/2 sub, writer      │
    │ vocab_version            │ phase1_main               │ phase1 sub, router        │
    │ diff_rate                │ phase1_main               │ router_phase1             │
    │ client_lsgs              │ phase2_sub (via merge)    │ phase2_main, writer       │
    │ diff_report              │ phase2_main               │ router_phase2, writer     │
    │ logic_diff_rate          │ phase2_main               │ router_phase2             │
    │ converged_phase1         │ router_phase1             │ graph routing             │
    │ converged_phase2         │ router_phase2             │ graph routing             │
    │ force_stopped            │ router_*                  │ exit nodes                │
    │ preprocess_done          │ preprocess_node           │ router                    │
    │ preprocess_status        │ preprocess_node           │ preprocess_node           │
    │ audit_log_paths          │ audit_logger callback     │ writer                    │
    │ discovery_reports        │ phase1_sub (via merge)    │ phase1_main               │
    │ a_class_feedback         │ phase2_main               │ phase2_sub                │
    └──────────────────────────┴───────────────────────────┴───────────────────────────┘
    """

    # ── Phase tracking ──────────────────────────────────────────────────
    current_phase: int                         # 0 = preprocess, 1 = phase1, 2 = phase2
    phase1_iteration: int
    phase2_iteration: int

    # ── Vocabulary (Phase 1 output) ─────────────────────────────────────
    guards: Annotated[list[dict], _merge_vocab]
    actions: Annotated[list[dict], _merge_vocab]
    vocab_version: int
    diff_rate: float

    # ── Per‑client LSGs (Phase 2) ───────────────────────────────────────
    client_lsgs: Annotated[dict[str, dict], _merge_dicts]

    # ── Diff report (Phase 2 output) ────────────────────────────────────
    diff_report: dict
    logic_diff_rate: float

    # ── Convergence / termination flags ─────────────────────────────────
    converged_phase1: bool
    converged_phase2: bool
    force_stopped: bool
    convergence_reason: str  # human-readable reason for convergence / force-stop

    # ── Phase 2 convergence tracking ─────────────────────────────────────
    a_class_count: Annotated[int, _replace]          # current iteration A-class diff count
    prev_a_class_count: Annotated[int, _replace]     # previous iteration A-class diff count
    iteration_history: Annotated[list[dict], _merge_lists]  # per-iter metrics

    # ── Preprocessing ───────────────────────────────────────────────────
    preprocess_done: bool
    preprocess_status: Annotated[dict[str, dict], _merge_dicts]

    # ── Audit logs ──────────────────────────────────────────────────────
    audit_log_paths: Annotated[list[str], _merge_lists]

    # ── Inter‑agent communication ───────────────────────────────────────
    discovery_reports: Annotated[list[dict], _merge_lists]
    a_class_feedback: Annotated[list[dict], _replace]
    sparsity_hints: Annotated[list[dict], _replace]
