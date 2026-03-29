"""Pydantic schemas for structured agent outputs."""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Literal

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


class Evidence(BaseModel):
    file: str
    function: str
    lines: tuple[int, int]


class VocabEntry(BaseModel):
    name: str
    category: str
    description: str
    aliases: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)


class VocabDiscoveryReport(BaseModel):
    client_name: ClientName
    discovered_guards: list[VocabEntry] = Field(default_factory=list)
    discovered_actions: list[VocabEntry] = Field(default_factory=list)
    notes: str = ""


class EnrichedSpec(BaseModel):
    version: int = 1
    vocab_version: int
    guards: list[VocabEntry]
    actions: list[VocabEntry]
    merge_summary: str = ""


class LSGTransition(BaseModel):
    guard: str
    actions: list[str]
    next_state: str
    evidence: Evidence


class LSGStateNode(BaseModel):
    id: str
    label: str
    category: str
    transitions: list[LSGTransition] = Field(default_factory=list)


class LSGWorkflow(BaseModel):
    id: WorkflowId
    name: str
    description: str
    mode: str
    initial_state: str
    states: list[LSGStateNode]


class LSGFileModel(BaseModel):
    version: int = 1
    client: str
    generated_at: str
    guards: list[VocabEntry] = Field(default_factory=list)
    actions: list[VocabEntry] = Field(default_factory=list)
    workflows: list[LSGWorkflow]


class DiffItem(BaseModel):
    diff_id: str
    diff_class: Literal["A", "B"]
    workflow_id: WorkflowId
    state_id: str
    transition_guard: str
    involved_clients: list[ClientName]
    summary: str
    expected_behavior: str = ""
    actual_behavior: str = ""
    evidence: dict[str, list[Evidence]] = Field(default_factory=dict)


class DiffReport(BaseModel):
    iteration: int
    compared_items: int
    a_diff_count: int
    b_diff_count: int
    logic_diff_rate: float
    class_a: list[DiffItem] = Field(default_factory=list)
    class_b: list[DiffItem] = Field(default_factory=list)
    notes: str = ""
