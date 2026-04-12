"""Tests for agent builder functions (Step 4 — mock mode).

These tests verify that the build_phase*_agent functions work in mock
mode (no LLM) and return expected data structures.
"""

from __future__ import annotations

from datetime import datetime, timezone

from agents.phase1_main_agent import build_phase1_main_agent
from agents.phase1_sub_agent import build_phase1_sub_agent
from agents.phase2_main_agent import build_phase2_main_agent
from agents.phase2_sub_agent import build_phase2_sub_agent
from agents.schemas import (
    DiffReport,
    EnrichedSpec,
    LSGFileModel,
    VocabDiscoveryReport,
)


def test_phase1_sub_agent_mock():
    """build_phase1_sub_agent with llm=None returns mock discovery report."""
    agent_fn = build_phase1_sub_agent(client_name="prysm", llm=None)
    state = {
        "guards": [],
        "actions": [],
        "vocab_version": 0,
        "phase1_iteration": 1,
    }
    result = agent_fn(state)
    assert "discovery_reports" in result
    reports = result["discovery_reports"]
    assert isinstance(reports, list)
    assert len(reports) == 1
    assert reports[0]["client_name"] == "prysm"


def test_phase1_main_agent_mock():
    """build_phase1_main_agent with llm=None returns deterministic merge."""
    agent_fn = build_phase1_main_agent(llm=None)
    state = {
        "guards": [],
        "actions": [],
        "vocab_version": 0,
        "discovery_reports": [
            {
                "client_name": "prysm",
                "new_guards": [
                    {"name": "G1", "category": "net", "description": "d"}
                ],
                "new_actions": [],
            },
        ],
    }
    result = agent_fn(state)
    assert "diff_rate" in result
    assert "vocab_version" in result
    assert result["vocab_version"] >= 1


def test_phase2_sub_agent_mock():
    """build_phase2_sub_agent with llm=None returns mock LSG for one workflow."""
    agent_fn = build_phase2_sub_agent(client_name="teku", llm=None)
    state = {
        "guards": [],
        "actions": [],
        "phase2_iteration": 1,
        "current_workflow": "initial_sync",
        "client_lsgs": {},
        "a_class_feedback": [],
        "sparsity_hints": [],
    }
    result = agent_fn(state)
    assert "client_lsgs" in result
    assert "teku" in result["client_lsgs"]
    lsg = result["client_lsgs"]["teku"]
    assert lsg["client"] == "teku"


def test_phase2_main_agent_mock():
    """build_phase2_main_agent with llm=None returns deterministic comparison."""
    # First build mock LSGs for all clients
    from config import CLIENT_NAMES

    client_lsgs = {}
    for client in CLIENT_NAMES:
        agent_fn = build_phase2_sub_agent(client_name=client, llm=None)
        state = {
            "guards": [],
            "actions": [],
            "phase2_iteration": 1,
            "current_workflow": "initial_sync",
            "client_lsgs": {},
            "a_class_feedback": [],
            "sparsity_hints": [],
        }
        result = agent_fn(state)
        client_lsgs.update(result["client_lsgs"])

    main_fn = build_phase2_main_agent(llm=None)
    state = {
        "guards": [],
        "actions": [],
        "phase2_iteration": 1,
        "current_workflow": "initial_sync",
        "client_lsgs": client_lsgs,
    }
    result = main_fn(state)
    assert "logic_diff_rate" in result
    assert result["logic_diff_rate"] == 0.0
