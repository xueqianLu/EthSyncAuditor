from __future__ import annotations

from datetime import datetime, timezone

from agents.phase1_main_agent import run_phase1_main_agent
from agents.phase1_sub_agent import run_phase1_sub_agent
from agents.phase2_main_agent import run_phase2_main_agent
from agents.phase2_sub_agent import run_phase2_sub_agent
from agents.schemas import (
    DiffReport,
    EnrichedSpec,
    LSGFileModel,
    VocabDiscoveryReport,
)


class MockStructuredLLM:
    def __init__(self, response):
        self.response = response

    def invoke(self, _messages):
        return self.response


class MockLLM:
    def __init__(self, response):
        self.response = response

    def with_structured_output(self, _schema):
        return MockStructuredLLM(self.response)


class MockReActExecutor:
    def __init__(self, response):
        self.response = response

    def invoke(self, _payload):
        return {"structured_response": self.response}


def test_phase1_sub_agent_with_mock_executor():
    mock_report = {
        "client_name": "prysm",
        "discovered_guards": [
            {
                "name": "ModeIsBootstrapping",
                "category": "mode",
                "description": "Node is in bootstrap mode",
                "aliases": [],
                "evidence": [
                    {
                        "file": "code/prysm/sync/mock.go",
                        "function": "runInitialSync",
                        "lines": [10, 20],
                    }
                ],
            }
        ],
        "discovered_actions": [],
        "notes": "ok",
    }

    report = run_phase1_sub_agent(
        client_name="prysm",
        current_vocab={"guards": [], "actions": []},
        iteration=1,
        llm=MockLLM(mock_report),
        agent_executor=MockReActExecutor(mock_report),
    )

    assert isinstance(report, VocabDiscoveryReport)
    assert report.client_name == "prysm"
    assert report.discovered_guards[0].name == "ModeIsBootstrapping"


def test_phase1_main_agent_with_mock_llm():
    mock_enriched = {
        "version": 1,
        "vocab_version": 2,
        "guards": [
            {
                "name": "RespRecv",
                "category": "network",
                "description": "response received",
                "aliases": [],
                "evidence": [],
            }
        ],
        "actions": [],
        "merge_summary": "merged",
    }

    out = run_phase1_main_agent(
        sub_reports=[],
        current_vocab={"guards": [], "actions": []},
        vocab_version=1,
        iteration=1,
        llm=MockLLM(mock_enriched),
    )

    assert isinstance(out, EnrichedSpec)
    assert out.vocab_version >= 2


def test_phase2_sub_agent_with_mock_executor():
    now = datetime.now(tz=timezone.utc).isoformat()
    mock_lsg = {
        "version": 1,
        "client": "teku",
        "generated_at": now,
        "guards": [],
        "actions": [],
        "workflows": [
            {
                "id": "initial_sync",
                "name": "Initial",
                "description": "desc",
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
                                "actions": [],
                                "next_state": "initial.done",
                                "evidence": {
                                    "file": "code/teku/a.java",
                                    "function": "doStart",
                                    "lines": [1, 5],
                                },
                            }
                        ],
                    }
                ],
            }
        ],
    }

    out = run_phase2_sub_agent(
        client_name="teku",
        enriched_spec={"guards": [], "actions": []},
        iteration=1,
        llm=MockLLM(mock_lsg),
        agent_executor=MockReActExecutor(mock_lsg),
    )

    assert isinstance(out, LSGFileModel)
    assert out.client == "teku"
    assert out.workflows[0].id == "initial_sync"


def test_phase2_main_agent_with_mock_llm():
    mock_report = {
        "iteration": 1,
        "compared_items": 20,
        "a_diff_count": 2,
        "b_diff_count": 1,
        "logic_diff_rate": 0.05,
        "class_a": [],
        "class_b": [],
        "notes": "ok",
    }

    out = run_phase2_main_agent(
        client_lsgs=[],
        iteration=1,
        llm=MockLLM(mock_report),
    )

    assert isinstance(out, DiffReport)
    assert out.logic_diff_rate == 0.05
