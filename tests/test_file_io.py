"""Tests for file_io modules — checkpoint, writer, audit_logger."""

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest
import yaml

import config as _config


@pytest.fixture(autouse=True)
def _use_temp_output(tmp_path, monkeypatch):
    """Redirect all output paths to a temporary directory."""
    monkeypatch.setattr(_config, "OUTPUT_PATH", tmp_path / "output")
    monkeypatch.setattr(_config, "CHECKPOINT_PATH", tmp_path / "output" / "checkpoints")
    monkeypatch.setattr(_config, "ITERATIONS_PATH", tmp_path / "output" / "iterations")
    monkeypatch.setattr(_config, "AUDIT_LOG_PATH", tmp_path / "output" / "audit_logs")
    yield


# ── Checkpoint ──────────────────────────────────────────────────────────

class TestCheckpoint:
    def test_save_and_load(self):
        from file_io.checkpoint import save_checkpoint, load_checkpoint

        state = {"current_phase": 1, "phase1_iteration": 3, "guards": [{"name": "G1"}]}
        path = save_checkpoint(state, phase=1, iteration=3)
        assert path.exists()
        assert "phase1" in path.name

        loaded = load_checkpoint(1, 3)
        assert loaded["current_phase"] == 1
        assert loaded["guards"][0]["name"] == "G1"

    def test_load_nonexistent_raises(self):
        from file_io.checkpoint import load_checkpoint
        with pytest.raises(FileNotFoundError):
            load_checkpoint(99, 99)

    def test_latest_checkpoint_empty(self, tmp_path, monkeypatch):
        """No checkpoints yet → returns None.

        Uses a dedicated empty directory to avoid interference from other tests.
        """
        from file_io.checkpoint import latest_checkpoint

        empty_dir = tmp_path / "empty_ckpts"
        empty_dir.mkdir()
        monkeypatch.setattr(_config, "CHECKPOINT_PATH", empty_dir)
        result = latest_checkpoint()
        assert result is None

    def test_latest_checkpoint_finds_newest(self):
        from file_io.checkpoint import save_checkpoint, latest_checkpoint

        save_checkpoint({"phase": 1}, phase=1, iteration=1)
        save_checkpoint({"phase": 2}, phase=2, iteration=3)

        result = latest_checkpoint()
        assert result is not None
        phase, iteration, state = result
        assert phase == 2
        assert iteration == 3

    def test_checkpoint_serializes_pydantic(self):
        from file_io.checkpoint import save_checkpoint, load_checkpoint
        from state import VocabEntry

        entry = VocabEntry(name="G1", category="net", description="d")
        state = {"guards": [entry]}
        save_checkpoint(state, phase=1, iteration=1)

        loaded = load_checkpoint(1, 1)
        assert loaded["guards"][0]["name"] == "G1"


# ── Writer ──────────────────────────────────────────────────────────────

class TestWriter:
    def test_write_enriched_spec(self):
        from file_io.writer import write_enriched_spec

        state = {
            "guards": [{"name": "G1", "category": "net", "description": "test"}],
            "actions": [{"name": "A1", "category": "block", "description": "test2"}],
        }
        path = write_enriched_spec(state)
        assert path.exists()

        with open(path) as f:
            data = yaml.safe_load(f)
        assert data["version"] == 1
        assert len(data["guards"]) == 1
        assert len(data["actions"]) == 1

    def test_write_client_lsg_final(self):
        from file_io.writer import write_client_lsg

        lsg = {"version": 1, "client": "prysm", "workflows": []}
        path = write_client_lsg("prysm", lsg, final=True)
        assert "final" in path.name
        assert path.exists()

    def test_write_client_lsg_iteration(self):
        from file_io.writer import write_client_lsg

        lsg = {"version": 1, "client": "prysm", "workflows": [], "_iteration": 3}
        path = write_client_lsg("prysm", lsg, final=False)
        assert "iter3" in path.name

    def test_write_all_final_lsgs(self):
        from file_io.writer import write_all_final_lsgs

        client_lsgs = {}
        for name in _config.CLIENT_NAMES:
            client_lsgs[name] = {"version": 1, "client": name, "workflows": []}

        state = {"client_lsgs": client_lsgs}
        paths = write_all_final_lsgs(state)
        assert len(paths) == 5

    def test_write_all_final_lsgs_backfills_vocab(self):
        """Empty guards/actions are backfilled from global vocabulary."""
        from file_io.writer import write_all_final_lsgs

        # Build a client LSG with empty guards/actions but referenced names
        client_lsgs = {}
        for name in _config.CLIENT_NAMES:
            client_lsgs[name] = {
                "version": 1,
                "client": name,
                "guards": [],  # empty — LLM didn't return them
                "actions": [],
                "workflows": [{
                    "id": "initial_sync",
                    "name": "Initial Sync",
                    "states": [{
                        "id": "initial_sync.init",
                        "label": "Init",
                        "category": "init",
                        "transitions": [{
                            "guard": "PeerAvailable",
                            "actions": ["StartSync", "LogEvent"],
                            "next_state": "initial_sync.syncing",
                        }],
                    }],
                }],
            }

        # Global vocab from Phase 1
        global_guards = [
            {"name": "PeerAvailable", "category": "net", "description": "Peer is up"},
            {"name": "UnusedGuard", "category": "net", "description": "Not used"},
        ]
        global_actions = [
            {"name": "StartSync", "category": "sync", "description": "Start syncing"},
            {"name": "LogEvent", "category": "misc", "description": "Log something"},
            {"name": "UnusedAction", "category": "misc", "description": "Not used"},
        ]

        state = {
            "client_lsgs": client_lsgs,
            "guards": global_guards,
            "actions": global_actions,
        }
        paths = write_all_final_lsgs(state)
        assert len(paths) == 5

        # Verify the backfilled data via re-reading the YAML
        with open(paths[0]) as f:
            data = yaml.safe_load(f)
        # Should have exactly 1 guard (PeerAvailable) and 2 actions (StartSync, LogEvent)
        # NOT the unused ones
        guard_names = [g["name"] for g in data.get("guards", [])]
        action_names = [a["name"] for a in data.get("actions", [])]
        assert "PeerAvailable" in guard_names
        assert "UnusedGuard" not in guard_names
        assert "StartSync" in action_names
        assert "LogEvent" in action_names
        assert "UnusedAction" not in action_names

    def test_write_diff_report(self):
        from file_io.writer import write_diff_report

        state = {
            "diff_report": {
                "a_class_diffs": [
                    {
                        "workflow_id": "initial_sync",
                        "state_id": "initial_sync.init",
                        "transition_guard": "OldGuard",
                        "diff_type": "A",
                        "description": "In prysm: rename guard `OldGuard` → `NewGuard`",
                        "involved_clients": ["prysm", "lighthouse"],
                        "evidence": {},
                    }
                ],
                "b_class_diffs": [
                    {
                        "workflow_id": "initial_sync",
                        "state_id": "s1",
                        "transition_guard": "G1",
                        "diff_type": "B",
                        "description": "Transition present in prysm but no equivalent in lighthouse",
                        "severity": "MINOR",
                        "involved_clients": ["prysm", "lighthouse"],
                        "evidence": {},
                    }
                ],
                "logic_diff_rate": 0.1,
                "total_transitions": 10,
            },
            "force_stopped": False,
            "convergence_reason": "A-class delta stabilized at iteration 3",
            "iteration_history": [
                {"iteration": 1, "a_class_count": 3, "b_class_count": 5, "logic_diff_rate": 0.5},
                {"iteration": 2, "a_class_count": 1, "b_class_count": 5, "logic_diff_rate": 0.4},
            ],
        }
        path = write_diff_report(state)
        assert path.exists()
        content = path.read_text()
        # Verify enriched sections
        assert "Executive Summary" in content
        assert "Per-Workflow Summary" in content
        assert "Per-Client Deviation Ranking" in content
        assert "A-Class Vocabulary Alignment Diffs" in content
        assert "B-Class Structural Logic Differences" in content
        assert "Agreement" in content
        assert "Iteration Trend" in content
        assert "initial_sync" in content
        assert "OldGuard" in content  # A-class diff shown
        # New: convergence reason and severity
        assert "Convergence Reason" in content
        assert "delta stabilized" in content
        assert "MINOR" in content or "Minor" in content

    def test_write_diff_report_severity_grouping(self):
        """B-class diffs are grouped by severity tier in the report."""
        from file_io.writer import write_diff_report

        state = {
            "diff_report": {
                "a_class_diffs": [],
                "b_class_diffs": [
                    {
                        "workflow_id": "initial_sync",
                        "state_id": "initial_sync.*",
                        "transition_guard": "*",
                        "diff_type": "B",
                        "description": "Workflow `initial_sync` is substantive in X but only a stub in Y",
                        "severity": "CRITICAL",
                        "involved_clients": ["prysm", "lighthouse"],
                        "evidence": {},
                    },
                    {
                        "workflow_id": "regular_sync",
                        "state_id": "regular_sync.validate",
                        "transition_guard": "HasBlock",
                        "diff_type": "B",
                        "description": "Different validation approach",
                        "severity": "MAJOR",
                        "involved_clients": ["prysm", "teku"],
                        "evidence": {},
                    },
                    {
                        "workflow_id": "aggregate",
                        "state_id": "aggregate.pool",
                        "transition_guard": "TRUE",
                        "diff_type": "B",
                        "description": "Transition present in prysm but no equivalent in teku",
                        "severity": "MINOR",
                        "involved_clients": ["prysm", "teku"],
                        "evidence": {},
                    },
                ],
                "logic_diff_rate": 0.5,
            },
            "force_stopped": False,
        }
        path = write_diff_report(state)
        content = path.read_text()
        # Verify severity sub-headings exist
        assert "Critical" in content
        assert "Major" in content
        assert "Minor" in content

    def test_write_diff_report_deduplication(self):
        """Symmetric B-class diffs are merged into one entry."""
        from file_io.writer import write_diff_report

        state = {
            "diff_report": {
                "a_class_diffs": [],
                "b_class_diffs": [
                    {
                        "workflow_id": "regular_sync",
                        "state_id": "regular_sync.check",
                        "transition_guard": "HasPeers",
                        "diff_type": "B",
                        "description": "Transition present in prysm but no equivalent in lighthouse",
                        "severity": "MINOR",
                        "involved_clients": ["prysm", "lighthouse"],
                        "evidence": {},
                    },
                    {
                        "workflow_id": "regular_sync",
                        "state_id": "regular_sync.check",
                        "transition_guard": "HasPeers",
                        "diff_type": "B",
                        "description": "Transition present in lighthouse but no equivalent in prysm",
                        "severity": "MINOR",
                        "involved_clients": ["lighthouse", "prysm"],
                        "evidence": {},
                    },
                ],
                "logic_diff_rate": 0.5,
            },
            "force_stopped": False,
        }
        path = write_diff_report(state)
        content = path.read_text()
        # Should only have B-1, not B-2 (deduped into one entry)
        assert "B-1:" in content
        assert "B-2:" not in content

    def test_write_diff_report_json(self):
        from file_io.writer import write_diff_report_json

        state = {
            "diff_report": {
                "a_class_diffs": [
                    {
                        "workflow_id": "regular_sync",
                        "state_id": "regular_sync.init",
                        "transition_guard": "X",
                        "diff_type": "A",
                        "description": "rename",
                        "involved_clients": ["prysm"],
                        "evidence": {},
                    }
                ],
                "b_class_diffs": [],
                "logic_diff_rate": 0.0,
            },
            "force_stopped": False,
            "convergence_reason": "Zero A-class diffs",
            "iteration_history": [],
        }
        path = write_diff_report_json(state)
        assert path.exists()
        assert path.suffix == ".json"

        import json
        with open(path) as f:
            data = json.load(f)
        assert "executive_summary" in data
        assert "per_workflow_summary" in data
        assert "per_client_ranking" in data
        assert "agreement_workflows" in data
        assert "convergence_reason" in data
        assert data["summary"]["a_class_count"] == 1
        assert data["summary"]["b_class_count"] == 0
        assert "b_class_critical" in data["summary"]
        assert "b_class_major" in data["summary"]
        assert "b_class_minor" in data["summary"]
        assert "total_transitions" in data["summary"]

    def test_normalize_severity_aliases(self):
        """Non-standard severity values from LLM (e.g. 'High') are normalized."""
        from file_io.writer import _normalize_severity

        # Canonical values pass through
        assert _normalize_severity({"severity": "CRITICAL"}) == "CRITICAL"
        assert _normalize_severity({"severity": "MAJOR"}) == "MAJOR"
        assert _normalize_severity({"severity": "MINOR"}) == "MINOR"

        # Common LLM quirks
        assert _normalize_severity({"severity": "High"}) == "MAJOR"
        assert _normalize_severity({"severity": "high"}) == "MAJOR"
        assert _normalize_severity({"severity": "Medium"}) == "MINOR"
        assert _normalize_severity({"severity": "low"}) == "MINOR"
        assert _normalize_severity({"severity": "severe"}) == "CRITICAL"

        # Empty/missing falls back to heuristic
        assert _normalize_severity({"severity": "", "state_id": "x.*"}) == "CRITICAL"
        assert _normalize_severity({"severity": None, "description": "some diff"}) == "MAJOR"

    def test_write_diff_report_severity_normalization(self):
        """Non-standard severity values are normalized in the final report."""
        from file_io.writer import write_diff_report_json

        state = {
            "diff_report": {
                "a_class_diffs": [],
                "b_class_diffs": [
                    {
                        "workflow_id": "block_generate",
                        "state_id": "teku.assembling_block",
                        "transition_guard": "TRUE",
                        "diff_type": "B",
                        "description": "Blob handling differs",
                        "severity": "High",  # Non-standard from LLM
                        "involved_clients": ["teku", "lighthouse"],
                        "evidence": {},
                    },
                ],
                "logic_diff_rate": 0.5,
            },
            "force_stopped": False,
        }
        path = write_diff_report_json(state)
        import json
        with open(path) as f:
            data = json.load(f)
        # "High" should have been normalized to "MAJOR"
        assert data["b_class_diffs"][0]["severity"] == "MAJOR"
        assert data["summary"]["b_class_major"] == 1

    def test_write_diff_report_empty_diffs(self):
        from file_io.writer import write_diff_report

        state = {
            "diff_report": {
                "a_class_diffs": [],
                "b_class_diffs": [],
                "logic_diff_rate": 0.0,
            },
            "force_stopped": False,
        }
        path = write_diff_report(state)
        content = path.read_text()
        assert "No vocabulary misalignment detected" in content
        assert "No structural logic differences found" in content

    def test_per_workflow_summary_uses_lsg_counts(self):
        """Per-workflow similarity uses actual transition counts from client_lsgs."""
        from file_io.writer import _per_workflow_summary

        # 2 B-class diffs in initial_sync, 1 in regular_sync
        b_diffs = [
            {"workflow_id": "initial_sync"},
            {"workflow_id": "initial_sync"},
            {"workflow_id": "regular_sync"},
        ]
        # initial_sync has 10 transitions (max across clients), regular_sync has 5
        client_lsgs = {
            "prysm": {
                "workflows": [
                    {"id": "initial_sync", "states": [{"transitions": [{}] * 10}]},
                    {"id": "regular_sync", "states": [{"transitions": [{}] * 5}]},
                ]
            },
            "lighthouse": {
                "workflows": [
                    {"id": "initial_sync", "states": [{"transitions": [{}] * 8}]},
                    {"id": "regular_sync", "states": [{"transitions": [{}] * 5}]},
                ]
            },
        }
        rows = _per_workflow_summary([], b_diffs, total_transitions=15, client_lsgs=client_lsgs)
        by_wf = {r["workflow_id"]: r for r in rows}
        # initial_sync: 2 B-class out of 10 → similarity = 0.8
        assert abs(by_wf["initial_sync"]["similarity"] - 0.8) < 0.01
        # regular_sync: 1 B-class out of 5 → similarity = 0.8
        assert abs(by_wf["regular_sync"]["similarity"] - 0.8) < 0.01

    def test_per_workflow_summary_without_lsg_falls_back(self):
        """Without client_lsgs, similarity falls back to even split."""
        from file_io.writer import _per_workflow_summary

        b_diffs = [
            {"workflow_id": "initial_sync"},
        ]
        rows = _per_workflow_summary([], b_diffs, total_transitions=70, client_lsgs=None)
        by_wf = {r["workflow_id"]: r for r in rows}
        # Even split: 70 / 7 = 10 per workflow; 1 B-class → similarity = 0.9
        assert abs(by_wf["initial_sync"]["similarity"] - 0.9) < 0.01

    def test_per_client_ranking_deviating_clients(self):
        """Ranking uses deviating_clients to differentiate ref from deviator."""
        from file_io.writer import _per_client_ranking

        b_diffs = [
            {
                "involved_clients": ["lighthouse", "prysm"],
                "deviating_clients": ["lighthouse"],
            },
            {
                "involved_clients": ["lighthouse", "prysm"],
                "deviating_clients": ["lighthouse"],
            },
            {
                "involved_clients": ["prysm", "teku"],
                "deviating_clients": ["teku"],
            },
        ]
        rows = _per_client_ranking([], b_diffs)
        by_client = {r["client"]: r for r in rows}
        # lighthouse: involved in 2, deviating in 2
        assert by_client["lighthouse"]["b_class"] == 2
        assert by_client["lighthouse"]["b_class_deviating"] == 2
        # prysm: involved in 3 (as ref), deviating in 0
        assert by_client["prysm"]["b_class"] == 3
        assert by_client["prysm"]["b_class_deviating"] == 0
        # teku: involved in 1, deviating in 1
        assert by_client["teku"]["b_class"] == 1
        assert by_client["teku"]["b_class_deviating"] == 1
        # Sorted by b_class_deviating: lighthouse (2) > teku (1) > prysm (0)
        assert rows[0]["client"] == "lighthouse"

    def test_per_client_ranking_no_deviating_falls_back(self):
        """Without deviating_clients, all involved_clients are counted as deviating."""
        from file_io.writer import _per_client_ranking

        b_diffs = [
            {"involved_clients": ["prysm", "lighthouse"]},  # no deviating_clients
        ]
        rows = _per_client_ranking([], b_diffs)
        by_client = {r["client"]: r for r in rows}
        # Both counted as deviating (fallback)
        assert by_client["prysm"]["b_class_deviating"] == 1
        assert by_client["lighthouse"]["b_class_deviating"] == 1

    def test_deduplication_preserves_deviating_clients(self):
        """Deduplication merges deviating_clients from symmetric diffs."""
        from file_io.writer import _deduplicate_b_diffs

        diffs = [
            {
                "workflow_id": "initial_sync",
                "state_id": "initial_sync.check",
                "transition_guard": "G1",
                "diff_type": "B",
                "description": "present in prysm but no equivalent in lighthouse",
                "severity": "MINOR",
                "involved_clients": ["lighthouse", "prysm"],
                "deviating_clients": ["lighthouse"],
                "evidence": {},
            },
            {
                "workflow_id": "initial_sync",
                "state_id": "initial_sync.check",
                "transition_guard": "G1",
                "diff_type": "B",
                "description": "present in lighthouse but no equivalent in teku",
                "severity": "MINOR",
                "involved_clients": ["lighthouse", "teku"],
                "deviating_clients": ["teku"],
                "evidence": {},
            },
        ]
        result = _deduplicate_b_diffs(diffs)
        assert len(result) == 1
        assert sorted(result[0]["deviating_clients"]) == ["lighthouse", "teku"]
        assert sorted(result[0]["involved_clients"]) == ["lighthouse", "prysm", "teku"]


# ── Audit Logger ────────────────────────────────────────────────────────

class TestAuditLogger:
    def test_on_llm_start(self):
        from file_io.audit_logger import AuditLogCallback

        cb = AuditLogCallback(phase=1, iteration=2, agent_type="sub_prysm")
        cb.on_llm_start({"model": "test"}, ["test prompt"])

        assert len(cb.paths) == 1
        path = Path(cb.paths[0])
        assert path.exists()

        with open(path) as f:
            data = json.load(f)
        assert data["event_type"] == "llm_start"
        assert data["phase"] == 1
        assert data["iteration"] == 2

    def test_on_llm_end(self):
        from file_io.audit_logger import AuditLogCallback

        cb = AuditLogCallback(phase=2, iteration=1, agent_type="main")
        cb.on_llm_end({"text": "response"})

        assert len(cb.paths) == 1
        path = Path(cb.paths[0])
        with open(path) as f:
            data = json.load(f)
        assert data["event_type"] == "llm_end"

    def test_on_llm_error(self):
        from file_io.audit_logger import AuditLogCallback

        cb = AuditLogCallback(phase=1, iteration=1, agent_type="test")
        cb.on_llm_error(RuntimeError("boom"))

        assert len(cb.paths) == 1
        path = Path(cb.paths[0])
        with open(path) as f:
            data = json.load(f)
        assert data["event_type"] == "llm_error"
        assert "boom" in data["payload"]["error"]

    def test_filename_format(self):
        from file_io.audit_logger import AuditLogCallback

        cb = AuditLogCallback(phase=2, iteration=5, agent_type="sub_teku")
        cb.on_llm_start({}, [])

        path = Path(cb.paths[0])
        assert "phase2" in path.name
        assert "iter5" in path.name
        assert "sub_teku" in path.name
