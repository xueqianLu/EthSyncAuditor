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
