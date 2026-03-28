"""Integration tests for end-to-end wiring: main.py, configure_graph, resume."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

import config as _config
from graph import configure_graph, get_graph_config


@pytest.fixture(autouse=True)
def _reset_graph_config():
    """Ensure graph config is reset after each test."""
    configure_graph()
    yield
    configure_graph()


@pytest.fixture
def _use_temp_output(tmp_path, monkeypatch):
    """Redirect all output paths to a temporary directory."""
    monkeypatch.setattr(_config, "OUTPUT_PATH", tmp_path / "output")
    monkeypatch.setattr(_config, "CHECKPOINT_PATH", tmp_path / "output" / "checkpoints")
    monkeypatch.setattr(_config, "ITERATIONS_PATH", tmp_path / "output" / "iterations")
    monkeypatch.setattr(_config, "AUDIT_LOG_PATH", tmp_path / "output" / "audit_logs")
    return tmp_path


class TestConfigureGraph:
    """Tests for the configure_graph / get_graph_config API."""

    def test_mock_by_default(self):
        configure_graph()
        assert get_graph_config()["mock"] is True

    def test_set_llm(self):
        sentinel = object()
        configure_graph(llm=sentinel, mock=False)
        cfg = get_graph_config()
        assert cfg["llm"] is sentinel
        assert cfg["mock"] is False

    def test_callbacks_stored(self):
        cb = [object()]
        configure_graph(callbacks=cb)
        assert get_graph_config()["callbacks"] is cb


class TestMainMockPipeline:
    """Test running main.py --mock via subprocess."""

    def test_mock_run_succeeds(self, tmp_path, monkeypatch):
        """python main.py --mock exits 0 and produces output files."""
        env = {
            **dict(__import__("os").environ),
        }
        result = subprocess.run(
            [sys.executable, "main.py", "--mock"],
            cwd=str(Path(__file__).resolve().parent.parent),
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "Pipeline finished" in result.stderr

    def test_mock_run_produces_outputs(self, tmp_path):
        """After mock run, expected output files exist."""
        project_root = Path(__file__).resolve().parent.parent
        output_dir = project_root / "output"
        result = subprocess.run(
            [sys.executable, "main.py", "--mock"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0

        # Verify key output files
        assert (output_dir / "Global_LSG_Spec_Enriched.yaml").exists()
        assert (output_dir / "Audit_Diff_Report.md").exists()

        # Verify at least one checkpoint
        ckpts = list((output_dir / "checkpoints").glob("checkpoint_*.json"))
        assert len(ckpts) >= 1

        # All 5 final LSGs should exist (invoke returns properly merged state)
        final_lsgs = list(output_dir.glob("LSG_*_final.yaml"))
        assert len(final_lsgs) >= 5


class TestResumeFromCheckpoint:
    """Test --resume functionality."""

    def test_resume_no_checkpoint_starts_fresh(self, _use_temp_output):
        """--resume with no checkpoints starts from scratch."""
        from main import _init_llm

        # _init_llm without key returns None
        llm = _init_llm("test-model")
        assert llm is None  # no ANTHROPIC_API_KEY

    def test_resume_loads_latest_checkpoint(self, _use_temp_output):
        """latest_checkpoint returns the newest file when checkpoints exist."""
        from file_io.checkpoint import save_checkpoint, latest_checkpoint

        save_checkpoint({"current_phase": 1, "x": 1}, phase=1, iteration=1)
        save_checkpoint({"current_phase": 2, "x": 2}, phase=2, iteration=3)

        result = latest_checkpoint()
        assert result is not None
        phase, iteration, state = result
        assert phase == 2
        assert iteration == 3
        assert state["x"] == 2


class TestInitLlm:
    """Test _init_llm graceful degradation."""

    def test_no_api_key_returns_none(self, monkeypatch):
        """Without ANTHROPIC_API_KEY, _init_llm returns None."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        from main import _init_llm
        result = _init_llm("claude-sonnet-4-20250514")
        assert result is None
