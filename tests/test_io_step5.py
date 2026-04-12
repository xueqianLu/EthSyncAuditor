"""Tests for I/O functions (Step 5) — checkpoint, writer, audit logger."""

from __future__ import annotations

from pathlib import Path

from eth_io import (
    make_audit_callback,
    save_checkpoint,
    load_checkpoint,
    write_diff_report,
    write_enriched_spec,
    write_final_lsgs,
    write_iteration_lsg,
)
from graph import make_initial_state


def test_checkpoint_save_and_load_roundtrip():
    state = make_initial_state()
    state["current_phase"] = 1
    state["phase1_iteration"] = 3

    path = save_checkpoint(state, phase=1, iteration=3)
    loaded = load_checkpoint(phase=1, iteration=3)

    assert path.exists()
    assert loaded["current_phase"] == 1
    assert loaded["phase1_iteration"] == 3


def test_writer_outputs_exist():
    spec_path = write_enriched_spec(
        {
            "guards": [{"name": "RespRecv", "category": "network", "description": "x"}],
            "actions": [{"name": "ApplyBatch", "category": "block", "description": "y"}],
        }
    )
    assert spec_path.exists()

    iter_path = write_iteration_lsg(
        "prysm",
        1,
        {
            "version": 1,
            "client": "prysm",
            "generated_at": "2026-01-01T00:00:00Z",
            "guards": [],
            "actions": [],
            "workflows": [],
        },
    )
    assert iter_path.exists()

    finals = write_final_lsgs(
        {
            "prysm": {
                "version": 1,
                "client": "prysm",
                "generated_at": "2026-01-01T00:00:00Z",
                "guards": [],
                "actions": [],
                "workflows": [],
            }
        }
    )
    assert finals and finals[0].exists()

    report = write_diff_report(
        diff_items_b=[],
        summary={"compared_items": 10, "a_diff_count": 2, "b_diff_count": 0, "logic_diff_rate": 0.0},
    )
    assert report.exists()


def test_audit_logger_callback_writes_file():
    context = {"phase": 2, "iteration": 4, "agent_type": "phase2_sub"}

    def state_getter():
        return context

    handler = make_audit_callback(state_getter)
    handler.on_llm_start({"name": "mock-llm"}, ["hello"], run_id="r1")
    handler.on_llm_end({"ok": True}, run_id="r1")

    audit_dir = Path(__file__).resolve().parents[1] / "output" / "audit_logs"
    files = list(audit_dir.glob("audit_phase2_iter4_phase2_sub_*.json"))
    assert files
