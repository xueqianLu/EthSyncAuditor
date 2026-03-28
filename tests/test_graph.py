"""Tests for graph.py — graph compilation, pipeline execution, convergence."""

import pytest
import config as _config
from graph import (
    build_graph,
    compile_graph,
    make_initial_state,
    phase1_sub_agent_node,
    phase1_main_agent_node,
    phase2_sub_agent_node,
    phase2_main_agent_node,
    route_after_phase1_main,
    route_after_phase2_main,
    preprocess_node,
    phase1_converged_node,
    phase1_force_stop_node,
    phase2_converged_node,
    phase2_force_stop_node,
    phase1_next_iter_node,
    phase2_next_iter_node,
)


# ── Initial state ──────────────────────────────────────────────────────

def test_make_initial_state_fields():
    """Initial state has all required keys."""
    s = make_initial_state()
    expected_keys = {
        "current_phase", "phase1_iteration", "phase2_iteration",
        "guards", "actions", "vocab_version", "diff_rate",
        "client_lsgs", "diff_report", "logic_diff_rate",
        "converged_phase1", "converged_phase2", "force_stopped",
        "preprocess_done", "preprocess_status",
        "audit_log_paths", "discovery_reports", "a_class_feedback",
    }
    assert expected_keys.issubset(set(s.keys()))


def test_make_initial_state_defaults():
    s = make_initial_state()
    assert s["current_phase"] == 0
    assert s["phase1_iteration"] == 0
    assert s["diff_rate"] == 1.0
    assert s["converged_phase1"] is False
    assert s["force_stopped"] is False
    assert s["guards"] == []


# ── Graph compilation ──────────────────────────────────────────────────

def test_graph_compiles():
    """build_graph() + compile() should not raise."""
    g = build_graph()
    app = g.compile()
    assert app is not None


# ── Node unit tests ────────────────────────────────────────────────────

def test_preprocess_node_first_run():
    state = make_initial_state()
    result = preprocess_node(state)
    assert result["preprocess_done"] is True
    assert result["current_phase"] == 1
    assert result["phase1_iteration"] == 1
    assert "prysm" in result["preprocess_status"]


def test_preprocess_node_skip_if_done():
    state = make_initial_state()
    state["preprocess_done"] = True
    result = preprocess_node(state)
    assert result == {}


def test_phase1_sub_agent_node_returns_discovery_report():
    state = make_initial_state()
    state["_client_name"] = "prysm"
    state["phase1_iteration"] = 1
    result = phase1_sub_agent_node(state)
    assert "discovery_reports" in result
    reports = result["discovery_reports"]
    assert isinstance(reports, list)
    assert len(reports) == 1
    assert reports[0]["client_name"] == "prysm"


def test_phase1_main_agent_node_computes_diff_rate():
    state = make_initial_state()
    state["discovery_reports"] = [
        {"client_name": "prysm", "new_guards": [
            {"name": "G1", "category": "net", "description": "d"}
        ], "new_actions": []},
    ]
    result = phase1_main_agent_node(state)
    assert "diff_rate" in result
    assert "vocab_version" in result
    assert result["vocab_version"] == 1  # bumped from 0


def test_phase2_sub_agent_node_returns_lsg():
    state = make_initial_state()
    state["_client_name"] = "lighthouse"
    result = phase2_sub_agent_node(state)
    assert "client_lsgs" in result
    assert "lighthouse" in result["client_lsgs"]
    lsg = result["client_lsgs"]["lighthouse"]
    assert lsg["client"] == "lighthouse"
    assert len(lsg["workflows"]) == 7


def test_phase2_main_agent_node_computes_logic_diff_rate():
    # All clients have the same mock workflows → no B-class diffs
    state = make_initial_state()
    lsgs = {}
    for client in _config.CLIENT_NAMES:
        s = {**state, "_client_name": client}
        r = phase2_sub_agent_node(s)
        lsgs.update(r["client_lsgs"])
    state["client_lsgs"] = lsgs

    result = phase2_main_agent_node(state)
    assert result["logic_diff_rate"] == 0.0


# ── Router logic ───────────────────────────────────────────────────────

def test_route_phase1_converged():
    state = make_initial_state()
    state["diff_rate"] = 0.0  # below threshold
    assert route_after_phase1_main(state) == "phase1_converged"


def test_route_phase1_next_iter():
    state = make_initial_state()
    state["diff_rate"] = 0.5  # above threshold
    state["phase1_iteration"] = 1
    assert route_after_phase1_main(state) == "phase1_next_iter"


def test_route_phase1_force_stop():
    state = make_initial_state()
    state["diff_rate"] = 0.5
    state["phase1_iteration"] = _config.MAX_ITER_PHASE1
    assert route_after_phase1_main(state) == "phase1_force_stop"


def test_route_phase2_converged():
    state = make_initial_state()
    state["logic_diff_rate"] = 0.0
    assert route_after_phase2_main(state) == "phase2_converged"


def test_route_phase2_force_stop():
    state = make_initial_state()
    state["logic_diff_rate"] = 0.5
    state["phase2_iteration"] = _config.MAX_ITER_PHASE2
    assert route_after_phase2_main(state) == "phase2_force_stop"


def test_route_phase2_next_iter():
    state = make_initial_state()
    state["logic_diff_rate"] = 0.5
    state["phase2_iteration"] = 1
    assert route_after_phase2_main(state) == "phase2_next_iter"


# ── Transition nodes ───────────────────────────────────────────────────

def test_phase1_converged_node():
    state = make_initial_state()
    result = phase1_converged_node(state)
    assert result["converged_phase1"] is True
    assert result["current_phase"] == 2


def test_phase1_force_stop_node():
    state = make_initial_state()
    result = phase1_force_stop_node(state)
    assert result["force_stopped"] is True
    assert result["converged_phase1"] is False


def test_phase2_converged_node():
    state = make_initial_state()
    result = phase2_converged_node(state)
    assert result["converged_phase2"] is True


def test_phase2_force_stop_node():
    state = make_initial_state()
    result = phase2_force_stop_node(state)
    assert result["force_stopped"] is True


def test_phase1_next_iter_bumps():
    state = make_initial_state()
    state["phase1_iteration"] = 3
    result = phase1_next_iter_node(state)
    assert result["phase1_iteration"] == 4


def test_phase2_next_iter_bumps():
    state = make_initial_state()
    state["phase2_iteration"] = 5
    result = phase2_next_iter_node(state)
    assert result["phase2_iteration"] == 6


# ── End-to-end pipeline ────────────────────────────────────────────────

@pytest.mark.timeout(30)
def test_full_pipeline_converges():
    """Mock pipeline runs preprocess → Phase 1 → Phase 2 and converges."""
    app = compile_graph()
    initial = make_initial_state()
    final = None
    for step in app.stream(initial, stream_mode="updates"):
        for _node, update in step.items():
            if isinstance(update, dict):
                final = {**(final or {}), **update}

    assert final is not None
    assert final.get("preprocess_done") is True
    assert final.get("converged_phase2") is True


@pytest.mark.timeout(30)
def test_force_stop_pipeline():
    """Force stop triggers when convergence threshold is unreachable."""
    saved = (_config.MAX_ITER_PHASE1, _config.CONVERGENCE_THRESHOLD)
    try:
        _config.MAX_ITER_PHASE1 = 1
        _config.CONVERGENCE_THRESHOLD = 0.0  # never converge

        app = compile_graph()
        initial = make_initial_state()
        initial["diff_rate"] = 1.0

        final = None
        for step in app.stream(initial, stream_mode="updates"):
            for _node, update in step.items():
                if isinstance(update, dict):
                    final = {**(final or {}), **update}

        assert final is not None
        assert final.get("force_stopped") is True
    finally:
        _config.MAX_ITER_PHASE1, _config.CONVERGENCE_THRESHOLD = saved
