"""Tests for graph.py — graph compilation, pipeline execution, convergence."""

import pytest
import config as _config
from graph import (
    build_graph,
    compile_graph,
    configure_graph,
    get_graph_config,
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
    phase2_enter_b_class_focus_node,
    phase1_next_iter_node,
    phase2_next_iter_node,
    _graph_config,
    _last_p2_convergence_reason,
)
import graph as _graph_module


# ── Initial state ──────────────────────────────────────────────────────

def test_make_initial_state_fields():
    """Initial state has all required keys."""
    s = make_initial_state()
    expected_keys = {
        "current_phase", "phase1_iteration", "phase2_iteration",
        "guards", "actions", "vocab_version", "diff_rate",
        "client_lsgs", "diff_report", "logic_diff_rate",
        "converged_phase1", "converged_phase2", "force_stopped",
        "convergence_reason",
        "a_class_count", "prev_a_class_count", "iteration_history",
        "preprocess_done", "preprocess_status",
        "audit_log_paths", "discovery_reports", "a_class_feedback",
        "b_class_focus", "b_class_focus_iteration", "prev_b_class_count",
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
    """A-class count == 0 → enter B-class focus (not final convergence)."""
    state = make_initial_state()
    state["a_class_count"] = 0
    assert route_after_phase2_main(state) == "phase2_enter_b_class_focus"


def test_route_phase2_converged_delta_stable():
    """A-class delta stable → enter B-class focus."""
    state = make_initial_state()
    state["a_class_count"] = 10
    state["prev_a_class_count"] = 10  # delta = 0 → below threshold
    state["phase2_iteration"] = 3
    assert route_after_phase2_main(state) == "phase2_enter_b_class_focus"
    assert "b-class" in _graph_module._last_p2_convergence_reason.lower()


def test_route_phase2_converged_oscillation():
    """A-class oscillation → enter B-class focus."""
    state = make_initial_state()
    state["a_class_count"] = 12
    state["prev_a_class_count"] = -1  # skip delta check
    state["phase2_iteration"] = 5
    state["iteration_history"] = [
        {"iteration": 3, "a_class_count": 11, "b_class_count": 40, "logic_diff_rate": 0.7},
        {"iteration": 4, "a_class_count": 12, "b_class_count": 40, "logic_diff_rate": 0.7},
        {"iteration": 5, "a_class_count": 11, "b_class_count": 40, "logic_diff_rate": 0.7},
    ]
    assert route_after_phase2_main(state) == "phase2_enter_b_class_focus"
    assert "oscillat" in _graph_module._last_p2_convergence_reason.lower()


def test_route_phase2_force_stop():
    state = make_initial_state()
    state["a_class_count"] = 20  # not zero
    state["prev_a_class_count"] = -1  # skip delta check
    state["phase2_iteration"] = _config.MAX_ITER_PHASE2
    assert route_after_phase2_main(state) == "phase2_force_stop"


def test_route_phase2_next_iter():
    state = make_initial_state()
    state["a_class_count"] = 20
    state["prev_a_class_count"] = -1  # skip delta check
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
    # Set up B-class focus convergence so the router returns "phase2_converged"
    state["b_class_focus"] = True
    state["b_class_focus_iteration"] = _config.MAX_ITER_B_CLASS
    state["phase2_iteration"] = 8
    state["prev_b_class_count"] = 5
    state["diff_report"] = {"b_class_diffs": [{}] * 10}
    route_after_phase2_main(state)
    result = phase2_converged_node(state)
    assert result["converged_phase2"] is True
    assert "convergence_reason" in result
    assert "b-class" in result["convergence_reason"].lower()


def test_phase2_force_stop_node():
    state = make_initial_state()
    state["a_class_count"] = 20
    state["prev_a_class_count"] = -1
    state["phase2_iteration"] = _config.MAX_ITER_PHASE2
    route_after_phase2_main(state)
    result = phase2_force_stop_node(state)
    assert result["force_stopped"] is True
    assert "convergence_reason" in result
    assert "MAX_ITER" in result["convergence_reason"]


def test_phase1_next_iter_bumps():
    state = make_initial_state()
    state["phase1_iteration"] = 3
    result = phase1_next_iter_node(state)
    assert result["phase1_iteration"] == 4


def test_phase2_next_iter_bumps():
    state = make_initial_state()
    state["phase2_iteration"] = 5
    state["a_class_count"] = 15
    state["diff_report"] = {"b_class_diffs": [{}] * 3}
    result = phase2_next_iter_node(state)
    assert result["phase2_iteration"] == 6
    assert result["prev_a_class_count"] == 15
    assert result["prev_b_class_count"] == 3


def test_route_phase2_b_class_focus_converged():
    """In B-class focus mode, B-class stable → final convergence."""
    state = make_initial_state()
    state["b_class_focus"] = True
    state["b_class_focus_iteration"] = 2
    state["phase2_iteration"] = 5
    state["prev_b_class_count"] = 7
    state["diff_report"] = {"b_class_diffs": [{}] * 7}  # 7 B-class
    state["iteration_history"] = [
        {"iteration": 4, "a_class_count": 0, "b_class_count": 7, "logic_diff_rate": 0.5},
        {"iteration": 5, "a_class_count": 0, "b_class_count": 7, "logic_diff_rate": 0.5},
    ]
    assert route_after_phase2_main(state) == "phase2_converged"
    assert "b-class" in _graph_module._last_p2_convergence_reason.lower()
    assert "converged" in _graph_module._last_p2_convergence_reason.lower()


def test_route_phase2_b_class_focus_max_iter():
    """In B-class focus mode, max iterations → final convergence."""
    state = make_initial_state()
    state["b_class_focus"] = True
    state["b_class_focus_iteration"] = _config.MAX_ITER_B_CLASS
    state["phase2_iteration"] = 8
    state["prev_b_class_count"] = 5
    state["diff_report"] = {"b_class_diffs": [{}] * 10}  # changed
    assert route_after_phase2_main(state) == "phase2_converged"


def test_phase2_next_iter_b_class_focus_bumps():
    """In B-class focus mode, phase2_next_iter also bumps b_class_focus_iteration."""
    state = make_initial_state()
    state["phase2_iteration"] = 5
    state["a_class_count"] = 0
    state["b_class_focus"] = True
    state["b_class_focus_iteration"] = 1
    state["diff_report"] = {"b_class_diffs": [{}] * 3}
    result = phase2_next_iter_node(state)
    assert result["phase2_iteration"] == 6
    assert result["b_class_focus_iteration"] == 2
    assert result["prev_b_class_count"] == 3


def test_phase2_enter_b_class_focus_node():
    """phase2_enter_b_class_focus_node activates B-class focus mode."""
    state = make_initial_state()
    state["phase2_iteration"] = 3
    state["a_class_count"] = 0
    state["diff_report"] = {"b_class_diffs": [{}] * 7}
    result = phase2_enter_b_class_focus_node(state)
    assert result["b_class_focus"] is True
    assert result["b_class_focus_iteration"] == 1
    assert result["prev_b_class_count"] == 7
    assert result["phase2_iteration"] == 4


# ── End-to-end pipeline ────────────────────────────────────────────────

@pytest.mark.timeout(30)
def test_full_pipeline_converges():
    """Mock pipeline runs preprocess → Phase 1 → Phase 2 and converges."""
    configure_graph(mock=True)
    app = compile_graph()
    initial = make_initial_state()
    final = app.invoke(initial)

    assert final is not None
    assert final.get("preprocess_done") is True
    assert final.get("converged_phase2") is True
    # Verify all 5 clients have LSGs
    assert len(final.get("client_lsgs", {})) == 5


@pytest.mark.timeout(30)
def test_force_stop_pipeline():
    """Force stop triggers when convergence threshold is unreachable."""
    saved = (_config.MAX_ITER_PHASE1, _config.CONVERGENCE_THRESHOLD)
    try:
        _config.MAX_ITER_PHASE1 = 1
        _config.CONVERGENCE_THRESHOLD = 0.0  # never converge

        configure_graph(mock=True)
        app = compile_graph()
        initial = make_initial_state()
        initial["diff_rate"] = 1.0

        final = app.invoke(initial)

        assert final is not None
        assert final.get("force_stopped") is True
    finally:
        _config.MAX_ITER_PHASE1, _config.CONVERGENCE_THRESHOLD = saved
        configure_graph()


# ── configure_graph / get_graph_config ─────────────────────────────────

def test_configure_graph_defaults():
    """Default config is mock=True, llm=None."""
    configure_graph()
    cfg = get_graph_config()
    assert cfg["mock"] is True
    assert cfg["llm"] is None
    assert cfg["callbacks"] is None


def test_configure_graph_custom():
    """configure_graph stores custom settings."""
    sentinel_llm = object()
    sentinel_cb = [object()]
    configure_graph(llm=sentinel_llm, mock=False, callbacks=sentinel_cb)
    cfg = get_graph_config()
    assert cfg["llm"] is sentinel_llm
    assert cfg["mock"] is False
    assert cfg["callbacks"] is sentinel_cb
    # Restore defaults
    configure_graph()


def test_get_graph_config_returns_copy():
    """get_graph_config returns a copy — mutations don't leak."""
    configure_graph()
    cfg = get_graph_config()
    cfg["mock"] = "TAMPERED"
    assert _graph_config["mock"] is True
    configure_graph()


# ── Preprocess node mock vs live ───────────────────────────────────────

def test_preprocess_node_mock_mode():
    """In mock mode (default), preprocess_node returns synthetic statuses."""
    configure_graph(mock=True)
    state = make_initial_state()
    result = preprocess_node(state)
    assert result["preprocess_done"] is True
    for client in _config.CLIENT_NAMES:
        assert result["preprocess_status"][client]["symbols_ready"] is True
    configure_graph()


# ── Per-iteration checkpointing ────────────────────────────────────────

def test_phase1_main_saves_checkpoint(tmp_path, monkeypatch):
    """phase1_main_agent_node writes a per-iteration checkpoint file."""
    monkeypatch.setattr(_config, "CHECKPOINT_PATH", tmp_path)
    configure_graph(mock=True)

    state = make_initial_state()
    state["phase1_iteration"] = 2
    state["discovery_reports"] = [
        {"client_name": "prysm", "new_guards": [
            {"name": "G1", "category": "net", "description": "d"}
        ], "new_actions": []},
    ]

    phase1_main_agent_node(state)

    ckpt_files = list(tmp_path.glob("checkpoint_phase1_iter2.json"))
    assert len(ckpt_files) == 1
    configure_graph()


def test_phase2_main_saves_checkpoint(tmp_path, monkeypatch):
    """phase2_main_agent_node writes a per-iteration checkpoint file."""
    monkeypatch.setattr(_config, "CHECKPOINT_PATH", tmp_path)
    configure_graph(mock=True)

    state = make_initial_state()
    state["phase2_iteration"] = 3
    # Build mock LSGs for all clients
    lsgs = {}
    for client in _config.CLIENT_NAMES:
        s = {**state, "_client_name": client}
        r = phase2_sub_agent_node(s)
        lsgs.update(r["client_lsgs"])
    state["client_lsgs"] = lsgs

    phase2_main_agent_node(state)

    ckpt_files = list(tmp_path.glob("checkpoint_phase2_iter3.json"))
    assert len(ckpt_files) == 1
    configure_graph()


# ── Intermediate LSG writing ───────────────────────────────────────────

def test_phase2_sub_writes_intermediate_lsg(tmp_path, monkeypatch):
    """phase2_sub_agent_node writes an intermediate LSG YAML."""
    monkeypatch.setattr(_config, "ITERATIONS_PATH", tmp_path)
    configure_graph(mock=True)

    state = make_initial_state()
    state["_client_name"] = "prysm"
    state["phase2_iteration"] = 4

    phase2_sub_agent_node(state)

    iter_files = list(tmp_path.glob("LSG_prysm_iter4.yaml"))
    assert len(iter_files) == 1
    configure_graph()
