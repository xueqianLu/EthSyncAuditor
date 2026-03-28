"""Tests for state.py — Pydantic models and GlobalState."""

from state import (
    DiffItem,
    DiffReport,
    EnrichedSpec,
    Evidence,
    GlobalState,
    LSGFile,
    LSGState,
    LSGWorkflow,
    PreprocessStatus,
    Transition,
    VocabDiscoveryReport,
    VocabEntry,
)


# ── VocabEntry ──────────────────────────────────────────────────────────

def test_vocab_entry_minimal():
    """VocabEntry with required fields only."""
    v = VocabEntry(name="TestGuard", category="network", description="Test")
    assert v.name == "TestGuard"
    assert v.evidence_file is None


def test_vocab_entry_with_evidence():
    """VocabEntry with evidence fields populated."""
    v = VocabEntry(
        name="G1", category="time", description="Timeout",
        evidence_file="sync.go", evidence_function="run", evidence_lines=[10, 20],
    )
    assert v.evidence_lines == [10, 20]


# ── Evidence ────────────────────────────────────────────────────────────

def test_evidence_defaults():
    e = Evidence(file="a.go", function="fn")
    assert e.lines == []


def test_evidence_with_lines():
    e = Evidence(file="a.go", function="fn", lines=[1, 50])
    assert e.lines == [1, 50]


# ── Transition ──────────────────────────────────────────────────────────

def test_transition_minimal():
    t = Transition(guard="TRUE", next_state="s2")
    assert t.actions == []
    assert t.evidence is None


def test_transition_full():
    t = Transition(
        guard="RespRecv",
        actions=["SendRangeRequest"],
        next_state="s3",
        evidence=Evidence(file="sync.go", function="recv", lines=[10, 20]),
    )
    assert t.guard == "RespRecv"
    assert len(t.actions) == 1


# ── LSGState / LSGWorkflow / LSGFile ────────────────────────────────────

def test_lsg_state():
    s = LSGState(id="initial.init", label="Init", category="init")
    assert s.transitions == []


def test_lsg_workflow():
    wf = LSGWorkflow(id="initial_sync", name="Initial Sync")
    assert wf.states == []
    assert wf.initial_state == ""


def test_lsg_file_defaults():
    f = LSGFile()
    assert f.version == 1
    assert f.guards == []
    assert f.workflows == []


def test_lsg_file_full():
    f = LSGFile(
        client="prysm",
        guards=[VocabEntry(name="G1", category="net", description="x")],
        workflows=[LSGWorkflow(id="initial_sync", name="IS")],
    )
    assert f.client == "prysm"
    assert len(f.guards) == 1
    assert len(f.workflows) == 1


# ── DiffItem / DiffReport ──────────────────────────────────────────────

def test_diff_item():
    d = DiffItem(
        workflow_id="initial_sync",
        state_id="s1",
        transition_guard="G1",
        diff_type="B",
        description="Missing",
    )
    assert d.involved_clients == []


def test_diff_report_defaults():
    r = DiffReport()
    assert r.logic_diff_rate == 1.0
    assert r.a_class_diffs == []
    assert r.b_class_diffs == []


# ── VocabDiscoveryReport ────────────────────────────────────────────────

def test_vocab_discovery_report():
    r = VocabDiscoveryReport(client_name="prysm")
    assert r.new_guards == []
    assert r.new_actions == []


# ── EnrichedSpec ────────────────────────────────────────────────────────

def test_enriched_spec_defaults():
    s = EnrichedSpec()
    assert s.version == 1
    assert s.guards == []


# ── PreprocessStatus ────────────────────────────────────────────────────

def test_preprocess_status_not_ready():
    ps = PreprocessStatus()
    assert not ps.all_ready


def test_preprocess_status_all_ready():
    ps = PreprocessStatus(
        symbols_ready=True,
        callgraph_ready=True,
        vector_index_ready=True,
        bm25_index_ready=True,
    )
    assert ps.all_ready


# ── Serialization round-trip ────────────────────────────────────────────

def test_vocab_entry_model_dump():
    v = VocabEntry(name="G1", category="net", description="d")
    d = v.model_dump()
    assert d["name"] == "G1"
    v2 = VocabEntry(**d)
    assert v2 == v


def test_lsg_file_model_dump():
    f = LSGFile(client="teku")
    d = f.model_dump()
    f2 = LSGFile(**d)
    assert f2.client == "teku"
