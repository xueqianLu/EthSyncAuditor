"""Unit test for the Phase 2 Main Agent comparison logic."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.phase2_main_agent import (
    _jaccard, _next_cat, _transition_similarity,
    _make_rename_description, _deterministic_compare,
    _classify_severity,
)


def test_jaccard():
    assert _jaccard(set(), set()) == 1.0
    assert _jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert abs(_jaccard({"a", "b"}, {"b", "c"}) - 1/3) < 1e-9
    assert _jaccard({"a"}, {"b"}) == 0.0
    print("  OK: _jaccard")


def test_next_cat():
    assert _next_cat("initial.peer_select") == "peer_select"
    assert _next_cat("done") == "done"
    assert _next_cat("exec.call_new_payload") == "call_new_payload"
    print("  OK: _next_cat")


def test_similarity_same_guard_diff_action():
    """A2: same guard, different action names, same destination → A-class."""
    s = _transition_similarity(
        "RespRecv", frozenset(["ValidateBatch"]), "import",
        "RespRecv", frozenset(["VerifyBlocks"]),  "import",
    )
    print(f"  A2 score: {s:.3f}")
    assert s >= 0.45, f"Expected A-class matchable, got {s}"


def test_similarity_diff_guard_same_action():
    """A1: different guard name, same actions, same destination → A-class."""
    s = _transition_similarity(
        "RespRecv",         frozenset(["ValidateBatch"]), "import",
        "ResponseReceived", frozenset(["ValidateBatch"]), "import",
    )
    print(f"  A1 score: {s:.3f}")
    assert s >= 0.45, f"Expected A-class matchable, got {s}"


def test_similarity_all_different():
    """Completely different transition → not matchable."""
    s = _transition_similarity(
        "RespRecv",      frozenset(["ValidateBatch"]), "import",
        "TimeoutExpired", frozenset(["PenalizePeer"]),  "peer_select",
    )
    print(f"  All-diff score: {s:.3f}")
    assert s < 0.45, f"Expected B-class, got {s}"


def test_rename_description():
    desc = _make_rename_description(
        "lighthouse",
        "RespRecv", "ResponseReceived",
        frozenset(["ValidateBatch"]), frozenset(["VerifyBlocks"]),
    )
    print(f"  desc: {desc}")
    assert "ResponseReceived" in desc
    assert "RespRecv" in desc
    assert "VerifyBlocks" in desc
    assert "ValidateBatch" in desc


def test_full_comparison():
    """End-to-end test with 3 clients having vocabulary + structural diffs."""
    client_lsgs = {
        "prysm": {"workflows": [{"id": "initial_sync", "states": [
            {"id": "initial.validate", "category": "validate", "transitions": [
                {"guard": "RespRecv", "actions": ["ValidateBatch"], "next_state": "initial.import"},
                {"guard": "TimeoutExpired", "actions": ["PenalizePeer"], "next_state": "initial.peer_select"},
            ]},
            {"id": "initial.import", "category": "import", "transitions": [
                {"guard": "TRUE", "actions": ["ApplyBatch", "UpdateForkChoice"], "next_state": "initial.progress"},
            ]},
            {"id": "initial.progress", "category": "progress", "transitions": []},
        ]}]},
        "lighthouse": {"workflows": [{"id": "initial_sync", "states": [
            {"id": "initial.verify", "category": "validate", "transitions": [
                {"guard": "ResponseReceived", "actions": ["VerifyBlocks"], "next_state": "initial.apply"},
                {"guard": "RequestTimeout", "actions": ["ScorePeer"], "next_state": "initial.select"},
            ]},
            {"id": "initial.apply", "category": "import", "transitions": [
                {"guard": "TRUE", "actions": ["ImportBlocks", "UpdateForkChoice"], "next_state": "initial.check"},
            ]},
            {"id": "initial.check", "category": "progress", "transitions": []},
        ]}]},
        "grandine": {"workflows": [{"id": "initial_sync", "states": [
            {"id": "initial.validate", "category": "validate", "transitions": [
                {"guard": "RespRecv", "actions": ["ValidateBatch"], "next_state": "initial.import"},
                # ← Missing TimeoutExpired → should be B-class
            ]},
            {"id": "initial.import", "category": "import", "transitions": [
                {"guard": "TRUE", "actions": ["ApplyBatch", "UpdateForkChoice"], "next_state": "initial.done"},
            ]},
            {"id": "initial.done", "category": "terminal", "transitions": []},
        ]}]},
    }

    result = _deterministic_compare(client_lsgs, [])
    dr = result["diff_report"]

    print(f"\n  A-class ({len(dr['a_class_diffs'])}):")
    for d in dr["a_class_diffs"]:
        print(f"    {d['description']}")
    print(f"  B-class ({len(dr['b_class_diffs'])}):")
    for d in dr["b_class_diffs"]:
        print(f"    [{d.get('severity', '?')}] {d['description']}")
    print(f"  logic_diff_rate: {dr['logic_diff_rate']:.2f}")
    print(f"  total_transitions: {dr['total_transitions']}")

    # lighthouse should have A-class diffs (vocabulary renames)
    a_descs = " ".join(d["description"] for d in dr["a_class_diffs"])
    assert "lighthouse" in a_descs, f"Expected A-class for lighthouse, got: {a_descs}"

    # grandine missing TimeoutExpired should be B-class
    b_descs = " ".join(d["description"] for d in dr["b_class_diffs"])
    assert "grandine" in b_descs, f"Expected B-class for grandine, got: {b_descs}"

    # B-class diffs should have severity assigned
    for d in dr["b_class_diffs"]:
        assert d.get("severity") in ("CRITICAL", "MAJOR", "MINOR"), \
            f"Expected severity, got: {d.get('severity')}"

    # B-class involved_clients should include both ref and non-ref client
    for d in dr["b_class_diffs"]:
        assert len(d.get("involved_clients", [])) >= 2, \
            f"Expected both clients in involved_clients, got: {d.get('involved_clients')}"

    # B-class diffs should have deviating_clients populated
    for d in dr["b_class_diffs"]:
        assert "deviating_clients" in d, f"Missing deviating_clients: {d}"
        assert len(d["deviating_clients"]) >= 1, \
            f"Expected at least one deviating client, got: {d['deviating_clients']}"
        # deviating clients should be subset of involved clients
        for dc in d["deviating_clients"]:
            assert dc in d["involved_clients"], \
                f"Deviating client {dc} not in involved_clients {d['involved_clients']}"

    # total_transitions should be present and positive
    assert dr["total_transitions"] > 0

    print("  OK: full comparison")


def test_classify_severity():
    """Test severity classification of B-class diffs."""
    # CRITICAL: stub workflow
    assert _classify_severity({
        "state_id": "initial_sync.*",
        "description": "Workflow is substantive in X but only a stub in Y",
    }) == "CRITICAL"

    # CRITICAL: missing state category
    assert _classify_severity({
        "state_id": "initial_sync.validate",
        "description": "State category `validate` exists in X but is missing in Y",
    }) == "CRITICAL"

    # MAJOR: transition present in one but not the other (security: missing guard)
    assert _classify_severity({
        "state_id": "initial_sync.validate",
        "description": "Transition present in prysm but no equivalent in lighthouse",
    }) == "MAJOR"

    # MINOR: default for non-security-relevant differences
    assert _classify_severity({
        "state_id": "initial_sync.validate",
        "description": "Different validation approach between prysm and lighthouse",
    }) == "MINOR"

    print("  OK: _classify_severity")


if __name__ == "__main__":
    print("Running comparison logic tests...")
    test_jaccard()
    test_next_cat()
    test_similarity_same_guard_diff_action()
    test_similarity_diff_guard_same_action()
    test_similarity_all_different()
    test_rename_description()
    test_full_comparison()
    test_classify_severity()
    print("\n=== ALL TESTS PASSED ===")

