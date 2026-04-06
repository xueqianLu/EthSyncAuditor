"""Unit test for the Phase 2 Main Agent comparison logic."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.phase2_main_agent import (
    _jaccard, _next_cat, _transition_similarity,
    _make_rename_description, _deterministic_compare,
    _classify_severity, _build_evidence_map,
    _backfill_evidence_from_lsgs, _infer_deviating_clients,
    _extract_vulnerability_patterns, _security_note_denies_impact,
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


def test_build_evidence_map():
    """_build_evidence_map filters None and empty entries."""
    ev = _build_evidence_map({
        "prysm": {"file": "sync.go", "function": "runSync", "lines": [10, 20]},
        "lighthouse": None,
        "grandine": {},
        "teku": {"file": "Sync.java", "function": "doSync", "lines": [5, 15]},
    })
    assert "prysm" in ev
    assert "teku" in ev
    assert "lighthouse" not in ev
    assert "grandine" not in ev
    assert ev["prysm"]["file"] == "sync.go"
    print("  OK: _build_evidence_map")


def test_backfill_evidence_from_lsgs():
    """_backfill_evidence_from_lsgs populates empty evidence from LSG transitions."""
    client_lsgs = {
        "prysm": {"workflows": [{"id": "initial_sync", "states": [
            {"id": "s1", "category": "validate", "transitions": [
                {"guard": "RespRecv", "actions": ["Validate"],
                 "next_state": "s2",
                 "evidence": {"file": "sync.go", "function": "onResp", "lines": [42, 60]}},
            ]},
        ]}]},
    }
    diffs = [
        {
            "workflow_id": "initial_sync",
            "transition_guard": "RespRecv",
            "involved_clients": ["prysm", "lighthouse"],
            "evidence": {},
        },
        {
            "workflow_id": "initial_sync",
            "transition_guard": "UnknownGuard",
            "involved_clients": ["prysm"],
            "evidence": {},
        },
    ]
    _backfill_evidence_from_lsgs(diffs, client_lsgs)

    # First diff should have evidence backfilled for prysm
    assert "prysm" in diffs[0]["evidence"]
    assert diffs[0]["evidence"]["prysm"]["file"] == "sync.go"

    # Second diff has no matching transition → still empty
    assert diffs[1]["evidence"] == {}
    print("  OK: _backfill_evidence_from_lsgs")


def test_backfill_evidence_preserves_existing():
    """_backfill_evidence_from_lsgs skips diffs that already have evidence."""
    client_lsgs = {
        "prysm": {"workflows": [{"id": "wf1", "states": [
            {"id": "s1", "category": "x", "transitions": [
                {"guard": "G1", "actions": [], "next_state": "s2",
                 "evidence": {"file": "new.go", "function": "f", "lines": [1]}},
            ]},
        ]}]},
    }
    original_ev = {"prysm": {"file": "old.go", "function": "g", "lines": [99]}}
    diffs = [{
        "workflow_id": "wf1",
        "transition_guard": "G1",
        "involved_clients": ["prysm"],
        "evidence": dict(original_ev),
    }]
    _backfill_evidence_from_lsgs(diffs, client_lsgs)
    # Should keep the original evidence, not overwrite
    assert diffs[0]["evidence"]["prysm"]["file"] == "old.go"
    print("  OK: _backfill_evidence_preserves_existing")


def test_infer_deviating_clients_contrast():
    """_infer_deviating_clients identifies minority clients from contrast markers."""
    diff = {
        "description": (
            "Prysm and Lighthouse model an explicit stalled recovery state. "
            "However, Grandine and Teku handle stall detection inline."
        ),
        "involved_clients": ["prysm", "lighthouse", "grandine", "teku"],
    }
    result = _infer_deviating_clients(diff)
    # "Prysm and Lighthouse" are before the contrast, "Grandine and Teku" after
    # Both groups have 2 clients — but the after-contrast group deviates
    assert len(result) == 2
    print(f"  Inferred deviating: {result}")
    print("  OK: _infer_deviating_clients (contrast)")


def test_infer_deviating_clients_unique():
    """_infer_deviating_clients identifies single unique client."""
    diff = {
        "description": (
            "Prysm's LSG includes an explicit stalled state. "
            "Other clients do not have an equivalent state."
        ),
        "involved_clients": ["prysm", "lighthouse", "grandine", "teku", "lodestar"],
    }
    result = _infer_deviating_clients(diff)
    assert result == ["prysm"]
    print("  OK: _infer_deviating_clients (unique)")


def test_infer_deviating_clients_already_set():
    """_infer_deviating_clients returns existing value if already set."""
    diff = {
        "description": "Some description",
        "involved_clients": ["prysm", "lighthouse"],
        "deviating_clients": ["prysm"],
    }
    result = _infer_deviating_clients(diff)
    assert result == ["prysm"]
    print("  OK: _infer_deviating_clients (already set)")


def test_infer_deviating_clients_no_marker():
    """_infer_deviating_clients returns empty when no contrast marker found."""
    diff = {
        "description": "All clients handle validation similarly.",
        "involved_clients": ["prysm", "lighthouse", "grandine"],
    }
    result = _infer_deviating_clients(diff)
    assert result == []
    print("  OK: _infer_deviating_clients (no marker)")


def test_evidence_in_deterministic_compare():
    """Evidence from LSG transitions flows into diff entries."""
    client_lsgs = {
        "prysm": {"workflows": [{"id": "initial_sync", "states": [
            {"id": "s1", "category": "validate", "transitions": [
                {"guard": "RespRecv", "actions": ["Validate"],
                 "next_state": "s2",
                 "evidence": {"file": "sync.go", "function": "onResp", "lines": [10, 30]}},
                {"guard": "Timeout", "actions": ["Penalize"],
                 "next_state": "s3",
                 "evidence": {"file": "timeout.go", "function": "onTimeout", "lines": [5, 12]}},
            ]},
        ]}]},
        "lighthouse": {"workflows": [{"id": "initial_sync", "states": [
            {"id": "s1", "category": "validate", "transitions": [
                {"guard": "RespRecv", "actions": ["Validate"],
                 "next_state": "s2",
                 "evidence": {"file": "sync.rs", "function": "on_resp", "lines": [20, 40]}},
                # Missing Timeout → B-class diff
            ]},
        ]}]},
    }
    result = _deterministic_compare(client_lsgs, [])
    dr = result["diff_report"]

    # B-class diff for missing Timeout should have evidence from prysm
    b_timeout = [d for d in dr["b_class_diffs"]
                 if "Timeout" in d.get("transition_guard", "")]
    assert len(b_timeout) >= 1, f"Expected B-class for Timeout, got: {dr['b_class_diffs']}"
    ev = b_timeout[0].get("evidence", {})
    assert "prysm" in ev, f"Expected prysm evidence, got: {ev}"
    assert ev["prysm"]["file"] == "timeout.go"

    print("  OK: evidence in deterministic compare")


def test_severity_downgrade_on_no_security_impact():
    """MAJOR keyword match is downgraded to MINOR when security_note denies impact."""
    # "architectural" matches MAJOR keywords, but security_note says no impact
    assert _classify_severity({
        "state_id": "attestation_generate.N/A",
        "description": "Prysm uses a terminal model, others use cyclic architectural model",
        "security_note": "This is a purely architectural difference with no direct security impact.",
    }) == "MINOR"

    # Same keywords WITHOUT denial → stays MAJOR
    assert _classify_severity({
        "state_id": "attestation_generate.N/A",
        "description": "Prysm uses a terminal model, others use cyclic architectural model",
        "security_note": "Could lead to missed duties under race conditions.",
    }) == "MAJOR"

    print("  OK: severity downgrade on no security impact")


def test_security_note_denies_impact():
    """_security_note_denies_impact detects denial phrases."""
    assert _security_note_denies_impact("no direct security impact") is True
    assert _security_note_denies_impact("purely architectural difference") is True
    assert _security_note_denies_impact("this could lead to eclipse attacks") is False
    assert _security_note_denies_impact("") is False
    print("  OK: _security_note_denies_impact")


def test_extract_vulnerability_patterns():
    """_extract_vulnerability_patterns extracts attack surface categories."""
    b_diffs = [
        {
            "workflow_id": "initial_sync",
            "transition_guard": "BlockBatchDoesNotConnect",
            "description": "Lighthouse bans the peer, Prysm only decreases score",
            "security_note": "eclipse attack surface due to lenient peer penalty",
            "severity": "MAJOR",
            "deviating_clients": ["lighthouse"],
        },
        {
            "workflow_id": "execute_layer_relation",
            "transition_guard": "OptimisticSyncDepthExceedsLimit",
            "description": "Guard missing in Grandine",
            "security_note": "missing safety guard — depth limit absent",
            "severity": "CRITICAL",
            "deviating_clients": ["grandine"],
        },
        {
            "workflow_id": "regular_sync",
            "transition_guard": "BlockIsAwaitingBlobs",
            "description": "Teku rejects block when blobs timeout",
            "security_note": "timeout rejection leads to minority fork",
            "severity": "CRITICAL",
            "deviating_clients": ["teku"],
        },
    ]
    patterns = _extract_vulnerability_patterns(b_diffs)
    categories = {p["category"] for p in patterns}

    assert "peer_penalty_divergence" in categories
    assert "missing_safety_guard" in categories
    assert "timeout_rejection_divergence" in categories
    assert len(patterns) == 3  # one per category

    # Check structure
    peer_pat = [p for p in patterns if p["category"] == "peer_penalty_divergence"][0]
    assert peer_pat["example_workflow"] == "initial_sync"
    assert peer_pat["severity"] == "MAJOR"

    print("  OK: _extract_vulnerability_patterns")


def test_extract_vulnerability_patterns_dedup():
    """_extract_vulnerability_patterns deduplicates by category."""
    b_diffs = [
        {
            "description": "Peer ban difference in initial_sync",
            "security_note": "eclipse via lenient peer penalty",
            "workflow_id": "initial_sync", "transition_guard": "G1",
            "severity": "MAJOR", "deviating_clients": ["prysm"],
        },
        {
            "description": "Peer ban difference in regular_sync",
            "security_note": "peer penalty divergence again",
            "workflow_id": "regular_sync", "transition_guard": "G2",
            "severity": "MAJOR", "deviating_clients": ["prysm"],
        },
    ]
    patterns = _extract_vulnerability_patterns(b_diffs)
    # Both match peer_penalty_divergence → only 1 pattern
    assert len(patterns) == 1
    assert patterns[0]["category"] == "peer_penalty_divergence"
    print("  OK: _extract_vulnerability_patterns dedup")


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
    test_build_evidence_map()
    test_backfill_evidence_from_lsgs()
    test_backfill_evidence_preserves_existing()
    test_infer_deviating_clients_contrast()
    test_infer_deviating_clients_unique()
    test_infer_deviating_clients_already_set()
    test_infer_deviating_clients_no_marker()
    test_evidence_in_deterministic_compare()
    test_severity_downgrade_on_no_security_impact()
    test_security_note_denies_impact()
    test_extract_vulnerability_patterns()
    test_extract_vulnerability_patterns_dedup()
    print("\n=== ALL TESTS PASSED ===")

