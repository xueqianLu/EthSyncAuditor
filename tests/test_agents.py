"""Tests for agent factory functions."""

import config as _config


# ── Phase 1 Sub-Agent ──────────────────────────────────────────────────

class TestPhase1SubAgent:
    def test_build_returns_callable(self):
        from agents.phase1_sub_agent import build_phase1_sub_agent

        agent_fn = build_phase1_sub_agent("prysm", llm=None)
        assert callable(agent_fn)

    def test_mock_returns_discovery_report(self):
        from agents.phase1_sub_agent import build_phase1_sub_agent

        agent_fn = build_phase1_sub_agent("prysm", llm=None)
        result = agent_fn({
            "guards": [], "actions": [], "vocab_version": 0,
        })
        assert "discovery_reports" in result
        reports = result["discovery_reports"]
        assert len(reports) == 1
        assert reports[0]["client_name"] == "prysm"

    def test_each_client_works(self):
        from agents.phase1_sub_agent import build_phase1_sub_agent

        for client in _config.CLIENT_NAMES:
            fn = build_phase1_sub_agent(client, llm=None)
            result = fn({"guards": [], "actions": [], "vocab_version": 0})
            assert result["discovery_reports"][0]["client_name"] == client


# ── Phase 1 Main Agent ────────────────────────────────────────────────

class TestPhase1MainAgent:
    def test_build_returns_callable(self):
        from agents.phase1_main_agent import build_phase1_main_agent

        agent_fn = build_phase1_main_agent(llm=None)
        assert callable(agent_fn)

    def test_merge_deduplicates(self):
        from agents.phase1_main_agent import build_phase1_main_agent

        agent_fn = build_phase1_main_agent(llm=None)
        result = agent_fn({
            "guards": [{"name": "G1", "category": "a", "description": "x"}],
            "actions": [],
            "vocab_version": 0,
            "discovery_reports": [
                {
                    "client_name": "prysm",
                    "new_guards": [
                        {"name": "G1", "category": "a", "description": "x"},  # dup
                        {"name": "G2", "category": "b", "description": "y"},  # new
                    ],
                    "new_actions": [],
                },
            ],
        })
        # G1 is duplicate, G2 is new
        new_guard_names = [g["name"] for g in result["guards"]]
        assert "G2" in new_guard_names
        assert "G1" not in new_guard_names  # already existed

    def test_diff_rate_computation(self):
        from agents.phase1_main_agent import build_phase1_main_agent

        agent_fn = build_phase1_main_agent(llm=None)
        # 0 existing, 2 new → diff_rate = 2/2 = 1.0
        result = agent_fn({
            "guards": [],
            "actions": [],
            "vocab_version": 0,
            "discovery_reports": [
                {
                    "client_name": "prysm",
                    "new_guards": [{"name": "G1", "category": "a", "description": "x"}],
                    "new_actions": [{"name": "A1", "category": "b", "description": "y"}],
                },
            ],
        })
        assert result["diff_rate"] == 1.0

    def test_empty_reports_zero_diff(self):
        from agents.phase1_main_agent import build_phase1_main_agent

        agent_fn = build_phase1_main_agent(llm=None)
        result = agent_fn({
            "guards": [{"name": "G1", "category": "a", "description": "x"}],
            "actions": [],
            "vocab_version": 0,
            "discovery_reports": [],
        })
        assert result["diff_rate"] == 0.0
        assert result["vocab_version"] == 1


# ── Phase 2 Sub-Agent ──────────────────────────────────────────────────

class TestPhase2SubAgent:
    def test_build_returns_callable(self):
        from agents.phase2_sub_agent import build_phase2_sub_agent

        agent_fn = build_phase2_sub_agent("lighthouse", llm=None)
        assert callable(agent_fn)

    def test_mock_returns_single_workflow(self):
        """Per-workflow architecture: mock returns 1 workflow per call."""
        from agents.phase2_sub_agent import build_phase2_sub_agent

        agent_fn = build_phase2_sub_agent("prysm", llm=None)
        result = agent_fn({
            "guards": [], "actions": [], "a_class_feedback": [],
            "current_workflow": "initial_sync", "client_lsgs": {},
            "sparsity_hints": [],
        })

        assert "client_lsgs" in result
        lsg = result["client_lsgs"]["prysm"]
        assert lsg["client"] == "prysm"
        # Mock returns exactly 1 workflow per call
        assert len(lsg["workflows"]) == 1
        wf = lsg["workflows"][0]
        assert wf["id"] == "initial_sync"
        # Each mock workflow has init and done states
        assert len(wf["states"]) == 2
        assert wf["states"][0]["category"] == "init"
        assert wf["states"][1]["category"] == "terminal"

    def test_each_workflow_can_be_extracted(self):
        """Each workflow ID can be extracted individually."""
        from agents.phase2_sub_agent import build_phase2_sub_agent

        agent_fn = build_phase2_sub_agent("teku", llm=None)
        collected_ids = set()
        for wf_id in _config.WORKFLOW_IDS:
            result = agent_fn({
                "guards": [], "actions": [], "a_class_feedback": [],
                "current_workflow": wf_id, "client_lsgs": {},
                "sparsity_hints": [],
            })
            lsg = result["client_lsgs"]["teku"]
            collected_ids.add(lsg["workflows"][-1]["id"])
        assert collected_ids == set(_config.WORKFLOW_IDS)


# ── Phase 2 Main Agent ────────────────────────────────────────────────

class TestPhase2MainAgent:
    def test_build_returns_callable(self):
        from agents.phase2_main_agent import build_phase2_main_agent

        agent_fn = build_phase2_main_agent(llm=None)
        assert callable(agent_fn)

    def test_identical_lsgs_zero_diff(self):
        """All clients with same structure → zero B-class diffs."""
        from agents.phase2_sub_agent import build_phase2_sub_agent
        from agents.phase2_main_agent import build_phase2_main_agent

        lsgs = {}
        for client in _config.CLIENT_NAMES:
            fn = build_phase2_sub_agent(client, llm=None)
            r = fn({
                "guards": [], "actions": [], "a_class_feedback": [],
                "current_workflow": "initial_sync", "client_lsgs": {},
                "sparsity_hints": [],
            })
            lsgs.update(r["client_lsgs"])

        main_fn = build_phase2_main_agent(llm=None)
        result = main_fn({
            "client_lsgs": lsgs,
            "current_workflow": "initial_sync",
            "guards": [], "actions": [],
        })

        assert result["logic_diff_rate"] == 0.0
        assert len(result["diff_report"]["b_class_diffs"]) == 0

    def test_missing_client_creates_b_diff(self):
        """If one client is missing a transition, B-class diff is produced."""
        from agents.phase2_sub_agent import build_phase2_sub_agent
        from agents.phase2_main_agent import build_phase2_main_agent

        lsgs = {}
        # Only add 4 of 5 clients with mock workflow
        for client in _config.CLIENT_NAMES[:4]:
            fn = build_phase2_sub_agent(client, llm=None)
            r = fn({
                "guards": [], "actions": [], "a_class_feedback": [],
                "current_workflow": "initial_sync", "client_lsgs": {},
                "sparsity_hints": [],
            })
            lsgs.update(r["client_lsgs"])

        # Add 5th client with empty workflows (no matching transitions)
        lsgs["lodestar"] = {
            "version": 1, "client": "lodestar",
            "workflows": [],
            "guards": [], "actions": [],
        }

        main_fn = build_phase2_main_agent(llm=None)
        result = main_fn({
            "client_lsgs": lsgs,
            "current_workflow": "initial_sync",
            "guards": [], "actions": [],
        })

        # lodestar is missing the workflow entirely → B-class diffs
        assert result["logic_diff_rate"] > 0.0
        assert len(result["diff_report"]["b_class_diffs"]) > 0
