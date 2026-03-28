"""Tests for config.py — configuration constants and mappings."""

import config


def test_iteration_limits():
    """MAX_ITER values are positive integers."""
    assert isinstance(config.MAX_ITER_PHASE1, int) and config.MAX_ITER_PHASE1 > 0
    assert isinstance(config.MAX_ITER_PHASE2, int) and config.MAX_ITER_PHASE2 > 0


def test_convergence_threshold():
    """Threshold is a float in (0, 1)."""
    assert 0 < config.CONVERGENCE_THRESHOLD < 1


def test_client_names():
    """All 5 clients are listed."""
    assert len(config.CLIENT_NAMES) == 5
    for name in ["prysm", "lighthouse", "grandine", "teku", "lodestar"]:
        assert name in config.CLIENT_NAMES


def test_workflow_ids():
    """All 7 reserved workflow IDs are present."""
    assert len(config.WORKFLOW_IDS) == 7
    expected = {
        "initial_sync", "regular_sync", "checkpoint_sync",
        "attestation_generate", "block_generate", "aggregate",
        "execute_layer_relation",
    }
    assert set(config.WORKFLOW_IDS) == expected


def test_language_grammars_cover_all_clients():
    """Every client has a language grammar mapping."""
    for client in config.CLIENT_NAMES:
        assert client in config.LANGUAGE_GRAMMARS
        lang, grammar = config.LANGUAGE_GRAMMARS[client]
        assert isinstance(lang, str)
        assert isinstance(grammar, str)


def test_bm25_vector_weights_sum_to_one():
    """BM25 + Vector weights should sum to 1.0."""
    assert abs(config.BM25_WEIGHT + config.VECTOR_WEIGHT - 1.0) < 1e-9


def test_entry_point_keywords_cover_all_workflows():
    """Every workflow has entry-point keywords."""
    for wf_id in config.WORKFLOW_IDS:
        assert wf_id in config.ENTRY_POINT_KEYWORDS
        assert len(config.ENTRY_POINT_KEYWORDS[wf_id]) > 0


def test_paths_are_path_objects():
    """All path configs are pathlib.Path."""
    from pathlib import Path
    assert isinstance(config.PROJECT_ROOT, Path)
    assert isinstance(config.CODE_BASE_PATH, Path)
    assert isinstance(config.OUTPUT_PATH, Path)
    assert isinstance(config.PREPROCESS_PATH, Path)
    assert isinstance(config.SPEC_PATH, Path)


def test_llm_provider():
    """LLM_PROVIDER is a valid provider string."""
    assert config.LLM_PROVIDER in ("anthropic", "gemini")


def test_llm_model_names():
    """LLM_MODEL and GEMINI_MODEL are non-empty strings."""
    assert isinstance(config.LLM_MODEL, str) and len(config.LLM_MODEL) > 0
    assert isinstance(config.GEMINI_MODEL, str) and len(config.GEMINI_MODEL) > 0
