"""Tests for tools modules — preprocessor and search."""

from tools.preprocessor import tokenize_identifier, tokenize_source


# ── Tokenizer ───────────────────────────────────────────────────────────

class TestTokenizeIdentifier:
    def test_camel_case(self):
        result = tokenize_identifier("runInitialSync")
        assert "runInitialSync" in result
        assert "run" in result
        assert "Initial" in result
        assert "Sync" in result

    def test_snake_case(self):
        result = tokenize_identifier("process_chain_segment")
        assert "process_chain_segment" in result
        assert "process" in result
        assert "chain" in result
        assert "segment" in result

    def test_single_word(self):
        result = tokenize_identifier("validate")
        assert result == ["validate"]

    def test_all_caps(self):
        result = tokenize_identifier("HTTP")
        assert "HTTP" in result

    def test_mixed_case_with_acronym(self):
        result = tokenize_identifier("parseHTTPResponse")
        assert "parseHTTPResponse" in result


class TestTokenizeSource:
    def test_basic_source(self):
        source = "func runInitialSync() { fetchBatch() }"
        tokens = tokenize_source(source)
        assert "runInitialSync" in tokens
        assert "fetchBatch" in tokens
        assert "func" in tokens

    def test_empty_source(self):
        assert tokenize_source("") == []

    def test_numeric_identifiers_ignored(self):
        """Pure numbers are not identifiers."""
        tokens = tokenize_source("x = 42")
        assert "42" not in tokens
        assert "x" in tokens


# ── Preprocessor data classes ───────────────────────────────────────────

def test_symbol_info_dataclass():
    from tools.preprocessor import SymbolInfo

    s = SymbolInfo(
        file="sync.go",
        function_name="runSync",
        qualified_name="(*Service).runSync",
        start_line=10,
        end_line=50,
    )
    assert s.calls == []
    assert s.called_by == []
    assert s.source_code == ""


def test_callgraph_dataclass():
    from tools.preprocessor import CallGraph

    cg = CallGraph()
    assert cg.nodes == []
    assert cg.edges == []
    assert cg.entry_points == {}


# ── Search module ───────────────────────────────────────────────────────

def test_search_result_dataclass():
    from tools.search import SearchResult

    sr = SearchResult(content="test code", metadata={"file": "a.go"}, score=0.8)
    assert sr.content == "test code"
    assert sr.score == 0.8


def test_search_codebase_no_index():
    """search_codebase gracefully handles missing indexes."""
    from tools.search import search_codebase

    results = search_codebase("runInitialSync", "prysm", top_k=5)
    assert results == []


def test_search_by_workflow_no_index():
    """search_codebase_by_workflow gracefully handles missing indexes."""
    from tools.search import search_codebase_by_workflow

    results = search_codebase_by_workflow(
        "initial_sync", "runInitialSync", "prysm",
    )
    assert results == []


# ── Preprocessor run_preprocessing ──────────────────────────────────────

def test_run_preprocessing_no_source(tmp_path, monkeypatch):
    """run_preprocessing handles missing source code gracefully."""
    import config as _config
    from tools.preprocessor import run_preprocessing

    monkeypatch.setattr(_config, "CODE_BASE_PATH", tmp_path / "code")
    monkeypatch.setattr(_config, "PREPROCESS_PATH", tmp_path / "preprocess")

    status = run_preprocessing("prysm", force_rebuild=True)
    assert status["symbols_ready"] is True
    assert status["callgraph_ready"] is True
