"""Offline preprocessing pipeline for EthAuditor.

Implements:
- AST symbol extraction (tree-sitter)
- callgraph construction + workflow entry point inference
- vector index building (Chroma persistent collections)
- BM25 index building (identifier tokenization)
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass
import importlib
import json
import os
from pathlib import Path
import pickle
import re
import shutil
from typing import Any, Iterable
from functools import lru_cache

from config import CLIENT_NAMES, ENTRY_POINT_OVERRIDES, PREPROCESS_PATH, WORKFLOW_IDS

ROOT_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT_DIR / "code"
PREPROCESS_DIR = ROOT_DIR / PREPROCESS_PATH.replace("./", "")


LANGUAGE_GRAMMARS: dict[str, tuple[str, str]] = {
    "prysm": ("go", "tree-sitter-go"),
    "lighthouse": ("rust", "tree-sitter-rust"),
    "grandine": ("rust", "tree-sitter-rust"),
    "teku": ("java", "tree-sitter-java"),
    "lodestar": ("typescript", "tree-sitter-typescript"),
}

LANGUAGE_FILE_EXTS: dict[str, tuple[str, ...]] = {
    "go": (".go",),
    "rust": (".rs",),
    "java": (".java",),
    "typescript": (".ts", ".tsx", ".js", ".mjs", ".cjs"),
}

LANGUAGE_FUNCTION_NODES: dict[str, tuple[str, ...]] = {
    "go": ("function_declaration", "method_declaration"),
    "rust": ("function_item",),
    "java": ("method_declaration",),
    "typescript": ("function_declaration", "method_definition"),
}

LANGUAGE_CALL_NODES: dict[str, tuple[str, ...]] = {
    "go": ("call_expression",),
    "rust": ("call_expression", "method_call_expression"),
    "java": ("method_invocation",),
    "typescript": ("call_expression",),
}

WORKFLOW_KEYWORDS: dict[str, list[str]] = {
    "initial_sync": ["initialsync", "runinitial", "startinitial"],
    "regular_sync": ["regularsync", "runregular", "gossipsync"],
    "checkpoint_sync": ["checkpointsync", "runcheckpoint"],
    "block_generate": ["proposeblock", "buildblock", "produceblock"],
    "attestation_generate": ["submitattestation", "createattestation"],
    "aggregate": ["aggregate", "computeaggregate"],
    "execute_layer_relation": ["engineapi", "executionengine", "forkchoiceupdate"],
}

WORKFLOW_FALLBACK_KEYWORDS: dict[str, list[str]] = {
    "initial_sync": ["initial", "sync", "start", "run"],
    "regular_sync": ["regular", "sync", "gossip", "run"],
    "checkpoint_sync": ["checkpoint", "sync"],
    "block_generate": ["block", "propose", "build", "produce"],
    "attestation_generate": ["attestation", "attest", "submit", "create"],
    "aggregate": ["aggregate", "committee"],
    "execute_layer_relation": ["engine", "execution", "forkchoice", "el"],
}


@dataclass(slots=True)
class SymbolInfo:
    file_path: str
    function_name: str
    qualified_name: str
    start_line: int
    end_line: int
    source_code: str
    calls: list[str]
    called_by: list[str]


@dataclass(slots=True)
class CallEdge:
    caller: str
    callee: str


@dataclass(slots=True)
class CallGraph:
    nodes: list[str]
    edges: list[CallEdge]
    entry_points: dict[str, list[str]]


def _ensure_dependencies() -> None:
    try:
        importlib.import_module("tree_sitter")
        importlib.import_module("rank_bm25")
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "Missing preprocessing dependencies. Install at least: tree-sitter, rank-bm25"
        ) from exc


def _get_language_and_parser(language_name: str):
    tree_sitter_mod = importlib.import_module("tree_sitter")
    Language = tree_sitter_mod.Language
    Parser = tree_sitter_mod.Parser

    parser = Parser()

    # Prefer dedicated grammar packages.
    if language_name == "go":
        tree_sitter_go = importlib.import_module("tree_sitter_go")

        language = Language(tree_sitter_go.language())
    elif language_name == "rust":
        tree_sitter_rust = importlib.import_module("tree_sitter_rust")

        language = Language(tree_sitter_rust.language())
    elif language_name == "java":
        tree_sitter_java = importlib.import_module("tree_sitter_java")

        language = Language(tree_sitter_java.language())
    elif language_name == "typescript":
        tree_sitter_typescript = importlib.import_module("tree_sitter_typescript")

        language = Language(tree_sitter_typescript.language_typescript())
    else:  # pragma: no cover - protected by caller mapping
        raise ValueError(f"Unsupported language: {language_name}")

    parser.language = language
    return language, parser


def _client_code_dir(client_name: str) -> Path:
    if client_name not in CLIENT_NAMES:
        raise ValueError(f"Unsupported client_name: {client_name}")
    return CODE_DIR / client_name


def _iter_source_files(client_name: str, language_name: str) -> Iterable[Path]:
    base = _client_code_dir(client_name)
    exts = LANGUAGE_FILE_EXTS[language_name]
    max_files_raw = os.getenv("ETHAUDITOR_MAX_SOURCE_FILES", "").strip()
    max_files = int(max_files_raw) if max_files_raw.isdigit() else None
    seen = 0

    for path in base.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts:
            if max_files is not None and seen >= max_files:
                break
            seen += 1
            yield path


def _node_text(node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")


def _find_first_identifier(node, source: bytes) -> str:
    queue = deque([node])
    while queue:
        cur = queue.popleft()
        if cur.type == "identifier":
            text = _node_text(cur, source).strip()
            if text:
                return text
        for child in cur.children:
            queue.append(child)
    return "anonymous"


def _extract_call_name(call_node, source: bytes, language_name: str) -> str | None:
    text = _node_text(call_node, source).strip()
    if not text:
        return None

    if language_name == "java":
        m = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", text)
        return m.group(1) if m else None

    m = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", text)
    if m:
        return m.group(1)

    if language_name == "rust":
        m2 = re.search(r"\.([A-Za-z_][A-Za-z0-9_]*)\s*\(", text)
        if m2:
            return m2.group(1)

    return None


def _collect_calls(func_node, source: bytes, language_name: str) -> list[str]:
    call_types = set(LANGUAGE_CALL_NODES[language_name])
    out: list[str] = []
    queue = deque([func_node])

    while queue:
        cur = queue.popleft()
        if cur.type in call_types:
            call_name = _extract_call_name(cur, source, language_name)
            if call_name:
                out.append(call_name)
        for child in cur.children:
            queue.append(child)

    # Preserve order + dedupe.
    seen = set()
    unique: list[str] = []
    for name in out:
        if name not in seen:
            seen.add(name)
            unique.append(name)
    return unique


def _to_relative(path: Path) -> str:
    return str(path.relative_to(ROOT_DIR))


def _extract_receiver_name(func_node, source: bytes, language_name: str) -> str | None:
    if language_name not in {"go", "typescript", "java"}:
        return None

    text = _node_text(func_node, source)

    if language_name == "go":
        # e.g. func (s *Service) runInitialSync(...)
        m = re.search(r"func\s*\(([^)]*)\)", text)
        if not m:
            return None
        receiver_block = m.group(1)
        rm = re.search(r"\*?([A-Za-z_][A-Za-z0-9_]*)\s*$", receiver_block.strip())
        return rm.group(1) if rm else None

    if language_name == "typescript":
        # best-effort class receiver from declaration context
        parent = func_node.parent
        while parent is not None:
            if parent.type == "class_declaration":
                class_name = _find_first_identifier(parent, source)
                return class_name
            parent = parent.parent
        return None

    if language_name == "java":
        parent = func_node.parent
        while parent is not None:
            if parent.type in {"class_declaration", "interface_declaration"}:
                class_name = _find_first_identifier(parent, source)
                return class_name
            parent = parent.parent
        return None

    return None


def _build_qualified_name(
    function_name: str,
    receiver_name: str | None,
    language_name: str,
) -> str:
    if receiver_name:
        if language_name == "go":
            return f"(*{receiver_name}).{function_name}"
        return f"{receiver_name}.{function_name}"
    return function_name


def _extract_symbols(client_name: str) -> list[SymbolInfo]:
    """Extract symbols with tree-sitter and persist to preprocess artifacts.

    Output: ./output/preprocess/<client>_symbols.json
    """

    language_name, _grammar_pkg = LANGUAGE_GRAMMARS[client_name]
    _language, parser = _get_language_and_parser(language_name)

    symbols: list[SymbolInfo] = []

    function_nodes = set(LANGUAGE_FUNCTION_NODES[language_name])

    for source_file in _iter_source_files(client_name, language_name):
        source = source_file.read_bytes()
        tree = parser.parse(source)
        root = tree.root_node

        queue = deque([root])
        while queue:
            node = queue.popleft()
            if node.type in function_nodes:
                function_name = _find_first_identifier(node, source)
                receiver_name = _extract_receiver_name(node, source, language_name)
                qualified_name = _build_qualified_name(function_name, receiver_name, language_name)

                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1
                source_code = _node_text(node, source)
                calls = _collect_calls(node, source, language_name)

                symbols.append(
                    SymbolInfo(
                        file_path=_to_relative(source_file),
                        function_name=function_name,
                        qualified_name=qualified_name,
                        start_line=start_line,
                        end_line=end_line,
                        source_code=source_code,
                        calls=calls,
                        called_by=[],
                    )
                )
            for child in node.children:
                queue.append(child)

    # Fill called_by reverse edges using short-name matching.
    by_short: dict[str, list[str]] = defaultdict(list)
    for s in symbols:
        by_short[s.function_name].append(s.qualified_name)

    called_by_map: dict[str, set[str]] = defaultdict(set)
    for s in symbols:
        for callee in s.calls:
            targets = by_short.get(callee, [])
            for target in targets:
                called_by_map[target].add(s.qualified_name)

    for i, s in enumerate(symbols):
        symbols[i].called_by = sorted(called_by_map.get(s.qualified_name, set()))

    PREPROCESS_DIR.mkdir(parents=True, exist_ok=True)
    symbols_path = PREPROCESS_DIR / f"{client_name}_symbols.json"
    symbols_path.write_text(
        json.dumps([asdict(s) for s in symbols], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return symbols


def _normalized_for_match(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _infer_entry_points(symbols: list[SymbolInfo], client_name: str) -> dict[str, list[str]]:
    by_workflow: dict[str, list[str]] = {wf: [] for wf in WORKFLOW_IDS}

    for sym in symbols:
        normalized_q = _normalized_for_match(sym.qualified_name)
        normalized_fn = _normalized_for_match(sym.function_name)

        for workflow_id, keywords in WORKFLOW_KEYWORDS.items():
            if any(k in normalized_q or k in normalized_fn for k in keywords):
                by_workflow[workflow_id].append(sym.qualified_name)

    # Manual overrides from config take precedence and are additive.
    overrides = ENTRY_POINT_OVERRIDES.get(client_name, {})
    for workflow_id, values in overrides.items():
        by_workflow.setdefault(workflow_id, [])
        by_workflow[workflow_id].extend(values)

    for workflow_id in by_workflow:
        # stable unique list
        seen = set()
        uniq = []
        for item in by_workflow[workflow_id]:
            if item not in seen:
                seen.add(item)
                uniq.append(item)
        by_workflow[workflow_id] = uniq

    # Fallback when strict keywords miss due naming variance.
    for workflow_id, entries in by_workflow.items():
        if entries:
            continue
        broad = WORKFLOW_FALLBACK_KEYWORDS.get(workflow_id, [])
        for sym in symbols:
            normalized = _normalized_for_match(sym.qualified_name)
            hits = sum(1 for k in broad if k in normalized)
            if hits >= 2:
                by_workflow[workflow_id].append(sym.qualified_name)
        # final dedupe
        seen = set()
        uniq = []
        for item in by_workflow[workflow_id]:
            if item not in seen:
                seen.add(item)
                uniq.append(item)
        by_workflow[workflow_id] = uniq[:50]

    return by_workflow


def _build_callgraph(client_name: str, symbols: list[SymbolInfo]) -> CallGraph:
    """Build callgraph and persist to preprocess artifacts.

    Output: ./output/preprocess/<client>_callgraph.json
    """

    by_short: dict[str, list[str]] = defaultdict(list)
    nodes = []

    for s in symbols:
        nodes.append(s.qualified_name)
        by_short[s.function_name].append(s.qualified_name)

    edges: list[CallEdge] = []
    seen_edges: set[tuple[str, str]] = set()

    for s in symbols:
        for callee_name in s.calls:
            for target in by_short.get(callee_name, []):
                pair = (s.qualified_name, target)
                if pair in seen_edges:
                    continue
                seen_edges.add(pair)
                edges.append(CallEdge(caller=s.qualified_name, callee=target))

    entry_points = _infer_entry_points(symbols, client_name)

    graph = CallGraph(
        nodes=sorted(set(nodes)),
        edges=edges,
        entry_points=entry_points,
    )

    payload = {
        "nodes": graph.nodes,
        "edges": [asdict(e) for e in graph.edges],
        "entry_points": graph.entry_points,
    }
    callgraph_path = PREPROCESS_DIR / f"{client_name}_callgraph.json"
    callgraph_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return graph


def _identifier_tokens(text: str) -> list[str]:
    raw = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)
    out: list[str] = []
    for token in raw:
        out.append(token)
        snake_parts = [p for p in token.split("_") if p]
        out.extend(snake_parts)
        camel_parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", token)
        out.extend(camel_parts)
    return out


def _compute_depths(callgraph: CallGraph) -> dict[str, dict[str, int]]:
    adj: dict[str, list[str]] = defaultdict(list)
    for e in callgraph.edges:
        adj[e.caller].append(e.callee)

    all_depths: dict[str, dict[str, int]] = {}
    for workflow_id, entries in callgraph.entry_points.items():
        depth = {n: 999 for n in callgraph.nodes}
        dq = deque()

        for ep in entries:
            if ep in depth:
                depth[ep] = 0
                dq.append(ep)

        while dq:
            cur = dq.popleft()
            for nxt in adj.get(cur, []):
                if depth[nxt] > depth[cur] + 1:
                    depth[nxt] = depth[cur] + 1
                    dq.append(nxt)

        all_depths[workflow_id] = depth

    return all_depths


@lru_cache(maxsize=1)
def _choose_embedding_model():
    # Priority:
    # 1) nomic-embed-code (Ollama)
    # 2) text-embedding-3-large (OpenAI)
    # 3) all-MiniLM-L6-v2 (HF)
    try:
        OllamaEmbeddings = importlib.import_module("langchain_ollama").OllamaEmbeddings

        return OllamaEmbeddings(model="nomic-embed-code")
    except Exception:
        pass

    try:
        if os.getenv("OPENAI_API_KEY"):
            OpenAIEmbeddings = importlib.import_module("langchain_openai").OpenAIEmbeddings

            return OpenAIEmbeddings(model="text-embedding-3-large")
    except Exception:
        pass

    # Prefer the new provider package to avoid deprecation warnings from
    # langchain_community.HuggingFaceEmbeddings.
    try:
        HuggingFaceEmbeddings = importlib.import_module(
            "langchain_huggingface"
        ).HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        pass

    # Backward-compatible fallback for environments that don't have
    # langchain-huggingface installed yet.
    try:
        HuggingFaceEmbeddings = importlib.import_module(
            "langchain_community.embeddings"
        ).HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        # Offline/CI fallback: deterministic fake embeddings to keep pipeline functional.
        FakeEmbeddings = importlib.import_module("langchain_community.embeddings").FakeEmbeddings
        return FakeEmbeddings(size=384)


def _language_for_text_splitter(language_name: str):
    Language = importlib.import_module("langchain_text_splitters").Language

    mapping = {
        "go": Language.GO,
        "rust": Language.RUST,
        "java": Language.JAVA,
        "typescript": Language.TS,
    }
    return mapping[language_name]


def _line_offsets(text: str) -> list[int]:
    offsets = [0]
    for idx, ch in enumerate(text):
        if ch == "\n":
            offsets.append(idx + 1)
    return offsets


def _byte_to_line(text: str, byte_offset: int) -> int:
    # byte_offset here is char offset for UTF-8-safe chunks from python slicing.
    # best effort for metadata usage.
    return text.count("\n", 0, max(0, byte_offset)) + 1


def _build_vector_index(client_name: str, symbols: list[SymbolInfo], callgraph: CallGraph):
    """Build persistent Chroma vector index with callgraph-enhanced metadata.

    Output: ./output/preprocess/<client>_chroma/
    """

    Chroma = importlib.import_module("langchain_chroma").Chroma
    Document = importlib.import_module("langchain_core.documents").Document
    RecursiveCharacterTextSplitter = importlib.import_module(
        "langchain_text_splitters"
    ).RecursiveCharacterTextSplitter

    language_name = LANGUAGE_GRAMMARS[client_name][0]
    lang_enum = _language_for_text_splitter(language_name)

    splitter = RecursiveCharacterTextSplitter.from_language(
        language=lang_enum,
        chunk_size=1200,
        chunk_overlap=120,
    )

    depths_by_workflow = _compute_depths(callgraph)

    docs: list[Any] = []

    for s in symbols:
        split_texts = splitter.split_text(s.source_code)
        cursor = 0

        callers = s.called_by
        callees = s.calls

        workflow_hints: list[str] = []
        nearest_depth = 999

        for wf in WORKFLOW_IDS:
            depth = depths_by_workflow.get(wf, {}).get(s.qualified_name, 999)
            if depth < 999:
                workflow_hints.append(wf)
            if depth < nearest_depth:
                nearest_depth = depth

        if nearest_depth == 999:
            nearest_depth = 999

        for chunk in split_texts:
            # best-effort locate to infer approximate line interval in full file.
            rel_pos = s.source_code.find(chunk, cursor)
            if rel_pos == -1:
                rel_pos = cursor
            cursor = rel_pos + len(chunk)

            chunk_start_line = s.start_line + _byte_to_line(s.source_code, rel_pos) - 1
            chunk_end_line = min(
                s.end_line,
                chunk_start_line + chunk.count("\n"),
            )

            metadata = {
                "client_name": client_name,
                "language": language_name,
                "file_path": s.file_path,
                "function_name": s.function_name,
                "qualified_name": s.qualified_name,
                "start_line": chunk_start_line,
                "end_line": chunk_end_line,
                "call_depth": nearest_depth,
                "workflow_hints": ",".join(workflow_hints) if workflow_hints else "",
                "callers": ",".join(callers) if callers else "",
                "callees": ",".join(callees) if callees else "",
            }
            docs.append(Document(page_content=chunk, metadata=metadata))

    persist_dir = PREPROCESS_DIR / f"{client_name}_chroma"
    if persist_dir.exists():
        shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    embeddings = _choose_embedding_model()

    # Recreate for deterministic behavior under force rebuild path.
    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=client_name,
        persist_directory=str(persist_dir),
    )


def _build_bm25_index(client_name: str, symbols: list[SymbolInfo]):
    """Build and persist BM25 index by identifier tokenization.

    Output: ./output/preprocess/<client>_bm25.pkl
    """

    BM25Okapi = importlib.import_module("rank_bm25").BM25Okapi

    tokenized_corpus: list[list[str]] = []
    payload_docs: list[dict[str, Any]] = []

    for s in symbols:
        tokens = _identifier_tokens(s.source_code)
        tokenized_corpus.append(tokens)
        payload_docs.append(
            {
                "page_content": s.source_code,
                "metadata": {
                    "client_name": client_name,
                    "language": LANGUAGE_GRAMMARS[client_name][0],
                    "file_path": s.file_path,
                    "function_name": s.function_name,
                    "qualified_name": s.qualified_name,
                    "start_line": s.start_line,
                    "end_line": s.end_line,
                    "callers": s.called_by,
                    "callees": s.calls,
                },
            }
        )

    bm25 = BM25Okapi(tokenized_corpus)

    bundle = {
        "tokenized_corpus": tokenized_corpus,
        "documents": payload_docs,
        "bm25": bm25,
    }

    out_path = PREPROCESS_DIR / f"{client_name}_bm25.pkl"
    with out_path.open("wb") as f:
        pickle.dump(bundle, f)


def _artifacts_exist(client_name: str) -> bool:
    return (
        (PREPROCESS_DIR / f"{client_name}_symbols.json").exists()
        and (PREPROCESS_DIR / f"{client_name}_callgraph.json").exists()
        and (PREPROCESS_DIR / f"{client_name}_bm25.pkl").exists()
        and (PREPROCESS_DIR / f"{client_name}_chroma").exists()
    )


def _load_symbols(client_name: str) -> list[SymbolInfo]:
    payload = json.loads((PREPROCESS_DIR / f"{client_name}_symbols.json").read_text(encoding="utf-8"))
    return [SymbolInfo(**item) for item in payload]


def _load_callgraph(client_name: str) -> CallGraph:
    payload = json.loads((PREPROCESS_DIR / f"{client_name}_callgraph.json").read_text(encoding="utf-8"))
    return CallGraph(
        nodes=payload["nodes"],
        edges=[CallEdge(**e) for e in payload["edges"]],
        entry_points=payload["entry_points"],
    )


def run_preprocessing(client_name: str, force_rebuild: bool = False) -> dict[str, Any]:
    """Run full preprocessing pipeline for one client.

    Tasks executed in order:
      A) _extract_symbols
      B) _build_callgraph
      C) _build_vector_index
      D) _build_bm25_index
    """

    if client_name not in CLIENT_NAMES:
        raise ValueError(f"Unsupported client_name: {client_name}")

    PREPROCESS_DIR.mkdir(parents=True, exist_ok=True)

    if _artifacts_exist(client_name) and not force_rebuild:
        return {
            "client_name": client_name,
            "skipped": True,
            "symbols": str(PREPROCESS_DIR / f"{client_name}_symbols.json"),
            "callgraph": str(PREPROCESS_DIR / f"{client_name}_callgraph.json"),
            "bm25": str(PREPROCESS_DIR / f"{client_name}_bm25.pkl"),
            "chroma": str(PREPROCESS_DIR / f"{client_name}_chroma"),
        }

    _ensure_dependencies()

    symbols = _extract_symbols(client_name)
    callgraph = _build_callgraph(client_name, symbols)
    _build_vector_index(client_name, symbols, callgraph)
    _build_bm25_index(client_name, symbols)

    return {
        "client_name": client_name,
        "skipped": False,
        "symbol_count": len(symbols),
        "node_count": len(callgraph.nodes),
        "edge_count": len(callgraph.edges),
        "symbols": str(PREPROCESS_DIR / f"{client_name}_symbols.json"),
        "callgraph": str(PREPROCESS_DIR / f"{client_name}_callgraph.json"),
        "bm25": str(PREPROCESS_DIR / f"{client_name}_bm25.pkl"),
        "chroma": str(PREPROCESS_DIR / f"{client_name}_chroma"),
    }


def load_bm25_bundle(client_name: str) -> dict[str, Any]:
    with (PREPROCESS_DIR / f"{client_name}_bm25.pkl").open("rb") as f:
        return pickle.load(f)


def load_callgraph(client_name: str) -> CallGraph:
    return _load_callgraph(client_name)
