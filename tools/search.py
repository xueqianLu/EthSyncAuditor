"""Hybrid retrieval tools for EthAuditor.

Provides two tool modes:
- search_codebase: BM25 + Vector hybrid retrieval
- search_codebase_by_workflow: callgraph-constrained hybrid retrieval
"""

from __future__ import annotations

from collections import deque
import importlib
from typing import Any

from langchain_core.documents import Document
from langchain_core.tools import tool

from config import BM25_VECTOR_WEIGHT
from tools.preprocessor import (
    PREPROCESS_DIR,
    WORKFLOW_IDS,
    WORKFLOW_FALLBACK_KEYWORDS,
    _identifier_tokens,
    _choose_embedding_model,
    load_bm25_bundle,
    load_callgraph,
)


def _load_chroma(client_name: str):
    Chroma = importlib.import_module("langchain_chroma").Chroma

    persist_dir = PREPROCESS_DIR / f"{client_name}_chroma"
    if not persist_dir.exists():
        raise FileNotFoundError(
            f"Missing Chroma artifacts for {client_name}. Run preprocessing first."
        )

    embeddings = _choose_embedding_model()
    return Chroma(
        collection_name=client_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )


def _bm25_search(client_name: str, query: str, top_k: int) -> list[Document]:
    bundle = load_bm25_bundle(client_name)
    bm25 = bundle["bm25"]
    docs = bundle["documents"]

    query_tokens = _identifier_tokens(query)
    scores = bm25.get_scores(query_tokens)

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    out: list[Document] = []
    for idx, score in ranked:
        item = docs[idx]
        md = dict(item["metadata"])
        md["_bm25_score"] = float(score)
        out.append(Document(page_content=item["page_content"], metadata=md))
    return out


def _vector_search(client_name: str, query: str, top_k: int) -> list[Document]:
    vectordb = _load_chroma(client_name)
    docs = vectordb.similarity_search(query, k=top_k)

    out: list[Document] = []
    for rank, doc in enumerate(docs, start=1):
        doc.metadata = dict(doc.metadata)
        doc.metadata["_vector_score"] = 1.0 / (50 + rank)
        out.append(doc)
    return out


def _hybrid_fuse(
    bm25_docs: list[Document],
    vector_docs: list[Document],
    top_k: int,
    bm25_weight: float,
    vector_weight: float,
) -> list[Document]:
    # Weighted reciprocal-rank style fusion with deterministic dedupe.
    bucket: dict[str, dict[str, Any]] = {}

    def key_of(d: Document) -> str:
        md = d.metadata
        return (
            f"{md.get('client_name','')}|{md.get('file_path','')}|"
            f"{md.get('function_name','')}|{md.get('start_line','')}|{md.get('end_line','')}"
        )

    for rank, doc in enumerate(bm25_docs, start=1):
        k = key_of(doc)
        entry = bucket.setdefault(k, {"doc": doc, "score": 0.0})
        entry["score"] += bm25_weight * (1.0 / (50 + rank))

    for rank, doc in enumerate(vector_docs, start=1):
        k = key_of(doc)
        entry = bucket.setdefault(k, {"doc": doc, "score": 0.0})
        entry["score"] += vector_weight * (1.0 / (50 + rank))

    ranked = sorted(bucket.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    out: list[Document] = []
    for item in ranked:
        doc = item["doc"]
        md = dict(doc.metadata)
        md["_hybrid_score"] = float(item["score"])
        doc.metadata = md
        out.append(doc)
    return out


def _hybrid_with_ensemble(
    client_name: str,
    query: str,
    top_k: int,
    allowed_nodes: set[str] | None = None,
) -> list[Document]:
    """Preferred hybrid retrieval implementation using EnsembleRetriever.

    Falls back to manual fusion if optional retriever packages are unavailable.
    """

    try:
        EnsembleRetriever = importlib.import_module("langchain.retrievers").EnsembleRetriever
        BM25Retriever = importlib.import_module("langchain_community.retrievers").BM25Retriever

        bundle = load_bm25_bundle(client_name)
        bm25_docs = [
            Document(page_content=item["page_content"], metadata=dict(item["metadata"]))
            for item in bundle["documents"]
        ]
        if allowed_nodes is not None:
            bm25_docs = _filter_docs_by_nodes(bm25_docs, allowed_nodes)

        bm25_retriever = BM25Retriever.from_documents(bm25_docs)
        bm25_retriever.k = top_k

        vectordb = _load_chroma(client_name)
        vector_retriever = vectordb.as_retriever(search_kwargs={"k": max(top_k * 4, top_k)})

        bm25_w, vector_w = BM25_VECTOR_WEIGHT
        ensemble = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[bm25_w, vector_w],
        )

        docs = ensemble.invoke(query)
        if allowed_nodes is not None:
            docs = _filter_docs_by_nodes(docs, allowed_nodes)
        return docs[:top_k]
    except Exception:
        # Manual fallback path still keeps BM25 + Vector weighted fusion.
        candidate_k = max(top_k * 4, top_k)
        bm25_docs = _bm25_search(client_name, query, top_k=candidate_k)
        vector_docs = _vector_search(client_name, query, top_k=candidate_k)

        if allowed_nodes is not None:
            bm25_docs = _filter_docs_by_nodes(bm25_docs, allowed_nodes)
            vector_docs = _filter_docs_by_nodes(vector_docs, allowed_nodes)

        bm25_w, vector_w = BM25_VECTOR_WEIGHT
        fused = _hybrid_fuse(
            bm25_docs=bm25_docs,
            vector_docs=vector_docs,
            top_k=top_k,
            bm25_weight=bm25_w,
            vector_weight=vector_w,
        )
        return fused


def _ensure_preprocessed(client_name: str) -> None:
    required = [
        PREPROCESS_DIR / f"{client_name}_symbols.json",
        PREPROCESS_DIR / f"{client_name}_callgraph.json",
        PREPROCESS_DIR / f"{client_name}_bm25.pkl",
        PREPROCESS_DIR / f"{client_name}_chroma",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Preprocessing artifacts missing for {client_name}: {missing}. "
            "Run run_preprocessing(client_name) first."
        )


@tool
def search_codebase(query: str, client_name: str, top_k: int = 5) -> list[Document]:
    """Hybrid Search for Phase 1 vocabulary discovery.

    1. BM25 retrieval from identifier-tokenized corpus
    2. Vector similarity retrieval from persistent Chroma index
    3. Weighted fusion (BM25:Vector = 4:6 by default)

    Returns Documents with evidence metadata fields.
    """

    if top_k <= 0:
        return []

    _ensure_preprocessed(client_name)

    return _hybrid_with_ensemble(client_name=client_name, query=query, top_k=top_k)


def _call_depth_from_entries(
    nodes: list[str],
    edges: list[tuple[str, str]],
    entries: list[str],
) -> dict[str, int]:
    adj: dict[str, list[str]] = {}
    for caller, callee in edges:
        adj.setdefault(caller, []).append(callee)

    depth = {n: 999 for n in nodes}
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

    return depth


def _filter_docs_by_nodes(docs: list[Document], allowed_nodes: set[str]) -> list[Document]:
    filtered = []
    for doc in docs:
        qn = doc.metadata.get("qualified_name")
        if qn in allowed_nodes:
            filtered.append(doc)
    return filtered


@tool
def search_codebase_by_workflow(
    workflow_id: str,
    query: str,
    client_name: str,
    max_call_depth: int = 5,
    top_k: int = 10,
) -> list[Document]:
    """Callgraph-guided hybrid retrieval for Phase 2 LSG extraction.

    Steps:
    1. Get workflow entry points from callgraph
    2. BFS callgraph to collect nodes with call_depth <= max_call_depth
    3. Run hybrid retrieval (BM25 + vector)
    4. Keep only docs in callgraph subgraph and sort by call_depth asc
    """

    if workflow_id not in WORKFLOW_IDS:
        raise ValueError(f"Unsupported workflow_id: {workflow_id}")
    if top_k <= 0:
        return []

    _ensure_preprocessed(client_name)

    cg = load_callgraph(client_name)
    entry_points = cg.entry_points.get(workflow_id, [])

    if not entry_points:
        broad = WORKFLOW_FALLBACK_KEYWORDS.get(workflow_id, [])
        for node in cg.nodes:
            normalized = "".join(ch for ch in node.lower() if ch.isalnum())
            hits = sum(1 for k in broad if k in normalized)
            if hits >= 2:
                entry_points.append(node)
        entry_points = entry_points[:50]

    edges = [(e.caller, e.callee) for e in cg.edges]
    depth_map = _call_depth_from_entries(cg.nodes, edges, entry_points)

    allowed = {node for node, d in depth_map.items() if d <= max_call_depth}
    if not allowed:
        allowed = set(cg.nodes)

    fused = _hybrid_with_ensemble(
        client_name=client_name,
        query=query,
        top_k=max(top_k * 2, top_k),
        allowed_nodes=allowed,
    )

    # Enrich + sort by call depth ascending.
    for doc in fused:
        qn = doc.metadata.get("qualified_name")
        d = depth_map.get(qn, 999)
        hints = [workflow_id] if d != 999 else []
        md = dict(doc.metadata)
        md["call_depth"] = d
        md["workflow_hints"] = hints
        doc.metadata = md

    fused.sort(key=lambda d: (d.metadata.get("call_depth", 999), -d.metadata.get("_hybrid_score", 0)))
    return fused[:top_k]
