"""EthAuditor — centralized configuration constants."""

from __future__ import annotations

from pathlib import Path

# ── Iteration limits & convergence ──────────────────────────────────────
MAX_ITER_PHASE1: int = 20
MAX_ITER_PHASE2: int = 20
CONVERGENCE_THRESHOLD: float = 0.05

# ── Client names (order used everywhere) ────────────────────────────────
CLIENT_NAMES: list[str] = [
    "prysm",
    "lighthouse",
    "grandine",
    "teku",
    "lodestar",
]

# ── Reserved workflow IDs (all clients must implement these) ────────────
WORKFLOW_IDS: list[str] = [
    "initial_sync",
    "regular_sync",
    "checkpoint_sync",
    "attestation_generate",
    "block_generate",
    "aggregate",
    "execute_layer_relation",
]

# ── Paths ───────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent
CODE_BASE_PATH: Path = PROJECT_ROOT / "code"
OUTPUT_PATH: Path = PROJECT_ROOT / "output"
PREPROCESS_PATH: Path = OUTPUT_PATH / "preprocess"
CHECKPOINT_PATH: Path = OUTPUT_PATH / "checkpoints"
ITERATIONS_PATH: Path = OUTPUT_PATH / "iterations"
AUDIT_LOG_PATH: Path = OUTPUT_PATH / "audit_logs"
SPEC_PATH: Path = PROJECT_ROOT / "docs" / "LSG_Schema_Spec.md"

# ── RAG weights ─────────────────────────────────────────────────────────
BM25_WEIGHT: float = 0.4
VECTOR_WEIGHT: float = 0.6

# ── Language → tree‑sitter grammar mapping ──────────────────────────────
LANGUAGE_GRAMMARS: dict[str, tuple[str, str]] = {
    "prysm":      ("go",         "tree-sitter-go"),
    "lighthouse": ("rust",       "tree-sitter-rust"),
    "grandine":   ("rust",       "tree-sitter-rust"),
    "teku":       ("java",       "tree-sitter-java"),
    "lodestar":   ("typescript", "tree-sitter-typescript"),
}

# ── Entry‑point keyword heuristics (case‑insensitive) ──────────────────
ENTRY_POINT_KEYWORDS: dict[str, list[str]] = {
    "initial_sync":            ["initialsync", "runinitial", "startinitial"],
    "regular_sync":            ["regularsync", "runregular", "gossipsync"],
    "checkpoint_sync":         ["checkpointsync", "runcheckpoint"],
    "block_generate":          ["proposeblock", "buildblock", "produceblock"],
    "attestation_generate":    ["submitattestation", "createattestation"],
    "aggregate":               ["aggregate", "computeaggregate"],
    "execute_layer_relation":  ["engineapi", "executionengine", "forkchoiceupdate"],
}

# ── User overrides for entry points (client → workflow → list[fn]) ─────
ENTRY_POINT_OVERRIDES: dict[str, dict[str, list[str]]] = {}

# ── Embedding model priority (first available wins) ─────────────────────
EMBEDDING_MODELS: list[str] = [
    "nomic-embed-code",
    "text-embedding-3-large",
    "all-MiniLM-L6-v2",
]

# ── LLM provider & model names ──────────────────────────────────────────
LLM_PROVIDER: str = "anthropic"  # "anthropic" or "gemini"
LLM_MODEL: str = "claude-sonnet-4-20250514"
GEMINI_MODEL: str = "gemini-3.1-pro-preview"

# ── API proxy / custom base URLs ────────────────────────────────────────
# Empty string means "use provider default".  Override via env vars or CLI.
ANTHROPIC_BASE_URL: str = ""
GEMINI_BASE_URL: str = ""

