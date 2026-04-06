"""EthAuditor — centralized configuration constants."""

from __future__ import annotations

from pathlib import Path

# ── Iteration limits & convergence ──────────────────────────────────────
MAX_ITER_PHASE1: int = 20
MAX_ITER_PHASE2: int = 20
CONVERGENCE_THRESHOLD: float = 0.05

# Phase 2 uses A-class delta stabilization instead of absolute B-class rate.
# Converge when the relative change in A-class count between two consecutive
# iterations drops below this threshold, OR when A-class count reaches 0.
P2_A_CLASS_CONVERGENCE_THRESHOLD: float = 0.10

# Oscillation detection: if the A-class count stays within ±BAND for WINDOW
# consecutive iterations, declare convergence (the system is not improving).
OSCILLATION_WINDOW: int = 3
OSCILLATION_BAND: int = 2

# ── B-class discovery iterations (post A-class convergence) ─────────────
# After A-class vocabulary alignment converges, the system continues for
# up to MAX_ITER_B_CLASS additional rounds focused on discovering more
# B-class (logic divergence) diffs.
MAX_ITER_B_CLASS: int = 3
# B-class focus converges when B-class count is stable (change ≤ threshold)
# for B_CLASS_STABLE_WINDOW consecutive iterations.
B_CLASS_STABLE_WINDOW: int = 2
B_CLASS_CHANGE_THRESHOLD: int = 1  # max allowed B-class count change per iter

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
# These are matched against function_name.lower().replace("_", "").
# IMPORTANT: keep keywords specific enough to avoid false positives
# (e.g. "aggregate" alone matches BLS aggregate functions).
ENTRY_POINT_KEYWORDS: dict[str, list[str]] = {
    "initial_sync": [
        "initialsync", "runinitial", "startinitial",
        # Lighthouse/Lodestar use "range sync" terminology
        "rangesync", "syncingchain", "syncchain",
        # Teku uses ForwardSync / SyncManager / PeerSync
        "forwardsync", "peersync", "syncmanager",
        # Grandine
        "syncblockbyrange",
    ],
    "regular_sync": [
        "regularsync", "runregular", "gossipsync",
        # Gossip block/attestation handlers
        "receiveblock", "receiveattestation", "processblock",
        "gossiphandler", "blockimporter", "gossipvalidator",
        # Reorg handling (newly emphasized)
        "handlereorg", "onreorg",
    ],
    "checkpoint_sync": [
        "checkpointsync", "runcheckpoint",
        # Backfill (Lighthouse/Lodestar)
        "backfillsync", "backfillbatch",
        # Weak subjectivity
        "weaksubjectivity",
    ],
    "block_generate": [
        "proposeblock", "buildblock", "produceblock",
        # Builder API / MEV-boost (newly emphasized)
        "builderapi", "builderbid", "mevboost",
        # Payload retrieval (specific Engine/Builder API calls)
        "enginegetpayload", "buildergetpayload",
        # Block production duty
        "blockproductionduty", "blockservice",
    ],
    "attestation_generate": [
        "submitattestation", "createattestation",
        # Service-level entry points
        "attestationservice", "attestationduty", "attestationproduction",
        "performattestationduty",
        # Slashing protection (newly emphasized)
        "slashingprotection", "issafetoattest",
    ],
    "aggregate": [
        # Specific: aggregation workflow, NOT BLS aggregate primitives
        "aggregateandproof", "submitaggregate",
        "aggregationduty", "aggregatorselection",
        "produceaggregate", "publishaggregate",
        "computeaggregate",
    ],
    "execute_layer_relation": [
        "engineapi", "executionengine", "forkchoiceupdate",
        # Specific Engine API calls (newly emphasized)
        "newpayload", "notifynewpayload", "payloadstatus",
        # Optimistic sync
        "optimisticsync", "optimisticimport",
        # Invalid payload handling
        "invalidpayload", "invalidateblock",
    ],
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
LLM_MODEL: str = "claude-opus-4-20250514"
GEMINI_MODEL: str = "gemini-2.5-pro"
# GEMINI_MODEL: str = "gemini-3.1-pro-preview"

# ── API proxy / custom base URLs ────────────────────────────────────────
# Empty string means "use provider default".  Override via env vars or CLI.
ANTHROPIC_BASE_URL: str = ""
GEMINI_BASE_URL: str = ""

