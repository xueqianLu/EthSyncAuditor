"""Global configuration for EthAuditor graph execution."""

from __future__ import annotations

MAX_ITER_PHASE1: int = 20
MAX_ITER_PHASE2: int = 20
CONVERGENCE_THRESHOLD: float = 0.05

CLIENT_NAMES: list[str] = [
    "prysm",
    "lighthouse",
    "grandine",
    "teku",
    "lodestar",
]

CODE_BASE_PATH: str = "./code"
OUTPUT_PATH: str = "./output"
PREPROCESS_PATH: str = "./output/preprocess"
SPEC_PATH: str = "./docs/LSG_Schema_Spec.md"

# Hybrid retrieval weight (BM25, Vector)
BM25_VECTOR_WEIGHT: tuple[float, float] = (0.4, 0.6)

WORKFLOW_IDS: list[str] = [
    "initial_sync",
    "regular_sync",
    "checkpoint_sync",
    "attestation_generate",
    "block_generate",
    "aggregate",
    "execute_layer_relation",
]

# Optional manual entry point overrides, filled in Step 3.
ENTRY_POINT_OVERRIDES: dict[str, dict[str, list[str]]] = {}
