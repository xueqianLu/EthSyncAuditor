"""EthAuditor — Checkpoint persistence.

Save / load GlobalState snapshots for resumable runs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from config import CHECKPOINT_PATH
from utils import safe_serialize

logger = logging.getLogger(__name__)


def save_checkpoint(state: dict[str, Any], phase: int, iteration: int) -> Path:
    """Serialize *state* to a checkpoint JSON file.

    Returns the path of the written file.
    """
    CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)
    filename = f"checkpoint_phase{phase}_iter{iteration}.json"
    path = CHECKPOINT_PATH / filename

    # Make state JSON-serializable (drop non-serializable objects)
    serializable = safe_serialize(state)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    logger.info("[save_checkpoint] phase=%d iter=%d → %s", phase, iteration, path)
    return path


def load_checkpoint(phase: int, iteration: int) -> dict[str, Any]:
    """Load a previously saved checkpoint.

    Raises FileNotFoundError if the checkpoint does not exist.
    """
    filename = f"checkpoint_phase{phase}_iter{iteration}.json"
    path = CHECKPOINT_PATH / filename
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    with open(path, encoding="utf-8") as f:
        state = json.load(f)

    logger.info("[load_checkpoint] phase=%d iter=%d ← %s", phase, iteration, path)
    return state


def latest_checkpoint() -> tuple[int, int, dict[str, Any]] | None:
    """Find and load the latest checkpoint (highest phase, then iteration).

    Returns (phase, iteration, state) or None if no checkpoints exist.
    """
    CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(CHECKPOINT_PATH.glob("checkpoint_phase*_iter*.json"))
    if not checkpoints:
        return None

    # Parse phase/iter from filename
    latest = checkpoints[-1]
    stem = latest.stem  # e.g. "checkpoint_phase2_iter5"
    parts = stem.split("_")
    phase = int(parts[1].replace("phase", ""))
    iteration = int(parts[2].replace("iter", ""))

    state = load_checkpoint(phase, iteration)
    return phase, iteration, state
