"""Checkpoint persistence for EthAuditor graph state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from state import GlobalState

ROOT_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT_DIR / "output" / "checkpoints"


def _checkpoint_path(phase: int, iteration: int) -> Path:
    return CHECKPOINT_DIR / f"checkpoint_phase{phase}_iter{iteration}.json"


def save_checkpoint(state: GlobalState, phase: int, iteration: int) -> Path:
    """Persist current state to output/checkpoints.

    File pattern: checkpoint_phase<P>_iter<N>.json
    """

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    target = _checkpoint_path(phase, iteration)
    target.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target


def load_checkpoint(phase: int, iteration: int) -> GlobalState:
    """Load state from an existing checkpoint file."""

    target = _checkpoint_path(phase, iteration)
    if not target.exists():
        raise FileNotFoundError(f"Checkpoint not found: {target}")

    payload: Any = json.loads(target.read_text(encoding="utf-8"))
    return cast(GlobalState, payload)
