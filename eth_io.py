"""Bridge module for project I/O helpers.

This avoids import name conflict with Python stdlib `io` module.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent
IO_DIR = ROOT_DIR / "io"


def _load(name: str, file_name: str) -> ModuleType:
    target = IO_DIR / file_name
    spec = importlib.util.spec_from_file_location(name, target)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module: {target}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_checkpoint = _load("ethauditor_checkpoint", "checkpoint.py")
_writer = _load("ethauditor_writer", "writer.py")
_audit = _load("ethauditor_audit", "audit_logger.py")

save_checkpoint = _checkpoint.save_checkpoint
load_checkpoint = _checkpoint.load_checkpoint

write_enriched_spec = _writer.write_enriched_spec
write_iteration_lsg = _writer.write_iteration_lsg
write_final_lsgs = _writer.write_final_lsgs
write_diff_report = _writer.write_diff_report

AuditCallbackHandler = _audit.AuditCallbackHandler
make_audit_callback = _audit.make_audit_callback
