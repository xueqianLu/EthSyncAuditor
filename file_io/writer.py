"""EthAuditor — File output writer.

Responsible for writing:
  - Global_LSG_Spec_Enriched.yaml  (Phase 1 exit)
  - LSG_<Client>_final.yaml        (Phase 2 exit × 5)
  - LSG_<Client>_iter<N>.yaml      (intermediate)
  - Audit_Diff_Report.md           (Phase 2 exit)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from config import CLIENT_NAMES, ITERATIONS_PATH, OUTPUT_PATH

logger = logging.getLogger(__name__)


def write_enriched_spec(state: dict[str, Any]) -> Path:
    """Write the enriched global vocabulary (Phase 1 output)."""
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_PATH / "Global_LSG_Spec_Enriched.yaml"

    spec = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "guards": list(state.get("guards", [])),
        "actions": list(state.get("actions", [])),
    }

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(spec, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    logger.info("[write_enriched_spec] → %s", path)
    return path


def write_client_lsg(client_name: str, lsg: dict[str, Any], final: bool = False) -> Path:
    """Write a client's LSG YAML (intermediate or final)."""
    if final:
        OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_PATH / f"LSG_{client_name}_final.yaml"
    else:
        ITERATIONS_PATH.mkdir(parents=True, exist_ok=True)
        iteration = lsg.get("_iteration", 0)
        path = ITERATIONS_PATH / f"LSG_{client_name}_iter{iteration}.yaml"

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(lsg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    logger.info("[write_client_lsg] client=%s final=%s → %s", client_name, final, path)
    return path


def write_all_final_lsgs(state: dict[str, Any]) -> list[Path]:
    """Write final LSG YAML for all clients."""
    paths: list[Path] = []
    client_lsgs = state.get("client_lsgs", {})
    for client_name in CLIENT_NAMES:
        lsg = client_lsgs.get(client_name)
        if lsg is not None:
            path = write_client_lsg(client_name, lsg, final=True)
            paths.append(path)
    return paths


def write_diff_report(state: dict[str, Any]) -> Path:
    """Write the Audit Diff Report (Phase 2 output) as Markdown."""
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_PATH / "Audit_Diff_Report.md"

    diff_report = state.get("diff_report", {})
    b_diffs = diff_report.get("b_class_diffs", [])
    a_diffs = diff_report.get("a_class_diffs", [])
    logic_diff_rate = diff_report.get("logic_diff_rate", 0.0)
    force_stopped = state.get("force_stopped", False)

    lines: list[str] = [
        "# EthAuditor — Audit Diff Report",
        "",
        f"**Generated at**: {datetime.now(timezone.utc).isoformat()}",
        f"**Logic Diff Rate**: {logic_diff_rate:.4f}",
        f"**Force Stopped**: {force_stopped}",
        "",
        "## Summary",
        "",
        f"| Metric | Count |",
        f"|--------|-------|",
        f"| A-class (implementation) diffs | {len(a_diffs)} |",
        f"| B-class (logic) diffs | {len(b_diffs)} |",
        f"| Total comparison items | {len(a_diffs) + len(b_diffs)} |",
        "",
    ]

    if b_diffs:
        lines.append("## B-Class Logic Differences")
        lines.append("")
        for i, diff in enumerate(b_diffs, 1):
            lines.append(f"### B-{i}: {diff.get('workflow_id', '?')} / {diff.get('state_id', '?')}")
            lines.append("")
            lines.append(f"- **Guard**: `{diff.get('transition_guard', '?')}`")
            lines.append(f"- **Clients involved**: {', '.join(diff.get('involved_clients', []))}")
            lines.append(f"- **Description**: {diff.get('description', '')}")
            evidence = diff.get("evidence", {})
            if evidence:
                lines.append("- **Evidence**:")
                for client, ev in evidence.items():
                    if ev:
                        lines.append(f"  - {client}: `{ev.get('file', '?')}` → `{ev.get('function', '?')}` L{ev.get('lines', [])}")
            lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("[write_diff_report] → %s (%d B-class diffs)", path, len(b_diffs))
    return path
