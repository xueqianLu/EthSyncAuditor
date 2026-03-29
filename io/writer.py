"""Output writers for enriched spec, iteration LSGs, finals and diff report."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "output"
ITER_DIR = OUTPUT_DIR / "iterations"


def _now_rfc3339() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ITER_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_vocab_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in entries:
        normalized.append(
            {
                "name": item.get("name", ""),
                "category": item.get("category", "unknown"),
                "description": item.get("description", ""),
            }
        )
    return normalized


def write_enriched_spec(spec: dict[str, Any]) -> Path:
    """Write Phase 1 final enriched vocabulary as LSG-compatible YAML.

    Path: ./output/Global_LSG_Spec_Enriched.yaml
    """

    _ensure_dirs()

    payload = {
        "version": 1,
        "client": "global",
        "generated_at": _now_rfc3339(),
        "guards": _normalize_vocab_entries(spec.get("guards", [])),
        "actions": _normalize_vocab_entries(spec.get("actions", [])),
        "workflows": [],
    }

    out = OUTPUT_DIR / "Global_LSG_Spec_Enriched.yaml"
    out.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out


def write_iteration_lsg(client_name: str, iteration: int, lsg: dict[str, Any]) -> Path:
    """Write per-iteration LSG output.

    Path: ./output/iterations/LSG_<ClientName>_iter<N>.yaml
    """

    _ensure_dirs()
    out = ITER_DIR / f"LSG_{client_name}_iter{iteration}.yaml"
    out.write_text(yaml.safe_dump(lsg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out


def write_final_lsgs(lsg_final: dict[str, dict[str, Any]]) -> list[Path]:
    """Write final LSG outputs for all clients.

    Paths: ./output/LSG_<ClientName>_final.yaml
    """

    _ensure_dirs()
    paths: list[Path] = []
    for client_name, lsg in lsg_final.items():
        out = OUTPUT_DIR / f"LSG_{client_name}_final.yaml"
        out.write_text(yaml.safe_dump(lsg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        paths.append(out)
    return paths


def write_diff_report(diff_items_b: list[dict[str, Any]], summary: dict[str, Any]) -> Path:
    """Write markdown report for B-class logic differences.

    Path: ./output/Audit_Diff_Report.md
    """

    _ensure_dirs()

    lines: list[str] = []
    lines.append("# Audit Diff Report")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Compared items | {summary.get('compared_items', 0)} |")
    lines.append(f"| A-class diffs | {summary.get('a_diff_count', 0)} |")
    lines.append(f"| B-class diffs | {summary.get('b_diff_count', 0)} |")
    lines.append(f"| Logic diff rate | {summary.get('logic_diff_rate', 0):.4f} |")
    lines.append("")

    lines.append("## B-class Differences")
    lines.append("")

    if not diff_items_b:
        lines.append("No B-class differences found.")
    else:
        for idx, item in enumerate(diff_items_b, start=1):
            lines.append(f"### B-{idx}: {item.get('summary', 'N/A')}")
            lines.append("")
            lines.append(f"- Involved clients: {', '.join(item.get('involved_clients', []))}")
            lines.append(f"- Workflow: `{item.get('workflow_id', 'unknown')}`")
            lines.append(f"- State: `{item.get('state_id', 'unknown')}`")
            lines.append(f"- Guard: `{item.get('transition_guard', 'unknown')}`")
            lines.append(f"- Expected behavior: {item.get('expected_behavior', '')}")
            lines.append(f"- Actual behavior: {item.get('actual_behavior', '')}")
            lines.append("")

            evidence = item.get("evidence", {})
            if evidence:
                lines.append("Evidence:")
                for client, records in evidence.items():
                    lines.append(f"- {client}:")
                    for e in records:
                        lines.append(
                            f"  - `{e.get('file', '')}`::{e.get('function', '')} lines {e.get('lines', [])}"
                        )
                lines.append("")

    out = OUTPUT_DIR / "Audit_Diff_Report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
