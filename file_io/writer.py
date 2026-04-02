"""EthAuditor — File output writer.

Responsible for writing:
  - Global_LSG_Spec_Enriched.yaml  (Phase 1 exit)
  - LSG_<Client>_final.yaml        (Phase 2 exit × 5)
  - LSG_<Client>_iter<N>.yaml      (intermediate)
  - Audit_Diff_Report.md           (Phase 2 exit)
  - Audit_Diff_Report.json         (Phase 2 exit — structured)
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

import config
from config import CLIENT_NAMES, WORKFLOW_IDS

logger = logging.getLogger(__name__)


def write_enriched_spec(state: dict[str, Any]) -> Path:
    """Write the enriched global vocabulary (Phase 1 output)."""
    config.OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    path = config.OUTPUT_PATH / "Global_LSG_Spec_Enriched.yaml"

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
        config.OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
        path = config.OUTPUT_PATH / f"LSG_{client_name}_final.yaml"
    else:
        config.ITERATIONS_PATH.mkdir(parents=True, exist_ok=True)
        iteration = lsg.get("_iteration", 0)
        path = config.ITERATIONS_PATH / f"LSG_{client_name}_iter{iteration}.yaml"

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


# ────────────────────────────────────────────────────────────────────────
# Internal analytics helpers for the enriched report
# ────────────────────────────────────────────────────────────────────────


def _per_workflow_summary(
    a_diffs: list[dict], b_diffs: list[dict],
    total_transitions: int = 0,
) -> list[dict]:
    """Return per-workflow summary sorted by total diff count (descending).

    ``similarity`` reflects structural agreement: the proportion of
    comparison items that are NOT B-class diffs.  A-class diffs (vocabulary
    misalignment) do not reduce similarity because they are auto-resolved.
    """
    wf_a: Counter = Counter()
    wf_b: Counter = Counter()
    for d in a_diffs:
        wf_a[d.get("workflow_id", "?")] += 1
    for d in b_diffs:
        wf_b[d.get("workflow_id", "?")] += 1

    all_wfs = sorted(set(wf_a.keys()) | set(wf_b.keys()) | set(WORKFLOW_IDS))
    total_b = sum(wf_b.values())

    # Distribute total_transitions proportionally across workflows for
    # similarity computation.  If not available, fall back to using total_b
    # as denominator (similarity is then 0% for any workflow with B-class diffs).
    rows: list[dict] = []
    for wf in all_wfs:
        a = wf_a.get(wf, 0)
        b = wf_b.get(wf, 0)
        total = a + b
        # Estimate per-workflow share of total_transitions proportionally
        if total_transitions > 0 and total_b > 0:
            # Allocate transitions proportionally to B-class count per workflow
            wf_transitions = max(
                int(total_transitions * (b / total_b)) if b > 0 else total_transitions // len(all_wfs),
                b,  # floor: at least as many as B-class diffs
            )
        elif total_transitions > 0:
            wf_transitions = total_transitions // max(len(all_wfs), 1)
        else:
            wf_transitions = max(a + b, 1)
        similarity = 1.0 - b / max(wf_transitions, 1)
        similarity = max(similarity, 0.0)
        rows.append({
            "workflow_id": wf,
            "a_class": a,
            "b_class": b,
            "total": total,
            "similarity": similarity,
        })
    rows.sort(key=lambda r: r["total"], reverse=True)
    return rows


def _per_client_ranking(
    a_diffs: list[dict], b_diffs: list[dict],
) -> list[dict]:
    """Return per-client deviation ranking (most involved first)."""
    client_a: Counter = Counter()
    client_b: Counter = Counter()
    for d in a_diffs:
        for c in d.get("involved_clients", []):
            client_a[c] += 1
    for d in b_diffs:
        for c in d.get("involved_clients", []):
            client_b[c] += 1

    all_clients = sorted(set(client_a.keys()) | set(client_b.keys()) | set(CLIENT_NAMES))
    rows: list[dict] = []
    for c in all_clients:
        a = client_a.get(c, 0)
        b = client_b.get(c, 0)
        rows.append({"client": c, "a_class": a, "b_class": b, "total": a + b})
    rows.sort(key=lambda r: r["total"], reverse=True)
    return rows


def _agreement_workflows(
    a_diffs: list[dict], b_diffs: list[dict],
) -> list[str]:
    """Return workflow IDs where ALL clients fully agree (0 diffs)."""
    diff_wfs = set()
    for d in a_diffs:
        diff_wfs.add(d.get("workflow_id", "?"))
    for d in b_diffs:
        diff_wfs.add(d.get("workflow_id", "?"))
    return sorted(wf for wf in WORKFLOW_IDS if wf not in diff_wfs)


def _classify_severity_fallback(diff: dict) -> str:
    """Fallback severity classification for B-class diffs missing the field."""
    desc_lower = (diff.get("description", "") or "").lower()
    state_id = diff.get("state_id", "")

    if state_id.endswith(".*") or "stub" in desc_lower:
        return "CRITICAL"
    if "missing in" in desc_lower and "state category" in desc_lower:
        return "CRITICAL"

    minor_keywords = [
        "present in", "but no equivalent", "not present in other",
        "granularity", "not explicitly", "implicitly",
    ]
    if any(kw in desc_lower for kw in minor_keywords):
        return "MINOR"
    return "MAJOR"


# Canonical severity values.
_VALID_SEVERITIES = {"CRITICAL", "MAJOR", "MINOR"}

# Map common non-standard LLM outputs to canonical values (case-insensitive).
_SEVERITY_ALIASES: dict[str, str] = {
    "high": "MAJOR",
    "medium": "MINOR",
    "low": "MINOR",
    "severe": "CRITICAL",
    "critical": "CRITICAL",
    "major": "MAJOR",
    "minor": "MINOR",
}


def _normalize_severity(diff: dict) -> str:
    """Return a canonical severity for *diff*, normalizing LLM quirks.

    If the value is already canonical (``CRITICAL``/``MAJOR``/``MINOR``),
    return it as-is.  Otherwise try alias lookup, then fall back to the
    heuristic classifier.
    """
    raw = (diff.get("severity") or "").strip()
    if raw in _VALID_SEVERITIES:
        return raw
    mapped = _SEVERITY_ALIASES.get(raw.lower())
    if mapped:
        return mapped
    return _classify_severity_fallback(diff)


def _deduplicate_b_diffs(b_diffs: list[dict]) -> list[dict]:
    """Deduplicate B-class diffs that describe the same structural difference.

    Diffs are grouped by ``(workflow_id, state_id, transition_guard)``.
    Within each group, entries whose descriptions describe opposite directions
    of the same gap (e.g. "present in X but not Y" and "present in Y but not X")
    are merged: ``involved_clients`` are unioned and descriptions concatenated.
    """
    from collections import OrderedDict

    groups: OrderedDict[tuple, list[dict]] = OrderedDict()
    for d in b_diffs:
        key = (
            d.get("workflow_id", "?"),
            d.get("state_id", "?"),
            d.get("transition_guard", "?"),
        )
        groups.setdefault(key, []).append(d)

    deduped: list[dict] = []
    for key, entries in groups.items():
        if len(entries) == 1:
            deduped.append(entries[0])
            continue

        # Merge: union involved_clients, pick highest severity, join descriptions
        merged_clients: set[str] = set()
        descriptions: list[str] = []
        best_severity = "MINOR"
        evidence: dict = {}
        severity_rank = {"CRITICAL": 3, "MAJOR": 2, "MINOR": 1, "": 0}

        seen_descs: set[str] = set()
        for e in entries:
            for c in e.get("involved_clients", []):
                merged_clients.add(c)
            desc = e.get("description", "")
            if desc and desc not in seen_descs:
                descriptions.append(desc)
                seen_descs.add(desc)
            sev = _normalize_severity(e)
            if severity_rank.get(sev, 0) > severity_rank.get(best_severity, 0):
                best_severity = sev
            if e.get("evidence"):
                evidence.update(e["evidence"])

        deduped.append({
            "workflow_id": key[0],
            "state_id": key[1],
            "transition_guard": key[2],
            "diff_type": "B",
            "description": " | ".join(descriptions) if len(descriptions) > 1 else (descriptions[0] if descriptions else ""),
            "severity": best_severity,
            "involved_clients": sorted(merged_clients),
            "evidence": evidence,
        })

    return deduped


def _generate_executive_summary(
    a_diffs: list[dict],
    b_diffs: list[dict],
    wf_summary: list[dict],
    client_ranking: list[dict],
    agreement_wfs: list[str],
    force_stopped: bool,
) -> str:
    """Generate a human-readable executive summary paragraph."""
    total_a = len(a_diffs)
    total_b = len(b_diffs)
    total = total_a + total_b

    parts: list[str] = []

    parts.append(
        f"This report compares {len(CLIENT_NAMES)} Ethereum consensus clients "
        f"({', '.join(CLIENT_NAMES)}) across {len(WORKFLOW_IDS)} core workflows. "
        f"A total of **{total}** differences were identified: "
        f"**{total_a}** A-class (vocabulary/naming misalignment, auto-resolved) "
        f"and **{total_b}** B-class (genuine structural/logic divergences "
        f"requiring human review)."
    )

    if agreement_wfs:
        parts.append(
            f"All clients are in **full agreement** on "
            f"{len(agreement_wfs)} workflow(s): {', '.join(f'`{w}`' for w in agreement_wfs)}."
        )

    if wf_summary:
        most_divergent = wf_summary[0]
        parts.append(
            f"The most divergent workflow is **`{most_divergent['workflow_id']}`** "
            f"with {most_divergent['total']} diffs "
            f"({most_divergent['b_class']} B-class)."
        )

    if client_ranking:
        most_unique = client_ranking[0]
        parts.append(
            f"The client with the most unique implementation choices is "
            f"**{most_unique['client']}** (involved in {most_unique['total']} diffs, "
            f"{most_unique['b_class']} B-class)."
        )

    if force_stopped:
        parts.append(
            "⚠️ The pipeline was **force-stopped** before natural convergence."
        )

    return " ".join(parts)


# ────────────────────────────────────────────────────────────────────────
# Public writers
# ────────────────────────────────────────────────────────────────────────


def write_diff_report(state: dict[str, Any]) -> Path:
    """Write the Audit Diff Report (Phase 2 output) as enriched Markdown.

    Sections:
    1. Executive Summary
    2. Summary table
    3. Per-Workflow Summary
    4. Per-Client Deviation Ranking
    5. A-Class Vocabulary Alignment Diffs
    6. B-Class Structural Logic Differences (grouped by severity)
    7. Agreement (fully matching workflows)
    8. Iteration Trend (if available)
    """
    config.OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    path = config.OUTPUT_PATH / "Audit_Diff_Report.md"

    diff_report = state.get("diff_report", {})
    b_diffs_raw = diff_report.get("b_class_diffs", [])
    a_diffs = diff_report.get("a_class_diffs", [])
    logic_diff_rate = diff_report.get("logic_diff_rate", 0.0)
    total_transitions = diff_report.get("total_transitions", 0)
    force_stopped = state.get("force_stopped", False)
    convergence_reason = state.get("convergence_reason", "")
    iteration_history = state.get("iteration_history", [])

    # ── Deduplicate B-class diffs ──────────────────────────────────────
    b_diffs = _deduplicate_b_diffs(b_diffs_raw)

    # ── Ensure all B-class diffs have canonical severity ─────────────
    for d in b_diffs:
        d["severity"] = _normalize_severity(d)

    # ── Compute analytics ──────────────────────────────────────────────
    wf_summary = _per_workflow_summary(a_diffs, b_diffs, total_transitions)
    client_ranking = _per_client_ranking(a_diffs, b_diffs)
    agreement_wfs = _agreement_workflows(a_diffs, b_diffs)
    exec_summary = _generate_executive_summary(
        a_diffs, b_diffs, wf_summary, client_ranking, agreement_wfs, force_stopped,
    )

    lines: list[str] = [
        "# EthAuditor — Audit Diff Report",
        "",
        f"**Generated at**: {datetime.now(timezone.utc).isoformat()}",
        f"**Logic Diff Rate (B-class)**: {logic_diff_rate:.4f}",
        f"**Force Stopped**: {force_stopped}",
    ]
    if convergence_reason:
        lines.append(f"**Convergence Reason**: {convergence_reason}")
    lines.append("")

    # ── 1. Executive Summary ───────────────────────────────────────────
    lines.extend([
        "## Executive Summary",
        "",
        exec_summary,
        "",
    ])

    # ── 2. Summary Table ───────────────────────────────────────────────
    # Count B-class by severity
    sev_counts = Counter(d.get("severity", "MAJOR") for d in b_diffs)
    lines.extend([
        "## Summary",
        "",
        "| Metric | Count |",
        "|--------|-------|",
        f"| A-class (vocabulary alignment) diffs | {len(a_diffs)} |",
        f"| B-class (structural logic) diffs | {len(b_diffs)} |",
        f"| — 🔴 CRITICAL | {sev_counts.get('CRITICAL', 0)} |",
        f"| — 🟠 MAJOR | {sev_counts.get('MAJOR', 0)} |",
        f"| — 🟡 MINOR | {sev_counts.get('MINOR', 0)} |",
        f"| Total diffs | {len(a_diffs) + len(b_diffs)} |",
        f"| Workflows with full agreement | {len(agreement_wfs)} |",
        "",
    ])

    # ── 3. Per-Workflow Summary ────────────────────────────────────────
    lines.extend([
        "## Per-Workflow Summary",
        "",
        "| Workflow | A-class | B-class | Total | Similarity |",
        "|----------|---------|---------|-------|------------|",
    ])
    for row in wf_summary:
        lines.append(
            f"| `{row['workflow_id']}` | {row['a_class']} | {row['b_class']} "
            f"| {row['total']} | {row['similarity']:.0%} |"
        )
    lines.append("")

    # ── 4. Per-Client Deviation Ranking ────────────────────────────────
    lines.extend([
        "## Per-Client Deviation Ranking",
        "",
        "| Client | A-class | B-class | Total Diffs Involved |",
        "|--------|---------|---------|----------------------|",
    ])
    for row in client_ranking:
        lines.append(
            f"| **{row['client']}** | {row['a_class']} | {row['b_class']} "
            f"| {row['total']} |"
        )
    lines.append("")

    # ── 5. A-Class Vocabulary Alignment Diffs ──────────────────────────
    if a_diffs:
        lines.extend([
            "## A-Class Vocabulary Alignment Diffs",
            "",
            "These differences reflect **naming misalignment only** — the "
            "underlying logic is equivalent. The system has generated rename "
            "directives to align vocabulary across clients.",
            "",
        ])
        for i, diff in enumerate(a_diffs, 1):
            lines.append(
                f"### A-{i}: {diff.get('workflow_id', '?')} / "
                f"{diff.get('state_id', '?')}"
            )
            lines.append("")
            lines.append(f"- **Guard**: `{diff.get('transition_guard', '?')}`")
            lines.append(
                f"- **Clients involved**: "
                f"{', '.join(diff.get('involved_clients', []))}"
            )
            lines.append(f"- **Rename directive**: {diff.get('description', '')}")
            lines.append("")
    else:
        lines.extend([
            "## A-Class Vocabulary Alignment Diffs",
            "",
            "✅ No vocabulary misalignment detected — all clients use "
            "consistent guard and action names.",
            "",
        ])

    # ── 6. B-Class Structural Logic Differences (by severity) ──────────
    if b_diffs:
        lines.extend([
            "## B-Class Structural Logic Differences",
            "",
            "These are **genuine design divergences** between client "
            "implementations that require human review. They reflect "
            "architectural choices, not naming inconsistencies.",
            "",
        ])

        severity_order = [
            ("CRITICAL", "🔴 Critical — Missing Workflows / State Categories"),
            ("MAJOR", "🟠 Major — Behavioral Divergences"),
            ("MINOR", "🟡 Minor — Extra/Missing Transitions, Granularity Differences"),
        ]
        b_idx = 1
        for sev_key, sev_label in severity_order:
            sev_diffs = [d for d in b_diffs if d.get("severity") == sev_key]
            if not sev_diffs:
                continue
            lines.extend([
                f"### {sev_label} ({len(sev_diffs)})",
                "",
            ])
            for diff in sev_diffs:
                lines.append(
                    f"#### B-{b_idx}: {diff.get('workflow_id', '?')} / "
                    f"{diff.get('state_id', '?')}"
                )
                lines.append("")
                lines.append(f"- **Guard**: `{diff.get('transition_guard', '?')}`")
                lines.append(
                    f"- **Clients involved**: "
                    f"{', '.join(diff.get('involved_clients', []))}"
                )
                lines.append(f"- **Description**: {diff.get('description', '')}")
                evidence = diff.get("evidence", {})
                if evidence:
                    lines.append("- **Evidence**:")
                    for client, ev in evidence.items():
                        if ev:
                            lines.append(
                                f"  - {client}: `{ev.get('file', '?')}` → "
                                f"`{ev.get('function', '?')}` "
                                f"L{ev.get('lines', [])}"
                            )
                lines.append("")
                b_idx += 1
    else:
        lines.extend([
            "## B-Class Structural Logic Differences",
            "",
            "✅ No structural logic differences found — all clients implement "
            "identical state machines.",
            "",
        ])

    # ── 7. Agreement ───────────────────────────────────────────────────
    lines.extend([
        "## Agreement",
        "",
    ])
    if agreement_wfs:
        lines.append(
            "The following workflows show **complete agreement** across all "
            "clients (no A-class or B-class diffs detected):"
        )
        lines.append("")
        for wf in agreement_wfs:
            lines.append(f"- ✅ `{wf}`")
        lines.append("")
    else:
        lines.append(
            "No workflows show complete agreement across all clients. "
            "Every workflow has at least one A-class or B-class diff."
        )
        lines.append("")

    # ── 8. Iteration Trend ─────────────────────────────────────────────
    if iteration_history:
        lines.extend([
            "## Iteration Trend",
            "",
            "| Iter | A-class | B-class | Logic Diff Rate |",
            "|------|---------|---------|-----------------|",
        ])
        for h in iteration_history:
            lines.append(
                f"| {h.get('iteration', '?')} "
                f"| {h.get('a_class_count', '?')} "
                f"| {h.get('b_class_count', '?')} "
                f"| {h.get('logic_diff_rate', 0.0):.4f} |"
            )
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(
        "[write_diff_report] → %s (%d A-class, %d B-class diffs)",
        path, len(a_diffs), len(b_diffs),
    )
    return path


def write_diff_report_json(state: dict[str, Any]) -> Path:
    """Write a structured JSON version of the Audit Diff Report.

    Provides the same data as the Markdown report in a machine-readable
    format for downstream tools, dashboards, and CI/CD integration.
    """
    config.OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    path = config.OUTPUT_PATH / "Audit_Diff_Report.json"

    diff_report = state.get("diff_report", {})
    b_diffs_raw = diff_report.get("b_class_diffs", [])
    a_diffs = diff_report.get("a_class_diffs", [])
    logic_diff_rate = diff_report.get("logic_diff_rate", 0.0)
    total_transitions = diff_report.get("total_transitions", 0)
    force_stopped = state.get("force_stopped", False)
    convergence_reason = state.get("convergence_reason", "")
    iteration_history = state.get("iteration_history", [])

    # Deduplicate and normalize severity
    b_diffs = _deduplicate_b_diffs(b_diffs_raw)
    for d in b_diffs:
        d["severity"] = _normalize_severity(d)

    wf_summary = _per_workflow_summary(a_diffs, b_diffs, total_transitions)
    client_ranking = _per_client_ranking(a_diffs, b_diffs)
    agreement_wfs = _agreement_workflows(a_diffs, b_diffs)
    exec_summary = _generate_executive_summary(
        a_diffs, b_diffs, wf_summary, client_ranking, agreement_wfs, force_stopped,
    )

    sev_counts = Counter(d.get("severity", "MAJOR") for d in b_diffs)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "logic_diff_rate": logic_diff_rate,
        "force_stopped": force_stopped,
        "convergence_reason": convergence_reason,
        "executive_summary": exec_summary,
        "summary": {
            "a_class_count": len(a_diffs),
            "b_class_count": len(b_diffs),
            "b_class_critical": sev_counts.get("CRITICAL", 0),
            "b_class_major": sev_counts.get("MAJOR", 0),
            "b_class_minor": sev_counts.get("MINOR", 0),
            "total_diffs": len(a_diffs) + len(b_diffs),
            "total_transitions": total_transitions,
            "agreement_workflows": len(agreement_wfs),
        },
        "per_workflow_summary": wf_summary,
        "per_client_ranking": client_ranking,
        "agreement_workflows": agreement_wfs,
        "a_class_diffs": a_diffs,
        "b_class_diffs": b_diffs,
        "iteration_history": iteration_history,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    logger.info("[write_diff_report_json] → %s", path)
    return path
