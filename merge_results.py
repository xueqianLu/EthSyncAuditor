#!/usr/bin/env python3
"""Merge Phase 1 & Phase 2 results from all experiment runs.

Produces:
  1. merged/Global_LSG_Spec_Enriched.yaml  — deduplicated global vocabulary
  2. merged/LSG_<client>_merged.yaml       — per-client merged LSG (×5)

Merge strategy:
  - Guards / Actions: deduplicate by `name`, keep the entry with the longest
    description (most informative).
  - Per-client LSG workflows: for each of the 7 workflow IDs, select the
    "richest" version across all runs (most states × transitions). All
    workflow versions from all runs are preserved for reference.
  - Per-client LSG guards/actions: regenerated from the merged global vocab,
    filtered to only those names referenced in the merged workflows.

Usage:
    python merge_results.py [--output-dir merged]
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml


# ── Configuration ───────────────────────────────────────────────────────

RESULT_DIRS = [
    "auditor_v0",
    "auditor_v1",
    "auditor_v2",
    "auditor_version_1",
    "auditor_version_2",
    "auditor_version_3",
    "auditor_version_4",
    "auditor_version_5",
    "auditor_version_6",
    "auditor_version_7",
]

CLIENT_NAMES = ["prysm", "lighthouse", "grandine", "teku", "lodestar"]

WORKFLOW_IDS = [
    "initial_sync",
    "regular_sync",
    "checkpoint_sync",
    "attestation_generate",
    "block_generate",
    "aggregate",
    "execute_layer_relation",
]

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_PATH = PROJECT_ROOT / "results"


# ── Helpers ─────────────────────────────────────────────────────────────


def load_yaml(path: Path) -> dict | None:
    """Load a YAML file, return None on failure."""
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"  ⚠ Failed to load {path}: {e}")
        return None


def merge_vocab_entries(all_entries: list[dict]) -> list[dict]:
    """Deduplicate vocabulary entries by name, keeping the best version.

    "Best" = longest description (most informative). If tied, last-seen wins.
    """
    by_name: dict[str, dict] = {}
    for entry in all_entries:
        name = entry.get("name", "")
        if not name:
            continue
        existing = by_name.get(name)
        if existing is None:
            by_name[name] = dict(entry)
        else:
            # Keep the one with longer description
            old_desc = existing.get("description", "") or ""
            new_desc = entry.get("description", "") or ""
            if len(new_desc) > len(old_desc):
                by_name[name] = dict(entry)
            # Also merge category if old was empty
            if not existing.get("category") and entry.get("category"):
                by_name[name]["category"] = entry["category"]
    return sorted(by_name.values(), key=lambda e: e.get("name", ""))


def workflow_richness(wf: dict) -> tuple[int, int]:
    """Return (num_states, num_transitions) as a richness metric."""
    states = wf.get("states", [])
    n_states = len(states)
    n_transitions = sum(len(st.get("transitions", [])) for st in states)
    return n_states, n_transitions


def collect_referenced_names(workflows: list[dict]) -> tuple[set[str], set[str]]:
    """Scan workflows and return (guard_names, action_names) actually used."""
    guard_names: set[str] = set()
    action_names: set[str] = set()
    for wf in workflows:
        for st in wf.get("states", []):
            for tr in st.get("transitions", []):
                g = tr.get("guard", "")
                if g and g != "TRUE":
                    guard_names.add(g)
                for a in tr.get("actions", []):
                    if a:
                        action_names.add(a)
    return guard_names, action_names


# ── Main merge logic ───────────────────────────────────────────────────


def merge_global_vocab(result_dirs: list[str]) -> tuple[list[dict], list[dict]]:
    """Merge Global_LSG_Spec_Enriched.yaml from all runs."""
    all_guards: list[dict] = []
    all_actions: list[dict] = []

    for dirname in result_dirs:
        spec_path = RESULTS_PATH / dirname / "Global_LSG_Spec_Enriched.yaml"
        data = load_yaml(spec_path)
        if data is None:
            print(f"  ⏭ Skipping {dirname} (no enriched spec)")
            continue
        guards = data.get("guards", [])
        actions = data.get("actions", [])
        print(f"  ✓ {dirname}: {len(guards)} guards, {len(actions)} actions")
        all_guards.extend(guards)
        all_actions.extend(actions)

    merged_guards = merge_vocab_entries(all_guards)
    merged_actions = merge_vocab_entries(all_actions)
    return merged_guards, merged_actions


def merge_client_lsg(
    client_name: str,
    result_dirs: list[str],
    global_guards: list[dict],
    global_actions: list[dict],
) -> dict:
    """Merge LSG_<client>_final.yaml from all runs for one client.

    For each workflow_id, picks the richest version (most states+transitions)
    across all runs.
    """
    # Collect all workflow versions: wf_id → [(richness, source_dir, wf_dict)]
    wf_candidates: dict[str, list[tuple[tuple[int, int], str, dict]]] = defaultdict(list)
    all_client_guards: list[dict] = []
    all_client_actions: list[dict] = []

    for dirname in result_dirs:
        lsg_path = RESULTS_PATH / dirname / f"LSG_{client_name}_final.yaml"
        data = load_yaml(lsg_path)
        if data is None:
            continue
        # Collect per-client vocab
        all_client_guards.extend(data.get("guards", []))
        all_client_actions.extend(data.get("actions", []))

        for wf in data.get("workflows", []):
            wf_id = wf.get("id", "")
            if not wf_id:
                continue
            richness = workflow_richness(wf)
            # Skip stub workflows (≤2 states)
            if richness[0] <= 2 and richness[1] <= 2:
                continue
            wf_candidates[wf_id].append((richness, dirname, wf))

    # Select the best workflow for each ID
    merged_workflows: list[dict] = []
    for wf_id in WORKFLOW_IDS:
        candidates = wf_candidates.get(wf_id, [])
        if not candidates:
            print(f"    ⚠ {client_name}/{wf_id}: no non-stub version found")
            # Create a minimal placeholder
            merged_workflows.append({
                "id": wf_id,
                "name": wf_id.replace("_", " ").title(),
                "description": f"No substantive version found for {client_name}",
                "mode": "",
                "initial_state": f"{wf_id}.init",
                "states": [],
            })
            continue

        # Sort by total richness (states * 10 + transitions), descending
        candidates.sort(key=lambda c: (c[0][0] * 10 + c[0][1]), reverse=True)
        best_richness, best_dir, best_wf = candidates[0]
        print(
            f"    ✓ {client_name}/{wf_id}: best from {best_dir} "
            f"({best_richness[0]} states, {best_richness[1]} transitions) "
            f"[{len(candidates)} candidates]"
        )
        merged_workflows.append(best_wf)

    # Filter global vocab to only names referenced in merged workflows
    ref_guards, ref_actions = collect_referenced_names(merged_workflows)

    # Merge client-level vocab with global vocab, filter to referenced names
    client_guards = merge_vocab_entries(all_client_guards + global_guards)
    client_actions = merge_vocab_entries(all_client_actions + global_actions)
    filtered_guards = [g for g in client_guards if g.get("name") in ref_guards]
    filtered_actions = [a for a in client_actions if a.get("name") in ref_actions]

    return {
        "version": 1,
        "client": client_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "guards": filtered_guards,
        "actions": filtered_actions,
        "workflows": merged_workflows,
    }


def print_summary(
    merged_guards: list[dict],
    merged_actions: list[dict],
    client_lsgs: dict[str, dict],
) -> None:
    """Print a summary table of the merge results."""
    print()
    print("=" * 70)
    print("MERGE SUMMARY")
    print("=" * 70)
    print(f"\nGlobal vocabulary: {len(merged_guards)} guards, {len(merged_actions)} actions")

    # Guard categories
    guard_cats: dict[str, int] = defaultdict(int)
    for g in merged_guards:
        guard_cats[g.get("category", "other")] += 1
    print(f"\n  Guard categories:")
    for cat, count in sorted(guard_cats.items(), key=lambda x: -x[1]):
        print(f"    {cat}: {count}")

    # Action categories
    action_cats: dict[str, int] = defaultdict(int)
    for a in merged_actions:
        action_cats[a.get("category", "other")] += 1
    print(f"\n  Action categories:")
    for cat, count in sorted(action_cats.items(), key=lambda x: -x[1]):
        print(f"    {cat}: {count}")

    print(f"\n{'Client':<15} {'Guards':<8} {'Actions':<8} ", end="")
    for wf_id in WORKFLOW_IDS:
        short = wf_id[:8]
        print(f"{short:<12}", end="")
    print()
    print("-" * (15 + 8 + 8 + 12 * 7))

    for client_name in CLIENT_NAMES:
        lsg = client_lsgs[client_name]
        g_count = len(lsg.get("guards", []))
        a_count = len(lsg.get("actions", []))
        print(f"{client_name:<15} {g_count:<8} {a_count:<8} ", end="")
        for wf in lsg.get("workflows", []):
            states = wf.get("states", [])
            n_st = len(states)
            n_tr = sum(len(s.get("transitions", [])) for s in states)
            print(f"{n_st}s/{n_tr}t      ", end="")
        print()

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge all experiment results")
    parser.add_argument(
        "--output-dir",
        default="merged",
        help="Output directory name under project root (default: merged)",
    )
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Merging Phase 1 results (Global Vocabulary)")
    print("=" * 70)
    merged_guards, merged_actions = merge_global_vocab(RESULT_DIRS)
    print(f"\n→ Merged: {len(merged_guards)} guards, {len(merged_actions)} actions")

    # Write merged global spec
    global_spec = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "merge_source": RESULT_DIRS,
        "guards": merged_guards,
        "actions": merged_actions,
    }
    global_spec_path = output_dir / "Global_LSG_Spec_Enriched.yaml"
    with open(global_spec_path, "w", encoding="utf-8") as f:
        yaml.dump(global_spec, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"→ Written to: {global_spec_path}")

    print()
    print("=" * 70)
    print("Merging Phase 2 results (Per-Client LSGs)")
    print("=" * 70)

    client_lsgs: dict[str, dict] = {}
    for client_name in CLIENT_NAMES:
        print(f"\n── {client_name} ──")
        merged_lsg = merge_client_lsg(
            client_name, RESULT_DIRS, merged_guards, merged_actions,
        )
        client_lsgs[client_name] = merged_lsg

        lsg_path = output_dir / f"LSG_{client_name}_merged.yaml"
        with open(lsg_path, "w", encoding="utf-8") as f:
            yaml.dump(
                merged_lsg, f,
                default_flow_style=False, allow_unicode=True, sort_keys=False,
            )
        print(f"  → Written to: {lsg_path}")

    # Print summary
    print_summary(merged_guards, merged_actions, client_lsgs)

    print(f"All files written to: {output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()

