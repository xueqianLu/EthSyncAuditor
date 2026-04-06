"""EthAuditor — shared utility functions."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def safe_serialize(obj: Any) -> Any:
    """Recursively convert an object to a JSON-serializable form."""
    if isinstance(obj, dict):
        return {k: safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_serialize(item) for item in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return {k: safe_serialize(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
    return str(obj)


def invoke_with_retry(
    chain: Any,
    prompt: Any,
    *,
    max_retries: int = 3,
    base_delay: float = 5.0,
    max_delay: float = 60.0,
    label: str = "LLM",
    callbacks: list[Any] | None = None,
) -> Any:
    """Invoke a LangChain chain/runnable with exponential-backoff retries.

    Retries on any exception (network errors, rate limits, server disconnects).

    Parameters
    ----------
    chain:
        A LangChain runnable (e.g. ``llm.with_structured_output(...)``).
    prompt:
        The prompt to pass to ``chain.invoke()``.
    max_retries:
        Maximum number of retry attempts after the first failure.
    base_delay:
        Initial delay in seconds before the first retry.
    max_delay:
        Upper cap for the delay between retries.
    label:
        Human-readable label for log messages.
    callbacks:
        Optional list of LangChain callback handlers to attach to this
        invocation (passed via ``config`` dict).

    Returns
    -------
    The result of ``chain.invoke(prompt)``.

    Raises
    ------
    Exception
        Re-raises the last exception if all retries are exhausted.
    """
    invoke_kwargs: dict[str, Any] = {}
    if callbacks:
        invoke_kwargs["config"] = {"callbacks": callbacks}

    last_exc: Exception | None = None
    for attempt in range(1 + max_retries):
        try:
            return chain.invoke(prompt, **invoke_kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                delay = min(base_delay * (2 ** attempt), max_delay)
                logger.warning(
                    "[%s] attempt %d/%d failed (%s: %s) — retrying in %.1fs",
                    label, attempt + 1, 1 + max_retries,
                    type(exc).__name__, exc, delay,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "[%s] all %d attempts failed — giving up",
                    label, 1 + max_retries,
                )
    raise last_exc  # type: ignore[misc]


def summarize_vocab_for_prompt(
    guards: list[dict],
    actions: list[dict],
    *,
    max_full_entries: int = 80,
) -> dict[str, Any]:
    """Produce a compact vocabulary representation for Phase 2 prompts.

    Instead of dumping all 500+ guards/actions inline (which blows up the
    prompt), this returns:

    - ``guard_summary``: ``{category: [name, …]}`` — names only, grouped.
    - ``action_summary``: ``{category: [name, …]}`` — names only, grouped.
    - ``guard_details``: Up to *max_full_entries / 2* full entries
      (name + category + description) drawn from the most common categories.
    - ``action_details``: Same for actions.
    - ``total_guards`` / ``total_actions``: integer counts.
    """
    def _group_by_category(entries: list[dict]) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = defaultdict(list)
        for e in entries:
            groups[e.get("category", "other")].append(e.get("name", "?"))
        return dict(groups)

    def _select_details(entries: list[dict], limit: int) -> list[dict]:
        """Pick entries from the largest categories first."""
        cat_counts: dict[str, int] = defaultdict(int)
        for e in entries:
            cat_counts[e.get("category", "other")] += 1
        # Sort categories by count (descending) to prioritise the richest
        ranked_cats = sorted(cat_counts, key=lambda c: cat_counts[c], reverse=True)
        selected: list[dict] = []
        per_cat = max(limit // max(len(ranked_cats), 1), 2)
        for cat in ranked_cats:
            cat_entries = [e for e in entries if e.get("category") == cat]
            selected.extend(cat_entries[:per_cat])
            if len(selected) >= limit:
                break
        return selected[:limit]

    half = max_full_entries // 2
    return {
        "guard_summary": _group_by_category(guards),
        "action_summary": _group_by_category(actions),
        "guard_details": _select_details(guards, half),
        "action_details": _select_details(actions, half),
        "total_guards": len(guards),
        "total_actions": len(actions),
    }


def serialize_lsg_compact(
    lsg: dict[str, Any],
    *,
    max_lines: int = 2500,
    strip_evidence: bool = False,
) -> str:
    """Serialize a client LSG dict to a compact YAML string for prompts.

    Parameters
    ----------
    lsg:
        The full LSG dict (as stored in ``client_lsgs[client]``).
    max_lines:
        If the serialized output exceeds this many lines, evidence fields
        are automatically stripped and the output is re-serialized.
    strip_evidence:
        If ``True``, always remove evidence blocks from transitions.
    """
    def _strip(obj: dict) -> dict:
        """Deep-copy the LSG with all evidence fields removed."""
        out = {}
        for k, v in obj.items():
            if k in ("evidence", "evidence_file", "evidence_function", "evidence_lines"):
                continue
            if k == "workflows":
                out[k] = []
                for wf in v:
                    wf_copy = dict(wf)
                    new_states = []
                    for st in wf_copy.get("states", []):
                        st_copy = dict(st)
                        new_trans = []
                        for tr in st_copy.get("transitions", []):
                            new_trans.append({
                                kk: vv for kk, vv in tr.items()
                                if kk not in ("evidence",)
                            })
                        st_copy["transitions"] = new_trans
                        new_states.append(st_copy)
                    wf_copy["states"] = new_states
                    out[k].append(wf_copy)
            elif k.startswith("_"):
                continue  # skip internal keys like _iteration
            else:
                out[k] = v
        return out

    data = _strip(lsg) if strip_evidence else {
        k: v for k, v in lsg.items() if not k.startswith("_")
    }
    text = yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)

    if text.count("\n") > max_lines and not strip_evidence:
        # Re-try without evidence to save tokens
        data = _strip(lsg)
        text = yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)

    return text


def compute_lsg_sparsity(client_lsgs: dict[str, dict]) -> list[dict]:
    """Return per-client, per-workflow sparsity hints.

    For each workflow with fewer than *min_states* states or *min_transitions*
    transitions, emit a hint dict that the Phase 2 sub-agent prompt can use
    to know where to focus expansion.
    """
    MIN_STATES = 4
    MIN_TRANSITIONS = 4
    hints: list[dict] = []
    for client, lsg in client_lsgs.items():
        for wf in lsg.get("workflows", []):
            wf_id = wf.get("id", "?")
            n_states = len(wf.get("states", []))
            n_trans = sum(
                len(st.get("transitions", []))
                for st in wf.get("states", [])
            )
            if n_states < MIN_STATES or n_trans < MIN_TRANSITIONS:
                hints.append({
                    "client": client,
                    "workflow_id": wf_id,
                    "states": n_states,
                    "transitions": n_trans,
                    "description": (
                        f"{client}/{wf_id} is sparse: only {n_states} states "
                        f"and {n_trans} transitions — needs expansion."
                    ),
                })
    return hints
