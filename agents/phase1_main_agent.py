"""Phase 1 main-agent: merge and normalize discovered vocabulary."""

from __future__ import annotations

from typing import Any

from agents.common import render_prompt
from agents.llm_factory import create_llm
from agents.schemas import EnrichedSpec, VocabDiscoveryReport


def run_phase1_main_agent(
    sub_reports: dict[str, VocabDiscoveryReport] | list[VocabDiscoveryReport],
    current_vocab: dict[str, list[dict]],
    vocab_version: int,
    iteration: int,
    llm: Any | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
) -> EnrichedSpec:
    """Merge sub-agent reports and produce EnrichedSpec via structured output."""

    if isinstance(sub_reports, dict):
        reports = list(sub_reports.values())
    else:
        reports = sub_reports

    prompt = render_prompt(
        "phase1_main.jinja2",
        iteration=iteration,
        vocab_version=vocab_version,
    )

    payload = {
        "current_vocab": current_vocab,
        "sub_reports": [r.model_dump() for r in reports],
    }

    llm = llm or create_llm(provider=llm_provider, model=llm_model)
    model = llm.with_structured_output(EnrichedSpec)
    enriched = model.invoke(
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": str(payload)},
        ]
    )

    if not isinstance(enriched, EnrichedSpec):
        enriched = EnrichedSpec.model_validate(enriched)

    if enriched.vocab_version <= vocab_version:
        enriched.vocab_version = vocab_version + 1

    return enriched
