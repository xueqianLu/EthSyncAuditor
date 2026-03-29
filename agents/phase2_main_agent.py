"""Phase 2 main-agent: compare multi-client LSGs and classify A/B diffs."""

from __future__ import annotations

from typing import Any

from agents.common import render_prompt
from agents.llm_factory import create_llm
from agents.schemas import DiffReport, LSGFileModel


def run_phase2_main_agent(
    client_lsgs: dict[str, LSGFileModel] | list[LSGFileModel],
    iteration: int,
    llm: Any | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
) -> DiffReport:
    """Compare LSGs using tri-key and return structured diff report."""

    if isinstance(client_lsgs, dict):
        lsgs = list(client_lsgs.values())
    else:
        lsgs = client_lsgs

    prompt = render_prompt(
        "phase2_main.jinja2",
        iteration=iteration,
    )

    payload = {
        "lsg_files": [l.model_dump() for l in lsgs],
        "comparison_key": ["workflow_id", "state_id", "transition_guard"],
    }

    llm = llm or create_llm(provider=llm_provider, model=llm_model)
    model = llm.with_structured_output(DiffReport)

    report = model.invoke(
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": str(payload)},
        ]
    )

    if not isinstance(report, DiffReport):
        report = DiffReport.model_validate(report)

    if report.iteration != iteration:
        report.iteration = iteration

    return report
