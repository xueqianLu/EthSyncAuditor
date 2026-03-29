"""Phase 1 sub-agent: discover missing Guard/Action vocabulary entries."""

from __future__ import annotations

from typing import Any

from agents.common import as_name_list, render_prompt
from agents.llm_factory import create_llm
from agents.schemas import VocabDiscoveryReport
from tools.search import search_codebase


def run_phase1_sub_agent(
    client_name: str,
    current_vocab: dict[str, list[dict]],
    iteration: int,
    llm: Any | None = None,
    agent_executor: Any | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
) -> VocabDiscoveryReport:
    """Run Phase 1 sub-agent with ReAct + structured output.

    - Uses search_codebase tool (mode A)
    - Returns VocabDiscoveryReport (Pydantic validated)
    """

    guard_names = as_name_list(current_vocab.get("guards", []))
    action_names = as_name_list(current_vocab.get("actions", []))

    prompt = render_prompt(
        "phase1_sub.jinja2",
        client_name=client_name,
        iteration=iteration,
        guard_names="\n".join(f"- {x}" for x in guard_names),
        action_names="\n".join(f"- {x}" for x in action_names),
    )

    llm = llm or create_llm(provider=llm_provider, model=llm_model)

    if agent_executor is None:
        from langgraph.prebuilt import create_react_agent

        agent_executor = create_react_agent(
            model=llm,
            tools=[search_codebase],
            prompt=prompt,
            response_format=VocabDiscoveryReport,
        )

    raw = agent_executor.invoke(
        {
            "messages": [
                {"role": "user", "content": prompt},
            ]
        }
    )

    if isinstance(raw, VocabDiscoveryReport):
        return raw

    structured = raw.get("structured_response") if isinstance(raw, dict) else raw
    report = VocabDiscoveryReport.model_validate(structured)

    if report.client_name != client_name:
        report.client_name = client_name  # normalize hard

    return report
