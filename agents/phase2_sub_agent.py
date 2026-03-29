"""Phase 2 sub-agent: extract client LSG workflows with mandatory evidence."""

from __future__ import annotations

from typing import Any

from agents.common import as_name_list, render_prompt, workflow_ids_text
from agents.llm_factory import create_llm
from agents.schemas import LSGFileModel
from tools.search import search_codebase_by_workflow


def run_phase2_sub_agent(
    client_name: str,
    enriched_spec: dict[str, list[dict]],
    iteration: int,
    llm: Any | None = None,
    agent_executor: Any | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
) -> LSGFileModel:
    """Run Phase 2 sub-agent with workflow-guided ReAct retrieval."""

    guard_names = as_name_list(enriched_spec.get("guards", []))
    action_names = as_name_list(enriched_spec.get("actions", []))

    prompt = render_prompt(
        "phase2_sub.jinja2",
        client_name=client_name,
        iteration=iteration,
        workflow_ids=workflow_ids_text(),
        guard_names="\n".join(f"- {x}" for x in guard_names),
        action_names="\n".join(f"- {x}" for x in action_names),
    )

    llm = llm or create_llm(provider=llm_provider, model=llm_model)

    if agent_executor is None:
        from langgraph.prebuilt import create_react_agent

        agent_executor = create_react_agent(
            model=llm,
            tools=[search_codebase_by_workflow],
            prompt=prompt,
            response_format=LSGFileModel,
        )

    raw = agent_executor.invoke(
        {
            "messages": [
                {"role": "user", "content": prompt},
            ]
        }
    )

    if isinstance(raw, LSGFileModel):
        result = raw
    else:
        structured = raw.get("structured_response") if isinstance(raw, dict) else raw
        result = LSGFileModel.model_validate(structured)

    if result.client != client_name:
        result.client = client_name

    return result
