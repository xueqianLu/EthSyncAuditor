"""Shared helpers for Step 4 agents."""

from __future__ import annotations

from pathlib import Path

from config import WORKFLOW_IDS

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


def render_prompt(template_name: str, **kwargs) -> str:
    from jinja2 import Environment, FileSystemLoader, StrictUndefined

    env = Environment(
        loader=FileSystemLoader(str(PROMPT_DIR)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template(template_name)
    return template.render(**kwargs)


def as_name_list(entries: list[dict] | list[object], field: str = "name") -> list[str]:
    names: list[str] = []
    for item in entries:
        if isinstance(item, dict):
            value = item.get(field)
        else:
            value = getattr(item, field, None)
        if isinstance(value, str) and value:
            names.append(value)
    return sorted(set(names))


def workflow_ids_text() -> str:
    return "\n".join(f"- {wf}" for wf in WORKFLOW_IDS)
