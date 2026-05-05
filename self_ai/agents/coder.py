# coding=utf-8
"""Coder role agent."""

from __future__ import annotations

from .base import AgentInput, BaseRoleAgent
from ..text_utils import shorten_text


class CoderAgent(BaseRoleAgent):
    name = "coder"
    role = "coder"
    model_category = "coding"
    output_type = "code_suggestion"

    def build_prompt(self, agent_input: AgentInput) -> str:
        task = shorten_text(agent_input.task, max_chars=600)
        plan_steps = agent_input.plan.get("steps", []) if isinstance(agent_input.plan, dict) else []
        plan_hint = ""
        if isinstance(plan_steps, list) and plan_steps:
            first = plan_steps[0] if isinstance(plan_steps[0], dict) else {}
            plan_hint = shorten_text(str(first.get("goal", "")), max_chars=160)
        context_preview = ""
        if isinstance(agent_input.context_pack, dict):
            context_preview = shorten_text(
                str(agent_input.context_pack.get("context_preview", "")),
                max_chars=320,
            )
        return (
            "You are CoderAgent.\n"
            "Focus: code generation/modification/debugging implementation suggestions.\n"
            "Do not act as reviewer or quality gate.\n"
            "Output practical implementation notes or code-oriented answer.\n\n"
            f"Task:\n{task}\n"
            f"Plan hint: {plan_hint}\n"
            f"Context preview: {context_preview}\n"
        )

