# coding=utf-8
"""Writer role agent."""

from __future__ import annotations

from .base import AgentInput, BaseRoleAgent
from ..text_utils import shorten_text


class WriterAgent(BaseRoleAgent):
    name = "writer"
    role = "writer"
    model_category = "fast"
    output_type = "document"

    def build_prompt(self, agent_input: AgentInput) -> str:
        task = shorten_text(agent_input.task, max_chars=600)
        style_hint = ""
        if isinstance(agent_input.task_profile, dict):
            style_hint = str(agent_input.task_profile.get("domain", "general"))
        return (
            "You are WriterAgent.\n"
            "Focus: documentation, explanation text, technical writing and rewriting.\n"
            "Do not act as reviewer or quality gate.\n"
            "Output polished, direct writing.\n\n"
            f"Task:\n{task}\n"
            f"Domain hint: {style_hint}\n"
        )

