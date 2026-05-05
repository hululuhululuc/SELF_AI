# coding=utf-8
"""Researcher role agent."""

from __future__ import annotations

from .base import AgentInput, BaseRoleAgent
from ..text_utils import shorten_text


class ResearcherAgent(BaseRoleAgent):
    name = "researcher"
    role = "researcher"
    model_category = "research"
    output_type = "research_summary"

    def build_prompt(self, agent_input: AgentInput) -> str:
        task = shorten_text(agent_input.task, max_chars=500)
        count = len(agent_input.research_results)
        context_preview = ""
        if isinstance(agent_input.context_pack, dict):
            context_preview = shorten_text(
                str(agent_input.context_pack.get("context_preview", "")),
                max_chars=360,
            )
        return (
            "You are ResearcherAgent.\n"
            "Focus: organize semantic evidence and summarize factual context.\n"
            "Do not act as reviewer or quality gate.\n"
            "Keep output concise and source-aware.\n\n"
            f"Task:\n{task}\n"
            f"Research result count: {count}\n"
            f"Evidence preview: {context_preview}\n"
        )

