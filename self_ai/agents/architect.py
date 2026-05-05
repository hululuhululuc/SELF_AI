# coding=utf-8
"""Architect role agent."""

from __future__ import annotations

from .base import AgentInput, BaseRoleAgent
from ..text_utils import shorten_text


class ArchitectAgent(BaseRoleAgent):
    name = "architect"
    role = "architect"
    model_category = "reasoning"
    output_type = "architecture_design"

    def build_prompt(self, agent_input: AgentInput) -> str:
        task = shorten_text(agent_input.task, max_chars=600)
        decision = agent_input.workflow_decision if isinstance(agent_input.workflow_decision, dict) else {}
        return (
            "You are ArchitectAgent.\n"
            "Focus: architecture boundaries, phased rollout, storage/retrieval workflow design.\n"
            "Do not act as reviewer or quality gate.\n"
            "Provide concise architecture-oriented output.\n\n"
            f"Task:\n{task}\n"
            f"Execution mode: {decision.get('execution_mode', 'general')}\n"
            f"Requires context: {decision.get('requires_context', False)}\n"
        )

