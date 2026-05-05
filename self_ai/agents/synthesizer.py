# coding=utf-8
"""Synthesizer role agent."""

from __future__ import annotations

from .base import AgentInput, BaseRoleAgent
from ..text_utils import shorten_text


class SynthesizerAgent(BaseRoleAgent):
    name = "synthesizer"
    role = "synthesizer"
    model_category = "reasoning"
    output_type = "synthesized_response"

    def build_prompt(self, agent_input: AgentInput) -> str:
        task = shorten_text(agent_input.task, max_chars=520)
        prior_outputs = []
        metadata = agent_input.metadata if isinstance(agent_input.metadata, dict) else {}
        raw_prior = metadata.get("prior_agent_outputs", [])
        if isinstance(raw_prior, list):
            for item in raw_prior[:5]:
                if not isinstance(item, dict):
                    continue
                prior_outputs.append(
                    f"[{item.get('agent_name', 'agent')}] {shorten_text(str(item.get('summary', '')), max_chars=140)}"
                )
        context_preview = ""
        if isinstance(agent_input.context_pack, dict):
            context_preview = shorten_text(
                str(agent_input.context_pack.get("context_preview", "")),
                max_chars=320,
            )
        prior_text = "\n".join(prior_outputs) if prior_outputs else "(none)"
        return (
            "You are SynthesizerAgent.\n"
            "Focus: merge role-agent outputs into one coherent response candidate.\n"
            "Do not do pass/fail judgment or quality gate.\n\n"
            f"Task:\n{task}\n"
            f"Prior role outputs:\n{prior_text}\n"
            f"Context preview: {context_preview}\n"
        )

