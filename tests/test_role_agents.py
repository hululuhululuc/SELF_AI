# coding=utf-8
"""Tests for Phase 6A role agents."""

from unittest.mock import AsyncMock

import pytest

from self_ai.agents.architect import ArchitectAgent
from self_ai.agents.base import AgentInput
from self_ai.agents.coder import CoderAgent
from self_ai.agents.researcher import ResearcherAgent
from self_ai.agents.synthesizer import SynthesizerAgent
from self_ai.agents.writer import WriterAgent


def _agent_input() -> AgentInput:
    return AgentInput(
        run_id="run-1",
        session_id="default",
        task="Design and implement a robust workflow",
        task_profile={"intent": "architecture_design", "domain": "software"},
        workflow_decision={"execution_mode": "architecture_design"},
        plan={"steps": [{"step_id": "s1", "goal": "analyze scope"}]},
        context_pack={"context_preview": "evidence summary"},
        research_results=[{"content": "result 1"}],
        metadata={
            "prior_agent_outputs": [
                {"agent_name": "architect", "summary": "architecture summary"}
            ]
        },
    )


@pytest.mark.parametrize(
    ("agent", "marker"),
    [
        (CoderAgent(), "You are CoderAgent"),
        (ArchitectAgent(), "You are ArchitectAgent"),
        (ResearcherAgent(), "You are ResearcherAgent"),
        (WriterAgent(), "You are WriterAgent"),
        (SynthesizerAgent(), "You are SynthesizerAgent"),
    ],
)
def test_role_agent_build_prompt(agent, marker: str) -> None:
    prompt = agent.build_prompt(_agent_input())
    assert marker in prompt
    assert "Do not act as reviewer" in prompt or "Do not do pass/fail judgment" in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "agent",
    [CoderAgent(), ArchitectAgent(), ResearcherAgent(), WriterAgent(), SynthesizerAgent()],
)
async def test_role_agent_run_returns_agent_result(agent) -> None:
    model_func = AsyncMock(return_value={"model": "mock-model", "response": "role output"})
    result = await agent.run(_agent_input(), route_model_func=model_func)
    assert result.run_id == "run-1"
    assert result.session_id == "default"
    assert result.agent_name == agent.name
    assert result.role == agent.role
    assert result.output == "role output"
    assert result.summary
    assert result.output_type == agent.output_type


def test_synthesizer_prompt_uses_prior_agent_outputs() -> None:
    agent = SynthesizerAgent()
    prompt = agent.build_prompt(_agent_input())
    assert "Prior role outputs:" in prompt
    assert "architecture summary" in prompt

