# coding=utf-8
"""Unit tests for Phase 6A agent base contracts."""

from unittest.mock import AsyncMock

import pytest

from self_ai.agents.base import AgentInput, AgentResult, BaseRoleAgent


class DummyAgent(BaseRoleAgent):
    name = "dummy"
    role = "dummy"
    model_category = "fast"

    def build_prompt(self, agent_input: AgentInput) -> str:
        return f"dummy prompt: {agent_input.task}"


def test_agent_input_defaults() -> None:
    value = AgentInput()
    dumped = value.model_dump(mode="json")
    assert dumped["session_id"] == "default"
    assert dumped["task"] == ""
    assert dumped["plan"] == {}
    assert dumped["context_pack"] == {}


def test_agent_result_json_stable() -> None:
    result = AgentResult(
        run_id="r1",
        session_id="s1",
        agent_name="coder",
        role="coder",
        output="x",
    )
    dumped = result.model_dump(mode="json")
    assert dumped["run_id"] == "r1"
    assert dumped["agent_name"] == "coder"
    assert "confidence" in dumped
    parsed = AgentResult(**dumped)
    assert parsed.agent_name == "coder"


@pytest.mark.asyncio
async def test_base_role_agent_run_success_with_injected_model() -> None:
    agent = DummyAgent()
    model_func = AsyncMock(return_value={"model": "mock-fast", "response": "done"})
    agent_input = AgentInput(
        run_id="r1",
        session_id="s1",
        task="build feature",
        context_pack={"context_preview": "ctx"},
        plan={"steps": [{"step_id": "s1"}]},
        research_results=[{"content": "evidence"}],
    )
    result = await agent.run(agent_input, route_model_func=model_func)
    assert result.agent_name == "dummy"
    assert result.role == "dummy"
    assert result.output == "done"
    assert result.output_type == "text"
    assert "context_pack" in result.used_context
    assert "plan" in result.used_context
    assert result.metadata["model"] == "mock-fast"


@pytest.mark.asyncio
async def test_base_role_agent_run_failure_returns_safe_result() -> None:
    agent = DummyAgent()
    model_func = AsyncMock(side_effect=RuntimeError("boom"))
    agent_input = AgentInput(run_id="r2", session_id="s2", task="x")
    result = await agent.run(agent_input, route_model_func=model_func)
    assert result.agent_name == "dummy"
    assert result.output == ""
    assert result.confidence <= 0.2
    assert result.warnings
    assert result.warnings[0].startswith("model_error:")

