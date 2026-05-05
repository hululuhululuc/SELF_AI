# coding=utf-8
"""Tests for AgentLoop tool-only execution behavior."""

from __future__ import annotations

from typing import Any

import pytest

from self_ai.agent_runtime.agent_loop import AgentLoop
from self_ai.agent_runtime.agent_session import AgentSession
from self_ai.adapters.agent_tools import register_agent_tools
from self_ai.adapters.model_tool import register_model_tools
from self_ai.kernel.run_context import RunContext
from self_ai.runtime.tool_runtime import ToolRuntime
from self_ai.runtime.tool_result import ToolResult


class _FakeToolRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def execute_by_name(self, name: str, arguments: dict[str, Any], **_kwargs: Any) -> ToolResult:
        self.calls.append((name, arguments))
        if name == "agent.run":
            if arguments.get("agent_name") == "broken":
                return ToolResult.failure(
                    call_id="c-fail",
                    tool_name=name,
                    error_type="AgentFailure",
                    message="broken agent",
                )
            agent_name = str(arguments.get("agent_name"))
            return ToolResult.success(
                call_id="c-ok",
                tool_name=name,
                data={
                    "agent_result": {
                        "agent_name": agent_name,
                        "role": agent_name,
                        "output": f"{agent_name} output",
                        "summary": f"{agent_name} summary",
                        "confidence": 0.8,
                    }
                },
            )
        if name == "agent.synthesize":
            return ToolResult.success(
                call_id="c-syn",
                tool_name=name,
                data={
                    "agent_result": {
                        "agent_name": "synthesizer",
                        "role": "synthesizer",
                        "output": "merged output",
                        "summary": "merged summary",
                        "confidence": 0.9,
                    }
                },
            )
        return ToolResult.failure(call_id="c-miss", tool_name=name, error_type="Missing", message="missing")


@pytest.mark.asyncio
async def test_agent_loop_runs_agents_and_synthesizer() -> None:
    runtime = _FakeToolRuntime()
    loop = AgentLoop(tool_runtime=runtime)
    session = AgentSession(run_id="r1", session_id="s1", task="task")
    session.selected_agents = ["coder", "synthesizer"]

    await loop.run_agents(session, agents=session.selected_agents)

    assert "coder" in session.agent_outputs
    assert "synthesizer" in session.agent_outputs
    assert any(call[0] == "agent.run" for call in runtime.calls)
    assert any(call[0] == "agent.synthesize" for call in runtime.calls)


@pytest.mark.asyncio
async def test_agent_loop_failure_adds_errors() -> None:
    runtime = _FakeToolRuntime()
    loop = AgentLoop(tool_runtime=runtime)
    session = AgentSession(run_id="r2", session_id="s2", task="task")

    ok = await loop.run_agent(session, agent_name="broken", instruction="fix this")
    assert ok is False
    assert len(session.errors) == 1
    assert session.errors[0]["type"] == "AgentFailure"


@pytest.mark.asyncio
async def test_agent_loop_uses_model_generate_under_agent_tools(tmp_path) -> None:
    calls: list[tuple[str, str]] = []

    async def _fake_route_model(prompt: str, category: str) -> dict[str, Any]:
        calls.append((prompt, category))
        return {"model": "mock-model", "response": "ok"}

    runtime = ToolRuntime()
    register_model_tools(runtime.registry, route_model_func=_fake_route_model)
    register_agent_tools(runtime.registry)
    ctx = RunContext(project_root=tmp_path, tool_runtime=runtime, permission_profile={"model_call": True})

    loop = AgentLoop(tool_runtime=runtime)
    session = AgentSession(run_id="r-model", session_id="s-model", task="implement function")
    session.selected_agents = ["coder", "synthesizer"]

    await loop.run_agents(session, agents=session.selected_agents, run_context=ctx)

    assert "coder" in session.agent_outputs
    assert "synthesizer" in session.agent_outputs
    assert len(calls) >= 2
    categories = {category for _, category in calls}
    assert "coding" in categories
    assert "reasoning" in categories
