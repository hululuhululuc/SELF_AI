# coding=utf-8
"""Tests for adapters.agent_tools model.generate path."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from self_ai.adapters.agent_tools import register_agent_tools
from self_ai.adapters.model_tool import register_model_tools
from self_ai.kernel.run_context import RunContext
from self_ai.runtime.tool_runtime import ToolRuntime


def _make_agent_input() -> dict[str, Any]:
    return {
        "run_id": "r-agent",
        "session_id": "s-agent",
        "task": "Write a function",
        "task_profile": {"intent": "code_generation"},
        "workflow_decision": {"execution_mode": "code_focused"},
        "plan": {"steps": []},
        "context_pack": {},
        "research_results": [],
        "debate_messages": [],
        "metadata": {},
    }


@pytest.mark.asyncio
async def test_agent_run_calls_model_generate(tmp_path: Path) -> None:
    model_calls: list[tuple[str, str]] = []

    async def _fake_route_model(prompt: str, category: str) -> dict[str, Any]:
        model_calls.append((prompt, category))
        return {"model": "mock-coder", "response": "def avg(xs): return sum(xs)/len(xs)"}

    runtime = ToolRuntime()
    register_model_tools(runtime.registry, route_model_func=_fake_route_model)
    register_agent_tools(runtime.registry)
    ctx = RunContext(project_root=tmp_path, tool_runtime=runtime, permission_profile={"model_call": True})

    result = await runtime.execute_by_name(
        "agent.run",
        {
            "agent_name": "coder",
            "agent_input": _make_agent_input(),
            "instruction": "handle empty list",
        },
        run_id="r-agent",
        session_id="s-agent",
        run_context=ctx,
    )

    assert result.ok is True
    agent_result = result.data["agent_result"]
    assert agent_result["agent_name"] == "coder"
    assert agent_result["metadata"]["model"] == "mock-coder"
    assert len(model_calls) == 1
    prompt, category = model_calls[0]
    assert "Revision instruction" in prompt
    assert category == "coding"


@pytest.mark.asyncio
async def test_agent_synthesize_calls_model_generate(tmp_path: Path) -> None:
    model_calls: list[tuple[str, str]] = []

    async def _fake_route_model(prompt: str, category: str) -> dict[str, Any]:
        model_calls.append((prompt, category))
        return {"model": "mock-synth", "response": "merged answer"}

    runtime = ToolRuntime()
    register_model_tools(runtime.registry, route_model_func=_fake_route_model)
    register_agent_tools(runtime.registry)
    ctx = RunContext(project_root=tmp_path, tool_runtime=runtime, permission_profile={"model_call": True})

    result = await runtime.execute_by_name(
        "agent.synthesize",
        {
            "agent_input": _make_agent_input(),
            "agent_outputs_summary": [{"agent_name": "coder", "summary": "code done"}],
        },
        run_id="r-agent",
        session_id="s-agent",
        run_context=ctx,
    )

    assert result.ok is True
    assert result.data["agent_result"]["agent_name"] == "synthesizer"
    assert len(model_calls) == 1
    assert model_calls[0][1] == "reasoning"


@pytest.mark.asyncio
async def test_agent_run_safe_failure_when_model_generate_fails(tmp_path: Path) -> None:
    async def _boom(_prompt: str, _category: str) -> dict[str, Any]:
        raise RuntimeError("model unavailable")

    runtime = ToolRuntime()
    register_model_tools(runtime.registry, route_model_func=_boom)
    register_agent_tools(runtime.registry)
    ctx = RunContext(project_root=tmp_path, tool_runtime=runtime, permission_profile={"model_call": True})

    result = await runtime.execute_by_name(
        "agent.run",
        {"agent_name": "coder", "agent_input": _make_agent_input()},
        run_id="r-agent",
        session_id="s-agent",
        run_context=ctx,
    )

    assert result.ok is False
    assert result.error is not None
    assert result.error["type"] == "RuntimeError"


def test_agent_tools_file_does_not_import_route_model() -> None:
    content = Path("adapters/agent_tools.py").read_text(encoding="utf-8")
    assert "from ..router import route_model" not in content
