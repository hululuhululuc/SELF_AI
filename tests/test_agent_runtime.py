# coding=utf-8
"""Tests for AgentRuntime integration unit."""

from __future__ import annotations

from pathlib import Path

import pytest

from self_ai.agent_runtime.runtime import AgentRuntime
from self_ai.kernel.engine_state import EngineState
from tests._engine_helpers import build_test_kernel


@pytest.mark.asyncio
async def test_agent_runtime_quick_answer_skips_execution(tmp_path: Path) -> None:
    kernel, redis_store = build_test_kernel(tmp_path)
    runtime = AgentRuntime(tool_runtime=kernel.tool_runtime, state_store=kernel.state_store)
    state = EngineState.from_input("What is Redis?", run_id="r-qa", session_id="s1")
    state.workflow_decision = {"use_quick_answer": True, "execution_mode": "quick_answer"}

    session = await runtime.run(state, run_context=kernel.new_run_context(run_id="r-qa", session_id="s1"))

    assert session.selected_agents == []
    assert session.review_reports == []
    assert session.quality_gate == {}
    node_names = [name for _, name, _ in redis_store.node_outputs]
    assert "agent_runtime" in node_names


@pytest.mark.asyncio
async def test_agent_runtime_non_quick_runs_agents_review_gate(tmp_path: Path) -> None:
    kernel, redis_store = build_test_kernel(tmp_path)
    runtime = AgentRuntime(tool_runtime=kernel.tool_runtime, state_store=kernel.state_store)
    state = EngineState.from_input("Architecture design for runtime", run_id="r-nq", session_id="s1")
    state.task_profile = {"intent": "architecture_design"}
    state.workflow_decision = {"use_quick_answer": False, "execution_mode": "architecture_design", "max_iterations": 1}

    session = await runtime.run(state, run_context=kernel.new_run_context(run_id="r-nq", session_id="s1"))

    assert session.selected_agents == ["architect", "synthesizer"]
    assert "architect" in session.agent_outputs
    assert len(session.review_reports) >= 1
    assert isinstance(session.quality_gate, dict)
    trace_events = [event["event"] for _, event in redis_store.traces]
    assert "agent_runtime.start" in trace_events
    assert "agent_runtime.end" in trace_events
    assert "revision_loop.start" in trace_events
    assert "revision_loop.end" in trace_events
