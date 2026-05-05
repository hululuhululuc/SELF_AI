# coding=utf-8
"""Tests for SelfAIKernel.run production path."""

from pathlib import Path

import pytest

from tests._engine_helpers import build_test_kernel


@pytest.mark.asyncio
async def test_kernel_run_returns_required_result_contract(tmp_path: Path) -> None:
    kernel, _ = build_test_kernel(tmp_path)

    result = await kernel.run("Architecture design for test system", session_id="session-k")

    assert result["response"]
    assert result["model"]
    assert result["run_id"]
    assert result["session_id"] == "session-k"
    assert isinstance(result["task_profile"], dict)
    assert isinstance(result["workflow_decision"], dict)
    assert isinstance(result["plan"], dict)
    assert isinstance(result["context_pack"], dict)
    assert isinstance(result["selected_agents"], list)
    assert isinstance(result["agent_outputs"], dict)
    assert isinstance(result["review_reports"], list)
    assert isinstance(result["quality_gate"], dict)
    assert isinstance(result["errors"], list)
    assert result["created_at"]
    assert result["updated_at"]
    assert isinstance(result["metadata"], dict)
    assert "agent_runtime_used" in result["metadata"]
    assert "revision_count" in result["metadata"]
    assert "max_revision_iterations" in result["metadata"]
    assert "revision_performed" in result["metadata"]
    assert "revision_target_agent" in result["metadata"]


@pytest.mark.asyncio
async def test_kernel_run_quick_answer_mode(tmp_path: Path) -> None:
    kernel, _ = build_test_kernel(tmp_path)

    result = await kernel.run("What is Qdrant?", session_id="session-q")

    assert result["workflow_decision"]["use_quick_answer"] is True
    assert result["selected_agents"] == []
    assert result["review_reports"] == []
    assert result["quality_gate"] == {}


@pytest.mark.asyncio
async def test_kernel_new_run_context_event_sink_defaults_and_override(tmp_path: Path) -> None:
    events: list[dict[str, str]] = []
    kernel, _ = build_test_kernel(tmp_path)
    kernel.event_callback = lambda payload: events.append(payload)

    ctx_default = kernel.new_run_context(session_id="s-default")
    assert ctx_default.event_sink is kernel.event_callback

    custom_sink = lambda payload: payload
    ctx_custom = kernel.new_run_context(session_id="s-custom", event_sink=custom_sink)
    assert ctx_custom.event_sink is custom_sink
