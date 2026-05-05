# coding=utf-8
"""Tests for run_autonomy_workflow -> Kernel delegation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from self_ai import main


@pytest.mark.asyncio
async def test_run_autonomy_workflow_delegates_to_kernel_run() -> None:
    fake_kernel = MagicMock()
    fake_kernel.run = AsyncMock(
        return_value={
            "response": "hello",
            "model": "mock",
            "run_id": "r1",
            "session_id": "s1",
            "task_profile": {"intent": "general_qa"},
            "workflow_decision": {"execution_mode": "quick_answer", "use_quick_answer": True},
            "plan": {},
            "context_pack": {},
            "selected_agents": [],
            "agent_outputs": {},
            "review_reports": [],
            "quality_gate": {},
            "errors": [],
        }
    )

    with (
        patch("self_ai.main.get_kernel", return_value=fake_kernel),
        patch("self_ai.main.get_chat_memory_store", return_value=None),
        patch("self_ai.main.get_chat_memory_service", return_value=None),
    ):
        result = await main.run_autonomy_workflow("What is this?")

    fake_kernel.run.assert_awaited_once()
    assert result["response"] == "hello"
    assert result["workflow_decision"]["execution_mode"] == "quick_answer"


@pytest.mark.asyncio
async def test_run_autonomy_workflow_applies_missing_field_defaults() -> None:
    fake_kernel = MagicMock()
    fake_kernel.run = AsyncMock(return_value={"response": "ok", "run_id": "r2", "session_id": "s2"})

    with (
        patch("self_ai.main.get_kernel", return_value=fake_kernel),
        patch("self_ai.main.get_chat_memory_store", return_value=None),
        patch("self_ai.main.get_chat_memory_service", return_value=None),
    ):
        result = await main.run_autonomy_workflow("Explain this")

    assert result["response"] == "ok"
    assert result["model"] == ""
    assert result["task_profile"] == {}
    assert result["workflow_decision"] == {}
    assert result["plan"] == {}
    assert result["context_pack"] == {}
    assert result["selected_agents"] == []
    assert result["agent_outputs"] == {}
    assert result["review_reports"] == []
    assert result["quality_gate"] == {}
    assert result["errors"] == []
    assert result["created_at"] == ""
    assert result["updated_at"] == ""
    assert result["metadata"]["agent_runtime_used"] is False
    assert result["metadata"]["revision_count"] == 0
    assert result["metadata"]["max_revision_iterations"] == 1
    assert result["metadata"]["revision_performed"] is False
    assert result["metadata"]["revision_target_agent"] is None
