# coding=utf-8
"""Role-agent output contract checks on MainLoop path."""

from pathlib import Path
from unittest.mock import patch

import pytest

from self_ai import main
from tests._engine_helpers import build_test_kernel


@pytest.mark.asyncio
async def test_architecture_design_returns_selected_agents_and_outputs_contract(tmp_path: Path) -> None:
    kernel, redis_store = build_test_kernel(tmp_path)
    with patch("self_ai.main.get_kernel", return_value=kernel):
        result = await main.run_autonomy_workflow("Architecture design for storage-aware agent system")

    assert isinstance(result["selected_agents"], list)
    assert isinstance(result["agent_outputs"], dict)

    node_names = [name for _, name, _ in redis_store.node_outputs]
    assert "mainloop_finalize" in node_names


@pytest.mark.asyncio
async def test_code_generation_returns_agent_contract_fields(tmp_path: Path) -> None:
    kernel, _ = build_test_kernel(tmp_path)
    with patch("self_ai.main.get_kernel", return_value=kernel):
        result = await main.run_autonomy_workflow("Write a Python function to compute average")

    assert isinstance(result["workflow_decision"], dict)
    assert isinstance(result["selected_agents"], list)
    assert isinstance(result["agent_outputs"], dict)


@pytest.mark.asyncio
async def test_quick_answer_keeps_agents_empty_or_safe(tmp_path: Path) -> None:
    kernel, _ = build_test_kernel(tmp_path)
    with patch("self_ai.main.get_kernel", return_value=kernel):
        result = await main.run_autonomy_workflow("What does Redis do?")

    assert result["workflow_decision"]["use_quick_answer"] is True
    assert isinstance(result["selected_agents"], list)
    assert isinstance(result["agent_outputs"], dict)
