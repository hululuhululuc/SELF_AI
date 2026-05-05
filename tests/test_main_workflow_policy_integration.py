# coding=utf-8
"""Workflow decision checks under MainLoop semantics."""

from pathlib import Path
from unittest.mock import patch

import pytest

from self_ai import main
from tests._engine_helpers import build_test_kernel


@pytest.mark.asyncio
async def test_quick_answer_path_sets_quick_answer_mode(tmp_path: Path) -> None:
    kernel, redis_store = build_test_kernel(tmp_path)
    with patch("self_ai.main.get_kernel", return_value=kernel):
        result = await main.run_autonomy_workflow("What is this project in one sentence?")

    decision = result["workflow_decision"]
    assert decision["execution_mode"] == "quick_answer"
    assert decision["use_quick_answer"] is True
    node_names = [name for _, name, _ in redis_store.node_outputs]
    assert "mainloop_final_answer" in node_names
    assert "mainloop_finalize" in node_names


@pytest.mark.asyncio
async def test_research_summary_path_keeps_mainloop_contract(tmp_path: Path) -> None:
    kernel, redis_store = build_test_kernel(tmp_path)
    with patch("self_ai.main.get_kernel", return_value=kernel):
        result = await main.run_autonomy_workflow("Please provide a research summary of storage layers")

    decision = result["workflow_decision"]
    assert isinstance(decision, dict)
    node_names = [name for _, name, _ in redis_store.node_outputs]
    assert "mainloop_finalize" in node_names
    assert result["response"]


@pytest.mark.asyncio
async def test_architecture_design_path_keeps_mainloop_result_contract(tmp_path: Path) -> None:
    kernel, redis_store = build_test_kernel(tmp_path)
    with patch("self_ai.main.get_kernel", return_value=kernel):
        result = await main.run_autonomy_workflow("Architecture design for next phase")

    decision = result["workflow_decision"]
    assert isinstance(decision, dict)
    node_names = [name for _, name, _ in redis_store.node_outputs]
    assert "mainloop_finalize" in node_names
    assert isinstance(result["review_reports"], list)
    assert isinstance(result["quality_gate"], dict)
