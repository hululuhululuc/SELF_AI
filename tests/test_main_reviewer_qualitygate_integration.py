# coding=utf-8
"""Review/quality-gate contract checks on MainLoop path."""

from pathlib import Path
from unittest.mock import patch

import pytest

from self_ai import main
from tests._engine_helpers import build_test_kernel


@pytest.mark.asyncio
async def test_full_path_keeps_review_and_quality_gate_contract(tmp_path: Path) -> None:
    kernel, redis_store = build_test_kernel(tmp_path)
    with patch("self_ai.main.get_kernel", return_value=kernel):
        result = await main.run_autonomy_workflow("Architecture design for next release")

    assert isinstance(result["review_reports"], list)
    assert isinstance(result["quality_gate"], dict)

    node_names = [name for _, name, _ in redis_store.node_outputs]
    assert "mainloop_finalize" in node_names


@pytest.mark.asyncio
async def test_quick_answer_keeps_empty_review_and_gate_safely(tmp_path: Path) -> None:
    kernel, _ = build_test_kernel(tmp_path)
    with patch("self_ai.main.get_kernel", return_value=kernel):
        result = await main.run_autonomy_workflow("Explain Qdrant briefly")

    assert result["workflow_decision"]["use_quick_answer"] is True
    assert isinstance(result["review_reports"], list)
    assert isinstance(result["quality_gate"], dict)


@pytest.mark.asyncio
async def test_workflow_completes_when_review_tool_fails(tmp_path: Path) -> None:
    kernel, _ = build_test_kernel(tmp_path)

    original = kernel.tool_runtime.execute_by_name

    async def _wrapped(name, arguments=None, **kwargs):
        if name == "review.run":
            from self_ai.runtime.tool_result import ToolResult

            return ToolResult.failure(
                call_id="review-fail",
                tool_name=name,
                error_type="ReviewError",
                message="review unavailable",
            )
        return await original(name, arguments or {}, **kwargs)

    kernel.tool_runtime.execute_by_name = _wrapped  # type: ignore[assignment]

    with patch("self_ai.main.get_kernel", return_value=kernel):
        result = await main.run_autonomy_workflow("Architecture design with review")

    assert result["response"]
    assert isinstance(result["errors"], list)
