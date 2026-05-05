# coding=utf-8
"""Kernel-path runtime state integration tests (MainLoop semantics)."""

from pathlib import Path
from unittest.mock import patch

import pytest

from self_ai import main
from tests._engine_helpers import build_test_kernel


@pytest.mark.asyncio
async def test_run_autonomy_workflow_persists_runtime_state_via_tools(tmp_path: Path) -> None:
    kernel, redis_store = build_test_kernel(tmp_path)
    with patch("self_ai.main.get_kernel", return_value=kernel):
        result = await main.run_autonomy_workflow("Design architecture for this system")

    assert result["run_id"]
    assert result["session_id"] == "default"
    assert redis_store.states
    assert redis_store.traces
    node_names = [name for _, name, _ in redis_store.node_outputs]
    assert "mainloop_finalize" in node_names


@pytest.mark.asyncio
async def test_run_autonomy_workflow_returns_error_list_on_model_tool_failures(tmp_path: Path) -> None:
    kernel, _ = build_test_kernel(tmp_path)

    original = kernel.tool_runtime.execute_by_name

    async def _wrapped(name, arguments=None, **kwargs):
        if name == "model.generate":
            from self_ai.runtime.tool_result import ToolResult

            return ToolResult.failure(
                call_id="forced-fail",
                tool_name=name,
                error_type="ForcedError",
                message="forced model failure",
            )
        return await original(name, arguments or {}, **kwargs)

    kernel.tool_runtime.execute_by_name = _wrapped  # type: ignore[assignment]

    with patch("self_ai.main.get_kernel", return_value=kernel):
        result = await main.run_autonomy_workflow("Summarize this project")

    assert isinstance(result["errors"], list)
    assert any(err.get("type") == "ForcedError" for err in result["errors"])
