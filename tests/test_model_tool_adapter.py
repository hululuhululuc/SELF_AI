# coding=utf-8
"""Tests for model.generate tool adapter."""

from unittest.mock import AsyncMock

import pytest

from self_ai.adapters.model_tool import register_model_tools
from self_ai.runtime.tool_runtime import ToolRuntime


@pytest.mark.asyncio
async def test_model_generate_adapter_uses_injected_route_model() -> None:
    route_model = AsyncMock(return_value={"model": "mock-model", "response": "hello"})
    runtime = ToolRuntime()
    register_model_tools(runtime.registry, route_model_func=route_model)

    result = await runtime.execute_by_name(
        "model.generate",
        {"task": "say hello", "category": "fast"},
    )
    assert result.ok is True
    assert result.data["model"] == "mock-model"
    assert result.data["response"] == "hello"
    route_model.assert_awaited_once()
