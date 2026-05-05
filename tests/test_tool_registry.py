# coding=utf-8
"""Tests for ToolRegistry behaviors."""

import pytest

from self_ai.runtime.tool_registry import DuplicateToolError, MissingToolError, ToolRegistry
from self_ai.runtime.tool_spec import ToolSpec


def _handler(_args, _ctx):
    return {"ok": True}


def _spec(name: str) -> ToolSpec:
    return ToolSpec(
        name=name,
        description="demo",
        permission="read_only",
        timeout_s=2,
        input_schema={},
        output_schema={},
        tags=[],
        handler=_handler,
    )


def test_registry_register_get_list_unregister() -> None:
    registry = ToolRegistry()
    registry.register(_spec("tool.a"))
    registry.register(_spec("tool.b"))
    assert registry.get("tool.a").name == "tool.a"
    assert [spec.name for spec in registry.list()] == ["tool.a", "tool.b"]
    registry.unregister("tool.a")
    assert [spec.name for spec in registry.list()] == ["tool.b"]


def test_registry_duplicate_name_rejected() -> None:
    registry = ToolRegistry()
    registry.register(_spec("tool.a"))
    with pytest.raises(DuplicateToolError):
        registry.register(_spec("tool.a"))


def test_registry_missing_tool_error() -> None:
    registry = ToolRegistry()
    with pytest.raises(MissingToolError):
        registry.get("missing.tool")
