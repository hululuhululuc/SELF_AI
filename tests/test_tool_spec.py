# coding=utf-8
"""Tests for ToolSpec / ToolCall / ToolResult contracts."""

from self_ai.runtime.tool_call import ToolCall
from self_ai.runtime.tool_result import ToolResult
from self_ai.runtime.tool_spec import ToolSpec


def _noop_handler(_args, _ctx):
    return {"ok": True}


def test_tool_spec_model_dump_json_stable() -> None:
    spec = ToolSpec(
        name="demo.tool",
        description="demo",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        permission="read_only",
        timeout_s=3,
        tags=["demo"],
        handler=_noop_handler,
    )
    dumped = spec.model_dump(mode="json")
    assert dumped["name"] == "demo.tool"
    assert dumped["permission"] == "read_only"
    assert "handler" not in dumped


def test_tool_call_model_dump_json_stable() -> None:
    call = ToolCall(
        call_id="c1",
        run_id="r1",
        session_id="s1",
        tool_name="demo.tool",
        arguments={"x": 1},
        metadata={"k": "v"},
    )
    dumped = call.model_dump(mode="json")
    assert dumped == {
        "call_id": "c1",
        "run_id": "r1",
        "session_id": "s1",
        "tool_name": "demo.tool",
        "arguments": {"x": 1},
        "metadata": {"k": "v"},
    }


def test_tool_result_model_dump_json_stable() -> None:
    result = ToolResult.success(
        call_id="c1",
        tool_name="demo.tool",
        data={"value": 42},
        latency_ms=12,
        metadata={"m": 1},
    )
    dumped = result.model_dump(mode="json")
    assert dumped["call_id"] == "c1"
    assert dumped["tool_name"] == "demo.tool"
    assert dumped["ok"] is True
    assert dumped["data"] == {"value": 42}
    assert dumped["latency_ms"] == 12
