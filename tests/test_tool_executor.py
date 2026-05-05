# coding=utf-8
"""Tests for ToolExecutor sync/async/error wrapping."""

import asyncio

import pytest

from self_ai.runtime.tool_call import ToolCall
from self_ai.runtime.tool_executor import ToolExecutor
from self_ai.runtime.tool_spec import ToolSpec


def _sync_handler(args, _ctx):
    return {"echo": args.get("x")}


async def _async_handler(args, _ctx):
    await asyncio.sleep(0.01)
    return {"echo": args.get("x")}


async def _slow_async_handler(_args, _ctx):
    await asyncio.sleep(0.2)
    return {"ok": True}


def _error_handler(_args, _ctx):
    raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_executor_sync_handler() -> None:
    executor = ToolExecutor()
    spec = ToolSpec(
        name="demo.sync",
        description="sync",
        input_schema={},
        output_schema={},
        permission="read_only",
        timeout_s=1,
        tags=[],
        handler=_sync_handler,
    )
    call = ToolCall(call_id="c1", tool_name="demo.sync", arguments={"x": 3})
    result = await executor.execute(spec, call)
    assert result.ok is True
    assert result.data["echo"] == 3


@pytest.mark.asyncio
async def test_executor_async_handler() -> None:
    executor = ToolExecutor()
    spec = ToolSpec(
        name="demo.async",
        description="async",
        input_schema={},
        output_schema={},
        permission="read_only",
        timeout_s=1,
        tags=[],
        handler=_async_handler,
    )
    call = ToolCall(call_id="c1", tool_name="demo.async", arguments={"x": 7})
    result = await executor.execute(spec, call)
    assert result.ok is True
    assert result.data["echo"] == 7


@pytest.mark.asyncio
async def test_executor_exception_wrapped_as_failure() -> None:
    events = []

    def _capture(event):
        events.append(event.event)

    executor = ToolExecutor(event_callback=_capture)
    spec = ToolSpec(
        name="demo.error",
        description="error",
        input_schema={},
        output_schema={},
        permission="read_only",
        timeout_s=1,
        tags=[],
        handler=_error_handler,
    )
    call = ToolCall(call_id="c1", tool_name="demo.error", arguments={})
    result = await executor.execute(spec, call)
    assert result.ok is False
    assert result.error is not None
    assert result.error["type"] == "RuntimeError"
    assert "tool.call.start" in events
    assert "tool.call.error" in events


@pytest.mark.asyncio
async def test_executor_timeout_wrapped_as_failure() -> None:
    executor = ToolExecutor()
    spec = ToolSpec(
        name="demo.timeout",
        description="timeout",
        input_schema={},
        output_schema={},
        permission="read_only",
        timeout_s=0.01,
        tags=[],
        handler=_slow_async_handler,
    )
    call = ToolCall(call_id="c-timeout", tool_name="demo.timeout", arguments={})
    result = await executor.execute(spec, call)
    assert result.ok is False
    assert result.error is not None
    assert result.error["type"] == "TimeoutError"
    assert "timeout" in result.error["message"].lower()
