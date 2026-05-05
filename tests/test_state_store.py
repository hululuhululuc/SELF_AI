# coding=utf-8
"""Tests for kernel.state_store."""

from __future__ import annotations

from pathlib import Path

import pytest

from self_ai.kernel.engine_state import EngineState
from self_ai.kernel.run_context import RunContext
from self_ai.kernel.state_store import StateStore
from self_ai.runtime.tool_result import ToolResult


class _FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def execute_by_name(self, name, arguments=None, **_kwargs):
        args = arguments or {}
        self.calls.append((name, dict(args)))
        return ToolResult.success(call_id="c1", tool_name=name, data={"ok": True})


@pytest.mark.asyncio
async def test_state_store_in_memory_save_trace_node_agent() -> None:
    store = StateStore()
    state = EngineState.from_input("hello", run_id="r1", session_id="s1")

    await store.save_state(state)
    await store.append_trace("r1", {"event": "e1"})
    await store.save_node_output("r1", "research", {"count": 1})
    await store.save_agent_output("r1", "coder", {"summary": "ok"})

    loaded = store.get_state("r1")
    assert loaded is not None
    assert loaded["run_id"] == "r1"
    assert store.list_trace("r1") == [{"event": "e1"}]
    assert store.get_node_output("r1", "research") == {"count": 1}
    assert store.get_agent_output("r1", "coder") == {"summary": "ok"}


@pytest.mark.asyncio
async def test_state_store_uses_tool_runtime_for_runtime_persistence(tmp_path: Path) -> None:
    runtime = _FakeRuntime()
    ctx = RunContext(project_root=tmp_path, run_id="r2", session_id="s2", tool_runtime=runtime)
    store = StateStore()
    state = EngineState.from_input("hello", run_id="r2", session_id="s2")

    await store.save_state(state, run_context=ctx)
    await store.append_trace("r2", {"event": "e2"}, run_context=ctx)
    await store.save_node_output("r2", "plan", {"step_count": 2}, run_context=ctx)
    await store.save_agent_output("r2", "synthesizer", {"summary": "done"}, run_context=ctx)

    names = [name for name, _ in runtime.calls]
    assert "storage.runtime.save_run_state" in names
    assert "storage.runtime.append_trace" in names
    assert "storage.runtime.save_node_output" in names
    assert "storage.runtime.save_agent_output" in names
