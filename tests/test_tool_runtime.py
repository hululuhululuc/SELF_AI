# coding=utf-8
"""Tests for ToolRuntime registration and execution."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from self_ai.kernel.run_context import RunContext
from self_ai.runtime.permission_guard import PermissionGuard
from self_ai.runtime.tool_call import ToolCall
from self_ai.runtime.tool_runtime import ToolRuntime


class _FakeRedisStore:
    def append_trace(self, run_id, trace):
        self.last = ("append_trace", run_id, trace)

    def save_node_output(self, run_id, node_name, output):
        self.last = ("save_node_output", run_id, node_name, output)

    def save_agent_output(self, run_id, agent_name, output):
        self.last = ("save_agent_output", run_id, agent_name, output)

    def save_run_state(self, run_id, state):
        self.last = ("save_run_state", run_id, state)

    def load_run_state(self, run_id):
        self.last = ("load_run_state", run_id)
        return {"run_id": run_id}


class _FakeQdrantMemory:
    async def search(self, **_kwargs):
        return []

    async def upsert_evidence(self, **_kwargs):
        return None

    async def upsert_memory(self, **_kwargs):
        return None


class _FakeGraphMemory:
    def find_task_lineage(self, *_args, **_kwargs):
        return None

    def find_related_symbols(self, *_args, **_kwargs):
        return []

    def find_prior_issues(self, *_args, **_kwargs):
        return []

    def record_issue_fix(self, *_args, **_kwargs):
        return None


@pytest.mark.asyncio
async def test_tool_runtime_register_default_tools_contains_required_names() -> None:
    route_model = AsyncMock(return_value={"model": "m", "response": "ok"})
    runtime = ToolRuntime(
        dependencies={
            "route_model_func": route_model,
            "redis_store": _FakeRedisStore(),
            "qdrant_memory": _FakeQdrantMemory(),
            "graph_memory": _FakeGraphMemory(),
            "project_root": ".",
            "web_search_func": lambda _q, _k: [],
        }
    )
    runtime.register_default_tools()
    names = {spec.name for spec in runtime.registry.list()}
    required = {
        "model.generate",
        "storage.runtime.append_trace",
        "storage.runtime.save_node_output",
        "storage.runtime.save_agent_output",
        "storage.runtime.save_run_state",
        "storage.runtime.load_run_state",
        "storage.semantic.search",
        "storage.semantic.upsert_evidence",
        "storage.semantic.upsert_memory",
        "storage.graph.find_task_lineage",
        "storage.graph.find_related_symbols",
        "storage.graph.find_prior_issues",
        "storage.graph.record_issue_fix",
        "retrieval.web.search",
        "workspace.file.read",
        "workspace.file.write",
        "workspace.file.edit",
        "workspace.file.rename",
        "workspace.file.delete",
        "workspace.shell.exec",
        "agent.run",
        "agent.synthesize",
        "review.run",
        "review.quality_gate.decide",
    }
    assert required.issubset(names)


@pytest.mark.asyncio
async def test_tool_runtime_permission_check_before_execute() -> None:
    route_model = AsyncMock(return_value={"model": "m", "response": "ok"})
    runtime = ToolRuntime(
        permission_guard=PermissionGuard(permission_profile={"model_call": False}),
        dependencies={"route_model_func": route_model},
    )
    runtime.register_default_tools()
    result = await runtime.execute_by_name(
        "model.generate",
        {"task": "hello", "category": "fast"},
    )
    assert result.ok is False
    assert result.error is not None
    assert result.error["type"] == "PermissionDenied"


@pytest.mark.asyncio
async def test_tool_runtime_missing_tool_returns_failure() -> None:
    runtime = ToolRuntime()
    call = ToolCall(tool_name="missing.tool", arguments={})
    result = await runtime.execute(call)
    assert result.ok is False
    assert result.error is not None
    assert result.error["type"] == "MissingToolError"


@pytest.mark.asyncio
async def test_tool_runtime_storage_runtime_load_run_state() -> None:
    runtime = ToolRuntime(dependencies={"redis_store": _FakeRedisStore()})
    runtime.register_default_tools()
    ctx = RunContext(project_root=".", metadata={})
    result = await runtime.execute_by_name(
        "storage.runtime.load_run_state",
        {"run_id": "r1"},
        run_id="r1",
        run_context=ctx,
    )
    assert result.ok is True
    assert result.data["state"]["run_id"] == "r1"


@pytest.mark.asyncio
async def test_tool_runtime_denies_workspace_write_by_default(tmp_path: Path) -> None:
    runtime = ToolRuntime(dependencies={"project_root": tmp_path})
    runtime.register_default_tools()
    ctx = RunContext(project_root=tmp_path)
    result = await runtime.execute_by_name(
        "workspace.file.write",
        {"path": "a.txt", "content": "hello"},
        run_context=ctx,
    )
    assert result.ok is False
    assert result.error is not None
    assert result.error["type"] == "PermissionDenied"
    assert result.metadata["permission"] == "workspace_write"


@pytest.mark.asyncio
async def test_tool_runtime_denies_network_by_default() -> None:
    runtime = ToolRuntime(dependencies={"web_search_func": lambda _q, _k: []})
    runtime.register_default_tools()
    result = await runtime.execute_by_name(
        "retrieval.web.search",
        {"query": "hello"},
    )
    assert result.ok is False
    assert result.error is not None
    assert result.error["type"] == "PermissionDenied"
    assert result.metadata["permission"] == "network"


@pytest.mark.asyncio
async def test_tool_runtime_denies_shell_exec_by_default(tmp_path: Path) -> None:
    runtime = ToolRuntime(dependencies={"project_root": tmp_path})
    runtime.register_default_tools()
    ctx = RunContext(project_root=tmp_path)
    result = await runtime.execute_by_name(
        "workspace.shell.exec",
        {"command": "git status"},
        run_context=ctx,
    )
    assert result.ok is False
    assert result.error is not None
    assert result.error["type"] == "PermissionDenied"
    assert result.metadata["permission"] == "shell_exec"
