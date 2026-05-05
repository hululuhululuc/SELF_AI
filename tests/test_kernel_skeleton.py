# coding=utf-8
"""Tests for SelfAIKernel skeleton wiring."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from self_ai.kernel.kernel import SelfAIKernel


class _FakeRedisStore:
    def append_trace(self, *_args, **_kwargs):
        return None

    def save_node_output(self, *_args, **_kwargs):
        return None

    def save_agent_output(self, *_args, **_kwargs):
        return None

    def save_run_state(self, *_args, **_kwargs):
        return None

    def load_run_state(self, run_id):
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
async def test_kernel_skeleton_registers_default_tools(tmp_path: Path) -> None:
    kernel = SelfAIKernel(
        project_root=tmp_path,
        dependencies={
            "route_model_func": AsyncMock(return_value={"model": "m", "response": "ok"}),
            "redis_store": _FakeRedisStore(),
            "qdrant_memory": _FakeQdrantMemory(),
            "graph_memory": _FakeGraphMemory(),
            "web_search_func": lambda _q, _k: [],
        },
    )
    info = kernel.dry_run()
    assert info["tool_count"] > 0
    assert "model.generate" in info["tools"]
    assert "review.quality_gate.decide" in info["tools"]


@pytest.mark.asyncio
async def test_kernel_run_once_wiring(tmp_path: Path) -> None:
    route_model = AsyncMock(return_value={"model": "mock", "response": "pong"})
    kernel = SelfAIKernel(
        project_root=tmp_path,
        dependencies={"route_model_func": route_model, "web_search_func": lambda _q, _k: []},
    )
    result = await kernel.run_once(
        tool_name="model.generate",
        arguments={"task": "ping", "category": "fast"},
    )
    assert result["ok"] is True
    assert result["data"]["response"] == "pong"
