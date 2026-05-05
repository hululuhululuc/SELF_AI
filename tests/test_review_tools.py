# coding=utf-8
"""Tests for adapters.review_tools model.generate path."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from self_ai.adapters.model_tool import register_model_tools
from self_ai.adapters.review_tools import register_review_tools
from self_ai.kernel.run_context import RunContext
from self_ai.runtime.tool_runtime import ToolRuntime


def _make_review_input() -> dict[str, Any]:
    return {
        "run_id": "r-review",
        "session_id": "s-review",
        "task": "Review architecture proposal",
        "task_profile": {"risk_level": "medium", "complexity": "medium"},
        "workflow_decision": {"selected_agents": ["architect", "synthesizer"]},
        "plan": {"steps": [{"goal": "design"}]},
        "context_pack": {},
        "agent_outputs": {"architect": {"summary": "proposal"}},
        "redis_context": {},
        "qdrant_review_memory": [],
        "neo4j_graph_context": [],
        "metadata": {"executed_role_agents": ["architect", "synthesizer"]},
    }


@pytest.mark.asyncio
async def test_review_run_calls_model_generate(tmp_path: Path) -> None:
    model_calls: list[tuple[str, str]] = []

    async def _fake_route_model(prompt: str, category: str) -> dict[str, Any]:
        model_calls.append((prompt, category))
        return {
            "model": "mock-review",
            "response": (
                '{"pass_review": true, "severity": "low", "major_issues": [], "minor_issues": [], '
                '"missing_requirements": [], "requires_revision": false, "revision_target_agent": null, '
                '"revision_instruction": "", "evidence_refs": [], "storage_evidence": [], "rationale": "ok"}'
            ),
        }

    runtime = ToolRuntime()
    register_model_tools(runtime.registry, route_model_func=_fake_route_model)
    register_review_tools(runtime.registry)
    ctx = RunContext(project_root=tmp_path, tool_runtime=runtime, permission_profile={"model_call": True})

    result = await runtime.execute_by_name(
        "review.run",
        {"review_input": _make_review_input()},
        run_id="r-review",
        session_id="s-review",
        run_context=ctx,
    )

    assert result.ok is True
    report = result.data["review_report"]
    assert report["pass_review"] is True
    assert report["metadata"]["model"] == "mock-review"
    assert len(model_calls) == 1
    assert model_calls[0][1] == "reasoning"


@pytest.mark.asyncio
async def test_review_run_safe_fallback_when_model_generate_fails(tmp_path: Path) -> None:
    async def _boom(_prompt: str, _category: str) -> dict[str, Any]:
        raise RuntimeError("llm down")

    runtime = ToolRuntime()
    register_model_tools(runtime.registry, route_model_func=_boom)
    register_review_tools(runtime.registry)
    ctx = RunContext(project_root=tmp_path, tool_runtime=runtime, permission_profile={"model_call": True})

    result = await runtime.execute_by_name(
        "review.run",
        {
            "review_input": {
                **_make_review_input(),
                "task_profile": {"risk_level": "high", "complexity": "high"},
            }
        },
        run_id="r-review",
        session_id="s-review",
        run_context=ctx,
    )

    assert result.ok is True
    report = result.data["review_report"]
    assert report["metadata"]["fallback"] is True
    assert report["pass_review"] is False


@pytest.mark.asyncio
async def test_quality_gate_decide_does_not_call_model(tmp_path: Path) -> None:
    model_calls = 0

    async def _fake_route_model(_prompt: str, _category: str) -> dict[str, Any]:
        nonlocal model_calls
        model_calls += 1
        return {"model": "mock", "response": "{}"}

    runtime = ToolRuntime()
    register_model_tools(runtime.registry, route_model_func=_fake_route_model)
    register_review_tools(runtime.registry)
    ctx = RunContext(project_root=tmp_path, tool_runtime=runtime, permission_profile={"model_call": True})

    result = await runtime.execute_by_name(
        "review.quality_gate.decide",
        {
            "review_report": {
                "pass_review": True,
                "severity": "low",
                "major_issues": [],
                "minor_issues": ["nit"],
                "missing_requirements": [],
                "requires_revision": False,
            },
            "task_profile": {"risk_level": "low"},
            "run_id": "r-review",
            "session_id": "s-review",
        },
        run_id="r-review",
        session_id="s-review",
        run_context=ctx,
    )

    assert result.ok is True
    assert result.data["quality_gate"]["decision"] in {"pass", "warn"}
    assert model_calls == 0


def test_review_tools_file_does_not_import_route_model() -> None:
    content = Path("adapters/review_tools.py").read_text(encoding="utf-8")
    assert "from ..router import route_model" not in content
