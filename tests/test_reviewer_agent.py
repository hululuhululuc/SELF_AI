# coding=utf-8
"""Tests for Phase 6B reviewer agent."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from self_ai.agents.reviewer import ReviewerAgent, ReviewerInput


def _review_input(*, risk_level: str = "medium", complexity: str = "medium") -> ReviewerInput:
    return ReviewerInput(
        run_id="run-review-1",
        session_id="default",
        task="Implement feature X with robust validation",
        task_profile={"risk_level": risk_level, "complexity": complexity},
        workflow_decision={"selected_agents": ["coder", "synthesizer"]},
        plan={"steps": [{"step_id": "s1", "goal": "implement"}]},
        context_pack={"context_preview": "ctx"},
        agent_outputs={"coder": {"summary": "implemented core logic"}},
        redis_context={"trace_count": 10},
        qdrant_review_memory=[{"content": "past issue"}],
        neo4j_graph_context=[{"query_type": "find_task_lineage"}],
        metadata={"executed_role_agents": ["coder", "synthesizer"]},
    )


def test_reviewer_input_model_dump_json_stable() -> None:
    value = _review_input()
    dumped = value.model_dump(mode="json")
    assert dumped["run_id"] == "run-review-1"
    assert dumped["session_id"] == "default"
    assert "agent_outputs" in dumped
    assert "qdrant_review_memory" in dumped


def test_reviewer_build_prompt_contains_storage_driven_instructions() -> None:
    agent = ReviewerAgent()
    prompt = agent.build_prompt(_review_input())
    assert "storage-driven reviewer" in prompt
    assert "Return ONLY valid JSON" in prompt
    assert "Required keys:" in prompt


def test_reviewer_prompt_does_not_contain_forbidden_instructions() -> None:
    agent = ReviewerAgent()
    prompt = agent.build_prompt(_review_input()).lower()
    assert "modify code" in prompt  # explicitly "do not modify code"
    assert "do not re-classify task intent" in prompt


@pytest.mark.asyncio
async def test_reviewer_run_parses_json_review_report() -> None:
    agent = ReviewerAgent()
    model_func = AsyncMock(
        return_value={
            "model": "mock-reasoning",
            "response": (
                '{"pass_review": true, "severity": "medium", "major_issues": [], '
                '"minor_issues": ["naming polish"], "missing_requirements": [], '
                '"requires_revision": false, "revision_target_agent": null, '
                '"revision_instruction": "", "evidence_refs": [{"source_type":"doc","ref":"r1"}], '
                '"storage_evidence": [{"content":"history evidence","source_type":"review"}], '
                '"rationale": "looks good"}'
            ),
        }
    )
    report = await agent.run(_review_input(), route_model_func=model_func)
    assert report["pass_review"] is True
    assert report["severity"] == "medium"
    assert report["minor_issues"] == ["naming polish"]
    assert report["metadata"]["model"] == "mock-reasoning"


@pytest.mark.asyncio
async def test_reviewer_model_failure_returns_fallback_report() -> None:
    agent = ReviewerAgent()
    model_func = AsyncMock(side_effect=RuntimeError("boom"))
    report = await agent.run(_review_input(), route_model_func=model_func)
    assert report["metadata"]["fallback"] is True
    assert report["metadata"]["error_type"] == "RuntimeError"
    assert "pass_review" in report


@pytest.mark.asyncio
async def test_high_risk_fallback_not_default_pass() -> None:
    agent = ReviewerAgent()
    model_func = AsyncMock(side_effect=ValueError("bad"))
    report = await agent.run(
        _review_input(risk_level="high", complexity="high"),
        route_model_func=model_func,
    )
    assert report["pass_review"] is False
    assert report["requires_revision"] is True
    assert report["severity"] == "high"


def test_reviewer_source_does_not_call_qdrant_or_neo4j_or_redis() -> None:
    source = Path("agents/reviewer.py").read_text(encoding="utf-8").lower()
    # Field names may contain storage keywords; assert no direct storage calls/imports.
    assert "from .tools import" not in source
    assert "from ..tools import" not in source
    assert "redis_store." not in source
    assert "qdrant_memory.search(" not in source
    assert "qdrant_memory.upsert_" not in source
    assert "graph_memory.find_task_lineage(" not in source
    assert "graph_memory.find_prior_issues(" not in source
    assert "graph_memory.find_related_symbols(" not in source


@pytest.mark.asyncio
async def test_review_report_does_not_store_large_raw_text() -> None:
    agent = ReviewerAgent()
    huge = "X" * 5000
    model_func = AsyncMock(
        return_value={
            "model": "mock",
            "response": (
                '{"pass_review": false, "severity": "high", "major_issues": ["a"], '
                '"minor_issues": [], "missing_requirements": [], "requires_revision": true, '
                '"revision_target_agent": "coder", "revision_instruction": "'
                + huge
                + '", "evidence_refs": [{"source_type":"doc","ref":"r1","preview":"'
                + huge
                + '"}], "storage_evidence": [{"content":"'
                + huge
                + '","source_type":"review"}], "rationale":"'
                + huge
                + '"}'
            ),
        }
    )
    report = await agent.run(_review_input(), route_model_func=model_func)
    assert len(report["revision_instruction"]) <= 500
    assert len(report["storage_evidence"][0]["content"]) <= 180
    assert len(report["evidence_refs"][0]["preview"]) <= 140
