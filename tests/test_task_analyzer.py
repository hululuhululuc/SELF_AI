# coding=utf-8
"""Unit tests for task analyzer with mocked model routing."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from self_ai.task_analyzer import analyze_task


def _mock_model_response(payload: dict) -> dict[str, str]:
    return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}


@pytest.mark.asyncio
async def test_task_analyzer_architecture_planning_case() -> None:
    payload = {
        "intent": "architecture_design",
        "domain": "ai_agent",
        "requires_code": False,
        "requires_repo_context": False,
        "requires_research": False,
        "requires_planning": True,
        "requires_review": True,
        "expected_artifact": "design_doc",
        "complexity": "high",
        "risk_level": "high",
        "confidence": 0.84,
        "rationale": "Workflow redesign and storage architecture planning.",
    }
    text = "多智能体项目流程重构，并用 Redis/Neo4j/Qdrant 增强存储能力"
    with patch("self_ai.task_analyzer.route_model", new_callable=AsyncMock) as mocked:
        mocked.return_value = _mock_model_response(payload)
        profile = await analyze_task(text, session_id="s-arch")

    assert profile.intent.value in {"architecture_design", "planning"}
    assert profile.requires_planning
    assert profile.requires_review
    assert profile.retrieval_policy.use_graph
    assert profile.identity.run_id
    assert profile.identity.session_id == "s-arch"


@pytest.mark.asyncio
async def test_task_analyzer_debugging_case() -> None:
    payload = {
        "intent": "debugging",
        "domain": "software",
        "requires_code": True,
        "requires_repo_context": True,
        "requires_research": False,
        "requires_planning": False,
        "requires_review": True,
        "expected_artifact": "code_review_report",
        "complexity": "medium",
        "risk_level": "medium",
        "confidence": 0.81,
        "rationale": "Runtime exception troubleshooting in repository context.",
    }
    text = "main.py 报 ValueError，请定位 bug"
    with patch("self_ai.task_analyzer.route_model", new_callable=AsyncMock) as mocked:
        mocked.return_value = _mock_model_response(payload)
        profile = await analyze_task(text, session_id="s-debug")

    assert profile.intent.value in {"debugging", "code_review"}
    assert profile.requires_code
    assert profile.requires_repo_context
    assert profile.requires_review
    assert profile.retrieval_policy.use_code_index
    assert profile.identity.session_id == "s-debug"


@pytest.mark.asyncio
async def test_task_analyzer_code_generation_case() -> None:
    payload = {
        "intent": "code_generation",
        "domain": "software",
        "requires_code": True,
        "requires_repo_context": False,
        "requires_research": False,
        "requires_planning": False,
        "requires_review": False,
        "expected_artifact": "code_patch",
        "complexity": "low",
        "risk_level": "low",
        "confidence": 0.88,
        "rationale": "Standalone function generation.",
    }
    with patch("self_ai.task_analyzer.route_model", new_callable=AsyncMock) as mocked:
        mocked.return_value = _mock_model_response(payload)
        profile = await analyze_task("写一个 Python 函数，输入列表返回平均值")
    assert profile.intent.value == "code_generation"
    assert profile.requires_code
    assert profile.identity.run_id


@pytest.mark.asyncio
async def test_task_analyzer_research_summary_case() -> None:
    payload = {
        "intent": "research_summary",
        "domain": "academic",
        "requires_code": False,
        "requires_repo_context": False,
        "requires_research": True,
        "requires_planning": False,
        "requires_review": False,
        "expected_artifact": "research_report",
        "complexity": "medium",
        "risk_level": "medium",
        "confidence": 0.86,
        "rationale": "Journal submission requirement summary.",
    }
    with patch("self_ai.task_analyzer.route_model", new_callable=AsyncMock) as mocked:
        mocked.return_value = _mock_model_response(payload)
        profile = await analyze_task("总结 Applied Mathematical Modelling 投稿要求")
    assert profile.intent.value == "research_summary"
    assert profile.requires_research
    assert profile.retrieval_policy.use_web


@pytest.mark.asyncio
async def test_task_analyzer_document_writing_case() -> None:
    payload = {
        "intent": "document_writing",
        "domain": "academic",
        "requires_code": False,
        "requires_repo_context": False,
        "requires_research": False,
        "requires_planning": False,
        "requires_review": False,
        "expected_artifact": "answer",
        "complexity": "medium",
        "risk_level": "low",
        "confidence": 0.8,
        "rationale": "Academic style translation and rewriting.",
    }
    with patch("self_ai.task_analyzer.route_model", new_callable=AsyncMock) as mocked:
        mocked.return_value = _mock_model_response(payload)
        profile = await analyze_task("将摘要翻译成符合拓扑优化期刊风格的英文")
    assert profile.intent.value == "document_writing"
    assert profile.domain.value == "academic"
    assert not profile.requires_code


@pytest.mark.asyncio
async def test_task_analyzer_fallback_on_invalid_json() -> None:
    with patch("self_ai.task_analyzer.route_model", new_callable=AsyncMock) as mocked:
        mocked.return_value = {"model": "mock-fast", "response": "not-json"}
        profile = await analyze_task("main.py 报 ValueError，请定位 bug", session_id="s-fallback")
    assert profile.identity.session_id == "s-fallback"
    assert profile.identity.run_id
    assert profile.memory_write_policy.write_redis
    assert profile.retrieval_policy.use_vector
