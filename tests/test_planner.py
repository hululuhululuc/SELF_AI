# coding=utf-8
"""Tests for Phase 5B planner module."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from self_ai.execution_modes import ExecutionMode
from self_ai.planner import Plan, build_plan, fallback_plan, should_plan
from self_ai.schemas import ComplexityLevel, RiskLevel, TaskIntent, TaskProfile


def _profile(
    intent: TaskIntent,
    *,
    complexity: ComplexityLevel = ComplexityLevel.MEDIUM,
    risk: RiskLevel = RiskLevel.MEDIUM,
    requires_planning: bool = False,
) -> TaskProfile:
    return TaskProfile(
        intent=intent,
        complexity=complexity,
        risk_level=risk,
        requires_planning=requires_planning,
    )


def test_should_plan_architecture_design_true() -> None:
    profile = _profile(TaskIntent.ARCHITECTURE_DESIGN, requires_planning=True)
    assert should_plan(task_profile=profile, workflow_decision={}) is True


def test_should_plan_planning_true() -> None:
    profile = _profile(TaskIntent.PLANNING, requires_planning=True)
    assert should_plan(task_profile=profile, workflow_decision={}) is True


def test_should_plan_high_complexity_true() -> None:
    profile = _profile(TaskIntent.GENERAL_QA, complexity=ComplexityLevel.HIGH)
    assert should_plan(task_profile=profile, workflow_decision={}) is True


def test_should_plan_quick_answer_false() -> None:
    profile = _profile(TaskIntent.GENERAL_QA, complexity=ComplexityLevel.HIGH)
    decision = {"use_quick_answer": True, "execution_mode": ExecutionMode.QUICK_ANSWER.value}
    assert should_plan(task_profile=profile, workflow_decision=decision) is False


def test_should_plan_document_writing_low_false() -> None:
    profile = _profile(TaskIntent.DOCUMENT_WRITING, complexity=ComplexityLevel.LOW)
    assert should_plan(task_profile=profile, workflow_decision={}) is False


@pytest.mark.asyncio
async def test_build_plan_parses_model_json_success() -> None:
    profile = _profile(TaskIntent.ARCHITECTURE_DESIGN, requires_planning=True)
    route_model_func = AsyncMock(
        return_value={
            "model": "mock-reasoning",
            "response": (
                '{"task_summary":"refactor workflow","execution_mode":"architecture_design",'
                '"rationale":"phased rollout","steps":[{"step_id":"s1","goal":"analyze",'
                '"agent_name":"architect","depends_on":[],"expected_output":"scope",'
                '"done_criteria":"scope ready"}]}'
            ),
        }
    )
    plan = await build_plan(
        task_summary="refactor workflow",
        run_id="r1",
        session_id="s1",
        task_profile=profile,
        workflow_decision={"execution_mode": ExecutionMode.ARCHITECTURE_DESIGN.value},
        route_model_func=route_model_func,
    )
    assert isinstance(plan, Plan)
    assert plan.run_id == "r1"
    assert plan.execution_mode == "architecture_design"
    assert len(plan.steps) == 1
    route_model_func.assert_awaited_once()


@pytest.mark.asyncio
async def test_build_plan_model_failure_falls_back() -> None:
    profile = _profile(TaskIntent.DEBUGGING, complexity=ComplexityLevel.HIGH)
    route_model_func = AsyncMock(side_effect=RuntimeError("planner failed"))
    plan = await build_plan(
        task_summary="debug value error",
        run_id="r2",
        session_id="s2",
        task_profile=profile,
        workflow_decision={"execution_mode": ExecutionMode.CODE_FOCUSED.value},
        route_model_func=route_model_func,
    )
    assert isinstance(plan, Plan)
    assert len(plan.steps) >= 3
    assert plan.metadata.get("fallback") is True


def test_fallback_plan_architecture_generates_multi_steps() -> None:
    plan = fallback_plan(
        run_id="r3",
        session_id="s3",
        task_summary="design architecture",
        task_profile=_profile(TaskIntent.ARCHITECTURE_DESIGN),
        workflow_decision={"execution_mode": ExecutionMode.ARCHITECTURE_DESIGN.value},
    )
    assert len(plan.steps) >= 3


def test_fallback_plan_debugging_contains_locate_fix_verify() -> None:
    plan = fallback_plan(
        run_id="r4",
        session_id="s4",
        task_summary="debug runtime error",
        task_profile=_profile(TaskIntent.DEBUGGING),
        workflow_decision={"execution_mode": ExecutionMode.CODE_FOCUSED.value},
    )
    goals = " ".join(step.goal.lower() for step in plan.steps)
    assert "locate" in goals or "定位" in goals
    assert "fix" in goals or "修复" in goals
    assert "validate" in goals or "验证" in goals


def test_plan_model_dump_json_stable() -> None:
    plan = fallback_plan(
        run_id="r5",
        session_id="s5",
        task_summary="summary",
        task_profile=_profile(TaskIntent.RESEARCH_SUMMARY),
        workflow_decision={"execution_mode": ExecutionMode.RESEARCH_THEN_IMPLEMENT.value},
    )
    dumped = plan.model_dump(mode="json")
    assert dumped["run_id"] == "r5"
    assert dumped["session_id"] == "s5"
    assert "steps" in dumped
    assert "planner_version" in dumped


def test_planner_source_does_not_reference_qdrant_or_neo4j() -> None:
    source = Path("planner.py").read_text(encoding="utf-8").lower()
    assert "qdrant" not in source
    assert "neo4j" not in source


def test_planner_source_does_not_use_legacy_keyword_helpers() -> None:
    source = Path("planner.py").read_text(encoding="utf-8")
    assert "is_coding_task" not in source
    assert "is_complex_task" not in source
