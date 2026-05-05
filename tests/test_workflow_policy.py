# coding=utf-8
"""Phase 5A tests for workflow policy decision rules."""

from pathlib import Path

from self_ai.schemas import (
    ComplexityLevel,
    RetrievalPolicy,
    RiskLevel,
    TaskIntent,
    TaskProfile,
)
from self_ai.workflow_policy import WorkflowDecision, decide_workflow


def _profile(intent: TaskIntent, **kwargs) -> TaskProfile:
    base = {
        "intent": intent,
        "complexity": ComplexityLevel.MEDIUM,
        "risk_level": RiskLevel.MEDIUM,
        "requires_code": False,
        "requires_research": False,
        "requires_planning": False,
        "requires_review": False,
    }
    base.update(kwargs)
    return TaskProfile(**base)


def test_general_qa_low_routes_to_quick_answer() -> None:
    profile = _profile(
        TaskIntent.GENERAL_QA,
        complexity=ComplexityLevel.LOW,
        risk_level=RiskLevel.LOW,
    )
    decision = decide_workflow(task_profile=profile, retrieval_policy=RetrievalPolicy())
    assert decision.execution_mode == "quick_answer"
    assert decision.use_quick_answer is True
    assert decision.skip_research is True
    assert decision.skip_debate is True
    assert decision.next_node == "quick_answer"


def test_document_writing_skips_debate() -> None:
    profile = _profile(TaskIntent.DOCUMENT_WRITING)
    decision = decide_workflow(task_profile=profile)
    assert decision.execution_mode == "document_writing"
    assert decision.skip_debate is True
    assert decision.skip_research is True
    assert decision.next_node == "implement"


def test_research_summary_routes_research_then_implement() -> None:
    profile = _profile(TaskIntent.RESEARCH_SUMMARY, requires_research=True)
    decision = decide_workflow(task_profile=profile)
    assert decision.execution_mode == "research_then_implement"
    assert decision.skip_research is False
    assert decision.skip_debate is True
    assert decision.next_node == "research"


def test_rag_answering_routes_research_then_implement() -> None:
    profile = _profile(TaskIntent.RAG_ANSWERING, requires_research=True)
    decision = decide_workflow(task_profile=profile)
    assert decision.execution_mode == "research_then_implement"
    assert decision.skip_research is False
    assert decision.skip_debate is True
    assert decision.next_node == "research"


def test_code_generation_low_skips_debate() -> None:
    profile = _profile(
        TaskIntent.CODE_GENERATION,
        complexity=ComplexityLevel.LOW,
        risk_level=RiskLevel.LOW,
        requires_code=True,
    )
    decision = decide_workflow(task_profile=profile)
    assert decision.execution_mode == "code_focused"
    assert decision.skip_research is False
    assert decision.skip_debate is True
    assert decision.next_node == "research"


def test_debugging_medium_or_high_requires_debate() -> None:
    profile = _profile(
        TaskIntent.DEBUGGING,
        complexity=ComplexityLevel.HIGH,
        risk_level=RiskLevel.MEDIUM,
        requires_code=True,
    )
    decision = decide_workflow(task_profile=profile)
    assert decision.execution_mode == "code_focused"
    assert decision.skip_research is False
    assert decision.skip_debate is False
    assert decision.requires_debate is True


def test_code_review_routes_code_focused() -> None:
    profile = _profile(TaskIntent.CODE_REVIEW, requires_code=True)
    decision = decide_workflow(task_profile=profile)
    assert decision.execution_mode == "code_focused"
    assert decision.skip_research is False


def test_code_modification_routes_code_focused() -> None:
    profile = _profile(TaskIntent.CODE_MODIFICATION, requires_code=True)
    decision = decide_workflow(task_profile=profile)
    assert decision.execution_mode == "code_focused"
    assert decision.skip_research is False


def test_architecture_design_requires_debate() -> None:
    profile = _profile(TaskIntent.ARCHITECTURE_DESIGN, requires_planning=True)
    decision = decide_workflow(task_profile=profile)
    assert decision.execution_mode == "architecture_design"
    assert decision.skip_research is False
    assert decision.skip_debate is False
    assert decision.requires_context is True
    assert decision.requires_debate is True


def test_planning_requires_debate() -> None:
    profile = _profile(TaskIntent.PLANNING, requires_planning=True)
    decision = decide_workflow(task_profile=profile)
    assert decision.execution_mode == "architecture_design"
    assert decision.skip_research is False
    assert decision.skip_debate is False


def test_missing_task_profile_safe_fallback() -> None:
    decision = decide_workflow(state={"run_id": "r1", "session_id": "s1"})
    assert decision.execution_mode == "general"
    assert decision.next_node == "research"
    assert decision.run_id == "r1"
    assert decision.session_id == "s1"


def test_workflow_decision_model_dump_json_stable() -> None:
    decision = WorkflowDecision(
        run_id="r1",
        session_id="s1",
        execution_mode="quick_answer",
        next_node="quick_answer",
    )
    dumped = decision.model_dump(mode="json")
    assert dumped["run_id"] == "r1"
    assert dumped["session_id"] == "s1"
    assert dumped["execution_mode"] == "quick_answer"
    assert "policy_version" in dumped
    assert "metadata" in dumped


def test_policy_does_not_use_legacy_keyword_router_helpers() -> None:
    source = Path("workflow_policy.py").read_text(encoding="utf-8")
    assert "is_coding_task" not in source
    assert "is_complex_task" not in source
