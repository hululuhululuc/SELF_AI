# coding=utf-8
"""Tests for pipeline selection."""

from self_ai.kernel.pipelines import select_pipeline


def test_select_pipeline_quick_answer() -> None:
    stages = select_pipeline(
        {"use_quick_answer": True, "execution_mode": "quick_answer"},
        {"intent": "general_qa"},
    )
    assert stages == ["quick_answer", "finalize"]


def test_select_pipeline_architecture_design_contains_expected_stages() -> None:
    stages = select_pipeline(
        {"execution_mode": "architecture_design", "skip_research": False},
        {"intent": "architecture_design"},
    )
    assert stages[0] == "research"
    assert "plan" in stages
    assert "agent_runtime" in stages
    assert stages[-1] == "finalize"


def test_select_pipeline_skip_research_removes_research_stage() -> None:
    stages = select_pipeline(
        {"execution_mode": "document_writing", "skip_research": True},
        {"intent": "document_writing"},
    )
    assert "research" not in stages
    assert "context_pack" in stages
