# coding=utf-8
"""Tests for Phase 6A role-agent selector."""

from pathlib import Path

from self_ai.agents.selector import select_agents_for_state


def _state(
    *,
    intent: str = "general_qa",
    use_quick_answer: bool = False,
    requires_research: bool = False,
    research_count: int = 0,
) -> dict:
    return {
        "workflow_decision": {"use_quick_answer": use_quick_answer},
        "task_profile": {
            "intent": intent,
            "requires_research": requires_research,
        },
        "research_results": [{"x": 1}] * research_count,
    }


def test_selector_quick_answer_returns_empty() -> None:
    assert select_agents_for_state(_state(use_quick_answer=True)) == []


def test_selector_architecture_returns_architect_and_synthesizer() -> None:
    agents = select_agents_for_state(_state(intent="architecture_design"))
    assert agents == ["architect", "synthesizer"]


def test_selector_planning_returns_architect_and_synthesizer() -> None:
    agents = select_agents_for_state(_state(intent="planning"))
    assert agents == ["architect", "synthesizer"]


def test_selector_code_generation_returns_coder_and_synthesizer() -> None:
    agents = select_agents_for_state(_state(intent="code_generation"))
    assert agents == ["coder", "synthesizer"]


def test_selector_debugging_with_research_adds_researcher() -> None:
    agents = select_agents_for_state(
        _state(intent="debugging", requires_research=True, research_count=6)
    )
    assert agents == ["researcher", "coder", "synthesizer"]


def test_selector_research_summary_returns_researcher_and_synthesizer() -> None:
    agents = select_agents_for_state(_state(intent="research_summary"))
    assert agents == ["researcher", "synthesizer"]


def test_selector_rag_answering_returns_researcher_and_synthesizer() -> None:
    agents = select_agents_for_state(_state(intent="rag_answering"))
    assert agents == ["researcher", "synthesizer"]


def test_selector_document_writing_returns_writer_and_synthesizer() -> None:
    agents = select_agents_for_state(_state(intent="document_writing"))
    assert agents == ["writer", "synthesizer"]


def test_selector_general_qa_returns_writer() -> None:
    agents = select_agents_for_state(_state(intent="general_qa"))
    assert agents == ["writer"]


def test_selector_fallback_returns_synthesizer() -> None:
    agents = select_agents_for_state(_state(intent="unknown_intent"))
    assert agents == ["synthesizer"]


def test_selector_does_not_use_legacy_helpers() -> None:
    source = Path("agents/selector.py").read_text(encoding="utf-8")
    assert "is_coding_task" not in source
    assert "is_complex_task" not in source

