# coding=utf-8
"""Tests for AgentPolicy rules."""

from self_ai.agent_runtime.agent_policy import AgentPolicy
from self_ai.agent_runtime.agent_session import AgentSession


def _session(intent: str, *, quick: bool = False, requires_research: bool = False) -> AgentSession:
    return AgentSession(
        run_id="r",
        session_id="s",
        task="t",
        task_profile={"intent": intent, "requires_research": requires_research},
        workflow_decision={"use_quick_answer": quick, "max_iterations": 2},
        context_pack={"evidence_items": []},
    )


def test_policy_quick_answer_disables_agents_review_gate_and_revision() -> None:
    policy = AgentPolicy()
    s = _session("general_qa", quick=True)
    assert policy.select_agents(s) == []
    assert policy.should_review(s) is False
    assert policy.should_run_quality_gate(s) is False


def test_policy_selects_agents_by_intent() -> None:
    policy = AgentPolicy()
    assert policy.select_agents(_session("document_writing")) == ["writer", "synthesizer"]
    assert policy.select_agents(_session("code_generation")) == ["coder", "synthesizer"]
    assert policy.select_agents(_session("debugging")) == ["coder", "synthesizer"]
    assert policy.select_agents(_session("architecture_design")) == ["architect", "synthesizer"]
    assert policy.select_agents(_session("research_summary")) == ["researcher", "synthesizer"]
    assert policy.select_agents(_session("general_qa")) == ["synthesizer"]


def test_policy_revision_controls_and_target() -> None:
    policy = AgentPolicy()
    s = _session("code_generation")
    s.selected_agents = ["coder", "synthesizer"]
    s.quality_gate = {"decision": "revise", "revision_target_agent": "coder", "requires_revision": True}
    assert policy.should_revise(s) is True
    assert policy.select_revision_target(s) == "coder"
    assert policy.max_revision_iterations(s) == 2

