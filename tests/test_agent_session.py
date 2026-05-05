# coding=utf-8
"""Tests for AgentSession mapping and mutation helpers."""

from self_ai.agent_runtime.agent_session import AgentSession
from self_ai.kernel.engine_state import EngineState


def test_agent_session_from_engine_state_and_updates() -> None:
    state = EngineState.from_input("task", run_id="r1", session_id="s1")
    state.task_profile = {"intent": "code_generation"}
    state.workflow_decision = {"execution_mode": "code_focused"}
    state.plan = {"steps": [{"step_id": "s1"}]}
    state.context_pack = {"evidence_items": [{"content": "x"}]}
    session = AgentSession.from_engine_state(state)

    assert session.run_id == "r1"
    assert session.session_id == "s1"
    assert session.task == state.normalized_task
    assert session.task_profile["intent"] == "code_generation"

    session.selected_agents = ["coder", "synthesizer"]
    session.add_agent_output("coder", {"output": "fix"})
    session.add_review_report({"pass_review": True})
    session.set_quality_gate({"decision": "warn"})
    updates = session.to_engine_state_updates()

    assert updates["selected_agents"] == ["coder", "synthesizer"]
    assert "coder" in updates["agent_outputs"]
    assert len(updates["review_reports"]) == 1
    assert updates["quality_gate"]["decision"] == "warn"


def test_agent_session_mutators_touch_updated_at() -> None:
    session = AgentSession(run_id="r2", session_id="s2", task="task")
    ts0 = session.updated_at
    session.append_message(sender="a", recipient="b", content="hello")
    assert session.updated_at >= ts0
    ts1 = session.updated_at
    session.add_error("E", "boom", stage="x")
    assert session.updated_at >= ts1

