# coding=utf-8
"""Tests for ResultAggregator."""

from self_ai.kernel.engine_state import EngineState
from self_ai.kernel.result_aggregator import ResultAggregator
from self_ai.runtime.tool_result import ToolResult


def test_result_aggregator_merges_model_generate_result() -> None:
    state = EngineState.from_input("task", run_id="r1", session_id="s1")
    agg = ResultAggregator()
    result = ToolResult.success(
        call_id="c1",
        tool_name="model.generate",
        data={"model": "mock", "response": "hello"},
    )

    agg.merge_tool_result(state, result, "final_response")

    assert state.model == "mock"
    assert state.response == "hello"
    assert "model_outputs" in state.metadata


def test_result_aggregator_merges_failed_tool_result_into_errors() -> None:
    state = EngineState.from_input("task", run_id="r1", session_id="s1")
    agg = ResultAggregator()
    result = ToolResult.failure(
        call_id="c2",
        tool_name="storage.semantic.search",
        error_type="SearchError",
        message="search failed",
    )

    agg.merge_tool_result(state, result, "research")

    assert len(state.errors) == 1
    assert state.errors[0]["type"] == "SearchError"
    assert state.errors[0]["stage"] == "research"


def test_result_aggregator_merges_agent_session_updates() -> None:
    state = EngineState.from_input("task", run_id="r3", session_id="s3")
    agg = ResultAggregator()
    updates = {
        "selected_agents": ["coder", "synthesizer"],
        "agent_outputs": {"coder": {"output": "fix"}, "synthesizer": {"output": "summary"}},
        "review_reports": [{"pass_review": True}],
        "quality_gate": {"decision": "warn"},
        "errors": [{"type": "Minor", "message": "minor issue", "stage": "agent_runtime"}],
        "metadata": {
            "revision_count": 1,
            "max_revision_iterations": 1,
            "revision_performed": True,
            "revision_target_agent": "coder",
        },
    }

    agg.merge_agent_session(state, updates)

    assert state.selected_agents == ["coder", "synthesizer"]
    assert "coder" in state.agent_outputs
    assert len(state.review_reports) == 1
    assert state.quality_gate["decision"] == "warn"
    assert state.metadata["revision_count"] == 1
    assert any(err["type"] == "Minor" for err in state.errors)
