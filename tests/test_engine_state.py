# coding=utf-8
"""Tests for EngineState model helpers."""

from datetime import datetime

from self_ai.kernel.engine_state import EngineState


def test_engine_state_from_input_and_to_result() -> None:
    state = EngineState.from_input("Hello", session_id="s1", run_id="r1")
    assert state.run_id == "r1"
    assert state.session_id == "s1"
    assert state.task == "Hello"
    assert state.normalized_task
    assert state.created_at
    assert state.updated_at

    state.response = "ok"
    state.model = "mock"
    result = state.to_result()
    assert result["response"] == "ok"
    assert result["model"] == "mock"
    assert result["run_id"] == "r1"
    assert result["created_at"]
    assert result["updated_at"]


def test_engine_state_trace_and_error_update_timestamp() -> None:
    state = EngineState.from_input("task", session_id="s2", run_id="r2")
    original_updated = state.updated_at
    state.append_trace("evt", stage="x", status="start")
    assert datetime.fromisoformat(state.updated_at) >= datetime.fromisoformat(original_updated)
    trace_updated = state.updated_at
    state.add_error("Boom", "failed", stage="x")
    assert datetime.fromisoformat(state.updated_at) >= datetime.fromisoformat(trace_updated)
    dumped = state.to_dict()
    assert dumped["run_id"] == "r2"
    assert len(dumped["trace"]) == 1
    assert len(dumped["errors"]) == 1
    assert dumped["created_at"]
    assert dumped["updated_at"]
