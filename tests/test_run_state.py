# coding=utf-8
"""Unit tests for runtime state models."""

from self_ai.memory.run_state import RunStateSnapshot, StoredOutput, TraceRecord


def test_run_state_snapshot_defaults() -> None:
    snap = RunStateSnapshot(run_id="r1")
    assert snap.session_id == "default"
    assert snap.status == "running"
    assert snap.current_node is None
    assert snap.research_result_count == 0
    assert snap.message_count == 0
    assert snap.error_count == 0
    assert snap.created_at
    assert snap.updated_at


def test_trace_record_has_default_ts() -> None:
    rec = TraceRecord(run_id="r1", event="workflow.node.start")
    assert rec.ts
    assert rec.node is None
    assert rec.fields == {}


def test_stored_output_serialization() -> None:
    output = StoredOutput(
        run_id="r1",
        name="implement",
        output_type="node",
        content={"model": "mock", "preview": "abc"},
    )
    dumped = output.model_dump(mode="json")
    assert dumped["run_id"] == "r1"
    assert dumped["name"] == "implement"
    assert dumped["output_type"] == "node"
    assert isinstance(dumped["content"], dict)
    assert dumped["created_at"]
