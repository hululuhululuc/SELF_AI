# coding=utf-8
"""Tests for Phase 4 graph data models."""

from self_ai.graph.graph_models import CodeSymbol, GraphEdge, GraphNode, GraphPath, TaskGraphRecord


def test_graph_node_serialization() -> None:
    node = GraphNode(id="n1", label="Task", properties={"run_id": "r1"})
    dumped = node.model_dump(mode="json")
    assert dumped["id"] == "n1"
    assert dumped["label"] == "Task"


def test_graph_edge_serialization() -> None:
    edge = GraphEdge(source_id="a", target_id="b", relation="HAS_PROFILE", properties={"k": 1})
    dumped = edge.model_dump(mode="json")
    assert dumped["relation"] == "HAS_PROFILE"
    assert dumped["source_id"] == "a"


def test_graph_path_serialization() -> None:
    path = GraphPath(
        nodes=[GraphNode(id="n1", label="Task")],
        edges=[GraphEdge(source_id="n1", target_id="n2", relation="RELATED_TO")],
        metadata={"run_id": "r1"},
    )
    dumped = path.model_dump(mode="json")
    assert dumped["metadata"]["run_id"] == "r1"
    assert len(dumped["nodes"]) == 1


def test_code_symbol_defaults() -> None:
    symbol = CodeSymbol(symbol_id="s1", name="f", symbol_type="function", file_path="a.py")
    assert symbol.decorators == []
    assert symbol.calls == []
    assert symbol.lineno == 0


def test_task_graph_record_defaults() -> None:
    record = TaskGraphRecord(run_id="r1", task_id="task:r1", task_profile_id="profile:r1")
    assert record.session_id == "default"
    assert record.evidence_refs == []
    assert record.agent_outputs == []
    assert record.review_reports == []
