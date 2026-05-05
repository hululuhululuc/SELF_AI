# coding=utf-8
"""Tests for task graph builder."""

from self_ai.graph.task_graph_builder import build_task_graph_record, task_graph_to_nodes_edges


def test_build_task_graph_record_from_state() -> None:
    state = {
        "run_id": "run-1",
        "session_id": "session-1",
        "task": "Refactor workflow",
        "task_profile": {"intent": "architecture_design", "domain": "ai_agent"},
        "research_results": [
            {
                "content": "evidence content " + "x" * 300,
                "source_type": "doc",
                "path": "docs/plan.md",
                "chunk_id": "c1",
                "score": 0.8,
            }
        ],
        "agent_outputs": {
            "architect": {"content": "design output " + "y" * 260},
        },
        "review_reports": [
            {
                "pass_review": False,
                "major_issues": ["missing safety"],
                "minor_issues": [],
                "missing_requirements": ["tests"],
                "revision_instruction": "revise the edge cases " + "z" * 240,
                "risk_level": "high",
            }
        ],
    }

    record = build_task_graph_record(state)

    assert record.run_id == "run-1"
    assert record.session_id == "session-1"
    assert record.intent == "architecture_design"
    assert record.domain == "ai_agent"
    assert record.evidence_refs
    assert record.agent_outputs
    assert record.review_reports


def test_task_graph_to_nodes_edges_generation() -> None:
    state = {
        "run_id": "run-2",
        "session_id": "session-2",
        "task": "Investigate issue",
        "task_profile": {"intent": "debugging", "domain": "software"},
        "research_results": [{"content": "evidence", "source_type": "doc", "chunk_id": "c2"}],
        "agent_outputs": {"coder": {"response": "fix applied"}},
        "review_reports": [{"pass_review": True, "major_issues": [], "minor_issues": []}],
    }
    record = build_task_graph_record(state)
    nodes, edges = task_graph_to_nodes_edges(record)

    labels = {n.label for n in nodes}
    relations = {e.relation for e in edges}

    assert "Task" in labels
    assert "TaskProfile" in labels
    assert "Evidence" in labels
    assert "AgentRun" in labels
    assert "ReviewReport" in labels

    assert "HAS_PROFILE" in relations
    assert "USED_EVIDENCE" in relations
    assert "HAS_OUTPUT" in relations
    assert "HAS_REVIEW" in relations


def test_task_graph_builder_does_not_store_large_raw_content() -> None:
    long_text = "A" * 5000
    state = {
        "run_id": "run-3",
        "task": long_text,
        "task_profile": {"intent": "general_qa", "domain": "general"},
        "research_results": [{"content": long_text, "source_type": "doc"}],
        "agent_outputs": {"assistant": {"content": long_text}},
        "review_reports": [{"revision_instruction": long_text}],
    }
    record = build_task_graph_record(state)

    assert len(record.task_preview) < 300
    assert "content" not in record.evidence_refs[0]
    assert len(record.evidence_refs[0]["preview"]) < 300
    assert len(record.agent_outputs[0]["preview"]) < 300
    assert len(record.review_reports[0]["preview"]) < 300


def test_task_graph_builder_handles_empty_collections() -> None:
    state = {
        "run_id": "run-4",
        "task": "simple task",
        "task_profile": {"intent": "general_qa", "domain": "general"},
        "research_results": [],
        "agent_outputs": {},
        "review_reports": [],
    }
    record = build_task_graph_record(state)
    nodes, edges = task_graph_to_nodes_edges(record)

    assert record.evidence_refs == []
    assert record.agent_outputs == []
    assert record.review_reports == []
    assert any(node.label == "Task" for node in nodes)
    assert any(edge.relation == "HAS_PROFILE" for edge in edges)
