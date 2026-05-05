# coding=utf-8
"""Tests for Phase 5B ContextPack building and prompt rendering."""

from self_ai.context_pack import (
    ContextPack,
    build_context_pack_from_state,
    context_pack_to_prompt_block,
)


def _state() -> dict:
    return {
        "run_id": "run-ctx-1",
        "session_id": "session-ctx-1",
        "task": "debug ValueError in main.py and provide fix",
        "task_profile": {"intent": "debugging", "requires_code": True},
        "workflow_decision": {"execution_mode": "code_focused", "skip_debate": False},
        "plan": {
            "planner_version": "phase5b.v1",
            "steps": [
                {"step_id": "s1", "goal": "Locate issue", "done_criteria": "root cause found"},
                {"step_id": "s2", "goal": "Apply fix", "done_criteria": "tests pass"},
            ],
        },
        "research_results": [
            {
                "content": "Traceback shows failure in parse_input() due to null value",
                "source_type": "code",
                "path": "main.py",
                "chunk_id": "c1",
                "score": 0.93,
            },
            {
                "content": "A prior fix used defensive checks before parsing payload",
                "source_type": "review",
                "chunk_id": "r1",
                "score": 0.71,
            },
        ],
        "graph_paths": [
            {
                "nodes": [{"id": "f:parse_input", "label": "Function"}],
                "edges": [{"source_id": "f:parse_input", "target_id": "f:validate", "relation": "CALLS"}],
                "metadata": {"score": 0.8},
            }
        ],
        "graph_summary": {"path_count": 1, "relation_type_counts": {"CALLS": 1}},
        "graph_context_preview": "parse_input CALLS validate before transform",
    }


def test_build_context_pack_reads_profile_decision_plan_and_research() -> None:
    cp = build_context_pack_from_state(_state(), token_budget=4000)
    assert cp.run_id == "run-ctx-1"
    assert cp.session_id == "session-ctx-1"
    assert cp.task_profile["intent"] == "debugging"
    assert cp.workflow_decision["execution_mode"] == "code_focused"
    assert cp.plan["planner_version"] == "phase5b.v1"
    assert len(cp.evidence_items) == 2


def test_build_context_pack_extracts_evidence_groups_and_preview() -> None:
    cp = build_context_pack_from_state(_state())
    assert cp.source_type_groups["code"] == 1
    assert cp.source_type_groups["review"] == 1
    assert cp.context_preview


def test_build_context_pack_extracts_graph_fields() -> None:
    cp = build_context_pack_from_state(_state())
    assert len(cp.graph_paths) == 1
    assert cp.graph_summary["path_count"] == 1
    assert cp.graph_context_preview


def test_build_context_pack_safe_fallback_without_research_results() -> None:
    state = _state()
    state["research_results"] = []
    cp = build_context_pack_from_state(state)
    assert cp.evidence_items == []
    assert cp.source_type_groups == {}
    assert cp.context_preview == ""


def test_context_pack_to_prompt_block_contains_required_sections() -> None:
    cp = build_context_pack_from_state(_state())
    block = context_pack_to_prompt_block(cp, max_chars=6000)
    assert "Task Summary:" in block
    assert "Workflow Decision:" in block
    assert "Plan:" in block
    assert "Semantic Evidence Summary:" in block
    assert "Graph Context Summary:" in block


def test_context_pack_to_prompt_block_does_not_output_large_raw_text() -> None:
    state = _state()
    huge = "X" * 12000
    state["research_results"] = [{"content": huge, "source_type": "doc"}]
    cp = build_context_pack_from_state(state)
    block = context_pack_to_prompt_block(cp, max_chars=2000)
    assert huge not in block


def test_context_pack_to_prompt_block_has_length_limit() -> None:
    state = _state()
    state["research_results"] = [
        {"content": "Y" * 5000, "source_type": "doc"},
        {"content": "Z" * 5000, "source_type": "web"},
    ]
    cp = build_context_pack_from_state(state)
    block = context_pack_to_prompt_block(cp, max_chars=1200)
    assert len(block) <= 1200


def test_context_pack_model_dump_json_stable() -> None:
    cp = build_context_pack_from_state(_state())
    dumped = cp.model_dump(mode="json")
    assert dumped["run_id"] == "run-ctx-1"
    assert "evidence_items" in dumped
    assert "graph_summary" in dumped
    assert "token_budget" in dumped
    parsed = ContextPack(**dumped)
    assert parsed.run_id == cp.run_id
