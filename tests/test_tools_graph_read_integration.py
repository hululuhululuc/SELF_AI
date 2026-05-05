# coding=utf-8
"""Phase 4.5 integration tests for graph read-only supplement in tools.graphrag_plus."""

from unittest.mock import AsyncMock, patch

import pytest

from self_ai import tools
from self_ai.graph.graph_models import GraphEdge, GraphNode, GraphPath
from self_ai.retrieval.graph_query_adapter import GraphQueryPlan
from self_ai.retrieval.query_planner import RetrievalPlan
from self_ai.schemas import RetrievalPolicy, TaskIntent, TaskProfile


class FakeRedisStore:
    def __init__(self) -> None:
        self.traces: list[dict] = []
        self.outputs: list[tuple[str, str, dict]] = []

    def append_trace(self, run_id: str, event: dict) -> None:
        self.traces.append(event)

    def save_node_output(self, run_id: str, node_name: str, output: dict) -> None:
        self.outputs.append((run_id, node_name, output))


def _graph_path() -> GraphPath:
    return GraphPath(
        nodes=[GraphNode(id="f:a", label="Function"), GraphNode(id="f:b", label="Function")],
        edges=[GraphEdge(source_id="f:a", target_id="f:b", relation="CALLS")],
        metadata={"score": 0.8},
    )


@pytest.mark.asyncio
async def test_graphrag_plus_keeps_interface_and_qdrant_main_path_with_graph_supplement() -> None:
    fake_store = FakeRedisStore()
    tools.set_retrieval_runtime_context(run_id="run-g1", session_id="session-g1", redis_store=fake_store)
    profile = TaskProfile(intent=TaskIntent.CODE_GENERATION)
    profile.retrieval_policy = RetrievalPolicy(use_graph=True)

    plan = RetrievalPlan(
        intent="code_generation",
        collections=["cf_code_chunks", "cf_doc_chunks"],
        top_k=3,
        min_score=0.0,
        use_vector=True,
        use_web=False,
        collection_weights={},
    )

    with (
        patch("self_ai.tools._encode_text", return_value=[0.1, 0.2]),
        patch("self_ai.tools._profile_from_content_type", return_value=profile),
        patch.object(tools.query_planner, "build", return_value=plan),
        patch("self_ai.tools._search_plan_collections", new=AsyncMock(return_value=[{"content": "code hit", "source_type": "code", "score": 0.9}])),
        patch.object(tools.settings, "graph_read_enabled", True),
        patch.object(tools.graph_query_adapter, "enabled", True),
        patch.object(
            tools.graph_query_adapter,
            "plan_graph_queries",
            return_value=GraphQueryPlan(
                enabled=True,
                query_types=["find_related_symbols"],
                symbols=["merge_k_lists"],
                run_id="run-g1",
                max_paths=3,
            ),
        ) as mocked_plan,
        patch.object(tools.graph_query_adapter, "run_graph_queries", return_value=[_graph_path()]) as mocked_run,
    ):
        result = await tools.graphrag_plus("merge_k_lists", content_type="code")

    tools.clear_retrieval_runtime_context()

    assert isinstance(result, list)
    assert result and isinstance(result[0], dict)
    assert result[0]["content"] == "code hit"
    mocked_plan.assert_called_once()
    mocked_run.assert_called_once()
    events = [item.get("event") for item in fake_store.traces]
    assert "retrieval.graph_query_planned" in events
    assert "retrieval.graph_query.start" in events
    assert "retrieval.graph_query.end" in events
    assert "retrieval.graph_context_built" in events


@pytest.mark.asyncio
async def test_graph_query_failure_does_not_break_evidence_return() -> None:
    plan = RetrievalPlan(
        intent="debugging",
        collections=["cf_code_chunks"],
        top_k=3,
        min_score=0.0,
        use_vector=True,
        use_web=False,
        collection_weights={},
    )
    with (
        patch("self_ai.tools._encode_text", return_value=[0.1, 0.2]),
        patch.object(tools.query_planner, "build", return_value=plan),
        patch("self_ai.tools._search_plan_collections", new=AsyncMock(return_value=[{"content": "keep evidence", "source_type": "code", "score": 0.91}])),
        patch.object(tools.settings, "graph_read_enabled", True),
        patch.object(tools.graph_query_adapter, "enabled", True),
        patch.object(
            tools.graph_query_adapter,
            "plan_graph_queries",
            return_value=GraphQueryPlan(
                enabled=True,
                query_types=["find_prior_issues"],
                issue_text="ValueError",
                run_id="run-g2",
                max_paths=3,
            ),
        ),
        patch.object(tools.graph_query_adapter, "run_graph_queries", side_effect=RuntimeError("neo4j down")),
    ):
        result = await tools.graphrag_plus("main.py ValueError", content_type="code")

    assert result
    assert result[0]["content"] == "keep evidence"


@pytest.mark.asyncio
async def test_use_graph_false_skips_graph_query_adapter() -> None:
    profile = TaskProfile(intent=TaskIntent.GENERAL_QA)
    profile.retrieval_policy = RetrievalPolicy(use_graph=False)
    plan = RetrievalPlan(
        intent="general_qa",
        collections=["cf_doc_chunks"],
        top_k=3,
        min_score=0.0,
        use_vector=True,
        use_web=False,
        collection_weights={},
    )
    with (
        patch("self_ai.tools._encode_text", return_value=[0.1, 0.2]),
        patch("self_ai.tools._profile_from_content_type", return_value=profile),
        patch.object(tools.query_planner, "build", return_value=plan),
        patch("self_ai.tools._search_plan_collections", new=AsyncMock(return_value=[{"content": "doc", "source_type": "doc", "score": 0.5}])),
        patch.object(tools.settings, "graph_read_enabled", True),
        patch.object(tools.graph_query_adapter, "enabled", True),
        patch.object(tools.graph_query_adapter, "plan_graph_queries") as mocked_plan,
    ):
        result = await tools.graphrag_plus("normal qa", content_type="general")

    assert result
    mocked_plan.assert_not_called()


@pytest.mark.asyncio
async def test_graph_trace_contains_summary_only() -> None:
    fake_store = FakeRedisStore()
    tools.set_retrieval_runtime_context(run_id="run-g3", session_id="session-g3", redis_store=fake_store)
    profile = TaskProfile(intent=TaskIntent.CODE_GENERATION)
    profile.retrieval_policy = RetrievalPolicy(use_graph=True)
    plan = RetrievalPlan(
        intent="code_generation",
        collections=["cf_code_chunks"],
        top_k=3,
        min_score=0.0,
        use_vector=True,
        use_web=False,
        collection_weights={},
    )
    with (
        patch("self_ai.tools._encode_text", return_value=[0.1, 0.2]),
        patch("self_ai.tools._profile_from_content_type", return_value=profile),
        patch.object(tools.query_planner, "build", return_value=plan),
        patch("self_ai.tools._search_plan_collections", new=AsyncMock(return_value=[{"content": "small evidence", "source_type": "code", "score": 0.9}])),
        patch.object(tools.settings, "graph_read_enabled", True),
        patch.object(tools.graph_query_adapter, "enabled", True),
        patch.object(
            tools.graph_query_adapter,
            "plan_graph_queries",
            return_value=GraphQueryPlan(
                enabled=True,
                query_types=["find_related_symbols"],
                symbols=["merge_k_lists"],
                run_id="run-g3",
                max_paths=2,
            ),
        ),
        patch.object(tools.graph_query_adapter, "run_graph_queries", return_value=[_graph_path()]),
    ):
        _ = await tools.graphrag_plus("merge_k_lists", content_type="code")

    tools.clear_retrieval_runtime_context()

    graph_events = [
        item for item in fake_store.traces if str(item.get("event", "")).startswith("retrieval.graph_")
    ]
    assert graph_events
    for item in graph_events:
        assert "run_id" in item
        assert "session_id" in item
    # Ensure node output summary exists and does not contain full graph path details.
    assert fake_store.outputs
    summary = fake_store.outputs[-1][2]
    assert "graph_path_count" in summary
    assert "graph_relation_type_counts" in summary
