# coding=utf-8
"""Phase 3 integration tests for graphrag_plus with mocked retrieval stack."""

from unittest.mock import AsyncMock, patch

import pytest

from self_ai import tools
from self_ai.retrieval.query_planner import RetrievalPlan
from self_ai.schemas import EvidenceItem


class FakeRedisStore:
    def __init__(self) -> None:
        self.traces: list[dict] = []
        self.outputs: list[tuple[str, str, dict]] = []

    def append_trace(self, run_id: str, event: dict) -> None:
        self.traces.append(event)

    def save_node_output(self, run_id: str, node_name: str, output: dict) -> None:
        self.outputs.append((run_id, node_name, output))


@pytest.mark.asyncio
async def test_graphrag_plus_returns_evidence_dicts_from_planned_collections() -> None:
    fake_store = FakeRedisStore()
    tools.set_retrieval_runtime_context(
        run_id="run-123",
        session_id="session-123",
        redis_store=fake_store,
    )

    plan = RetrievalPlan(
        intent="code_generation",
        collections=["cf_code_chunks", "cf_doc_chunks"],
        top_k=4,
        min_score=0.0,
        use_vector=True,
        use_web=False,
        collection_weights={},
    )

    search_mock = AsyncMock(
        side_effect=[
            [
                EvidenceItem(
                    content="code hit",
                    source_type="code",
                    chunk_id="c1",
                    score=0.92,
                )
            ],
            [
                EvidenceItem(
                    content="doc hit",
                    source_type="doc",
                    chunk_id="d1",
                    score=0.81,
                )
            ],
        ]
    )

    with (
        patch("self_ai.tools._encode_text", return_value=[0.1, 0.2, 0.3]),
        patch.object(tools.query_planner, "build", return_value=plan),
        patch.object(tools.qdrant_memory, "search", new=search_mock),
    ):
        result = await tools.graphrag_plus("write function", content_type="code")

    tools.clear_retrieval_runtime_context()

    assert isinstance(result, list)
    assert result
    assert all(isinstance(item, dict) for item in result)
    searched_collections = [call.kwargs["collection_name"] for call in search_mock.call_args_list]
    assert searched_collections[:2] == ["cf_code_chunks", "cf_doc_chunks"]
    assert "docs" not in searched_collections[:2]

    events = [item.get("event") for item in fake_store.traces]
    assert "retrieval.query_planned" in events
    assert "retrieval.qdrant_search.start" in events
    assert "retrieval.qdrant_search.end" in events
    assert "retrieval.context_built" in events
    assert fake_store.outputs


@pytest.mark.asyncio
async def test_graphrag_plus_web_fallback_when_primary_empty() -> None:
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
        patch("self_ai.tools._encode_text", return_value=[0.1, 0.2, 0.3]),
        patch.object(tools.query_planner, "build", return_value=plan),
        patch.object(tools.settings, "disable_legacy_retrieval", False),
        patch(
            "self_ai.tools._search_plan_collections",
            new=AsyncMock(
                side_effect=[
                    [],
                    [{"content": "from web", "source_type": "web", "score": 0.5}],
                ]
            ),
        ),
        patch("self_ai.tools._index_web_result", new=AsyncMock()) as mocked_index,
        patch(
            "self_ai.tools.tavily.search",
            return_value={"results": [{"content": "web content"}]},
        ) as mocked_web,
    ):
        result = await tools.graphrag_plus("latest policy", content_type="general")

    assert result
    mocked_web.assert_called_once()
    mocked_index.assert_called_once()


@pytest.mark.asyncio
async def test_graphrag_plus_no_legacy_fallback_returns_empty_when_primary_empty() -> None:
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
        patch("self_ai.tools._encode_text", return_value=[0.1, 0.2, 0.3]),
        patch.object(tools.query_planner, "build", return_value=plan),
        patch.object(tools.settings, "disable_legacy_retrieval", True),
        patch("self_ai.tools._search_plan_collections", new=AsyncMock(return_value=[])),
        patch("self_ai.tools.tavily.search") as mocked_web,
    ):
        result = await tools.graphrag_plus("latest policy", content_type="general")

    assert result == []
    mocked_web.assert_not_called()
