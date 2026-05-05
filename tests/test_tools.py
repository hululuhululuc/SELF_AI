# coding=utf-8
"""Unit tests for tools module with full mocking."""

from unittest.mock import AsyncMock, patch

import pytest

from self_ai import tools
from self_ai.retrieval.query_planner import RetrievalPlan


@pytest.mark.asyncio
async def test_graphrag_plus_code_query_skips_web_fallback() -> None:
    plan = RetrievalPlan(
        intent="code_generation",
        collections=["cf_code_chunks", "cf_doc_chunks"],
        top_k=5,
        min_score=0.0,
        use_vector=True,
        use_web=False,
        collection_weights={},
    )
    with (
        patch("self_ai.tools._encode_text", return_value=[0.1, 0.2]),
        patch.object(tools.query_planner, "build", return_value=plan),
        patch(
            "self_ai.tools._search_plan_collections",
            new=AsyncMock(return_value=[{"content": "code chunk", "source_type": "code", "score": 0.9}]),
        ),
        patch("self_ai.tools.tavily.search") as mocked_web,
    ):
        result = await tools.graphrag_plus("write python function", content_type="code")
    assert len(result) == 1
    mocked_web.assert_not_called()


@pytest.mark.asyncio
async def test_graphrag_plus_general_query_uses_web_fallback_on_empty_vector() -> None:
    plan = RetrievalPlan(
        intent="general_qa",
        collections=["cf_doc_chunks"],
        top_k=5,
        min_score=0.0,
        use_vector=True,
        use_web=False,
        collection_weights={},
    )
    with (
        patch("self_ai.tools._encode_text", return_value=[0.1, 0.2]),
        patch.object(tools.query_planner, "build", return_value=plan),
        patch.object(tools.settings, "disable_legacy_retrieval", False),
        patch(
            "self_ai.tools._search_plan_collections",
            new=AsyncMock(side_effect=[[], [{"content": "from web", "source_type": "web", "score": 0.5}]]),
        ),
        patch("self_ai.tools._index_web_result", new=AsyncMock()),
        patch(
            "self_ai.tools.tavily.search",
            return_value={"results": [{"content": "web content"}]},
        ) as mocked_web,
    ):
        result = await tools.graphrag_plus("latest policy news", content_type="general")
    assert result
    mocked_web.assert_called_once()
