# coding=utf-8
"""Web retrieval tool adapters."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from ..runtime.tool_registry import ToolRegistry
from ..runtime.tool_spec import ToolSpec

WebSearchFunc = Callable[[str, int], Any] | Callable[[str, int], Awaitable[Any]]


def register_web_tools(
    registry: ToolRegistry,
    *,
    web_search_func: WebSearchFunc | None = None,
) -> None:
    """Register web retrieval tools.

    Tests must inject fake web_search_func; default behavior is safe no-network fallback.
    """

    async def _web_search(args: dict[str, Any], _run_context: Any | None) -> dict[str, Any]:
        query = str(args.get("query", "")).strip()
        max_results = int(args.get("max_results", 5))
        if not query:
            raise ValueError("retrieval.web.search requires query")
        if web_search_func is None:
            return {"results": [], "source": "disabled"}

        result = web_search_func(query, max_results)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, dict):
            return {"results": result.get("results", []), "source": result.get("source", "custom")}
        if isinstance(result, list):
            return {"results": result, "source": "custom"}
        return {"results": [], "source": "custom"}

    registry.register(
        ToolSpec(
            name="retrieval.web.search",
            description="Search web via injected provider adapter.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer"},
                },
                "required": ["query"],
            },
            output_schema={"type": "object", "properties": {"results": {"type": "array"}}},
            permission="network",
            timeout_s=20,
            tags=["web", "retrieval"],
            handler=_web_search,
        )
    )
