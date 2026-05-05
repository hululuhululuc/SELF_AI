# coding=utf-8
"""Neo4j graph memory tool adapters."""

from __future__ import annotations

from typing import Any

from ..runtime.tool_registry import ToolRegistry
from ..runtime.tool_spec import ToolSpec


def _resolve_graph_memory(explicit_memory: Any | None, run_context: Any | None) -> Any:
    if explicit_memory is not None:
        return explicit_memory
    if run_context is not None:
        metadata = getattr(run_context, "metadata", None)
        if isinstance(metadata, dict) and metadata.get("graph_memory") is not None:
            return metadata["graph_memory"]
    raise RuntimeError("graph_memory is not configured")


def _dump_graph_item(item: Any) -> dict[str, Any]:
    if hasattr(item, "model_dump"):
        return item.model_dump(mode="json")
    if isinstance(item, dict):
        return item
    return {"repr": repr(item)}


def register_neo4j_tools(registry: ToolRegistry, *, graph_memory: Any | None = None) -> None:
    """Register Neo4j graph tools."""

    def _find_task_lineage(args: dict[str, Any], run_context: Any | None) -> dict[str, Any]:
        try:
            memory = _resolve_graph_memory(graph_memory, run_context)
            path = memory.find_task_lineage(
                str(args.get("run_id", "")),
                limit=int(args.get("limit", 10)),
            )
            if path is None:
                return {"paths": []}
            return {"paths": [_dump_graph_item(path)]}
        except Exception:
            return {"paths": []}

    def _find_related_symbols(args: dict[str, Any], run_context: Any | None) -> dict[str, Any]:
        try:
            memory = _resolve_graph_memory(graph_memory, run_context)
            paths = memory.find_related_symbols(
                str(args.get("symbol_name", "")),
                limit=int(args.get("limit", 10)),
            )
            return {"paths": [_dump_graph_item(item) for item in (paths or [])]}
        except Exception:
            return {"paths": []}

    def _find_prior_issues(args: dict[str, Any], run_context: Any | None) -> dict[str, Any]:
        try:
            memory = _resolve_graph_memory(graph_memory, run_context)
            paths = memory.find_prior_issues(
                str(args.get("query", "")),
                limit=int(args.get("limit", 10)),
            )
            return {"paths": [_dump_graph_item(item) for item in (paths or [])]}
        except Exception:
            return {"paths": []}

    def _record_issue_fix(args: dict[str, Any], run_context: Any | None) -> dict[str, Any]:
        memory = _resolve_graph_memory(graph_memory, run_context)
        ok = memory.record_issue_fix(
            run_id=str(args.get("run_id", "")),
            issue=args.get("issue", {}),
            fix=args.get("fix"),
            test=args.get("test"),
        )
        if ok is False:
            raise RuntimeError("Neo4j record_issue_fix failed")
        return {"recorded": True}

    registry.register(
        ToolSpec(
            name="storage.graph.find_task_lineage",
            description="Find task lineage from Neo4j graph memory.",
            input_schema={
                "type": "object",
                "properties": {"run_id": {"type": "string"}, "limit": {"type": "integer"}},
                "required": ["run_id"],
            },
            output_schema={"type": "object", "properties": {"paths": {"type": "array"}}},
            permission="memory_read",
            timeout_s=10,
            tags=["neo4j", "graph_read"],
            handler=_find_task_lineage,
        )
    )
    registry.register(
        ToolSpec(
            name="storage.graph.find_related_symbols",
            description="Find related symbols from Neo4j graph memory.",
            input_schema={
                "type": "object",
                "properties": {"symbol_name": {"type": "string"}, "limit": {"type": "integer"}},
                "required": ["symbol_name"],
            },
            output_schema={"type": "object", "properties": {"paths": {"type": "array"}}},
            permission="memory_read",
            timeout_s=10,
            tags=["neo4j", "graph_read"],
            handler=_find_related_symbols,
        )
    )
    registry.register(
        ToolSpec(
            name="storage.graph.find_prior_issues",
            description="Find prior issues from Neo4j graph memory.",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
                "required": ["query"],
            },
            output_schema={"type": "object", "properties": {"paths": {"type": "array"}}},
            permission="memory_read",
            timeout_s=10,
            tags=["neo4j", "graph_read"],
            handler=_find_prior_issues,
        )
    )
    registry.register(
        ToolSpec(
            name="storage.graph.record_issue_fix",
            description="Record issue/fix summary into Neo4j graph memory.",
            input_schema={
                "type": "object",
                "properties": {
                    "run_id": {"type": "string"},
                    "issue": {"type": "object"},
                    "fix": {"type": ["object", "null"]},
                    "test": {"type": ["object", "null"]},
                },
                "required": ["run_id", "issue"],
            },
            output_schema={"type": "object", "properties": {"recorded": {"type": "boolean"}}},
            permission="memory_write",
            timeout_s=10,
            tags=["neo4j", "graph_write"],
            handler=_record_issue_fix,
        )
    )
