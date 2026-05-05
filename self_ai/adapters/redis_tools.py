# coding=utf-8
"""Redis runtime state tool adapters."""

from __future__ import annotations

from typing import Any

from ..runtime.tool_registry import ToolRegistry
from ..runtime.tool_spec import ToolSpec


def _resolve_redis_store(
    explicit_store: Any | None,
    run_context: Any | None,
) -> Any:
    if explicit_store is not None:
        return explicit_store
    if run_context is not None:
        metadata = getattr(run_context, "metadata", None)
        if isinstance(metadata, dict) and metadata.get("redis_store") is not None:
            return metadata["redis_store"]
    raise RuntimeError("redis_store is not configured")


def register_redis_tools(registry: ToolRegistry, *, redis_store: Any | None = None) -> None:
    """Register runtime store tools."""

    def _append_trace(args: dict[str, Any], run_context: Any | None) -> dict[str, Any]:
        store = _resolve_redis_store(redis_store, run_context)
        run_id = str(args.get("run_id", "")).strip()
        trace_payload = args.get("trace", {})
        store.append_trace(run_id, trace_payload)
        return {"appended": True}

    def _save_node_output(args: dict[str, Any], run_context: Any | None) -> dict[str, Any]:
        store = _resolve_redis_store(redis_store, run_context)
        store.save_node_output(
            str(args.get("run_id", "")),
            str(args.get("node_name", "")),
            args.get("output", {}),
        )
        return {"saved": True}

    def _save_agent_output(args: dict[str, Any], run_context: Any | None) -> dict[str, Any]:
        store = _resolve_redis_store(redis_store, run_context)
        store.save_agent_output(
            str(args.get("run_id", "")),
            str(args.get("agent_name", "")),
            args.get("output", {}),
        )
        return {"saved": True}

    def _save_run_state(args: dict[str, Any], run_context: Any | None) -> dict[str, Any]:
        store = _resolve_redis_store(redis_store, run_context)
        store.save_run_state(
            str(args.get("run_id", "")),
            args.get("state", {}),
        )
        return {"saved": True}

    def _load_run_state(args: dict[str, Any], run_context: Any | None) -> dict[str, Any]:
        store = _resolve_redis_store(redis_store, run_context)
        state = store.load_run_state(str(args.get("run_id", "")))
        return {"state": state}

    runtime_write_schema = {
        "type": "object",
        "properties": {"run_id": {"type": "string"}},
        "required": ["run_id"],
    }

    registry.register(
        ToolSpec(
            name="storage.runtime.append_trace",
            description="Append runtime trace to RedisStore.",
            input_schema={
                **runtime_write_schema,
                "properties": {
                    **runtime_write_schema["properties"],
                    "trace": {"type": "object"},
                },
                "required": ["run_id", "trace"],
            },
            output_schema={"type": "object", "properties": {"appended": {"type": "boolean"}}},
            permission="runtime_write",
            timeout_s=5,
            tags=["redis", "trace"],
            handler=_append_trace,
        )
    )
    registry.register(
        ToolSpec(
            name="storage.runtime.save_node_output",
            description="Save node output summary to RedisStore.",
            input_schema={
                **runtime_write_schema,
                "properties": {
                    **runtime_write_schema["properties"],
                    "node_name": {"type": "string"},
                    "output": {"type": "object"},
                },
                "required": ["run_id", "node_name", "output"],
            },
            output_schema={"type": "object", "properties": {"saved": {"type": "boolean"}}},
            permission="runtime_write",
            timeout_s=5,
            tags=["redis", "node_output"],
            handler=_save_node_output,
        )
    )
    registry.register(
        ToolSpec(
            name="storage.runtime.save_agent_output",
            description="Save agent output summary to RedisStore.",
            input_schema={
                **runtime_write_schema,
                "properties": {
                    **runtime_write_schema["properties"],
                    "agent_name": {"type": "string"},
                    "output": {"type": "object"},
                },
                "required": ["run_id", "agent_name", "output"],
            },
            output_schema={"type": "object", "properties": {"saved": {"type": "boolean"}}},
            permission="runtime_write",
            timeout_s=5,
            tags=["redis", "agent_output"],
            handler=_save_agent_output,
        )
    )
    registry.register(
        ToolSpec(
            name="storage.runtime.save_run_state",
            description="Save run state snapshot to RedisStore.",
            input_schema={
                **runtime_write_schema,
                "properties": {
                    **runtime_write_schema["properties"],
                    "state": {"type": "object"},
                },
                "required": ["run_id", "state"],
            },
            output_schema={"type": "object", "properties": {"saved": {"type": "boolean"}}},
            permission="runtime_write",
            timeout_s=5,
            tags=["redis", "run_state"],
            handler=_save_run_state,
        )
    )
    registry.register(
        ToolSpec(
            name="storage.runtime.load_run_state",
            description="Load run state snapshot from RedisStore.",
            input_schema=runtime_write_schema,
            output_schema={"type": "object", "properties": {"state": {"type": ["object", "null"]}}},
            permission="read_only",
            timeout_s=5,
            tags=["redis", "run_state"],
            handler=_load_run_state,
        )
    )
