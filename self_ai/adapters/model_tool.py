# coding=utf-8
"""Model tool adapter (temporary bridge to existing route_model)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from ..runtime.tool_registry import ToolRegistry
from ..runtime.tool_spec import ToolSpec

RouteModelFunc = Callable[..., Awaitable[dict[str, Any]]]


def register_model_tools(
    registry: ToolRegistry,
    *,
    route_model_func: RouteModelFunc | None = None,
) -> None:
    """Register model.generate tool.

    Step-1 note:
    This still bridges to route_model internally. Step-2 should remove this dependency
    by introducing direct ModelGateway.
    """

    async def _model_generate(args: dict[str, Any], _run_context: Any | None) -> dict[str, Any]:
        task = str(args.get("prompt") or args.get("task") or "").strip()
        if not task:
            raise ValueError("model.generate requires prompt or task")
        stage = str(args.get("stage") or args.get("category") or "general")
        hints = args.get("hints", {})
        call = route_model_func
        if call is None:
            from ..router import route_model as call  # lazy import

        # Keep backward compatibility with 2-arg injected callables in tests.
        try:
            result = await call(task, stage, hints)
        except TypeError:
            result = await call(task, stage)
        return {
            "model": str(result.get("model", "")),
            "response": str(result.get("response", "")),
        }

    registry.register(
        ToolSpec(
            name="model.generate",
            description="Generate response from configured model gateway.",
            input_schema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "task": {"type": "string"},
                    "category": {"type": "string"},
                    "stage": {"type": "string"},
                    "hints": {"type": "object"},
                },
                "required": [],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "model": {"type": "string"},
                    "response": {"type": "string"},
                },
                "required": ["model", "response"],
            },
            permission="model_call",
            timeout_s=120,
            tags=["model", "llm"],
            handler=_model_generate,
        )
    )
