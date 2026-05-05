# coding=utf-8
"""Tool result object for Self AI ToolRuntime."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Standardized tool execution result envelope."""

    call_id: str
    tool_name: str
    ok: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error: dict[str, Any] | None = None
    latency_ms: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def success(
        cls,
        *,
        call_id: str,
        tool_name: str,
        data: dict[str, Any] | None = None,
        latency_ms: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> "ToolResult":
        return cls(
            call_id=call_id,
            tool_name=tool_name,
            ok=True,
            data=data or {},
            error=None,
            latency_ms=max(0, int(latency_ms)),
            metadata=metadata or {},
        )

    @classmethod
    def failure(
        cls,
        *,
        call_id: str,
        tool_name: str,
        error_type: str,
        message: str,
        latency_ms: int = 0,
        metadata: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
    ) -> "ToolResult":
        return cls(
            call_id=call_id,
            tool_name=tool_name,
            ok=False,
            data=data or {},
            error={"type": error_type, "message": message},
            latency_ms=max(0, int(latency_ms)),
            metadata=metadata or {},
        )
