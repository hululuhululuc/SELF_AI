# coding=utf-8
"""Tool specification contract for Self AI ToolRuntime."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field, field_validator

ToolHandler = Callable[[dict[str, Any], Any | None], Any]


class ToolSpec(BaseModel):
    """Describes one runtime tool."""

    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    permission: str = "read_only"
    timeout_s: float = 30.0
    tags: list[str] = Field(default_factory=list)
    handler: ToolHandler = Field(exclude=True, repr=False)

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        name = value.strip()
        if not name:
            raise ValueError("tool name cannot be empty")
        return name

    @field_validator("timeout_s")
    @classmethod
    def _validate_timeout(cls, value: float) -> float:
        timeout = float(value)
        if timeout <= 0:
            raise ValueError("timeout_s must be > 0")
        return timeout
