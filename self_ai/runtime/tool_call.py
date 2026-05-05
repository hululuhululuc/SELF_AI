# coding=utf-8
"""Tool call object for Self AI ToolRuntime."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """One execution request to a tool."""

    call_id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str = ""
    session_id: str = "default"
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
