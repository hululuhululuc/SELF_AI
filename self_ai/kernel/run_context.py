# coding=utf-8
"""Run context model for Kernel + EngineLoop."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class RunContext(BaseModel):
    """Execution context detached from orchestration framework."""

    project_root: Path
    run_id: str = ""
    session_id: str = "default"
    settings: Any | None = None
    permission_profile: dict[str, bool] = Field(default_factory=dict)
    tool_runtime: Any | None = Field(default=None, exclude=True)
    state_store: Any | None = Field(default=None, exclude=True)
    event_sink: Any | None = Field(default=None, exclude=True)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}
