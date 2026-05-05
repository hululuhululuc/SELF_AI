# coding=utf-8
"""Engine event contract for kernel/runtime event emission."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class EngineEvent(BaseModel):
    """Structured engine event record."""

    event: str
    ts: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    run_id: str = ""
    session_id: str = "default"
    call_id: str | None = None
    tool_name: str | None = None
    level: str = "info"
    fields: dict[str, Any] = Field(default_factory=dict)
