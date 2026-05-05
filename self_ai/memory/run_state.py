# coding=utf-8
"""Runtime state models for Redis-backed workflow tracing."""

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


class RunStateSnapshot(BaseModel):
    run_id: str
    session_id: str = "default"
    status: str = "running"  # running | completed | failed
    current_node: str | None = None
    task: str = ""
    task_profile: dict[str, Any] = Field(default_factory=dict)
    retrieval_policy: dict[str, Any] = Field(default_factory=dict)
    memory_write_policy: dict[str, Any] = Field(default_factory=dict)
    research_result_count: int = 0
    message_count: int = 0
    model: str = ""
    has_response: bool = False
    error_count: int = 0
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TraceRecord(BaseModel):
    run_id: str
    event: str
    ts: str = Field(default_factory=utc_now_iso)
    node: str | None = None
    status: str | None = None
    fields: dict[str, Any] = Field(default_factory=dict)


class StoredOutput(BaseModel):
    run_id: str
    name: str
    output_type: str
    content: dict[str, Any] | str
    created_at: str = Field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = Field(default_factory=dict)
