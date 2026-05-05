# coding=utf-8
"""Core graph data models for Neo4j structural memory."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class GraphNode(BaseModel):
    id: str
    label: str
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    source_id: str
    target_id: str
    relation: str
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphPath(BaseModel):
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CodeSymbol(BaseModel):
    symbol_id: str
    name: str
    symbol_type: str
    file_path: str
    module_name: str | None = None
    class_name: str | None = None
    decorators: list[str] = Field(default_factory=list)
    calls: list[str] = Field(default_factory=list)
    lineno: int = 0
    end_lineno: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskGraphRecord(BaseModel):
    run_id: str
    session_id: str = "default"
    task_id: str
    task_profile_id: str
    task_preview: str = ""
    intent: str = ""
    domain: str = ""
    evidence_refs: list[dict[str, Any]] = Field(default_factory=list)
    agent_outputs: list[dict[str, Any]] = Field(default_factory=list)
    review_reports: list[dict[str, Any]] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = Field(default_factory=dict)
