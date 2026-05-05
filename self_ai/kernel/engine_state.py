# coding=utf-8
"""Engine state model for Kernel + EngineLoop runtime."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from ..text_utils import normalize_text


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class EngineState(BaseModel):
    """Single runtime state structure used by EngineLoop."""

    run_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str = "default"
    task: str = ""
    normalized_task: str = ""
    messages: list[dict[str, Any]] = Field(default_factory=list)

    task_profile: dict[str, Any] = Field(default_factory=dict)
    retrieval_policy: dict[str, Any] = Field(default_factory=dict)
    memory_write_policy: dict[str, Any] = Field(default_factory=dict)

    workflow_decision: dict[str, Any] = Field(default_factory=dict)
    plan: dict[str, Any] = Field(default_factory=dict)
    context_pack: dict[str, Any] = Field(default_factory=dict)

    selected_agents: list[str] = Field(default_factory=list)
    research_results: list[dict[str, Any]] = Field(default_factory=list)
    agent_outputs: dict[str, Any] = Field(default_factory=dict)
    review_reports: list[dict[str, Any]] = Field(default_factory=list)
    quality_gate: dict[str, Any] = Field(default_factory=dict)

    response: str = ""
    model: str = ""
    errors: list[dict[str, Any]] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_utc_now_iso)
    updated_at: str = Field(default_factory=_utc_now_iso)

    @classmethod
    def from_input(
        cls,
        input_text: str,
        *,
        session_id: str = "default",
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "EngineState":
        normalized = normalize_text(input_text)
        now = _utc_now_iso()
        return cls(
            run_id=run_id or str(uuid4()),
            session_id=session_id,
            task=input_text,
            normalized_task=normalized,
            metadata=metadata or {},
            created_at=now,
            updated_at=now,
        )

    def append_trace(
        self,
        event: str,
        *,
        stage: str | None = None,
        status: str | None = None,
        fields: dict[str, Any] | None = None,
    ) -> None:
        self.trace.append(
            {
                "event": event,
                "stage": stage,
                "status": status,
                "run_id": self.run_id,
                "session_id": self.session_id,
                "ts": _utc_now_iso(),
                "fields": fields or {},
            }
        )
        self.updated_at = _utc_now_iso()

    def add_error(
        self,
        error_type: str,
        message: str,
        *,
        stage: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.errors.append(
            {
                "type": error_type,
                "message": str(message)[:500],
                "stage": stage,
                "metadata": metadata or {},
            }
        )
        self.updated_at = _utc_now_iso()

    def add_control_event(
        self,
        event_type: str,
        *,
        stage: str | None = None,
        reason: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        events = self.metadata.setdefault("control_events", [])
        if not isinstance(events, list):
            events = []
            self.metadata["control_events"] = events
        events.append(
            {
                "type": event_type,
                "stage": stage,
                "reason": str(reason)[:300],
                "metadata": metadata or {},
                "ts": _utc_now_iso(),
            }
        )
        self.updated_at = _utc_now_iso()

    def to_result(self) -> dict[str, Any]:
        """Return frontend/API-compatible result payload."""
        return {
            "response": self.response,
            "model": self.model,
            "run_id": self.run_id,
            "session_id": self.session_id,
            "task_profile": self.task_profile,
            "workflow_decision": self.workflow_decision,
            "plan": self.plan,
            "context_pack": self.context_pack,
            "selected_agents": self.selected_agents,
            "agent_outputs": self.agent_outputs,
            "review_reports": self.review_reports,
            "quality_gate": self.quality_gate,
            "errors": self.errors,
            "trace": self.trace,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
