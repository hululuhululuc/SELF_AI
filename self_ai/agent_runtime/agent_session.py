# coding=utf-8
"""Agent runtime session model."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AgentSession(BaseModel):
    """State container for role-agent execution and controlled revision."""

    session_id: str = "default"
    run_id: str = ""
    task: str = ""
    task_profile: dict[str, Any] = Field(default_factory=dict)
    workflow_decision: dict[str, Any] = Field(default_factory=dict)
    plan: dict[str, Any] = Field(default_factory=dict)
    context_pack: dict[str, Any] = Field(default_factory=dict)

    selected_agents: list[str] = Field(default_factory=list)
    agent_outputs: dict[str, Any] = Field(default_factory=dict)
    review_reports: list[dict[str, Any]] = Field(default_factory=list)
    quality_gate: dict[str, Any] = Field(default_factory=dict)

    revision_count: int = 0
    max_revision_iterations: int = 1
    current_target_agent: str | None = None

    messages: list[dict[str, Any]] = Field(default_factory=list)
    tasks: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    created_at: str = Field(default_factory=_now_iso)
    updated_at: str = Field(default_factory=_now_iso)

    @classmethod
    def from_engine_state(cls, state: Any) -> "AgentSession":
        raw: dict[str, Any]
        if isinstance(state, dict):
            raw = state
        elif hasattr(state, "model_dump"):
            raw = state.model_dump(mode="json")
        else:
            raw = {}

        now = _now_iso()
        return cls(
            session_id=str(raw.get("session_id", "default")),
            run_id=str(raw.get("run_id", "")),
            task=str(raw.get("normalized_task") or raw.get("task") or ""),
            task_profile=raw.get("task_profile", {}) if isinstance(raw.get("task_profile"), dict) else {},
            workflow_decision=raw.get("workflow_decision", {})
            if isinstance(raw.get("workflow_decision"), dict)
            else {},
            plan=raw.get("plan", {}) if isinstance(raw.get("plan"), dict) else {},
            context_pack=raw.get("context_pack", {}) if isinstance(raw.get("context_pack"), dict) else {},
            selected_agents=list(raw.get("selected_agents", []))
            if isinstance(raw.get("selected_agents"), list)
            else [],
            agent_outputs=raw.get("agent_outputs", {}) if isinstance(raw.get("agent_outputs"), dict) else {},
            review_reports=list(raw.get("review_reports", []))
            if isinstance(raw.get("review_reports"), list)
            else [],
            quality_gate=raw.get("quality_gate", {}) if isinstance(raw.get("quality_gate"), dict) else {},
            errors=list(raw.get("errors", [])) if isinstance(raw.get("errors"), list) else [],
            metadata=raw.get("metadata", {}) if isinstance(raw.get("metadata"), dict) else {},
            created_at=str(raw.get("created_at") or now),
            updated_at=str(raw.get("updated_at") or now),
        )

    def touch(self) -> None:
        self.updated_at = _now_iso()

    def append_message(
        self,
        *,
        sender: str,
        recipient: str,
        content: str,
        message_type: str = "info",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.messages.append(
            {
                "sender": sender,
                "recipient": recipient,
                "content": str(content)[:1000],
                "message_type": message_type,
                "metadata": metadata or {},
                "ts": _now_iso(),
            }
        )
        self.touch()

    def add_agent_output(self, agent_name: str, output: dict[str, Any]) -> None:
        self.agent_outputs[str(agent_name)] = output
        self.touch()

    def add_review_report(self, report: dict[str, Any]) -> None:
        self.review_reports.append(report)
        self.touch()

    def set_quality_gate(self, gate: dict[str, Any]) -> None:
        self.quality_gate = gate
        self.touch()

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
        self.touch()

    def to_engine_state_updates(self) -> dict[str, Any]:
        return {
            "selected_agents": list(self.selected_agents),
            "agent_outputs": dict(self.agent_outputs),
            "review_reports": list(self.review_reports),
            "quality_gate": dict(self.quality_gate),
            "errors": list(self.errors),
            "metadata": {
                **self.metadata,
                "revision_count": self.revision_count,
                "max_revision_iterations": self.max_revision_iterations,
                "revision_performed": self.revision_count > 0,
                "revision_target_agent": self.current_target_agent,
            },
            "updated_at": self.updated_at,
        }

