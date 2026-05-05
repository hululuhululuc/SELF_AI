# coding=utf-8
"""Simple in-memory task board for AgentRuntime."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


ALLOWED_TASK_STATUS = {"pending", "claimed", "running", "done", "blocked", "failed"}


class TaskBoard:
    """Task tracker for agent execution and revisions."""

    def __init__(self) -> None:
        self._tasks: dict[str, dict[str, Any]] = {}

    def create_task(
        self,
        *,
        goal: str,
        agent_name: str,
        status: str = "pending",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized = status if status in ALLOWED_TASK_STATUS else "pending"
        task_id = str(uuid4())
        task = {
            "task_id": task_id,
            "goal": str(goal),
            "agent_name": str(agent_name),
            "status": normalized,
            "metadata": metadata or {},
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
        }
        self._tasks[task_id] = task
        return dict(task)

    def claim_task(self, task_id: str, *, claimer: str | None = None) -> dict[str, Any] | None:
        task = self._tasks.get(task_id)
        if task is None:
            return None
        if task["status"] not in {"pending", "blocked"}:
            return dict(task)
        task["status"] = "claimed"
        if claimer:
            task["metadata"]["claimer"] = claimer
        task["updated_at"] = _now_iso()
        return dict(task)

    def update_task(
        self,
        task_id: str,
        *,
        status: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        task = self._tasks.get(task_id)
        if task is None:
            return None
        if status is not None and status in ALLOWED_TASK_STATUS:
            task["status"] = status
        if metadata:
            task["metadata"].update(metadata)
        task["updated_at"] = _now_iso()
        return dict(task)

    def list_tasks(self, *, status: str | None = None) -> list[dict[str, Any]]:
        tasks = list(self._tasks.values())
        if status is not None:
            tasks = [t for t in tasks if t.get("status") == status]
        return [dict(item) for item in tasks]

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        task = self._tasks.get(task_id)
        return dict(task) if task is not None else None

