# coding=utf-8
"""Simple in-memory message bus for agent coordination."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MessageBus:
    """Send/inbox/broadcast helper for AgentRuntime."""

    def __init__(self) -> None:
        self._messages: list[dict[str, Any]] = []

    def send(
        self,
        *,
        sender: str,
        recipient: str,
        content: str,
        message_type: str = "info",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        msg = {
            "message_id": str(uuid4()),
            "sender": sender,
            "recipient": recipient,
            "content": str(content)[:1000],
            "message_type": message_type,
            "metadata": metadata or {},
            "ts": _now_iso(),
        }
        self._messages.append(msg)
        return dict(msg)

    def inbox(self, recipient: str) -> list[dict[str, Any]]:
        return [dict(m) for m in self._messages if m.get("recipient") == recipient]

    def broadcast(
        self,
        *,
        sender: str,
        recipients: list[str],
        content: str,
        message_type: str = "broadcast",
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        sent: list[dict[str, Any]] = []
        for recipient in recipients:
            sent.append(
                self.send(
                    sender=sender,
                    recipient=recipient,
                    content=content,
                    message_type=message_type,
                    metadata=metadata,
                )
            )
        return sent

    def list_messages(self) -> list[dict[str, Any]]:
        return [dict(m) for m in self._messages]

