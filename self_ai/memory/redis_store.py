# coding=utf-8
"""Redis-backed runtime state persistence with fail-safe behavior."""

import json
from typing import Any


class RedisStore:
    """Redis runtime store with dependency injection and tolerant I/O."""

    def __init__(
        self,
        client: Any,
        *,
        key_prefix: str = "cf",
        default_ttl_seconds: int = 86400,
    ) -> None:
        self.client = client
        self.key_prefix = key_prefix
        self.default_ttl_seconds = default_ttl_seconds

    def _k_run_state(self, run_id: str) -> str:
        return f"{self.key_prefix}:run:{run_id}:state"

    def _k_run_trace(self, run_id: str) -> str:
        return f"{self.key_prefix}:run:{run_id}:trace"

    def _k_node_output(self, run_id: str, node_name: str) -> str:
        return f"{self.key_prefix}:run:{run_id}:node:{node_name}:output"

    def _k_agent_output(self, run_id: str, agent_name: str) -> str:
        return f"{self.key_prefix}:run:{run_id}:agent:{agent_name}:output"

    def _k_review(self, run_id: str, iteration: int) -> str:
        return f"{self.key_prefix}:run:{run_id}:review:{iteration}"

    def _k_errors(self, run_id: str) -> str:
        return f"{self.key_prefix}:run:{run_id}:errors"

    def _k_session_messages(self, session_id: str) -> str:
        return f"{self.key_prefix}:session:{session_id}:messages"

    def _k_model_cache(self, request_hash: str) -> str:
        return f"{self.key_prefix}:model_cache:{request_hash}"

    @staticmethod
    def _dump(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, default=str)

    @staticmethod
    def _to_text(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="strict")
        return str(value)

    def _apply_ttl(self, key: str, ttl: int | None) -> None:
        applied_ttl = self.default_ttl_seconds if ttl is None else ttl
        if applied_ttl and applied_ttl > 0:
            self.client.expire(key, applied_ttl)

    def _parse_dict(self, raw: Any) -> dict[str, Any] | None:
        if raw is None:
            return None
        try:
            text = self._to_text(raw)
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
            return None
        except Exception:
            return None

    def _parse_list(self, raw_items: Any) -> list[dict[str, Any]]:
        if not raw_items:
            return []
        results: list[dict[str, Any]] = []
        for raw in raw_items:
            item = self._parse_dict(raw)
            if item is not None:
                results.append(item)
        return results

    def save_run_state(
        self,
        run_id: str,
        state: dict[str, Any],
        ttl: int | None = None,
    ) -> None:
        try:
            key = self._k_run_state(run_id)
            self.client.set(key, self._dump(state))
            self._apply_ttl(key, ttl)
        except Exception:
            return

    def load_run_state(self, run_id: str) -> dict[str, Any] | None:
        try:
            key = self._k_run_state(run_id)
            raw = self.client.get(key)
            return self._parse_dict(raw)
        except Exception:
            return None

    def append_trace(
        self,
        run_id: str,
        event: dict[str, Any],
        ttl: int | None = None,
    ) -> None:
        try:
            key = self._k_run_trace(run_id)
            self.client.rpush(key, self._dump(event))
            self._apply_ttl(key, ttl)
        except Exception:
            return

    def list_trace(self, run_id: str, limit: int = 200) -> list[dict[str, Any]]:
        try:
            key = self._k_run_trace(run_id)
            raw_items = self.client.lrange(key, 0, -1)
            parsed = self._parse_list(raw_items)
            if limit <= 0:
                return parsed
            return parsed[-limit:]
        except Exception:
            return []

    def save_node_output(
        self,
        run_id: str,
        node_name: str,
        output: dict[str, Any],
        ttl: int | None = None,
    ) -> None:
        try:
            key = self._k_node_output(run_id, node_name)
            self.client.set(key, self._dump(output))
            self._apply_ttl(key, ttl)
        except Exception:
            return

    def get_node_output(self, run_id: str, node_name: str) -> dict[str, Any] | None:
        try:
            key = self._k_node_output(run_id, node_name)
            raw = self.client.get(key)
            return self._parse_dict(raw)
        except Exception:
            return None

    def save_agent_output(
        self,
        run_id: str,
        agent_name: str,
        output: dict[str, Any],
        ttl: int | None = None,
    ) -> None:
        try:
            key = self._k_agent_output(run_id, agent_name)
            self.client.set(key, self._dump(output))
            self._apply_ttl(key, ttl)
        except Exception:
            return

    def get_agent_output(self, run_id: str, agent_name: str) -> dict[str, Any] | None:
        try:
            key = self._k_agent_output(run_id, agent_name)
            raw = self.client.get(key)
            return self._parse_dict(raw)
        except Exception:
            return None

    def append_error(
        self,
        run_id: str,
        error: dict[str, Any],
        ttl: int | None = None,
    ) -> None:
        try:
            key = self._k_errors(run_id)
            self.client.rpush(key, self._dump(error))
            self._apply_ttl(key, ttl)
        except Exception:
            return

    def list_errors(self, run_id: str, limit: int = 100) -> list[dict[str, Any]]:
        try:
            key = self._k_errors(run_id)
            raw_items = self.client.lrange(key, 0, -1)
            parsed = self._parse_list(raw_items)
            if limit <= 0:
                return parsed
            return parsed[-limit:]
        except Exception:
            return []

    def append_session_message(
        self,
        session_id: str,
        message: dict[str, Any],
        ttl: int | None = 604800,
    ) -> None:
        try:
            key = self._k_session_messages(session_id)
            self.client.rpush(key, self._dump(message))
            self._apply_ttl(key, ttl)
        except Exception:
            return

    def list_session_messages(
        self,
        session_id: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        try:
            key = self._k_session_messages(session_id)
            raw_items = self.client.lrange(key, 0, -1)
            parsed = self._parse_list(raw_items)
            if limit <= 0:
                return parsed
            return parsed[-limit:]
        except Exception:
            return []


class NullRedisStore:
    """No-op runtime store used when Redis is disabled or unavailable."""

    def save_run_state(self, run_id: str, state: dict[str, Any], ttl: int | None = None) -> None:
        return

    def load_run_state(self, run_id: str) -> dict[str, Any] | None:
        return None

    def append_trace(self, run_id: str, event: dict[str, Any], ttl: int | None = None) -> None:
        return

    def list_trace(self, run_id: str, limit: int = 200) -> list[dict[str, Any]]:
        return []

    def save_node_output(
        self, run_id: str, node_name: str, output: dict[str, Any], ttl: int | None = None
    ) -> None:
        return

    def get_node_output(self, run_id: str, node_name: str) -> dict[str, Any] | None:
        return None

    def save_agent_output(
        self, run_id: str, agent_name: str, output: dict[str, Any], ttl: int | None = None
    ) -> None:
        return

    def get_agent_output(self, run_id: str, agent_name: str) -> dict[str, Any] | None:
        return None

    def append_error(self, run_id: str, error: dict[str, Any], ttl: int | None = None) -> None:
        return

    def list_errors(self, run_id: str, limit: int = 100) -> list[dict[str, Any]]:
        return []

    def append_session_message(
        self,
        session_id: str,
        message: dict[str, Any],
        ttl: int | None = 604800,
    ) -> None:
        return

    def list_session_messages(self, session_id: str, limit: int = 50) -> list[dict[str, Any]]:
        return []
