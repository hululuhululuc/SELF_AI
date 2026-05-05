# coding=utf-8
"""Unit tests for RedisStore with fake Redis client."""

from typing import Any

from self_ai.memory.redis_store import NullRedisStore, RedisStore


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, Any] = {}
        self.lists: dict[str, list[Any]] = {}
        self.expire_calls: list[tuple[str, int]] = []
        self.raise_ops: set[str] = set()

    def _maybe_raise(self, op: str) -> None:
        if op in self.raise_ops:
            raise RuntimeError(f"boom:{op}")

    def set(self, key: str, value: Any) -> None:
        self._maybe_raise("set")
        self.values[key] = value

    def get(self, key: str) -> Any:
        self._maybe_raise("get")
        return self.values.get(key)

    def rpush(self, key: str, value: Any) -> None:
        self._maybe_raise("rpush")
        self.lists.setdefault(key, []).append(value)

    def lrange(self, key: str, start: int, end: int) -> list[Any]:
        self._maybe_raise("lrange")
        arr = self.lists.get(key, [])
        if end == -1:
            end = len(arr) - 1
        if end < start:
            return []
        return arr[start : end + 1]

    def expire(self, key: str, ttl: int) -> None:
        self._maybe_raise("expire")
        self.expire_calls.append((key, ttl))


def test_save_and_load_run_state() -> None:
    fake = FakeRedis()
    store = RedisStore(fake, key_prefix="cf", default_ttl_seconds=123)
    store.save_run_state("r1", {"run_id": "r1", "task": "中文任务"})
    loaded = store.load_run_state("r1")

    assert loaded is not None
    assert loaded["run_id"] == "r1"
    assert "中文任务" in fake.values["cf:run:r1:state"]
    assert ("cf:run:r1:state", 123) in fake.expire_calls


def test_append_and_list_trace() -> None:
    fake = FakeRedis()
    store = RedisStore(fake)
    store.append_trace("r1", {"event": "a"})
    store.append_trace("r1", {"event": "b"})
    traces = store.list_trace("r1")
    assert [x["event"] for x in traces] == ["a", "b"]


def test_save_get_node_and_agent_output() -> None:
    fake = FakeRedis()
    store = RedisStore(fake)
    store.save_node_output("r1", "research", {"count": 2})
    store.save_agent_output("r1", "coder", {"content": "ok"})
    assert store.get_node_output("r1", "research") == {"count": 2}
    assert store.get_agent_output("r1", "coder") == {"content": "ok"}


def test_append_list_errors_and_session_messages() -> None:
    fake = FakeRedis()
    store = RedisStore(fake)
    store.append_error("r1", {"error": "ValueError"})
    store.append_session_message("s1", {"role": "user", "content": "hi"})
    assert store.list_errors("r1")[0]["error"] == "ValueError"
    assert store.list_session_messages("s1")[0]["role"] == "user"


def test_bytes_decode_and_missing_key_behavior() -> None:
    fake = FakeRedis()
    store = RedisStore(fake)
    fake.values["cf:run:r2:state"] = b'{"run_id":"r2"}'
    fake.lists["cf:run:r2:trace"] = [b'{"event":"x"}']
    assert store.load_run_state("r2") == {"run_id": "r2"}
    assert store.list_trace("r2") == [{"event": "x"}]
    assert store.load_run_state("missing") is None
    assert store.list_errors("missing") == []


def test_malformed_json_tolerant_reads() -> None:
    fake = FakeRedis()
    store = RedisStore(fake)
    fake.values["cf:run:r3:state"] = "{bad-json"
    fake.lists["cf:run:r3:trace"] = ["{bad-json", '{"event":"ok"}']
    assert store.load_run_state("r3") is None
    assert store.list_trace("r3") == [{"event": "ok"}]


def test_redis_exceptions_do_not_raise() -> None:
    fake = FakeRedis()
    store = RedisStore(fake)
    fake.raise_ops.update({"set", "get", "rpush", "lrange", "expire"})

    store.save_run_state("r1", {"x": 1})
    assert store.load_run_state("r1") is None
    store.append_trace("r1", {"event": "x"})
    assert store.list_trace("r1") == []
    store.save_node_output("r1", "assign", {"a": 1})
    assert store.get_node_output("r1", "assign") is None
    store.save_agent_output("r1", "coder", {"a": 1})
    assert store.get_agent_output("r1", "coder") is None
    store.append_error("r1", {"e": 1})
    assert store.list_errors("r1") == []
    store.append_session_message("s1", {"m": 1})
    assert store.list_session_messages("s1") == []


def test_ttl_override_is_used() -> None:
    fake = FakeRedis()
    store = RedisStore(fake, default_ttl_seconds=100)
    store.save_run_state("r1", {"a": 1}, ttl=9)
    assert ("cf:run:r1:state", 9) in fake.expire_calls


def test_null_redis_store_noop_methods() -> None:
    null_store = NullRedisStore()
    null_store.save_run_state("r", {"a": 1})
    null_store.append_trace("r", {"e": 1})
    null_store.save_node_output("r", "n", {"o": 1})
    null_store.save_agent_output("r", "a", {"o": 1})
    null_store.append_error("r", {"e": 1})
    null_store.append_session_message("s", {"m": 1})
    assert null_store.load_run_state("r") is None
    assert null_store.list_trace("r") == []
    assert null_store.get_node_output("r", "n") is None
    assert null_store.get_agent_output("r", "a") is None
    assert null_store.list_errors("r") == []
    assert null_store.list_session_messages("s") == []
