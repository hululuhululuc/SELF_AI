# coding=utf-8
"""Production workflow entrypoint backed by SelfAIKernel + EngineLoop."""

from __future__ import annotations

import asyncio
import json
import locale
import os
from pathlib import Path
import sys
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from redis import Redis

from .config import resolve_project_path, settings
from .kernel import SelfAIKernel
from .memory.chat_memory_service import ChatMemoryService
from .memory.chat_memory_store import ChatMemoryStore
from .memory.memory_contract import allowed_memory_collections
from .memory.memory_sidecar import MemorySidecarAgent
from .memory.redis_store import NullRedisStore, RedisStore
from .observability import trace
from .router import route_model
from .text_utils import is_probably_garbled, normalize_text, shorten_text

_kernel: SelfAIKernel | None = None
_utf8_configured = False
_chat_memory_store: ChatMemoryStore | None = None
_chat_memory_service: ChatMemoryService | None = None
_memory_sidecar_agent: MemorySidecarAgent | None = None
_memory_vector_encoder: Any | None = None


def _ensure_utf8_runtime() -> None:
    global _utf8_configured
    if _utf8_configured:
        return
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    encoding_before = {
        "stdout": str(getattr(sys.stdout, "encoding", "") or ""),
        "stderr": str(getattr(sys.stderr, "encoding", "") or ""),
        "preferred": str(locale.getpreferredencoding(False) or ""),
    }
    codepage: int | None = None
    if os.name == "nt":
        try:
            import ctypes

            codepage = int(ctypes.windll.kernel32.GetConsoleOutputCP())  # type: ignore[attr-defined]
        except Exception:
            codepage = None
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
    encoding_after = {
        "stdout": str(getattr(sys.stdout, "encoding", "") or ""),
        "stderr": str(getattr(sys.stderr, "encoding", "") or ""),
        "preferred": str(locale.getpreferredencoding(False) or ""),
    }
    trace(
        "runtime.encoding.health",
        platform=os.name,
        codepage=codepage,
        pythonutf8=os.getenv("PYTHONUTF8", ""),
        pyioencoding=os.getenv("PYTHONIOENCODING", ""),
        before=encoding_before,
        after=encoding_after,
    )
    _utf8_configured = True


def _parse_shell_whitelist(raw: str) -> tuple[tuple[str, ...], ...]:
    entries: list[tuple[str, ...]] = []
    for item in (raw or "").split(";"):
        text = item.strip()
        if not text:
            continue
        parts = tuple(token for token in text.split(" ") if token)
        if parts:
            entries.append(parts)
    return tuple(entries)


def _parse_shell_denylist(raw: str) -> set[str]:
    values: set[str] = set()
    for item in (raw or "").split(","):
        token = item.strip().lower()
        if token:
            values.add(token)
    return values


def _build_dependencies() -> dict[str, Any]:
    redis_store: RedisStore | NullRedisStore
    if settings.redis_enabled:
        redis_store = RedisStore(
            Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
            ),
            key_prefix=settings.redis_key_prefix,
            default_ttl_seconds=settings.redis_default_ttl_seconds,
        )
    else:
        redis_store = NullRedisStore()

    qdrant_memory = None
    graph_memory = None
    web_search_func = None
    try:
        from .tools import graph_memory as imported_graph_memory
        from .tools import qdrant_memory as imported_qdrant_memory
        from .tools import tavily as imported_tavily

        qdrant_memory = imported_qdrant_memory
        graph_memory = imported_graph_memory
        web_search_func = lambda query, max_results: imported_tavily.search(  # noqa: E731
            query=query,
            max_results=max_results,
        )
    except Exception:
        # Safe fallback for environments without external services.
        web_search_func = lambda _query, _max_results: []  # noqa: E731

    # Phase 3 write path: graph write must not be blocked by graph-read feature flag.
    try:
        is_null_graph = graph_memory is None or graph_memory.__class__.__name__.lower().startswith("nullneo4j")
        if is_null_graph and settings.neo4j_enabled:
            from neo4j import GraphDatabase

            from .memory.neo4j_store import Neo4jGraphMemory

            driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
            )
            graph_memory = Neo4jGraphMemory(
                driver=driver,
                database=settings.neo4j_database,
                enabled=True,
            )
    except Exception:
        pass

    return {
        "route_model_func": route_model,
        "redis_store": redis_store,
        "qdrant_memory": qdrant_memory,
        "graph_memory": graph_memory,
        "web_search_func": web_search_func,
        "project_root": str(Path(__file__).resolve().parent),
        "shell_whitelist": _parse_shell_whitelist(settings.shell_allowlist),
        "shell_denylist": _parse_shell_denylist(settings.shell_denylist),
        "shell_enabled": bool(settings.shell_enabled),
    }


def _build_permission_profile() -> dict[str, bool]:
    mode = str(settings.permission_mode or "dev_write").strip().lower()
    profile: dict[str, bool] = {
        "read_only": True,
        "runtime_write": True,
        "memory_read": True,
        "memory_write": False,
        "workspace_read": True,
        "workspace_write": False,
        "shell_exec": False,
        "network": False,
        "model_call": True,
        "shell_enabled": bool(settings.shell_enabled),
    }
    if mode == "dev_write":
        profile["workspace_write"] = True
        profile["memory_write"] = True
        profile["network"] = True
        profile["shell_exec"] = bool(settings.shell_enabled)
    elif mode == "dev_full":
        profile["workspace_write"] = True
        profile["shell_exec"] = bool(settings.shell_enabled)
        profile["network"] = True
        profile["memory_write"] = True
    return profile


def get_kernel() -> SelfAIKernel:
    global _kernel
    if _kernel is None:
        _ensure_utf8_runtime()
        _kernel = SelfAIKernel(
            project_root=Path(__file__).resolve().parent,
            settings=settings,
            permission_profile=_build_permission_profile(),
            dependencies=_build_dependencies(),
        )
    return _kernel


def get_chat_memory_store() -> ChatMemoryStore | None:
    global _chat_memory_store
    if not settings.chat_memory_enabled:
        return None
    if _chat_memory_store is None:
        _chat_memory_store = ChatMemoryStore(
            root_dir=resolve_project_path(settings.chat_memory_root),
            max_shard_bytes=settings.chat_memory_max_shard_bytes,
            max_turns_per_shard=settings.chat_memory_max_turns_per_shard,
        )
    return _chat_memory_store


def get_chat_memory_service() -> ChatMemoryService | None:
    global _chat_memory_service
    store = get_chat_memory_store()
    if store is None:
        return None
    if _chat_memory_service is None:
        _chat_memory_service = ChatMemoryService(store)
    return _chat_memory_service


def get_memory_sidecar_agent() -> MemorySidecarAgent:
    global _memory_sidecar_agent
    if _memory_sidecar_agent is None:
        _memory_sidecar_agent = MemorySidecarAgent()
    return _memory_sidecar_agent


def _json_preview(value: Any, *, max_chars: int = 1200) -> str:
    text = str(value or "")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...(truncated)"


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = str(text or "").strip()
    if not stripped:
        return {}
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(stripped[start : end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _render_global_constraint_list(items: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for idx, row in enumerate(items[:24], start=1):
        if not isinstance(row, dict):
            continue
        content = shorten_text(str(row.get("content", "") or "").strip(), max_chars=260)
        if content:
            lines.append(f"{idx}. [{str(row.get('id', '') or '')}] {content}")
    return "\n".join(lines) if lines else "(none)"


def _load_session_global_constraints(
    *,
    chat_id: str,
    chat_memory_service: Any | None,
    chat_memory_store: Any | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    status: dict[str, Any] = {
        "attempted": False,
        "succeeded": True,
        "source": "none",
        "count": 0,
        "error_type": "",
        "error_message": "",
    }

    loader = getattr(chat_memory_service, "load_global_constraints", None)
    source = "service"
    if not callable(loader):
        loader = getattr(chat_memory_store, "read_global_constraints", None)
        source = "store"

    if not callable(loader):
        return [], status

    status["attempted"] = True
    status["source"] = source
    try:
        payload = loader(chat_id=chat_id)
    except Exception as exc:
        status.update(
            {
                "succeeded": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:240],
            }
        )
        trace(
            "chat_memory.global_constraints.read_error",
            chat_id=chat_id,
            source=source,
            error=type(exc).__name__,
            message=str(exc)[:220],
        )
        return [], status

    if not isinstance(payload, dict):
        status.update(
            {
                "succeeded": False,
                "error_type": "InvalidGlobalConstraintsPayload",
                "error_message": "global constraints loader returned non-dict payload",
            }
        )
        return [], status

    raw_items = payload.get("items", [])
    if not isinstance(raw_items, list):
        status.update(
            {
                "succeeded": False,
                "error_type": "InvalidGlobalConstraintsItems",
                "error_message": "global constraints payload items must be a list",
            }
        )
        return [], status

    items = [x for x in raw_items if isinstance(x, dict)]
    status["count"] = len(items)
    return items, status


def _resolve_memory_write_policy(
    result: dict[str, Any],
    *,
    committed_memories: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    default_collection = str(
        getattr(settings, "chat_memory_l2_collection_default", "cf_chat_memory") or "cf_chat_memory"
    )
    profile = result.get("task_profile", {})
    if not isinstance(profile, dict):
        profile = {}
    policy = profile.get("memory_write_policy", {})
    if not isinstance(policy, dict):
        policy = {}
    def _is_issue_like_memory(item: dict[str, Any]) -> bool:
        memory_type = str(item.get("memory_type", "") or "").strip().lower()
        memory_class = str(item.get("memory_class", "") or "").strip().lower()
        if memory_type == "issue":
            return True
        return any(token in memory_class for token in ("issue", "bug", "error", "failure"))

    committed_memories = committed_memories or []
    has_issue_memory = any(
        isinstance(item, dict)
        and _is_issue_like_memory(item)
        for item in committed_memories
    )
    fallback_write_qdrant = bool(committed_memories)
    fallback_write_neo4j = has_issue_memory
    fallback_collection = "cf_review_memory" if has_issue_memory else default_collection
    return {
        "write_qdrant": bool(policy.get("write_qdrant", fallback_write_qdrant)),
        "write_neo4j": bool(policy.get("write_neo4j", fallback_write_neo4j)),
        "qdrant_collection": str(policy.get("qdrant_collection", "") or fallback_collection),
        "neo4j_label": str(policy.get("neo4j_label", "") or ("Issue" if has_issue_memory else "Task")),
        "reason": str(policy.get("reason", "") or ""),
    }


def _get_memory_vector_encode_func() -> Any | None:
    global _memory_vector_encoder
    if _memory_vector_encoder is not None:
        return _memory_vector_encoder
    try:
        from .tools import _encode_text as encode_text  # type: ignore

        _memory_vector_encoder = encode_text
    except Exception:
        _memory_vector_encoder = None
    return _memory_vector_encoder


def _build_memory_evidence(
    *,
    memory_item: dict[str, Any],
    chat_id: str,
    run_id: str,
    turn_id: int,
) -> dict[str, Any]:
    raw_content = str(
        memory_item.get("content_preview", "")
        or memory_item.get("retrieval_summary", "")
        or ""
    ).strip()
    memory_type = str(memory_item.get("memory_type", "turn_summary") or "turn_summary")
    memory_class = str(memory_item.get("memory_class", "fact") or "fact")
    memory_id = str(memory_item.get("memory_id", "") or f"{chat_id}-turn-{turn_id}-{memory_type}")
    timestamp = str(memory_item.get("timestamp", "") or "")
    importance = float(memory_item.get("importance", 0.5) or 0.5)
    confidence = float(memory_item.get("confidence", 0.7) or 0.7)
    refs = memory_item.get("refs", {}) if isinstance(memory_item.get("refs"), dict) else {}
    files: list[str] = []
    entities: list[str] = []
    for key in ("path", "file", "target_path"):
        value = refs.get(key)
        text = str(value or "").strip()
        if text and text not in files:
            files.append(text)
    for key in ("error_type", "stage", "quality_gate_decision"):
        value = refs.get(key)
        text = str(value or "").strip()
        if text and text not in entities:
            entities.append(f"{key}:{text}")

    summary_parts = [
        f"[{memory_class}/{memory_type}]",
        raw_content,
    ]
    if files:
        summary_parts.append(f"files={','.join(files[:3])}")
    if entities:
        summary_parts.append(f"signals={','.join(entities[:4])}")
    content = shorten_text(" | ".join(x for x in summary_parts if x), max_chars=420)

    metadata = {
        "chat_id": chat_id,
        "run_id": run_id,
        "turn_id": turn_id,
        "memory_type": memory_type,
        "memory_class": memory_class,
        "target_collection": str(memory_item.get("target_collection", "") or ""),
        "importance": max(0.0, min(importance, 1.0)),
        "decay_mode": str(memory_item.get("decay_mode", "normal") or "normal"),
        "pinned": bool(memory_item.get("pinned", False)),
        "confidence": max(0.0, min(confidence, 1.0)),
        "decision_source": str(memory_item.get("decision_source", "system_default_v1") or "system_default_v1"),
        "rationale": str(memory_item.get("rationale", "") or "")[:240],
        "refs": refs,
        "entities": entities[:8],
        "files": files[:8],
    }
    return {
        "content": content,
        "source_type": f"chat_memory:{memory_type}",
        "source": f"chat:{chat_id}",
        "chunk_id": memory_id,
        "score": metadata["importance"],
        "token_count": max(0, len(content) // 4),
        "created_at": timestamp,
        "metadata": metadata,
    }


def _memory_point_id(memory_id: str) -> str:
    text = str(memory_id or "").strip()
    if not text:
        text = "memory:unknown"
    return str(uuid5(NAMESPACE_URL, text))


async def _persist_committed_memories_l2_l3(
    *,
    kernel: SelfAIKernel,
    result: dict[str, Any],
    session_id: str,
    chat_id: str,
    memory_write: dict[str, Any] | None = None,
) -> dict[str, Any]:
    def _is_issue_like_memory(item: dict[str, Any]) -> bool:
        memory_type = str(item.get("memory_type", "") or "").strip().lower()
        memory_class = str(item.get("memory_class", "") or "").strip().lower()
        if memory_type == "issue":
            return True
        return any(token in memory_class for token in ("issue", "bug", "error", "failure"))

    default_collection = str(
        getattr(settings, "chat_memory_l2_collection_default", "cf_chat_memory") or "cf_chat_memory"
    )
    default_issue_collection = "cf_review_memory"
    memory_write = memory_write or {}
    strict_required = bool(memory_write.get("strict_required", True))
    out: dict[str, Any] = {
        "enabled": True,
        "strict_required": strict_required,
        "failed": False,
        "write_qdrant": bool(memory_write.get("write_qdrant", False)),
        "write_neo4j": bool(memory_write.get("write_neo4j", False)),
        "qdrant_collection": str(memory_write.get("qdrant_collection", "") or default_collection),
        "neo4j_label": str(memory_write.get("neo4j_label", "") or "Issue"),
        "committed_count": 0,
        "qdrant_attempted": 0,
        "qdrant_ok": 0,
        "qdrant_failed": 0,
        "neo4j_attempted": 0,
        "neo4j_ok": 0,
        "neo4j_failed": 0,
        "errors": [],
        "qdrant_collection_counts": {},
    }
    metadata = result.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    chat_memory = metadata.get("chat_memory", {})
    if not isinstance(chat_memory, dict):
        return out
    committed = chat_memory.get("committed_memories", [])
    if not isinstance(committed, list):
        committed = []
    if not memory_write:
        memory_write = _resolve_memory_write_policy(
            result,
            committed_memories=[x for x in committed if isinstance(x, dict)],
        )
        out["write_qdrant"] = bool(memory_write.get("write_qdrant", False))
        out["write_neo4j"] = bool(memory_write.get("write_neo4j", False))
        out["qdrant_collection"] = str(memory_write.get("qdrant_collection", "") or out["qdrant_collection"])
        out["neo4j_label"] = str(memory_write.get("neo4j_label", "") or out["neo4j_label"])
    out["committed_count"] = len(committed)
    if not committed:
        committed = []
    if not committed:
        return out

    run_id = str(result.get("run_id", "") or "")
    turn_id = int(chat_memory.get("turn_id", 0) or 0)

    encode_func = _get_memory_vector_encode_func()
    can_write_qdrant = bool(memory_write.get("write_qdrant", False)) and callable(encode_func)
    if bool(memory_write.get("write_qdrant", False)) and not callable(encode_func):
        out["errors"].append({"stage": "qdrant", "type": "EmbedderUnavailable", "message": "memory embedder unavailable"})

    allowed_collections = allowed_memory_collections(chat_collection=default_collection)

    def _qdrant_collection_for_item(item: dict[str, Any]) -> tuple[str | None, str | None]:
        requested = str(item.get("target_collection", "") or "").strip()
        if not requested:
            requested = (
                default_issue_collection
                if _is_issue_like_memory(item)
                else str(memory_write.get("qdrant_collection", "") or "").strip() or default_collection
            )
        if requested not in allowed_collections:
            return None, requested
        return requested, None

    for item in committed:
        if not isinstance(item, dict):
            continue
        evidence = _build_memory_evidence(
            memory_item=item,
            chat_id=chat_id,
            run_id=run_id,
            turn_id=turn_id,
        )
        content = str(evidence.get("content", "") or "").strip()
        if not content:
            continue

        if can_write_qdrant:
            collection_name, invalid_collection = _qdrant_collection_for_item(item)
            out["qdrant_attempted"] += 1
            if not collection_name:
                out["qdrant_failed"] += 1
                out["errors"].append(
                    {
                        "stage": "qdrant",
                        "type": "MemoryContractValidationError",
                        "message": "invalid target_collection from memory decision",
                        "collection": str(invalid_collection or ""),
                    }
                )
                continue
            out["qdrant_collection_counts"][collection_name] = int(
                out["qdrant_collection_counts"].get(collection_name, 0) or 0
            ) + 1
            try:
                dense_vector = encode_func(content)  # type: ignore[misc]
                tool_ret = await kernel.run_once(
                    tool_name="storage.semantic.upsert_memory",
                    arguments={
                        "collection_name": collection_name,
                        "dense_vector": dense_vector,
                        "evidence": evidence,
                        "point_id": _memory_point_id(str(evidence.get("chunk_id", ""))),
                    },
                    run_id=run_id,
                    session_id=session_id,
                    metadata={"source": "chat_memory_phase3_l2_write"},
                )
                if bool(tool_ret.get("ok", False)):
                    out["qdrant_ok"] += 1
                else:
                    out["qdrant_failed"] += 1
                    out["errors"].append(
                        {
                            "stage": "qdrant",
                            "type": str((tool_ret.get("error", {}) or {}).get("type", "ToolCallFailed")),
                            "message": str((tool_ret.get("error", {}) or {}).get("message", ""))[:240],
                            "collection": collection_name,
                        }
                    )
            except Exception as exc:
                out["qdrant_failed"] += 1
                out["errors"].append(
                    {
                        "stage": "qdrant",
                        "type": type(exc).__name__,
                        "message": str(exc)[:240],
                        "collection": collection_name,
                    }
                )

        if bool(memory_write.get("write_neo4j", False)) and _is_issue_like_memory(item):
            out["neo4j_attempted"] += 1
            try:
                tool_ret = await kernel.run_once(
                    tool_name="storage.graph.record_issue_fix",
                    arguments={
                        "run_id": run_id,
                        "issue": {
                            "title": f"chat_memory_issue:{item.get('memory_type', 'issue')}",
                            "summary": content,
                            "chat_id": chat_id,
                            "turn_id": turn_id,
                            "memory_id": str(item.get("memory_id", "")),
                        },
                        "fix": None,
                        "test": None,
                    },
                    run_id=run_id,
                    session_id=session_id,
                    metadata={"source": "chat_memory_phase3_l3_write"},
                )
                if bool(tool_ret.get("ok", False)):
                    out["neo4j_ok"] += 1
                else:
                    out["neo4j_failed"] += 1
                    out["errors"].append(
                        {
                            "stage": "neo4j",
                            "type": str((tool_ret.get("error", {}) or {}).get("type", "ToolCallFailed")),
                            "message": str((tool_ret.get("error", {}) or {}).get("message", ""))[:240],
                        }
                    )
            except Exception as exc:
                out["neo4j_failed"] += 1
                out["errors"].append(
                    {
                        "stage": "neo4j",
                        "type": type(exc).__name__,
                        "message": str(exc)[:240],
                    }
                )
    out["failed"] = bool(out["qdrant_failed"] or out["neo4j_failed"])
    return out


async def _run_memory_sidecar_decision(
    *,
    kernel: SelfAIKernel,
    session_id: str,
    task: str,
    result: dict[str, Any],
    recent_turns_payload: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    status: dict[str, Any] = {
        "attempted": False,
        "enabled": bool(settings.chat_memory_sidecar_enabled),
        "succeeded": False,
        "timeout": False,
        "attempt_count": 0,
        "fast_retry_used": False,
        "fallback_used": False,
        "error_type": "",
        "error_message": "",
        "decision_source": "none",
    }
    if not settings.chat_memory_sidecar_enabled:
        return None, status
    agent = get_memory_sidecar_agent()
    status["attempted"] = True
    metadata = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
    execution_state = metadata.get("execution_state", {}) if isinstance(metadata.get("execution_state"), dict) else {}
    recent_tool_results = metadata.get("recent_tool_results", [])
    if not isinstance(recent_tool_results, list):
        recent_tool_results = []
    execution_summary_full = {
        "execution_state": execution_state,
        "recent_tool_results": recent_tool_results[-6:],
        "error_count": len(result.get("errors", [])) if isinstance(result.get("errors"), list) else 0,
        "control_event_count": len(metadata.get("control_events", []))
        if isinstance(metadata.get("control_events"), list)
        else 0,
        "quality_gate": result.get("quality_gate", {}) if isinstance(result.get("quality_gate"), dict) else {},
    }
    response_full = str(result.get("response", "") or "")
    candidates_summary = [
        {
            "memory_type": "turn_summary",
            "default_decay_mode": "fast",
            "default_target_collection": str(
                getattr(settings, "chat_memory_l2_collection_default", "cf_chat_memory")
                or "cf_chat_memory"
            ),
        },
        {
            "memory_type": "response_summary",
            "default_decay_mode": "normal",
            "default_target_collection": str(
                getattr(settings, "chat_memory_l2_collection_default", "cf_chat_memory")
                or "cf_chat_memory"
            ),
        },
        {
            "memory_type": "issue",
            "default_decay_mode": "slow",
            "default_target_collection": "cf_review_memory",
        },
    ]
    recent_turns_full: list[dict[str, Any]] = []
    turns = recent_turns_payload.get("turns", []) if isinstance(recent_turns_payload, dict) else []
    if isinstance(turns, list):
        for item in turns[-4:]:
            if not isinstance(item, dict):
                continue
            recent_turns_full.append(
                {
                    "turn_id": item.get("turn_id"),
                    "retrieval_summary": _json_preview(item.get("retrieval_summary", ""), max_chars=180),
                    "memory_profile": item.get("memory_profile", {})
                    if isinstance(item.get("memory_profile"), dict)
                    else {},
                }
            )

    async def _attempt(
        *,
        timeout_s: float,
        force_tier: str,
        response_text: str,
        execution_summary: dict[str, Any],
        recent_turns_summary: list[dict[str, Any]],
        mark_fast_retry: bool,
    ) -> dict[str, Any]:
        async def _model_generate(prompt: str) -> dict[str, Any]:
            return await kernel.run_once(
                tool_name="model.generate",
                arguments={
                    "prompt": prompt,
                    "stage": "memory_sidecar",
                    "hints": {
                        "goal": "memory_profile_decision",
                        "prefer_structured_json": True,
                        "force_tier": force_tier,
                        "enable_thinking": False,
                        "max_tokens": 1024 if mark_fast_retry else 2048,
                    },
                },
                session_id=session_id,
                metadata={"source": "memory_sidecar", "attempt": "fast_retry" if mark_fast_retry else "primary"},
            )

        status["attempt_count"] = int(status.get("attempt_count", 0) or 0) + 1
        if mark_fast_retry:
            status["fast_retry_used"] = True
        return await asyncio.wait_for(
            agent.decide(
                task=task,
                response=response_text,
                execution_summary=execution_summary,
                candidates_summary=candidates_summary,
                recent_turns_summary=recent_turns_summary,
                model_generate=_model_generate,
                chat_collection=str(
                    getattr(settings, "chat_memory_l2_collection_default", "cf_chat_memory")
                    or "cf_chat_memory"
                ),
            ),
            timeout=timeout_s,
        )

    total_timeout = float(settings.chat_memory_sidecar_timeout_s)
    fast_retry_timeout = float(settings.chat_memory_sidecar_fast_retry_timeout_s)
    primary_timeout = max(1.0, total_timeout - fast_retry_timeout)
    primary_tier = "fast"

    try:
        decision = await _attempt(
            timeout_s=primary_timeout,
            force_tier=primary_tier,
            response_text=response_full,
            execution_summary=execution_summary_full,
            recent_turns_summary=recent_turns_full,
            mark_fast_retry=False,
        )
        if isinstance(decision, dict):
            status["succeeded"] = True
            status["timeout"] = False
            status["error_type"] = ""
            status["error_message"] = ""
            status["decision_source"] = str(
                decision.get("decision_source", "model_sidecar_v1") or "model_sidecar_v1"
            )
            return decision, status
        status["error_type"] = "MemorySidecarDecisionEmpty"
        status["error_message"] = "memory sidecar returned empty decision"
    except Exception as exc:
        status["error_type"] = type(exc).__name__
        status["error_message"] = str(exc)[:220]
        status["timeout"] = isinstance(exc, TimeoutError)

    if bool(settings.chat_memory_sidecar_retry_enabled):
        remaining_timeout = max(1.0, total_timeout - primary_timeout)
        retry_timeout = min(fast_retry_timeout, remaining_timeout)
        try:
            decision = await _attempt(
                timeout_s=retry_timeout,
                force_tier="fast",
                response_text=_json_preview(response_full, max_chars=260),
                execution_summary={
                    "execution_state": execution_summary_full.get("execution_state", {}),
                    "recent_tool_results": execution_summary_full.get("recent_tool_results", [])[-2:],
                    "error_count": execution_summary_full.get("error_count", 0),
                    "control_event_count": execution_summary_full.get("control_event_count", 0),
                },
                recent_turns_summary=recent_turns_full[-2:],
                mark_fast_retry=True,
            )
            if isinstance(decision, dict):
                status["succeeded"] = True
                status["timeout"] = False
                status["error_type"] = ""
                status["error_message"] = ""
                status["decision_source"] = str(
                    decision.get("decision_source", "model_sidecar_v1") or "model_sidecar_v1"
                )
                return decision, status
            status["error_type"] = "MemorySidecarDecisionEmpty"
            status["error_message"] = "memory sidecar returned empty decision"
        except Exception as exc:
            status["error_type"] = type(exc).__name__
            status["error_message"] = str(exc)[:220]
            status["timeout"] = isinstance(exc, TimeoutError) or bool(status.get("timeout", False))
    return None, status


def _constraint_maintenance_requested(result: dict[str, Any]) -> tuple[bool, str]:
    workflow = result.get("workflow_decision", {})
    if isinstance(workflow, dict):
        maintenance = workflow.get("constraint_maintenance", {})
        if isinstance(maintenance, dict):
            return bool(maintenance.get("enabled", False)), str(maintenance.get("reason", "") or "")
    metadata = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
    payload = metadata.get("last_model_payload", {})
    if isinstance(payload, dict):
        maintenance = payload.get("constraint_maintenance", {})
        if isinstance(maintenance, dict):
            return bool(maintenance.get("enabled", False)), str(maintenance.get("reason", "") or "")
    return False, ""


async def _run_global_constraint_maintainer(
    *,
    kernel: SelfAIKernel,
    session_id: str,
    task: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    response_text = str(result.get("response", "") or "").strip()
    current_items = []
    metadata = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
    raw_items = metadata.get("chat_global_constraints", [])
    if isinstance(raw_items, list):
        current_items = [x for x in raw_items if isinstance(x, dict)]
    if not task.strip():
        return {"ops": [{"op": "noop"}], "error": {"type": "EmptyTask"}}
    prompt = (
        "You are the session Global Constraint Maintainer for Self AI.\n"
        "Maintain one concise numbered list of durable session-level user preferences and constraints.\n"
        "Return JSON only. No markdown.\n\n"
        "Definitions:\n"
        "- A global constraint is a durable preference, rule, convention, or boundary for future turns in this same chat session.\n"
        "- It is not a one-off task request.\n"
        "- It should be short, clear, and directly actionable by the main model.\n"
        "- The list is session-scoped. Do not infer preferences for other chats.\n\n"
        "Current Global Constraints:\n"
        f"{_render_global_constraint_list(current_items)}\n\n"
        "Current user message:\n"
        f"{task[:1600]}\n\n"
        "Assistant response:\n"
        f"{response_text[:1600]}\n\n"
        "Rules:\n"
        "- Add a constraint only when the user clearly establishes a durable future preference or standing rule.\n"
        "- Update an existing constraint when the new message refines, replaces, or narrows it.\n"
        "- Delete a constraint only when the user clearly cancels it.\n"
        "- Keep the list concise. Avoid duplicates.\n"
        "- Prefer one clear sentence per constraint.\n"
        "- Do not include examples, explanations, or temporary task details.\n"
        "- Do not mention storage, implementation, or internal memory mechanics in constraint content.\n"
        "- Latest added or updated items should appear first after ops are applied.\n\n"
        "Output schema:\n"
        "{\n"
        '  "ops": [\n'
        '    {"op":"add|update|delete|noop","id":"existing id for update/delete","content":"required for add/update","reason":"short reason","confidence":0.0}\n'
        "  ]\n"
        "}\n"
    )
    ret = await kernel.run_once(
        tool_name="model.generate",
        arguments={
            "prompt": prompt,
            "stage": "global_constraint_maintainer",
            "hints": {
                "goal": "global_constraint_maintenance_ops_json",
                "prefer_structured_json": True,
                "force_tier": "fast",
                "enable_thinking": False,
                "max_tokens": 1024,
            },
        },
        session_id=session_id,
        metadata={"source": "global_constraint_maintainer"},
    )
    if not bool(ret.get("ok", False)):
        return {
            "ops": [{"op": "noop"}],
            "error": ret.get("error", {}) if isinstance(ret.get("error"), dict) else {"type": "ToolCallFailed"},
        }
    data = ret.get("data", {}) if isinstance(ret.get("data"), dict) else {}
    raw = str(data.get("response", "") or "")
    parsed = _extract_json_object(raw)
    ops = parsed.get("ops", [])
    if not isinstance(ops, list):
        ops = [{"op": "noop"}]
    normalized_ops: list[dict[str, Any]] = []
    for item in ops[:12]:
        if not isinstance(item, dict):
            continue
        op = str(item.get("op", "") or "").strip().lower()
        if op not in {"add", "update", "delete", "noop"}:
            continue
        row: dict[str, Any] = {
            "op": op,
            "reason": str(item.get("reason", "") or "")[:240],
        }
        if "id" in item:
            row["id"] = str(item.get("id", "") or "").strip()
        if "content" in item:
            row["content"] = str(item.get("content", "") or "").strip()[:320]
        if "confidence" in item:
            try:
                row["confidence"] = max(0.0, min(float(item.get("confidence", 0.85) or 0.85), 1.0))
            except Exception:
                row["confidence"] = 0.85
        normalized_ops.append(row)
    return {
        "ops": normalized_ops or [{"op": "noop"}],
        "raw_response_preview": raw[:500],
        "model": str(data.get("model", "") or ""),
    }


def get_chat_manifest(*, chat_id: str) -> dict[str, Any]:
    service = get_chat_memory_service()
    if service is None:
        return {"chat_id": chat_id, "enabled": False, "shards": []}
    return service.get_manifest(chat_id=chat_id)


def replay_chat_turns(
    *,
    chat_id: str,
    start_turn_id: int | None = None,
    end_turn_id: int | None = None,
    limit: int = 200,
    ascending: bool = True,
) -> list[dict[str, Any]]:
    service = get_chat_memory_service()
    if service is None:
        return []
    return service.replay_turns(
        chat_id=chat_id,
        start_turn_id=start_turn_id,
        end_turn_id=end_turn_id,
        limit=limit,
        ascending=ascending,
    )


def load_recent_chat_turns(
    *,
    chat_id: str,
    last_n: int | None = None,
) -> dict[str, Any]:
    service = get_chat_memory_service()
    if service is None:
        return {
            "chat_id": chat_id,
            "requested_turns": 0,
            "kept_turns": 0,
            "dropped_turns": 0,
            "budget": {
                "max_chars_total": 0,
                "max_item_chars": 0,
                "used_chars": 0,
            },
            "turns": [],
        }
    return service.load_recent_turns_for_injection(
        chat_id=chat_id,
        last_n=int(last_n or settings.chat_memory_recent_turns),
        max_chars_total=settings.chat_memory_injection_max_chars,
        max_item_chars=settings.chat_memory_injection_item_max_chars,
    )


def _schedule_chat_memory_compaction(
    *,
    chat_memory_service: ChatMemoryService | None,
    chat_id: str,
) -> None:
    if chat_memory_service is None or not bool(settings.chat_memory_compact_enabled):
        return
    compact_fn = getattr(chat_memory_service, "compact_chat_memory", None)
    if not callable(compact_fn):
        return

    async def _runner() -> None:
        try:
            report = await asyncio.to_thread(
                compact_fn,
                chat_id=chat_id,
                min_shards=int(settings.chat_memory_compact_min_shards),
                summary_max_chars=int(settings.chat_memory_summary_max_chars),
                max_docs=int(settings.chat_memory_compact_max_docs),
            )
            trace(
                "chat_memory.compaction",
                chat_id=chat_id,
                compacted=bool((report or {}).get("compacted", False)),
                archived_count=int((report or {}).get("archived_count", 0) or 0),
                block_count=int((report or {}).get("block_count", 0) or 0),
                reason=str((report or {}).get("reason", "") or ""),
            )
        except Exception as exc:
            trace(
                "chat_memory.compaction.error",
                chat_id=chat_id,
                error=type(exc).__name__,
                message=str(exc)[:220],
            )

    asyncio.create_task(_runner())


def _build_chat_recent_turns_stats(recent_turns_payload: dict[str, Any]) -> dict[str, Any]:
    raw_stats = recent_turns_payload.get("stats", {})
    stats = dict(raw_stats) if isinstance(raw_stats, dict) else {}
    stats.update(
        {
            "requested_turns": recent_turns_payload.get("requested_turns", 0),
            "kept_turns": recent_turns_payload.get("kept_turns", 0),
            "dropped_turns": recent_turns_payload.get("dropped_turns", 0),
            "source_type_groups": recent_turns_payload.get("source_type_groups", {}),
            "fusion_failed": bool(stats.get("fusion_failed", False)),
        }
    )
    return stats


async def run_autonomy_workflow(
    input: str,
    *,
    session_id: str = "default",
    chat_id: str | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    """Run production workflow through Kernel.run -> EngineLoop."""
    normalized = normalize_text(input)
    effective_session_id = str(chat_id or session_id or "default")
    garbled_warning = False
    if is_probably_garbled(normalized):
        # Soft-fail instead of hard-stop: many Windows terminals can transiently
        # degrade CJK to '?' while the user intent is still recoverable.
        garbled_warning = True
    kernel = get_kernel()
    chat_memory_store = get_chat_memory_store()
    chat_memory_service = get_chat_memory_service()
    recent_turns_payload = {
        "chat_id": effective_session_id,
        "requested_turns": 0,
        "kept_turns": 0,
        "dropped_turns": 0,
        "budget": {
            "max_chars_total": settings.chat_memory_injection_max_chars,
            "max_item_chars": settings.chat_memory_injection_item_max_chars,
            "used_chars": 0,
        },
        "turns": [],
        "source_type_groups": {},
        "stats": {
            "fusion_failed": False,
            "degraded": False,
            "degraded_reasons": [],
            "layer_health": {
                "l1_ok": True,
                "l2_ok": False,
                "l3_ok": False,
                "l1_degraded": False,
                "l2_degraded": False,
                "l3_degraded": False,
                "l2_error_type": "",
                "l3_error_type": "",
            },
        },
    }
    global_constraint_items, global_constraints_status = _load_session_global_constraints(
        chat_id=effective_session_id,
        chat_memory_service=chat_memory_service,
        chat_memory_store=chat_memory_store,
    )
    recent_turns_payload["global_constraints"] = global_constraint_items
    if chat_memory_service is not None:
        if bool(settings.chat_memory_fusion_enabled):
            l2_read_collections: list[str] = []
            for name in (
                str(settings.chat_memory_l2_collection_default or "").strip(),
                "cf_review_memory",
                "cf_task_memory",
            ):
                if name and name not in l2_read_collections:
                    l2_read_collections.append(name)
            try:
                recent_turns_payload = await chat_memory_service.retrieve_memory_for_turn(
                    chat_id=effective_session_id,
                    query=normalized,
                    last_n=settings.chat_memory_recent_turns,
                    max_chars_total=settings.chat_memory_injection_max_chars,
                    max_item_chars=settings.chat_memory_injection_item_max_chars,
                    kernel=kernel,
                    session_id=effective_session_id,
                    encode_func=_get_memory_vector_encode_func(),
                    l2_collection=str(settings.chat_memory_l2_collection_default or "cf_chat_memory"),
                    l2_collections=l2_read_collections,
                    global_constraints_payload=None,
                )
                recent_turns_payload["global_constraints"] = global_constraint_items
            except Exception as exc:
                recent_turns_payload["stats"] = {
                    "fusion_failed": True,
                    "degraded": True,
                    "degraded_reasons": ["fusion_read_failed"],
                    "layer_health": {
                        "l1_ok": False,
                        "l2_ok": False,
                        "l3_ok": False,
                        "l1_degraded": True,
                        "l2_degraded": True,
                        "l3_degraded": True,
                        "l2_error_type": "",
                        "l3_error_type": "",
                    },
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:220],
                }
                trace(
                    "chat_memory.fusion.error",
                    chat_id=effective_session_id,
                    error=type(exc).__name__,
                    message=str(exc)[:220],
                )
                recent_turns_payload["global_constraints"] = global_constraint_items
        else:
            recent_turns_payload = chat_memory_service.load_recent_turns_for_injection(
                chat_id=effective_session_id,
                last_n=settings.chat_memory_recent_turns,
                max_chars_total=settings.chat_memory_injection_max_chars,
                max_item_chars=settings.chat_memory_injection_item_max_chars,
            )
            recent_turns_payload["global_constraints"] = global_constraint_items
    result = await kernel.run(
        normalized,
        session_id=effective_session_id,
        metadata={
            "input_garbled_warning": garbled_warning,
            "chat_id": effective_session_id,
            "user_id": user_id or "anonymous",
            "chat_recent_turns": recent_turns_payload.get("turns", []),
            "chat_recent_turns_budget": recent_turns_payload.get("budget", {}),
            "chat_recent_turns_stats": _build_chat_recent_turns_stats(recent_turns_payload),
            "chat_global_constraints": recent_turns_payload.get("global_constraints", []),
            "chat_global_constraints_source": "l1_session_list",
            "chat_global_constraints_status": global_constraints_status,
        },
    )
    # Keep response contract stable for frontend and external callers.
    result.setdefault("response", "")
    result.setdefault("model", "")
    result.setdefault("run_id", "")
    result.setdefault("session_id", effective_session_id)
    result.setdefault("task_profile", {})
    result.setdefault("workflow_decision", {})
    result.setdefault("plan", {})
    result.setdefault("context_pack", {})
    result.setdefault("selected_agents", [])
    result.setdefault("agent_outputs", {})
    result.setdefault("review_reports", [])
    result.setdefault("quality_gate", {})
    result.setdefault("errors", [])
    result.setdefault("created_at", "")
    result.setdefault("updated_at", "")
    result.setdefault("metadata", {})
    if isinstance(result["metadata"], dict):
        result["metadata"].setdefault("input_garbled_warning", garbled_warning)
        result["metadata"].setdefault("agent_runtime_used", False)
        result["metadata"].setdefault("revision_count", 0)
        result["metadata"].setdefault("max_revision_iterations", 1)
        result["metadata"].setdefault("revision_performed", False)
        result["metadata"].setdefault("revision_target_agent", None)
        result["metadata"].setdefault("chat_id", effective_session_id)
        result["metadata"].setdefault("user_id", user_id or "anonymous")
        result["metadata"].setdefault(
            "chat_recent_turns_stats",
            _build_chat_recent_turns_stats(recent_turns_payload),
        )
        result["metadata"].setdefault(
            "chat_recent_turns_budget",
            recent_turns_payload.get("budget", {}),
        )
        result["metadata"].setdefault(
            "chat_global_constraints",
            recent_turns_payload.get("global_constraints", []),
        )
        result["metadata"].setdefault("chat_global_constraints_source", "l1_session_list")
        result["metadata"].setdefault(
            "chat_global_constraints_status",
            global_constraints_status,
        )

    if chat_memory_store is not None:
        memory_decision: dict[str, Any] | None = None
        memory_sidecar_status: dict[str, Any] = {
            "attempted": False,
            "enabled": bool(settings.chat_memory_sidecar_enabled),
            "succeeded": False,
            "timeout": False,
            "attempt_count": 0,
            "fast_retry_used": False,
            "fallback_used": False,
            "error_type": "",
            "error_message": "",
            "decision_source": "none",
        }
        if chat_memory_service is not None:
            memory_decision, memory_sidecar_status = await _run_memory_sidecar_decision(
                kernel=kernel,
                session_id=effective_session_id,
                task=normalized,
                result=result,
                recent_turns_payload=recent_turns_payload,
            )
            if memory_decision is None and bool(settings.chat_memory_sidecar_enabled):
                fallback_decision = get_memory_sidecar_agent().normalize_decision({})
                error_type = str(memory_sidecar_status.get("error_type", "") or "")
                fallback_source = "system_default_v1"
                if error_type.lower() == "timeouterror":
                    fallback_source = "system_default_timeout_v1"
                elif error_type:
                    fallback_source = "system_default_error_v1"
                fallback_decision["decision_source"] = fallback_source
                memory_decision = fallback_decision
                memory_sidecar_status["fallback_used"] = True
                memory_sidecar_status["decision_source"] = fallback_source
            maintenance_requested, maintenance_reason = _constraint_maintenance_requested(result)
            constraint_maintenance_status: dict[str, Any] = {
                "requested": maintenance_requested,
                "reason": maintenance_reason[:240],
                "attempted": False,
                "changed": False,
                "applied_ops": [],
                "error": {},
            }
            if maintenance_requested:
                constraint_maintenance_status["attempted"] = True
                maintainer_result = await _run_global_constraint_maintainer(
                    kernel=kernel,
                    session_id=effective_session_id,
                    task=normalized,
                    result=result,
                )
                ops = maintainer_result.get("ops", []) if isinstance(maintainer_result, dict) else []
                if not isinstance(ops, list):
                    ops = [{"op": "noop"}]
                applied = chat_memory_store.apply_global_constraint_ops(
                    chat_id=effective_session_id,
                    ops=[x for x in ops if isinstance(x, dict)],
                    source="constraint_maintainer",
                )
                constraint_maintenance_status.update(
                    {
                        "changed": bool(applied.get("changed", False)),
                        "applied_ops": applied.get("applied_ops", []),
                        "error": maintainer_result.get("error", {})
                        if isinstance(maintainer_result, dict) and isinstance(maintainer_result.get("error", {}), dict)
                        else {},
                    }
                )
                refreshed_constraints = chat_memory_store.read_global_constraints(
                    chat_id=effective_session_id
                )
                if isinstance(result.get("metadata"), dict):
                    result["metadata"]["chat_global_constraints"] = refreshed_constraints.get("items", [])
                recent_turns_payload["global_constraints"] = refreshed_constraints.get("items", [])
            if isinstance(result.get("metadata"), dict):
                result["metadata"]["global_constraint_maintenance"] = constraint_maintenance_status
        try:
            memory_write = chat_memory_store.append_turn(
                chat_id=effective_session_id,
                user_id=user_id,
                run_id=str(result.get("run_id", "")),
                task=normalized,
                result=result,
                logs=result.get("trace", []) if isinstance(result.get("trace"), list) else [],
                memory_decision=memory_decision,
            )
            if isinstance(result["metadata"], dict):
                result["metadata"]["chat_memory"] = memory_write
                result["metadata"]["chat_memory_sidecar"] = memory_sidecar_status
                can_compact = bool(
                    settings.chat_memory_compact_enabled
                    and chat_memory_service is not None
                    and callable(getattr(chat_memory_service, "compact_chat_memory", None))
                )
                result["metadata"]["chat_memory_compaction_scheduled"] = can_compact
                _schedule_chat_memory_compaction(
                    chat_memory_service=chat_memory_service,
                    chat_id=effective_session_id,
                )
                if isinstance(memory_decision, dict):
                    result["metadata"]["chat_memory_decision"] = {
                        "decision_source": str(memory_decision.get("decision_source", "model_sidecar_v1")),
                        "global_notes_preview": _json_preview(memory_decision.get("global_notes", ""), max_chars=200),
                        "validation_error_count": len(
                            memory_decision.get("validation_errors", {})
                        )
                        if isinstance(memory_decision.get("validation_errors"), dict)
                        else 0,
                    }
                committed_memories = memory_write.get("committed_memories", [])
                if not isinstance(committed_memories, list):
                    committed_memories = []
                l2l3_write = await _persist_committed_memories_l2_l3(
                    kernel=kernel,
                    result=result,
                    session_id=effective_session_id,
                    chat_id=effective_session_id,
                    memory_write=_resolve_memory_write_policy(
                        result,
                        committed_memories=[x for x in committed_memories if isinstance(x, dict)],
                    ),
                )
                result["metadata"]["chat_memory_l2l3_write"] = l2l3_write
                if bool(l2l3_write.get("failed", False)):
                    strict_required = bool(l2l3_write.get("strict_required", True))
                    failed_errors = l2l3_write.get("errors", [])
                    if not isinstance(failed_errors, list):
                        failed_errors = []
                    if isinstance(result.get("errors"), list):
                        for err in failed_errors:
                            if not isinstance(err, dict):
                                continue
                            result["errors"].append(
                                {
                                    "type": "ChatMemoryL2L3WriteFailed",
                                    "stage": "chat_memory_l2l3",
                                    "message": str(err.get("message", ""))[:300],
                                    "metadata": {
                                        "backend": str(err.get("stage", "")),
                                        "error_type": str(err.get("type", "")),
                                        "collection": str(err.get("collection", "")),
                                        "strict_required": strict_required,
                                    },
                                }
                            )
                    if isinstance(result["metadata"], dict):
                        result["metadata"]["chat_memory_l2l3_failed"] = True
                        result["metadata"]["degraded"] = False
                        result["metadata"]["failed"] = bool(strict_required)
        except Exception as exc:
            if isinstance(result["metadata"], dict):
                result["metadata"]["chat_memory_error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc)[:300],
                }
            if isinstance(result.get("errors"), list):
                result["errors"].append(
                    {
                        "type": "ChatMemoryWriteError",
                        "message": str(exc)[:300],
                        "stage": "chat_memory",
                        "metadata": {"chat_id": effective_session_id},
                    }
                )
    return result


async def run_tool_once(
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    *,
    session_id: str = "default",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one tool through Kernel/ToolRuntime with full permission guard."""
    kernel = get_kernel()
    return await kernel.run_once(
        tool_name=tool_name,
        arguments=arguments or {},
        session_id=session_id,
        metadata=metadata or {"source": "frontend_tool_panel"},
    )
