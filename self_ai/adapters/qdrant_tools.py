# coding=utf-8
"""Qdrant semantic memory tool adapters."""

from __future__ import annotations

from typing import Any
from uuid import uuid5, NAMESPACE_URL

from qdrant_client.http import models as qmodels

from ..config import settings
from ..runtime.tool_registry import ToolRegistry
from ..runtime.tool_spec import ToolSpec
from ..schemas import EvidenceItem


def _safe_text(value: Any) -> str:
    return "" if value is None else str(value)


def _resolve_qdrant_memory(explicit_memory: Any | None, run_context: Any | None) -> Any:
    if explicit_memory is not None:
        return explicit_memory
    if run_context is not None:
        metadata = getattr(run_context, "metadata", None)
        if isinstance(metadata, dict) and metadata.get("qdrant_memory") is not None:
            return metadata["qdrant_memory"]
    raise RuntimeError("qdrant_memory is not configured")


def _resolve_encode_func(run_context: Any | None) -> Any | None:
    if run_context is not None:
        metadata = getattr(run_context, "metadata", None)
        if isinstance(metadata, dict):
            func = metadata.get("encode_func")
            if callable(func):
                return func
    try:
        from ..tools import _encode_text as encode_text  # type: ignore

        return encode_text
    except Exception:
        return None


def _coerce_dense_vector(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    out: list[float] = []
    for item in value:
        try:
            out.append(float(item))
        except Exception:
            return []
    return out


def _build_dense_vector(
    *,
    explicit_dense_vector: Any,
    text_for_encoding: str,
    run_context: Any | None,
) -> list[float]:
    vec = _coerce_dense_vector(explicit_dense_vector)
    if vec:
        return vec
    if not text_for_encoding.strip():
        return []
    encode_func = _resolve_encode_func(run_context)
    if not callable(encode_func):
        return []
    try:
        encoded = encode_func(text_for_encoding)
    except Exception:
        return []
    return _coerce_dense_vector(encoded)


def _allowed_collections(memory: Any) -> set[str]:
    allowed = set(getattr(memory, "DEFAULT_COLLECTIONS", ()) or ())
    allowed.update(
        {
            "cf_chat_memory",
            "cf_preference_memory",
            "cf_task_memory",
            "cf_review_memory",
            "cf_doc_chunks",
            "cf_code_chunks",
            "cf_web_chunks",
            "cf_eval_cases",
        }
    )
    default_chat_collection = str(
        getattr(settings, "chat_memory_l2_collection_default", "cf_chat_memory")
        or "cf_chat_memory"
    ).strip()
    if default_chat_collection:
        allowed.add(default_chat_collection)
    return allowed


def _normalize_collection_name(
    memory: Any,
    requested: Any,
    *,
    fallback: str,
    for_issue: bool = False,
) -> tuple[str, str]:
    candidate = str(requested or "").strip()
    if not candidate:
        candidate = "cf_review_memory" if for_issue else fallback
    allowed = _allowed_collections(memory)
    if candidate in allowed:
        return candidate, ""

    normalized_from = candidate
    lowered = candidate.lower()
    default_chat_collection = str(
        getattr(settings, "chat_memory_l2_collection_default", "cf_chat_memory")
        or "cf_chat_memory"
    ).strip()
    # Deterministic normalization for common invalid inputs from LLM.
    if for_issue:
        candidate = "cf_review_memory"
    elif lowered in {"default", "preferences", "preference", "chat", "memory", "session"}:
        candidate = default_chat_collection
    elif lowered in {"review", "issues", "issue", "errors"}:
        candidate = "cf_review_memory"
    else:
        candidate = default_chat_collection

    if candidate not in allowed:
        raise RuntimeError(
            f"collection_not_allowed:{normalized_from}. allowed={sorted(allowed)}"
        )
    return candidate, normalized_from


def _normalize_query_filter(raw: Any) -> Any | None:
    if raw is None:
        return None
    if isinstance(raw, qmodels.Filter):
        return raw
    if isinstance(raw, dict):
        must: list[Any] = []
        for key, value in raw.items():
            if value is None:
                continue
            must.append(
                qmodels.FieldCondition(
                    key=str(key),
                    match=qmodels.MatchValue(value=value),
                )
            )
        if must:
            return qmodels.Filter(must=must)
    return None


def _with_chat_scope_filter(
    query_filter: Any | None,
    *,
    run_context: Any | None,
) -> Any | None:
    session_id = _safe_text(getattr(run_context, "session_id", "") if run_context is not None else "")
    if not session_id:
        return query_filter
    scoped_condition = qmodels.FieldCondition(
        key="metadata.chat_id",
        match=qmodels.MatchValue(value=session_id),
    )
    if query_filter is None:
        return qmodels.Filter(must=[scoped_condition])
    if isinstance(query_filter, qmodels.Filter):
        must = list(getattr(query_filter, "must", None) or [])
        must.append(scoped_condition)
        query_filter.must = must
        return query_filter
    return query_filter


def _build_evidence(args: dict[str, Any], *, fallback_content: str = "") -> EvidenceItem:
    evidence_raw = args.get("evidence", {})
    if isinstance(evidence_raw, EvidenceItem):
        return evidence_raw
    if isinstance(evidence_raw, dict) and evidence_raw.get("content"):
        return EvidenceItem(**evidence_raw)

    content = str(args.get("content") or args.get("text") or fallback_content or "").strip()
    if not content:
        raise RuntimeError("missing_evidence_content")
    metadata = args.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {"raw_metadata": str(metadata)}
    return EvidenceItem(
        content=content,
        source_type=str(args.get("source_type") or "chat_memory"),
        source=args.get("source"),
        path=args.get("path"),
        symbol=args.get("symbol"),
        chunk_id=args.get("chunk_id"),
        score=float(args.get("score", 0.0) or 0.0) if args.get("score") is not None else None,
        title=args.get("title"),
        url=args.get("url"),
        token_count=int(args.get("token_count", 0) or 0),
        created_at=args.get("created_at"),
        embedding_model=args.get("embedding_model"),
        metadata=metadata,
    )


def register_qdrant_tools(registry: ToolRegistry, *, qdrant_memory: Any | None = None) -> None:
    """Register Qdrant semantic tools."""

    async def _search(args: dict[str, Any], run_context: Any | None) -> dict[str, Any]:
        memory = _resolve_qdrant_memory(qdrant_memory, run_context)
        collection_name, normalized_from = _normalize_collection_name(
            memory,
            args.get("collection_name"),
            fallback=str(getattr(settings, "chat_memory_l2_collection_default", "cf_chat_memory") or "cf_chat_memory"),
        )
        query_text = str(args.get("query_text") or args.get("query") or "").strip()
        dense_vector = _build_dense_vector(
            explicit_dense_vector=args.get("dense_vector"),
            text_for_encoding=query_text,
            run_context=run_context,
        )
        if not dense_vector:
            return {
                "evidence": [],
                "error": {
                    "type": "VectorUnavailable",
                    "message": "dense_vector missing and query_text encoding unavailable",
                },
            }
        query_filter = _normalize_query_filter(args.get("filter") or args.get("query_filter"))
        query_filter = _with_chat_scope_filter(query_filter, run_context=run_context)
        hits = await memory.search(
            collection_name=collection_name,
            dense_vector=dense_vector,
            limit=int(args.get("limit", 5)),
            min_score=float(args.get("min_score", 0.0)),
            query_filter=query_filter,
        )
        response = {
            "evidence": [
                item.model_dump(mode="json") if hasattr(item, "model_dump") else item
                for item in (hits or [])
            ]
        }
        if normalized_from:
            response["collection_normalized"] = {
                "from": normalized_from,
                "to": collection_name,
            }
        return response

    async def _upsert_evidence(args: dict[str, Any], run_context: Any | None) -> dict[str, Any]:
        memory = _resolve_qdrant_memory(qdrant_memory, run_context)
        source_type = str(args.get("source_type") or "").strip().lower()
        for_issue = "issue" in source_type or "error" in source_type or "failure" in source_type
        collection_name, normalized_from = _normalize_collection_name(
            memory,
            args.get("collection_name"),
            fallback=str(getattr(settings, "chat_memory_l2_collection_default", "cf_chat_memory") or "cf_chat_memory"),
            for_issue=for_issue,
        )
        evidence = _build_evidence(args)
        dense_vector = _build_dense_vector(
            explicit_dense_vector=args.get("dense_vector"),
            text_for_encoding=evidence.content,
            run_context=run_context,
        )
        if not dense_vector:
            raise RuntimeError("vector_unavailable_for_upsert_evidence")
        point_id = args.get("point_id")
        if point_id is None:
            point_id = str(uuid5(NAMESPACE_URL, f"{collection_name}:{evidence.content[:120]}"))
        ok = await memory.upsert_evidence(
            collection_name=collection_name,
            dense_vector=dense_vector,
            evidence=evidence,
            point_id=point_id,
        )
        if ok is False:
            raise RuntimeError("Qdrant upsert_evidence failed")
        response = {"upserted": True, "collection_name": collection_name}
        if normalized_from:
            response["collection_normalized"] = {"from": normalized_from, "to": collection_name}
        return response

    async def _upsert_memory(args: dict[str, Any], run_context: Any | None) -> dict[str, Any]:
        memory = _resolve_qdrant_memory(qdrant_memory, run_context)
        evidence_raw = args.get("evidence", {})
        evidence_meta = evidence_raw.get("metadata", {}) if isinstance(evidence_raw, dict) else {}
        memory_type = str(evidence_meta.get("memory_type", "") or "").strip().lower()
        memory_class = str(evidence_meta.get("memory_class", "") or "").strip().lower()
        for_issue = memory_type == "issue" or any(t in memory_class for t in ("issue", "error", "failure", "bug"))
        collection_name, normalized_from = _normalize_collection_name(
            memory,
            args.get("collection_name"),
            fallback=str(getattr(settings, "chat_memory_l2_collection_default", "cf_chat_memory") or "cf_chat_memory"),
            for_issue=for_issue,
        )
        evidence = _build_evidence(args)
        dense_vector = _build_dense_vector(
            explicit_dense_vector=args.get("dense_vector"),
            text_for_encoding=evidence.content,
            run_context=run_context,
        )
        if not dense_vector:
            raise RuntimeError("vector_unavailable_for_upsert_memory")
        point_id = args.get("point_id")
        if point_id is None:
            point_id = str(uuid5(NAMESPACE_URL, f"{collection_name}:{evidence.content[:120]}"))
        ok = await memory.upsert_memory(
            collection_name=collection_name,
            evidence=evidence,
            dense_vector=dense_vector,
            point_id=point_id,
        )
        if ok is False:
            raise RuntimeError("Qdrant upsert_memory failed")
        response = {"upserted": True, "collection_name": collection_name}
        if normalized_from:
            response["collection_normalized"] = {"from": normalized_from, "to": collection_name}
        return response

    registry.register(
        ToolSpec(
            name="storage.semantic.search",
            description="Search semantic evidence from QdrantMemory.",
            input_schema={
                "type": "object",
                "properties": {
                    "collection_name": {"type": "string"},
                    "query_text": {"type": "string"},
                    "query": {"type": "string"},
                    "dense_vector": {"type": "array", "items": {"type": "number"}},
                    "limit": {"type": "integer"},
                    "min_score": {"type": "number"},
                    "filter": {"type": ["object", "null"]},
                    "query_filter": {"type": ["object", "null"]},
                },
                "required": [],
            },
            output_schema={"type": "object", "properties": {"evidence": {"type": "array"}}},
            permission="memory_read",
            timeout_s=20,
            tags=["qdrant", "retrieval"],
            handler=_search,
        )
    )
    registry.register(
        ToolSpec(
            name="storage.semantic.upsert_evidence",
            description="Upsert evidence into Qdrant semantic collection.",
            input_schema={
                "type": "object",
                "properties": {
                    "collection_name": {"type": "string"},
                    "content": {"type": "string"},
                    "text": {"type": "string"},
                    "dense_vector": {"type": "array", "items": {"type": "number"}},
                    "evidence": {"type": "object"},
                    "point_id": {"type": ["string", "integer", "null"]},
                    "source_type": {"type": "string"},
                    "metadata": {"type": "object"},
                },
                "required": [],
            },
            output_schema={"type": "object", "properties": {"upserted": {"type": "boolean"}}},
            permission="memory_write",
            timeout_s=20,
            tags=["qdrant", "memory_write"],
            handler=_upsert_evidence,
        )
    )
    registry.register(
        ToolSpec(
            name="storage.semantic.upsert_memory",
            description="Upsert memory record into Qdrant semantic collection.",
            input_schema={
                "type": "object",
                "properties": {
                    "collection_name": {"type": "string"},
                    "content": {"type": "string"},
                    "text": {"type": "string"},
                    "dense_vector": {"type": "array", "items": {"type": "number"}},
                    "evidence": {"type": "object"},
                    "point_id": {"type": ["string", "integer", "null"]},
                    "source_type": {"type": "string"},
                    "metadata": {"type": "object"},
                },
                "required": [],
            },
            output_schema={"type": "object", "properties": {"upserted": {"type": "boolean"}}},
            permission="memory_write",
            timeout_s=20,
            tags=["qdrant", "memory_write"],
            handler=_upsert_memory,
        )
    )
