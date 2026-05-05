# coding=utf-8
"""Qdrant semantic memory layer for Phase 3.

This module provides a dependency-injected Qdrant memory store that supports
multiple collections and returns standardized EvidenceItem objects.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any
from uuid import uuid4

from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from ..observability import trace
from ..schemas import EvidenceItem


class QdrantMemory:
    """Semantic memory store backed by Qdrant.

    The Qdrant client must be injected from outside for testability.
    """

    DEFAULT_COLLECTIONS: tuple[str, ...] = (
        "cf_code_chunks",
        "cf_doc_chunks",
        "cf_web_chunks",
        "cf_task_memory",
        "cf_review_memory",
        "cf_chat_memory",
        "cf_preference_memory",
        "cf_eval_cases",
    )

    def __init__(
        self,
        client: Any,
        *,
        default_vector_size: int = 1024,
        default_limit: int = 5,
        embedding_model: str = "bge-m3",
        strict_schema: bool = True,
    ) -> None:
        self.client = client
        self.default_vector_size = default_vector_size
        self.default_limit = default_limit
        self.embedding_model = embedding_model
        self.strict_schema = strict_schema

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    async def _call(self, method_name: str, /, *args: Any, **kwargs: Any) -> Any:
        method = getattr(self.client, method_name)
        attempts = 0
        last_exc: Exception | None = None
        while attempts < 3:
            attempts += 1
            try:
                return await self._maybe_await(method(*args, **kwargs))
            except UnexpectedResponse as exc:
                status_code = int(getattr(exc, "status_code", 0) or 0)
                retryable = status_code in {500, 502, 503, 504}
                if not retryable or attempts >= 3:
                    raise
                last_exc = exc
                trace(
                    "qdrant_memory.transport.retry",
                    method=method_name,
                    attempt=attempts,
                    status_code=status_code,
                )
                await asyncio.sleep(0.25 * attempts)
            except ResponseHandlingException as exc:
                if attempts >= 3:
                    raise
                last_exc = exc
                trace(
                    "qdrant_memory.transport.retry",
                    method=method_name,
                    attempt=attempts,
                    error=type(exc).__name__,
                )
                await asyncio.sleep(0.25 * attempts)
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"qdrant call failed unexpectedly: {method_name}")

    @staticmethod
    def _vector_params_from_collection_info(collection_info: Any) -> tuple[int | None, str | None, bool]:
        config = getattr(collection_info, "config", None)
        params = getattr(config, "params", None) if config is not None else None
        vectors = getattr(params, "vectors", None) if params is not None else None
        sparse_vectors = getattr(params, "sparse_vectors", None) if params is not None else None
        has_sparse = bool(sparse_vectors)

        if isinstance(vectors, qmodels.VectorParams):
            distance = str(getattr(vectors, "distance", "") or "")
            size = int(getattr(vectors, "size", 0) or 0)
            return (size if size > 0 else None, distance, has_sparse)

        if vectors is not None and hasattr(vectors, "size"):
            distance = str(getattr(vectors, "distance", "") or "")
            size = int(getattr(vectors, "size", 0) or 0)
            return (size if size > 0 else None, distance, has_sparse)

        if isinstance(vectors, dict) and vectors:
            first = next(iter(vectors.values()))
            if isinstance(first, qmodels.VectorParams):
                distance = str(getattr(first, "distance", "") or "")
                size = int(getattr(first, "size", 0) or 0)
                return (size if size > 0 else None, distance, has_sparse)

        return (None, None, has_sparse)

    async def _validate_existing_collection_schema(
        self,
        *,
        collection_name: str,
        expected_vector_size: int,
    ) -> tuple[bool, dict[str, Any]]:
        try:
            collection_info = await self._call("get_collection", collection_name=collection_name)
        except Exception as exc:
            return (
                False,
                {
                    "error_type": type(exc).__name__,
                    "message": str(exc)[:200],
                    "reason": "get_collection_failed",
                },
            )

        size, distance, has_sparse = self._vector_params_from_collection_info(collection_info)
        normalized_distance = str(distance or "").strip().lower()
        if normalized_distance.startswith("distance."):
            normalized_distance = normalized_distance.split(".", 1)[1]
        size_match = int(size or 0) == int(expected_vector_size)
        distance_match = normalized_distance in {"cosine", "distance.cosine"}
        schema_ok = bool(size_match and distance_match)
        detail = {
            "actual_vector_size": size,
            "expected_vector_size": expected_vector_size,
            "actual_distance": distance,
            "expected_distance": "cosine",
            "has_sparse_vectors": has_sparse,
            "size_match": size_match,
            "distance_match": distance_match,
        }
        return (schema_ok, detail)

    @staticmethod
    def _extract_points(response: Any) -> list[Any]:
        if response is None:
            return []
        if hasattr(response, "points"):
            points = getattr(response, "points")
            return list(points or [])
        if isinstance(response, list):
            return response
        return []

    async def ensure_collection(
        self,
        collection_name: str,
        vector_size: int | None = None,
    ) -> bool:
        """Ensure one collection exists with dense vector config."""
        size = vector_size or self.default_vector_size
        try:
            exists = await self._call("collection_exists", collection_name=collection_name)
            if not bool(exists):
                await self._call(
                    "create_collection",
                    collection_name=collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=size,
                        distance=qmodels.Distance.COSINE,
                    ),
                )
                trace(
                    "qdrant_memory.collection.created",
                    collection=collection_name,
                    vector_size=size,
                )
            elif self.strict_schema:
                schema_ok, detail = await self._validate_existing_collection_schema(
                    collection_name=collection_name,
                    expected_vector_size=size,
                )
                if not schema_ok:
                    trace(
                        "qdrant_memory.collection.schema_mismatch",
                        collection=collection_name,
                        error="VectorSchemaMismatch",
                        **detail,
                    )
                    return False
            return True
        except Exception as exc:
            trace(
                "qdrant_memory.collection.error",
                collection=collection_name,
                error=type(exc).__name__,
                message=str(exc)[:200],
            )
            return False

    async def ensure_collections(
        self,
        vector_size: int | None = None,
        collections: list[str] | None = None,
    ) -> None:
        """Ensure required semantic memory collections exist."""
        target = collections or list(self.DEFAULT_COLLECTIONS)
        for name in target:
            await self.ensure_collection(name, vector_size=vector_size)

    @staticmethod
    def _payload_to_evidence(
        point: Any,
        payload: dict[str, Any],
        collection_name: str,
        embedding_model: str,
    ) -> EvidenceItem:
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {"raw_metadata": str(metadata)}
        metadata = {
            **metadata,
            "collection": collection_name,
            "point_id": str(getattr(point, "id", "")),
        }
        content = str(payload.get("content") or payload.get("text") or "")
        return EvidenceItem(
            content=content,
            source_type=str(payload.get("source_type") or payload.get("source") or "qdrant"),
            source=payload.get("source"),
            path=payload.get("path"),
            symbol=payload.get("symbol"),
            chunk_id=(payload.get("chunk_id") or str(getattr(point, "id", "")) or None),
            score=float(getattr(point, "score", 0.0) or 0.0),
            title=payload.get("title"),
            url=payload.get("url"),
            token_count=int(payload.get("token_count", 0) or 0),
            created_at=payload.get("created_at"),
            embedding_model=str(payload.get("embedding_model") or embedding_model),
            metadata=metadata,
        )

    async def search(
        self,
        *,
        collection_name: str,
        dense_vector: list[float],
        limit: int | None = None,
        min_score: float = 0.0,
        query_filter: Any | None = None,
    ) -> list[EvidenceItem]:
        """Search one collection and return structured evidence.

        On Qdrant exceptions this method returns an empty list and never raises.
        """
        if not dense_vector:
            return []

        top_k = limit or self.default_limit
        try:
            ensured = await self.ensure_collection(collection_name, vector_size=len(dense_vector))
            if not ensured:
                trace(
                    "qdrant_memory.search.error",
                    collection=collection_name,
                    error="VectorSchemaMismatch",
                    message="ensure_collection failed for current vector schema",
                )
                return []
            query_kwargs: dict[str, Any] = {
                "collection_name": collection_name,
                "query": dense_vector,
                "limit": top_k,
                "with_payload": True,
            }
            if query_filter is not None:
                query_kwargs["query_filter"] = query_filter
            response = await self._call("query_points", **query_kwargs)
            points = self._extract_points(response)
            items: list[EvidenceItem] = []
            for point in points:
                payload = getattr(point, "payload", None) or {}
                if not isinstance(payload, dict):
                    payload = {}
                evidence = self._payload_to_evidence(
                    point,
                    payload,
                    collection_name,
                    self.embedding_model,
                )
                if evidence.score is None or evidence.score < min_score:
                    continue
                items.append(evidence)
            return items
        except Exception as exc:
            trace(
                "qdrant_memory.search.error",
                collection=collection_name,
                error=type(exc).__name__,
                message=str(exc)[:200],
            )
            return []

    async def upsert_evidence(
        self,
        *,
        collection_name: str,
        evidence: EvidenceItem | dict[str, Any],
        dense_vector: list[float],
        point_id: str | None = None,
        wait: bool = True,
    ) -> bool:
        """Upsert one evidence item into a collection."""
        if not dense_vector:
            return False
        try:
            ensured = await self.ensure_collection(collection_name, vector_size=len(dense_vector))
            if not ensured:
                trace(
                    "qdrant_memory.upsert.error",
                    collection=collection_name,
                    error="VectorSchemaMismatch",
                    message=f"ensure_collection failed for vector_size={len(dense_vector)} embedding_model={self.embedding_model}",
                )
                return False
            evidence_item = evidence if isinstance(evidence, EvidenceItem) else EvidenceItem(**evidence)
            payload = evidence_item.model_dump(mode="json")
            if not payload.get("embedding_model"):
                payload["embedding_model"] = self.embedding_model
            point = qmodels.PointStruct(
                id=point_id or evidence_item.chunk_id or str(uuid4()),
                vector=dense_vector,
                payload=payload,
            )
            await self._call(
                "upsert",
                collection_name=collection_name,
                points=[point],
                wait=wait,
            )
            return True
        except Exception as exc:
            trace(
                "qdrant_memory.upsert.error",
                collection=collection_name,
                error=type(exc).__name__,
                message=str(exc)[:200],
            )
            return False

    async def upsert_memory(
        self,
        *,
        collection_name: str,
        evidence: EvidenceItem | dict[str, Any],
        dense_vector: list[float],
        point_id: str | None = None,
        wait: bool = True,
    ) -> bool:
        """Alias for upsert_evidence to keep API naming flexible."""
        return await self.upsert_evidence(
            collection_name=collection_name,
            evidence=evidence,
            dense_vector=dense_vector,
            point_id=point_id,
            wait=wait,
        )
