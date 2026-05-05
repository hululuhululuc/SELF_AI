# coding=utf-8
"""Tests for Phase 3 QdrantMemory with mocked clients only."""

from types import SimpleNamespace

import pytest

from self_ai.memory.qdrant_store import QdrantMemory
from self_ai.schemas import EvidenceItem


class FakeScoredPoint:
    def __init__(self, point_id: str, score: float, payload: dict):
        self.id = point_id
        self.score = score
        self.payload = payload


class FakeQdrantClient:
    def __init__(self) -> None:
        self.collections: set[str] = set()
        self.collection_sizes: dict[str, int] = {}
        self.create_calls: list[tuple[str, int]] = []
        self.query_results: dict[str, list[FakeScoredPoint]] = {}
        self.upsert_calls: list[tuple[str, list, bool]] = []
        self.raise_on_query = False

    def collection_exists(self, *, collection_name: str) -> bool:
        return collection_name in self.collections

    def create_collection(self, *, collection_name: str, vectors_config) -> None:
        self.collections.add(collection_name)
        self.collection_sizes[collection_name] = int(vectors_config.size)
        self.create_calls.append((collection_name, vectors_config.size))

    def get_collection(self, *, collection_name: str):
        size = int(self.collection_sizes.get(collection_name, 0) or 0)
        vectors = SimpleNamespace(size=size, distance="cosine")
        params = SimpleNamespace(vectors=vectors, sparse_vectors=None)
        config = SimpleNamespace(params=params)
        return SimpleNamespace(config=config)

    def query_points(self, *, collection_name: str, query, limit: int, with_payload: bool):
        if self.raise_on_query:
            raise RuntimeError("query failed")
        points = self.query_results.get(collection_name, [])[:limit]
        return SimpleNamespace(points=points)

    def upsert(self, *, collection_name: str, points: list, wait: bool):
        self.upsert_calls.append((collection_name, points, wait))


@pytest.mark.asyncio
async def test_qdrant_memory_supports_dependency_injection() -> None:
    client = FakeQdrantClient()
    memory = QdrantMemory(client=client)
    assert memory.client is client


@pytest.mark.asyncio
async def test_qdrant_memory_ensure_collection_creates_missing_collection() -> None:
    client = FakeQdrantClient()
    memory = QdrantMemory(client=client)

    ok = await memory.ensure_collection("cf_doc_chunks", vector_size=768)

    assert ok
    assert ("cf_doc_chunks", 768) in client.create_calls


@pytest.mark.asyncio
async def test_qdrant_memory_search_returns_evidence_items() -> None:
    client = FakeQdrantClient()
    client.collections.add("cf_doc_chunks")
    client.collection_sizes["cf_doc_chunks"] = 3
    client.query_results["cf_doc_chunks"] = [
        FakeScoredPoint(
            "p1",
            0.91,
            {
                "content": "retrieved content",
                "source_type": "doc",
                "chunk_id": "chunk-1",
                "path": "docs/a.md",
            },
        )
    ]
    memory = QdrantMemory(client=client)

    results = await memory.search(
        collection_name="cf_doc_chunks",
        dense_vector=[0.1, 0.2, 0.3],
        limit=5,
        min_score=0.0,
    )

    assert len(results) == 1
    assert results[0].content == "retrieved content"
    assert results[0].source_type == "doc"
    assert results[0].chunk_id == "chunk-1"


@pytest.mark.asyncio
async def test_qdrant_memory_search_exception_returns_empty_list() -> None:
    client = FakeQdrantClient()
    client.collections.add("cf_doc_chunks")
    client.collection_sizes["cf_doc_chunks"] = 2
    client.raise_on_query = True
    memory = QdrantMemory(client=client)

    results = await memory.search(
        collection_name="cf_doc_chunks",
        dense_vector=[0.1, 0.2],
    )

    assert results == []


@pytest.mark.asyncio
async def test_qdrant_memory_schema_mismatch_returns_safe_failure() -> None:
    client = FakeQdrantClient()
    client.collections.add("cf_doc_chunks")
    client.collection_sizes["cf_doc_chunks"] = 16
    memory = QdrantMemory(client=client)

    results = await memory.search(
        collection_name="cf_doc_chunks",
        dense_vector=[0.1, 0.2, 0.3],
        limit=3,
    )

    assert results == []


@pytest.mark.asyncio
async def test_qdrant_memory_upsert_serializes_evidence_payload() -> None:
    client = FakeQdrantClient()
    memory = QdrantMemory(client=client)
    evidence = EvidenceItem(
        content="new memory",
        source_type="task",
        chunk_id="chunk-upsert",
        metadata={"k": "v"},
    )

    ok = await memory.upsert_evidence(
        collection_name="cf_task_memory",
        evidence=evidence,
        dense_vector=[0.4, 0.5, 0.6],
    )

    assert ok
    assert client.upsert_calls
    collection, points, wait = client.upsert_calls[0]
    assert collection == "cf_task_memory"
    assert wait is True
    payload = points[0].payload
    assert payload["content"] == "new memory"
    assert payload["source_type"] == "task"
    assert payload["embedding_model"] == "bge-m3"


@pytest.mark.asyncio
async def test_qdrant_memory_upsert_fails_on_schema_mismatch_without_fallback() -> None:
    client = FakeQdrantClient()
    client.collections.add("cf_task_memory")
    client.collection_sizes["cf_task_memory"] = 16
    memory = QdrantMemory(client=client)

    ok = await memory.upsert_memory(
        collection_name="cf_task_memory",
        evidence={"content": "x", "source_type": "task"},
        dense_vector=[0.1, 0.2, 0.3],
    )

    assert ok is False
    assert client.upsert_calls == []
