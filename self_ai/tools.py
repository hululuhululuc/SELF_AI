# coding=utf-8
"""Tools for retrieval and RAG in Self AI.

Phase 3 keeps the public graphrag_plus interface unchanged while moving retrieval
internals to QdrantMemory + QueryPlanner + ContextBuilder.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from qdrant_client import AsyncQdrantClient, QdrantClient
from tavily import TavilyClient
from tenacity import retry, stop_after_attempt, wait_fixed

from .config import resolve_project_path, settings
from .memory.neo4j_store import Neo4jGraphMemory, NullNeo4jGraphMemory
from .memory.qdrant_store import QdrantMemory
from .observability import make_trace_payload, now_ms, trace
from .retrieval.context_builder import ContextBuilder
from .retrieval.graph_query_adapter import GraphQueryAdapter, GraphQueryPlan
from .retrieval.query_planner import QueryPlanner
from .schemas import TaskDomain, TaskIntent, TaskProfile
from .storage_policy import enrich_profile_with_storage_policies
from .text_utils import normalize_text

qdrant: AsyncQdrantClient | QdrantClient = (
    AsyncQdrantClient(
        url=settings.qdrant_url,
        check_compatibility=False,
        trust_env=False,
        timeout=settings.qdrant_timeout_s,
    )
    if settings.use_async
    else QdrantClient(
        url=settings.qdrant_url,
        check_compatibility=False,
        trust_env=False,
        timeout=settings.qdrant_timeout_s,
    )
)

qdrant_memory = QdrantMemory(client=qdrant, embedding_model="BAAI/bge-m3")
qdrant_memory.strict_schema = bool(settings.qdrant_strict_schema)
query_planner = QueryPlanner()
context_builder = ContextBuilder()


class _LazyTavilyClient:
    def __init__(self) -> None:
        self._client: TavilyClient | None = None

    def _get_client(self) -> TavilyClient:
        if not settings.tavily_api_key:
            raise RuntimeError("TAVILY_API_KEY is required for web fallback retrieval.")
        if self._client is None:
            self._client = TavilyClient(api_key=settings.tavily_api_key)
        return self._client

    def search(self, *args: Any, **kwargs: Any) -> Any:
        return self._get_client().search(*args, **kwargs)


tavily = _LazyTavilyClient()


def _build_graph_memory() -> Neo4jGraphMemory | NullNeo4jGraphMemory:
    if not settings.graph_read_enabled or not settings.neo4j_enabled:
        return NullNeo4jGraphMemory()
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        return Neo4jGraphMemory(
            driver=driver,
            database=settings.neo4j_database,
            enabled=True,
        )
    except Exception as exc:
        trace(
            "retrieval.graph_memory.init.error",
            error=type(exc).__name__,
            message=str(exc)[:200],
        )
        return NullNeo4jGraphMemory()


graph_memory = _build_graph_memory()
graph_query_adapter = GraphQueryAdapter(
    graph_memory=graph_memory,
    max_paths=settings.graph_max_paths,
    enabled=settings.graph_read_enabled and settings.neo4j_enabled,
)

if TYPE_CHECKING:
    from FlagEmbedding import BGEM3FlagModel

_embedder: Optional["BGEM3FlagModel"] = None
_runtime_context: dict[str, Any] = {
    "run_id": None,
    "session_id": "default",
    "redis_store": None,
}


def set_retrieval_runtime_context(
    *,
    run_id: str | None,
    session_id: str = "default",
    redis_store: Any | None = None,
) -> None:
    """Inject runtime context for Redis trace/output write-through."""
    _runtime_context["run_id"] = run_id
    _runtime_context["session_id"] = session_id
    _runtime_context["redis_store"] = redis_store


def clear_retrieval_runtime_context() -> None:
    """Clear runtime context after one research pass."""
    _runtime_context["run_id"] = None
    _runtime_context["session_id"] = "default"
    _runtime_context["redis_store"] = None


def _append_runtime_trace(event: str, **fields: Any) -> None:
    run_id = _runtime_context.get("run_id")
    session_id = _runtime_context.get("session_id", "default")
    trace_fields = dict(fields)
    if run_id and "run_id" not in trace_fields:
        trace_fields["run_id"] = run_id
    if "session_id" not in trace_fields:
        trace_fields["session_id"] = session_id

    trace(event, **trace_fields)

    redis_store = _runtime_context.get("redis_store")
    if not run_id or redis_store is None:
        return
    try:
        payload = make_trace_payload(event, **trace_fields)
        redis_store.append_trace(run_id, payload)
    except Exception:
        return


def _save_runtime_retrieval_output(summary: dict[str, Any]) -> None:
    run_id = _runtime_context.get("run_id")
    redis_store = _runtime_context.get("redis_store")
    if not run_id or redis_store is None:
        return
    try:
        redis_store.save_node_output(run_id, "retrieval", summary)
    except Exception:
        return


def _get_embedder() -> "BGEM3FlagModel":
    """Lazily load local BGE-M3 model to avoid import-time stalls."""
    global _embedder
    if _embedder is None:
        trace("embedder.import.start", module="FlagEmbedding")
        from FlagEmbedding import BGEM3FlagModel

        trace("embedder.import.end", module="FlagEmbedding")
        try:
            from transformers.utils import logging as hf_logging

            hf_logging.set_verbosity_error()
            hf_logging.disable_progress_bar()
        except Exception:
            pass

        model_path = resolve_project_path(settings.embedder_path)
        trace("embedder.load.start", path=str(model_path), fp16=settings.embedder_fp16)
        if not model_path.exists():
            raise FileNotFoundError(
                f"BGE-M3 model path not found: {model_path}. "
                "Please download model to this path first."
            )
        _embedder = BGEM3FlagModel(
            str(model_path),
            use_fp16=settings.embedder_fp16,
        )
        trace("embedder.load.end", path=str(model_path))
    return _embedder


def prewarm_embedder() -> bool:
    """Best-effort embedder prewarm with explicit observability.

    Returns True only if model is fully loaded and ready.
    """
    trace("embedder.warmup.start", enabled=bool(settings.embedder_prewarm), path=settings.embedder_path)
    try:
        _get_embedder()
        trace("embedder.warmup.end", ok=True, path=settings.embedder_path)
        return True
    except Exception as exc:
        trace(
            "embedder.warmup.error",
            ok=False,
            error=type(exc).__name__,
            message=str(exc)[:220],
            path=settings.embedder_path,
        )
        return False


def _encode_text(text: str) -> list[float]:
    """Encode text into dense vector using local BGE-M3."""
    model = _get_embedder()
    output = model.encode(
        [text],
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    dense_raw = output["dense_vecs"][0]
    return dense_raw.tolist() if hasattr(dense_raw, "tolist") else list(dense_raw)


def _profile_from_content_type(content_type: str) -> TaskProfile:
    """Build a lightweight TaskProfile for retrieval planning in tools layer."""
    if content_type == "code":
        profile = TaskProfile(
            intent=TaskIntent.CODE_GENERATION,
            domain=TaskDomain.SOFTWARE,
            requires_code=True,
        )
        return enrich_profile_with_storage_policies(profile)
    if content_type == "research":
        profile = TaskProfile(
            intent=TaskIntent.RESEARCH_SUMMARY,
            domain=TaskDomain.GENERAL,
            requires_research=True,
        )
        return enrich_profile_with_storage_policies(profile)

    profile = TaskProfile(
        intent=TaskIntent.GENERAL_QA,
        domain=TaskDomain.GENERAL,
    )
    return enrich_profile_with_storage_policies(profile)


def _extract_web_contents(web_result: Any) -> list[str]:
    """Normalize Tavily responses into content strings."""
    if isinstance(web_result, dict):
        items = web_result.get("results", [])
    elif isinstance(web_result, list):
        items = web_result
    else:
        items = []

    contents: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        content = item.get("content") or item.get("raw_content") or ""
        if content:
            contents.append(str(content))
    return contents


async def _index_web_result(content: str) -> None:
    """Index one web result into semantic web collection."""
    dense_vector = _encode_text(content)
    await qdrant_memory.upsert_evidence(
        collection_name="cf_web_chunks",
        dense_vector=dense_vector,
        evidence={
            "content": content,
            "source_type": "web",
            "source": "tavily",
            "embedding_model": "BAAI/bge-m3",
            "metadata": {"ingested_from": "tavily"},
        },
    )


async def _search_plan_collections(
    *,
    dense_query: list[float],
    collections: list[str],
    limit: int,
    min_score: float,
    collection_weights: dict[str, float],
) -> list[dict[str, Any]]:
    evidence_items: list[dict[str, Any]] = []
    for collection in collections:
        _append_runtime_trace(
            "retrieval.qdrant_search.start",
            collection=collection,
            limit=limit,
            min_score=min_score,
        )
        hits = await qdrant_memory.search(
            collection_name=collection,
            dense_vector=dense_query,
            limit=limit,
            min_score=min_score,
        )
        weight = float(collection_weights.get(collection, 1.0))
        weighted_hits: list[dict[str, Any]] = []
        for item in hits:
            if item.score is not None:
                item.score = item.score * weight
            weighted_hits.append(item.model_dump(mode="json"))
        _append_runtime_trace(
            "retrieval.qdrant_search.end",
            collection=collection,
            result_count=len(weighted_hits),
        )
        evidence_items.extend(weighted_hits)
    return evidence_items


def _graph_path_signature(path: Any) -> str:
    if hasattr(path, "model_dump"):
        payload = path.model_dump(mode="json")
    elif isinstance(path, dict):
        payload = path
    else:
        payload = {"repr": repr(path)}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def _dedup_graph_paths(paths: list[Any], limit: int) -> list[Any]:
    deduped: dict[str, Any] = {}
    for path in paths:
        deduped[_graph_path_signature(path)] = path
    return list(deduped.values())[: max(1, int(limit))]


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
async def graphrag_plus(
    query: str, content_type: str = "general"
) -> list[dict[str, Any]]:
    """Perform semantic retrieval and return list[EvidenceItem-dict]."""
    normalized_query = normalize_text(query)
    profile = _profile_from_content_type(content_type)
    plan = query_planner.build(profile, profile.retrieval_policy)

    _append_runtime_trace(
        "retrieval.query_planned",
        query_preview=normalized_query[:120],
        intent=plan.intent,
        collections=plan.collections,
        top_k=plan.top_k,
        min_score=plan.min_score,
    )

    dense_query = _encode_text(normalized_query)

    evidence_dicts = await _search_plan_collections(
        dense_query=dense_query,
        collections=plan.collections,
        limit=plan.top_k,
        min_score=plan.min_score,
        collection_weights=plan.collection_weights,
    )

    used_web_fallback = False
    allow_web_fallback = not bool(settings.disable_legacy_retrieval)
    if not allow_web_fallback and not evidence_dicts:
        _append_runtime_trace(
            "retrieval.fallback.disabled",
            reason="SELF_AI_DISABLE_LEGACY_RETRIEVAL=true",
            content_type=content_type,
            query_preview=normalized_query[:100],
        )
    if allow_web_fallback and not evidence_dicts and content_type != "code":
        used_web_fallback = True
        try:
            web_result = tavily.search(query=normalized_query, max_results=5)
            contents = _extract_web_contents(web_result)
            if contents:
                await _index_web_result(contents[0])
                evidence_dicts = await _search_plan_collections(
                    dense_query=dense_query,
                    collections=plan.collections,
                    limit=plan.top_k,
                    min_score=plan.min_score,
                    collection_weights=plan.collection_weights,
                )
        except Exception as exc:
            trace(
                "retrieval.web_fallback.error",
                error=type(exc).__name__,
                message=str(exc)[:200],
            )

    graph_paths: list[Any] = []
    graph_plan: GraphQueryPlan
    graph_enabled = bool(
        settings.graph_read_enabled
        and graph_query_adapter.enabled
        and profile.retrieval_policy.use_graph
    )
    if graph_enabled:
        graph_plan = graph_query_adapter.plan_graph_queries(
            query=normalized_query,
            task_profile=profile,
            retrieval_policy=profile.retrieval_policy,
            evidence_items=evidence_dicts,
        )
    else:
        graph_plan = GraphQueryPlan(
            enabled=False,
            rationale="graph read disabled or retrieval_policy.use_graph is false",
            max_paths=max(1, int(settings.graph_max_paths)),
        )

    _append_runtime_trace(
        "retrieval.graph_query_planned",
        intent=profile.intent.value,
        enabled=graph_plan.enabled,
        planned_queries=list(graph_plan.query_types),
        symbols_count=len(graph_plan.symbols),
    )

    if graph_plan.enabled and graph_plan.query_types:
        raw_graph_paths: list[Any] = []
        for query_type in graph_plan.query_types:
            start_ms = now_ms()
            error_type: str | None = None
            path_count = 0
            _append_runtime_trace(
                "retrieval.graph_query.start",
                intent=profile.intent.value,
                enabled=True,
                query_type=query_type,
                symbols_count=len(graph_plan.symbols),
            )
            try:
                partial_plan = graph_plan.model_copy(update={"query_types": [query_type]})
                partial_paths = graph_query_adapter.run_graph_queries(partial_plan)
                path_count = len(partial_paths)
                raw_graph_paths.extend(partial_paths)
            except Exception as exc:
                error_type = type(exc).__name__
            _append_runtime_trace(
                "retrieval.graph_query.end",
                intent=profile.intent.value,
                enabled=True,
                query_type=query_type,
                path_count=path_count,
                elapsed_ms=now_ms() - start_ms,
                error_type=error_type,
            )
        graph_paths = _dedup_graph_paths(raw_graph_paths, graph_plan.max_paths)

    context_pack = context_builder.build(
        query=normalized_query,
        profile=profile,
        retrieval_policy=profile.retrieval_policy,
        evidence_items=evidence_dicts,
        graph_paths=graph_paths,
    )

    graph_summary = context_pack.get("graph_summary", {})
    _append_runtime_trace(
        "retrieval.graph_context_built",
        intent=profile.intent.value,
        enabled=graph_plan.enabled,
        path_count=graph_summary.get("path_count", 0),
        relation_type_counts=graph_summary.get("relation_type_counts", {}),
        context_preview_len=len(str(context_pack.get("graph_context_preview", ""))),
    )

    summary = {
        "result_count": context_pack.get("selected_evidence_count", 0),
        "collections": plan.collections,
        "source_type_counts": context_pack.get("source_type_counts", {}),
        "score_summary": context_pack.get("score_summary", {}),
        "context_preview": context_pack.get("context_preview", ""),
        "used_web_fallback": used_web_fallback,
        "graph_path_count": graph_summary.get("path_count", 0),
        "graph_relation_type_counts": graph_summary.get("relation_type_counts", {}),
    }
    _append_runtime_trace("retrieval.context_built", **summary)
    _save_runtime_retrieval_output(summary)

    return list(context_pack.get("evidence", []))
