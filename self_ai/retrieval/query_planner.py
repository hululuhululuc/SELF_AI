# coding=utf-8
"""Retrieval query planning for Phase 3 semantic memory layer."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..schemas import RetrievalPolicy, TaskIntent, TaskProfile


class RetrievalPlan(BaseModel):
    """Executable retrieval plan derived from TaskProfile and policy."""

    intent: str
    collections: list[str] = Field(default_factory=list)
    top_k: int = 5
    min_score: float = 0.0
    use_vector: bool = True
    use_web: bool = False
    collection_weights: dict[str, float] = Field(default_factory=dict)


def _intent_to_collections(intent: str) -> list[str]:
    if intent == TaskIntent.CODE_GENERATION.value:
        return ["cf_code_chunks", "cf_doc_chunks"]
    if intent in {
        TaskIntent.DEBUGGING.value,
        TaskIntent.CODE_REVIEW.value,
        TaskIntent.CODE_MODIFICATION.value,
    }:
        return ["cf_code_chunks", "cf_review_memory"]
    if intent in {TaskIntent.ARCHITECTURE_DESIGN.value, TaskIntent.PLANNING.value}:
        return ["cf_task_memory", "cf_doc_chunks", "cf_review_memory"]
    if intent in {TaskIntent.RESEARCH_SUMMARY.value, TaskIntent.RAG_ANSWERING.value}:
        return ["cf_doc_chunks", "cf_web_chunks"]
    if intent == TaskIntent.DOCUMENT_WRITING.value:
        return ["cf_doc_chunks", "cf_task_memory"]
    return ["cf_doc_chunks"]


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def build_query_plan(
    profile: TaskProfile,
    retrieval_policy: RetrievalPolicy | None = None,
) -> RetrievalPlan:
    """Build a retrieval plan without any keyword-based legacy routing."""
    policy = retrieval_policy or profile.retrieval_policy
    intent = profile.intent.value if hasattr(profile.intent, "value") else str(profile.intent)

    mapped_collections = _intent_to_collections(intent)
    collections = _unique([*mapped_collections, *policy.qdrant_collections])

    return RetrievalPlan(
        intent=intent,
        collections=collections,
        top_k=max(1, int(policy.top_k_vector or 5)),
        min_score=float(policy.min_score),
        use_vector=bool(policy.use_vector),
        use_web=bool(policy.use_web),
        collection_weights=dict(policy.collection_weights),
    )


class QueryPlanner:
    """Small object wrapper to keep calling style explicit in tools."""

    def build(self, profile: TaskProfile, retrieval_policy: RetrievalPolicy | None = None) -> RetrievalPlan:
        return build_query_plan(profile, retrieval_policy=retrieval_policy)
