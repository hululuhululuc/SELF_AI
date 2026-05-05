# coding=utf-8
"""Tests for context builder dedup/sort/group/truncate behavior."""

from self_ai.retrieval.context_builder import build_context_pack
from self_ai.schemas import EvidenceItem, RetrievalPolicy, TaskIntent, TaskProfile


def test_context_builder_dedup_sort_and_group() -> None:
    profile = TaskProfile(intent=TaskIntent.RESEARCH_SUMMARY)
    policy = RetrievalPolicy(max_context_chars=500)
    items = [
        EvidenceItem(
            content="duplicate low score",
            source_type="doc",
            chunk_id="dup-1",
            score=0.2,
        ),
        EvidenceItem(
            content="duplicate high score",
            source_type="doc",
            chunk_id="dup-1",
            score=0.9,
        ),
        EvidenceItem(
            content="web evidence content that is intentionally long for truncation",
            source_type="web",
            chunk_id="web-1",
            score=0.8,
        ),
    ]

    pack = build_context_pack(
        query="test query",
        profile=profile,
        retrieval_policy=policy,
        evidence_items=items,
    )

    assert pack["total_evidence"] == 3
    assert pack["selected_evidence_count"] == 2
    assert pack["evidence"][0]["content"] == "duplicate high score"
    assert pack["source_type_counts"]["doc"] == 1
    assert pack["source_type_counts"]["web"] == 1
    assert len(pack["context"]) <= 500
    assert pack["max_context_chars"] == 500


def test_context_builder_truncates_by_max_context_chars() -> None:
    profile = TaskProfile(intent=TaskIntent.RESEARCH_SUMMARY)
    policy = RetrievalPolicy(max_context_chars=40)
    items = [
        EvidenceItem(
            content="long content for truncation case",
            source_type="doc",
            chunk_id="x1",
            score=0.9,
        )
    ]

    pack = build_context_pack(
        query="truncate query",
        profile=profile,
        retrieval_policy=policy,
        evidence_items=items,
    )

    assert len(pack["context"]) <= 40
    assert pack["max_context_chars"] == 40
