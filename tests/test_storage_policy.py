# coding=utf-8
"""Unit tests for storage policy builders."""

from self_ai.schemas import TaskIntent, TaskProfile
from self_ai.storage_policy import (
    build_memory_write_policy,
    build_retrieval_policy,
    enrich_profile_with_storage_policies,
)


def test_architecture_design_memory_policy() -> None:
    profile = TaskProfile(intent=TaskIntent.ARCHITECTURE_DESIGN)
    policy = build_memory_write_policy(profile)
    assert policy.write_redis
    assert policy.write_qdrant
    assert policy.write_neo4j
    assert policy.qdrant_collection == "cf_task_memory"


def test_debugging_policies() -> None:
    profile = TaskProfile(intent=TaskIntent.DEBUGGING)
    mem = build_memory_write_policy(profile)
    ret = build_retrieval_policy(profile)
    assert mem.qdrant_collection == "cf_review_memory"
    assert mem.neo4j_label == "Issue"
    assert ret.use_code_index
    assert ret.use_graph


def test_research_summary_retrieval_policy() -> None:
    profile = TaskProfile(intent=TaskIntent.RESEARCH_SUMMARY)
    ret = build_retrieval_policy(profile)
    assert ret.use_web
    assert "cf_doc_chunks" in ret.qdrant_collections
    assert "cf_web_chunks" in ret.qdrant_collections


def test_general_qa_memory_policy() -> None:
    profile = TaskProfile(intent=TaskIntent.GENERAL_QA)
    mem = build_memory_write_policy(profile)
    assert mem.write_redis
    assert not mem.write_qdrant
    assert not mem.write_neo4j


def test_enrich_profile_with_policies() -> None:
    profile = TaskProfile(intent=TaskIntent.PLANNING)
    enriched = enrich_profile_with_storage_policies(profile)
    assert enriched.memory_write_policy.write_redis
    assert enriched.retrieval_policy.use_vector
