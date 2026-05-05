# coding=utf-8
"""Tests for retrieval query planner mappings."""

from unittest.mock import patch

from self_ai.retrieval.query_planner import build_query_plan
from self_ai.schemas import TaskIntent, TaskProfile
from self_ai.storage_policy import enrich_profile_with_storage_policies


def _profile(intent: TaskIntent) -> TaskProfile:
    return enrich_profile_with_storage_policies(TaskProfile(intent=intent))


def test_query_planner_code_generation_collections() -> None:
    plan = build_query_plan(_profile(TaskIntent.CODE_GENERATION))
    assert plan.collections[:2] == ["cf_code_chunks", "cf_doc_chunks"]


def test_query_planner_debugging_collections() -> None:
    plan = build_query_plan(_profile(TaskIntent.DEBUGGING))
    assert "cf_code_chunks" in plan.collections
    assert "cf_review_memory" in plan.collections


def test_query_planner_architecture_collections() -> None:
    plan = build_query_plan(_profile(TaskIntent.ARCHITECTURE_DESIGN))
    assert plan.collections == ["cf_task_memory", "cf_doc_chunks", "cf_review_memory"]


def test_query_planner_research_collections() -> None:
    plan = build_query_plan(_profile(TaskIntent.RESEARCH_SUMMARY))
    assert plan.collections == ["cf_doc_chunks", "cf_web_chunks"]


def test_query_planner_document_writing_collections() -> None:
    plan = build_query_plan(_profile(TaskIntent.DOCUMENT_WRITING))
    assert plan.collections == ["cf_doc_chunks", "cf_task_memory"]


def test_query_planner_does_not_use_legacy_keyword_helpers() -> None:
    profile = _profile(TaskIntent.GENERAL_QA)
    with patch("self_ai.text_utils.is_coding_task", side_effect=AssertionError("must not call")):
        with patch("self_ai.text_utils.is_complex_task", side_effect=AssertionError("must not call")):
            plan = build_query_plan(profile)
    assert plan.collections == ["cf_doc_chunks"]
