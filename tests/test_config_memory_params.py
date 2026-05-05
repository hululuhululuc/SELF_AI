# coding=utf-8
"""Validation tests for chat-memory injection parameter boundaries."""

from __future__ import annotations

import pytest

from self_ai.config import PROJECT_ROOT, Settings, resolve_project_path


def test_chat_memory_injection_defaults_are_within_prompt_budget() -> None:
    s = Settings()
    dynamic_budget = int(s.prompt_max_chars_mainloop * s.prompt_dynamic_budget_ratio)
    assert s.chat_memory_injection_max_chars <= int(dynamic_budget * 0.9)
    assert s.chat_memory_injection_item_max_chars <= s.chat_memory_injection_max_chars
    assert s.prompt_max_chars_mainloop >= 180000
    assert s.prompt_dynamic_budget_ratio >= 0.6
    assert s.prompt_recent_tool_results >= 24
    assert s.chat_memory_recent_turns >= 16
    assert s.chat_memory_injection_max_chars >= 96000
    assert s.chat_memory_injection_item_max_chars >= 6000
    assert s.chat_memory_summary_max_chars >= 4000


def test_chat_memory_injection_item_cannot_exceed_total() -> None:
    with pytest.raises(ValueError):
        Settings(
            chat_memory_injection_max_chars=800,
            chat_memory_injection_item_max_chars=900,
        )


def test_chat_memory_injection_total_cannot_exceed_dynamic_budget() -> None:
    with pytest.raises(ValueError):
        Settings(
            prompt_max_chars_mainloop=5000,
            prompt_dynamic_budget_ratio=0.35,
            chat_memory_injection_max_chars=5000,
        )


def test_strict_runtime_defaults_enabled() -> None:
    s = Settings(_env_file=None)
    assert s.qdrant_strict_schema is True
    assert s.disable_legacy_retrieval is True
    assert s.embedder_prewarm is False


def test_relative_runtime_paths_resolve_from_project_root() -> None:
    assert resolve_project_path("./memory/chats") == (PROJECT_ROOT / "memory" / "chats").resolve()
    assert resolve_project_path("./self_ai/model/bge-m3") == (PROJECT_ROOT / "self_ai" / "model" / "bge-m3").resolve()
