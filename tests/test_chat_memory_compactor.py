# coding=utf-8
"""Tests for Phase-4 chat memory compactor."""

from __future__ import annotations

from pathlib import Path

from self_ai.memory.chat_memory_compactor import compact_chat
from self_ai.memory.chat_memory_store import ChatMemoryStore


def _mk_turn(store: ChatMemoryStore, chat_id: str, idx: int) -> None:
    store.append_turn(
        chat_id=chat_id,
        user_id="u",
        run_id=f"run-{idx}",
        task=f"task-{idx}",
        result={"response": f"resp-{idx}", "errors": [], "metadata": {}},
        logs=[],
    )


def test_compact_chat_creates_summary_and_archive_index(tmp_path: Path) -> None:
    store = ChatMemoryStore(
        root_dir=tmp_path / "memory" / "chats",
        max_shard_bytes=8,  # force shard roll quickly
        max_turns_per_shard=1,
    )
    chat_id = "chat-compact"
    for i in range(1, 6):
        _mk_turn(store, chat_id, i)

    report = compact_chat(
        store=store,
        chat_id=chat_id,
        min_shards=3,
        summary_max_chars=300,
        max_docs=20,
    )
    assert report["compacted"] is True
    assert report["archived_count"] >= 1
    assert str(report["summary_file"]).startswith("summary-v")

    manifest = store.read_manifest(chat_id=chat_id)
    assert isinstance(manifest["archived_shards"], dict)
    assert len(manifest["archived_shards"]) >= 1
    assert len(manifest["summary_versions"]) >= 1
    assert int(manifest["last_compacted_turn_id"]) >= 1
    summary = store.read_summary(chat_id=chat_id)
    assert summary["chat_id"] == chat_id
    assert len(summary.get("blocks", [])) >= 1


def test_compact_chat_no_candidates_is_not_fake_success(tmp_path: Path) -> None:
    store = ChatMemoryStore(root_dir=tmp_path / "memory" / "chats")
    chat_id = "chat-no-compact"
    _mk_turn(store, chat_id, 1)
    _mk_turn(store, chat_id, 2)

    report = compact_chat(
        store=store,
        chat_id=chat_id,
        min_shards=5,
        summary_max_chars=300,
        max_docs=20,
    )
    assert report["compacted"] is False
    assert report["reason"] == "no_candidates"
