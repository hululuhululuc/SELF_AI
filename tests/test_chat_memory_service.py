# coding=utf-8
"""Tests for chat memory service layer APIs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from self_ai.memory.chat_memory_service import ChatMemoryService
from self_ai.memory.chat_memory_store import ChatMemoryStore


def test_service_replay_and_manifest(tmp_path: Path) -> None:
    store = ChatMemoryStore(root_dir=tmp_path / "memory" / "chats")
    service = ChatMemoryService(store)
    for idx in range(1, 4):
        store.append_turn(
            chat_id="chat-svc",
            user_id="u",
            run_id=f"run-{idx}",
            task=f"task-{idx}",
            result={"response": f"resp-{idx}", "errors": [], "metadata": {}},
            logs=[],
        )

    manifest = service.get_manifest(chat_id="chat-svc")
    assert manifest["chat_id"] == "chat-svc"
    assert manifest["next_turn_id"] == 4

    turns = service.replay_turns(chat_id="chat-svc", start_turn_id=2, end_turn_id=3)
    assert [x["turn_id"] for x in turns] == [2, 3]


def test_service_recent_turns_budget_control(tmp_path: Path) -> None:
    store = ChatMemoryStore(root_dir=tmp_path / "memory" / "chats")
    service = ChatMemoryService(store)
    long_text = "A" * 800
    for idx in range(1, 6):
        store.append_turn(
            chat_id="chat-budget",
            user_id="u",
            run_id=f"run-{idx}",
            task=long_text,
            result={"response": long_text, "errors": [], "metadata": {}},
            logs=[],
        )

    payload = service.load_recent_turns_for_injection(
        chat_id="chat-budget",
        last_n=5,
        max_chars_total=700,
        max_item_chars=120,
    )
    assert payload["chat_id"] == "chat-budget"
    assert payload["requested_turns"] == 5
    assert payload["kept_turns"] >= 1
    assert payload["budget"]["used_chars"] <= 700
    assert len(payload["turns"]) == payload["kept_turns"]
    assert payload["dropped_turns"] >= 0
    assert payload["turns"][0].get("score")
    assert isinstance(payload["turns"][0].get("memory_profile", {}), dict)


def test_manifest_self_check_repair(tmp_path: Path) -> None:
    store = ChatMemoryStore(root_dir=tmp_path / "memory" / "chats")
    chat_dir = tmp_path / "memory" / "chats" / "chat-repair"
    chat_dir.mkdir(parents=True, exist_ok=True)
    shard = chat_dir / "turns-20260429-01.ndjson"
    shard.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "memory_id": "m1",
                        "chat_id": "chat-repair",
                        "run_id": "r1",
                        "turn_id": 1,
                        "retrieval_summary": "s1",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "memory_id": "m2",
                        "chat_id": "chat-repair",
                        "run_id": "r2",
                        "turn_id": 2,
                        "retrieval_summary": "s2",
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    # broken manifest: wrong next_turn_id, missing active shard, broken chat_id.
    (chat_dir / "manifest.json").write_text(
        json.dumps(
            {
                "chat_id": "wrong",
                "next_turn_id": 1,
                "active_shard_file": "missing.ndjson",
                "shards": [],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    report = store.repair_manifest(chat_id="chat-repair")
    assert report["changed"] is True
    assert report["next_turn_id"] == 3
    assert report["active_shard_file"] == "turns-20260429-01.ndjson"
    manifest = store.read_manifest(chat_id="chat-repair")
    assert manifest["chat_id"] == "chat-repair"
    assert manifest["next_turn_id"] == 3
    assert len(manifest["shards"]) == 1


def test_decay_scoring_prefers_pinned_over_old_memory(tmp_path: Path) -> None:
    store = ChatMemoryStore(root_dir=tmp_path / "memory" / "chats")
    service = ChatMemoryService(store)
    out1 = store.append_turn(
        chat_id="chat-decay",
        user_id="u",
        run_id="run-1",
        task="task-old",
        result={"response": "old", "errors": [], "metadata": {}},
        logs=[],
    )
    out2 = store.append_turn(
        chat_id="chat-decay",
        user_id="u",
        run_id="run-2",
        task="task-new",
        result={"response": "new", "errors": [], "metadata": {}},
        logs=[],
    )
    shard_path = tmp_path / "memory" / "chats" / "chat-decay" / out2["shard_file"]
    lines = [json.loads(x) for x in shard_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    # Make turn-1 very old and unpinned.
    lines[0]["timestamp"] = "2020-01-01T00:00:00+00:00"
    lines[0]["memory_profile"] = {
        "primary_class": "fact",
        "importance": 0.4,
        "decay_mode": "fast",
        "pinned": False,
        "confidence": 0.6,
    }
    # Make turn-2 pinned.
    lines[1]["memory_profile"] = {
        "primary_class": "preference",
        "importance": 0.9,
        "decay_mode": "none",
        "pinned": True,
        "confidence": 0.9,
    }
    shard_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in lines) + "\n",
        encoding="utf-8",
    )

    payload = service.load_recent_turns_for_injection(
        chat_id="chat-decay",
        last_n=2,
        max_chars_total=1200,
        max_item_chars=180,
    )
    assert payload["kept_turns"] >= 1
    top = payload["turns"][0]
    assert top["turn_id"] == 2
    assert top["score"]["pinned"] is True


@pytest.mark.asyncio
async def test_retrieve_memory_for_turn_fuses_l1_l2_l3(tmp_path: Path) -> None:
    store = ChatMemoryStore(root_dir=tmp_path / "memory" / "chats")
    service = ChatMemoryService(store)
    store.append_turn(
        chat_id="chat-fusion",
        user_id="u",
        run_id="run-fusion-1",
        task="create demo.py",
        result={"response": "created demo.py", "errors": [], "metadata": {}},
        logs=[],
    )

    class _FakeKernel:
        async def run_once(self, *, tool_name, **_kwargs):
            if tool_name == "storage.semantic.search":
                return {
                    "ok": True,
                    "data": {
                        "evidence": [
                            {
                                "content": "User prefers concise python style.",
                                "source_type": "chat_memory:preference",
                                "chunk_id": "c1",
                                "score": 0.82,
                                "metadata": {"chat_id": "chat-fusion", "run_id": "run-fusion-1"},
                            }
                        ]
                    },
                }
            if tool_name == "storage.graph.find_task_lineage":
                return {
                    "ok": True,
                    "data": {
                        "paths": [
                            {
                                "nodes": [{"id": "task:1", "label": "Task"}],
                                "edges": [],
                                "metadata": {"run_id": "run-fusion-1"},
                            }
                        ]
                    },
                }
            return {"ok": False, "error": {"type": "UnknownTool", "message": tool_name}}

    payload = await service.retrieve_memory_for_turn(
        chat_id="chat-fusion",
        query="what did we do before?",
        last_n=4,
        max_chars_total=1800,
        max_item_chars=220,
        kernel=_FakeKernel(),
        session_id="chat-fusion",
        encode_func=lambda _q: [0.1, 0.2, 0.3],
    )
    assert payload["chat_id"] == "chat-fusion"
    assert payload["kept_turns"] >= 1
    assert isinstance(payload["source_type_groups"], dict)
    assert payload["source_type_groups"].get("L1", 0) >= 1
    assert payload["source_type_groups"].get("L2", 0) >= 1
    assert payload["source_type_groups"].get("L3", 0) >= 1


def test_service_compact_chat_memory_and_summary_read(tmp_path: Path) -> None:
    store = ChatMemoryStore(
        root_dir=tmp_path / "memory" / "chats",
        max_shard_bytes=8,
        max_turns_per_shard=1,
    )
    service = ChatMemoryService(store)
    for idx in range(1, 5):
        store.append_turn(
            chat_id="chat-comp-svc",
            user_id="u",
            run_id=f"run-{idx}",
            task=f"task-{idx}",
            result={"response": f"resp-{idx}", "errors": [], "metadata": {}},
            logs=[],
        )
    candidates = service.list_compaction_candidates(chat_id="chat-comp-svc", min_shards=3)
    assert len(candidates) >= 1
    report = service.compact_chat_memory(
        chat_id="chat-comp-svc",
        min_shards=3,
        summary_max_chars=260,
        max_docs=20,
    )
    assert report["compacted"] is True
    summary = service.read_summary(chat_id="chat-comp-svc")
    assert summary["chat_id"] == "chat-comp-svc"
    assert len(summary.get("blocks", [])) >= 1
