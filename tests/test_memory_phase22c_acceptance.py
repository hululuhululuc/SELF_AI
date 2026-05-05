# coding=utf-8
"""Phase 2.2-C acceptance tests."""

from __future__ import annotations

import json
from pathlib import Path

from self_ai.memory.chat_memory_service import ChatMemoryService
from self_ai.memory.chat_memory_store import ChatMemoryStore


def _patch_profile(
    shard_path: Path,
    *,
    turn_id: int,
    memory_profile: dict,
    timestamp: str | None = None,
) -> None:
    rows = [json.loads(x) for x in shard_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    for row in rows:
        if int(row.get("turn_id", 0) or 0) == turn_id:
            row["memory_profile"] = memory_profile
            if timestamp is not None:
                row["timestamp"] = timestamp
    shard_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in rows) + "\n",
        encoding="utf-8",
    )


def test_acceptance_decay_old_memory_weight_drops(tmp_path: Path) -> None:
    store = ChatMemoryStore(root_dir=tmp_path / "memory" / "chats")
    service = ChatMemoryService(store)
    out1 = store.append_turn(
        chat_id="chat-a",
        user_id="u",
        run_id="r1",
        task="old task",
        result={"response": "old response", "errors": [], "metadata": {}},
        logs=[],
    )
    out2 = store.append_turn(
        chat_id="chat-a",
        user_id="u",
        run_id="r2",
        task="new task",
        result={"response": "new response", "errors": [], "metadata": {}},
        logs=[],
    )
    shard = tmp_path / "memory" / "chats" / "chat-a" / out2["shard_file"]
    _patch_profile(
        shard,
        turn_id=1,
        timestamp="2020-01-01T00:00:00+00:00",
        memory_profile={
            "primary_class": "fact",
            "importance": 0.55,
            "decay_mode": "fast",
            "pinned": False,
            "confidence": 0.7,
        },
    )
    _patch_profile(
        shard,
        turn_id=2,
        memory_profile={
            "primary_class": "fact",
            "importance": 0.55,
            "decay_mode": "normal",
            "pinned": False,
            "confidence": 0.7,
        },
    )
    payload = service.load_recent_turns_for_injection(
        chat_id="chat-a",
        last_n=2,
        max_chars_total=2800,
        max_item_chars=220,
    )
    assert payload["kept_turns"] >= 2
    assert payload["turns"][0]["turn_id"] == 2
    assert payload["turns"][0]["score"]["final_score"] >= payload["turns"][1]["score"]["final_score"]


def test_acceptance_preference_pinned_stays_high_weight(tmp_path: Path) -> None:
    store = ChatMemoryStore(root_dir=tmp_path / "memory" / "chats")
    service = ChatMemoryService(store)
    out1 = store.append_turn(
        chat_id="chat-b",
        user_id="u",
        run_id="r1",
        task="用户偏好 Python 风格",
        result={"response": "记住偏好", "errors": [], "metadata": {}},
        logs=[],
    )
    out2 = store.append_turn(
        chat_id="chat-b",
        user_id="u",
        run_id="r2",
        task="普通任务",
        result={"response": "普通回复", "errors": [], "metadata": {}},
        logs=[],
    )
    shard = tmp_path / "memory" / "chats" / "chat-b" / out2["shard_file"]
    _patch_profile(
        shard,
        turn_id=1,
        timestamp="2021-01-01T00:00:00+00:00",
        memory_profile={
            "primary_class": "preference",
            "importance": 0.95,
            "decay_mode": "none",
            "pinned": True,
            "confidence": 0.95,
        },
    )
    _patch_profile(
        shard,
        turn_id=2,
        memory_profile={
            "primary_class": "ephemeral",
            "importance": 0.4,
            "decay_mode": "fast",
            "pinned": False,
            "confidence": 0.6,
        },
    )
    payload = service.load_recent_turns_for_injection(
        chat_id="chat-b",
        last_n=2,
        max_chars_total=1400,
        max_item_chars=220,
    )
    assert payload["turns"][0]["turn_id"] == 1
    assert payload["turns"][0]["score"]["pinned"] is True
    assert payload["turns"][0]["score"]["decay"] == 1.0


def test_acceptance_oral_completion_without_evidence_not_fact_committed(tmp_path: Path) -> None:
    store = ChatMemoryStore(root_dir=tmp_path / "memory" / "chats")
    out = store.append_turn(
        chat_id="chat-c",
        user_id="u",
        run_id="r1",
        task="请创建 c.txt 并写入 hello",
        result={
            "response": "已完成",
            "errors": [],
            "metadata": {
                "goal_contract": {"requires_side_effect": True},
                "recent_tool_results": [
                    {
                        "tool_name": "workspace.file.write",
                        "ok": False,
                        "execution_flag": {"ok": False, "state_change_committed": False},
                    }
                ],
            },
        },
        logs=[],
    )
    assert out["write_policy"]["requires_state_change"] is True
    assert out["write_policy"]["fact_write_allowed"] is False
    assert "response_summary" not in out["committed_memory_types"]
    assert "turn_summary" not in out["committed_memory_types"]
