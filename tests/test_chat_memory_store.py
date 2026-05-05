# coding=utf-8
"""Tests for Phase-1 chat memory store."""

from __future__ import annotations

import json
from pathlib import Path

from self_ai.memory.chat_memory_store import ChatMemoryStore


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_append_turn_creates_manifest_shard_and_summary(tmp_path: Path) -> None:
    store = ChatMemoryStore(root_dir=tmp_path / "memory" / "chats")
    out = store.append_turn(
        chat_id="chat-A",
        user_id="u1",
        run_id="run-1",
        task="请创建一个 hello.py 并写入打印语句",
        result={
            "response": "已创建文件 hello.py",
            "errors": [],
            "selected_agents": ["coder"],
            "metadata": {
                "execution_state": {"side_effect_done": True, "goal_aligned_side_effect": True},
                "recent_tool_results": [{"ok": True}, {"ok": True}],
            },
            "quality_gate": {"decision": "pass"},
            "review_reports": [],
        },
        logs=[{"event": "engine.run.end"}],
    )

    chat_dir = tmp_path / "memory" / "chats" / "chat-A"
    manifest_path = chat_dir / "manifest.json"
    summary_path = chat_dir / "summary-v1.json"
    shard_path = chat_dir / out["shard_file"]

    assert manifest_path.exists()
    assert summary_path.exists()
    assert shard_path.exists()
    assert out["turn_id"] == 1
    assert out["memory_id"] == "chat-A-turn-1"
    assert out["retrieval_summary"]

    manifest = _read_json(manifest_path)
    assert manifest["chat_id"] == "chat-A"
    assert manifest["next_turn_id"] == 2
    assert len(manifest["shards"]) == 1
    assert manifest["latest_summary"]

    line = shard_path.read_text(encoding="utf-8").strip().splitlines()[0]
    payload = json.loads(line)
    assert payload["chat_id"] == "chat-A"
    assert payload["turn_id"] == 1
    assert payload["retrieval_summary"]
    assert payload["execution"]["goal_aligned_side_effect"] is True


def test_shard_rolls_by_turn_limit(tmp_path: Path) -> None:
    store = ChatMemoryStore(
        root_dir=tmp_path / "memory" / "chats",
        max_shard_bytes=16,
    )
    common = {
        "user_id": "u2",
        "task": "记录一次运行",
        "result": {"response": "ok", "errors": [], "metadata": {}},
    }
    out1 = store.append_turn(chat_id="chat-roll", run_id="run-1", logs=[], **common)
    out2 = store.append_turn(chat_id="chat-roll", run_id="run-2", logs=[], **common)

    assert out1["shard_file"] != out2["shard_file"]
    manifest = _read_json(tmp_path / "memory" / "chats" / "chat-roll" / "manifest.json")
    assert len(manifest["shards"]) == 2
    assert manifest["next_turn_id"] == 3


def test_manifest_read_replay_and_recent_loader(tmp_path: Path) -> None:
    store = ChatMemoryStore(root_dir=tmp_path / "memory" / "chats")
    for idx in range(1, 5):
        store.append_turn(
            chat_id="chat-replay",
            user_id="u3",
            run_id=f"run-{idx}",
            task=f"task-{idx}",
            result={"response": f"resp-{idx}", "errors": [], "metadata": {}},
            logs=[],
        )

    manifest = store.read_manifest(chat_id="chat-replay")
    assert manifest["chat_id"] == "chat-replay"
    assert manifest["next_turn_id"] == 5
    assert len(manifest["shards"]) >= 1
    assert store.list_shards(chat_id="chat-replay")

    replay = store.replay_turns(chat_id="chat-replay", start_turn_id=2, end_turn_id=3, limit=10)
    assert [x["turn_id"] for x in replay] == [2, 3]
    assert all(x["retrieval_summary"] for x in replay)

    recent = store.load_recent_turns(chat_id="chat-replay", last_n=2)
    assert [x["turn_id"] for x in recent] == [3, 4]
    assert recent[-1]["response_preview"].startswith("resp-4")


def test_phase2_trusted_write_suppresses_issue_for_control_event_only(tmp_path: Path) -> None:
    store = ChatMemoryStore(root_dir=tmp_path / "memory" / "chats")
    out = store.append_turn(
        chat_id="chat-phase2-a",
        user_id="u4",
        run_id="run-p2-a",
        task="请创建 a.txt 并写入 hello",
        result={
            "response": "已完成",
            "errors": [],
            "metadata": {
                "goal_contract": {"requires_side_effect": True},
                "control_events": [
                    {
                        "type": "CompletionGateBlocked",
                        "stage": "mainloop",
                        "reason": "missing proof",
                    }
                ],
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
    assert out["write_policy"]["issue_write_allowed"] is False
    assert out["write_policy"]["issue_suppressed_due_to_terminal_success"] is True
    assert out["committed_memory_count"] == 0


def test_phase2_trusted_write_commits_issue_for_terminal_error(tmp_path: Path) -> None:
    store = ChatMemoryStore(root_dir=tmp_path / "memory" / "chats")
    out = store.append_turn(
        chat_id="chat-phase2-terminal",
        user_id="u4",
        run_id="run-p2-terminal",
        task="Create a.txt with hello",
        result={
            "response": "Execution is not verified",
            "errors": [
                {
                    "type": "CompletionNotVerifiedTerminal",
                    "message": "missing proof",
                    "stage": "mainloop",
                }
            ],
            "metadata": {
                "goal_contract": {"requires_side_effect": True},
                "recent_tool_results": [],
            },
        },
        logs=[],
    )
    assert out["write_policy"]["terminal_error_count"] == 1
    assert out["write_policy"]["issue_write_allowed"] is True
    assert out["committed_memory_count"] == 1
    assert out["committed_memory_types"] == ["issue"]


def test_phase2_trusted_write_commits_fact_with_state_change_proof(tmp_path: Path) -> None:
    store = ChatMemoryStore(root_dir=tmp_path / "memory" / "chats")
    out = store.append_turn(
        chat_id="chat-phase2-b",
        user_id="u5",
        run_id="run-p2-b",
        task="请创建 b.txt 并写入 hello",
        result={
            "response": "已创建 b.txt",
            "errors": [],
            "metadata": {
                "goal_contract": {"requires_side_effect": True},
                "recent_tool_results": [
                    {
                        "tool_name": "workspace.file.write",
                        "ok": True,
                        "execution_flag": {"ok": True, "state_change_committed": True},
                        "goal_check": {"aligned": True},
                    }
                ],
            },
            "quality_gate": {"decision": "pass"},
        },
        logs=[],
    )
    assert out["write_policy"]["requires_state_change"] is True
    assert out["write_policy"]["fact_write_allowed"] is True
    assert out["committed_memory_count"] >= 2


def test_phase22b_memory_decision_overrides_profile(tmp_path: Path) -> None:
    store = ChatMemoryStore(root_dir=tmp_path / "memory" / "chats")
    out = store.append_turn(
        chat_id="chat-phase22b",
        user_id="u6",
        run_id="run-p22b",
        task="记住这是长期偏好",
        result={
            "response": "收到",
            "errors": [],
            "metadata": {
                "goal_contract": {"requires_side_effect": False},
                "recent_tool_results": [{"tool_name": "workspace.file.read", "ok": True}],
            },
        },
        logs=[],
        memory_decision={
            "profiles": {
                "turn_summary": {
                    "memory_class": "preference",
                    "importance": 0.92,
                    "decay_mode": "none",
                    "pinned": True,
                    "confidence": 0.93,
                    "decision_source": "model_sidecar_v1",
                }
            }
        },
    )
    assert out["committed_memory_count"] >= 1
    shard = tmp_path / "memory" / "chats" / "chat-phase22b" / out["shard_file"]
    row = json.loads(shard.read_text(encoding="utf-8").splitlines()[-1])
    assert row["memory_profile"]["primary_class"] == "preference"
    assert row["memory_candidates"][0]["decision_source"] == "model_sidecar_v1"


def test_global_constraints_are_session_scoped_and_latest_first(tmp_path: Path) -> None:
    store = ChatMemoryStore(root_dir=tmp_path / "memory" / "chats")

    first = store.apply_global_constraint_ops(
        chat_id="chat-gc-a",
        ops=[
            {"op": "add", "content": "Prefer concise answers."},
            {"op": "add", "content": "Use TypeScript for source files."},
        ],
    )
    other = store.apply_global_constraint_ops(
        chat_id="chat-gc-b",
        ops=[{"op": "add", "content": "Use Python for examples."}],
    )

    a_items = store.read_global_constraints(chat_id="chat-gc-a")["items"]
    b_items = store.read_global_constraints(chat_id="chat-gc-b")["items"]

    assert first["changed"] is True
    assert other["changed"] is True
    assert [x["content"] for x in a_items] == [
        "Use TypeScript for source files.",
        "Prefer concise answers.",
    ]
    assert [x["content"] for x in b_items] == ["Use Python for examples."]
    assert (tmp_path / "memory" / "chats" / "chat-gc-a" / "global_constraints.json").exists()
    assert (tmp_path / "memory" / "chats" / "chat-gc-b" / "global_constraints.json").exists()


def test_global_constraint_update_moves_item_to_front_and_delete_removes(tmp_path: Path) -> None:
    store = ChatMemoryStore(root_dir=tmp_path / "memory" / "chats")
    store.apply_global_constraint_ops(
        chat_id="chat-gc-edit",
        ops=[
            {"op": "add", "content": "Prefer short answers."},
            {"op": "add", "content": "Use Markdown lists."},
        ],
    )
    before = store.read_global_constraints(chat_id="chat-gc-edit")["items"]
    old_id = before[-1]["id"]
    deleted_id = before[0]["id"]

    store.apply_global_constraint_ops(
        chat_id="chat-gc-edit",
        ops=[
            {"op": "update", "id": old_id, "content": "Prefer concise answers with the conclusion first."},
            {"op": "delete", "id": deleted_id},
        ],
    )

    after = store.read_global_constraints(chat_id="chat-gc-edit")["items"]
    assert [x["content"] for x in after] == ["Prefer concise answers with the conclusion first."]
