# coding=utf-8
"""MainLoop semantic tests for EngineLoop."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from self_ai.kernel import SelfAIKernel
from self_ai.kernel.engine_state import EngineState
from tests._engine_helpers import FakeGraphMemory, FakeQdrantMemory, FakeRedisStore


def _build_kernel(
    tmp_path: Path,
    *,
    route_model_func: Any,
    permission_profile: dict[str, bool] | None = None,
) -> tuple[SelfAIKernel, FakeRedisStore]:
    redis_store = FakeRedisStore()
    kernel = SelfAIKernel(
        project_root=tmp_path,
        permission_profile=permission_profile
        or {
            "read_only": True,
            "runtime_write": True,
            "memory_read": True,
            "memory_write": False,
            "workspace_read": True,
            "workspace_write": False,
            "shell_exec": False,
            "network": False,
            "model_call": True,
        },
        dependencies={
            "route_model_func": route_model_func,
            "redis_store": redis_store,
            "qdrant_memory": FakeQdrantMemory(),
            "graph_memory": FakeGraphMemory(),
            "web_search_func": lambda _query, _max_results: [],
            "project_root": str(tmp_path),
        },
    )
    return kernel, redis_store


@pytest.mark.asyncio
async def test_engine_loop_tool_call_then_final_answer(tmp_path: Path) -> None:
    (tmp_path / "demo.txt").write_text("demo", encoding="utf-8")

    async def _route_model(prompt: str, stage: str, hints: dict[str, Any] | None = None) -> dict[str, Any]:
        turn = int((hints or {}).get("turn_index", 0) or 0)
        if stage == "mainloop_decide" and turn == 1:
            payload = {
                "type": "tool_calls",
                "calls": [
                    {
                        "tool_name": "workspace.file.list",
                        "arguments": {"path": ".", "recursive": False, "max_entries": 50},
                        "reason": "Inspect workspace first",
                    }
                ],
                "execution_mode": "model_driven",
            }
            return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}

        payload = {
            "type": "final_answer",
            "response": "done",
            "execution_mode": "quick_answer",
        }
        return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}

    kernel, redis_store = _build_kernel(tmp_path, route_model_func=_route_model)
    state = EngineState.from_input("List files and then answer", run_id="r-loop-1", session_id="s-loop")
    ctx = kernel.new_run_context(run_id="r-loop-1", session_id="s-loop")

    final = await kernel.engine_loop.run(state, run_context=ctx)

    assert final.response == "done"
    assert final.model == "mock-fast"
    assert final.workflow_decision.get("execution_mode") in {"quick_answer", "model_driven"}

    node_names = [name for _, name, _ in redis_store.node_outputs]
    assert "mainloop_turn_1" in node_names
    assert "mainloop_final_answer" in node_names
    assert "mainloop_finalize" in node_names


@pytest.mark.asyncio
async def test_engine_loop_parse_failure_does_not_fake_final_answer(tmp_path: Path) -> None:
    async def _route_model(_prompt: str, stage: str, _hints: dict[str, Any] | None = None) -> dict[str, Any]:
        if stage == "mainloop_json_repair":
            return {"model": "mock-fast", "response": "still-not-json"}
        return {"model": "mock-fast", "response": "not-json-response"}

    kernel, _ = _build_kernel(tmp_path, route_model_func=_route_model)
    state = EngineState.from_input("fallback test", run_id="r-loop-2", session_id="s-loop")
    ctx = kernel.new_run_context(run_id="r-loop-2", session_id="s-loop")

    final = await kernel.engine_loop.run(state, run_context=ctx)

    assert final.response == "No response generated."
    assert any((err.get("type") or "").startswith("RuntimeError") for err in final.errors)


@pytest.mark.asyncio
async def test_engine_loop_emits_mainloop_trace_events(tmp_path: Path) -> None:
    async def _route_model(_prompt: str, _stage: str, _hints: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = {
            "type": "final_answer",
            "response": "ok",
            "execution_mode": "quick_answer",
        }
        return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}

    kernel, _ = _build_kernel(tmp_path, route_model_func=_route_model)
    state = EngineState.from_input("trace test", run_id="r-loop-3", session_id="s-loop")
    ctx = kernel.new_run_context(run_id="r-loop-3", session_id="s-loop")

    final = await kernel.engine_loop.run(state, run_context=ctx)
    events = [item.get("event") for item in final.trace]

    assert "engine.run.start" in events
    assert "engine.pipeline.selected" in events
    assert "prompt.mainloop.stats" in events
    assert "loop.turn.start" in events
    assert "loop.turn.end" in events
    assert "engine.run.end" in events


@pytest.mark.asyncio
async def test_engine_loop_blocks_unverified_completion_and_requires_tool_evidence(tmp_path: Path) -> None:
    async def _route_model(_prompt: str, stage: str, hints: dict[str, Any] | None = None) -> dict[str, Any]:
        turn = int((hints or {}).get("turn_index", 0) or 0)
        if stage == "mainloop_decide" and turn == 1:
            payload = {
                "type": "tool_calls",
                "calls": [
                    {
                        "tool_name": "workspace.file.write",
                        "arguments": {"path": "notes.txt", "content": "hello"},
                        "reason": "need real write evidence",
                    }
                ],
                "execution_mode": "model_driven",
            }
            return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}
        payload = {
            "type": "final_answer",
            "response": "created notes.txt with hello",
            "execution_mode": "model_driven",
        }
        return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}

    kernel, _ = _build_kernel(
        tmp_path,
        route_model_func=_route_model,
        permission_profile={
            "read_only": True,
            "runtime_write": True,
            "memory_read": True,
            "memory_write": False,
            "workspace_read": True,
            "workspace_write": True,
            "shell_exec": False,
            "network": False,
            "model_call": True,
        },
    )
    state = EngineState.from_input("Create notes.txt and write hello", run_id="r-loop-4", session_id="s-loop")
    ctx = kernel.new_run_context(run_id="r-loop-4", session_id="s-loop")

    final = await kernel.engine_loop.run(state, run_context=ctx)

    assert final.response.startswith("created notes.txt")
    assert (tmp_path / "notes.txt").exists()
    assert not any(item.get("event") == "mainloop.final_answer.blocked" for item in final.trace)


@pytest.mark.asyncio
async def test_engine_loop_blocks_final_answer_after_failed_tool_loop(tmp_path: Path) -> None:
    async def _route_model(_prompt: str, stage: str, hints: dict[str, Any] | None = None) -> dict[str, Any]:
        turn = int((hints or {}).get("turn_index", 0) or 0)
        if stage == "mainloop_decide" and turn == 1:
            payload = {
                "type": "tool_calls",
                "calls": [
                    {
                        "tool_name": "workspace.file.read",
                        "arguments": {"path": "missing.txt"},
                        "reason": "start evidence loop",
                    }
                ],
                "execution_mode": "model_driven",
            }
            return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}
        payload = {
            "type": "final_answer",
            "response": "read missing.txt",
            "execution_mode": "model_driven",
        }
        return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}

    kernel, _ = _build_kernel(tmp_path, route_model_func=_route_model)
    state = EngineState.from_input("Read missing.txt", run_id="r-loop-4-fail", session_id="s-loop")
    state.metadata["max_turns"] = 3
    ctx = kernel.new_run_context(run_id="r-loop-4", session_id="s-loop")

    final = await kernel.engine_loop.run(state, run_context=ctx)

    assert "Execution is not verified" in final.response
    assert any(item.get("event") == "mainloop.final_answer.blocked" for item in final.trace)


@pytest.mark.asyncio
async def test_engine_loop_allows_failed_outcome_with_failed_tool_evidence(tmp_path: Path) -> None:
    async def _route_model(_prompt: str, stage: str, hints: dict[str, Any] | None = None) -> dict[str, Any]:
        turn = int((hints or {}).get("turn_index", 0) or 0)
        if stage == "mainloop_decide" and turn == 1:
            payload = {
                "type": "tool_calls",
                "calls": [
                    {
                        "tool_name": "workspace.file.rename",
                        "arguments": {"src": "missing.txt", "dst": "recovered.txt"},
                        "reason": "attempt requested rename",
                    }
                ],
                "execution_mode": "model_driven",
            }
            return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}
        payload = {
            "type": "final_answer",
            "response": "无法完成重命名：工具证据显示源文件 missing.txt 不存在。",
            "execution_mode": "model_driven",
            "completion": {
                "outcome": "failed",
                "is_complete": False,
                "needs_side_effect_proof": True,
                "evidence_refs": ["ev:1:workspace.file.rename"],
                "evidence_note": "workspace.file.rename failed because source file is missing",
                "remaining_blockers": ["missing.txt does not exist"],
            },
        }
        return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}

    kernel, _ = _build_kernel(
        tmp_path,
        route_model_func=_route_model,
        permission_profile={
            "read_only": True,
            "runtime_write": True,
            "memory_read": True,
            "memory_write": False,
            "workspace_read": True,
            "workspace_write": True,
            "shell_exec": False,
            "network": False,
            "model_call": True,
        },
    )
    state = EngineState.from_input(
        "Rename missing.txt to recovered.txt. If you cannot complete it, explain the exact tool evidence.",
        run_id="r-loop-failed-outcome",
        session_id="s-loop",
    )
    ctx = kernel.new_run_context(run_id="r-loop-failed-outcome", session_id="s-loop")

    final = await kernel.engine_loop.run(state, run_context=ctx)

    assert "无法完成重命名" in final.response
    assert final.metadata.get("completion_gate_blocked_count", 0) == 0
    assert any(
        isinstance(item, dict)
        and item.get("tool_name") == "workspace.file.rename"
        and item.get("ok") is False
        for item in final.metadata.get("recent_tool_results", [])
    )
    assert not any(item.get("event") == "mainloop.final_answer.blocked" for item in final.trace)


@pytest.mark.asyncio
async def test_engine_loop_allows_preference_update_without_tool_evidence(tmp_path: Path) -> None:
    async def _route_model(_prompt: str, stage: str, _hints: dict[str, Any] | None = None) -> dict[str, Any]:
        assert stage != "mainloop_goal_contract"
        payload = {
            "type": "final_answer",
            "response": "好的，后续我会默认使用 Python 和 .py 文件。",
            "execution_mode": "preference_update",
            "workflow_decision": {"requires_context": False, "use_quick_answer": False},
            "constraint_maintenance": {
                "enabled": True,
                "reason": "user updated a durable session file-language preference",
            },
        }
        return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}

    kernel, _ = _build_kernel(tmp_path, route_model_func=_route_model)
    state = EngineState.from_input(
        "我要修改我的偏好，后面的文件用python语言的py文件创建",
        run_id="r-loop-pref-update",
        session_id="s-loop",
    )
    ctx = kernel.new_run_context(run_id="r-loop-pref-update", session_id="s-loop")

    final = await kernel.engine_loop.run(state, run_context=ctx)

    assert final.response == "好的，后续我会默认使用 Python 和 .py 文件。"
    assert final.workflow_decision.get("constraint_maintenance", {}).get("enabled") is True
    assert final.metadata.get("completion_gate_blocked_count", 0) == 0
    assert not any(item.get("event") == "mainloop.final_answer.blocked" for item in final.trace)


@pytest.mark.asyncio
async def test_engine_loop_requires_success_after_tool_loop_has_started(tmp_path: Path) -> None:
    async def _route_model(_prompt: str, stage: str, hints: dict[str, Any] | None = None) -> dict[str, Any]:
        turn = int((hints or {}).get("turn_index", 0) or 0)
        if stage == "mainloop_decide" and turn == 1:
            payload = {
                "type": "tool_calls",
                "calls": [
                    {
                        "tool_name": "workspace.file.read",
                        "arguments": {"path": "missing.txt"},
                        "reason": "inspect before answering",
                    }
                ],
                "execution_mode": "model_driven",
            }
            return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}
        payload = {
            "type": "final_answer",
            "response": "I inspected the file.",
            "execution_mode": "model_driven",
        }
        return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}

    kernel, _ = _build_kernel(tmp_path, route_model_func=_route_model)
    state = EngineState.from_input("Inspect missing.txt and summarize it", run_id="r-loop-tool-fail", session_id="s-loop")
    state.metadata["max_turns"] = 3
    ctx = kernel.new_run_context(run_id="r-loop-tool-fail", session_id="s-loop")

    final = await kernel.engine_loop.run(state, run_context=ctx)

    assert "Execution is not verified" in final.response
    assert final.metadata.get("completion_gate_blocked_count", 0) >= 1
    assert any(item.get("event") == "mainloop.final_answer.blocked" for item in final.trace)


@pytest.mark.asyncio
async def test_engine_loop_treats_semantic_search_as_verified_read_evidence(tmp_path: Path) -> None:
    async def _route_model(_prompt: str, stage: str, hints: dict[str, Any] | None = None) -> dict[str, Any]:
        turn = int((hints or {}).get("turn_index", 0) or 0)
        if stage == "mainloop_goal_contract":
            payload = {
                "intent_kind": "read_only",
                "requires_side_effect": False,
                "requires_read_proof": True,
                "target_paths": [],
                "rationale": "read-only memory retrieval task",
            }
            return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}
        if stage == "mainloop_decide" and turn == 1:
            payload = {
                "type": "tool_calls",
                "calls": [
                    {
                        "tool_name": "storage.semantic.search",
                        "arguments": {"collection_name": "cf_chat_memory", "query_text": "preference", "limit": 3},
                        "reason": "retrieve memory evidence",
                    }
                ],
                "execution_mode": "model_driven",
            }
            return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}
        payload = {
            "type": "final_answer",
            "response": "Found no preference marker in memory evidence.",
            "execution_mode": "model_driven",
            "completion": {"needs_read_proof": True, "is_complete": True},
        }
        return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}

    kernel, _ = _build_kernel(tmp_path, route_model_func=_route_model)
    state = EngineState.from_input(
        "Search memory and summarize user preference markers",
        run_id="r-loop-semantic-read-proof",
        session_id="s-loop",
    )
    state.metadata["max_turns"] = 4
    ctx = kernel.new_run_context(run_id="r-loop-semantic-read-proof", session_id="s-loop")

    final = await kernel.engine_loop.run(state, run_context=ctx)

    assert final.response.startswith("Found no preference marker")
    assert final.metadata.get("completion_gate_blocked_count", 0) == 0
    recent = final.metadata.get("recent_tool_results", [])
    assert isinstance(recent, list)
    semantic_items = [item for item in recent if isinstance(item, dict) and item.get("tool_name") == "storage.semantic.search"]
    assert semantic_items
    flag = semantic_items[-1].get("execution_flag", {})
    assert isinstance(flag, dict)
    assert flag.get("verified_read") is True


@pytest.mark.asyncio
async def test_engine_loop_ignores_model_declared_read_proof_without_system_requirement(tmp_path: Path) -> None:
    async def _route_model(_prompt: str, stage: str, _hints: dict[str, Any] | None = None) -> dict[str, Any]:
        if stage == "mainloop_goal_contract":
            payload = {
                "intent_kind": "analysis",
                "requires_side_effect": False,
                "requires_read_proof": False,
                "target_paths": [],
                "rationale": "analysis-only task",
            }
            return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}
        payload = {
            "type": "final_answer",
            "response": "Here is the conceptual explanation.",
            "execution_mode": "model_driven",
            "completion": {
                "needs_read_proof": True,
                "is_complete": True,
            },
        }
        return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}

    kernel, _ = _build_kernel(tmp_path, route_model_func=_route_model)
    state = EngineState.from_input(
        "Explain why vector search helps memory recall in plain language",
        run_id="r-loop-ignore-model-read-proof",
        session_id="s-loop",
    )
    state.metadata["max_turns"] = 2
    ctx = kernel.new_run_context(run_id="r-loop-ignore-model-read-proof", session_id="s-loop")

    final = await kernel.engine_loop.run(state, run_context=ctx)

    assert final.response == "Here is the conceptual explanation."
    assert final.metadata.get("completion_gate_blocked_count", 0) == 0
    assert not any(item.get("event") == "mainloop.final_answer.blocked" for item in final.trace)


@pytest.mark.asyncio
async def test_engine_loop_does_not_auto_require_read_proof_for_read_only_intent(tmp_path: Path) -> None:
    async def _route_model(_prompt: str, stage: str, _hints: dict[str, Any] | None = None) -> dict[str, Any]:
        if stage == "mainloop_goal_contract":
            payload = {
                "intent_kind": "read_only",
                "requires_side_effect": False,
                "requires_read_proof": False,
                "target_paths": [],
                "rationale": "read-only answer does not require external proof",
            }
            return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}
        payload = {
            "type": "final_answer",
            "response": "Direct read-only answer without tool loop.",
            "execution_mode": "model_driven",
            "completion": {
                "needs_read_proof": True,
                "is_complete": True,
            },
        }
        return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}

    kernel, _ = _build_kernel(tmp_path, route_model_func=_route_model)
    state = EngineState.from_input(
        "Provide a direct answer from context only",
        run_id="r-loop-readonly-no-auto-proof",
        session_id="s-loop",
    )
    state.metadata["max_turns"] = 2
    ctx = kernel.new_run_context(run_id="r-loop-readonly-no-auto-proof", session_id="s-loop")

    final = await kernel.engine_loop.run(state, run_context=ctx)

    assert final.response == "Direct read-only answer without tool loop."
    assert final.metadata.get("completion_gate_blocked_count", 0) == 0
    assert not any(item.get("event") == "mainloop.final_answer.blocked" for item in final.trace)


@pytest.mark.asyncio
async def test_engine_loop_honors_explicit_read_proof_requirement(tmp_path: Path) -> None:
    async def _route_model(_prompt: str, stage: str, _hints: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = {
            "type": "final_answer",
            "response": "Read-only answer without any tool evidence.",
            "execution_mode": "model_driven",
            "completion": {
                "needs_read_proof": False,
                "is_complete": True,
            },
        }
        return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}

    kernel, _ = _build_kernel(tmp_path, route_model_func=_route_model)
    state = EngineState.from_input(
        "Answer with explicit read-proof requirement",
        run_id="r-loop-readonly-explicit-proof",
        session_id="s-loop",
    )
    state.metadata["goal_contract"] = {
        "intent_kind": "read_only",
        "requires_side_effect": False,
        "requires_read_proof": True,
        "target_paths": [],
    }
    state.metadata["max_turns"] = 2
    ctx = kernel.new_run_context(run_id="r-loop-readonly-explicit-proof", session_id="s-loop")

    final = await kernel.engine_loop.run(state, run_context=ctx)

    assert "Execution is not verified" in final.response
    assert final.metadata.get("completion_gate_blocked_count", 0) >= 1
    assert any(item.get("event") == "mainloop.final_answer.blocked" for item in final.trace)
@pytest.mark.asyncio
async def test_engine_loop_allows_evidence_backed_final_answer_without_semantic_second_gate(
    tmp_path: Path,
) -> None:
    verify_calls = {"count": 0}

    async def _route_model(_prompt: str, stage: str, hints: dict[str, Any] | None = None) -> dict[str, Any]:
        turn = int((hints or {}).get("turn_index", 0) or 0)
        if stage == "mainloop_goal_contract":
            payload = {
                "intent_kind": "state_change",
                "requires_side_effect": True,
                "requires_read_proof": False,
                "target_paths": ["sorting.py"],
                "rationale": "task asks file create and rewrite",
            }
            return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}
        if stage == "goal_contract_verify":
            payload = {"requires_side_effect": True, "confidence": 0.95, "reason": "explicit mutation request"}
            return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}
        if stage == "mainloop_completion_verify":
            verify_calls["count"] += 1
            if verify_calls["count"] == 1:
                payload = {
                    "can_commit": False,
                    "reason": "response misses required behavior: empty list should return None",
                    "missing": ["empty list returns None"],
                }
                return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}
            payload = {"can_commit": True, "reason": "", "missing": []}
            return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}
        if stage == "mainloop_decide" and turn == 1:
            payload = {
                "type": "tool_calls",
                "calls": [
                    {
                        "tool_name": "workspace.file.write",
                        "arguments": {
                            "path": "sorting.py",
                            "content": "def sort_numbers(items):\n    return sorted(items)\n",
                        },
                        "reason": "create target file",
                    }
                ],
                "execution_mode": "model_driven",
                "workflow_decision": {"requires_context": True, "use_quick_answer": False},
            }
            return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}
        if stage == "mainloop_decide" and turn == 2:
            payload = {
                "type": "final_answer",
                "response": "created sorting.py but omitted empty-list handling",
                "execution_mode": "model_driven",
                "workflow_decision": {"requires_context": True, "use_quick_answer": False},
            }
            return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}
        payload = {
            "type": "final_answer",
            "response": "created sorting.py and handled empty list returning None",
            "execution_mode": "model_driven",
            "workflow_decision": {"requires_context": True, "use_quick_answer": False},
            "completion": {"evidence_refs": ["ev:1:workspace.file.write"]},
        }
        return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}

    kernel, _ = _build_kernel(
        tmp_path,
        route_model_func=_route_model,
        permission_profile={
            "read_only": True,
            "runtime_write": True,
            "memory_read": True,
            "memory_write": False,
            "workspace_read": True,
            "workspace_write": True,
            "shell_exec": False,
            "network": False,
            "model_call": True,
        },
    )
    state = EngineState.from_input(
        "Create sorting.py and ensure empty list returns None; then report completion path.",
        run_id="r-loop-4b",
        session_id="s-loop",
    )
    ctx = kernel.new_run_context(run_id="r-loop-4b", session_id="s-loop")

    final = await kernel.engine_loop.run(state, run_context=ctx)

    assert final.response.startswith("created sorting.py")
    assert verify_calls["count"] == 0
    blocked_events = [item for item in final.trace if item.get("event") == "mainloop.final_answer.blocked"]
    assert blocked_events == []
    assert (tmp_path / "sorting.py").exists()
@pytest.mark.asyncio
async def test_engine_loop_trusts_direct_final_answer_before_tool_loop(tmp_path: Path) -> None:
    async def _route_model(_prompt: str, _stage: str, _hints: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = {
            "type": "final_answer",
            "response": "已经把项目里的数据库都更新好了",
            "execution_mode": "model_driven",
        }
        return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}

    kernel, _ = _build_kernel(tmp_path, route_model_func=_route_model)
    state = EngineState.from_input("请更新项目数据库并完成迁移", run_id="r-loop-5", session_id="s-loop")
    state.metadata["max_turns"] = 2
    ctx = kernel.new_run_context(run_id="r-loop-5", session_id="s-loop")

    final = await kernel.engine_loop.run(state, run_context=ctx)

    assert final.response == "已经把项目里的数据库都更新好了"
    assert final.metadata.get("completion_gate_blocked_count", 0) == 0




@pytest.mark.asyncio
async def test_engine_loop_stops_after_three_blocked_final_answers_after_tool_loop(tmp_path: Path) -> None:
    async def _route_model(_prompt: str, _stage: str, _hints: dict[str, Any] | None = None) -> dict[str, Any]:
        turn = int((_hints or {}).get("turn_index", 0) or 0)
        if _stage == "mainloop_decide" and turn == 1:
            payload = {
                "type": "tool_calls",
                "calls": [
                    {
                        "tool_name": "workspace.file.read",
                        "arguments": {"path": "missing.txt"},
                        "reason": "start tool loop",
                    }
                ],
                "execution_mode": "model_driven",
            }
            return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}
        payload = {
            "type": "final_answer",
            "response": "claimed done without evidence",
            "execution_mode": "model_driven",
        }
        return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}

    kernel, _ = _build_kernel(tmp_path, route_model_func=_route_model)
    state = EngineState.from_input("Please create file demo.txt with hello", run_id="r-loop-5b", session_id="s-loop")
    state.metadata["max_turns"] = 8
    ctx = kernel.new_run_context(run_id="r-loop-5b", session_id="s-loop")

    final = await kernel.engine_loop.run(state, run_context=ctx)

    assert final.metadata.get("completion_gate_blocked_count", 0) >= 3
    assert final.metadata.get("completion_gate_terminal") is True
    assert "Execution not completed with verifiable evidence." in final.response
    events = [item.get("event") for item in final.trace]
    assert "mainloop.final_answer.terminal_block" in events


@pytest.mark.asyncio
async def test_engine_loop_blocks_repeated_workspace_list_without_workspace_change(tmp_path: Path) -> None:
    async def _route_model(_prompt: str, stage: str, hints: dict[str, Any] | None = None) -> dict[str, Any]:
        turn = int((hints or {}).get("turn_index", 0) or 0)
        if stage == "mainloop_decide" and turn == 1:
            payload = {
                "type": "tool_calls",
                "calls": [
                    {
                        "tool_name": "workspace.file.list",
                        "arguments": {"path": ".", "recursive": False, "max_entries": 50},
                        "reason": "discover files",
                    }
                ],
                "execution_mode": "model_driven",
            }
            return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}
        if stage == "mainloop_decide" and turn == 2:
            payload = {
                "type": "tool_calls",
                "calls": [
                    {
                        "tool_name": "workspace.file.list",
                        "arguments": {"path": ".", "recursive": False, "max_entries": 50},
                        "reason": "repeat same list",
                    }
                ],
                "execution_mode": "model_driven",
            }
            return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}
        payload = {
            "type": "final_answer",
            "response": "done after one valid read/list",
            "execution_mode": "model_driven",
        }
        return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}

    kernel, _ = _build_kernel(tmp_path, route_model_func=_route_model)
    state = EngineState.from_input("list workspace once and finish", run_id="r-loop-6", session_id="s-loop")
    ctx = kernel.new_run_context(run_id="r-loop-6", session_id="s-loop")

    final = await kernel.engine_loop.run(state, run_context=ctx)

    events = [item.get("event") for item in final.trace]
    assert "loop.tool.blocked_repeated_success" in events

    recent = final.metadata.get("recent_tool_results", [])
    assert isinstance(recent, list)
    list_ok_count = sum(
        1
        for item in recent
        if isinstance(item, dict) and item.get("tool_name") == "workspace.file.list" and bool(item.get("ok", False))
    )
    assert list_ok_count == 1


