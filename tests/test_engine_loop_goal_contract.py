# coding=utf-8
"""Goal-contract checks for mainloop side-effect consistency."""

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
) -> SelfAIKernel:
    return SelfAIKernel(
        project_root=tmp_path,
        permission_profile=permission_profile
        or {
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
        dependencies={
            "route_model_func": route_model_func,
            "redis_store": FakeRedisStore(),
            "qdrant_memory": FakeQdrantMemory(),
            "graph_memory": FakeGraphMemory(),
            "web_search_func": lambda _query, _max_results: [],
            "project_root": str(tmp_path),
        },
    )


@pytest.mark.asyncio
async def test_mainloop_blocks_final_answer_when_write_path_mismatches_goal(tmp_path: Path) -> None:
    async def _route_model(
        _prompt: str,
        stage: str,
        hints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        turn = int((hints or {}).get("turn_index", 0) or 0)
        if stage == "mainloop_completion_verify":
            payload = {"can_commit": True, "reason": "", "missing": []}
            return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}
        if stage == "mainloop_decide" and turn == 1:
            payload = {
                "type": "tool_calls",
                "calls": [
                    {
                        "tool_name": "workspace.file.write",
                        "arguments": {"path": "wrong.txt", "content": "hello"},
                        "reason": "write file",
                    }
                ],
                "execution_mode": "model_driven",
                "completion": {"needs_side_effect_proof": True},
            }
            return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}
        payload = {
            "type": "final_answer",
            "response": "Done.",
            "execution_mode": "model_driven",
            "completion": {"needs_side_effect_proof": True},
        }
        return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}

    kernel = _build_kernel(tmp_path, route_model_func=_route_model)
    state = EngineState.from_input(
        "Create file artifacts/target.txt with hello",
        run_id="r-goal-1",
        session_id="s-goal",
    )
    state.metadata["max_turns"] = 2
    ctx = kernel.new_run_context(run_id="r-goal-1", session_id="s-goal")

    final = await kernel.engine_loop.run(state, run_context=ctx)

    assert (tmp_path / "wrong.txt").exists()
    assert not (tmp_path / "artifacts" / "target.txt").exists()
    control_events = final.metadata.get("control_events", [])
    assert any(event.get("type") == "CompletionGateBlocked" for event in control_events)
    assert any(err.get("type") == "CompletionNotVerifiedTerminal" for err in final.errors)
    assert "Execution is not verified" in final.response


@pytest.mark.asyncio
async def test_mainloop_allows_final_answer_when_write_path_matches_goal(tmp_path: Path) -> None:
    async def _route_model(
        _prompt: str,
        stage: str,
        hints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        turn = int((hints or {}).get("turn_index", 0) or 0)
        if stage == "mainloop_completion_verify":
            payload = {"can_commit": True, "reason": "", "missing": []}
            return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}
        if stage == "mainloop_decide" and turn == 1:
            payload = {
                "type": "tool_calls",
                "calls": [
                    {
                        "tool_name": "workspace.file.write",
                        "arguments": {"path": "artifacts/target.txt", "content": "hello"},
                        "reason": "write target file",
                    }
                ],
                "execution_mode": "model_driven",
                "completion": {"needs_side_effect_proof": True},
            }
            return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}
        payload = {
            "type": "final_answer",
            "response": "Created artifacts/target.txt",
            "execution_mode": "model_driven",
            "completion": {"needs_side_effect_proof": True},
        }
        return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}

    kernel = _build_kernel(tmp_path, route_model_func=_route_model)
    state = EngineState.from_input(
        "Create file artifacts/target.txt with hello",
        run_id="r-goal-2",
        session_id="s-goal",
    )
    ctx = kernel.new_run_context(run_id="r-goal-2", session_id="s-goal")

    final = await kernel.engine_loop.run(state, run_context=ctx)

    assert final.response == "Created artifacts/target.txt"
    assert (tmp_path / "artifacts" / "target.txt").exists()


@pytest.mark.asyncio
async def test_mainloop_allows_related_read_before_mutation(tmp_path: Path) -> None:
    async def _route_model(
        _prompt: str,
        stage: str,
        hints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        turn = int((hints or {}).get("turn_index", 0) or 0)
        if stage == "mainloop_completion_verify":
            payload = {"can_commit": True, "reason": "", "missing": []}
            return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}
        if stage == "mainloop_decide" and turn == 1:
            payload = {
                "type": "tool_calls",
                "calls": [
                    {
                        "tool_name": "workspace.file.read",
                        "arguments": {"path": "artifacts/target2.txt"},
                        "reason": "try read first",
                    }
                ],
                "execution_mode": "model_driven",
                "completion": {"needs_side_effect_proof": True},
            }
            return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}
        if stage == "mainloop_decide" and turn == 2:
            payload = {
                "type": "tool_calls",
                "calls": [
                    {
                        "tool_name": "workspace.file.write",
                        "arguments": {"path": "artifacts/target2.txt", "content": "ok"},
                        "reason": "write target",
                    }
                ],
                "execution_mode": "model_driven",
                "completion": {"needs_side_effect_proof": True},
            }
            return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}
        payload = {
            "type": "final_answer",
            "response": "Created artifacts/target2.txt",
            "execution_mode": "model_driven",
            "completion": {"needs_side_effect_proof": True},
        }
        return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}

    kernel = _build_kernel(tmp_path, route_model_func=_route_model)
    state = EngineState.from_input(
        "Create file artifacts/target2.txt",
        run_id="r-goal-3",
        session_id="s-goal",
    )
    state.metadata["max_turns"] = 3
    ctx = kernel.new_run_context(run_id="r-goal-3", session_id="s-goal")

    final = await kernel.engine_loop.run(state, run_context=ctx)

    events = [item.get("event") for item in final.trace]
    assert "loop.tool.blocked_goal_mismatch" not in events
    assert (tmp_path / "artifacts" / "target2.txt").exists()
    assert final.response == "Created artifacts/target2.txt"


def test_goal_contract_resolves_basename_path_from_session_history(tmp_path: Path) -> None:
    async def _route_model(_prompt: str, _stage: str, _hints: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"model": "mock-fast", "response": "{}"}

    kernel = _build_kernel(tmp_path, route_model_func=_route_model)
    state = EngineState.from_input(
        "Ensure b_note.txt still has exactly those three lines and no extra lines.",
        run_id="r-goal-anchor-1",
        session_id="s-goal-anchor",
        metadata={
            "chat_recent_turns": [
                {
                    "preview": "Created artifacts/memory_eval_v2_workspace/b_note.txt with content hello.",
                }
            ]
        },
    )

    contract = kernel.engine_loop._derive_goal_contract(state)

    assert contract["target_paths"] == ["artifacts/memory_eval_v2_workspace/b_note.txt"]
    assert contract["has_explicit_target_path"] is True


def test_goal_contract_anchors_path_for_final_file_continuation_task(tmp_path: Path) -> None:
    async def _route_model(_prompt: str, _stage: str, _hints: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"model": "mock-fast", "response": "{}"}

    kernel = _build_kernel(tmp_path, route_model_func=_route_model)
    state = EngineState.from_input(
        "Return final file path and repeat each line verbatim.",
        run_id="r-goal-anchor-2",
        session_id="s-goal-anchor",
        metadata={
            "chat_recent_turns": [
                {
                    "preview": "Renamed artifacts/memory_eval_v2_workspace/b_note.txt to artifacts/memory_eval_v2_workspace/b_final_note.txt.",
                }
            ]
        },
    )

    contract = kernel.engine_loop._derive_goal_contract(state)

    assert contract["target_paths"][0] == "artifacts/memory_eval_v2_workspace/b_final_note.txt"
    assert contract["has_explicit_target_path"] is True


@pytest.mark.asyncio
async def test_mainloop_blocks_unrelated_read_before_mutation(tmp_path: Path) -> None:
    async def _route_model(
        _prompt: str,
        stage: str,
        hints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        turn = int((hints or {}).get("turn_index", 0) or 0)
        if stage == "mainloop_decide" and turn == 1:
            payload = {
                "type": "tool_calls",
                "calls": [
                    {
                        "tool_name": "workspace.file.read",
                        "arguments": {"path": "README.md"},
                        "reason": "unrelated read",
                    }
                ],
                "execution_mode": "model_driven",
                "completion": {"needs_side_effect_proof": True},
            }
            return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}
        if stage == "mainloop_decide" and turn == 2:
            payload = {
                "type": "tool_calls",
                "calls": [
                    {
                        "tool_name": "workspace.file.write",
                        "arguments": {"path": "artifacts/target3.txt", "content": "ok"},
                        "reason": "write target",
                    }
                ],
                "execution_mode": "model_driven",
                "completion": {"needs_side_effect_proof": True},
            }
            return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}
        payload = {
            "type": "final_answer",
            "response": "Created artifacts/target3.txt",
            "execution_mode": "model_driven",
            "completion": {"needs_side_effect_proof": True},
        }
        return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}

    kernel = _build_kernel(tmp_path, route_model_func=_route_model)
    state = EngineState.from_input(
        "Create file artifacts/target3.txt",
        run_id="r-goal-33",
        session_id="s-goal",
    )
    state.metadata["max_turns"] = 3
    ctx = kernel.new_run_context(run_id="r-goal-33", session_id="s-goal")

    final = await kernel.engine_loop.run(state, run_context=ctx)
    events = [item.get("event") for item in final.trace]
    assert "loop.tool.blocked_goal_mismatch" in events
    assert (tmp_path / "artifacts" / "target3.txt").exists()


@pytest.mark.asyncio
async def test_mainloop_blocks_repeated_successful_same_call(tmp_path: Path) -> None:
    async def _route_model(
        _prompt: str,
        stage: str,
        _hints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if stage != "mainloop_decide":
            return {"model": "mock-fast", "response": "{}"}
        payload = {
            "type": "tool_calls",
            "calls": [
                {
                    "tool_name": "workspace.file.write",
                    "arguments": {"path": "artifacts/repeat.txt", "content": "hello"},
                    "reason": "write repeatedly",
                }
            ],
            "execution_mode": "model_driven",
            "completion": {"needs_side_effect_proof": True},
            "goal_contract": {"intent_kind": "state_change", "target_paths": ["artifacts/repeat.txt"]},
        }
        return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}

    kernel = _build_kernel(tmp_path, route_model_func=_route_model)
    state = EngineState.from_input(
        "Create file artifacts/repeat.txt",
        run_id="r-goal-4",
        session_id="s-goal",
    )
    state.metadata["max_turns"] = 6
    ctx = kernel.new_run_context(run_id="r-goal-4", session_id="s-goal")

    final = await kernel.engine_loop.run(state, run_context=ctx)

    assert (tmp_path / "artifacts" / "repeat.txt").exists()
    assert any(err.get("type") == "RepeatedToolCallNoProgress" for err in final.errors)
    assert final.metadata.get("auto_finalized_due_to_no_progress") is True
    assert final.response.startswith("Done. Verified execution evidence exists")


@pytest.mark.asyncio
async def test_mainloop_quick_answer_flag_cannot_bypass_side_effect_goal_contract(tmp_path: Path) -> None:
    async def _route_model(
        _prompt: str,
        stage: str,
        hints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        turn = int((hints or {}).get("turn_index", 0) or 0)
        if stage == "mainloop_decide" and turn == 1:
            payload = {
                "type": "final_answer",
                "response": "done without tools",
                "workflow_decision": {
                    "execution_mode": "quick_answer",
                    "use_quick_answer": True,
                    "requires_context": False,
                },
            }
            return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}
        payload = {
            "type": "final_answer",
            "response": "still no tools",
            "workflow_decision": {
                "execution_mode": "quick_answer",
                "use_quick_answer": True,
                "requires_context": False,
            },
        }
        return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}

    kernel = _build_kernel(tmp_path, route_model_func=_route_model)
    state = EngineState.from_input(
        "Create file artifacts/quick_guard.txt with hello",
        run_id="r-goal-quick-guard",
        session_id="s-goal",
    )
    state.metadata["goal_contract"] = {
        "requires_side_effect": True,
        "requires_read_proof": False,
        "target_paths": ["artifacts/quick_guard.txt"],
        "has_explicit_target_path": True,
        "intent_kind": "state_change",
        "source": "test",
    }
    state.metadata["max_turns"] = 2
    ctx = kernel.new_run_context(run_id="r-goal-quick-guard", session_id="s-goal")
    if getattr(ctx, "settings", None) is not None:
        setattr(ctx.settings, "mainloop_goal_contract_model_enabled", False)

    final = await kernel.engine_loop.run(state, run_context=ctx)
    assert "Execution is not verified" in final.response
    control_events = final.metadata.get("control_events", [])
    assert any(event.get("type") == "CompletionGateBlocked" for event in control_events)
    assert any(err.get("type") == "CompletionNotVerifiedTerminal" for err in final.errors)
