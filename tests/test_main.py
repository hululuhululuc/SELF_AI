# coding=utf-8
"""Tests for main entrypoint on Kernel path."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from self_ai import main
from self_ai.memory.chat_memory_service import ChatMemoryService
from self_ai.memory.chat_memory_store import ChatMemoryStore


class _FakeKernel:
    def __init__(self) -> None:
        self.run = AsyncMock(
            return_value={
                "response": "ok",
                "model": "mock-model",
                "run_id": "run-1",
                "session_id": "session-1",
                "workflow_decision": {"execution_mode": "quick_answer"},
            }
        )
        self.run_once = AsyncMock(
            return_value={
                "ok": True,
                "data": {
                    "response": (
                        '{"profiles":{"turn_summary":{"memory_class":"preference","importance":0.9,'
                        '"decay_mode":"none","pinned":true,"confidence":0.9}}}'
                    )
                },
            }
        )


@pytest.mark.asyncio
async def test_run_autonomy_workflow_calls_kernel_run_and_keeps_contract() -> None:
    fake_kernel = _FakeKernel()
    with (
        patch("self_ai.main.get_kernel", return_value=fake_kernel),
        patch("self_ai.main.get_chat_memory_store", return_value=None),
    ):
        result = await main.run_autonomy_workflow("Explain this project")

    fake_kernel.run.assert_awaited_once()
    assert result["response"] == "ok"
    assert result["model"] == "mock-model"
    assert result["run_id"] == "run-1"
    assert result["session_id"] == "session-1"
    assert "task_profile" in result
    assert "plan" in result
    assert "context_pack" in result
    assert "selected_agents" in result
    assert "agent_outputs" in result
    assert "review_reports" in result
    assert "quality_gate" in result
    assert "errors" in result


@pytest.mark.asyncio
async def test_run_autonomy_workflow_prefers_fusion_memory_loader() -> None:
    class _FakeChatService:
        def __init__(self) -> None:
            self.called = False

        async def retrieve_memory_for_turn(self, **kwargs):
            self.called = True
            return {
                "chat_id": kwargs.get("chat_id", ""),
                "requested_turns": 1,
                "kept_turns": 1,
                "dropped_turns": 0,
                "budget": {"max_chars_total": 1000, "max_item_chars": 200, "used_chars": 120},
                "turns": [{"turn_id": 1, "preview": "recent memory"}],
                "source_type_groups": {"L1": 1},
                "stats": {
                    "fusion_failed": False,
                    "degraded": False,
                    "degraded_reasons": [],
                    "layer_health": {"l1_ok": True, "l2_ok": False, "l3_ok": False},
                },
            }

    fake_kernel = _FakeKernel()
    fake_service = _FakeChatService()
    with (
        patch("self_ai.main.get_kernel", return_value=fake_kernel),
        patch("self_ai.main.get_chat_memory_store", return_value=None),
        patch("self_ai.main.get_chat_memory_service", return_value=fake_service),
        patch.object(main.settings, "chat_memory_fusion_enabled", True),
    ):
        result = await main.run_autonomy_workflow("use memory", session_id="chat-fusion-main")

    assert fake_service.called is True
    assert result["metadata"]["chat_recent_turns_stats"]["kept_turns"] == 1
    assert result["metadata"]["chat_recent_turns_stats"]["fusion_failed"] is False
    assert result["metadata"]["chat_recent_turns_stats"]["layer_health"]["l1_ok"] is True


@pytest.mark.asyncio
async def test_run_autonomy_workflow_marks_fusion_error_visible() -> None:
    class _FailingChatService:
        async def retrieve_memory_for_turn(self, **_kwargs):
            raise RuntimeError("fusion broken")

    fake_kernel = _FakeKernel()
    with (
        patch("self_ai.main.get_kernel", return_value=fake_kernel),
        patch("self_ai.main.get_chat_memory_store", return_value=None),
        patch("self_ai.main.get_chat_memory_service", return_value=_FailingChatService()),
        patch.object(main.settings, "chat_memory_fusion_enabled", True),
    ):
        result = await main.run_autonomy_workflow("use memory", session_id="chat-fusion-fail")

    stats = result["metadata"]["chat_recent_turns_stats"]
    assert stats["fusion_failed"] is True
    assert stats["degraded"] is True
    assert "fusion_read_failed" in stats["degraded_reasons"]


@pytest.mark.asyncio
async def test_run_autonomy_workflow_marks_garbled_input_warning() -> None:
    fake_kernel = _FakeKernel()
    with (
        patch("self_ai.main.get_kernel", return_value=fake_kernel),
        patch("self_ai.main.get_chat_memory_store", return_value=None),
    ):
        result = await main.run_autonomy_workflow("????????")

    fake_kernel.run.assert_awaited_once()
    assert result["metadata"]["input_garbled_warning"] is True


@pytest.mark.asyncio
async def test_run_autonomy_workflow_passes_session_id_and_chat_metadata() -> None:
    fake_kernel = _FakeKernel()
    with (
        patch("self_ai.main.get_kernel", return_value=fake_kernel),
        patch("self_ai.main.get_chat_memory_store", return_value=None),
    ):
        await main.run_autonomy_workflow(
            "Create a file",
            session_id="chat-123",
            user_id="u-9",
        )

    _, kwargs = fake_kernel.run.await_args
    assert kwargs["session_id"] == "chat-123"
    assert kwargs["metadata"]["chat_id"] == "chat-123"
    assert kwargs["metadata"]["user_id"] == "u-9"
    assert isinstance(kwargs["metadata"]["chat_recent_turns"], list)


def test_chat_memory_service_interfaces_safe_when_disabled() -> None:
    with patch("self_ai.main.get_chat_memory_service", return_value=None):
        manifest = main.get_chat_manifest(chat_id="chat-x")
        replay = main.replay_chat_turns(chat_id="chat-x")
        recent = main.load_recent_chat_turns(chat_id="chat-x")

    assert manifest["chat_id"] == "chat-x"
    assert manifest["enabled"] is False
    assert replay == []
    assert recent["chat_id"] == "chat-x"
    assert recent["kept_turns"] == 0


@pytest.mark.asyncio
async def test_run_autonomy_workflow_applies_memory_sidecar_decision() -> None:
    class _FakeChatStore:
        def __init__(self) -> None:
            self.captured_decision = None

        def append_turn(self, **kwargs):
            self.captured_decision = kwargs.get("memory_decision")
            return {"turn_id": 1, "committed_memory_count": 1}

    class _FakeChatService:
        def load_recent_turns_for_injection(self, **kwargs):
            return {
                "chat_id": kwargs.get("chat_id", ""),
                "requested_turns": 0,
                "kept_turns": 0,
                "dropped_turns": 0,
                "budget": {"max_chars_total": 1400, "max_item_chars": 220, "used_chars": 0},
                "turns": [],
            }

    fake_kernel = _FakeKernel()
    fake_store = _FakeChatStore()
    fake_service = _FakeChatService()
    with (
        patch("self_ai.main.get_kernel", return_value=fake_kernel),
        patch("self_ai.main.get_chat_memory_store", return_value=fake_store),
        patch("self_ai.main.get_chat_memory_service", return_value=fake_service),
        patch.object(main.settings, "chat_memory_sidecar_enabled", True),
    ):
        result = await main.run_autonomy_workflow("remember preference", session_id="chat-sidecar")

    assert isinstance(fake_store.captured_decision, dict)
    assert "profiles" in fake_store.captured_decision
    assert result["metadata"]["chat_memory"]["turn_id"] == 1


@pytest.mark.asyncio
async def test_run_autonomy_workflow_schedules_chat_compaction() -> None:
    class _FakeChatStore:
        def append_turn(self, **kwargs):
            return {"turn_id": 11, "committed_memory_count": 0}

    class _FakeChatService:
        async def retrieve_memory_for_turn(self, **kwargs):
            return {
                "chat_id": kwargs.get("chat_id", ""),
                "requested_turns": 0,
                "kept_turns": 0,
                "dropped_turns": 0,
                "budget": {"max_chars_total": 1000, "max_item_chars": 200, "used_chars": 0},
                "turns": [],
                "source_type_groups": {},
                "stats": {"fusion_failed": False},
            }

        def compact_chat_memory(self, **_kwargs):
            return {"compacted": False, "reason": "no_candidates"}

    fake_kernel = _FakeKernel()
    with (
        patch("self_ai.main.get_kernel", return_value=fake_kernel),
        patch("self_ai.main.get_chat_memory_store", return_value=_FakeChatStore()),
        patch("self_ai.main.get_chat_memory_service", return_value=_FakeChatService()),
        patch("self_ai.main._schedule_chat_memory_compaction") as schedule_mock,
        patch.object(main.settings, "chat_memory_sidecar_enabled", False),
        patch.object(main.settings, "chat_memory_fusion_enabled", True),
        patch.object(main.settings, "chat_memory_compact_enabled", True),
    ):
        result = await main.run_autonomy_workflow("task", session_id="chat-comp-main")

    assert result["metadata"]["chat_memory_compaction_scheduled"] is True
    schedule_mock.assert_called_once()


@pytest.mark.asyncio
async def test_run_autonomy_workflow_writes_committed_memory_to_l2_l3() -> None:
    class _FakeKernelL2L3:
        def __init__(self) -> None:
            self.run = AsyncMock(
                return_value={
                    "response": "ok",
                    "model": "mock-model",
                    "run_id": "run-l2l3",
                    "session_id": "session-l2l3",
                    "task_profile": {
                        "memory_write_policy": {
                            "write_qdrant": True,
                            "write_neo4j": True,
                            "qdrant_collection": "cf_task_memory",
                            "neo4j_label": "Issue",
                        }
                    },
                }
            )

            async def _run_once(*, tool_name, **_kwargs):
                if tool_name == "model.generate":
                    return {"ok": True, "data": {"response": "{}"}}
                if tool_name == "storage.semantic.upsert_memory":
                    return {"ok": True, "data": {"upserted": True}}
                if tool_name == "storage.graph.record_issue_fix":
                    return {"ok": True, "data": {"recorded": True}}
                return {"ok": False, "error": {"type": "UnknownTool", "message": tool_name}}

            self.run_once = AsyncMock(side_effect=_run_once)

    class _FakeChatStoreL2L3:
        def append_turn(self, **_kwargs):
            return {
                "turn_id": 2,
                "committed_memory_count": 2,
                "committed_memories": [
                    {
                        "memory_id": "chat-l2l3-turn-2-response",
                        "memory_type": "response_summary",
                        "memory_class": "fact",
                        "content_preview": "created file x.py",
                        "timestamp": "2026-04-29T00:00:00Z",
                        "importance": 0.8,
                        "confidence": 0.9,
                    },
                    {
                        "memory_id": "chat-l2l3-turn-2-issue",
                        "memory_type": "issue",
                        "memory_class": "issue",
                        "content_preview": "failed previous write due to permission",
                        "timestamp": "2026-04-29T00:00:01Z",
                        "importance": 0.7,
                        "confidence": 0.85,
                    },
                ],
            }

    class _FakeChatServiceL2L3:
        def load_recent_turns_for_injection(self, **kwargs):
            return {
                "chat_id": kwargs.get("chat_id", ""),
                "requested_turns": 0,
                "kept_turns": 0,
                "dropped_turns": 0,
                "budget": {"max_chars_total": 1400, "max_item_chars": 220, "used_chars": 0},
                "turns": [],
            }

    fake_kernel = _FakeKernelL2L3()
    with (
        patch("self_ai.main.get_kernel", return_value=fake_kernel),
        patch("self_ai.main.get_chat_memory_store", return_value=_FakeChatStoreL2L3()),
        patch("self_ai.main.get_chat_memory_service", return_value=_FakeChatServiceL2L3()),
        patch("self_ai.main._get_memory_vector_encode_func", return_value=lambda _text: [0.1, 0.2, 0.3]),
        patch.object(main.settings, "chat_memory_sidecar_enabled", False),
    ):
        result = await main.run_autonomy_workflow("create file", session_id="chat-l2l3")

    l2l3 = result["metadata"]["chat_memory_l2l3_write"]
    assert l2l3["committed_count"] == 2
    assert l2l3["qdrant_attempted"] == 2
    assert l2l3["qdrant_ok"] == 2
    assert l2l3["neo4j_attempted"] == 1
    assert l2l3["neo4j_ok"] == 1
    called_tools = [call.kwargs.get("tool_name", "") for call in fake_kernel.run_once.await_args_list]
    assert "storage.semantic.upsert_memory" in called_tools
    assert "storage.graph.record_issue_fix" in called_tools


@pytest.mark.asyncio
async def test_run_autonomy_workflow_exposes_l2l3_failure_without_fallback() -> None:
    class _FakeKernelL2Fail:
        def __init__(self) -> None:
            self.run = AsyncMock(
                return_value={
                    "response": "ok",
                    "model": "mock-model",
                    "run_id": "run-l2-fail",
                    "session_id": "session-l2-fail",
                    "task_profile": {
                        "memory_write_policy": {
                            "write_qdrant": True,
                            "write_neo4j": False,
                            "qdrant_collection": "cf_task_memory",
                        }
                    },
                }
            )

            async def _run_once(*, tool_name, **_kwargs):
                if tool_name == "storage.semantic.upsert_memory":
                    return {
                        "ok": False,
                        "error": {
                            "type": "VectorSchemaMismatch",
                            "message": "vector size mismatch",
                        },
                    }
                return {"ok": True, "data": {"response": "{}"}}

            self.run_once = AsyncMock(side_effect=_run_once)

    class _FakeChatStore:
        def append_turn(self, **_kwargs):
            return {
                "turn_id": 3,
                "committed_memory_count": 1,
                "committed_memories": [
                    {
                        "memory_id": "chat-l2-fail-turn-3-response",
                        "memory_type": "response_summary",
                        "memory_class": "fact",
                        "content_preview": "response memory",
                        "timestamp": "2026-04-29T00:00:02Z",
                        "importance": 0.7,
                        "confidence": 0.9,
                    }
                ],
            }

    class _FakeChatService:
        def load_recent_turns_for_injection(self, **kwargs):
            return {
                "chat_id": kwargs.get("chat_id", ""),
                "requested_turns": 0,
                "kept_turns": 0,
                "dropped_turns": 0,
                "budget": {"max_chars_total": 1400, "max_item_chars": 220, "used_chars": 0},
                "turns": [],
            }

    fake_kernel = _FakeKernelL2Fail()
    with (
        patch("self_ai.main.get_kernel", return_value=fake_kernel),
        patch("self_ai.main.get_chat_memory_store", return_value=_FakeChatStore()),
        patch("self_ai.main.get_chat_memory_service", return_value=_FakeChatService()),
        patch("self_ai.main._get_memory_vector_encode_func", return_value=lambda _text: [0.1, 0.2, 0.3]),
        patch.object(main.settings, "chat_memory_sidecar_enabled", False),
    ):
        result = await main.run_autonomy_workflow("store memory", session_id="chat-l2-fail")

    l2l3 = result["metadata"]["chat_memory_l2l3_write"]
    assert l2l3["failed"] is True
    assert l2l3["qdrant_failed"] == 1
    assert result["metadata"]["chat_memory_l2l3_failed"] is True
    assert result["metadata"]["failed"] is True
    assert "qdrant_retry_fallback_used" not in l2l3
    assert any(
        str(err.get("type", "")) == "ChatMemoryL2L3WriteFailed"
        for err in result.get("errors", [])
        if isinstance(err, dict)
    )


@pytest.mark.asyncio
async def test_memory_sidecar_uses_fast_retry_after_primary_timeout() -> None:
    class _FakeAgent:
        async def decide(
            self,
            *,
            task,
            response,
            execution_summary,
            candidates_summary,
            recent_turns_summary,
            model_generate,
            chat_collection="cf_chat_memory",
        ):
            _ = (task, execution_summary, candidates_summary, recent_turns_summary, chat_collection)
            if len(str(response or "")) > 300:
                # Force primary timeout branch.
                await asyncio.sleep(1.2)
            tool_ret = await model_generate("return json")
            assert bool(tool_ret.get("ok", False)) is True
            return {"profiles": {}, "decision_source": "model_sidecar_v1"}

    class _FakeKernel:
        def __init__(self) -> None:
            self.force_tiers: list[str] = []

        async def run_once(self, **kwargs):
            assert kwargs.get("tool_name") == "model.generate"
            hints = kwargs.get("arguments", {}).get("hints", {})
            if isinstance(hints, dict):
                self.force_tiers.append(str(hints.get("force_tier", "")))
            return {"ok": True, "data": {"response": "{}"}}

    fake_kernel = _FakeKernel()
    with (
        patch("self_ai.main.get_memory_sidecar_agent", return_value=_FakeAgent()),
        patch.object(main.settings, "chat_memory_sidecar_enabled", True),
        patch.object(main.settings, "chat_memory_sidecar_retry_enabled", True),
        patch.object(main.settings, "chat_memory_sidecar_timeout_s", 2),
        patch.object(main.settings, "chat_memory_sidecar_fast_retry_timeout_s", 1),
    ):
        decision, status = await main._run_memory_sidecar_decision(
            kernel=fake_kernel,
            session_id="chat-x",
            task="remember preference",
            result={
                "response": "x" * 800,
                "metadata": {"execution_state": {}, "recent_tool_results": []},
                "errors": [],
                "quality_gate": {},
            },
            recent_turns_payload={"turns": []},
        )

    assert isinstance(decision, dict)
    assert status["succeeded"] is True
    assert status["fast_retry_used"] is True
    assert status["attempt_count"] >= 2
    assert fake_kernel.force_tiers
    assert set(fake_kernel.force_tiers) == {"fast"}


@pytest.mark.asyncio
async def test_run_autonomy_workflow_uses_default_memory_decision_when_sidecar_fails() -> None:
    class _FakeChatStore:
        def __init__(self) -> None:
            self.captured_decision = None

        def append_turn(self, **kwargs):
            self.captured_decision = kwargs.get("memory_decision")
            return {"turn_id": 7, "committed_memory_count": 1}

    class _FakeChatService:
        def load_recent_turns_for_injection(self, **kwargs):
            return {
                "chat_id": kwargs.get("chat_id", ""),
                "requested_turns": 0,
                "kept_turns": 0,
                "dropped_turns": 0,
                "budget": {"max_chars_total": 1400, "max_item_chars": 220, "used_chars": 0},
                "turns": [],
            }

    fake_kernel = _FakeKernel()
    fake_store = _FakeChatStore()
    with (
        patch("self_ai.main.get_kernel", return_value=fake_kernel),
        patch("self_ai.main.get_chat_memory_store", return_value=fake_store),
        patch("self_ai.main.get_chat_memory_service", return_value=_FakeChatService()),
        patch("self_ai.main._run_memory_sidecar_decision", new=AsyncMock(return_value=(None, {
            "attempted": True,
            "enabled": True,
            "succeeded": False,
            "timeout": True,
            "attempt_count": 2,
            "fast_retry_used": True,
            "fallback_used": False,
            "error_type": "TimeoutError",
            "error_message": "",
            "decision_source": "none",
        }))),
        patch.object(main.settings, "chat_memory_sidecar_enabled", True),
    ):
        result = await main.run_autonomy_workflow("save memory", session_id="chat-sidecar-fail")

    assert isinstance(fake_store.captured_decision, dict)
    assert fake_store.captured_decision.get("decision_source") == "system_default_timeout_v1"
    status = result["metadata"]["chat_memory_sidecar"]
    assert status["succeeded"] is False
    assert status["fallback_used"] is True


@pytest.mark.asyncio
async def test_run_autonomy_workflow_updates_session_global_constraints_when_mainloop_requests(
    tmp_path: Path,
) -> None:
    store = ChatMemoryStore(root_dir=tmp_path / "memory")
    service = ChatMemoryService(store)

    class _ConstraintKernel:
        def __init__(self) -> None:
            self.run = AsyncMock(
                return_value={
                    "response": "我会在这个会话里按这个偏好处理。",
                    "model": "mock-model",
                    "run_id": "run-gc",
                    "session_id": "chat-gc-main",
                    "workflow_decision": {
                        "constraint_maintenance": {
                            "enabled": True,
                            "reason": "user set a durable response preference",
                        }
                    },
                    "metadata": {
                        "last_model_payload": {
                            "constraint_maintenance": {
                                "enabled": True,
                                "reason": "user set a durable response preference",
                            }
                        }
                    },
                }
            )

            async def _run_once(*, tool_name, **_kwargs):
                if tool_name == "model.generate":
                    return {
                        "ok": True,
                        "data": {
                            "response": (
                                '{"ops":[{"op":"add","content":"Prefer answers that start with a concise conclusion.",'
                                '"confidence":0.93}]}'
                            )
                        },
                    }
                return {"ok": False, "error": {"type": "UnknownTool", "message": tool_name}}

            self.run_once = AsyncMock(side_effect=_run_once)

    fake_kernel = _ConstraintKernel()
    with (
        patch("self_ai.main.get_kernel", return_value=fake_kernel),
        patch("self_ai.main.get_chat_memory_store", return_value=store),
        patch("self_ai.main.get_chat_memory_service", return_value=service),
        patch.object(main.settings, "chat_memory_sidecar_enabled", False),
        patch.object(main.settings, "chat_memory_fusion_enabled", False),
        patch.object(main.settings, "chat_memory_compact_enabled", False),
    ):
        result = await main.run_autonomy_workflow(
            "以后回答先给结论，记住这个偏好",
            session_id="chat-gc-main",
        )

    current_items = store.read_global_constraints(chat_id="chat-gc-main")["items"]
    other_items = store.read_global_constraints(chat_id="chat-gc-other")["items"]

    assert current_items[0]["content"] == "Prefer answers that start with a concise conclusion."
    assert current_items[0]["id"] == "gc-1"
    assert other_items == []
    assert result["metadata"]["global_constraint_maintenance"]["requested"] is True
    assert result["metadata"]["global_constraint_maintenance"]["changed"] is True
    assert result["metadata"]["chat_global_constraints"][0]["content"] == current_items[0]["content"]
