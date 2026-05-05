# coding=utf-8
"""Tests for frontend_app helpers without UI automation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from self_ai import frontend_app


def test_run_frontend_workflow_once_calls_run_autonomy_workflow() -> None:
    async def _fake_run(_question: str, *, session_id: str = "default", chat_id: str | None = None) -> dict:
        assert session_id == "test-session"
        assert chat_id == "test-session"
        return {"response": "ok", "run_id": "r1", "session_id": "s1", "metadata": {}}

    with patch("self_ai.frontend_app.run_autonomy_workflow", side_effect=_fake_run) as mocked:
        out = frontend_app.run_frontend_workflow_once("hello", session_id="test-session")

    assert mocked.call_count == 1
    assert out["error"] is None
    assert out["result"]["response"] == "ok"
    assert isinstance(out["elapsed_ms"], int)


def test_execute_question_catches_exception() -> None:
    async def _boom(_question: str, *, session_id: str = "default", chat_id: str | None = None) -> dict:
        assert session_id == "test-session"
        assert chat_id == "test-session"
        raise RuntimeError("backend failed")

    with patch("self_ai.frontend_app.run_autonomy_workflow", side_effect=_boom):
        result, error, logs, elapsed_ms = frontend_app._execute_question(
            "hello",
            session_id="test-session",
            chat_id="test-session",
        )

    assert result is None
    assert isinstance(error, dict)
    assert error["error_type"] == "RuntimeError"
    assert isinstance(logs, list)
    assert isinstance(elapsed_ms, int)


def test_execute_question_collects_trace_logs() -> None:
    captured_listener = {"fn": None}

    def _register(listener):
        captured_listener["fn"] = listener
        listener({"event": "engine.run.start", "ts": "2026-01-01T00:00:00Z"})

    def _unregister(_listener):
        return None

    async def _fake_run(_question: str, *, session_id: str = "default", chat_id: str | None = None) -> dict:
        assert session_id == "test-session"
        assert chat_id == "isolated-chat"
        return {"response": "ok", "run_id": "r1", "session_id": "s1", "metadata": {}}

    with (
        patch("self_ai.frontend_app.register_trace_listener", side_effect=_register),
        patch("self_ai.frontend_app.unregister_trace_listener", side_effect=_unregister),
        patch("self_ai.frontend_app.run_autonomy_workflow", side_effect=_fake_run),
    ):
        result, error, logs, _elapsed_ms = frontend_app._execute_question(
            "hello",
            session_id="test-session",
            chat_id="isolated-chat",
        )

    assert error is None
    assert result["response"] == "ok"
    assert any("engine.run.start" in line for line in logs)


def test_append_chat_run_keeps_chat_history_isolated() -> None:
    chat_a = frontend_app._create_chat_record("chat-a")
    chat_b = frontend_app._create_chat_record("chat-b")
    result = {
        "run_id": "run-a",
        "model": "mock-model",
        "metadata": {
            "execution_state": {"terminal_error_count": 0, "control_event_count": 1},
            "chat_recent_turns_stats": {"fusion_failed": False, "degraded": False},
        },
        "errors": [],
    }

    frontend_app._append_chat_run(
        chat_a,
        question="first task",
        result=result,
        error=None,
        logs=[],
        elapsed_ms=42,
    )

    assert chat_a["turn_count"] == 1
    assert chat_b["turn_count"] == 0
    assert chat_a["title"] == "first task"
    assert chat_a["runs"][0]["metrics"]["control_event_count"] == 1


def test_load_latest_benchmark_snapshot_reads_newest_run(tmp_path: Path) -> None:
    old_run = tmp_path / "20260101_old"
    new_run = tmp_path / "20260102_new"
    old_run.mkdir()
    new_run.mkdir()
    (old_run / "metrics.json").write_text(json.dumps({"verified_success_rate": 0.5}), encoding="utf-8")
    (new_run / "metrics.json").write_text(json.dumps({"verified_success_rate": 1.0}), encoding="utf-8")
    (new_run / "case_results.json").write_text(
        json.dumps([{"case_id": "core.create", "validation_ok": True}]),
        encoding="utf-8",
    )
    (new_run / "failures.json").write_text(json.dumps([]), encoding="utf-8")

    snapshot = frontend_app._load_latest_benchmark_snapshot(tmp_path)

    assert snapshot["run_dir"].endswith("20260102_new")
    assert snapshot["metrics"]["verified_success_rate"] == 1.0
    assert snapshot["cases"][0]["case_id"] == "core.create"
    assert snapshot["error"] == ""
