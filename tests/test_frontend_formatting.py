# coding=utf-8
"""Tests for Agent Cockpit frontend formatting helpers."""

from __future__ import annotations

from self_ai.frontend_formatting import (
    as_dict,
    as_list,
    format_agent_turns,
    format_engine_timeline,
    format_memory_context,
    format_quality_gate,
    format_review_reports,
    format_revision_summary,
    format_runtime_signals,
    format_run_overview,
    format_tool_calls,
    redact_sensitive,
    safe_get,
    sanitize_payload,
    truncate_text,
)


def _sample_result() -> dict:
    return {
        "run_id": "run-1",
        "session_id": "session-1",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:01+00:00",
        "model": "mock-model",
        "workflow_decision": {"execution_mode": "architecture_design"},
        "metadata": {
            "revision_count": 1,
            "max_revision_iterations": 1,
            "revision_performed": True,
            "revision_target_agent": "architect",
            "pipeline": ["research", "plan", "context_pack", "agent_runtime", "final_response", "finalize"],
            "execution_state": {
                "progress": "complete",
                "goal_progress": "verified",
                "successful_calls": 3,
                "failed_calls": 0,
                "control_event_count": 2,
                "terminal_error_count": 0,
                "completion_gate_blocked_count": 2,
            },
            "chat_recent_turns_stats": {
                "requested_turns": 16,
                "kept_turns": 8,
                "fusion_failed": False,
                "degraded": True,
                "degraded_reasons": ["l2_search_timeout"],
                "layer_health": {
                    "l1_ok": True,
                    "l2_ok": False,
                    "l3_ok": True,
                    "l1_degraded": False,
                    "l2_degraded": True,
                    "l3_degraded": False,
                    "l2_error_type": "TimeoutError",
                    "l3_error_type": "",
                },
            },
            "chat_memory": {
                "committed_memory_count": 1,
                "write_policy": {
                    "issue_write_allowed": False,
                    "issue_suppressed_due_to_terminal_success": True,
                },
            },
            "chat_memory_l2l3_write": {"qdrant_ok": 1, "neo4j_ok": 1},
            "chat_memory_sidecar": {"attempted": True, "succeeded": True, "fallback_used": False},
            "prompt_stats": {"chars_total": 12345, "dynamic_ratio": 0.42},
        },
        "selected_agents": ["architect", "synthesizer"],
        "agent_outputs": {
            "architect": {"agent_name": "architect", "summary": "design", "output": "design details", "warnings": []},
            "synthesizer": {"agent_name": "synthesizer", "summary": "merged", "output": "merged details", "warnings": []},
        },
        "review_reports": [
            {
                "pass_review": False,
                "severity": "high",
                "major_issues": ["issue1"],
                "minor_issues": ["minor1"],
                "missing_requirements": ["req1"],
                "requires_revision": True,
                "revision_target_agent": "architect",
                "revision_instruction": "revise module split",
                "rationale": "high risk",
                "metadata": {},
            }
        ],
        "quality_gate": {
            "decision": "revise",
            "passed": False,
            "severity": "high",
            "requires_revision": True,
            "revision_target_agent": "architect",
            "current_iteration": 1,
            "max_revision_iterations": 1,
            "blocking_issues": ["issue1"],
            "warnings": [],
            "rationale": "needs fix",
            "metadata": {},
        },
        "context_pack": {
            "evidence_items": [{"source_type": "doc", "content": "very long text " * 100, "score": 0.9}],
            "source_type_groups": {"doc": ["d1"]},
            "context_preview": "context preview",
            "graph_paths": [{"relation": "RELATED_TO"}],
            "graph_summary": {"relation_type_counts": {"RELATED_TO": 1}},
            "graph_context_preview": "graph preview",
            "token_budget": 3000,
        },
        "errors": [],
        "response": "final response",
    }


def test_safe_get_and_collections_helpers() -> None:
    payload = {"a": {"b": [{"c": 1}]}}
    assert safe_get(payload, "a.b.0.c", None) == 1
    assert safe_get(payload, "a.b.2.c", "x") == "x"
    assert as_dict(None) == {}
    assert as_list(None) == []


def test_truncate_text() -> None:
    text = "a" * 300
    out = truncate_text(text, max_chars=120)
    assert len(out) <= 120
    assert out.endswith("...")


def test_redact_sensitive_values() -> None:
    assert redact_sensitive("api_key=123") == "[REDACTED]"
    assert redact_sensitive("token abc") == "[REDACTED]"
    assert redact_sensitive("password=xyz") == "[REDACTED]"
    assert redact_sensitive("normal-text") == "normal-text"


def test_sanitize_payload_redacts_sensitive_keys() -> None:
    payload = {"token": "abc", "nested": {"api_key": "x", "ok": "v"}}
    sanitized = sanitize_payload(payload)
    assert sanitized["token"] == "[REDACTED]"
    assert sanitized["nested"]["api_key"] == "[REDACTED]"
    assert sanitized["nested"]["ok"] == "v"


def test_format_run_overview_missing_safe() -> None:
    overview = format_run_overview({}, logs=None)
    assert overview["run_id"] == ""
    assert overview["execution_mode"] == ""
    assert overview["revision_count"] == 0


def test_format_engine_timeline_parses_and_skips_bad_json() -> None:
    logs = [
        '[self-ai] {"event":"engine.run.start","ts":"2026-01-01T00:00:00Z"}',
        "bad json line",
        '[self-ai] {"event":"engine.stage.end","stage":"plan","status":"end"}',
    ]
    timeline = format_engine_timeline({}, logs)
    assert len(timeline) == 2
    assert timeline[0]["event"] == "engine.run.start"
    assert timeline[1]["event"] == "engine.stage.end"


def test_format_tool_calls_extracts_start_end_error() -> None:
    logs = [
        '[self-ai] {"event":"tool.call.start","call_id":"c1","tool_name":"model.generate","fields":{"arguments_preview":"x"}}',
        '[self-ai] {"event":"tool.call.end","call_id":"c1","tool_name":"model.generate","fields":{"ok":true,"latency_ms":12}}',
        '[self-ai] {"event":"tool.call.error","call_id":"c2","tool_name":"review.run","fields":{"error_type":"TimeoutError","latency_ms":50}}',
    ]
    calls = format_run_overview({}, logs=None)  # ensure independent
    assert calls["run_id"] == ""
    extracted = format_tool_calls({}, logs)
    assert len(extracted) == 2
    assert extracted[0]["tool_name"] == "model.generate"
    assert extracted[0]["ok"] is True
    assert extracted[1]["ok"] is False


def test_format_tool_calls_extracts_current_engine_tool_events() -> None:
    logs = [
        (
            '[self-ai] {"event":"engine.stage.tool.start","stage":"tool","fields":'
            '{"tool":"workspace.file.read","arguments_preview":"README.md"}}'
        ),
        (
            '[self-ai] {"event":"engine.stage.tool.end","stage":"tool","fields":'
            '{"tool":"workspace.file.read","ok":true,"latency_ms":8,"response_preview":"content"}}'
        ),
    ]

    extracted = format_tool_calls({}, logs)

    assert len(extracted) == 1
    assert extracted[0]["tool_name"] == "workspace.file.read"
    assert extracted[0]["ok"] is True
    assert extracted[0]["latency_ms"] == 8


def test_format_runtime_signals_surfaces_p0_health_fields() -> None:
    logs = [
        '[self-ai] {"event":"engine.stage.tool.end","fields":{"tool":"model.generate","ok":true}}',
        '[self-ai] {"event":"engine.stage.tool.end","fields":{"tool":"workspace.file.read","ok":true}}',
    ]

    runtime = format_runtime_signals(_sample_result(), logs=logs)

    assert runtime["progress"] == "complete"
    assert runtime["control_event_count"] == 2
    assert runtime["terminal_error_count"] == 0
    assert runtime["fusion_failed"] is False
    assert runtime["degraded"] is True
    assert runtime["degraded_reasons"] == ["l2_search_timeout"]
    assert runtime["layer_health"]["l2_degraded"] is True
    assert runtime["issue_suppressed_due_to_terminal_success"] is True
    assert runtime["model_call_count"] == 1
    assert runtime["tool_call_count"] == 2


def test_format_agent_turns_shows_selected_agents_and_outputs() -> None:
    turns = format_agent_turns(_sample_result())
    assert turns["selected_agents"] == ["architect", "synthesizer"]
    assert turns["agent_output_count"] == 2
    assert len(turns["turns"]) == 2


def test_format_revision_summary_no_revision_safe() -> None:
    result = {"metadata": {}, "quality_gate": {}, "review_reports": []}
    summary = format_revision_summary(result, logs=[])
    assert summary["revision_count"] == 0
    assert summary["max_revision_iterations"] == 1


def test_format_review_reports_missing_safe() -> None:
    summary = format_review_reports({"review_reports": None})
    assert summary["count"] == 0
    assert summary["reports"] == []


def test_format_quality_gate_decisions() -> None:
    for decision in ("pass", "warn", "revise", "fail"):
        gate = format_quality_gate({"quality_gate": {"decision": decision}})
        assert gate["decision"] == decision


def test_format_memory_context_not_expose_large_evidence() -> None:
    memory = format_memory_context(_sample_result())
    assert memory["evidence_count"] == 1
    assert len(memory["evidence_preview"]) == 1
    assert len(memory["evidence_preview"][0]["content_preview"]) < 200
